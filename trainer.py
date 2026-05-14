"""
Features:
  - Complete training/validation loop (includes AMP mixed precision)
  - CosineAnnealing learning rate scheduling + Warmup
  - Early Stopping (monitors val_auc)
  - Layered learning rates (high lr for classifier head, low lr for backbone)
  - Evaluation metrics: AUC-ROC, F1-Score, MCC (Matthews Correlation Coefficient)
  - TensorBoard training log recording
  - Auto-save best weights
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig, cfg
from model import BrainCancerModel


# ──────────────────────────────────────────────────────
# Early Stopping Monitor
# ──────────────────────────────────────────────────────
class EarlyStopping:
    """
    Monitors validation metrics, triggers stop signal when no improvement for `patience` epochs.

    Args:
        patience:   Max waiting epochs
        mode:       "max" (larger is better, e.g., AUC) or "min" (smaller is better, e.g., Loss)
        delta:      Minimum improvement amount (less than this is considered "no improvement")
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.triggered = False

    def step(self, current_value: float) -> bool:
        """
        Check if current value has improved.

        Returns:
            True  = Improved (save model)
            False = No improvement (counter +1)
        """
        if self.mode == "max":
            improved = current_value > self.best_value + self.delta
        else:
            improved = current_value < self.best_value - self.delta

        if improved:
            self.best_value = current_value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
            return False


# ──────────────────────────────────────────────────────
# Evaluation Function: Compute all metrics
# ──────────────────────────────────────────────────────
def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification evaluation metrics.

    Args:
        all_labels:  Ground truth label array (0 or 1)
        all_probs:   Model predicted positive class probability (0.0 ~ 1.0)
        threshold:   Threshold to convert probability to category (default 0.5)

    Returns:
        metrics dict:  Contains auc, f1, mcc, accuracy
    """
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {}

    # AUC-ROC: More sensitive to imbalanced data, closer to 1.0 is better
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.5  # AUC is meaningless when there is only one class

    # F1-Score: Harmonic mean of precision and recall, suitable for imbalanced data
    metrics["f1"] = f1_score(all_labels, all_preds, zero_division=0)

    # MCC (Matthews Correlation Coefficient): More comprehensive than F1, -1 ~ +1, 1 is perfect
    metrics["mcc"] = matthews_corrcoef(all_labels, all_preds)

    # Accuracy
    metrics["accuracy"] = float((all_labels == all_preds).mean())

    return metrics


# ──────────────────────────────────────────────────────
# Core Trainer Class
# ──────────────────────────────────────────────────────
class Trainer:
    """
    Complete training manager, encapsulates:
    - Optimizer construction (layered learning rates)
    - Learning rate scheduling (CosineAnnealing)
    - AMP Mixed Precision Training
    - Early Stopping
    - Metric computation and TensorBoard logging
    - Best weight auto-saving and restoring

    Args:
        model:        BrainCancerModel instance
        train_cfg:    TrainConfig configuration
        device:       Execution device
        pos_weight:   Positive sample weight (handles class imbalance)
    """

    def __init__(
        self,
        model: BrainCancerModel,
        train_cfg: TrainConfig,
        device: torch.device,
        pos_weight: Optional[float] = None,
    ) -> None:
        self.model = model
        self.cfg = train_cfg
        self.device = device
        self.current_epoch = 0

        # ── Loss Function: BCEWithLogitsLoss (built-in numerically stable Sigmoid) ──
        # pos_weight: If positive samples are few, give positive samples higher loss weight
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=device)
            print(f"    Using weighted loss function (pos_weight={pos_weight:.2f})")
        else:
            pw = None

        # label_smoothing: Soften hard labels 0/1 to ε/2 and 1-ε/2 to prevent overfitting
        ls = train_cfg.label_smoothing
        if ls > 0:
            print(f"    Label smoothing enabled (label_smoothing={ls})")

        class _BCEWithLabelSmoothing(nn.Module):
            def __init__(self, pos_weight, smoothing, device):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                self.smoothing = smoothing

            def forward(self, logits, targets):
                if self.smoothing > 0:
                    targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
                return self.bce(logits, targets)

        self.criterion = _BCEWithLabelSmoothing(pw, ls, device)

        # ── Build Optimizer (layered learning rates) ──
        self._build_optimizer()

        # ── CosineAnnealing Learning Rate Scheduling ──
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.num_epochs,
            eta_min=train_cfg.lr_min,
        )

        # ── Mixed Precision Scaler (AMP) ──
        # Auto-degrades to regular float32 training when enabled=False
        self._use_amp = train_cfg.use_amp and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self._use_amp) if device.type == "cuda" else GradScaler("cpu", enabled=False)
        if self._use_amp:
            print("    Mixed Precision Training (AMP) enabled")
        else:
            print("    AMP not enabled (CPU training or explicitly disabled)")

        # ── Early Stopping ──
        self.early_stopping = EarlyStopping(
            patience=train_cfg.early_stopping_patience,
            mode="max",   # Monitor val_auc (larger is better)
        )

        # ── TensorBoard Logger ──
        self.writer = SummaryWriter(log_dir=str(train_cfg.log_dir))
        print(f"    TensorBoard log path: {train_cfg.log_dir}")
        print(f"     Run tensorboard --logdir=\"{train_cfg.log_dir}\" to view training curves\n")

        # ── Best validation metric (for saving best weights) ──
        self.best_val_auc = 0.0

    def _build_optimizer(self) -> None:
        """
        Build layered learning rate optimizer.
        Classifier head uses high learning rate (lr_head),
        Backbone (if unfrozen) uses low learning rate (lr_backbone).
        """
        backbone_params, classifier_params = self.model.get_trainable_params()

        param_groups = [
            {
                "params": classifier_params,
                "lr": self.cfg.lr_head,
                "name": "classifier",
            }
        ]

        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": self.cfg.lr_backbone,
                "name": "backbone",
            })

        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.cfg.weight_decay,
        )

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Single epoch training loop.

        Returns:
            Dict containing train_loss, train_auc, train_f1, train_mcc
        """
        self.model.train()

        total_loss = 0.0
        all_labels = []
        all_probs = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # ── Forward pass (AMP auto precision reduction) ──
            with autocast(device_type=self.device.type, enabled=self._use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # ── Backward pass (AMP Scaler handles gradient scaling) ──
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping (prevents gradient explosion, especially important for Transformers)
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.grad_clip_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Record Loss and prediction results
            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_probs),
        )
        metrics["loss"] = avg_loss

        return metrics

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validation/Test evaluation loop (no gradient calculation, saves VRAM).

        Returns:
            Dict containing loss, auc, f1, mcc, accuracy
        """
        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_probs = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self._use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(loader)
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_probs),
        )
        metrics["loss"] = avg_loss

        return metrics

    def _log_metrics(
        self,
        phase: str,
        metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        """Write metrics to TensorBoard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, epoch)

    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        elapsed: float,
    ) -> None:
        """Format and print summary per epoch"""
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch [{epoch:>3}/{total_epochs}] | "
            f"LR={current_lr:.2e} | "
            f"Time={elapsed:.1f}s\n"
            f"    Train → Loss={train_metrics['loss']:.4f} | "
            f"AUC={train_metrics['auc']:.4f} | "
            f"F1={train_metrics['f1']:.4f} | "
            f"MCC={train_metrics['mcc']:.4f}\n"
            f"    Val   → Loss={val_metrics['loss']:.4f} | "
            f"AUC={val_metrics['auc']:.4f} | "
            f"F1={val_metrics['f1']:.4f} | "
            f"MCC={val_metrics['mcc']:.4f}"
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """
        Execute the complete training process.

        Args:
            train_loader:  Training set DataLoader
            val_loader:    Validation set DataLoader

        Returns:
            history:  Dictionary containing history records of train/val metrics per epoch
        """
        history = {
            "train_loss": [], "train_auc": [], "train_f1": [], "train_mcc": [],
            "val_loss": [], "val_auc": [], "val_f1": [], "val_mcc": [],
        }

        print("\n    Starting Training...")

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.current_epoch = epoch
            t_start = time.time()

            # ── Phase 1 → Phase 2: Unfreeze backbone when reaching unfreeze epoch ──
            if epoch == self.model.model_cfg.unfreeze_at_epoch + 1:
                print(f"\n    Epoch {epoch}: Starting Fine-tuning, unfreezing backbone network!")
                self.model.unfreeze_backbone()
                # Rebuild optimizer to include backbone parameters
                self._build_optimizer()
                # Reset scheduler
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=(self.cfg.num_epochs - epoch),
                    eta_min=self.cfg.lr_min,
                )

            # ── Train one epoch ──
            train_metrics = self._train_one_epoch(train_loader)

            # ── Validate ──
            val_metrics = self._evaluate(val_loader)

            # ── Learning Rate Scheduling ──
            self.scheduler.step()

            # ── Record History ──
            for key in ["loss", "auc", "f1", "mcc"]:
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])

            # ── TensorBoard Logging ──
            self._log_metrics("Train", train_metrics, epoch)
            self._log_metrics("Val", val_metrics, epoch)
            self.writer.add_scalar(
                "LR/head",
                self.optimizer.param_groups[0]["lr"],
                epoch,
            )

            elapsed = time.time() - t_start
            self._print_epoch_summary(
                epoch, self.cfg.num_epochs,
                train_metrics, val_metrics, elapsed,
            )

            # ── Early Stopping Check + Best Model Save ──
            val_auc = val_metrics["auc"]
            improved = self.early_stopping.step(val_auc)

            if improved:
                self.best_val_auc = val_auc
                self._save_checkpoint(epoch, val_metrics)
                print(f"    Best model saved! (val_auc={val_auc:.4f})")
            else:
                print(
                    f"    No improvement ({self.early_stopping.counter}/{self.early_stopping.patience}) "
                    f"| Historical best val_auc={self.best_val_auc:.4f}"
                )

            if self.early_stopping.triggered:
                print(f"\n    Early Stopping triggered! No improvement for {self.early_stopping.patience} consecutive epochs.")
                break

            print()  # Empty line to separate epoch outputs

        self.writer.close()

        print(f"\n    Training Complete! Best Val AUC = {self.best_val_auc:.4f}")
        print(f"     Best weights saved to: {self.cfg.checkpoint_path}\n")

        return history

    def evaluate_test(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate loaded best model on the test set.

        First restore best weights, then evaluate to ensure results are from the best model.
        """
        print("\n    Loading best weights for test set evaluation...")
        self._load_checkpoint()

        test_metrics = self._evaluate(test_loader)

        print("\n    Final Test Set Results:")
        print(f"     AUC-ROC  : {test_metrics['auc']:.4f}")
        print(f"     F1-Score : {test_metrics['f1']:.4f}")
        print(f"     MCC      : {test_metrics['mcc']:.4f}")
        print(f"     Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"     Loss     : {test_metrics['loss']:.4f}\n")

        return test_metrics

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
    ) -> None:
        """Save model weights and training state"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "val_mcc": val_metrics["mcc"],
        }
        torch.save(checkpoint, self.cfg.checkpoint_path)

    def _load_checkpoint(self) -> None:
        """Load best model weights from disk"""
        if not self.cfg.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Weight file not found: {self.cfg.checkpoint_path}"
            )
        checkpoint = torch.load(
            self.cfg.checkpoint_path,
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"    Loaded epoch {checkpoint['epoch']} weights "
              f"(val_auc={checkpoint['val_auc']:.4f})")
