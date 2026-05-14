"""
Features:
  - Loads pretrained Swin Transformer weights from timm library
  - Replaces classification head with custom binary classification head (includes Dropout + BatchNorm)
  - Supports selectively freezing/unfreezing backbone parameters (Two-stage training strategy)
  - Provides parameter statistics tools
"""

from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn

from config import ModelConfig, cfg


# ──────────────────────────────────────────────────────
# Custom Classification Head: More powerful than timm's default single linear layer
# ──────────────────────────────────────────────────────
class ClassifierHead(nn.Module):
    """
    Two-layer MLP classification head, includes BatchNorm + Dropout + GELU activation.

    Structure: [in_features] → FC(512) → BN → GELU → Dropout → FC(1)

    Why this design:
    - BatchNorm: Stabilizes training, accelerates convergence
    - Dropout: Regularization, prevents overfitting
    - GELU: Smoother than ReLU, standard choice for Transformer series
    - Outputs 1 logit (BCEWithLogitsLoss does not require manual Sigmoid)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),   # Binary classification: output single logit
        )

        # Kaiming initialization for linear layers (suitable for GELU activation)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ──────────────────────────────────────────────────────
# Core: Swin Transformer Brain Cancer Detection Model
# ──────────────────────────────────────────────────────
class BrainCancerModel(nn.Module):
    """
    Brain Cancer binary classification model based on Swin Transformer.

    Two-stage training strategy (controlled in Trainer):
      Stage 1 (Freeze backbone): Train only ClassifierHead, quickly establish basic classification capability
      Stage 2 (Unfreeze and fine-tune): Fine-tune entire network with very small learning rate to maximize performance

    Args:
        model_cfg:  ModelConfig instance
    """

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()

        self.model_cfg = model_cfg

        # ── Load timm pretrained Swin Transformer backbone ──
        print(f"    Loading model: {model_cfg.model_name}")
        print(f"     Pretrained weights: {'ImageNet-1K' if model_cfg.pretrained else 'Random Initialization'}")

        # Use num_classes=0 to remove timm's default classification head, get pure feature extractor
        self.backbone = timm.create_model(
            model_cfg.model_name,
            pretrained=model_cfg.pretrained,
            num_classes=0,    # 0 = remove original classification head, return only feature vectors
            drop_path_rate=getattr(model_cfg, 'drop_path_rate', 0.2), # Stochastic Depth to prevent overfitting
        )

        # Get the feature dimension of the backbone output (Swin Base = 1024)
        in_features = self.backbone.num_features
        print(f"     Backbone output feature dimension: {in_features}")

        # ── Custom classification head ──
        self.classifier = ClassifierHead(
            in_features=in_features,
            hidden_dim=model_cfg.classifier_hidden_dim,
            dropout_rate=model_cfg.dropout_rate,
        )

        # ── Initial state: decide whether to freeze backbone based on config ──
        if model_cfg.freeze_backbone:
            self.freeze_backbone()
            print(f"     Backbone state: Frozen (training classifier head only)")
        else:
            print(f"     Backbone state: Fully unfreezed (end-to-end training)")

        # Print parameter statistics
        self._print_param_stats()

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the backbone network (stop gradient updates)"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = -1) -> None:
        """
        Unfreeze backbone network parameters (used in Fine-tuning stage).

        Args:
            unfreeze_last_n_layers:  -1 = Unfreeze all layers;
                                      N = Only unfreeze the last N submodules (more conservative fine-tuning)
        """
        if unfreeze_last_n_layers == -1:
            # Unfreeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("    Backbone network fully unfrozen (Fine-tuning mode)")
        else:
            # Only unfreeze the last N submodules (saves VRAM)
            layers = list(self.backbone.children())
            for layer in layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"    Last {unfreeze_last_n_layers} layers of backbone network unfrozen")

    def get_trainable_params(self) -> Tuple[list, list]:
        """
        Returns a list of grouped parameters (for setting different learning rates).

        Returns:
            backbone_params:    Backbone parameters (use low lr during Fine-tuning)
            classifier_params:  Classifier head parameters (always use high lr)
        """
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        classifier_params = list(self.classifier.parameters())
        return backbone_params, classifier_params

    def _print_param_stats(self) -> None:
        """Print model parameter statistics"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"\n    Model parameter statistics:")
        print(f"     Total parameters:      {total:>12,}")
        print(f"     Trainable parameters:  {trainable:>12,}  ({100*trainable/total:.1f}%)")
        print(f"     Frozen parameters:     {frozen:>12,}  ({100*frozen/total:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:  Input image Tensor, shape = (batch_size, 3, H, W)

        Returns:
            logits: Raw output before Sigmoid, shape = (batch_size,)
                    (BCEWithLogitsLoss will automatically apply Sigmoid internally)
        """
        # Backbone feature extraction: (B, C, H, W) → (B, in_features)
        features = self.backbone(x)

        # Classification head: (B, in_features) → (B, 1) → squeeze → (B,)
        logits = self.classifier(features).squeeze(1)

        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference interface: returns probability values (after Sigmoid).

        Args:
            x:  Input image Tensor

        Returns:
            proba:  Tumor probability, range [0, 1], shape = (batch_size,)
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ──────────────────────────────────────────────────────
# Factory Function: Quickly build model and move to target device
# ──────────────────────────────────────────────────────
def build_model(
    model_cfg: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple["BrainCancerModel", torch.device]:
    """
    Build model and automatically select execution device (GPU preferred).

    Args:
        model_cfg:  Model configuration, use global default config if None
        device:     Specify device, auto-detect if None

    Returns:
        model:   Initialized model instance
        device:  Actual device used (cuda / cpu)
    """
    if model_cfg is None:
        model_cfg = cfg.model

    # Automatic device selection
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n    GPU detected: {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
        else:
            device = torch.device("cpu")
            print("\n    No CUDA GPU detected, using CPU for training (slow)")

    print("\n    Building Brain Cancer Detection Model...")

    model = BrainCancerModel(model_cfg).to(device)

    print(f"\n    Model loaded to: {device}")


    return model, device
