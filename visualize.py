"""
Features:
  - Implements Grad-CAM (Gradient-weighted Class Activation Mapping), showing "where the model is looking"
  - Overlays the heatmap on the original MRI image to generate medically interpretable visualizations
  - Supports batch processing, saves as PNG images
  - Supports separate visualization for correct and incorrect predictions

Grad-CAM Principle (Brief):
  1. Compute gradient of the target class logit
  2. Perform global average pooling on gradients over the spatial dimensions of the last conv layer feature map -> obtain weights
  3. Perform weighted sum of feature maps using weights -> obtain activation heatmap
  4. Upsample to original image size using bilinear interpolation, overlay and display
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import VizConfig, cfg
from dataset import BrainMRIDataset
from model import BrainCancerModel


# ──────────────────────────────────────────────────────
# Grad-CAM Implementation
# ──────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM Visualization Tool.

    Uses PyTorch's hook mechanism to capture forward feature maps and
    backward gradients without modifying the model structure.

    Args:
        model:        BrainCancerModel instance
        target_layer: Target conv layer (auto-inferred if None)
        device:       Execution device
    """

    def __init__(
        self,
        model: BrainCancerModel,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device

        # Automatically find the last layer that can produce Grad-CAM
        if target_layer is None:
            target_layer = self._find_target_layer()
        self.target_layer = target_layer

        # Store intermediate values captured by hooks
        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register forward hook (captures feature maps)
        self._forward_hook = target_layer.register_forward_hook(
            self._forward_hook_fn
        )
        # Register backward hook (captures gradients)
        self._backward_hook = target_layer.register_full_backward_hook(
            self._backward_hook_fn
        )

    def _find_target_layer(self) -> nn.Module:
        """
        Automatically find the last suitable layer for Grad-CAM in Swin Transformer.

        The LayerNorm before the last norm layer of Swin Transformer is suitable for CAM.
        For Swin, we choose the last BasicLayer or PatchMerging in the backbone.
        """
        # Attempt to find the last stage of Swin
        target = None

        # Iterate through backbone submodules, find the last stage with the 'blocks' attribute
        for name, module in self.model.backbone.named_modules():
            # Swin Transformer's stages are usually named layers.x or stages.x
            if hasattr(module, "blocks") and hasattr(module, "downsample"):
                target = module
            elif "norm" in name.lower() and isinstance(module, nn.LayerNorm):
                target = module

        if target is None:
            # Fallback: directly use the last submodule of the backbone
            target = list(self.model.backbone.children())[-1]

        print(f"    Grad-CAM Target Layer: {type(target).__name__}")
        return target

    def _forward_hook_fn(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook: capture output feature map of target layer"""
        # output could be Tensor or tuple, handle uniformly
        if isinstance(output, tuple):
            output = output[0]
        self._feature_maps = output.detach()

    def _backward_hook_fn(
        self,
        module: nn.Module,
        grad_input: Tuple,
        grad_output: Tuple,
    ) -> None:
        """Backward hook: capture gradient of target layer"""
        if grad_output[0] is not None:
            self._gradients = grad_output[0].detach()

    def generate(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image:         Preprocessed image Tensor, shape = (1, C, H, W)
            target_class:  Target class (None = use model predicted class)

        Returns:
            cam:    Normalized Grad-CAM heatmap, shape = (H, W), range [0, 1]
            prob:   Probability of being predicted as tumor
        """
        self.model.eval()
        image = image.to(self.device)
        image.requires_grad_(True)

        # Forward pass
        logit = self.model(image)          # shape = (1,)
        prob = torch.sigmoid(logit).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass on prediction score (compute gradient)
        logit.backward()

        # ── Compute Grad-CAM ──
        gradients = self._gradients        # shape: (1, ..., C) or (1, C, H, W)
        feature_maps = self._feature_maps  # shape: (1, ..., C) or (1, C, H, W)

        if gradients is None or feature_maps is None:
            raise RuntimeError(
                "Unable to capture gradients or feature maps! Please check if target_layer is correct."
            )

        # Process Swin Transformer's special output format (B, H*W, C) -> (B, C, H, W)
        if feature_maps.ndim == 3:
            # Swin output format: (batch, seq_len, channels)
            B, N, C = feature_maps.shape
            spatial_size = int(math.sqrt(N))
            feature_maps = feature_maps.reshape(B, spatial_size, spatial_size, C)
            feature_maps = feature_maps.permute(0, 3, 1, 2)  # (B, C, H, W)

            if gradients.ndim == 3:
                gradients = gradients.reshape(B, spatial_size, spatial_size, C)
                gradients = gradients.permute(0, 3, 1, 2)

        # Global average pooling on gradients -> weights (C,)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps -> heatmap
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Keep only positive activations (negative activations don't contribute to target class)

        # Upsample to original image size
        h, w = image.shape[2], image.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, prob

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            original_image:  Original RGB image, shape = (H, W, 3), range [0, 255]
            cam:             Grad-CAM heatmap, shape = (H, W), range [0, 1]
            alpha:           Heatmap transparency
            colormap:        OpenCV colormap

        Returns:
            overlaid:  Overlaid RGB image, shape = (H, W, 3)
        """
        # Convert cam to colored heatmap
        heatmap = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).clip(0, 255).astype(np.uint8)

        # Resize heatmap to match original image size
        h, w = original_image.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))

        # Weighted overlay
        overlaid = (original_image * (1 - alpha) + heatmap * alpha).clip(0, 255)
        return overlaid.astype(np.uint8)

    def remove_hooks(self) -> None:
        """Remove registered hooks (free memory)"""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ──────────────────────────────────────────────────────
# Batch Visualization Function
# ──────────────────────────────────────────────────────
def visualize_gradcam_batch(
    model: BrainCancerModel,
    dataset: BrainMRIDataset,
    device: torch.device,
    viz_cfg: Optional[VizConfig] = None,
    num_images: int = 8,
    output_dir: Optional[Path] = None,
    filename: str = "gradcam_visualization.png",
) -> None:
    """
    Generate Grad-CAM visualization for random samples in the dataset and save as image.

    Displays 4 columns:
      [Original MRI] [Grad-CAM Heatmap] [Overlay] [Prediction Info]

    Args:
        model:       Model loaded with best weights
        dataset:     BrainMRIDataset instance (contains original path info)
        device:      Execution device
        num_images:  Number of images to display (recommend multiples of 4)
        output_dir:  Output directory
        filename:    Output filename
    """
    if viz_cfg is None:
        viz_cfg = cfg.viz
    if output_dir is None:
        output_dir = viz_cfg.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n    Generating Grad-CAM visualization ({num_images} images)...")

    # Initialize Grad-CAM
    grad_cam = GradCAM(model=model, device=device)

    # Randomly select sample indices
    indices = np.random.choice(len(dataset), size=min(num_images, len(dataset)), replace=False)

    # Create visualization canvas
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        num_images, 3,
        figsize=(15, num_images * 5),
        facecolor="#1a1a2e",
    )
    fig.suptitle(
        "    Brain Cancer MRI - Grad-CAM Visualization",
        fontsize=18, fontweight="bold", color="white", y=0.98,
    )

    # Denormalization function (restore Tensor to displayable image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for row, idx in enumerate(indices):
        # Load preprocessed Tensor image and label
        img_tensor, label = dataset[idx]
        img_path = dataset.get_image_path(idx)

        # ── Read original image (unnormalized, resized only) ──
        original_bgr = cv2.imread(str(img_path))
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.resize(original_rgb, (224, 224))

        # ── Generate Grad-CAM ──
        img_batch = img_tensor.unsqueeze(0)  # (1, C, H, W)
        cam, prob = grad_cam.generate(img_batch)

        # ── Generate Overlay ──
        overlaid = grad_cam.overlay_on_image(
            original_rgb, cam,
            alpha=viz_cfg.heatmap_alpha,
        )

        # ── Heatmap Color Display ──
        heatmap_color = (cam * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # ── Prediction Info ──
        true_label = "Tumor (Yes)" if int(label.item()) == 1 else "Normal (No)"
        pred_label = "Tumor (Yes)" if prob >= 0.5 else "Normal (No)"
        is_correct = (prob >= 0.5) == (int(label.item()) == 1)
        border_color = "#00ff88" if is_correct else "#ff4444"

        ax_row = axes[row] if num_images > 1 else axes

        # Column 0: Original MRI
        ax_row[0].imshow(original_rgb)
        ax_row[0].set_title(
            f"Original MRI\nTrue: {true_label}",
            fontsize=10, color="white",
        )
        ax_row[0].axis("off")
        for spine in ax_row[0].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

        # Column 1: Grad-CAM Heatmap
        ax_row[1].imshow(heatmap_color)
        ax_row[1].set_title(
            f"Grad-CAM Heatmap\n(Red = Model Focus Area)",
            fontsize=10, color="white",
        )
        ax_row[1].axis("off")

        # Column 2: Overlay + Prediction Result
        ax_row[2].imshow(overlaid)
        status_icon = "OK" if is_correct else "FAIL"
        ax_row[2].set_title(
            f"Overlay | {status_icon} Pred: {pred_label}\n"
            f"Tumor Prob: {prob:.1%}",
            fontsize=10,
            color=border_color,
        )
        ax_row[2].axis("off")

        # Set background color for each row
        for ax in ax_row:
            ax.set_facecolor("#1a1a2e")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = output_dir / filename
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="#1a1a2e",
    )
    plt.close(fig)

    grad_cam.remove_hooks()

    print(f"    Grad-CAM visualization saved: {save_path}")


# ──────────────────────────────────────────────────────
# Training Curve Visualization
# ──────────────────────────────────────────────────────
def plot_training_history(
    history: dict,
    output_dir: Path,
    filename: str = "training_curves.png",
) -> None:
    """
    Plot training/validation curves (Loss, AUC, F1, MCC).

    Args:
        history:    History dictionary returned by Trainer.train()
        output_dir: Output directory
        filename:   Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["loss", "auc", "f1", "mcc"]
    titles = ["Loss (Lower is better)", "AUC-ROC (Higher is better)", "F1-Score (Higher is better)", "MCC (Higher is better)"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#1a1a2e")
    fig.suptitle(
        "    Brain Cancer Model - Training Curves",
        fontsize=16, fontweight="bold", color="white",
    )

    colors = {"train": "#00d4ff", "val": "#ff6b6b"}

    for ax, metric, title in zip(axes.flat, metrics, titles):
        train_values = history.get(f"train_{metric}", [])
        val_values = history.get(f"val_{metric}", [])
        epochs = range(1, len(train_values) + 1)

        ax.plot(epochs, train_values, color=colors["train"], linewidth=2, label="Train Set", alpha=0.9)
        ax.plot(epochs, val_values, color=colors["val"], linewidth=2, label="Val Set", alpha=0.9)

        # Mark the optimal point
        if metric != "loss":
            best_val_idx = int(np.argmax(val_values))
            ax.axvline(x=best_val_idx + 1, color="#ffff00", linestyle="--", alpha=0.5)
            ax.scatter(
                best_val_idx + 1, val_values[best_val_idx],
                color="#ffff00", s=100, zorder=5,
                label=f"Best {val_values[best_val_idx]:.4f}",
            )

        ax.set_title(title, fontsize=12, color="white")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(metric.upper(), color="white")
        ax.legend(fontsize=10)
        ax.tick_params(colors="white")
        ax.set_facecolor("#0f0f23")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(alpha=0.2)

    plt.tight_layout()

    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    print(f"    Training curves saved: {save_path}")
