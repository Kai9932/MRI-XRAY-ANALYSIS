
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent          # d:\tools\ML
DATA_DIR = Path(r"C:\Users\User\Desktop\WK xmum\ML\MRI-XRAY-ANALYSIS\archive\brain_dataset")              # Raw data root directory
OUTPUT_DIR = BASE_DIR / "outputs"                   # Output directory (weights, logs, plots)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"         # Model weight saving path
LOG_DIR = OUTPUT_DIR / "logs"                       # TensorBoard logs path
VIZ_DIR = OUTPUT_DIR / "visualizations"             # Grad-CAM visualization output

# Automatically create all output directories
for _dir in [CHECKPOINT_DIR, LOG_DIR, VIZ_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────
# Data Configuration
# ──────────────────────────────────────────────────────
@dataclass
class DataConfig:
    # Dataset directory (contains yes/ and no/ subfolders)
    data_dir: Path = DATA_DIR

    # Uniform image resize dimension (Swin Transformer standard input)
    image_size: Tuple[int, int] = (224, 224)

    # Train/Val/Test split ratio (sum must be 1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # DataLoader parameters
    batch_size: int = 16              # Number of samples per batch (reduce if VRAM is insufficient)
    num_workers: int = 4              # Data loading parallel workers
    pin_memory: bool = True           # Pin memory to accelerate GPU transfer

    # ImageNet normalization mean and std (must be consistent for transfer learning)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Class names (folder name is label name)
    class_names: List[str] = field(default_factory=lambda: ["no", "yes"])  # no=0, yes=1
    num_classes: int = 1              # Binary classification uses a single neuron Sigmoid output

    # Random seed (ensure experiment reproducibility)
    seed: int = 42


# ──────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # timm model name (swin_base is ~88M parameters, feasible for consumer GPUs)
    # If GPU VRAM < 8GB, consider changing to swin_tiny_patch4_window7_224
    model_name: str = "swin_tiny_patch4_window7_224"

    # Whether to use ImageNet pretrained weights (Highly recommended True, significantly accelerates convergence)
    pretrained: bool = True

    # Whether to freeze the backbone network in the initial training phase (True = train only classifier head, fast, prevents overfitting)
    freeze_backbone: bool = True

    # Number of epochs to train in the frozen phase before unfreezing the backbone (Fine-tuning)
    unfreeze_at_epoch: int = 5

    # Dropout rate (increase to prevent overfitting)
    dropout_rate: float = 0.5

    # Classifier head hidden dimension
    classifier_hidden_dim: int = 512

    # Drop path rate (Stochastic Depth) for Transformer backbone to prevent overfitting
    drop_path_rate: float = 0.2


# ──────────────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Total number of training epochs
    num_epochs: int = 50

    # Initial learning rate (use larger lr for training the classifier head during frozen phase)
    lr_head: float = 1e-3

    # Learning rate for Fine-tuning phase (use very small lr to fine-tune backbone after unfreezing)
    lr_backbone: float = 1e-5

    # Weight decay (L2 regularization, increase to prevent overfitting)
    weight_decay: float = 1e-3

    # Minimum learning rate for CosineAnnealing learning rate scheduler
    lr_min: float = 1e-7

    # Automatic Mixed Precision training (AMP): True saves VRAM and speeds up, requires CUDA GPU support
    use_amp: bool = True

    # Early Stopping: terminate early if validation AUC does not improve for N consecutive epochs
    early_stopping_patience: int = 10

    # Model weights save path
    checkpoint_path: Path = CHECKPOINT_DIR / "best_model.pth"

    # Maximum norm for gradient clipping (prevents exploding gradients)
    grad_clip_norm: float = 1.0

    # Positive class weight (used to handle class imbalance, None = auto calculate)
    pos_weight: Optional[float] = None

    # Label Smoothing coefficient, increase to prevent overfitting
    label_smoothing: float = 0.2

    # TensorBoard logs directory
    log_dir: Path = LOG_DIR

    # Random seed
    seed: int = 42


# ──────────────────────────────────────────────────────
# Data Augmentation Configuration (Albumentations Parameters)
# ──────────────────────────────────────────────────────
@dataclass
class AugConfig:
    # ── Training Set Augmentation (aggressive augmentation, improves generalization) ──
    # Random rotation angle range
    rotate_limit: int = 30
    # Horizontal/Vertical flip probability
    flip_prob: float = 0.5
    # Random brightness/contrast adjustment range
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    # Gaussian noise standard deviation range (simulates MRI scanner noise)
    gauss_noise_var: Tuple[float, float] = (10.0, 50.0)
    # CLAHE contrast enhancement (common in medical imaging, highlights lesion boundaries)
    clahe_prob: float = 0.3
    clahe_clip_limit: float = 4.0
    # Elastic deformation (simulates slight soft tissue deformation)
    elastic_prob: float = 0.2
    elastic_alpha: float = 120.0
    elastic_sigma: float = 6.0
    # Grid distortion (supplements elastic deformation)
    grid_distort_prob: float = 0.2
    # Random crop and resize back to original dimension (enhances local feature learning)
    random_resized_crop_prob: float = 0.3
    # CoarseDropout (random occlusion, equivalent to Cutout, prevents overfitting)
    cutout_prob: float = 0.3
    cutout_max_holes: int = 8
    cutout_max_height: int = 32
    cutout_max_width: int = 32


# ──────────────────────────────────────────────────────
# Visualization Configuration
# ──────────────────────────────────────────────────────
@dataclass
class VizConfig:
    # Grad-CAM target layer name (None = automatically infer the last convolutional layer)
    target_layer: str = None
    # Number of images displayed per visualization
    num_images: int = 8
    # Visualization output directory
    output_dir: Path = VIZ_DIR
    # Heatmap transparency (0 = completely original image, 1 = completely heatmap)
    heatmap_alpha: float = 0.4
    # Colormap scheme
    colormap: str = "jet"


# ──────────────────────────────────────────────────────
# Main Configuration Class integrating all configs
# ──────────────────────────────────────────────────────
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    viz: VizConfig = field(default_factory=VizConfig)


# Global default configuration instance (can be imported directly by other modules)
cfg = Config()
