"""
Features:
  - Automatically scan and collect all image paths and labels from yes/no folders
  - Supports Albumentations data augmentation (specialized for medical imaging)
  - Automatically handles class imbalance (calculates pos_weight)
  - Stratified splitting of dataset according to train/val/test ratio
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import AugConfig, DataConfig, cfg


# ──────────────────────────────────────────────────────
# Helper Function: Set global random seed (ensure reproducibility)
# ──────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix all random sources to make experimental results reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────
# Core Function: Scan folders, collect all image paths and corresponding labels
# ──────────────────────────────────────────────────────
def collect_image_paths(
    data_dir: Path,
    class_names: List[str],
) -> Tuple[List[Path], List[int]]:
    """
    Collect images from class_names subfolders under data_dir directory.

    Args:
        data_dir:     Root directory (contains yes/ and no/ subfolders)
        class_names:  List of class names, corresponding to label indices (e.g., ["no", "yes"])

    Returns:
        image_paths:  List of absolute paths for all images
        labels:       List of corresponding labels (integer, no=0, yes=1)
    """
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_paths: List[Path] = []
    labels: List[int] = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"[WARNING] Directory does not exist, skipping: {class_dir}")
            continue

        # Traverse all images in this class folder
        found = 0
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                image_paths.append(img_path)
                labels.append(label_idx)
                found += 1

        print(f"    Class '{class_name}' (label={label_idx}): found {found} images")

    print(f"\n  Total: {len(image_paths)} images | Number of classes: {len(class_names)}")
    return image_paths, labels


# ──────────────────────────────────────────────────────
# Data Augmentation Transform Builder
# ──────────────────────────────────────────────────────
def build_transforms(
    aug_cfg: AugConfig,
    data_cfg: DataConfig,
    phase: str,  # "train" | "val" | "test"
) -> A.Compose:
    """
    Build Albumentations data augmentation pipeline based on the phase.

    Training set: Aggressive augmentation to improve generalization
    Validation/Test set: Only resize and normalize to ensure fair evaluation
    """
    h, w = data_cfg.image_size

    if phase == "train":
        transforms = A.Compose([
            # ── Geometric Transformations (simulate different scan angles and patient poses) ──
            A.Resize(height=h, width=w),
            A.HorizontalFlip(p=aug_cfg.flip_prob),
            A.VerticalFlip(p=aug_cfg.flip_prob * 0.5),   # Vertical flip probability is half
            A.Rotate(limit=aug_cfg.rotate_limit, p=0.7,
                     border_mode=cv2.BORDER_REFLECT_101),

            # ── Medical Imaging Specialized: CLAHE Contrast Enhancement ──
            # Can significantly improve lesion visibility in low-contrast MRI images
            A.CLAHE(
                clip_limit=aug_cfg.clahe_clip_limit,
                tile_grid_size=(8, 8),
                p=aug_cfg.clahe_prob,
            ),

            # ── Random Brightness/Contrast Jitter (simulate MRI machine parameter differences) ──
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.brightness_limit,
                contrast_limit=aug_cfg.contrast_limit,
                p=0.5,
            ),

            # ── Elastic Deformation (simulate real slight soft tissue deformation) ──
            A.ElasticTransform(
                alpha=aug_cfg.elastic_alpha,
                sigma=aug_cfg.elastic_sigma,
                p=aug_cfg.elastic_prob,
            ),

            # ── Grid Distortion (supplements elastic deformation, more diverse shape changes) ──
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=aug_cfg.grid_distort_prob,
            ),

            # ── Gaussian Noise (simulate MRI scanner thermal noise) ──
            A.GaussNoise(
                var_limit=aug_cfg.gauss_noise_var,
                p=0.4,
            ),

            # ── Gaussian Blur (simulate motion artifacts) ──
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),

            # ── Cutout / CoarseDropout (random occlusion to prevent overfitting) ──
            A.CoarseDropout(
                max_holes=aug_cfg.cutout_max_holes,
                max_height=aug_cfg.cutout_max_height,
                max_width=aug_cfg.cutout_max_width,
                fill_value=0,
                p=aug_cfg.cutout_prob,
            ),

            # ── Normalization (ImageNet mean/std, aligns with pretrained weights) ──
            A.Normalize(mean=data_cfg.mean, std=data_cfg.std),
            ToTensorV2(),  # Convert HWC numpy array to CHW PyTorch Tensor
        ])
    else:
        # Validation/Test phase: Only Resize + Normalize, no random augmentation
        transforms = A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(mean=data_cfg.mean, std=data_cfg.std),
            ToTensorV2(),
        ])

    return transforms


# ──────────────────────────────────────────────────────
# Dataset Class: Encapsulates image reading and preprocessing
# ──────────────────────────────────────────────────────
class BrainMRIDataset(Dataset):
    """
    Brain MRI Image Dataset Class.

    Args:
        image_paths:  List of absolute image paths
        labels:       List of integer labels (0=No Tumor, 1=Tumor)
        transform:    Albumentations transform pipeline
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # ── Read image (OpenCV default is BGR, needs conversion to RGB) ──
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))

        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB

        # ── Apply data augmentation ──
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Type: torch.Tensor, shape: (C, H, W)

        # ── Convert label to float32 Tensor (BCEWithLogitsLoss requires float type) ──
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    def get_image_path(self, idx: int) -> Path:
        """Return the original image path for the given index (used by visualization module)"""
        return self.image_paths[idx]


# ──────────────────────────────────────────────────────
# Dataset Builder: One-click builder for train/val/test DataLoader
# ──────────────────────────────────────────────────────
def build_dataloaders(
    data_cfg: DataConfig,
    aug_cfg: AugConfig,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train/val/test DataLoader from raw folders.

    Returns:
        train_loader, val_loader, test_loader, info_dict
        info_dict contains: Class statistics, pos_weight (used to handle class imbalance)
    """
    set_seed(seed)

    print("\n    Scanning dataset directory...")

    # Step 1: Collect all image paths and labels
    all_paths, all_labels = collect_image_paths(data_cfg.data_dir, data_cfg.class_names)

    if len(all_paths) == 0:
        raise RuntimeError(f"No images found in {data_cfg.data_dir}! Please check the path.")

    # Step 2: Stratified split train/val+test (maintain class proportions)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=(1.0 - data_cfg.train_ratio),
        stratify=all_labels,           # Stratified sampling to maintain class ratio
        random_state=seed,
    )

    # Then split val+test evenly
    val_ratio_adjusted = data_cfg.val_ratio / (data_cfg.val_ratio + data_cfg.test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1.0 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=seed,
    )

    # Step 3: Count class distribution, calculate pos_weight (handle class imbalance)
    train_labels_arr = np.array(train_labels)
    num_neg = int((train_labels_arr == 0).sum())   # Number of negative samples in train set (No Tumor)
    num_pos = int((train_labels_arr == 1).sum())   # Number of positive samples in train set (Tumor)
    pos_weight = num_neg / max(num_pos, 1)         # pos_weight = neg/pos, auto compensates imbalance

    print(f"\n    Dataset Split Results:")
    print(f"     Train Set : {len(train_paths):>5} images (Neg={num_neg}, Pos={num_pos}, pos_weight={pos_weight:.2f})")
    print(f"     Val Set   : {len(val_paths):>5} images")
    print(f"     Test Set  : {len(test_paths):>5} images")


    # Step 4: Build Dataset for each phase (assigning corresponding augmentation transforms)
    train_dataset = BrainMRIDataset(
        train_paths, train_labels,
        transform=build_transforms(aug_cfg, data_cfg, phase="train"),
    )
    val_dataset = BrainMRIDataset(
        val_paths, val_labels,
        transform=build_transforms(aug_cfg, data_cfg, phase="val"),
    )
    test_dataset = BrainMRIDataset(
        test_paths, test_labels,
        transform=build_transforms(aug_cfg, data_cfg, phase="test"),
    )

    # Step 5: Build DataLoader (shuffle train set, do not shuffle val/test sets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,                             # Random shuffle per epoch
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=True,                           # Drop last incomplete batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    info = {
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "num_pos": num_pos,
        "num_neg": num_neg,
        "pos_weight": pos_weight,
        "class_names": data_cfg.class_names,
    }

    return train_loader, val_loader, test_loader, info
