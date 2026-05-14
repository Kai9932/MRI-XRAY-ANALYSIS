"""
Usage:
  conda activate PY31115
  python train.py

Optional arguments (modify corresponding Config class fields directly in config.py):
  - Modify ModelConfig.model_name to change model architecture
  - Modify TrainConfig.num_epochs to adjust training epochs
  - Modify DataConfig.batch_size to adjust batch size (reduce if VRAM is insufficient)
"""

import sys
import warnings
from pathlib import Path

import torch

# Add current directory to module search path (ensure peer .py files can be found)
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    """Main Training Function"""

    # ── Step 0: Import all modules (delayed to here for quick import error location) ──
    from config import cfg
    from dataset import build_dataloaders, set_seed
    from model import build_model
    from trainer import Trainer
    from visualize import plot_training_history, visualize_gradcam_batch

    print("\n      Brain Cancer MRI Detection Pipeline")
    print("  Tech Stack: PyTorch + Swin Transformer + Albumentations\n")

    # ── Step 1: Set Random Seed ──
    set_seed(cfg.train.seed)
    print(f"    Random seed fixed: {cfg.train.seed}")

    # ── Step 2: Print Current Configuration Summary ──
    print(f"\n    Training Configuration Summary:")
    print(f"     Data Directory  : {cfg.data.data_dir}")
    print(f"     Image Size      : {cfg.data.image_size}")
    print(f"     Batch Size      : {cfg.data.batch_size}")
    print(f"     Model Arch      : {cfg.model.model_name}")
    print(f"     Total Epochs    : {cfg.train.num_epochs}")
    print(f"     Initial LR      : Head={cfg.train.lr_head:.0e} | Backbone={cfg.train.lr_backbone:.0e}")
    print(f"     Mixed Precision : {'Enabled' if cfg.train.use_amp else 'Disabled'}")
    print(f"     Early Stop      : {cfg.train.early_stopping_patience} epochs")

    # ── Step 3: Build Dataset ──
    train_loader, val_loader, test_loader, data_info = build_dataloaders(
        data_cfg=cfg.data,
        aug_cfg=cfg.aug,
        seed=cfg.train.seed,
    )

    # Use auto-calculated value from dataset if pos_weight is not manually set in config
    pos_weight = cfg.train.pos_weight or data_info["pos_weight"]

    # ── Step 4: Build Model ──
    model, device = build_model(model_cfg=cfg.model)

    # ── Step 5: Initialize Trainer ──
    trainer = Trainer(
        model=model,
        train_cfg=cfg.train,
        device=device,
        pos_weight=pos_weight,
    )

    # ── Step 6: Start Training ──
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # ── Step 7: Final Evaluation on Test Set ──
    test_metrics = trainer.evaluate_test(test_loader=test_loader)

    # ── Step 8: Plot Training Curves ──
    from config import OUTPUT_DIR
    plot_training_history(
        history=history,
        output_dir=OUTPUT_DIR / "visualizations",
        filename="training_curves.png",
    )

    # ── Step 9: Generate Grad-CAM Visualization (using test set images) ──
    print("\n    Generating Grad-CAM visualizations...")

    # Reload best weights (already loaded in evaluate_test)
    test_dataset = test_loader.dataset

    visualize_gradcam_batch(
        model=model,
        dataset=test_dataset,
        device=device,
        viz_cfg=cfg.viz,
        num_images=8,
        output_dir=cfg.viz.output_dir,
        filename="gradcam_test_samples.png",
    )

    # ── Done! Print Summary ──
    print("\n      Training Complete! Results Summary")
    print(f"  Test Set AUC-ROC  : {test_metrics['auc']:.4f}")
    print(f"  Test Set F1-Score : {test_metrics['f1']:.4f}")
    print(f"  Test Set MCC      : {test_metrics['mcc']:.4f}")
    print(f"  Test Set Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"     Best Weights: outputs/checkpoints/best_model.pth")
    print(f"     Train Curves: outputs/visualizations/training_curves.png")
    print(f"     Grad-CAM    : outputs/visualizations/gradcam_*.png\n")


if __name__ == "__main__":
    main()
