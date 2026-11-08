"""
Evaluation script for TerraShield AI Dual-Head Siamese U-Net.
Evaluates on the test split, prints metrics, and exports visual confusion matrices and predictions.
Usage:
    python -m src.evaluate --checkpoint checkpoints/dual_head_siamese_best.pth --split test
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import (
    CHECKPOINT_ROOT,
    CLASS_MAPPING_5,
    DEVICE,
    NUM_TOTAL_CLASSES,
    OUTPUT_ROOT,
    USE_AMP,
)
from src.dataset import get_dataloaders
from src.metrics import calculate_metrics_from_confusion, compute_confusion_matrix
from src.model import DualHeadSiameseUNet, EnsembleDualHeadModel


def plot_confusion_matrix(confusion: np.ndarray, save_path: Path):
    class_names = [CLASS_MAPPING_5[i] for i in range(NUM_TOTAL_CLASSES)]
    row_sums = confusion.sum(axis=1, keepdims=True)
    norm_confusion = np.divide(
        confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums > 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    im0 = axes[0].imshow(confusion, cmap="Blues")
    axes[0].set_title("Test Confusion Matrix (Pixel Counts)")
    axes[0].set_xticks(range(NUM_TOTAL_CLASSES))
    axes[0].set_yticks(range(NUM_TOTAL_CLASSES))
    axes[0].set_xticklabels(class_names, rotation=35, ha="right")
    axes[0].set_yticklabels(class_names)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # Normalized recall %
    im1 = axes[1].imshow(norm_confusion, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Test Confusion Matrix (Row-Normalized / Recall)")
    axes[1].set_xticks(range(NUM_TOTAL_CLASSES))
    axes[1].set_yticks(range(NUM_TOTAL_CLASSES))
    axes[1].set_xticklabels(class_names, rotation=35, ha="right")
    axes[1].set_yticklabels(class_names)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    for i in range(NUM_TOTAL_CLASSES):
        for j in range(NUM_TOTAL_CLASSES):
            val_raw = f"{confusion[i, j]:,}"
            val_norm = f"{norm_confusion[i, j]:.1%}"
            color_raw = "white" if confusion[i, j] > confusion.max() / 2 else "black"
            color_norm = "white" if norm_confusion[i, j] > 0.5 else "black"
            axes[0].text(j, i, val_raw, ha="center", va="center", color=color_raw, fontsize=8)
            axes[1].text(j, i, val_norm, ha="center", va="center", color=color_norm, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_visual_samples(samples, save_path: Path):
    """Plots side-by-side visual comparisons: Pre, Post, Ground Truth, Prediction."""
    import matplotlib.colors as mcolors
    colors = ["#2b2b2b", "#2ca02c", "#ffbb78", "#ff7f0e", "#d62728"]
    cmap = mcolors.ListedColormap(colors)

    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, s in enumerate(samples):
        # Denormalize RGB
        pre_rgb = s["pre_img"].permute(1, 2, 0).cpu().numpy()
        post_rgb = s["post_img"].permute(1, 2, 0).cpu().numpy()
        pre_rgb = np.clip(pre_rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        post_rgb = np.clip(post_rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        gt = s["gt_mask"].cpu().numpy()
        pred = s["pred_mask"].cpu().numpy()

        axes[i, 0].imshow(pre_rgb)
        axes[i, 0].set_title(f"Pre-Disaster: {s['tile_id']}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(post_rgb)
        axes[i, 1].set_title("Post-Disaster")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gt, cmap=cmap, vmin=0, vmax=4)
        axes[i, 2].set_title("Ground Truth Mask")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred, cmap=cmap, vmin=0, vmax=4)
        axes[i, 3].set_title("Predicted Mask (Dual-Head)")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def run_evaluation(
    checkpoint_path: Path,
    split: str = "test",
    threshold: float = 0.35,
    use_tta: bool = True,
    backbone: str = "resnet34",
    ensemble: bool = False,
    majority_vote: bool = False,
):
    print("=" * 70)
    print("TERRASHIELD AI — EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Evaluation mode    : {'ENSEMBLE (ResNet-34 + EfficientNet-B2)' if ensemble else f'SINGLE MODEL ({backbone.upper()})'}")
    print(f"Evaluation split   : {split.upper()}")
    print(f"Localization Thresh: {threshold} | TTA: {use_tta} | Majority Voting: {majority_vote}")

    _, hold_loader, test_loader = get_dataloaders()
    loader = test_loader if split == "test" else hold_loader

    if ensemble:
        m1 = DualHeadSiameseUNet(backbone="resnet34", pretrained=False).to(DEVICE)
        ckpt1_path = CHECKPOINT_ROOT / "dual_head_siamese_best.pth"
        c1 = torch.load(ckpt1_path, map_location=DEVICE, weights_only=False)
        m1.load_state_dict(c1["model_state_dict"] if "model_state_dict" in c1 else c1)

        m2 = DualHeadSiameseUNet(backbone="efficientnet_b2", pretrained=False).to(DEVICE)
        ckpt2_path = CHECKPOINT_ROOT / "dual_head_efficientnet_b2_best.pth"
        if not ckpt2_path.exists():
            ckpt2_path = checkpoint_path
        c2 = torch.load(ckpt2_path, map_location=DEVICE, weights_only=False)
        m2.load_state_dict(c2["model_state_dict"] if "model_state_dict" in c2 else c2)

        model = EnsembleDualHeadModel([m1, m2]).eval()
    else:
        model = DualHeadSiameseUNet(backbone=backbone, pretrained=False).to(DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

    if majority_vote:
        from src.vectorize import vectorize_predictions

    confusion = np.zeros((NUM_TOTAL_CLASSES, NUM_TOTAL_CLASSES), dtype=np.int64)
    visual_samples = []

    pbar = tqdm(loader, desc=f"Evaluating on {split}")
    for batch in pbar:
        pre_img = batch["pre_image"].to(DEVICE, non_blocking=True)
        post_img = batch["post_image"].to(DEVICE, non_blocking=True)
        raw_tgt = batch["raw_target"].to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP, dtype=torch.float16):
            if use_tta:
                final_pred, loc_probs, damage_probs = model.predict_tta(
                    pre_img, post_img, loc_threshold=threshold
                )
            else:
                final_pred, loc_probs, damage_probs = model.predict(
                    pre_img, post_img, loc_threshold=threshold
                )

        if majority_vote:
            pred_np = final_pred[0].cpu().numpy()
            loc_np = (loc_probs[0].cpu().numpy() >= threshold).astype(np.uint8)
            dmg_np = damage_probs[0].cpu().numpy()
            voted_np, _ = vectorize_predictions(loc_np, dmg_np, min_area=15)
            final_pred = torch.from_numpy(voted_np).unsqueeze(0).to(DEVICE)

        confusion += compute_confusion_matrix(final_pred, raw_tgt)

        # Collect 3 representative samples with damage for visualization
        if len(visual_samples) < 3 and torch.any(raw_tgt >= 2):
            visual_samples.append({
                "tile_id": batch["tile_id"][0],
                "pre_img": pre_img[0],
                "post_img": post_img[0],
                "gt_mask": raw_tgt[0],
                "pred_mask": final_pred[0],
            })

    metrics = calculate_metrics_from_confusion(confusion)

    print("\n" + "=" * 70)
    print(f"{split.upper()} CLASS-WISE PERFORMANCE")
    print("=" * 70)
    print(metrics["class_table"].to_string(index=False))

    print("\n" + "=" * 70)
    print("KEY PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Building Localization F1 : {metrics['loc_f1']:.4f} (IoU: {metrics['loc_iou']:.4f})")
    print(f"Damage Macro F1          : {metrics['fg_macro_f1']:.4f}")
    print(f"Foreground mIoU          : {metrics['fg_miou']:.4f}")
    print(f"Official xView2 Score    : {metrics['xview2_score']:.4f} (0.3*Loc + 0.7*Damage)")
    print("=" * 70)

    # Save artifacts
    cm_path = OUTPUT_ROOT / f"{split}_confusion_matrix.png"
    plot_confusion_matrix(confusion, cm_path)
    print(f"[*] Confusion matrix saved: {cm_path}")

    if visual_samples:
        vis_path = OUTPUT_ROOT / f"{split}_predictions_visual.png"
        plot_visual_samples(visual_samples, vis_path)
        print(f"[*] Prediction visual saved: {vis_path}")

    # Save json summary
    json_path = OUTPUT_ROOT / f"{split}_metrics_report.json"
    summary_data = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "loc_f1": metrics["loc_f1"],
        "loc_iou": metrics["loc_iou"],
        "fg_macro_f1": metrics["fg_macro_f1"],
        "fg_miou": metrics["fg_miou"],
        "xview2_score": metrics["xview2_score"],
        "class_metrics": metrics["class_table"].to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[*] Metrics JSON report saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dual-Head Siamese U-Net")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_ROOT / "dual_head_siamese_best.pth"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "hold"], help="Dataset split")
    parser.add_argument("--threshold", type=float, default=0.35, help="Localization probability threshold")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet34", "efficientnet_b2"], help="Model backbone")
    parser.add_argument("--ensemble", action="store_true", help="Evaluate ensemble of ResNet-34 and EfficientNet-B2")
    parser.add_argument("--majority-vote", action="store_true", help="Apply building-level majority voting")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        threshold=args.threshold,
        use_tta=not args.no_tta,
        backbone=args.backbone,
        ensemble=args.ensemble,
        majority_vote=args.majority_vote,
    )

