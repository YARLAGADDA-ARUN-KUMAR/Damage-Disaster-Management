"""
Standalone Inference and Disaster Assessment Tool for TerraShield AI.
Generates pixel-level damage overlays and executive impact reports from satellite image pairs.

Usage:
    python -m src.predict --tile-id hurricane-florence_00000029
    python -m src.predict --pre path/to/pre.png --post path/to/post.png --output outputs/report.png --tta
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import torch

from src.config import (
    CHECKPOINT_ROOT,
    DAMAGE_CLASSES,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OUTPUT_ROOT,
    USE_AMP,
    DATASET_ROOT,
)
from src.model import DualHeadSiameseUNet

NORM_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
NORM_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

COLOR_PALETTE = {
    0: (0, 0, 0, 0),             # Background (transparent)
    1: (46, 204, 113, 220),       # No damage: Emerald Green (#2ECC71)
    2: (241, 196, 15, 220),       # Minor damage: Sunflower Yellow (#F1C40F)
    3: (230, 126, 34, 220),       # Major damage: Carrot Orange (#E67E22)
    4: (231, 76, 60, 220),        # Destroyed: Alizarin Red (#E74C3C)
}


def load_and_preprocess_image(image_path: Path):
    """Loads an RGB image and converts it into a standardized PyTorch tensor."""
    raw_img = Image.open(image_path).convert("RGB")
    raw_np = np.array(raw_img)
    t = torch.from_numpy(raw_np.astype(np.float32) / 255.0).permute(2, 0, 1)
    t_norm = (t - NORM_MEAN) / NORM_STD
    return raw_np, t_norm.unsqueeze(0)


def create_damage_overlay(background_np: np.ndarray, pred_mask_np: np.ndarray) -> np.ndarray:
    """Overlays semi-transparent color-coded damage masks onto the post-disaster image."""
    h, w, _ = background_np.shape
    overlay = background_np.astype(np.float32)

    for class_id in range(1, 5):
        mask = pred_mask_np == class_id
        if not np.any(mask):
            continue
        r, g, b, alpha = COLOR_PALETTE[class_id]
        alpha_factor = alpha / 255.0
        overlay[mask] = (1.0 - alpha_factor) * overlay[mask] + alpha_factor * np.array([r, g, b])

    return np.clip(overlay, 0, 255).astype(np.uint8)


def calculate_damage_statistics(pred_mask_np: np.ndarray):
    """Calculates pixel counts, percentages, and severity classification."""
    total_pixels = pred_mask_np.size
    building_mask = pred_mask_np > 0
    total_building_pixels = int(np.sum(building_mask))

    class_counts = {}
    class_pcts = {}

    for c in range(1, 5):
        cnt = int(np.sum(pred_mask_np == c))
        class_counts[c] = cnt
        pct = (cnt / total_building_pixels * 100.0) if total_building_pixels > 0 else 0.0
        class_pcts[c] = pct

    # Damaged pixels (classes 2, 3, 4)
    damaged_pixels = class_counts[2] + class_counts[3] + class_counts[4]
    damage_ratio = (damaged_pixels / total_building_pixels * 100.0) if total_building_pixels > 0 else 0.0

    if damage_ratio < 10.0:
        severity = "LOW / MINIMAL"
        severity_color = "#27ae60"
    elif damage_ratio < 30.0:
        severity = "MODERATE"
        severity_color = "#f39c12"
    elif damage_ratio < 55.0:
        severity = "SEVERE"
        severity_color = "#e67e22"
    else:
        severity = "CATASTROPHIC"
        severity_color = "#c0392b"

    return {
        "total_building_pixels": total_building_pixels,
        "class_counts": class_counts,
        "class_pcts": class_pcts,
        "damaged_pixels": damaged_pixels,
        "damage_ratio": damage_ratio,
        "severity": severity,
        "severity_color": severity_color,
    }


def plot_executive_dashboard(
    pre_np: np.ndarray,
    post_np: np.ndarray,
    overlay_np: np.ndarray,
    stats: dict,
    save_path: Path,
    tile_name: str,
):
    """Generates an executive 4-panel disaster assessment dashboard."""
    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.9])

    # 1. Pre-Disaster
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(pre_np)
    ax0.set_title("Pre-Disaster Imagery", fontsize=13, fontweight="bold", pad=8)
    ax0.axis("off")

    # 2. Post-Disaster
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(post_np)
    ax1.set_title("Post-Disaster Imagery", fontsize=13, fontweight="bold", pad=8)
    ax1.axis("off")

    # 3. Damage Overlay
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(overlay_np)
    ax2.set_title("AI Damage Assessment Overlay", fontsize=13, fontweight="bold", pad=8)
    ax2.axis("off")

    # Legend on overlay
    legend_patches = [
        mpatches.Patch(color=np.array([46, 204, 113])/255, label="No Damage"),
        mpatches.Patch(color=np.array([241, 196, 15])/255, label="Minor Damage"),
        mpatches.Patch(color=np.array([230, 126, 34])/255, label="Major Damage"),
        mpatches.Patch(color=np.array([231, 76, 60])/255, label="Destroyed"),
    ]
    ax2.legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=9)

    # 4. Analytics Panel
    ax3 = fig.add_subplot(gs[3])
    ax3.set_facecolor("#f8f9fa")
    ax3.axis("off")

    # Title & Severity banner
    ax3.text(0.5, 0.95, "DISASTER IMPACT REPORT", ha="center", va="top", fontsize=13, fontweight="bold")
    ax3.text(0.5, 0.88, f"Tile: {tile_name}", ha="center", va="top", fontsize=9, color="#7f8c8d")

    # Severity pill
    ax3.text(
        0.5, 0.78,
        f"SEVERITY: {stats['severity']}",
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=stats["severity_color"], edgecolor="none"),
    )

    # Damage Breakdown Bars
    categories = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
    pcts = [stats["class_pcts"][i] for i in range(1, 5)]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    y_positions = [0.60, 0.48, 0.36, 0.24]
    for y, cat, pct, color in zip(y_positions, categories, pcts, colors):
        ax3.text(0.05, y + 0.04, f"{cat}: {pct:.1f}%", fontsize=9, fontweight="bold", color="#2c3e50")
        # Draw background bar
        ax3.barh(y, 0.9, height=0.03, left=0.05, color="#ecf0f1", align="edge")
        # Draw fill bar
        bar_len = (pct / 100.0) * 0.9
        if bar_len > 0:
            ax3.barh(y, bar_len, height=0.03, left=0.05, color=color, align="edge")

    # Total footprint area
    ax3.text(
        0.5, 0.08,
        f"Building Footprint Area: {stats['total_building_pixels']:,} px\nImpact Ratio: {stats['damage_ratio']:.1f}% Damaged",
        ha="center", va="bottom", fontsize=9, color="#34495e",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#ffffff", edgecolor="#bdc3c7"),
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def find_tile_paths(tile_id: str):
    """Searches train, hold, and test splits for the specified tile ID."""
    for split in ["test", "hold", "train"]:
        split_dir = DATASET_ROOT / split / "images"
        pre_path = split_dir / f"{tile_id}_pre_disaster.png"
        post_path = split_dir / f"{tile_id}_post_disaster.png"
        if pre_path.exists() and post_path.exists():
            return pre_path, post_path
    raise FileNotFoundError(f"Could not find tile '{tile_id}' in dataset (checked test, hold, train).")


def run_prediction(
    pre_path: Path,
    post_path: Path,
    checkpoint_path: Path = CHECKPOINT_ROOT / "dual_head_siamese_best.pth",
    output_path: Path = None,
    threshold: float = 0.35,
    use_tta: bool = True,
):
    print("=" * 70)
    print("TERRASHIELD AI — RAPID DISASTER DAMAGE ASSESSMENT")
    print("=" * 70)
    print(f"Pre-disaster image : {pre_path}")
    print(f"Post-disaster image: {post_path}")
    print(f"Checkpoint         : {checkpoint_path}")
    print(f"Localization Thresh: {threshold} | TTA: {use_tta}")

    tile_name = pre_path.stem.replace("_pre_disaster", "")
    if output_path is None:
        output_path = OUTPUT_ROOT / f"assessment_{tile_name}.png"

    # 1. Load model
    model = DualHeadSiameseUNet(pretrained=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 2. Preprocess images
    pre_np, pre_t = load_and_preprocess_image(pre_path)
    post_np, post_t = load_and_preprocess_image(post_path)

    pre_t = pre_t.to(DEVICE)
    post_t = post_t.to(DEVICE)

    # 3. Model Inference
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP, dtype=torch.float16):
            if use_tta:
                pred_mask, _, _ = model.predict_tta(pre_t, post_t, loc_threshold=threshold)
            else:
                pred_mask, _, _ = model.predict(pre_t, post_t, loc_threshold=threshold)

    pred_mask_np = pred_mask[0].cpu().numpy()

    # 4. Generate damage statistics
    stats = calculate_damage_statistics(pred_mask_np)

    # 5. Generate visual overlay
    overlay_np = create_damage_overlay(post_np, pred_mask_np)

    # 6. Save executive dashboard
    plot_executive_dashboard(pre_np, post_np, overlay_np, stats, output_path, tile_name)

    # 7. Print formatted terminal summary
    print("\n" + "=" * 70)
    print("ASSESSMENT SUMMARY")
    print("=" * 70)
    print(f"Total Building Pixels: {stats['total_building_pixels']:,}")
    print(f"  - No Damage (Intact): {stats['class_counts'][1]:8,d} ({stats['class_pcts'][1]:5.1f}%)")
    print(f"  - Minor Damage      : {stats['class_counts'][2]:8,d} ({stats['class_pcts'][2]:5.1f}%)")
    print(f"  - Major Damage      : {stats['class_counts'][3]:8,d} ({stats['class_pcts'][3]:5.1f}%)")
    print(f"  - Destroyed         : {stats['class_counts'][4]:8,d} ({stats['class_pcts'][4]:5.1f}%)")
    print("-" * 70)
    print(f"Damage Ratio         : {stats['damage_ratio']:.1f}%")
    print(f"Severity Assessment  : {stats['severity']}")
    print(f"Dashboard Saved To   : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraShield AI - Satellite Disaster Assessment")
    parser.add_argument("--tile-id", type=str, default=None, help="Tile ID in dataset (e.g. hurricane-florence_00000029)")
    parser.add_argument("--pre", type=str, default=None, help="Path to pre-disaster RGB image")
    parser.add_argument("--post", type=str, default=None, help="Path to post-disaster RGB image")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_ROOT / "dual_head_siamese_best.pth"))
    parser.add_argument("--output", type=str, default=None, help="Path to save assessment visualization")
    parser.add_argument("--threshold", type=float, default=0.35, help="Localization probability threshold")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    args = parser.parse_args()

    if args.tile_id:
        pre_img_path, post_img_path = find_tile_paths(args.tile_id)
    elif args.pre and args.post:
        pre_img_path = Path(args.pre)
        post_img_path = Path(args.post)
    else:
        # Default test tile
        pre_img_path, post_img_path = find_tile_paths("hurricane-florence_00000029")

    out_p = Path(args.output) if args.output else None
    run_prediction(
        pre_path=pre_img_path,
        post_path=post_img_path,
        checkpoint_path=Path(args.checkpoint),
        output_path=out_p,
        threshold=args.threshold,
        use_tta=not args.no_tta,
    )
