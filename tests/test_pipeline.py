"""
Comprehensive verification test suite for the TerraShield AI Dual-Head pipeline.
"""

import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAIN_CROP_SIZE,
    DAMAGE_CLASS_WEIGHTS,
)
from src.dataset import build_split_dataframe, DualHeadXView2Dataset, get_dataloaders
from src.model import DualHeadSiameseUNet
from src.loss import DualHeadLoss
from src.metrics import compute_confusion_matrix, calculate_metrics_from_confusion


def test_dataset_and_normalization():
    print("\n--- Test 1: Dataset & Normalization ---")
    df = build_split_dataframe("train")
    assert len(df) == 2799, f"Expected 2799 training samples, got {len(df)}"
    print(f"[PASSED] Indexed {len(df)} train samples.")

    ds = DualHeadXView2Dataset(df.iloc[:10], split="train", augment=True, crop_size=TRAIN_CROP_SIZE)
    sample = ds[0]

    pre = sample["pre_image"]
    post = sample["post_image"]
    loc_tgt = sample["loc_target"]
    dmg_tgt = sample["damage_target"]
    raw_tgt = sample["raw_target"]

    assert pre.shape == (3, 512, 512), f"Unexpected pre shape: {pre.shape}"
    assert post.shape == (3, 512, 512), f"Unexpected post shape: {post.shape}"
    assert loc_tgt.shape == (512, 512), f"Unexpected loc target shape: {loc_tgt.shape}"
    assert dmg_tgt.shape == (512, 512), f"Unexpected dmg target shape: {dmg_tgt.shape}"

    # Check normalization: ImageNet standard values range roughly between -2.5 and +2.5
    print(f"Pre-image mean: {pre.mean():.3f}, std: {pre.std():.3f}")
    assert -3.0 < pre.mean() < 3.0
    print("[PASSED] Dataset shapes and ImageNet normalization verified.")


def test_model_forward_backward():
    print("\n--- Test 2: Model Forward & Backward Pass ---")
    model = DualHeadSiameseUNet(pretrained=False).to(DEVICE)
    model.train()

    pre = torch.randn(2, 3, 512, 512, device=DEVICE)
    post = torch.randn(2, 3, 512, 512, device=DEVICE)

    loc_logits, dmg_logits = model(pre, post)
    assert loc_logits.shape == (2, 1, 512, 512), f"Unexpected loc logits shape: {loc_logits.shape}"
    assert dmg_logits.shape == (2, 4, 512, 512), f"Unexpected dmg logits shape: {dmg_logits.shape}"
    print(f"[PASSED] Forward pass logits: Loc={loc_logits.shape}, Dmg={dmg_logits.shape}")

    # Loss
    criterion = DualHeadLoss().to(DEVICE)
    loc_tgt = torch.randint(0, 2, (2, 512, 512), device=DEVICE, dtype=torch.float32)
    dmg_tgt = torch.randint(0, 4, (2, 512, 512), device=DEVICE, dtype=torch.long)

    loss_dict = criterion(loc_logits, dmg_logits, loc_tgt, dmg_tgt)
    total_loss = loss_dict["total_loss"]
    assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss}"

    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    print(f"[PASSED] Loss: {total_loss.item():.4f} with clean gradients across all parameters.")


def test_predict_and_metrics():
    print("\n--- Test 3: Prediction Logic & Metrics ---")
    model = DualHeadSiameseUNet(pretrained=False).to(DEVICE)
    model.eval()

    pre = torch.randn(2, 3, 512, 512, device=DEVICE)
    post = torch.randn(2, 3, 512, 512, device=DEVICE)

    final_pred, loc_probs, dmg_probs = model.predict(pre, post)
    assert final_pred.shape == (2, 512, 512)
    assert loc_probs.shape == (2, 512, 512)
    assert dmg_probs.shape == (2, 4, 512, 512)

    # Values in final_pred must be in range 0..4
    unique_vals = set(final_pred.cpu().numpy().flatten())
    assert unique_vals.issubset({0, 1, 2, 3, 4})

    # Metrics
    targets = torch.randint(0, 5, (2, 512, 512), device=DEVICE)
    confusion = compute_confusion_matrix(final_pred, targets)
    assert confusion.shape == (5, 5)

    metrics = calculate_metrics_from_confusion(confusion)
    assert "loc_f1" in metrics
    assert "fg_macro_f1" in metrics
    assert "xview2_score" in metrics
    assert 0.0 <= metrics["xview2_score"] <= 1.0
    print(f"[PASSED] Metrics computation verified (xView2 Score: {metrics['xview2_score']:.4f}).")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TERRASHIELD AI PIPELINE TESTS")
    print("=" * 60)
    test_dataset_and_normalization()
    test_model_forward_backward()
    test_predict_and_metrics()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
