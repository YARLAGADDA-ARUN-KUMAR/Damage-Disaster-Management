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
from src.model import DualHeadSiameseUNet, EnsembleDualHeadModel
from src.loss import DualHeadLoss
from src.metrics import compute_confusion_matrix, calculate_metrics_from_confusion
from src.vectorize import (
    classify_existing_polygons,
    export_geojson,
    majority_vote_damage,
    vectorize_predictions,
)


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
    print("\n--- Test 2: Model Forward & Backward Pass (ResNet-34 & EfficientNet-B2) ---")
    for backbone in ["resnet34", "efficientnet_b2"]:
        model = DualHeadSiameseUNet(backbone=backbone, pretrained=False).to(DEVICE)
        model.train()

        pre = torch.randn(2, 3, 512, 512, device=DEVICE)
        post = torch.randn(2, 3, 512, 512, device=DEVICE)

        loc_logits, dmg_logits = model(pre, post)
        assert loc_logits.shape == (2, 1, 512, 512), f"Unexpected loc logits shape for {backbone}"
        assert dmg_logits.shape == (2, 4, 512, 512), f"Unexpected dmg logits shape for {backbone}"

        criterion = DualHeadLoss().to(DEVICE)
        loc_tgt = torch.randint(0, 2, (2, 512, 512), device=DEVICE, dtype=torch.float32)
        dmg_tgt = torch.randint(0, 4, (2, 512, 512), device=DEVICE, dtype=torch.long)

        loss_dict = criterion(loc_logits, dmg_logits, loc_tgt, dmg_tgt)
        total_loss = loss_dict["total_loss"]
        assert torch.isfinite(total_loss), f"Loss is not finite for {backbone}"

        total_loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient in {backbone} for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {backbone} for {name}"

        print(f"[PASSED] {backbone.upper()} forward/backward pass verified (loss: {total_loss.item():.4f}).")


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

    # Ensemble test
    m_eff = DualHeadSiameseUNet(backbone="efficientnet_b2", pretrained=False).to(DEVICE)
    ens = EnsembleDualHeadModel([model, m_eff])
    ens_pred, ens_loc_p, ens_dmg_p = ens.predict(pre, post)
    assert ens_pred.shape == (2, 512, 512)
    assert ens_loc_p.shape == (2, 512, 512)
    assert ens_dmg_p.shape == (2, 4, 512, 512)
    print(f"[PASSED] Ensemble inference verified (xView2 Score: {metrics['xview2_score']:.4f}).")


def test_vectorize_and_majority_voting():
    print("\n--- Test 4: Object Vectorization & Majority Voting ---")
    labeled_mask = np.zeros((100, 100), dtype=np.int32)
    labeled_mask[10:30, 10:30] = 1
    labeled_mask[60:80, 60:80] = 2

    damage_probs = np.zeros((4, 100, 100), dtype=np.float32)
    damage_probs[0, 10:30, 10:30] = 0.95
    damage_probs[3, 60:80, 60:80] = 0.90

    voted_mask, best_classes, best_confs, areas = majority_vote_damage(labeled_mask, damage_probs)
    assert voted_mask.shape == (100, 100)
    assert best_classes[0] == 1
    assert best_classes[1] == 4
    assert np.all(voted_mask[10:30, 10:30] == 1)
    assert np.all(voted_mask[60:80, 60:80] == 4)

    loc_mask = (labeled_mask > 0).astype(np.uint8)
    clean_mask, features = vectorize_predictions(loc_mask, damage_probs, min_area=10)
    assert len(features) == 2
    assert features[0]["properties"]["damage_class"] == "no-damage"
    assert features[1]["properties"]["damage_class"] == "destroyed"
    print(f"[PASSED] Vectorized {len(features)} buildings with majority voting.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TERRASHIELD AI PIPELINE TESTS")
    print("=" * 60)
    test_dataset_and_normalization()
    test_model_forward_backward()
    test_predict_and_metrics()
    test_vectorize_and_majority_voting()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

