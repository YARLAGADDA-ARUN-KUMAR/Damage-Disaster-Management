"""
Evaluation metrics for TerraShield AI:
Computes Localization F1, Damage Macro F1, Foreground mIoU, and Official xView2 Challenge Score.
"""

import numpy as np
import pandas as pd
import torch

from src.config import CLASS_MAPPING_5, DAMAGE_CLASSES, IGNORE_INDEX, NUM_TOTAL_CLASSES


def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Accumulates 5x5 confusion matrix for valid pixels (targets != 255).
    Rows: Ground Truth | Columns: Predictions
    """
    pred_flat = predictions.view(-1).long()
    target_flat = targets.view(-1).long()

    valid = target_flat != IGNORE_INDEX
    pred_valid = pred_flat[valid]
    target_valid = target_flat[valid]

    encoded = target_valid * NUM_TOTAL_CLASSES + pred_valid
    bincount = torch.bincount(encoded, minlength=NUM_TOTAL_CLASSES ** 2)
    return bincount.reshape(NUM_TOTAL_CLASSES, NUM_TOTAL_CLASSES).cpu().numpy()


def calculate_metrics_from_confusion(confusion: np.ndarray) -> dict:
    """
    Calculates comprehensive metrics from a 5x5 confusion matrix:
      - Localization F1 & IoU
      - Class-wise IoU, F1, Precision, Recall
      - Foreground Macro F1 (damage classes 1..4)
      - Foreground mIoU (damage classes 1..4)
      - xView2 Combined Score: 0.3 * F1_loc + 0.7 * F1_damage_macro
    """
    tp = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)

    union = actual + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)

    denominator = actual + predicted
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)

    # Localization metric (Building vs Background):
    # Classes 1..4 = building, Class 0 = background
    loc_tp = tp[1:].sum()
    loc_actual = actual[1:].sum()
    loc_predicted = predicted[1:].sum()
    loc_union = loc_actual + loc_predicted - loc_tp
    loc_iou = loc_tp / max(loc_union, 1.0)
    loc_precision = loc_tp / max(loc_predicted, 1.0)
    loc_recall = loc_tp / max(loc_actual, 1.0)
    loc_f1 = (2.0 * loc_tp) / max(loc_actual + loc_predicted, 1.0)

    # Foreground damage metrics (classes 1..4)
    fg_iou = iou[1:]
    fg_f1 = f1[1:]
    fg_precision = precision[1:]
    fg_recall = recall[1:]

    fg_miou = float(np.mean(fg_iou))
    fg_macro_f1 = float(np.mean(fg_f1))

    # Official xView2 Challenge Score
    xview2_score = 0.3 * loc_f1 + 0.7 * fg_macro_f1

    # Format class-wise DataFrame
    class_df = pd.DataFrame({
        "Class": [CLASS_MAPPING_5[i] for i in range(NUM_TOTAL_CLASSES)],
        "IoU": np.round(iou, 4),
        "F1": np.round(f1, 4),
        "Precision": np.round(precision, 4),
        "Recall": np.round(recall, 4),
        "Actual Pixels": actual.astype(np.int64),
        "Predicted Pixels": predicted.astype(np.int64),
    })

    return {
        "confusion_matrix": confusion,
        "class_table": class_df,
        "class_iou": iou,
        "class_f1": f1,
        "class_precision": precision,
        "class_recall": recall,
        "loc_iou": float(loc_iou),
        "loc_f1": float(loc_f1),
        "loc_precision": float(loc_precision),
        "loc_recall": float(loc_recall),
        "fg_miou": fg_miou,
        "fg_macro_f1": fg_macro_f1,
        "xview2_score": float(xview2_score),
    }
