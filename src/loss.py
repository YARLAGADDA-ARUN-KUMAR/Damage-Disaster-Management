"""
Loss functions for Dual-Head building localization and damage assessment.
Eliminates background dominance and false-positive blindspots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DAMAGE_CLASS_WEIGHTS, IGNORE_INDEX


class BinaryDiceLoss(nn.Module):
    """Numerically stable binary Dice loss with valid-pixel masking."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        p = probs[valid_mask]
        t = targets[valid_mask]
        if p.numel() == 0:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        intersection = (p * t).sum()
        cardinality = p.sum() + t.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class MulticlassDiceLoss(nn.Module):
    """Batch-level multiclass Dice loss that penalizes false positives across all classes."""

    def __init__(self, num_classes: int = 4, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # valid_mask: [B, H, W] boolean
        if not valid_mask.any():
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        # Permute probs to [B*H*W, C] and select valid pixels
        p = probs.permute(0, 2, 3, 1)[valid_mask]  # [N_valid, 4]
        t = targets[valid_mask]  # [N_valid]

        t_onehot = F.one_hot(t, num_classes=self.num_classes).float()  # [N_valid, 4]

        intersection = (p * t_onehot).sum(dim=0)
        cardinality = p.sum(dim=0) + t_onehot.sum(dim=0)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


class DualHeadLoss(nn.Module):
    """
    Combined Loss for Dual-Head Siamese U-Net:
      Total Loss = Localization Loss (BCE + Dice) + Damage Loss (Weighted CE + Dice)
    """

    def __init__(
        self,
        damage_weights=DAMAGE_CLASS_WEIGHTS,
        loc_weight: float = 1.0,
        damage_weight: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.loc_weight = loc_weight
        self.damage_weight = damage_weight
        self.ignore_index = ignore_index

        self.binary_dice = BinaryDiceLoss()
        self.multiclass_dice = MulticlassDiceLoss(num_classes=4)

        weights_t = torch.tensor(damage_weights, dtype=torch.float32)
        self.register_buffer("dmg_weights", weights_t)

    def forward(
        self,
        loc_logits: torch.Tensor,
        damage_logits: torch.Tensor,
        loc_targets: torch.Tensor,
        damage_targets: torch.Tensor,
    ):
        # 1. LOCALIZATION LOSS (pre-disaster building vs background)
        valid_loc = loc_targets != self.ignore_index
        loc_logits_squeezed = loc_logits.squeeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(
            loc_logits_squeezed[valid_loc],
            loc_targets[valid_loc],
        )

        loc_probs = torch.sigmoid(loc_logits_squeezed)
        dice_loc = self.binary_dice(loc_probs, loc_targets, valid_loc)
        loss_loc = bce_loss + dice_loc

        # 2. DAMAGE LOSS (evaluated ONLY on building pixels: damage_targets != 255)
        valid_dmg = damage_targets != self.ignore_index
        if valid_dmg.any():
            ce_damage = F.cross_entropy(
                damage_logits,
                damage_targets,
                weight=self.dmg_weights,
                ignore_index=self.ignore_index,
            )

            dmg_probs = torch.softmax(damage_logits, dim=1)
            dice_damage = self.multiclass_dice(dmg_probs, damage_targets, valid_dmg)
            loss_dmg = ce_damage + dice_damage
        else:
            loss_dmg = damage_logits.sum() * 0.0

        # Total combined loss
        total_loss = self.loc_weight * loss_loc + self.damage_weight * loss_dmg

        return {
            "total_loss": total_loss,
            "loss_loc": loss_loc,
            "bce_loc": bce_loss,
            "dice_loc": dice_loc,
            "loss_dmg": loss_dmg,
        }

