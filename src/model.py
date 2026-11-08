"""
Dual-Head Siamese U-Net Architecture for TerraShield AI.
Decouples building footprint localization from building damage classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class FeatureFusion(nn.Module):
    """Fuses multi-temporal feature maps using concatenation and absolute difference."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre_features: torch.Tensor, post_features: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(post_features - pre_features)
        fused = torch.cat([pre_features, post_features, diff], dim=1)
        return self.projection(fused)


class DecoderBlock(nn.Module):
    """U-Net decoder block with bilinear upsampling and skip connection fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class DualHeadSiameseUNet(nn.Module):
    """
    Dual-Head Siamese U-Net:
      - Shared ResNet-34 encoder for pre-disaster and post-disaster images.
      - Localization Branch: Segments pre-disaster building footprints (binary).
      - Damage Branch: Classifies damage into 4 classes strictly within building regions.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        # Shared encoder layers
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Temporal feature fusion modules for damage branch
        self.fuse_stem = FeatureFusion(64, 32)
        self.fuse1 = FeatureFusion(64, 32)
        self.fuse2 = FeatureFusion(128, 64)
        self.fuse3 = FeatureFusion(256, 128)
        self.fuse4 = FeatureFusion(512, 256)

        # Localization decoder (processes pre-disaster features)
        self.loc_dec4 = DecoderBlock(512, 256, 128)
        self.loc_dec3 = DecoderBlock(128, 128, 64)
        self.loc_dec2 = DecoderBlock(64, 64, 32)
        self.loc_dec1 = DecoderBlock(32, 64, 32)

        # Damage decoder (processes fused multi-temporal features)
        self.dmg_dec4 = DecoderBlock(256, 128, 128)
        self.dmg_dec3 = DecoderBlock(128, 64, 64)
        self.dmg_dec2 = DecoderBlock(64, 32, 32)
        self.dmg_dec1 = DecoderBlock(32, 32, 32)

        # Output Heads
        self.loc_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self.damage_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),  # 4 damage classes
        )

    def extract_features(self, x: torch.Tensor):
        stem = self.stem(x)
        x_pool = self.maxpool(stem)
        f1 = self.layer1(x_pool)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return stem, f1, f2, f3, f4

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor):
        # Extract features independently with shared encoder
        p_stem, p1, p2, p3, p4 = self.extract_features(pre_image)
        q_stem, q1, q2, q3, q4 = self.extract_features(post_image)

        # 1. Localization branch (from pre-disaster features)
        lx = self.loc_dec4(p4, p3)
        lx = self.loc_dec3(lx, p2)
        lx = self.loc_dec2(lx, p1)
        lx = self.loc_dec1(lx, p_stem)
        lx = F.interpolate(lx, size=pre_image.shape[-2:], mode="bilinear", align_corners=False)
        loc_logits = self.loc_head(lx)

        # 2. Damage branch (from fused temporal features)
        f_stem = self.fuse_stem(p_stem, q_stem)
        f1 = self.fuse1(p1, q1)
        f2 = self.fuse2(p2, q2)
        f3 = self.fuse3(p3, q3)
        f4 = self.fuse4(p4, q4)

        dx = self.dmg_dec4(f4, f3)
        dx = self.dmg_dec3(dx, f2)
        dx = self.dmg_dec2(dx, f1)
        dx = self.dmg_dec1(dx, f_stem)
        dx = F.interpolate(dx, size=post_image.shape[-2:], mode="bilinear", align_corners=False)
        damage_logits = self.damage_head(dx)

        return loc_logits, damage_logits

    @torch.no_grad()
    def predict(self, pre_image: torch.Tensor, post_image: torch.Tensor, loc_threshold: float = 0.5):
        """
        Inference helper:
        Produces full 5-class segmentation (0=background, 1..4=damage) by masking
        damage predictions with localized building footprints.
        """
        loc_logits, damage_logits = self.forward(pre_image, post_image)
        loc_probs = torch.sigmoid(loc_logits).squeeze(1)  # [B, H, W]
        damage_probs = torch.softmax(damage_logits, dim=1)  # [B, 4, H, W]

        is_building = loc_probs >= loc_threshold
        damage_preds = torch.argmax(damage_probs, dim=1) + 1  # map 0..3 -> 1..4

        final_mask = torch.where(is_building, damage_preds, torch.zeros_like(damage_preds))
        return final_mask, loc_probs, damage_probs

