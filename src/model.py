from typing import List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet34,
    ResNet34_Weights,
    efficientnet_b2,
    EfficientNet_B2_Weights,
)


class FeatureFusion(nn.Module):
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
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            res = resnet34(weights=weights)
            self.stem = nn.Sequential(res.conv1, res.bn1, res.relu)
            self.maxpool = res.maxpool
            self.layer1 = res.layer1
            self.layer2 = res.layer2
            self.layer3 = res.layer3
            self.layer4 = res.layer4

            self.fuse_stem = FeatureFusion(64, 32)
            self.fuse1 = FeatureFusion(64, 32)
            self.fuse2 = FeatureFusion(128, 64)
            self.fuse3 = FeatureFusion(256, 128)
            self.fuse4 = FeatureFusion(512, 256)

            self.loc_dec4 = DecoderBlock(512, 256, 128)
            self.loc_dec3 = DecoderBlock(128, 128, 64)
            self.loc_dec2 = DecoderBlock(64, 64, 32)
            self.loc_dec1 = DecoderBlock(32, 64, 32)

            self.dmg_dec4 = DecoderBlock(256, 128, 128)
            self.dmg_dec3 = DecoderBlock(128, 64, 64)
            self.dmg_dec2 = DecoderBlock(64, 32, 32)
            self.dmg_dec1 = DecoderBlock(32, 32, 32)

        elif self.backbone_name == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            eff = efficientnet_b2(weights=weights)
            self.eff_stage0 = eff.features[0:2]
            self.eff_stage1 = eff.features[2:3]
            self.eff_stage2 = eff.features[3:4]
            self.eff_stage3 = eff.features[4:6]
            self.eff_stage4 = eff.features[6:9]

            self.fuse_stem = FeatureFusion(16, 16)
            self.fuse1 = FeatureFusion(24, 24)
            self.fuse2 = FeatureFusion(48, 48)
            self.fuse3 = FeatureFusion(120, 64)
            self.fuse4 = FeatureFusion(1408, 256)

            self.loc_dec4 = DecoderBlock(1408, 120, 128)
            self.loc_dec3 = DecoderBlock(128, 48, 64)
            self.loc_dec2 = DecoderBlock(64, 24, 32)
            self.loc_dec1 = DecoderBlock(32, 16, 32)

            self.dmg_dec4 = DecoderBlock(256, 64, 128)
            self.dmg_dec3 = DecoderBlock(128, 48, 64)
            self.dmg_dec2 = DecoderBlock(64, 24, 32)
            self.dmg_dec1 = DecoderBlock(32, 16, 32)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet34' or 'efficientnet_b2'.")

        self.loc_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        self.damage_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )

    def extract_features(self, x: torch.Tensor):
        if self.backbone_name == "resnet34":
            stem = self.stem(x)
            x_pool = self.maxpool(stem)
            f1 = self.layer1(x_pool)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return stem, f1, f2, f3, f4
        else:
            e0 = self.eff_stage0(x)
            e1 = self.eff_stage1(e0)
            e2 = self.eff_stage2(e1)
            e3 = self.eff_stage3(e2)
            e4 = self.eff_stage4(e3)
            return e0, e1, e2, e3, e4

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor):
        p0, p1, p2, p3, p4 = self.extract_features(pre_image)
        q0, q1, q2, q3, q4 = self.extract_features(post_image)

        lx = self.loc_dec4(p4, p3)
        lx = self.loc_dec3(lx, p2)
        lx = self.loc_dec2(lx, p1)
        lx = self.loc_dec1(lx, p0)
        lx = F.interpolate(lx, size=pre_image.shape[-2:], mode="bilinear", align_corners=False)
        loc_logits = self.loc_head(lx)

        f0 = self.fuse_stem(p0, q0)
        f1 = self.fuse1(p1, q1)
        f2 = self.fuse2(p2, q2)
        f3 = self.fuse3(p3, q3)
        f4 = self.fuse4(p4, q4)

        dx = self.dmg_dec4(f4, f3)
        dx = self.dmg_dec3(dx, f2)
        dx = self.dmg_dec2(dx, f1)
        dx = self.dmg_dec1(dx, f0)
        dx = F.interpolate(dx, size=post_image.shape[-2:], mode="bilinear", align_corners=False)
        damage_logits = self.damage_head(dx)

        return loc_logits, damage_logits

    @torch.no_grad()
    def predict(self, pre_image: torch.Tensor, post_image: torch.Tensor, loc_threshold: float = 0.35):
        loc_logits, damage_logits = self.forward(pre_image, post_image)
        loc_probs = torch.sigmoid(loc_logits).squeeze(1)
        damage_probs = torch.softmax(damage_logits, dim=1)

        is_building = loc_probs >= loc_threshold
        damage_preds = torch.argmax(damage_probs, dim=1) + 1

        final_mask = torch.where(is_building, damage_preds, torch.zeros_like(damage_preds))
        return final_mask, loc_probs, damage_probs

    @torch.no_grad()
    def predict_tta(self, pre_image: torch.Tensor, post_image: torch.Tensor, loc_threshold: float = 0.35):
        loc_l1, dmg_l1 = self.forward(pre_image, post_image)
        loc_p = torch.sigmoid(loc_l1)
        dmg_p = torch.softmax(dmg_l1, dim=1)

        pre_h = torch.flip(pre_image, dims=[3])
        post_h = torch.flip(post_image, dims=[3])
        loc_l2, dmg_l2 = self.forward(pre_h, post_h)
        loc_p += torch.flip(torch.sigmoid(loc_l2), dims=[3])
        dmg_p += torch.flip(torch.softmax(dmg_l2, dim=1), dims=[3])

        pre_v = torch.flip(pre_image, dims=[2])
        post_v = torch.flip(post_image, dims=[2])
        loc_l3, dmg_l3 = self.forward(pre_v, post_v)
        loc_p += torch.flip(torch.sigmoid(loc_l3), dims=[2])
        dmg_p += torch.flip(torch.softmax(dmg_l3, dim=1), dims=[2])

        loc_probs = (loc_p / 3.0).squeeze(1)
        dmg_probs = dmg_p / 3.0

        is_building = loc_probs >= loc_threshold
        damage_preds = torch.argmax(dmg_probs, dim=1) + 1

        final_mask = torch.where(is_building, damage_preds, torch.zeros_like(damage_preds))
        return final_mask, loc_probs, dmg_probs


class EnsembleDualHeadModel:
    def __init__(self, models: List[DualHeadSiameseUNet], weights: Optional[List[float]] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self

    @torch.no_grad()
    def predict(self, pre_image: torch.Tensor, post_image: torch.Tensor, loc_threshold: float = 0.35):
        blended_loc_probs = None
        blended_dmg_probs = None

        for model, w in zip(self.models, self.weights):
            _, loc_p, dmg_p = model.predict(pre_image, post_image, loc_threshold=loc_threshold)
            if blended_loc_probs is None:
                blended_loc_probs = w * loc_p
                blended_dmg_probs = w * dmg_p
            else:
                blended_loc_probs += w * loc_p
                blended_dmg_probs += w * dmg_p

        is_building = blended_loc_probs >= loc_threshold
        damage_preds = torch.argmax(blended_dmg_probs, dim=1) + 1
        final_mask = torch.where(is_building, damage_preds, torch.zeros_like(damage_preds))
        return final_mask, blended_loc_probs, blended_dmg_probs

    @torch.no_grad()
    def predict_tta(self, pre_image: torch.Tensor, post_image: torch.Tensor, loc_threshold: float = 0.35):
        blended_loc_probs = None
        blended_dmg_probs = None

        for model, w in zip(self.models, self.weights):
            _, loc_p, dmg_p = model.predict_tta(pre_image, post_image, loc_threshold=loc_threshold)
            if blended_loc_probs is None:
                blended_loc_probs = w * loc_p
                blended_dmg_probs = w * dmg_p
            else:
                blended_loc_probs += w * loc_p
                blended_dmg_probs += w * dmg_p

        is_building = blended_loc_probs >= loc_threshold
        damage_preds = torch.argmax(blended_dmg_probs, dim=1) + 1
        final_mask = torch.where(is_building, damage_preds, torch.zeros_like(damage_preds))
        return final_mask, blended_loc_probs, blended_dmg_probs


def build_model(backbone: str = "resnet34", pretrained: bool = True) -> DualHeadSiameseUNet:
    return DualHeadSiameseUNet(backbone=backbone, pretrained=pretrained)
