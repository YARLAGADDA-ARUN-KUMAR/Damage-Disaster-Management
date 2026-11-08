"""
Data loading and preprocessing pipeline for TerraShield AI.
Includes ImageNet normalization, guided building-focused patch cropping, and dual-head targets.
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    DATASET_ROOT,
    TARGET_CACHE_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IGNORE_INDEX,
    TRAIN_CROP_SIZE,
    BUILDING_CROP_PROBABILITY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)

# Precomputed tensor normalization constants [3, 1, 1]
NORM_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
NORM_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


def build_split_dataframe(split: str) -> pd.DataFrame:
    """Indexes pre/post images, targets, and metadata for a given split."""
    split_root = DATASET_ROOT / split
    images_dir = split_root / "images"
    targets_dir = split_root / "targets"
    labels_dir = split_root / "labels"

    pre_images = {
        p.stem.replace("_pre_disaster", ""): p
        for p in images_dir.glob("*_pre_disaster.png")
    }
    post_images = {
        p.stem.replace("_post_disaster", ""): p
        for p in images_dir.glob("*_post_disaster.png")
    }

    tile_ids = sorted(set(pre_images.keys()) & set(post_images.keys()))
    records = []
    for tile_id in tile_ids:
        records.append({
            "tile_id": tile_id,
            "pre_image": str(pre_images[tile_id]),
            "post_image": str(post_images[tile_id]),
            "post_mask": str(targets_dir / f"{tile_id}_post_disaster_target.png"),
            "pre_mask": str(targets_dir / f"{tile_id}_pre_disaster_target.png"),
            "post_label": str(labels_dir / f"{tile_id}_post_disaster.json"),
        })

    df = pd.DataFrame(records)
    return df


def apply_spatial_augmentation(combined_images: torch.Tensor, target: torch.Tensor):
    """Applies synchronized spatial transformations (flips and 90-degree rotations)."""
    # Horizontal flip
    if random.random() < 0.5:
        combined_images = torch.flip(combined_images, dims=[2])
        target = torch.flip(target, dims=[1])

    # Vertical flip
    if random.random() < 0.5:
        combined_images = torch.flip(combined_images, dims=[1])
        target = torch.flip(target, dims=[0])

    # 90-degree rotations
    rot = random.randint(0, 3)
    if rot > 0:
        combined_images = torch.rot90(combined_images, k=rot, dims=[1, 2])
        target = torch.rot90(target, k=rot, dims=[0, 1])

    return combined_images.contiguous(), target.contiguous()


class DualHeadXView2Dataset(Dataset):
    """
    Dataset for Dual-Head Siamese U-Net.
    Generates:
      1. Normalized pre-disaster RGB tensor
      2. Normalized post-disaster RGB tensor
      3. Binary building localization target (0=background, 1=building, 255=ignore)
      4. 4-class building damage target (0..3=damage, 255=background & ignore)
      5. Full 5-class target mask (0..4, 255=ignore) for evaluation
    """

    def __init__(self, dataframe: pd.DataFrame, split: str, augment: bool = False, crop_size=TRAIN_CROP_SIZE):
        self.dataframe = dataframe.reset_index(drop=True)
        self.split = split
        self.augment = augment
        self.crop_size = crop_size
        self.cache_dir = TARGET_CACHE_ROOT / split

    def __len__(self):
        return len(self.dataframe)

    def _sample_crop_coords(self, target_mask: np.ndarray, crop_h: int, crop_w: int):
        img_h, img_w = target_mask.shape
        max_y = max(0, img_h - crop_h)
        max_x = max(0, img_w - crop_w)

        # Guided building sampling: focus on building and damage pixels
        has_buildings = np.any((target_mask >= 1) & (target_mask <= 4))
        if has_buildings and random.random() < BUILDING_CROP_PROBABILITY:
            building_ys, building_xs = np.where((target_mask >= 1) & (target_mask <= 4))
            idx = random.randint(0, len(building_ys) - 1)
            cy, cx = building_ys[idx], building_xs[idx]
            top = int(np.clip(cy - crop_h // 2 + random.randint(-64, 64), 0, max_y))
            left = int(np.clip(cx - crop_w // 2 + random.randint(-64, 64), 0, max_x))
        else:
            top = random.randint(0, max_y)
            left = random.randint(0, max_x)

        return top, left

    def __getitem__(self, index: int):
        sample = self.dataframe.iloc[index]
        tile_id = sample["tile_id"]

        # Load RGB images
        pre_img = np.array(Image.open(sample["pre_image"]).convert("RGB"))
        post_img = np.array(Image.open(sample["post_image"]).convert("RGB"))

        # Load cached target mask (values 0..4, 255)
        target_path = self.cache_dir / f"{tile_id}.png"
        if target_path.exists():
            target_mask = np.array(Image.open(target_path).convert("L"))
        else:
            target_mask = np.array(Image.open(sample["post_mask"]).convert("L"))

        # ImageNet normalization: scale to [0, 1] then standardize
        pre_t = torch.from_numpy(pre_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        post_t = torch.from_numpy(post_img.astype(np.float32) / 255.0).permute(2, 0, 1)

        pre_t = (pre_t - NORM_MEAN) / NORM_STD
        post_t = (post_t - NORM_MEAN) / NORM_STD

        target_t = torch.from_numpy(target_mask.astype(np.int64))

        # Training crops & augmentations
        if self.augment and self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            top, left = self._sample_crop_coords(target_mask, crop_h, crop_w)

            pre_t = pre_t[:, top:top + crop_h, left:left + crop_w]
            post_t = post_t[:, top:top + crop_h, left:left + crop_w]
            target_t = target_t[top:top + crop_h, left:left + crop_w]

            combined = torch.cat([pre_t, post_t], dim=0)
            combined, target_t = apply_spatial_augmentation(combined, target_t)
            pre_t = combined[:3]
            post_t = combined[3:]

        # Derive dual-head targets
        # 1. Binary building footprint: 1 for building (classes 1..4), 0 for background (0), 255 for ignore
        loc_target = torch.zeros_like(target_t, dtype=torch.float32)
        loc_target[(target_t >= 1) & (target_t <= 4)] = 1.0
        loc_target[target_t == IGNORE_INDEX] = float(IGNORE_INDEX)

        # 2. Damage target: 0..3 on buildings, 255 on background & ignore
        damage_target = torch.full_like(target_t, IGNORE_INDEX, dtype=torch.int64)
        building_mask = (target_t >= 1) & (target_t <= 4)
        damage_target[building_mask] = target_t[building_mask] - 1  # maps 1..4 -> 0..3

        return {
            "pre_image": pre_t,
            "post_image": post_t,
            "loc_target": loc_target,
            "damage_target": damage_target,
            "raw_target": target_t,
            "tile_id": tile_id,
        }


def get_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    crop_size=TRAIN_CROP_SIZE,
):
    """Creates training, holdout, and test dataloaders."""
    train_df = build_split_dataframe("train")
    hold_df = build_split_dataframe("hold")
    test_df = build_split_dataframe("test")

    train_ds = DualHeadXView2Dataset(train_df, split="train", augment=True, crop_size=crop_size)
    hold_ds = DualHeadXView2Dataset(hold_df, split="hold", augment=False, crop_size=None)
    test_ds = DualHeadXView2Dataset(test_df, split="test", augment=False, crop_size=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    hold_loader = DataLoader(
        hold_ds,
        batch_size=1,  # Full 1024x1024 evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Full 1024x1024 evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, hold_loader, test_loader
