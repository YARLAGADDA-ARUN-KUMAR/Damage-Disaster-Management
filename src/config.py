"""
Central configuration for TerraShield AI (xView2 Disaster Assessment).
"""

from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
PROCESSED_ROOT = PROJECT_ROOT / "processed"
TARGET_CACHE_ROOT = PROCESSED_ROOT / "targets"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Ensure runtime directories exist
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Image parameters
IMAGE_SIZE = (1024, 1024)
TRAIN_CROP_SIZE = (512, 512)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class mappings
IGNORE_INDEX = 255

CLASS_MAPPING_5 = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

DAMAGE_CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

NUM_DAMAGE_CLASSES = 4
NUM_TOTAL_CLASSES = 5

# Empirically verified normalized inverse square root class weights for damage classes
# (excludes background so gradient is not diluted)
DAMAGE_CLASS_WEIGHTS = [0.3987, 1.1309, 1.0064, 1.4641]

# Hardware & Training
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 0  # Safe default on Windows
DEFAULT_LR = 2e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_EPOCHS = 20
DEFAULT_PATIENCE = 6
BUILDING_CROP_PROBABILITY = 0.85

