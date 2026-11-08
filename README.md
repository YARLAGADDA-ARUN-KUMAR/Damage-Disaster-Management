# TerraShield AI: Satellite Disaster Damage Assessment

TerraShield AI is a deep learning system for rapid, pixel-level disaster damage assessment from pre- and post-disaster satellite imagery on the **xView2** dataset.

The system uses a **Dual-Head Siamese U-Net** that decouples building footprint localization from building damage classification, overcoming class imbalance and background swallow.

---

## 1. Explanation: Architecture & Problem Formulation

### The Limitation of Single-Stage 5-Class Segmentation
In satellite disaster damage assessment:
- **94.0%** of all pixels are background (unbuilt terrain, water, roads, vegetation).
- Heavily damaged or destroyed structures collapse into rubble that closely resembles bare dirt or gravel.
- A single-stage 5-class model (`[background, no-damage, minor, major, destroyed]`) experiences severe class collision:
  - Over 20% to 31% of damaged and destroyed building pixels are swallowed by background.
  - Minor damage (often subtle roof discolouration or broken tiles) is overwhelmingly misclassified as undamaged.

### The Dual-Head Siamese U-Net Solution
TerraShield AI resolves this using a **Dual-Head Decoupled Architecture**:

```
Pre-Disaster Image  [3x512x512] ──► [Shared ResNet-34 Encoder] ──► [Localization Decoder] ──► Binary Building Mask
                                                ▲
                                                │ (Multiscale Fusion: concat + |pre - post|)
                                                ▼
Post-Disaster Image [3x512x512] ──► [Shared ResNet-34 Encoder] ──► [Damage Decoder]       ──► 4-Class Damage Logits
                                                                                                      │
                                                                  [Masked Multiplication]  ◄──────────┘
                                                                              ▼
                                                                Final 5-Class Prediction
```

1. **Pre-Disaster Building Localization:** Intact buildings in pre-disaster images have sharp geometric edges. The localization branch achieves high F1 without confusion from post-disaster rubble.
2. **Decoupled Damage Assessment:** The damage branch evaluates damage levels (0: No Damage, 1: Minor, 2: Major, 3: Destroyed) **only on building pixels**, completely removing background interference from the damage loss.
3. **Building-Focused Patch Cropping:** Training on 512×512 guided crops with 85% building probability increases building pixel density from 5.9% to ~40%, fits into 4GB VRAM with `batch_size = 4`, and speeds up training by 4x–8x.
4. **Standardized Normalization:** Full ImageNet mean/variance normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

---

## 2. Tutorials: Quickstart

### Prerequisites & Environment
Ensure PyTorch and CUDA dependencies are installed:
```powershell
uv sync
```

### Running the Test Suite
Verify model forward/backward passes, dataloaders, and metric calculations:
```powershell
.venv\Scripts\python.exe tests/test_pipeline.py
```

### Training the Model
Train the Dual-Head Siamese U-Net using mixed precision (AMP):
```powershell
.venv\Scripts\python.exe -m src.train --epochs 20 --batch-size 4 --lr 2e-4
```
Checkpoints will be saved automatically to `checkpoints/dual_head_siamese_best.pth`.

### Resuming & Fine-Tuning
Resume training from an existing checkpoint with a fine-tuning learning rate (e.g. up to 35 epochs):
```powershell
.venv\Scripts\python.exe -m src.train --epochs 35 --lr 5e-5 --resume checkpoints/dual_head_siamese_best.pth
```

### Evaluating on the Test Split
Run evaluation on the test set (933 satellite tiles) with calibrated threshold and optional TTA:
```powershell
.venv\Scripts\python.exe -m src.evaluate --checkpoint checkpoints/dual_head_siamese_best.pth --split test
```

---

## 3. How-To Guides: Interactive Notebooks

If you prefer an interactive Jupyter environment:

1. **Dataset Exploration & Preprocessing:**
   - [`01_dataset_understanding.ipynb`](file:///d:/ML_Project/notebooks/01_dataset_understanding.ipynb): File structure, polygon annotations, and class distributions.
   - [`02_exploratory_data_analysis.ipynb`](file:///d:/ML_Project/notebooks/02_exploratory_data_analysis.ipynb): Spectral analysis, building sizes, and regional disaster heatmaps.
   - [`03_data_preprocessing.ipynb`](file:///d:/ML_Project/notebooks/03_data_preprocessing.ipynb): Target rasterization and caching unclassified building ignore masks.
2. **Model Training & Evaluation:**
   - [`04_model_development.ipynb`](file:///d:/ML_Project/notebooks/04_model_development.ipynb): Baseline experiments (01–04) analysis and diagnostic retrospective.
   - [`05_model_training.ipynb`](file:///d:/ML_Project/notebooks/05_model_training.ipynb): Interactive training of the Dual-Head architecture with real-time loss, F1, and xView2 score curves.
   - [`06_model_evaluation.ipynb`](file:///d:/ML_Project/notebooks/06_model_evaluation.ipynb): Test set evaluation, normalized confusion matrices, and qualitative side-by-side visual comparisons.

---

## 4. Reference

### Evaluation Metrics
- **Localization F1 ($F1_{\text{loc}}$):** Harmonic mean of building detection precision and recall.
- **Damage Macro F1 ($F1_{\text{dmg}}$):** Unweighted average of F1 scores across the 4 damage categories (`no-damage`, `minor-damage`, `major-damage`, `destroyed`).
- **Official xView2 Challenge Score:**
  $$\text{xView2 Score} = 0.3 \times F1_{\text{loc}} + 0.7 \times F1_{\text{dmg}}$$

### Class Mapping
| ID | Label | Description |
| :---: | :--- | :--- |
| `0` | `background` | Non-building pixels (terrain, water, vegetation, roads) |
| `1` | `no-damage` | Structurally intact building with no visible signs of damage |
| `2` | `minor-damage` | Water-surrounded, partially missing shingles, minor exterior debris |
| `3` | `major-damage` | Partial structural collapse, missing roof, significant damage |
| `4` | `destroyed` | Complete collapse, leveled structure, roof and walls destroyed |
| `255` | `ignored` | Unclassified buildings excluded from loss and evaluation |

