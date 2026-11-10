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

## 2. Notebook Execution Guide

The entire end-to-end pipeline is structured into 6 sequential, fully self-contained Jupyter notebooks:

1. **[`01_dataset_understanding.ipynb`](file:///d:/ML_Project/notebooks/01_dataset_understanding.ipynb):**
   - File hierarchy inspection, satellite metadata parsing, and polygon annotation verification.
2. **[`02_exploratory_data_analysis.ipynb`](file:///d:/ML_Project/notebooks/02_exploratory_data_analysis.ipynb):**
   - Spectral band distribution, building area histograms, and disaster-specific damage ratios.
3. **[`03_data_preprocessing.ipynb`](file:///d:/ML_Project/notebooks/03_data_preprocessing.ipynb):**
   - WKT polygon rasterization into pixel target masks (`processed/targets/`) and unclassified building mask caching.
4. **[`04_model_development.ipynb`](file:///d:/ML_Project/notebooks/04_model_development.ipynb):**
   - Single-stage baseline experiments (01–04) analysis and diagnostic failure retrospective.
5. **[`05_model_training.ipynb`](file:///d:/ML_Project/notebooks/05_model_training.ipynb):**
   - Interactive training of both the **ResNet-34** and **EfficientNet-B2** Dual-Head Siamese U-Nets with mixed precision (AMP) and guided building cropping.
   - Saves model checkpoints to `checkpoints/dual_head_siamese_best.pth` and `checkpoints/dual_head_efficientnet_b2_best.pth`.
6. **[`06_model_evaluation.ipynb`](file:///d:/ML_Project/notebooks/06_model_evaluation.ipynb):**
   - Complete evaluation suite on 933 untouched test tiles: threshold calibration, multi-angle TTA, multi-backbone ensembling (ResNet-34 + EfficientNet-B2), object-level polygon vectorization (GeoJSON export), and polygon-guided cadastre evaluation.

---

## 3. Reference

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

### Benchmark Results on Untouched Test Set (933 Tiles)

| Model Architecture | Training Epochs | Localization F1 ($F1_{\text{loc}}$) | Damage Macro F1 ($F1_{\text{dmg}}$) | Destroyed F1 | Minor Damage F1 | Official xView2 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Single-Stage ResNet-34 Baseline | 20 | 0.6120 | 0.3700 | 0.5410 | 0.1008 | 0.4430 |
| Dual-Head Siamese U-Net (Initial) | 20 | 0.7432 | 0.6294 | 0.7015 | 0.4109 | 0.6635 |
| Dual-Head Siamese U-Net (Fine-Tuned) | 31 | 0.7588 | 0.6556 | 0.7102 | 0.4385 | 0.6866 |
| Dual-Head Siamese U-Net + TTA + Calibrated | 31 | 0.7652 | 0.6624 | 0.7173 | 0.4468 | 0.6932 |
| **Multi-Backbone Ensemble (ResNet-34 + EfficientNet-B2)** | **31 / 26** | **0.8487** | **0.6716** | **0.7209** | **0.4810** | **0.7247** |
| **Polygon-Guided Cadastre (Dataset Footprints Given)** | **31** | **1.0000** | **0.7667** | **0.8501** | **0.5432** | **0.8367** |


