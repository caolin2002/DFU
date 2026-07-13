# DFU — Diabetic Foot Ulcer Wagner Grading

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Ordinal regression for diabetic foot ulcer severity grading (Wagner 0–5)** using ConvNeXt-Tiny with CORN (Conditional Ordinal Regression for Neural networks), two-stage inference, and unsupervised clustering for dataset curation.

## Overview

Accurate Wagner grading is critical for DFU treatment decisions — Grade 0 requires follow-up while Grade 3 demands emergency hospitalization. This project provides a complete pipeline from multi-source data acquisition to clinical report generation:

- **Backbone**: ConvNeXt-Tiny (ImageNet-1K pretrained, frozen)
- **Classifier**: CORN ordinal head + Binary screening head
- **Inference**: Two-stage (benign/ulcer screening → ordinal grading)
- **Augmentation**: Three-tier offline R3 pipeline for rare class expansion
- **Clustering**: 5 unsupervised strategies for unlabeled data curation
- **Explainability**: Grad-CAM visualization via `torch.autograd.grad()`
- **Reports**: Self-contained HTML clinical reports with base64-embedded images

## Key Features

| Layer | Technology | Description |
|:---|:---|:---|
| Data Acquisition | Kaggle / Mendeley / HuggingFace / Open-i / Wikimedia | 28K+ foot images from 5 sources |
| Auto-Labeling | Rule-based pipeline (R2) | 27 source categories → 7 Wagner classes |
| Offline Augmentation | R3 three-tier pipeline | Light / medium / heavy, rare class targeted expansion (grade4: 1→200) |
| Unsupervised Clustering | K-Means × 5 strategies | HSV+GLCM+Deep hybrid features → cluster assignment |
| Online Augmentation | RandAugment + RandomErasing | Per-epoch stochastic augmentation |
| Backbone | ConvNeXt-Tiny | Frozen ImageNet-1K weights |
| Ordinal Head | CORN (BCE with monotonic bias) | 6 binary subtasks with ordinal constraints |
| Binary Head | Sigmoid classifier | Two-stage screening (benign vs. ulcer) |
| Loss | CORN BCE + class weights | Inverse-frequency weighting for imbalance |
| Training | AdamW + CosineAnnealingWarmRestarts + AMP | Mixed precision with GradScaler |
| Validation | 3-Fold CV (patient-level split) | Mean ± 95% CI via t-distribution |
| TTA | 6-view test-time augmentation | Logit averaging for robust inference |
| Grad-CAM | `torch.autograd.grad()` | Clinically interpretable heatmaps |
| Reports | Self-contained HTML | Base64 images, printable to PDF |

## 7-Class Wagner Grading Schema

| Class | Wagner Grade | Clinical Meaning |
|:---|:---|:---|
| normal | — | Healthy foot, no DFU |
| grade0 | Wagner 0 | High-risk foot, no ulceration |
| grade1 | Wagner 1 | Superficial ulcer |
| grade2 | Wagner 2 | Deep ulcer |
| grade3 | Wagner 3 | Deep infection |
| grade4 | Wagner 4 | Localized gangrene |
| grade5 | Wagner 5 | Whole-foot gangrene (reserved) |

## Project Structure

```
DFU/
├── config.yaml              # Model & training configuration
├── PLAN.md                  # Implementation plan & design decisions
├── src/
│   ├── model.py             # ConvNeXt-Tiny + CORNHead + BinaryHead + losses
│   ├── train.py             # Training loop with AMP
│   ├── inference.py         # Two-stage inference + Grad-CAM + HTML reports
│   ├── eval_tta.py          # TTA evaluation script
│   ├── tta.py               # 6-view test-time augmentation
│   ├── dataset.py           # Group-based sampling + RandAugment
│   ├── gradcam.py           # Grad-CAM via autograd
│   ├── report.py            # Self-contained HTML clinical report
│   ├── cross_validate.py    # 3-Fold CV with patient-level split
│   ├── split_data.py        # Stratified train/val/test split
│   ├── cluster_split.py     # 5 unsupervised clustering strategies
│   ├── augmentation/
│   │   └── r3_augment.py    # Three-tier offline augmentation
│   ├── labeling/
│   │   └── r2_labeling.py   # Rule-based auto-labeling pipeline
│   └── download/
│       ├── download_all.py      # Data acquisition orchestrator
│       ├── download_kaggle.py   # kagglehub downloads
│       ├── download_mendeley.py # Cloudscraper-based download
│       ├── download_gangrene.py # Gangrene image crawler
│       └── extract_new_data.py  # MD5 dedup & extraction
├── docs/
│   └── DFU_技术架构总览.md   # Full technical architecture (Chinese)
├── reports/
│   ├── cluster_report.json  # Clustering results
│   └── cluster_assignments.csv
├── data/                    # (gitignored — 2.4G images)
└── models/                  # (gitignored — 1.3G checkpoints)
```

## Installation

```bash
# Clone repository
git clone git@github.com:caolin2002/DFU.git
cd DFU

# Install dependencies
pip install torch torchvision
pip install numpy pandas scipy scikit-learn tqdm pillow pyyaml
pip install kagglehub cloudscraper huggingface_hub opencv-python-headless
```

## Usage

### Training

```bash
# Full 3-fold cross-validation
python src/cross_validate.py --config config.yaml

# Single fold (for testing)
python src/cross_validate.py --config config.yaml --fold 0
```

### Inference

```bash
# Single image inference with Grad-CAM
python src/inference.py \
    --image path/to/foot_image.jpg \
    --checkpoint models/corn_v2/best_model.pth \
    --output report.html
```

### TTA Evaluation

```bash
# Evaluate with 6-view test-time augmentation
python src/eval_tta.py --checkpoint models/corn_v2/best_model.pth
```

### Data Acquisition

```bash
# Download all available datasets
python src/download/download_all.py

# List available sources
python src/download/download_all.py --list
```

## Configuration

All settings are in [`config.yaml`](config.yaml):

```yaml
model:
  name: convnext_tiny
  num_classes: 7
  binary_head: true

training:
  batch_size: 64
  epochs: 80
  learning_rate: 1.0e-3
  use_amp: true
  use_corn: true
  use_class_weights: true
```

## Two-Stage Inference Pipeline

```
Input Image
    │
    ▼
┌──────────────┐
│  BinaryHead  │─── benign ──→ "Normal foot"
│              │
│              │─── ulcer ──→ ┌────────────────┐
└──────────────┘              │  CORN Ordinal  │
                              │  Head          │
                              │                │
                              │  6 binary      │
                              │  subtasks:     │
                              │  ≥0? ≥1? ≥2?   │
                              │  ≥3? ≥4? ≥5?   │
                              └───────┬────────┘
                                      │
                                      ▼
                              Wagner 0–5 Grade
```

## Key Design Decisions

1. **CORN over standard CE** — ordinal regression preserves the natural ordering of Wagner grades (grade3 > grade2 > grade1), improving accuracy on adjacent classes
2. **Frozen backbone** — ConvNeXt-Tiny features are generic enough for medical images; training only the head (1.7M params) prevents overfitting on ~5K training images
3. **Patient-level splitting** — wounds from the same patient are grouped into the same fold, preventing data leakage from patient-specific features
4. **Group-based sampling** — multiple augmentations of the same wound are treated as one group; each `__getitem__` samples one variant randomly
5. **Two-stage inference** — binary screening (benign/ulcer) as first stage filters out healthy images before expensive ordinal grading

## Clustering Strategies (Unsupervised)

Five clustering strategies were compared for unlabeled data curation:

| Strategy | Features | Best For |
|:---|:---|:---|
| K-Means (Deep) | ConvNeXt 768d features | Semantic grouping |
| K-Means (CV) | HSV(96d) + GLCM(10d) + Color(30d) | Texture-based grouping |
| K-Means (Hybrid) | Deep + CV features concatenated | Balanced approach |
| Threshold Ranking | CV feature scalar ranking | Severity ordering |
| Feature+Prob Ranking | CV + CORN ordinal probabilities | Ordinal-aware ranking |

## Performance

Results from 3-fold cross-validation (mean ± 95% CI, t-distribution):

| Metric | Mean ± 95% CI |
|:---|:---|
| Accuracy | Reported in `reports/` |
| Macro F1 | Reported in `reports/` |
| Quadratic Kappa | Reported in `reports/` |

Full per-class F1 scores and confusion matrices are saved to `models/corn_v2/cv_results.csv`.

## Technical Documentation

For a comprehensive technical deep-dive (in Chinese), see:
→ **[docs/DFU_技术架构总览.md](docs/DFU_技术架构总览.md)**

Covers all 15 technical layers in detail: data acquisition, auto-labeling, offline augmentation, unsupervised clustering, online augmentation, backbone, CORN ordinal regression, two-stage inference, loss functions, training strategy (AMP + CV), TTA, Grad-CAM, HTML reports, complete data flow, and key design philosophy.
