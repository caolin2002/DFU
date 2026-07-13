#!/usr/bin/env python3
"""
PyTorch Dataset for DFU Wagner 0-5 grading system.

Label schema (7 classes — full Wagner 0-5 granularity):
  normal   = 0 — healthy foot/skin, no DFU pathology
  grade0   = 1 — Wagner 0, high-risk foot, callus, deformity, NO ulcer
  grade1   = 2 — Wagner 1, superficial ulcer (cluster + human-labeled)
  grade2   = 3 — Wagner 2, deep ulcer (cluster + human-labeled)
  grade3   = 4 — Wagner 3, deep infection (cluster + human-labeled)
  grade4   = 5 — Wagner 4-5, localized gangrene (aug + human-labeled)
  grade5   = 6 — Wagner 4-5, full-foot gangrene (reserved placeholder)

Binary mode:
  benign (grade0 + normal) vs ulcer (grade1 + grade2 + grade3 + grade4 + grade5)

Directory structure expected:
  data/processed/{train,val,test}/{normal,grade0,grade1,grade2,grade3,grade4,grade5}/*.jpg

Supports:
  - Group-based sampling (one variant per wound per epoch) for R3 augmented data
  - RandAugment online augmentation (training)
  - Class weights for imbalanced data (esp. grade4/grade5)
  - Binary and ordinal (CORN) modes
"""

import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ─── Label Maps ───────────────────────────────────────────────────────
# 7-class schema — ordered by clinical severity (Wagner 0-5 full granularity):
#   normal = 0 — healthy, no DFU
#   grade0 = 1 — Wagner 0, high-risk foot, NO ulcer
#   grade1 = 2 — Wagner 1, superficial ulcer (cluster + human-labeled)
#   grade2 = 3 — Wagner 2, deep ulcer (cluster + human-labeled)
#   grade3 = 4 — Wagner 3, deep infection (cluster + human-labeled)
#   grade4 = 5 — Wagner 4-5, localized gangrene (aug + human-labeled)
#   grade5 = 6 — Wagner 4-5, full-foot gangrene (reserved placeholder)
LABEL_TO_IDX = {
    "normal": 0,
    "grade0": 1,
    "grade1": 2,
    "grade2": 3,
    "grade3": 4,
    "grade4": 5,
    "grade5": 6,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# Human-readable names
LABEL_NAMES = {
    "normal": "Normal (healthy)",
    "grade0": "Wagner 0 (high-risk)",
    "grade1": "Wagner 1 (superficial, cluster + labeled)",
    "grade2": "Wagner 2 (deep ulcer, cluster + labeled)",
    "grade3": "Wagner 3 (deep infection, cluster + labeled)",
    "grade4": "Wagner 4-5 (localized gangrene, aug + labeled)",
    "grade5": "Wagner 4-5 (full-foot, reserved)",
}

# Binary mode mapping
# 0 = benign (no ulcer): normal + grade0
# 1 = ulcer (ulcer present): grade1 + grade2 + grade3 + grade4 + grade5
BINARY_MAP = {
    0: 0,  # normal  → benign
    1: 0,  # grade0  → benign
    2: 1,  # grade1  → ulcer
    3: 1,  # grade2  → ulcer
    4: 1,  # grade3  → ulcer
    5: 1,  # grade4  → ulcer
    6: 1,  # grade5  → ulcer
}

NUM_CLASSES = 7
NUM_BINARY = 2

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Helpers ──────────────────────────────────────────────────────────
def get_original_id(filename: str) -> str:
    """
    Extract original image ID from augmented filename.
    Handles three augmentation naming patterns:
      1. Roboflow:   'wound_A_jpg.rf.aaa111.jpg' → 'wound_A_jpg'
      2. DM pre-aug:  'DM001_M_L-rotated1.png'    → 'DM001_M_L'
      3. R3 offline:  'DM001_M_L_aug0001.jpg'     → 'DM001_M_L'
    For non-augmented files, returns the stem.
    """
    # Roboflow pattern: <id>.rf.<hash>.ext
    idx = filename.find(".rf.")
    if idx != -1:
        return filename[:idx]

    # DM pre-augmentation pattern: <id>-rotated1.ext, <id>-sharpened.ext
    stem = Path(filename).stem
    for suffix in ('-rotated1', '-rotated2', '-sharpened'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]

    # R3 offline augmentation pattern: <id>_augNNNN.ext
    aug_match = re.search(r'_aug\d{4,}$', stem)
    if aug_match:
        return stem[:aug_match.start()]

    return stem


def compute_class_weights(data_dir: str, binary: bool = False) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced data.

    Args:
        data_dir: path to processed dataset root (contains train/val/test)
        binary: if True, return 2-class weights (benign/ulcer)
    Returns:
        tensor of class weights (higher weight for rare classes)
    """
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        return torch.ones(NUM_BINARY if binary else NUM_CLASSES)

    counts = defaultdict(int)
    for label_name, idx in LABEL_TO_IDX.items():
        label_dir = train_dir / label_name
        if label_dir.exists():
            counts[idx] = len(list(label_dir.iterdir()))

    if binary:
        benign_count = counts.get(0, 0) + counts.get(1, 0)
        ulcer_count = counts.get(2, 0) + counts.get(3, 0) + counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0)
        total = benign_count + ulcer_count
        if total == 0:
            return torch.ones(2)
        # Inverse frequency
        w_benign = total / (2 * max(benign_count, 1))
        w_ulcer = total / (2 * max(ulcer_count, 1))
        return torch.tensor([w_benign, w_ulcer], dtype=torch.float32)

    total = sum(counts.values())
    n = len(LABEL_TO_IDX)
    weights = []
    for i in range(n):
        c = counts.get(i, 0)
        if c == 0:
            weights.append(0.0)  # 空占位类不参与 loss 计算
        else:
            weights.append(total / (n * c))
    return torch.tensor(weights, dtype=torch.float32)


# ─── Transforms ───────────────────────────────────────────────────────
def get_train_transforms(input_size: int = 224):
    """Training augmentation: offline variants + online RandAugment + RandomErasing."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandAugment(num_ops=3, magnitude=12),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])


def get_val_transforms(input_size: int = 224):
    """Validation/test — resize + normalize only."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Dataset ──────────────────────────────────────────────────────────
class DFUDataset(Dataset):
    """
    DFU Wagner grading dataset.

    Supports two modes:
      binary=True  → 0=benign (grade0+normal), 1=ulcer (wound+gangrene)
      binary=False → 0=grade0, 1=normal, 2=wound, 3=gangrene

    Group-based sampling:
      Augmented variants of the same wound are grouped. Each __getitem__
      randomly picks one variant from the group, preventing memorization
      of fixed augmentations. (Becomes relevant after R3 augmentation.)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        input_size: int = 224,
        binary: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.binary = binary
        self.is_train = (split == "train")
        self.num_classes = NUM_BINARY if binary else NUM_CLASSES

        # self.groups: [(variant_paths_list, label_idx)]
        self.groups: list[tuple[list[str], int]] = []

        # Flat label list for class-weight computation
        self.all_targets: list[int] = []

        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # Collect files per label
        for label_name, label_idx in LABEL_TO_IDX.items():
            grade_dir = split_dir / label_name
            if not grade_dir.exists() or not grade_dir.is_dir():
                continue

            # Group files by original ID (for augmented variants)
            grade_groups: dict[str, list[str]] = defaultdict(list)
            for img_path in grade_dir.iterdir():
                if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}:
                    continue
                orig_id = get_original_id(img_path.name)
                grade_groups[orig_id].append(str(img_path))

            for paths in grade_groups.values():
                mapped_label = BINARY_MAP[label_idx] if binary else label_idx
                self.groups.append((paths, mapped_label))
                self.all_targets.append(mapped_label)

        if self.is_train:
            self.transform = get_train_transforms(input_size)
        else:
            self.transform = get_val_transforms(input_size)

        # Log
        label_counts = defaultdict(int)
        for _, lbl in self.groups:
            label_counts[lbl] += 1
        mode_str = "binary" if binary else "multiclass"
        print(f"  [{split}] {len(self.groups)} groups, {len(self.all_targets)} files ({mode_str})")
        if not binary:
            for lbl_idx, count in sorted(label_counts.items()):
                print(f"    {IDX_TO_LABEL[lbl_idx]:<10} ({LABEL_NAMES[IDX_TO_LABEL[lbl_idx]]}): {count}")
        else:
            names = {0: "benign", 1: "ulcer"}
            for lbl_idx, count in sorted(label_counts.items()):
                print(f"    {names[lbl_idx]:<10}: {count}")

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        paths, label = self.groups[idx]

        # Pick one variant randomly (for augmented data)
        chosen_path = random.choice(paths)

        img = Image.open(chosen_path).convert("RGB")
        return self.transform(img), label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights from group labels. Empty classes get zero weight."""
        counts = defaultdict(int)
        for _, lbl in self.groups:
            counts[lbl] += 1
        total = len(self.groups)
        n = self.num_classes
        weights = []
        for i in range(n):
            c = counts.get(i, 0)
            if c == 0:
                weights.append(0.0)  # 空占位类不参与损失计算
            else:
                weights.append(total / (n * c))
        return torch.tensor(weights, dtype=torch.float32)
