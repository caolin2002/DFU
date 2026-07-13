#!/usr/bin/env python3
"""
3-Fold Cross-Validation for DFU Wagner grading model.

Uses hybrid splitting strategy:
  - Real patients (DM/CG): patient-level — all wounds from same patient in same fold
  - Cluster-labeled data: wound-level — each cluster ID treated independently
    (original patient IDs not available; cluster IDs are 1:1 with wounds)

Reports mean ± 95% CI for Accuracy, Macro F1, Kappa, and per-class F1.

Usage:
    python src/cross_validate.py                     # full 3-fold CV
    python src/cross_validate.py --fold 0            # single fold (testing)
"""

import argparse
import csv
import os
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import (
    DFUDataset, IDX_TO_LABEL, LABEL_NAMES, LABEL_TO_IDX, NUM_CLASSES,
    get_original_id,
)
from model import corn_loss, get_convnext_tiny
from train import train_epoch, validate_epoch, set_seed


# ─── Patient/wound helpers ──────────────────────────────────────────────

def extract_patient(filename: str) -> str:
    """Extract patient ID from filename."""
    stem = Path(filename).stem
    for suffix in ('-rotated1', '-rotated2', '-sharpened'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    aug_match = re.search(r'_aug\d{4,}$', stem)
    if aug_match:
        stem = stem[:aug_match.start()]
    idx = stem.find(".rf.")
    if idx != -1:
        stem = stem[:idx]
    match = re.match(r'^([A-Z]{2,4}\d{2,4})', stem)
    if match:
        return match.group(1)
    parts = stem.split('_')
    return parts[0] if parts else stem


def is_real_patient(pid: str) -> bool:
    """Check if this is a real patient ID (DM/CG prefix)."""
    return bool(re.match(r'^[A-Z]{2,4}\d{2,4}$', pid))


# ─── Fold generation ────────────────────────────────────────────────────

def build_folds(data_dir: str, n_folds: int = 3, seed: int = 42):
    """
    Build balanced 3-fold splits.

    Strategy:
      1. Collect all files, grouped by (grade, original_id).
      2. For real patients (DM/CG), link all their wounds together.
      3. Shuffle and distribute wounds into folds, stratified by grade.
      4. Post-process to ensure real patient wounds stay together.
    """
    random.seed(seed)
    data_dir = Path(data_dir)

    # ── Collect all files ──────────────────────────────────────────
    # wound_map: (grade, oid) → [(src_path, pid), ...]
    wound_map = defaultdict(list)
    # real_patient_oids: pid → {(grade, oid), ...}
    real_patient_oids = defaultdict(set)

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for grade_dir in split_dir.iterdir():
            if not grade_dir.is_dir():
                continue
            grade = grade_dir.name
            for f in grade_dir.iterdir():
                if f.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}:
                    continue
                pid = extract_patient(f.name)
                oid = get_original_id(f.name)
                wound_map[(grade, oid)].append((f, pid))
                if is_real_patient(pid):
                    real_patient_oids[pid].add((grade, oid))

    # ── Build wound list ───────────────────────────────────────────
    # Each wound is a (grade, oid) key. Real patients have multiple wounds.
    wounds_by_grade = defaultdict(list)
    for (grade, oid), files in wound_map.items():
        wounds_by_grade[grade].append((grade, oid))

    print(f"Total wounds: {sum(len(v) for v in wounds_by_grade.values())}")
    for grade in ['normal', 'grade0', 'grade1', 'grade2', 'grade3', 'grade4']:
        print(f"  {grade}: {len(wounds_by_grade[grade])} wounds")

    # ── Assign wounds to folds (stratified by grade, shuffled) ─────
    wound_fold = {}  # (grade, oid) → fold_idx

    for grade, wounds in wounds_by_grade.items():
        shuffled = wounds.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        for i, wound in enumerate(shuffled):
            wound_fold[wound] = i % n_folds

    # ── Post-process: ensure real patient wounds in same fold ──────
    # For each real patient, move all their wounds to the fold that
    # contains the majority of their wounds.
    for pid, wound_set in real_patient_oids.items():
        fold_votes = defaultdict(int)
        for wound in wound_set:
            fold_votes[wound_fold[wound]] += 1
        # Assign all wounds to the most common fold
        best_fold = max(fold_votes, key=fold_votes.get)
        for wound in wound_set:
            wound_fold[wound] = best_fold

    # ── Build fold file lists ──────────────────────────────────────
    folds = []
    for fold_idx in range(n_folds):
        train_wounds = set()
        test_wounds = set()

        for wound, f_idx in wound_fold.items():
            if f_idx == fold_idx:
                test_wounds.add(wound)
            else:
                train_wounds.add(wound)

        # Split train into train/val (85/15, wound-level)
        train_list = sorted(train_wounds)
        random.shuffle(train_list)
        n_val = max(1, int(len(train_list) * 0.15))
        val_wounds = set(train_list[:n_val])
        train_wounds_final = set(train_list[n_val:])

        def build_files(wound_set):
            result = []
            for wound in wound_set:
                for src_path, pid in wound_map[wound]:
                    grade, oid = wound
                    result.append((src_path, grade, oid, pid))
            return result

        folds.append({
            'train': build_files(train_wounds_final),
            'val': build_files(val_wounds),
            'test': build_files(test_wounds),
        })

    return folds


def create_fold_dir(files, target_dir: Path):
    """Create directory structure with symlinks."""
    for src_path, grade, oid, pid in files:
        grade_dir = target_dir / grade
        grade_dir.mkdir(parents=True, exist_ok=True)
        dst = grade_dir / src_path.name
        if not dst.exists():
            dst.symlink_to(src_path.resolve())


# ─── Model evaluation helper ────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run evaluation, return predictions and labels."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images, task="ordinal")
            preds = model.ordinal_head.predict(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return all_preds, all_labels


def compute_all_metrics(labels, preds):
    """Compute comprehensive metrics dict."""
    n_cls = NUM_CLASSES
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    prec, rec, f1s, supp = precision_recall_fscore_support(
        labels, preds, labels=list(range(n_cls)), zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    return {
        "accuracy": acc, "macro_f1": f1_macro, "kappa": kappa,
        "precision": prec, "recall": rec, "f1_per_class": f1s,
        "support": supp, "confusion_matrix": cm,
    }


# ─── Single fold training ───────────────────────────────────────────────

def train_fold(fold_idx: int, fold_data: dict, config: dict, device: torch.device,
               fold_dir: Path):
    """Train a single CV fold. Returns test metrics."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}")
    print(f"{'='*60}")

    # Create fold directories
    for split_name, files in [('train', fold_data['train']),
                               ('val', fold_data['val']),
                               ('test', fold_data['test'])]:
        create_fold_dir(files, fold_dir / split_name)

    # Count and print
    for split_name in ['train', 'val', 'test']:
        groups = defaultdict(set)
        for _, grade, oid, _ in fold_data[split_name]:
            groups[grade].add(oid)
        counts = {g: len(oids) for g, oids in sorted(groups.items())}
        total = sum(counts.values())
        n_files = sum(1 for _ in fold_data[split_name])
        print(f"  {split_name}: {total} groups, {n_files} files — {counts}")

    # ── Datasets ───────────────────────────────────────────────────
    input_size = config["data"]["input_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = min(4, os.cpu_count() or 4)

    train_ds = DFUDataset(str(fold_dir), "train", input_size, binary=False)
    val_ds = DFUDataset(str(fold_dir), "val", input_size, binary=False)
    test_ds = DFUDataset(str(fold_dir), "test", input_size, binary=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────
    num_classes = config["model"]["num_classes"]
    binary_head = config["model"].get("binary_head", True)
    model = get_convnext_tiny(num_classes=num_classes, binary=binary_head)
    model = model.to(device)

    class_weights = train_ds.get_class_weights().to(device)

    # ── Optimizer ──────────────────────────────────────────────────
    opt_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["learning_rate"],
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=opt_cfg["lr_t_0"], T_mult=opt_cfg["lr_t_mult"],
    )
    scaler = torch.amp.GradScaler("cuda") if opt_cfg["use_amp"] and torch.cuda.is_available() else None

    # ── Train ──────────────────────────────────────────────────────
    best_val_f1 = 0.0
    patience_counter = 0
    patience = opt_cfg["early_stopping_patience"]
    epochs = opt_cfg["epochs"]
    best_state = None

    for epoch in range(epochs):
        tr_loss, tr_acc, tr_f1, _, _ = train_epoch(
            model, train_loader, corn_loss, optimizer, scaler, device,
            binary_mode=False, log_interval=1000,
            classify_mode=False, class_weights=class_weights,
        )
        vl_loss, vl_acc, vl_f1, vl_kappa, _, _ = validate_epoch(
            model, val_loader, corn_loss, device,
            binary_mode=False, classify_mode=False,
        )
        scheduler.step()

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1} (best val_f1={best_val_f1:.4f})")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} "
                  f"vl_loss={vl_loss:.4f} vl_f1={vl_f1:.4f} vl_kappa={vl_kappa:.4f}")

    # ── Test evaluation ────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    preds, labels = evaluate_model(model, test_loader, device)
    metrics = compute_all_metrics(labels, preds)

    print(f"  Fold {fold_idx+1} Test: Acc={metrics['accuracy']:.4f} "
          f"F1={metrics['macro_f1']:.4f} Kappa={metrics['kappa']:.4f}")

    return metrics


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-Fold CV for DFU model")
    parser.add_argument("--config", type=str, default="/root/dfu/config.yaml")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold only (0-indexed, for testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="/root/dfu/models/corn_v2/cv_results.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Folds: {args.n_folds}")

    data_dir = config["data"]["data_dir"]

    # ── Build folds ────────────────────────────────────────────────
    print("\n=== Building balanced wound-level folds ===")
    folds = build_folds(data_dir, n_folds=args.n_folds, seed=args.seed)

    for i, fold_data in enumerate(folds):
        print(f"\nFold {i+1}:")
        for split_name in ['train', 'val', 'test']:
            groups = defaultdict(set)
            for _, grade, oid, _ in fold_data[split_name]:
                groups[grade].add(oid)
            counts = {g: len(oids) for g, oids in sorted(groups.items())}
            total = sum(counts.values())
            n_files = sum(1 for _ in fold_data[split_name])
            print(f"  {split_name}: {total} grp, {n_files} files")

    # ── Run folds ──────────────────────────────────────────────────
    fold_indices = [args.fold] if args.fold is not None else range(args.n_folds)
    all_metrics = []

    for fold_idx in fold_indices:
        set_seed(args.seed + fold_idx)
        fold_dir = Path(tempfile.mkdtemp(prefix=f"dfu_cv_f{fold_idx}_"))
        try:
            metrics = train_fold(fold_idx, folds[fold_idx], config, device, fold_dir)
            all_metrics.append(metrics)
        finally:
            if fold_dir.exists():
                shutil.rmtree(fold_dir, ignore_errors=True)

    if len(all_metrics) < 2:
        print("\nSingle fold only — run all 3 for full CV report.")
        return

    # ── Aggregate results ──────────────────────────────────────────
    print(f"\n{'='*75}")
    print("3-Fold Cross-Validation Results (Mean ± 95% CI)")
    print(f"{'='*75}")

    metric_names = ["accuracy", "macro_f1", "kappa"]
    agg = {}

    print(f"\n{'Metric':<18}", end="")
    for i in range(len(all_metrics)):
        print(f"{'Fold '+str(i+1):>12}", end="")
    print(f"{'Mean ± 95% CI':>24}")
    print("-" * 75)

    for name in metric_names:
        vals = [m[name] for m in all_metrics]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ci = scipy_stats.t.ppf(0.975, len(vals)-1) * std / np.sqrt(len(vals))
        agg[name] = {"mean": mean, "std": std, "ci": ci}
        print(f"{name:<18}", end="")
        for v in vals:
            print(f"{v:>12.4f}", end="")
        print(f"{mean:>12.4f} ± {ci:.4f}")

    print(f"\n{'Per-Class F1':<18}", end="")
    for i in range(len(all_metrics)):
        print(f"{'Fold '+str(i+1):>12}", end="")
    print(f"{'Mean ± 95% CI':>24}")
    print("-" * 75)

    for cls_idx in range(NUM_CLASSES - 1):  # Skip grade5
        cls_name = IDX_TO_LABEL[cls_idx]
        vals = [m["f1_per_class"][cls_idx] for m in all_metrics]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ci = scipy_stats.t.ppf(0.975, len(vals)-1) * std / np.sqrt(len(vals))
        print(f"{cls_name:<18}", end="")
        for v in vals:
            print(f"{v:>12.4f}", end="")
        print(f"{mean:>12.4f} ± {ci:.4f}")

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [f"fold_{i}" for i in range(len(all_metrics))]
                         + ["mean", "std", "ci95"])
        for name in metric_names:
            vals = [m[name] for m in all_metrics]
            writer.writerow([name] + vals + [agg[name]["mean"], agg[name]["std"], agg[name]["ci"]])
        for cls_idx in range(NUM_CLASSES - 1):
            vals = [m["f1_per_class"][cls_idx] for m in all_metrics]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            ci = scipy_stats.t.ppf(0.975, len(vals)-1) * std / np.sqrt(len(vals))
            writer.writerow([f"f1_{IDX_TO_LABEL[cls_idx]}"] + vals + [mean, std, ci])

    print(f"\n✅ CV results saved to {args.output}")


if __name__ == "__main__":
    main()
