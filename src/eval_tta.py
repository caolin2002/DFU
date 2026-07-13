#!/usr/bin/env python3
"""
TTA (Test-Time Augmentation) evaluation for existing DFU model.

Loads a trained CORN model, evaluates on test set with and without TTA,
and compares results.

Usage:
    python src/eval_tta.py
    python src/eval_tta.py --checkpoint models/corn_v2/best_model.pth --n_views 6
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DFUDataset, IDX_TO_LABEL, LABEL_NAMES, NUM_CLASSES
from model import get_convnext_tiny, get_efficientnet_b0, get_resnet50
from tta import tta_predict_ordinal, get_tta_views


@torch.no_grad()
def evaluate_standard(model, loader, device):
    """Standard evaluation (single forward pass per image)."""
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Standard eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images, task="ordinal")
        preds = model.ordinal_head.predict(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


@torch.no_grad()
def evaluate_tta(model, loader, device, n_views: int = 4):
    """
    TTA evaluation — for each image, average predictions across N views.
    Operates per-image (not batched) to keep memory low.
    """
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc=f"TTA ({n_views} views)", leave=False):
        for i in range(images.size(0)):
            img = images[i]  # [3, H, W]
            label = labels[i].item()

            pred, _ = tta_predict_ordinal(model, img, device, n_views=n_views)
            all_preds.append(pred)
            all_labels.append(label)

    return all_preds, all_labels


def compute_metrics(all_labels, all_preds):
    """Compute and return all metrics."""
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    prec, rec, f1s, supp = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "kappa": kappa,
        "precision": prec,
        "recall": rec,
        "f1_per_class": f1s,
        "support": supp,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="TTA Evaluation for DFU model")
    parser.add_argument("--checkpoint", type=str,
                        default="/root/dfu/models/corn_v2/best_model.pth")
    parser.add_argument("--config", type=str,
                        default="/root/dfu/config.yaml")
    parser.add_argument("--n_views", type=int, default=4,
                        help="Number of TTA views (default 4)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"TTA views: {args.n_views}")

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", config)
    model_name = model_cfg["model"]["name"]
    num_classes = model_cfg["model"]["num_classes"]
    binary_head = model_cfg["model"].get("binary_head", True)

    if model_name == "convnext_tiny":
        model = get_convnext_tiny(num_classes=num_classes, binary=binary_head)
    elif model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, binary=binary_head)
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_b0(num_classes=num_classes, binary=binary_head)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Model: {model_name} | Epoch saved: {checkpoint.get('epoch', 'N/A')}")

    # Data
    data_dir = config["data"]["data_dir"]
    input_size = config["data"]["input_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = min(4, os.cpu_count() or 4)

    test_ds = DFUDataset(data_dir, "test", input_size, binary=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Standard Evaluation ────────────────────────────────────────────
    print("\n=== Standard Evaluation ===")
    std_preds, std_labels = evaluate_standard(model, test_loader, device)
    std_metrics = compute_metrics(std_labels, std_preds)

    print(f"Accuracy: {std_metrics['accuracy']:.4f}")
    print(f"Macro F1: {std_metrics['macro_f1']:.4f}")
    print(f"Kappa:    {std_metrics['kappa']:.4f}")

    # ── TTA Evaluation ─────────────────────────────────────────────────
    print(f"\n=== TTA Evaluation ({args.n_views} views) ===")
    tta_preds, tta_labels = evaluate_tta(model, test_loader, device, n_views=args.n_views)
    tta_metrics = compute_metrics(tta_labels, tta_preds)

    print(f"Accuracy: {tta_metrics['accuracy']:.4f}")
    print(f"Macro F1: {tta_metrics['macro_f1']:.4f}")
    print(f"Kappa:    {tta_metrics['kappa']:.4f}")

    # ── Comparison ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Metric':<15} {'Standard':>10} {'TTA':>10} {'Δ':>10}")
    print(f"{'-'*45}")
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Kappa", "kappa"),
    ]:
        s = std_metrics[key]
        t = tta_metrics[key]
        delta = t - s
        sign = "+" if delta >= 0 else ""
        print(f"{metric_name:<15} {s:>10.4f} {t:>10.4f} {sign}{delta:>9.4f}")

    print(f"\n=== Per-Class F1 Comparison ===")
    print(f"{'Class':<15} {'Standard':>10} {'TTA':>10} {'Δ':>10}")
    print(f"{'-'*45}")
    for i in range(NUM_CLASSES):
        name = IDX_TO_LABEL[i]
        s = std_metrics["f1_per_class"][i]
        t = tta_metrics["f1_per_class"][i]
        delta = t - s
        sign = "+" if delta >= 0 else ""
        if std_metrics["support"][i] > 0:
            print(f"{name:<15} {s:>10.4f} {t:>10.4f} {sign}{delta:>9.4f}")

    print(f"\n=== TTA Confusion Matrix ===")
    cm = tta_metrics["confusion_matrix"]
    class_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    header = "      " + "  ".join(f"Pred {n:>8}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i][j]:>13d}" for j in range(NUM_CLASSES))
        print(f"True {name:<5}: {row}")


if __name__ == "__main__":
    main()
