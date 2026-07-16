#!/usr/bin/env python3
"""
Independent evaluation of deployed DFU model weights.

Loads a trained checkpoint, runs full evaluation on the test set, and
produces a comprehensive report: accuracy, per-class F1, sensitivity,
specificity, confusion matrix, and comparison with CV results.

This solves Problem 3: the deployed weights' metrics should be independently
verified and compared against the cross-validation report.

Usage:
    # Standard evaluation
    python src/test_deployed_model.py

    # With TTA
    python src/test_deployed_model.py --tta --tta-views 6

    # Custom checkpoint
    python src/test_deployed_model.py --checkpoint models/corn_v3/best_model.pth

    # Save report to file
    python src/test_deployed_model.py --output reports/deployment_test.json
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
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

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import DFUDataset, IDX_TO_LABEL, LABEL_NAMES, NUM_CLASSES
from model import get_convnext_tiny, get_efficientnet_b0, get_resnet50
from tta import tta_predict_ordinal


# ═══════════════════════════════════════════════════════════════════════
# Core Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_standard(model, loader, device, joint_mode: bool = False):
    """Standard single-pass evaluation. Returns (preds, labels, binary_preds, binary_labels)."""
    model.eval()
    all_preds, all_labels = [], []
    all_bin_preds, all_bin_labels = [], []

    for images, labels in tqdm(loader, desc="Standard evaluation", leave=False):
        images, labels = images.to(device), labels.to(device)

        if joint_mode:
            bin_logits, ord_logits = model(images, task="both")
            preds = model.ordinal_head.predict(ord_logits)
            bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
            bin_labels = (labels >= 2).long()
            all_bin_preds.extend(bin_preds.cpu().numpy())
            all_bin_labels.extend(bin_labels.cpu().numpy())
        else:
            logits = model(images, task="ordinal")
            preds = model.ordinal_head.predict(logits)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    result = (all_preds, all_labels)
    if joint_mode:
        result += (all_bin_preds, all_bin_labels)
    return result


@torch.no_grad()
def evaluate_tta(model, loader, device, n_views: int = 4,
                 joint_mode: bool = False):
    """TTA evaluation — per-image multi-view averaging."""
    model.eval()
    all_preds, all_labels = [], []
    all_bin_preds, all_bin_labels = [], []

    for images, labels in tqdm(loader, desc=f"TTA ({n_views} views)", leave=False):
        for i in range(images.size(0)):
            img = images[i]  # [3, H, W]
            label = labels[i].item()

            pred, _ = tta_predict_ordinal(model, img, device, n_views=n_views)
            all_preds.append(pred)
            all_labels.append(label)

            if joint_mode:
                # TTA binary: average logit across views
                from tta import get_tta_views
                views = get_tta_views(img, n_views)
                bin_logits = []
                for view in views:
                    view = view.to(device)
                    logit = model(view, task="binary")
                    bin_logits.append(logit)
                avg_bin = torch.stack(bin_logits).mean()
                bin_pred = 1 if torch.sigmoid(avg_bin).item() >= 0.5 else 0
                all_bin_preds.append(bin_pred)
                all_bin_labels.append(1 if label >= 2 else 0)

    result = (all_preds, all_labels)
    if joint_mode:
        result += (all_bin_preds, all_bin_labels)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Metrics Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_ordinal_metrics(labels, preds, class_names):
    """Compute all ordinal (Wagner grading) metrics."""
    n_cls = NUM_CLASSES

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")

    prec, rec, f1s, supp = precision_recall_fscore_support(
        labels, preds, labels=list(range(n_cls)), zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))

    # Sensitivity = Recall per class
    # Specificity = TN / (TN + FP) per class
    specificities = []
    for i in range(n_cls):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / max(tn + fp, 1)
        specificities.append(spec)

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "kappa": kappa,
        "per_class": {
            IDX_TO_LABEL[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),          # = sensitivity
                "sensitivity": float(rec[i]),
                "specificity": float(specificities[i]),
                "f1": float(f1s[i]),
                "support": int(supp[i]),
            }
            for i in range(n_cls)
            if class_names[i] is not None
        },
        "confusion_matrix": cm.tolist(),
    }


def compute_binary_metrics(labels, preds):
    """Compute binary screening metrics."""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    prec, rec, f1s, supp = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": acc,
        "f1": f1,
        "benign": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "specificity": float(specificity),
            "f1": float(f1s[0]),
            "support": int(supp[0]),
        },
        "ulcer": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),          # = sensitivity
            "sensitivity": float(rec[1]),
            "f1": float(f1s[1]),
            "support": int(supp[1]),
        },
        "confusion_matrix": cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Report Formatting
# ═══════════════════════════════════════════════════════════════════════

def print_ordinal_report(metrics, title="Ordinal Wagner Grading"):
    """Pretty-print ordinal metrics."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"  Cohen's Kappa:{metrics['kappa']:.4f}")

    print(f"\n  {'Class':<12} {'Prec':>8} {'Sens(Rec)':>10} {'Spec':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*60}")
    for label, m in metrics["per_class"].items():
        if m["support"] > 0:
            print(f"  {label:<12} {m['precision']:>8.4f} {m['sensitivity']:>10.4f} "
                  f"{m['specificity']:>8.4f} {m['f1']:>8.4f} {m['support']:>8d}")
        else:
            print(f"  {label:<12} {'—':>8} {'—':>10} {'—':>8} {'—':>8} {'(empty)':>8}")

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    labels = list(metrics["per_class"].keys())
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    header = "  " + "".join(f"{l:>8}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        row = "".join(f"{cm[i][j]:>8d}" for j in range(len(labels)))
        print(f"  {l:<10}{row}")

    # Normalized confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0)
    print(f"\n  Normalized Confusion Matrix (row-wise):")
    print(header)
    for i, l in enumerate(labels):
        row = "".join(f"{cm_norm[i][j]:>8.2f}" for j in range(len(labels)))
        print(f"  {l:<10}{row}")


def print_binary_report(metrics):
    """Pretty-print binary screening metrics."""
    print(f"\n  --- Binary Screening Head ---")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  Benign: P={metrics['benign']['precision']:.4f} "
          f"R={metrics['benign']['recall']:.4f} "
          f"F1={metrics['benign']['f1']:.4f} "
          f"N={metrics['benign']['support']}")
    print(f"  Ulcer:  P={metrics['ulcer']['precision']:.4f} "
          f"R={metrics['ulcer']['recall']:.4f} "
          f"Sens={metrics['ulcer']['sensitivity']:.4f} "
          f"F1={metrics['ulcer']['f1']:.4f} "
          f"N={metrics['ulcer']['support']}")

    cm = np.array(metrics["confusion_matrix"])
    print(f"\n  Binary Confusion Matrix:")
    print(f"              Pred Benign  Pred Ulcer")
    print(f"  True Benign  {cm[0,0]:>11d}  {cm[0,1]:>10d}")
    print(f"  True Ulcer   {cm[1,0]:>11d}  {cm[1,1]:>10d}")


def compare_with_cv(metrics, cv_path: str):
    """Compare deployed model metrics with CV report."""
    if not os.path.exists(cv_path):
        print(f"\n  ⚠️ CV results not found at {cv_path} — skipping comparison")
        return None

    import pandas as pd
    cv = pd.read_csv(cv_path, index_col=0)

    comparison = {}
    print(f"\n{'='*70}")
    print(f"  Comparison: Deployed Model vs CV Report (Mean ± 95% CI)")
    print(f"{'='*70}")
    print(f"  {'Metric':<18} {'Deployed':>10} {'CV Mean':>10} {'CV 95% CI':>12} {'In CI?':>8}")
    print(f"  {'-'*60}")

    mapping = {
        "accuracy": "accuracy",
        "macro_f1": "macro_f1",
        "kappa": "kappa",
    }

    for disp_name, cv_name in mapping.items():
        deployed_val = metrics[disp_name]
        if cv_name in cv.index:
            cv_mean = cv.loc[cv_name, "mean"]
            cv_ci = cv.loc[cv_name, "ci95"]
            in_ci = abs(deployed_val - cv_mean) <= cv_ci
            print(f"  {disp_name:<18} {deployed_val:>10.4f} {cv_mean:>10.4f} "
                  f"{cv_ci:>12.4f} {'✅' if in_ci else '⚠️':>8}")
            comparison[cv_name] = {
                "deployed": deployed_val,
                "cv_mean": cv_mean,
                "cv_ci95": cv_ci,
                "in_ci": bool(in_ci),
            }

    # Per-class F1
    for label, m in metrics["per_class"].items():
        cv_name = f"f1_{label}"
        if cv_name in cv.index:
            deployed_val = m["f1"]
            cv_mean = cv.loc[cv_name, "mean"]
            cv_ci = cv.loc[cv_name, "ci95"]
            in_ci = abs(deployed_val - cv_mean) <= cv_ci
            print(f"  f1_{label:<13} {deployed_val:>10.4f} {cv_mean:>10.4f} "
                  f"{cv_ci:>12.4f} {'✅' if in_ci else '⚠️':>8}")
            comparison[cv_name] = {
                "deployed": deployed_val,
                "cv_mean": cv_mean,
                "cv_ci95": cv_ci,
                "in_ci": bool(in_ci),
            }

    n_outside = sum(1 for v in comparison.values() if not v["in_ci"])
    if n_outside > 0:
        print(f"\n  ⚠️ {n_outside} metric(s) outside CV 95% CI — deployment weights differ from CV folds")
    else:
        print(f"\n  ✅ All metrics within CV 95% CI")

    return comparison


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Independent evaluation of deployed DFU model weights"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="/root/dfu/models/corn_v3/best_model.pth")
    parser.add_argument("--config", type=str,
                        default="/root/dfu/config.yaml")
    parser.add_argument("--cv-results", type=str,
                        default="/root/dfu/models/corn_v3/cv_results.csv")
    parser.add_argument("--tta", action="store_true",
                        help="Enable TTA evaluation")
    parser.add_argument("--tta-views", type=int, default=4,
                        help="Number of TTA views")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to file")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to evaluate (test/val/train)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    if args.tta:
        print(f"TTA: {args.tta_views} views")

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", config)
    model_name = model_cfg["model"]["name"]
    num_classes = model_cfg["model"]["num_classes"]
    binary_head = model_cfg["model"].get("binary_head", True)
    joint_mode = model_cfg["training"].get("joint_training", False)

    print(f"  Model: {model_name}")
    print(f"  Classes: {num_classes}")
    print(f"  Binary head: {binary_head}")
    print(f"  Joint training: {joint_mode}")
    print(f"  Saved epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Saved val_f1: {checkpoint.get('val_f1', 'N/A')}")
    if "temperature" in checkpoint:
        print(f"  Temperature: {checkpoint.get('temperature', 1.0):.4f}")

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

    # Apply temperature scaling if calibrated
    temperature = checkpoint.get("temperature", 1.0)
    if temperature != 1.0:
        print(f"  Using temperature scaling: T = {temperature:.4f}")
        # Wrap ordinal head forward to apply temperature
        _orig_forward = model.ordinal_head.forward
        def _calibrated_forward(x):
            return _orig_forward(x) / temperature
        model.ordinal_head.forward = _calibrated_forward

    # Data
    data_dir = config["data"]["data_dir"]
    input_size = config["data"]["input_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = min(4, os.cpu_count() or 4)

    test_ds = DFUDataset(data_dir, args.split, input_size, binary=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Evaluate ────────────────────────────────────────────────────
    class_names = [IDX_TO_LABEL.get(i) for i in range(NUM_CLASSES)]

    if args.tta:
        print(f"\n=== TTA Evaluation ({args.tta_views} views) ===")
        eval_result = evaluate_tta(model, test_loader, device,
                                   n_views=args.tta_views,
                                   joint_mode=joint_mode)
    else:
        print(f"\n=== Standard Evaluation ===")
        eval_result = evaluate_standard(model, test_loader, device,
                                        joint_mode=joint_mode)

    if joint_mode:
        preds, labels, bin_preds, bin_labels = eval_result
    else:
        preds, labels = eval_result
        bin_preds, bin_labels = None, None

    # Compute metrics
    ordinal_metrics = compute_ordinal_metrics(labels, preds, class_names)
    print_ordinal_report(ordinal_metrics)

    binary_metrics = None
    if joint_mode and bin_preds is not None:
        binary_metrics = compute_binary_metrics(bin_labels, bin_preds)
        print_binary_report(binary_metrics)

    # ── Compare with CV ─────────────────────────────────────────────
    cv_comparison = None
    cv_path = args.cv_results
    if os.path.exists(cv_path):
        cv_comparison = compare_with_cv(ordinal_metrics, cv_path)

    # ── Save report ─────────────────────────────────────────────────
    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "num_classes": num_classes,
        "joint_mode": joint_mode,
        "temperature": temperature,
        "tta_used": args.tta,
        "tta_views": args.tta_views if args.tta else 0,
        "ordinal_metrics": ordinal_metrics,
    }

    if binary_metrics is not None:
        report["binary_metrics"] = binary_metrics
    if cv_comparison is not None:
        report["cv_comparison"] = cv_comparison

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Report saved to: {args.output}")

    # Default: always save alongside checkpoint
    default_output = os.path.join(
        os.path.dirname(args.checkpoint),
        f"deployment_test_{args.split}.json"
    )
    os.makedirs(os.path.dirname(default_output), exist_ok=True)
    with open(default_output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ Report also saved to: {default_output}")


if __name__ == "__main__":
    main()
