#!/usr/bin/env python3
"""
Probability calibration for CORN ordinal models.

Temperature Scaling: learns a single scalar T > 0 that is applied to
the CORN logits before sigmoid → probability decoding.  This sharpens
(T < 1) or smooths (T > 1) the predicted probabilities without changing
the class prediction (argmax is invariant to T).

Also provides Expected Calibration Error (ECE) to measure calibration
quality before and after scaling.

Usage:
    # 1. Train model normally, then calibrate on validation set:
    python src/calibrate.py --checkpoint models/corn_v3/best_model.pth

    # 2. Or use programmatically:
    from calibrate import calibrate_temperature, CalibratedModel
    model, T = calibrate_temperature(model, val_loader, device)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DFUDataset, NUM_CLASSES


# ═══════════════════════════════════════════════════════════════════════
# Temperature Scaling
# ═══════════════════════════════════════════════════════════════════════

class CalibratedCORN(nn.Module):
    """
    Wraps a CORNHead with a learned temperature parameter.

    At inference, logits are scaled:  logits_cal = logits / T
    Temperature T is learned on a calibration set to minimize NLL.
    """

    def __init__(self, corn_head: nn.Module, init_temperature: float = 1.0):
        super().__init__()
        self.corn_head = corn_head
        # Temperature is stored as log(T) so T = exp(log_T) > 0 always
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))

    @property
    def temperature(self) -> float:
        return torch.exp(self.log_temperature).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated logits = raw_logits / T."""
        raw_logits = self.corn_head(x)
        return raw_logits / torch.exp(self.log_temperature)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Predicted class — invariant to temperature."""
        return self.corn_head.predict(logits)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrated class probabilities via temperature scaling."""
        return self.corn_head.predict_proba(logits)


def calibrate_temperature(
    model,
    loader: DataLoader,
    device: torch.device,
    lr: float = 0.01,
    max_iter: int = 200,
    verbose: bool = True,
) -> float:
    """
    Learn the optimal temperature T on the given (validation) loader.

    Optimizes T to minimize NLL of the true class under CORN ordinal probabilities.
    The model backbone is frozen; only the temperature parameter is learned.

    Args:
        model: DFUModel with ordinal_head (already trained)
        loader: DataLoader for calibration set (typically validation)
        device: torch device
        lr: learning rate for temperature (LBFGS)
        max_iter: max LBFGS iterations
        verbose: print progress

    Returns:
        Optimal temperature T (float)
    """
    # ── Collect all logits and labels ────────────────────────────────
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting logits", leave=False):
            images = images.to(device)
            features = model.backbone(images)
            logits = model.ordinal_head(features)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)  # [N, K-1]
    labels = torch.cat(all_labels, dim=0)  # [N]

    if verbose:
        print(f"  Collected {logits.size(0)} samples for calibration")

    # ── Pre-calibration ECE ──────────────────────────────────────────
    ece_before = compute_ece(logits, labels, model.ordinal_head.num_classes)

    # ── Learn temperature via LBFGS ──────────────────────────────────
    log_T = nn.Parameter(torch.tensor(0.0))  # T = exp(0) = 1
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def nll_loss():
        """Negative log-likelihood of true class under temperature-scaled probs."""
        scaled_logits = logits / torch.exp(log_T)
        probs = model.ordinal_head.predict_proba(scaled_logits)  # [N, K]
        # Gather probability of true class
        true_probs = probs[torch.arange(probs.size(0)), labels]
        # Avoid log(0)
        true_probs = torch.clamp(true_probs, min=1e-8)
        return -torch.log(true_probs).mean()

    def closure():
        optimizer.zero_grad()
        loss = nll_loss()
        loss.backward()
        return loss

    initial_loss = nll_loss().item()
    optimizer.step(closure)
    final_loss = nll_loss().item()
    T_opt = torch.exp(log_T).item()

    # ── Post-calibration ECE ─────────────────────────────────────────
    scaled_logits = logits / T_opt
    ece_after = compute_ece(scaled_logits, labels, model.ordinal_head.num_classes)

    if verbose:
        print(f"  NLL: {initial_loss:.4f} → {final_loss:.4f}")
        print(f"  ECE: {ece_before:.4f} → {ece_after:.4f}")
        print(f"  Optimal T: {T_opt:.4f}")

    return T_opt


# ═══════════════════════════════════════════════════════════════════════
# Expected Calibration Error (ECE)
# ═══════════════════════════════════════════════════════════════════════

def compute_ece(logits: torch.Tensor, labels: torch.Tensor,
                num_classes: int, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    For each sample, takes confidence = max class probability.
    Bins samples by confidence, computes |accuracy - confidence| per bin,
    then averages weighted by bin size.

    Args:
        logits: [N, K-1] CORN threshold logits
        labels: [N] true class indices
        num_classes: K
        n_bins: number of confidence bins

    Returns:
        ECE score (lower = better calibrated)
    """
    probs = torch.sigmoid(logits)

    # Build class probabilities using telescoping CORN decode
    class_probs_list = []
    for i in range(num_classes):
        if i == 0:
            class_probs_list.append(1 - probs[:, 0])
        elif i == num_classes - 1:
            class_probs_list.append(probs[:, -1])
        else:
            class_probs_list.append(probs[:, i - 1] - probs[:, i])
    class_probs = torch.stack(class_probs_list, dim=1)  # [N, K]

    # Clamp negatives (shouldn't happen with shared-weight CORN, but safety)
    class_probs = class_probs.clamp(0, 1)
    # Normalize per row
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)

    confidences, predictions = class_probs.max(dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum().item()
        if bin_size > 0:
            bin_acc = accuracies[in_bin].mean().item()
            bin_conf = confidences[in_bin].mean().item()
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)

    return ece


# ═══════════════════════════════════════════════════════════════════════
# Reliability Diagram Data
# ═══════════════════════════════════════════════════════════════════════

def reliability_curve(logits: torch.Tensor, labels: torch.Tensor,
                      num_classes: int, n_bins: int = 10):
    """
    Compute per-bin accuracy and confidence for reliability diagram.

    Returns:
        bin_confidences: list of mean confidence per bin
        bin_accuracies: list of mean accuracy per bin
        bin_sizes: list of sample counts per bin
    """
    probs = torch.sigmoid(logits)

    class_probs_list = []
    for i in range(num_classes):
        if i == 0:
            class_probs_list.append(1 - probs[:, 0])
        elif i == num_classes - 1:
            class_probs_list.append(probs[:, -1])
        else:
            class_probs_list.append(probs[:, i - 1] - probs[:, i])
    class_probs = torch.stack(class_probs_list, dim=1).clamp(0, 1)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)

    confidences, predictions = class_probs.max(dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_sizes = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum().item()
        bin_sizes.append(bin_size)
        if bin_size > 0:
            bin_confidences.append(confidences[in_bin].mean().item())
            bin_accuracies.append(accuracies[in_bin].mean().item())
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    return bin_confidences, bin_accuracies, bin_sizes


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from model import get_convnext_tiny, get_efficientnet_b0, get_resnet50

    parser = argparse.ArgumentParser(
        description="Temperature calibration for CORN DFU model"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="/root/dfu/models/corn_v3/best_model.pth")
    parser.add_argument("--config", type=str,
                        default="/root/dfu/config.yaml")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Path to save calibrated checkpoint (default: <ckpt>_calibrated.pth)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
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

    # Calibration data (validation set)
    data_dir = config["data"]["data_dir"]
    input_size = config["data"]["input_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = min(4, os.cpu_count() or 4)

    val_ds = DFUDataset(data_dir, "val", input_size, binary=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Calibrate
    print(f"\n=== Temperature Calibration ===")
    T_opt = calibrate_temperature(model, val_loader, device)

    # Save calibrated checkpoint
    output_path = args.output or args.checkpoint.replace(".pth", "_calibrated.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    checkpoint["temperature"] = T_opt
    torch.save(checkpoint, output_path)
    print(f"\n✅ Calibrated model saved to: {output_path}")
    print(f"   Temperature: {T_opt:.4f}")
    print(f"   Usage: divide CORN logits by {T_opt:.4f} before predict_proba")


if __name__ == "__main__":
    import sys
    main()
