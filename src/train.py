#!/usr/bin/env python3
"""
Training script for DFU Wagner 0-5 grading system.

Supports:
  - Binary mode: benign (normal+grade0) vs ulcer (wound+gangrene)
  - Ordinal mode: 4-class CORN (normal < grade0 < wound < gangrene)
  - Focal Loss for class imbalance
  - Mixed precision (AMP)
  - Cosine warm-restart LR scheduler
  - Early stopping on validation F1
"""

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import (
    DFUDataset,
    IDX_TO_LABEL,
    LABEL_NAMES,
    LABEL_TO_IDX,
    NUM_BINARY,
    NUM_CLASSES,
)
from model import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    corn_loss,
    get_convnext_tiny,
    get_efficientnet_b0,
    get_resnet50,
)


# ─── Helpers ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Train/Val Epochs ─────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device,
                binary_mode: bool, log_interval: int,
                classify_mode: bool = False,
                class_weights: torch.Tensor | None = None,
                joint_mode: bool = False,
                binary_criterion=None,
                binary_loss_weight: float = 0.5):
    """Train one epoch. Returns (loss, acc, f1, preds, labels)
       plus (binary_acc, binary_f1) in joint mode."""
    model.train()
    running_loss = 0.0
    running_binary_loss = 0.0
    running_ordinal_loss = 0.0
    all_preds, all_labels = [], []
    all_binary_preds, all_binary_labels = [], []

    if joint_mode:
        task = "both"
    elif binary_mode:
        task = "binary"
    elif classify_mode:
        task = "classify"
    else:
        task = "ordinal"

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if joint_mode:
                bin_logits, ord_logits = model(images, task="both")
                # Binary labels: normal(0)+grade0(1) → benign(0); grade1+(2+) → ulcer(1)
                binary_labels = (labels >= 2).long()
                loss_bin = binary_criterion(bin_logits, binary_labels.float())
                if class_weights is not None:
                    sample_w = class_weights[labels]
                    loss_ord = criterion(ord_logits, labels, sample_w)
                else:
                    loss_ord = criterion(ord_logits, labels)
                loss = binary_loss_weight * loss_bin + loss_ord
                # Predictions (ordinal for primary task)
                preds = model.ordinal_head.predict(ord_logits)
                bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
                running_binary_loss += loss_bin.item()
                running_ordinal_loss += loss_ord.item()
            elif binary_mode:
                logits = model(images, task="binary")          # [B]
                loss = criterion(logits, labels.float())
                preds = (torch.sigmoid(logits) >= 0.5).long()
            elif classify_mode:
                logits = model(images, task="classify")         # [B, num_classes]
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
            else:
                logits = model(images, task="ordinal")          # [B, K-1]
                if class_weights is not None:
                    sample_w = class_weights[labels]            # [B]
                    loss = criterion(logits, labels, sample_w)
                else:
                    loss = criterion(logits, labels)
                preds = model.ordinal_head.predict(logits)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        if joint_mode:
            all_binary_preds.extend(bin_preds.detach().cpu().numpy())
            all_binary_labels.extend(binary_labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if joint_mode:
        bin_acc = accuracy_score(all_binary_labels, all_binary_preds)
        bin_f1 = f1_score(all_binary_labels, all_binary_preds, average="binary", zero_division=0)
        return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels, bin_acc, bin_f1

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, binary_mode: bool,
                   classify_mode: bool = False,
                   joint_mode: bool = False,
                   binary_criterion=None,
                   binary_loss_weight: float = 0.5):
    """Validate one epoch. Returns (loss, acc, f1, kappa, preds, labels)
       plus (binary_acc, binary_f1) in joint mode."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_binary_preds, all_binary_labels = [], []

    if joint_mode:
        task = "both"
    elif binary_mode:
        task = "binary"
    elif classify_mode:
        task = "classify"
    else:
        task = "ordinal"

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            if joint_mode:
                bin_logits, ord_logits = model(images, task="both")
                binary_labels = (labels >= 2).long()
                loss_bin = binary_criterion(bin_logits, binary_labels.float())
                loss_ord = criterion(ord_logits, labels)
                loss = binary_loss_weight * loss_bin + loss_ord
                preds = model.ordinal_head.predict(ord_logits)
                bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
                all_binary_preds.extend(bin_preds.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())
            elif binary_mode:
                logits = model(images, task="binary")
                loss = criterion(logits, labels.float())
                preds = (torch.sigmoid(logits) >= 0.5).long()
            elif classify_mode:
                logits = model(images, task="classify")
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
            else:
                logits = model(images, task="ordinal")
                loss = criterion(logits, labels)
                preds = model.ordinal_head.predict(logits)

        running_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    if joint_mode:
        bin_acc = accuracy_score(all_binary_labels, all_binary_preds)
        bin_f1 = f1_score(all_binary_labels, all_binary_preds, average="binary", zero_division=0)
        return epoch_loss, epoch_acc, epoch_f1, kappa, all_preds, all_labels, bin_acc, bin_f1

    return epoch_loss, epoch_acc, epoch_f1, kappa, all_preds, all_labels


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    import yaml

    with open("/root/dfu/config.yaml") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────
    data_dir = config["data"]["data_dir"]
    input_size = config["data"]["input_size"]
    binary_mode = config["data"].get("binary", False)
    batch_size = config["training"]["batch_size"]
    num_classes = NUM_BINARY if binary_mode else NUM_CLASSES
    model_name = config["model"]["name"]
    use_corn = config["training"].get("use_corn", False) and not binary_mode
    classify_mode = not use_corn and not binary_mode  # standard CE / Focal Loss
    joint_mode = config["training"].get("joint_training", False) and not binary_mode
    binary_loss_weight = config["training"].get("binary_loss_weight", 0.5)
    num_workers = min(4, os.cpu_count() or 4)

    if joint_mode:
        mode_str = f"Joint (Binary + CORN ordinal, λ_bin={binary_loss_weight})"
    else:
        mode_str = ("Binary (benign/ulcer)" if binary_mode
                    else ("CORN ordinal" if use_corn else "Standard classification (Focal/CE)"))
    print(f"Mode: {mode_str} ({num_classes}-class)")
    print(f"Data:  {data_dir}")
    print(f"Model: {model_name}")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = DFUDataset(data_dir, "train", input_size, binary=binary_mode)
    val_ds = DFUDataset(data_dir, "val", input_size, binary=binary_mode)
    test_ds = DFUDataset(data_dir, "test", input_size, binary=binary_mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\n=== Building {model_name} ===")
    binary_head = config["model"].get("binary_head", True)

    if model_name == "convnext_tiny":
        model = get_convnext_tiny(num_classes=num_classes, binary=binary_head,
                                   classify_head=classify_mode)
    elif model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, binary=binary_head,
                              classify_head=classify_mode)
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_b0(num_classes=num_classes, binary=binary_head,
                                     classify_head=classify_mode)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────
    corn_class_weights = None  # only used for CORN mode
    binary_criterion = None     # only used for joint mode

    if binary_mode:
        # Binary classification — use BCEWithLogitsLoss with pos_weight
        # Count positives for pos_weight
        all_labels = torch.tensor(train_ds.all_targets)
        n_pos = all_labels.sum().item()
        n_neg = len(all_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.2f})")
    elif joint_mode:
        # Joint training: binary BCE + CORN ordinal
        criterion = corn_loss
        corn_class_weights = train_ds.get_class_weights().to(device)
        # Binary criterion for screening head
        all_labels = torch.tensor(train_ds.all_targets)
        binary_labels = (all_labels >= 2).long()  # grade1+ = ulcer
        n_pos = binary_labels.sum().item()
        n_neg = len(binary_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
        binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: Joint BCE(pos_weight={pos_weight.item():.2f}) + λ_bin={binary_loss_weight} * CORN")
        print(f"  CORN class_weights={corn_class_weights.tolist()}")
    elif use_corn:
        criterion = corn_loss
        corn_class_weights = train_ds.get_class_weights().to(device)
        print(f"Loss: CORN binary cross-entropy (ordinal, class_weights={corn_class_weights.tolist()})")
    else:
        # Standard classification with optional focal loss
        focal_gamma = config["training"].get("focal_gamma", 0)
        if config["training"].get("use_class_weights", False):
            class_weights = train_ds.get_class_weights().to(device)
        else:
            class_weights = None

        if focal_gamma > 0:
            alpha = class_weights if config["training"].get("use_class_weights") else None
            criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
            print(f"Loss: FocalLoss (γ={focal_gamma}, class_weights={alpha is not None})")
        else:
            smoothing = config["training"].get("label_smoothing", 0.0)
            if smoothing > 0:
                criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
                print(f"Loss: LabelSmoothingCrossEntropy (α={smoothing})")
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"Loss: CrossEntropyLoss (class_weights={class_weights is not None})")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    opt_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["learning_rate"],
        weight_decay=opt_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=opt_cfg["lr_t_0"], T_mult=opt_cfg["lr_t_mult"],
    )

    # ── AMP ───────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if config["training"]["use_amp"] and torch.cuda.is_available() else None

    # ── Logging ────────────────────────────────────────────────────────
    ckpt_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(config["logging"]["tensorboard_dir"])
    csv_file = open(config["logging"]["csv_log"], "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "train_loss", "train_acc", "train_f1",
        "val_loss", "val_acc", "val_f1", "val_kappa", "lr",
    ] + (["train_bin_acc", "train_bin_f1", "val_bin_acc", "val_bin_f1"] if joint_mode else []))

    # ── Training Loop ─────────────────────────────────────────────────
    best_val_f1 = 0.0
    patience_counter = 0
    patience = opt_cfg["early_stopping_patience"]
    epochs = opt_cfg["epochs"]

    print(f"\n=== Training ({epochs} epochs, patience={patience}) ===\n")
    for epoch in range(epochs):
        if joint_mode:
            tr_loss, tr_acc, tr_f1, _, _, tr_bin_acc, tr_bin_f1 = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                binary_mode=False, log_interval=config["logging"]["log_interval"],
                classify_mode=False,
                class_weights=corn_class_weights,
                joint_mode=True, binary_criterion=binary_criterion,
                binary_loss_weight=binary_loss_weight,
            )
            vl_loss, vl_acc, vl_f1, vl_kappa, _, _, vl_bin_acc, vl_bin_f1 = validate_epoch(
                model, val_loader, criterion, device, binary_mode=False,
                classify_mode=False, joint_mode=True,
                binary_criterion=binary_criterion,
                binary_loss_weight=binary_loss_weight,
            )
        else:
            tr_loss, tr_acc, tr_f1, _, _ = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                binary_mode, config["logging"]["log_interval"],
                classify_mode=classify_mode,
                class_weights=corn_class_weights if use_corn else None,
            )
            vl_loss, vl_acc, vl_f1, vl_kappa, _, _ = validate_epoch(
                model, val_loader, criterion, device, binary_mode,
                classify_mode=classify_mode,
            )

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # TensorBoard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", vl_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", vl_acc, epoch)
        writer.add_scalar("F1/train", tr_f1, epoch)
        writer.add_scalar("F1/val", vl_f1, epoch)
        writer.add_scalar("Kappa/val", vl_kappa, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if joint_mode:
            csv_writer.writerow(
                [epoch, tr_loss, tr_acc, tr_f1, vl_loss, vl_acc, vl_f1, vl_kappa, current_lr,
                 tr_bin_acc, tr_bin_f1, vl_bin_acc, vl_bin_f1]
            )
            print(
                f"Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.2e} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} "
                f"Bin: {tr_bin_acc:.4f}/{tr_bin_f1:.4f} | "
                f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} F1: {vl_f1:.4f} Kappa: {vl_kappa:.4f} "
                f"Bin: {vl_bin_acc:.4f}/{vl_bin_f1:.4f}"
            )
        else:
            csv_writer.writerow(
                [epoch, tr_loss, tr_acc, tr_f1, vl_loss, vl_acc, vl_f1, vl_kappa, current_lr]
            )
            print(
                f"Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.2e} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | "
                f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} F1: {vl_f1:.4f} Kappa: {vl_kappa:.4f}"
            )

        csv_file.flush()

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            patience_counter = 0
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": vl_f1,
                "val_acc": vl_acc,
                "val_kappa": vl_kappa,
                "config": config,
            }
            if joint_mode:
                ckpt_data["val_bin_acc"] = vl_bin_acc
                ckpt_data["val_bin_f1"] = vl_bin_f1
            torch.save(ckpt_data, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  ✅ Best model saved (F1={vl_f1:.4f})" +
                  (f" BinF1={vl_bin_f1:.4f}" if joint_mode else ""))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹  Early stopping at epoch {epoch+1}")
                break

    csv_file.close()
    writer.close()

    # ── Final Test Evaluation ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")

    checkpoint = torch.load(
        os.path.join(ckpt_dir, "best_model.pth"), weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    all_binary_preds, all_binary_labels = [], []

    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                if joint_mode:
                    bin_logits, ord_logits = model(images, task="both")
                    binary_targets = (labels >= 2).long()
                    preds = model.ordinal_head.predict(ord_logits)
                    bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
                    all_binary_preds.extend(bin_preds.cpu().numpy())
                    all_binary_labels.extend(binary_targets.numpy())
                elif binary_mode:
                    logits = model(images, task="binary")
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                elif classify_mode:
                    logits = model(images, task="classify")
                    preds = logits.argmax(dim=1)
                else:
                    logits = model(images, task="ordinal")
                    preds = model.ordinal_head.predict(logits)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    n_cls = NUM_BINARY if binary_mode else NUM_CLASSES
    prec, rec, f1s, supp = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(n_cls)), zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_cls)))

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Test Kappa:    {kappa:.4f}")

    if joint_mode:
        bin_acc = accuracy_score(all_binary_labels, all_binary_preds)
        bin_f1 = f1_score(all_binary_labels, all_binary_preds, average="binary", zero_division=0)
        bin_prec, bin_rec, bin_f1s, bin_supp = precision_recall_fscore_support(
            all_binary_labels, all_binary_preds, labels=[0, 1], zero_division=0,
        )
        bin_cm = confusion_matrix(all_binary_labels, all_binary_preds, labels=[0, 1])
        print(f"\n--- Binary Screening Head ---")
        print(f"Binary Accuracy: {bin_acc:.4f}")
        print(f"Binary F1:       {bin_f1:.4f}")
        print(f"  Benign: Prec={bin_prec[0]:.4f} Rec={bin_rec[0]:.4f} F1={bin_f1s[0]:.4f} N={bin_supp[0]}")
        print(f"  Ulcer:  Prec={bin_prec[1]:.4f} Rec={bin_rec[1]:.4f} F1={bin_f1s[1]:.4f} N={bin_supp[1]}")
        print(f"\nBinary Confusion Matrix:")
        print(f"            Pred Benign  Pred Ulcer")
        print(f"True Benign  {bin_cm[0][0]:>11d}  {bin_cm[0][1]:>10d}")
        print(f"True Ulcer   {bin_cm[1][0]:>11d}  {bin_cm[1][1]:>10d}")

    print(f"\n=== Per-Class Metrics ===")
    if binary_mode:
        class_names = ["Benign", "Ulcer"]
    else:
        class_names = [f"{IDX_TO_LABEL[i]} ({LABEL_NAMES[IDX_TO_LABEL[i]]})" for i in range(n_cls)]
    for i, name in enumerate(class_names):
        if i < len(prec):
            print(f"  {name}: Prec={prec[i]:.4f} Rec={rec[i]:.4f} F1={f1s[i]:.4f} N={supp[i]}")

    print(f"\n=== Confusion Matrix ===")
    header = "      " + "  ".join(f"Pred {n.split('(')[0].strip():>8}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        short = name.split("(")[0].strip()
        row = "  ".join(f"{cm[i][j]:>13d}" for j in range(n_cls))
        print(f"True {short:<5}: {row}")

    # Save weights-only for inference
    torch.save(checkpoint["model_state_dict"],
               os.path.join(ckpt_dir, "best_model_weights.pth"))

    print(f"\n✅ Training complete!")
    print(f"   Model:  {ckpt_dir}/best_model.pth")
    print(f"   Weights:{ckpt_dir}/best_model_weights.pth")
    print(f"   CSV:    {config['logging']['csv_log']}")


if __name__ == "__main__":
    main()
