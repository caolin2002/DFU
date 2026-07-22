#!/usr/bin/env python3
"""
Fine-tuning script for DFU Wagner 0-5 grading system — FULL backbone unfrozen.

Key differences from train.py:
  1. Loads v4 checkpoint as hot-start (Head weights already converged)
  2. Differential learning rates: backbone 1e-5, head 1e-4
  3. Fewer epochs (20) with shorter early-stopping patience
  4. Independent output directory (models/corn_v4_finetune)

Usage:
  python src/train_finetune.py          # reads config_finetune_all.yaml
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
                binary_labels = (labels >= 2).long()
                loss_bin = binary_criterion(bin_logits, binary_labels.float())
                if class_weights is not None:
                    sample_w = class_weights[labels]
                    loss_ord = criterion(ord_logits, labels, sample_w)
                else:
                    loss_ord = criterion(ord_logits, labels)
                loss = binary_loss_weight * loss_bin + loss_ord
                preds = model.ordinal_head.predict(ord_logits)
                bin_preds = (torch.sigmoid(bin_logits) >= 0.5).long()
                running_binary_loss += loss_bin.item()
                running_ordinal_loss += loss_ord.item()
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
                if class_weights is not None:
                    sample_w = class_weights[labels]
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

    # Use finetune-specific config
    config_path = "/root/dfu/config_finetune_all.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"Config: {config_path}")

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
    classify_mode = not use_corn and not binary_mode
    joint_mode = config["training"].get("joint_training", False) and not binary_mode
    binary_loss_weight = config["training"].get("binary_loss_weight", 0.5)
    num_workers = min(4, os.cpu_count() or 4)

    # Differential LR settings
    lr_head = config["training"]["learning_rate"]
    lr_backbone = config["training"]["learning_rate_backbone"]
    wd_head = config["training"]["weight_decay"]
    wd_backbone = config["training"]["weight_decay_backbone"]
    resume_ckpt = config["training"]["resume_checkpoint"]

    if joint_mode:
        mode_str = f"Joint (Binary + CORN ordinal, λ_bin={binary_loss_weight})"
    else:
        mode_str = ("Binary (benign/ulcer)" if binary_mode
                    else ("CORN ordinal" if use_corn else "Standard classification (Focal/CE)"))
    print(f"Mode: {mode_str} ({num_classes}-class)")
    print(f"Data:  {data_dir}")
    print(f"Model: {model_name}")
    print(f"  freeze_backbone = {config['model']['freeze_backbone']}  ← FINETUNE MODE")
    print(f"  LR: backbone={lr_backbone:.1e}, head={lr_head:.1e}")
    print(f"  Weight decay: backbone={wd_backbone:.1e}, head={wd_head:.1e}")
    print(f"  Hot-start from: {resume_ckpt}")

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

    # ── Model (with backbone unfrozen) ─────────────────────────────────
    print(f"\n=== Building {model_name} (full fine-tune) ===")
    binary_head = config["model"].get("binary_head", True)
    freeze_backbone = config["model"].get("freeze_backbone", False)

    if model_name == "convnext_tiny":
        model = get_convnext_tiny(
            num_classes=num_classes,
            binary=binary_head,
            classify_head=classify_mode,
            freeze_backbone=freeze_backbone,   # False → all params trainable
        )
    elif model_name == "resnet50":
        model = get_resnet50(
            num_classes=num_classes,
            binary=binary_head,
            classify_head=classify_mode,
            freeze_early=False,                 # unfreeze all for fine-tuning
        )
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_b0(
            num_classes=num_classes,
            binary=binary_head,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)

    # ── Hot-start: Load v4 checkpoint ─────────────────────────────────
    print(f"\n=== Loading v4 checkpoint: {resume_ckpt} ===")
    checkpoint = torch.load(resume_ckpt, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")
    v4_val_f1 = checkpoint.get("val_f1", "N/A")
    v4_val_acc = checkpoint.get("val_acc", "N/A")
    print(f"  v4 baseline: val_f1={v4_val_f1}, val_acc={v4_val_acc}")

    # ── Loss ──────────────────────────────────────────────────────────
    corn_class_weights = None
    binary_criterion = None

    if binary_mode:
        all_labels = torch.tensor(train_ds.all_targets)
        n_pos = all_labels.sum().item()
        n_neg = len(all_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.2f})")
    elif joint_mode:
        criterion = corn_loss
        corn_class_weights = train_ds.get_class_weights().to(device)
        all_labels = torch.tensor(train_ds.all_targets)
        binary_labels = (all_labels >= 2).long()
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

    # ── Optimizer (differential LR) ───────────────────────────────────
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"\n=== Optimizer (differential LR) ===")
    print(f"  Backbone: {n_backbone:,} params @ lr={lr_backbone:.1e}, wd={wd_backbone:.1e}")
    print(f"  Head:     {n_head:,} params @ lr={lr_head:.1e}, wd={wd_head:.1e}")

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone},
        {"params": head_params, "lr": lr_head, "weight_decay": wd_head},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["training"]["lr_t_0"], T_mult=config["training"]["lr_t_mult"],
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
        "val_loss", "val_acc", "val_f1", "val_kappa", "lr_head", "lr_backbone",
    ] + (["train_bin_acc", "train_bin_f1", "val_bin_acc", "val_bin_f1"] if joint_mode else []))

    # ── Training Loop ─────────────────────────────────────────────────
    opt_cfg = config["training"]
    best_val_f1 = 0.0
    patience_counter = 0
    patience = opt_cfg["early_stopping_patience"]
    epochs = opt_cfg["epochs"]

    print(f"\n=== Fine-tuning ({epochs} epochs, patience={patience}) ===\n")
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

        # Get current LRs from both param groups
        lr_head_now = optimizer.param_groups[1]["lr"]  # head is group 1
        lr_backbone_now = optimizer.param_groups[0]["lr"]  # backbone is group 0
        scheduler.step()

        # TensorBoard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", vl_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", vl_acc, epoch)
        writer.add_scalar("F1/train", tr_f1, epoch)
        writer.add_scalar("F1/val", vl_f1, epoch)
        writer.add_scalar("Kappa/val", vl_kappa, epoch)
        writer.add_scalar("LR/head", lr_head_now, epoch)
        writer.add_scalar("LR/backbone", lr_backbone_now, epoch)

        if joint_mode:
            csv_writer.writerow(
                [epoch, tr_loss, tr_acc, tr_f1, vl_loss, vl_acc, vl_f1, vl_kappa,
                 lr_head_now, lr_backbone_now,
                 tr_bin_acc, tr_bin_f1, vl_bin_acc, vl_bin_f1]
            )
            print(
                f"Epoch {epoch+1:3d}/{epochs} | LR_h={lr_head_now:.2e} LR_b={lr_backbone_now:.2e} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} "
                f"Bin: {tr_bin_acc:.4f}/{tr_bin_f1:.4f} | "
                f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} F1: {vl_f1:.4f} Kappa: {vl_kappa:.4f} "
                f"Bin: {vl_bin_acc:.4f}/{vl_bin_f1:.4f}"
            )
        else:
            csv_writer.writerow(
                [epoch, tr_loss, tr_acc, tr_f1, vl_loss, vl_acc, vl_f1, vl_kappa,
                 lr_head_now, lr_backbone_now]
            )
            print(
                f"Epoch {epoch+1:3d}/{epochs} | LR_h={lr_head_now:.2e} LR_b={lr_backbone_now:.2e} | "
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
                "finetune_mode": "full_unfrozen",
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
    print("Final Test Evaluation (Fine-tuned Model)")
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

    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model:  {ckpt_dir}/best_model.pth")
    print(f"   Weights:{ckpt_dir}/best_model_weights.pth")
    print(f"   CSV:    {config['logging']['csv_log']}")

    # Quick comparison with v4 baseline (from the loaded checkpoint)
    print(f"\n{'='*60}")
    print("Comparison: v4 (frozen) vs Fine-tuned (unfrozen)")
    print(f"{'='*60}")
    print(f"  v4 val_f1 (checkpoint):  {v4_val_f1}")
    print(f"  Finetune best val_f1:    {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
