#!/usr/bin/env python3
"""
Model definitions for DFU Wagner 0-5 grading system.

Architecture — Two-stage design:
  Stage 1 (binary):    benign (normal + grade0) vs ulcer (wound + gangrene)
  Stage 2 (ordinal):   4-class CORN — normal < grade0 < wound < gangrene

Supported backbones:
  - ResNet-50 (standard or CORN ordinal head)
  - ConvNeXt-Tiny + CORN (recommended — modern, efficient)

Key components:
  - CORNHead: Conditional Ordinal Regression for Neural networks
    Enforces monotonic thresholds: P(normal) >= P(grade0) >= P(wound) >= P(gangrene)
  - BinaryHead: Standard binary classifier for ulcer screening
  - DFUModel: Combined backbone with both heads for two-stage inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ===========================================================================
# CORN Ordinal Regression Head
# ===========================================================================

class CORNHead(nn.Module):
    """
    CORN (Conditional Ordinal Regression for Neural networks) head.

    For K classes, learns K-1 binary classifiers with monotonic biases:
      "Is severity >= grade0?" → ">= grade1?" → ">= grade2?" → ... → ">= grade5?"

    Bias constraint: bias[0] <= bias[1] <= ... <= bias[K-2] via softplus chaining.
    This enforces: P(>=grade0) >= P(>=grade1) >= ... >= P(>=grade5)
    """

    def __init__(self, in_features: int, num_classes: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # 7 classes → 6 binary tasks

        # Shared weight matrix: [in_features] → [num_tasks] logits
        self.linear = nn.Linear(in_features, self.num_tasks)

        # Non-decreasing bias chain:
        #   bias[0] = base_bias
        #   bias[k] = bias[k-1] + softplus(delta[k-1])
        self.base_bias = nn.Parameter(torch.zeros(1))
        self.bias_deltas = nn.Parameter(torch.zeros(self.num_tasks - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # [B, K-1]

        # Build non-decreasing biases
        biases = [self.base_bias]
        for delta in self.bias_deltas:
            biases.append(biases[-1] + F.softplus(delta))
        bias = torch.cat(biases)  # [K-1]

        return logits + bias

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Sum of passed thresholds = predicted class index."""
        probs = torch.sigmoid(logits)
        preds = probs.round().sum(dim=1).long()
        return preds.clamp(0, self.num_classes - 1)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert threshold logits to per-class probabilities.
        P(class 0) = 1 - P(>=1)
        P(class 1) = P(>=1) - P(>=2)
        P(class 2) = P(>=2) - P(>=3)
        P(class 3) = P(>=3)
        """
        probs = torch.sigmoid(logits)  # [B, K-1]
        class_probs = []
        for i in range(self.num_classes):
            if i == 0:
                class_probs.append(1 - probs[:, 0])
            elif i == self.num_classes - 1:
                class_probs.append(probs[:, -1])
            else:
                class_probs.append(probs[:, i - 1] - probs[:, i])
        return torch.stack(class_probs, dim=1).clamp(0, 1)


# ===========================================================================
# Binary Classification Head
# ===========================================================================

class BinaryHead(nn.Module):
    """Standard binary classifier for ulcer screening."""

    def __init__(self, in_features: int, dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # [B] logits


# ===========================================================================
# Loss Functions
# ===========================================================================

def corn_loss(logits: torch.Tensor, labels: torch.Tensor,
              sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    CORN binary cross-entropy loss with optional sample weights.

    Targets for 7 classes (K-1 = 6 binary tasks):
        normal (0)  → [0, 0, 0, 0, 0, 0]
        grade0 (1)  → [1, 0, 0, 0, 0, 0]
        grade1 (2)  → [1, 1, 0, 0, 0, 0]
        grade2 (3)  → [1, 1, 1, 0, 0, 0]
        grade3 (4)  → [1, 1, 1, 1, 0, 0]
        grade4 (5)  → [1, 1, 1, 1, 1, 0]
        grade5 (6)  → [1, 1, 1, 1, 1, 1]

    Args:
        logits: [B, K-1] threshold logits
        labels: [B] integer class labels (0..K-1)
        sample_weights: [B] per-sample weight (e.g., inverse-frequency class weight)
    """
    num_tasks = logits.size(1)
    targets = (labels.unsqueeze(1) > torch.arange(num_tasks, device=labels.device)).float()

    if sample_weights is not None:
        # Per-sample BCE with weights
        loss_per_task = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # [B, K-1]
        loss_per_sample = loss_per_task.mean(dim=1)  # [B] — average over tasks
        return (loss_per_sample * sample_weights).mean()
    else:
        return F.binary_cross_entropy_with_logits(logits, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples, focuses on hard ones (e.g., gangrene).
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # class weights [C]
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        if target.dim() == 1:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (n_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        else:
            true_dist = target
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# ===========================================================================
# Combined DFU Model
# ===========================================================================

class DFUModel(nn.Module):
    """
    Two-stage DFU grading model.

    Shared backbone → BinaryHead (benign/ulcer) + CORNHead (Wagner 0-5 ordinal, 7-class).

    Supports:
      - Single-task training: binary only or ordinal only
      - Joint training: binary + ordinal losses combined
      - Two-stage inference: screen first, then grade
    """

    def __init__(self, backbone: nn.Module, in_features: int,
                 num_classes: int = 7, binary: bool = True,
                 classify_head: bool = False, dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.num_classes = num_classes
        self.has_binary = binary
        self.has_classify = classify_head

        if binary:
            self.binary_head = BinaryHead(in_features, dropout)
        self.ordinal_head = CORNHead(in_features, num_classes)
        if classify_head:
            self.classify_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x: torch.Tensor, task: str = "ordinal"):
        """
        Args:
            x: image batch [B, 3, H, W]
            task: "binary" | "ordinal" | "classify" | "both"
        Returns:
            "binary"   → (binary_logits,)
            "ordinal"  → (ordinal_logits,)
            "classify" → (class_logits,)  [B, num_classes]
            "both"     → (binary_logits, ordinal_logits)
        """
        features = self.backbone(x)  # [B, in_features]

        if task == "binary":
            if not self.has_binary:
                raise ValueError("Binary head not enabled")
            return self.binary_head(features)
        elif task == "ordinal":
            return self.ordinal_head(features)
        elif task == "classify":
            if not self.has_classify:
                raise ValueError("Classify head not enabled")
            return self.classify_head(features)
        elif task == "both":
            bin_out = self.binary_head(features) if self.has_binary else None
            ord_out = self.ordinal_head(features)
            return bin_out, ord_out
        else:
            raise ValueError(f"Unknown task: {task}")

    def predict_staged(self, x: torch.Tensor,
                       binary_thresh: float = 0.5) -> torch.Tensor:
        """
        Two-stage inference:
          1. Binary screening: if benign → return class 0 or 1
          2. If ulcer → ordinal grading to get Wagner level
        """
        features = self.backbone(x)

        if self.has_binary:
            bin_logit = self.binary_head(features)
            bin_prob = torch.sigmoid(bin_logit)
            is_ulcer = bin_prob >= binary_thresh
        else:
            is_ulcer = torch.ones(x.size(0), device=x.device, dtype=torch.bool)

        ord_logits = self.ordinal_head(features)
        wagner_preds = self.ordinal_head.predict(ord_logits)  # 0-3

        # For benign cases, the ordinal prediction is already correct (0=normal, 1=grade0)
        # For ulcer cases, map ordinal 0-3 to Wagner grade
        return wagner_preds


# ===========================================================================
# Model Constructors
# ===========================================================================

def get_resnet50(num_classes: int = 7, pretrained: bool = True,
                 binary: bool = True, classify_head: bool = False,
                 freeze_early: bool = True):
    """
    ResNet-50 backbone with CORN ordinal + optional binary/classify heads.

    Args:
        num_classes: number of ordinal categories (7)
        pretrained: use ImageNet-1K weights
        binary: include binary screening head
        classify_head: include standard classification head (for CE/Focal loss)
        freeze_early: freeze conv1, bn1, layer1, layer2
    """
    model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)

    if freeze_early:
        for name, param in model.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                param.requires_grad = False

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Flatten(start_dim=1)  # Replace classification head

    dfu_model = DFUModel(model, in_features, num_classes, binary=binary,
                         classify_head=classify_head)

    trainable = sum(p.numel() for p in dfu_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in dfu_model.parameters())
    mode = "standard" if classify_head else "CORN"
    print(f"  ResNet-50 DFU ({mode}) | Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return dfu_model


def get_convnext_tiny(num_classes: int = 7, pretrained: bool = True,
                      binary: bool = True, classify_head: bool = False,
                      freeze_backbone: bool = True):
    """
    ConvNeXt-Tiny backbone with CORN ordinal + optional binary/classify heads.

    Recommended for DFU grading — modern architecture, efficient, good with medical images.

    Args:
        num_classes: number of ordinal categories (7)
        pretrained: use ImageNet-1K weights
        binary: include binary screening head
        classify_head: include standard classification head (for CE/Focal loss)
        freeze_backbone: freeze all backbone, train only heads
    """
    model = models.convnext_tiny(
        weights="IMAGENET1K_V1" if pretrained else None
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ConvNeXt feature dim
    in_features = model.classifier[2].in_features  # 768

    # Replace classifier with flatten
    model.classifier = nn.Flatten(start_dim=1)

    dfu_model = DFUModel(model, in_features, num_classes, binary=binary,
                         classify_head=classify_head)

    trainable = sum(p.numel() for p in dfu_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in dfu_model.parameters())
    mode = "standard" if classify_head else "CORN"
    print(f"  ConvNeXt-Tiny DFU ({mode}) | Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return dfu_model


def get_efficientnet_b0(num_classes: int = 7, pretrained: bool = True,
                        binary: bool = True, freeze_backbone: bool = True):
    """
    EfficientNet-B0 backbone — lightweight alternative for quick experiments.
    """
    model = models.efficientnet_b0(
        weights="IMAGENET1K_V1" if pretrained else None
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Flatten(start_dim=1)

    dfu_model = DFUModel(model, in_features, num_classes, binary=binary)

    trainable = sum(p.numel() for p in dfu_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in dfu_model.parameters())
    print(f"  EfficientNet-B0 DFU | Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return dfu_model
