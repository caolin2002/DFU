#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) utilities for DFU grading.

For each image, generates multiple augmented views and averages predictions
to reduce variance and improve robustness. No additional training needed.

Usage:
    from tta import tta_predict, TTA_TRANSFORMS

    preds, probs = tta_predict(model, image_tensor, device, n_views=4)
"""

import torch
import torch.nn.functional as F
from torchvision import transforms

# ─── TTA View Transforms ────────────────────────────────────────────────
# Each transform takes a [3, H, W] normalized tensor and returns the same.
# Applied AFTER ImageNet normalization (re-normalize after spatial ops when needed).

def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _rotate_plus5(x: torch.Tensor) -> torch.Tensor:
    """Rotate +5 degrees with reflection padding."""
    return transforms.functional.rotate(x, angle=5.0, fill=0)


def _rotate_minus5(x: torch.Tensor) -> torch.Tensor:
    """Rotate -5 degrees."""
    return transforms.functional.rotate(x, angle=-5.0, fill=0)


def _brightness_up(x: torch.Tensor) -> torch.Tensor:
    """Increase brightness by factor 1.1, clamp to valid range."""
    return torch.clamp(x * 1.1, 0.0, 1.0)


def _brightness_down(x: torch.Tensor) -> torch.Tensor:
    """Decrease brightness by factor 0.9."""
    return torch.clamp(x * 0.9, 0.0, 1.0)


# Available TTA views (all assume input is already ToTensor + Normalize'd)
TTA_TRANSFORMS = {
    "identity": _identity,
    "hflip": _hflip,
    "rotate+5": _rotate_plus5,
    "rotate-5": _rotate_minus5,
    "brightness+": _brightness_up,
    "brightness-": _brightness_down,
}


def get_tta_views(tensor: torch.Tensor, n_views: int = 4) -> list[torch.Tensor]:
    """
    Generate `n_views` augmented versions of a single image tensor.

    Args:
        tensor: [3, H, W] normalized image tensor
        n_views: number of augmented views (default 4, max 6)
    Returns:
        list of [1, 3, H, W] tensors ready for model input
    """
    views = [tensor.unsqueeze(0)]  # Always include original
    transforms_list = list(TTA_TRANSFORMS.values())[1:]  # Skip identity

    for i in range(min(n_views - 1, len(transforms_list))):
        t = transforms_list[i]
        view = t(tensor.clone())
        views.append(view.unsqueeze(0))

    return views


def tta_predict_ordinal(model, image_tensor: torch.Tensor, device: torch.device,
                        n_views: int = 4) -> tuple[int, torch.Tensor]:
    """
    TTA prediction for CORN ordinal model.

    Averages CORN logits across TTA views, then predicts class.

    Args:
        model: DFUModel (with ordinal_head)
        image_tensor: [3, H, W] normalized tensor
        device: torch device
        n_views: number of augmented views
    Returns:
        (predicted_class: int, class_probabilities: [num_classes])
    """
    views = get_tta_views(image_tensor, n_views)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for view in views:
            view = view.to(device)
            logits = model(view, task="ordinal")  # [1, K-1]
            all_logits.append(logits)

        # Average logits across views
        avg_logits = torch.stack(all_logits).mean(dim=0)  # [1, K-1]

        # Predict
        pred = model.ordinal_head.predict(avg_logits).item()
        probs = model.ordinal_head.predict_proba(avg_logits).squeeze(0)  # [K]

    return pred, probs


def tta_predict_classify(model, image_tensor: torch.Tensor, device: torch.device,
                         n_views: int = 4) -> tuple[int, torch.Tensor]:
    """
    TTA prediction for standard classification model.
    Averages class logits across views.
    """
    views = get_tta_views(image_tensor, n_views)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for view in views:
            view = view.to(device)
            logits = model(view, task="classify")  # [1, num_classes]
            all_logits.append(logits)

        avg_logits = torch.stack(all_logits).mean(dim=0)
        probs = F.softmax(avg_logits, dim=1).squeeze(0)
        pred = avg_logits.argmax(dim=1).item()

    return pred, probs


def tta_predict_binary(model, image_tensor: torch.Tensor, device: torch.device,
                       n_views: int = 4) -> tuple[int, float]:
    """
    TTA prediction for binary model.
    Averages logit across views.
    """
    views = get_tta_views(image_tensor, n_views)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for view in views:
            view = view.to(device)
            logit = model(view, task="binary")  # [1]
            all_logits.append(logit)

        avg_logit = torch.stack(all_logits).mean()
        prob = torch.sigmoid(avg_logit).item()
        pred = 1 if prob >= 0.5 else 0

    return pred, prob
