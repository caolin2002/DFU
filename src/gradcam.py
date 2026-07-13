#!/usr/bin/env python3
"""
Grad-CAM visualization for DFU ConvNeXt-Tiny model.

Generates heatmaps showing which image regions the model focuses on
for each Wagner grade prediction. Essential for clinical trust —
verifies the model looks at wounds, not background artifacts.

Uses torch.autograd.grad() instead of backward hooks for robust gradient capture.

Usage:
    python src/gradcam.py                                    # all classes
    python src/gradcam.py --class grade4 --n_samples 10      # specific class
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import DFUDataset, IDX_TO_LABEL, LABEL_NAMES, NUM_CLASSES
from model import get_convnext_tiny

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GradCAM:
    """
    Grad-CAM for ConvNeXt-Tiny backbone using torch.autograd.grad.

    Captures feature maps from the last spatial layer via forward hook,
    then computes gradients of the target logit w.r.t. those feature maps.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        # Hook into the last spatial layer
        self.target_layer = model.backbone.features
        self.handle = self.target_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.activations = output  # Keep on graph, don't detach!

    def __call__(self, x: torch.Tensor, target_class: int | None = None):
        """
        Args:
            x: [1, 3, H, W] normalized image tensor
            target_class: class to explain (None = use predicted class)
        Returns:
            heatmap: [H, W] numpy (0-1), pred_class, confidence
        """
        # Ensure gradients flow — backbone is frozen but we need gradients
        # of the target score w.r.t. spatial feature maps.
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Forward: manually run backbone components to intercept activations
        activations = self.model.backbone.features(x)         # [1, C, H', W']
        pooled = self.model.backbone.avgpool(activations)      # [1, C, 1, 1]
        flat = self.model.backbone.classifier(pooled)          # [1, C]
        ord_logits = self.model.ordinal_head(flat)             # [1, K-1]

        pred_class = self.model.ordinal_head.predict(ord_logits).item()
        if target_class is None:
            target_class = pred_class

        # CORN: class k depends on logits[k-1] (the ">=k" threshold)
        # class 0 → -logits[0]; class k>0 → logits[k-1]
        if target_class == 0:
            score = -ord_logits[:, 0]
        else:
            score = ord_logits[:, target_class - 1]

        # Gradient of score w.r.t. activations
        grads = torch.autograd.grad(
            outputs=score,
            inputs=activations,
            retain_graph=False,
            create_graph=False,
        )[0]  # [1, C, H', W']

        # Pool gradients spatially → weights
        weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination + ReLU
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cmin, cmax = cam.min(), cam.max()
        if cmax > cmin:
            cam = (cam - cmin) / (cmax - cmin)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode='bilinear', align_corners=False)

        heatmap = cam.squeeze().detach().cpu().numpy()

        # Confidence
        probs = self.model.ordinal_head.predict_proba(ord_logits).squeeze(0)
        confidence = probs[target_class].item()

        return heatmap, target_class, confidence

    def remove_hooks(self):
        self.handle.remove()


# ─── Image helpers ──────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized tensor → uint8 numpy [H, W, 3]."""
    img = tensor.clone()
    for t, m, s in zip(img, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """Blend jet-colored heatmap onto original image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    overlaid = ((1 - alpha) * image_np.astype(float) +
                alpha * heatmap_colored.astype(float))
    return overlaid.clip(0, 255).astype(np.uint8)


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for DFU model")
    parser.add_argument("--checkpoint", type=str,
                        default="/root/dfu/models/corn_v2/best_model.pth")
    parser.add_argument("--config", type=str,
                        default="/root/dfu/config.yaml")
    parser.add_argument("--class", dest="target_class", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str,
                        default="/root/dfu/models/corn_v2/gradcam")
    parser.add_argument("--correct_only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", config)
    num_classes = model_cfg["model"]["num_classes"]
    binary_head = model_cfg["model"].get("binary_head", True)

    model = get_convnext_tiny(num_classes=num_classes, binary=binary_head)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Data
    data_dir = config["data"]["data_dir"]
    input_size = config["data"]["input_size"]
    num_workers = min(2, os.cpu_count() or 2)

    test_ds = DFUDataset(data_dir, "test", input_size, binary=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True,
                             num_workers=num_workers, pin_memory=True)

    # Target classes
    if args.target_class:
        target_classes = [int(args.target_class)] if args.target_class.isdigit() else \
                         [k for k, v in IDX_TO_LABEL.items() if v == args.target_class]
    else:
        target_classes = [i for i in range(NUM_CLASSES - 1)]  # Skip grade5

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAM(model)

    for target_cls in target_classes:
        cls_name = IDX_TO_LABEL[target_cls]
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {cls_name} ({LABEL_NAMES[cls_name]}) ===")
        collected = 0
        seen_in_class = 0
        max_look = len(test_loader)

        pbar = tqdm(total=args.n_samples, desc=f"  {cls_name}")
        for images, labels in test_loader:
            if labels[0].item() != target_cls:
                continue
            seen_in_class += 1
            if seen_in_class > max_look:
                break

            image = images[0]
            label = labels[0].item()

            try:
                heatmap, pred_cls, conf = gradcam(image.unsqueeze(0).to(device))

                if args.correct_only and pred_cls != label:
                    continue

                img_np = denormalize(image.cpu())
                overlaid = overlay_heatmap(img_np, heatmap)

                status = "correct" if pred_cls == label else "wrong"
                fname = (f"{cls_name}_{status}_conf{conf:.2f}_"
                         f"pred{IDX_TO_LABEL[pred_cls]}_{collected+1:02d}.jpg")
                Image.fromarray(overlaid).save(cls_dir / fname)

                collected += 1
                pbar.update(1)
                if collected >= args.n_samples:
                    break
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
        pbar.close()
        print(f"  ✅ Saved {collected} samples to {cls_dir}")

    gradcam.remove_hooks()
    print(f"\n✅ Done! Output: {output_dir}")


if __name__ == "__main__":
    main()
