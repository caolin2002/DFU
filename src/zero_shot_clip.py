#!/usr/bin/env python3
"""
Zero-shot evaluation of DFU Wagner grading using CLIP with clinical text prompts.

Tests whether text descriptions can improve grade0↔normal distinction
before committing to a full BiomedCLIP fine-tuning pipeline.

Usage:
    python src/zero_shot_clip.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip

# ─── Clinical Text Prompts ──────────────────────────────────────────────
# These are the "text descriptions" the user asked about.
# Each class gets multiple prompt templates for robustness.

PROMPT_TEMPLATES = [
    "a photo of {}",
    "a clinical photograph of {}",
    "a dermatology image showing {}",
    "a foot examination image of {}",
]

CLASS_PROMPTS = {
    "normal": [
        "a healthy foot sole",
        "normal foot skin without any pathology",
        "healthy plantar skin with normal texture",
        "a clean foot with no wounds or lesions",
    ],
    "grade0": [
        "a diabetic high-risk foot with callus",
        "a foot with callus formation and dry scaly skin",
        "a foot with fungal nail infection onychomycosis",
        "a diabetic foot with corn and callus but no open wound",
        "a foot with thickened nails and dry cracked skin",
        "a high-risk foot with deformity but no ulcer",
    ],
    "grade1": [
        "a superficial diabetic foot ulcer on skin surface",
        "a small shallow wound on the foot",
        "a Wagner grade 1 superficial ulcer limited to skin",
    ],
    "grade2": [
        "a deep diabetic foot ulcer extending below skin",
        "a Wagner grade 2 deep ulcer exposing tendon or joint",
        "a deep open wound on diabetic foot",
    ],
    "grade3": [
        "a diabetic foot with deep infection and abscess",
        "a Wagner grade 3 ulcer with osteomyelitis and pus",
        "an infected deep foot wound with bone involvement",
    ],
    "grade4": [
        "a diabetic foot with localized gangrene on toes or forefoot",
        "a Wagner grade 4 foot with black necrotic tissue limited area",
        "partial foot gangrene with dry black eschar",
    ],
    "grade5": [
        "a diabetic foot with full foot gangrene",
        "a Wagner grade 5 foot with extensive necrosis entire foot",
        "complete foot gangrene with widespread black necrotic tissue",
    ],
}

CLASS_NAMES = ["normal", "grade0", "grade1", "grade2", "grade3", "grade4", "grade5"]

# ─── Dataset (minimal, no augmentation) ─────────────────────────────────


def build_dataloader(split: str, batch_size: int = 32):
    """Build a simple dataloader for zero-shot evaluation."""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    data_dir = Path("/root/dfu/data/processed") / split
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP stats
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # Use flat image loading (no group sampling for zero-shot eval)
    dataset = ImageFolder(root=str(data_dir), transform=transform)

    # Map ImageFolder alphabetical class indices to our 0-6 schema
    # ImageFolder sorts alphabetically: grade0, grade1, grade2, grade3, grade4, grade5, normal
    # We need: normal=0, grade0=1, grade1=2, grade2=3, grade3=4, grade4=5, grade5=6
    alphabetical_order = ["grade0", "grade1", "grade2", "grade3", "grade4", "grade5", "normal"]
    remap = {alphabetical_order[i]: i for i in range(7)}  # folder_name -> imgfolder_idx
    label_map = {}
    for folder_name, imgfolder_idx in dataset.class_to_idx.items():
        label_map[imgfolder_idx] = CLASS_NAMES.index(folder_name)

    # Remap dataset targets
    original_targets = dataset.targets.copy()
    for i, t in enumerate(original_targets):
        dataset.targets[i] = label_map[t]  # noqa

    dataset.samples = [(path, label_map[dataset.class_to_idx[cls_name]])
                       for path, cls_name in zip(dataset.imgs,
                       [dataset.classes[dataset.class_to_idx[c]]
                        if hasattr(dataset, 'classes') else None
                        for c in [str(t) for t in original_targets]])]

    # Fix: rebuild samples properly
    dataset.samples = []
    for path, target in zip(dataset.imgs, dataset.targets):
        dataset.samples.append((path, target))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader, dataset


def remap_imagefolder_labels(dataset):
    """ImageFolder sorts alphabetically. Remap to our clinical ordering.

    Alphabetical: grade0, grade1, grade2, grade3, grade4, grade5, normal
    Our order:    normal=0, grade0=1, grade1=2, grade2=3, grade3=4, grade4=5, grade5=6
    """
    our_order = {"normal": 0, "grade0": 1, "grade1": 2, "grade2": 3, "grade3": 4, "grade4": 5, "grade5": 6}

    # ImageFolder internally maps class names alphabetically
    classes = sorted(dataset.classes)  # alphabetical
    imgfolder_to_ours = {}
    for imgfolder_idx, class_name in enumerate(classes):
        imgfolder_to_ours[imgfolder_idx] = our_order[class_name]

    # Remap targets
    for i in range(len(dataset.targets)):
        dataset.targets[i] = imgfolder_to_ours[dataset.targets[i]]

    # Remap samples
    new_samples = []
    for path, old_label in dataset.samples:
        new_samples.append((path, imgfolder_to_ours[old_label]))
    dataset.samples = new_samples

    # Remap class_to_idx
    new_class_to_idx = {}
    for name, idx in dataset.class_to_idx.items():
        new_class_to_idx[name] = our_order[name]
    dataset.class_to_idx = new_class_to_idx


# ─── Zero-shot Inference ────────────────────────────────────────────────


@torch.no_grad()
def zero_shot_evaluate(model, tokenizer, dataloader, device: str = "cuda"):
    """Run zero-shot classification using text prompts."""
    model.eval()

    # Build text embeddings for all classes
    all_text_embeddings = []
    n_classes = len(CLASS_NAMES)

    for class_name in CLASS_NAMES:
        prompts = CLASS_PROMPTS[class_name]
        class_embeds = []
        for prompt_text in prompts:
            for template in PROMPT_TEMPLATES:
                full_text = template.format(prompt_text)
                text_tokens = tokenizer([full_text]).to(device)
                text_embed = model.encode_text(text_tokens)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                class_embeds.append(text_embed)
        # Average across all prompts + templates for this class
        class_embeds = torch.stack(class_embeds).mean(dim=0)
        class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
        all_text_embeddings.append(class_embeds)

    text_features = torch.cat(all_text_embeddings, dim=0)  # [7, dim]

    # Run inference
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="Zero-shot inference"):
        images = images.to(device)

        # Encode images
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity → probabilities
        logits = (100.0 * image_features @ text_features.T)  # CLIP temperature ~100
        probs = logits.softmax(dim=-1)

        preds = probs.argmax(dim=-1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)

    return np.array(all_labels), np.array(all_preds), all_probs.numpy()


# ─── Metrics ────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred, y_probs, class_names):
    """Compute comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Build per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        cls = report[name]
        per_class[name] = {
            "precision": round(cls["precision"], 4),
            "recall": round(cls["recall"], 4),
            "f1": round(cls["f1-score"], 4),
            "support": int(cls["support"]),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "kappa": round(kappa, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


# ─── Main ───────────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")

    # ── Load model ──────────────────────────────────────────────────
    print("\n[1/4] Loading CLIP ViT-B/32 (openai)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    print("  Model loaded ✓")

    # ── Data ────────────────────────────────────────────────────────
    print("\n[2/4] Loading test set...")
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    data_dir = Path("/root/dfu/data/processed/test")

    # Use CLIP preprocessing
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    dataset = ImageFolder(root=str(data_dir), transform=clip_transform)
    remap_imagefolder_labels(dataset)

    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Count per class
    from collections import Counter
    label_counts = Counter(dataset.targets)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10}: {label_counts.get(i, 0)} images")
    print(f"  Total: {len(dataset)} images ✓")

    # ── Run zero-shot ───────────────────────────────────────────────
    print("\n[3/4] Running zero-shot classification...")
    print(f"  Prompts per class: {[len(CLASS_PROMPTS[c]) for c in CLASS_NAMES]}")
    print(f"  Templates: {len(PROMPT_TEMPLATES)}")
    t0 = time.time()

    y_true, y_pred, y_probs = zero_shot_evaluate(model, tokenizer, loader, device)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ✓")

    # ── Report ──────────────────────────────────────────────────────
    print("\n[4/4] Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_probs, CLASS_NAMES)

    print("\n" + "=" * 70)
    print("ZERO-SHOT CLIP RESULTS (ViT-B/32, openai weights)")
    print("=" * 70)
    print(f"\n  Accuracy:     {metrics['accuracy']:.2%}")
    print(f"  Macro F1:     {metrics['macro_f1']:.2%}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.2%}")
    print(f"  Kappa:        {metrics['kappa']:.2%}")

    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 52)
    for name in CLASS_NAMES:
        cls = metrics["per_class"][name]
        print(f"  {name:<10} {cls['precision']:>10.2%} {cls['recall']:>10.2%} {cls['f1']:>10.2%} {cls['support']:>8}")

    # Confusion matrix
    print(f"\n{'Confusion Matrix (rows=true, cols=pred)':^52}")
    print(f"{'':>10}", end="")
    for name in CLASS_NAMES:
        print(f"{name[:6]:>8}", end="")
    print()
    cm = metrics["confusion_matrix"]
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<8}", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()

    # ── Key comparison: grade0 recall ───────────────────────────────
    grade0_recall = metrics["per_class"]["grade0"]["recall"]
    normal_f1 = metrics["per_class"]["normal"]["f1"]
    print(f"\n{'─' * 60}")
    print(f"KEY METRICS (for Phase 2 decision):")
    print(f"  grade0 Recall:  {grade0_recall:.2%}  (v4: 25.45%)")
    print(f"  normal F1:      {normal_f1:.2%}     (v4: 91.96%)")
    if grade0_recall > 0.30:
        print(f"  → grade0 recall > 30% — PROCEED to Phase 2 (BiomedCLIP fine-tune)")
    else:
        print(f"  → grade0 recall ≤ 30% — text prompts alone insufficient")
    print(f"{'─' * 60}")

    # ── Save report ─────────────────────────────────────────────────
    report = {
        "model": "ViT-B-32 (openai CLIP)",
        "method": "zero_shot",
        "timestamp:": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
        "prompts_per_class": {c: CLASS_PROMPTS[c] for c in CLASS_NAMES},
        "templates": PROMPT_TEMPLATES,
        "metrics": metrics,
        "inference_time_s": round(elapsed, 1),
    }

    out_path = Path("/root/dfu/models/corn_v4/zero_shot_clip_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
