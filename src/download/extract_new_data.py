#!/usr/bin/env python3
"""
Extract useful data from /root/dfu/image, image1, image2 into /root/dfu/data/raw/

Sources and their value:
  image/TestSet          — ~20 Grade 0 (corn/callus/pre-ulcer) images
  image/TL/internetSet   — ~38 Grade 0 images
  image/TL/Wound Images2 — 677 wound images, some Wagner-labeled
  image/TL/Wound Images  — 109 wound images
  image1/train+val/DM Group    — 976 diabetic foot (Grade 0!) NOT in existing data
  image1/train+val/Control     — 890 normal foot (additional)
  image2/Labeled               — 110 DFU with callus/fibrin/granulation masks
  image2/Unlabeled             — 600 unlabeled wound images

Strategy:
  - Deduplicate by MD5 hash across all existing data + new sources
  - Skip images already present in /root/dfu/data/raw/
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

RAW = Path("/root/dfu/data/raw")
IMAGE = Path("/root/dfu/image")
IMAGE1 = Path("/root/dfu/image1")
IMAGE2 = Path("/root/dfu/image2")

# ─── Collect existing MD5 hashes ────────────────────────────────────
def collect_existing_hashes(base_dir: Path) -> dict[str, Path]:
    """Return {md5hex: path} for all image files under base_dir."""
    hashes: dict[str, Path] = {}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}
    for f in base_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in exts:
            try:
                h = hashlib.md5(f.read_bytes()).hexdigest()
                hashes[h] = f
            except Exception:
                pass
    return hashes

print("🔍 Collecting existing MD5 hashes from /root/dfu/data/raw/ ...")
existing = collect_existing_hashes(RAW)
print(f"   Found {len(existing)} existing images")

# ─── Copy with dedup ────────────────────────────────────────────────
def copy_dedup(src_files: list[Path], dest_dir: Path, seen: dict[str, Path]) -> int:
    """Copy files to dest_dir, skipping duplicates. Returns count copied."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for src in src_files:
        if not src.is_file():
            continue
        try:
            h = hashlib.md5(src.read_bytes()).hexdigest()
        except Exception:
            continue
        if h in seen:
            skipped += 1
            continue
        seen[h] = src
        # Keep original name; resolve conflicts
        dest = dest_dir / src.name
        if dest.exists():
            # Add hash suffix to avoid name collision
            stem, ext = os.path.splitext(src.name)
            dest = dest_dir / f"{stem}_{h[:8]}{ext}"
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception as e:
            print(f"   ✗ {src.name}: {e}")
    if skipped:
        print(f"   ⏭  Skipped {skipped} duplicates")
    return copied


# ─── Grade 0 keywords for filtering ────────────────────────────────
GRADE0_KEYWORDS = [
    'corn', 'callus', 'callo', 'pre-ulcer', 'preulcer', 'pre_ulcer',
    'nail', 'claw', 'hammer', 'bunion', 'charcot', 'deform',
    'bleeding', 'risk', 'dry skin', 'fissure', 'crack',
    'grade-0', 'grade0', 'grade 0', 'class-1', 'class1',
]

def is_grade0(filename: str) -> bool:
    lower = filename.lower()
    return any(kw in lower for kw in GRADE0_KEYWORDS)


# ─── Extraction plans ───────────────────────────────────────────────

PLANS = [
    # ── Plan 1: DM Group (diabetic foot, Grade 0) from image1 ──
    {
        "name": "DM Foot Grade 0 (image1 DM Group)",
        "dest": "dm_foot_grade0",
        "files": list(IMAGE1.rglob('DM Group/*')),
        "desc": "Diabetic foot without ulcer — perfect Grade 0 data",
    },
    # ── Plan 2: Control Group (normal feet) from image1 ──
    {
        "name": "Normal Foot Control (image1 Control Group)",
        "dest": "normal_foot_control",
        "files": list(IMAGE1.rglob('Control Group/*')),
        "desc": "Normal foot control group for binary classification",
    },
    # ── Plan 3: Grade 0 from image/TestSet ──
    {
        "name": "Grade 0 from TestSet",
        "dest": "dermnet_grade0",  # merge into existing
        "files": [f for f in IMAGE.glob('TestSet/*') if f.is_file() and is_grade0(f.name)],
        "desc": "Corn/callus/pre-ulcer images found in TestSet",
    },
    # ── Plan 4: Grade 0 from image/TL/internetSet ──
    {
        "name": "Grade 0 from internetSet",
        "dest": "dermnet_grade0",  # merge into existing
        "files": [f for f in (IMAGE / 'Transfer-Learning images' / 'internetSet').glob('*')
                  if f.is_file() and is_grade0(f.name)],
        "desc": "Corn/callus/pre-ulcer images from internet crawl",
    },
    # ── Plan 5: Non-Grade0 from TestSet (other wound images) ──
    {
        "name": "TestSet other images",
        "dest": "internet_wound",
        "files": [f for f in IMAGE.glob('TestSet/*') if f.is_file() and not is_grade0(f.name)],
        "desc": "Remaining TestSet images (mostly wounds)",
    },
    # ── Plan 6: Non-Grade0 from internetSet ──
    {
        "name": "internetSet other images",
        "dest": "internet_wound",
        "files": [f for f in (IMAGE / 'Transfer-Learning images' / 'internetSet').glob('*')
                  if f.is_file() and not is_grade0(f.name)],
        "desc": "Remaining internetSet images",
    },
    # ── Plan 7: Wound Images2 (677 wound images, some Wagner-labeled) ──
    {
        "name": "Wound Images2",
        "dest": "internet_wound",
        "files": list((IMAGE / 'Transfer-Learning images' / 'Wound Images2').glob('*')),
        "desc": "Wound images from internet, some Wagner-labeled",
    },
    # ── Plan 8: Wound Images (109) ──
    {
        "name": "Wound Images",
        "dest": "internet_wound",
        "files": list((IMAGE / 'Transfer-Learning images' / 'Wound Images').glob('*')),
        "desc": "Additional wound images from internet",
    },
    # ── Plan 9: image2 Labeled Original Images ──
    {
        "name": "Wound Tissue Seg Images",
        "dest": "wound_seg_callus/images",
        "files": list((IMAGE2 / 'Labeled' / 'Original' / 'Images').rglob('*')),
        "desc": "DFU images with callus/fibrin/granulation masks",
    },
    # ── Plan 10: image2 Labeled Original Annotations ──
    {
        "name": "Wound Tissue Seg Annotations",
        "dest": "wound_seg_callus/annotations",
        "files": list((IMAGE2 / 'Labeled' / 'Original' / 'Annotations').rglob('*')),
        "desc": "Pixel-level masks (Red=Fibrin, Green=Granulation, Blue=Callus)",
    },
    # ── Plan 11: image2 Labeled Padded Images ──
    {
        "name": "Wound Tissue Seg Padded Images",
        "dest": "wound_seg_callus/padded/images",
        "files": list((IMAGE2 / 'Labeled' / 'Padded' / 'Images').rglob('*')),
        "desc": "Padded versions for segmentation training",
    },
    # ── Plan 12: image2 Labeled Padded Annotations ──
    {
        "name": "Wound Tissue Seg Padded Annotations",
        "dest": "wound_seg_callus/padded/annotations",
        "files": list((IMAGE2 / 'Labeled' / 'Padded' / 'Annotations').rglob('*')),
        "desc": "Padded mask versions",
    },
    # ── Plan 13: image2 Unlabeled ──
    {
        "name": "Wound Unlabeled",
        "dest": "wound_unlabeled",
        "files": list((IMAGE2 / 'Unlabeled').glob('*')),
        "desc": "600 unlabeled wound images for semi-supervised learning",
    },
]

# ─── Execute ─────────────────────────────────────────────────────────
print()
total_copied = 0
total_skipped = 0

for plan in PLANS:
    files = [f for f in plan["files"] if f.is_file()]
    if not files:
        print(f"⏭  {plan['name']}: no files found")
        continue
    print(f"📋 {plan['name']} ({len(files)} files)")
    print(f"   → {RAW / plan['dest']}")
    print(f"   {plan['desc']}")
    copied = copy_dedup(files, RAW / plan["dest"], existing)
    print(f"   ✓ Copied {copied}/{len(files)} files")
    total_copied += copied

# ─── Also save the palette and metadata for image2 ──────────────────
for meta_file in IMAGE2.rglob('palette_colorCode.txt'):
    dest = RAW / 'wound_seg_callus' / meta_file.name
    shutil.copy2(meta_file, dest)
    print(f"📄 Saved palette: {dest}")

for names_file in IMAGE2.glob('Labeled/*_names.txt'):
    dest = RAW / 'wound_seg_callus' / names_file.name
    shutil.copy2(names_file, dest)
    print(f"📄 Saved split: {dest}")

for citation in IMAGE2.glob('Citation.txt'):
    dest = RAW / 'wound_seg_callus' / citation.name
    shutil.copy2(citation, dest)
    print(f"📄 Saved citation: {dest}")

# ─── Summary ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"✅ Extraction complete: {total_copied} new images added")
print(f"   Total existing images: {len(existing)}")
print()
print("📊 New data inventory:")
for plan in PLANS:
    dest_dir = RAW / plan["dest"]
    count = sum(1 for _ in dest_dir.rglob('*')) if dest_dir.exists() else 0
    if count:
        print(f"   {plan['dest']}: {count} files — {plan['desc']}")
