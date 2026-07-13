#!/usr/bin/env python3
"""
R2: Auto-Labeling Pipeline for DFU Wagner 0-5 Grading System
=============================================================

Three phases:
  A1 — Build manifest: scan all 28K+ images, assign labels by source + heuristics
  A2 — Quality check: label distribution, imbalance analysis, outlier detection
  A3 — Stratified split: 70/15/15 train/val/test preserving class ratios

Label Schema (7 classes — Wagner 0-5 full granularity):
  normal   = Healthy foot/skin — no DFU pathology
  grade0   = Wagner 0 — high-risk foot, callus, deformity, NO ulcer
  grade1   = Wagner 1-3 — ulcer present (sub-grade uncertain; starting point)
  grade2   = Wagner 1-3 — (reserved, empty for now)
  grade3   = Wagner 1-3 — (reserved, empty for now)
  grade4   = Wagner 4-5 — gangrene/necrosis (starting point)
  grade5   = Wagner 4-5 — (reserved, empty for now)
  non_dfu  = Other foot conditions (eczema, psoriasis, etc.) — exclude from training

Strategy:
  Only high-confidence labels from dataset metadata are assigned.
  Current wound images → grade1 (all Wagner 1-3 pooled here initially).
  Current gangrene images → grade4 (all Wagner 4-5 pooled here initially).
  grade2, grade3, grade5 are empty placeholders for future model-predicted refinement.
  In R4/R5 a two-stage model handles this: binary first, then ordinal grading.

Output:
  data/manifest.csv        — full inventory (path, label, confidence, source)
  data/manifest_report.txt — label distribution & warnings
  data/processed/           — organized train/val/test/{label}/
"""

import csv
import hashlib
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# ─── Config ──────────────────────────────────────────────────────────
RAW = Path("/root/dfu/data/raw")
MANIFEST = Path("/root/dfu/data/manifest.csv")
REPORT = Path("/root/dfu/data/manifest_report.txt")
PROCESSED = Path("/root/dfu/data/processed")
RANDOM_SEED = 42

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}
MASK_EXTS = {'.png'}  # masks are always PNG

# ─── Label definitions ────────────────────────────────────────────────
# Each rule: (label, confidence, condition_fn)
# Processed in order — first match wins

def build_rules() -> list[dict]:
    """Return label assignment rules in priority order."""
    rules = []

    # ═══ EXCLUDE: masks, annotations, pre-augmented variants, non-image data ═══
    def _exclude_masks(rel_path: str, fname: str) -> str | None:
        if 'wound_mask' in rel_path or 'train_masks' in rel_path or 'test_masks' in rel_path:
            return 'mask'
        # wound_seg_callus annotations (any path containing 'annotations') — masks, not photos
        if 'annotations' in rel_path:
            return 'mask'
        # wound_seg_callus padded — duplicates of original images
        if 'padded' in rel_path.lower():
            return 'padded_duplicate'
        if 'palette' in rel_path.lower() or fname.endswith('.txt'):
            return 'metadata'
        # Pre-augmented variants from dm_foot_grade0 / normal_foot_control:
        #   DM001_M_L.png → original (keep)
        #   DM001_M_L-rotated1.png → discard (R3 applies our own RandAugment)
        #   DM001_M_L-rotated2.png → discard
        #   DM001_M_L-sharpened.png → discard
        stem = Path(fname).stem
        for suffix in ('-rotated1', '-rotated2', '-sharpened'):
            if stem.endswith(suffix):
                return 'pre_augmented'
        return None

    rules.append({'label': 'exclude', 'confidence': 'high', 'match_fn': _exclude_masks})

    # ═══ EXCLUDE: Heel X-ray — separate modality, not RGB DFU photos ═══
    def _exclude_heel_xray(rel_path: str, fname: str) -> str | None:
        if 'heel_dataset' in rel_path:
            return 'xray_modality'
        return None
    rules.append({'label': 'exclude', 'confidence': 'high', 'match_fn': _exclude_heel_xray})

    # ═══ GRADE 4: Wagner 4-5 gangrene/necrosis ═══
    def _match_grade4(rel_path: str, fname: str) -> str | None:
        if 'gangrene' in fname.lower() or 'gangarene' in fname.lower():
            return 'gangrene_keyword'
        return None
    rules.append({'label': 'grade4', 'confidence': 'high', 'match_fn': _match_grade4})

    # ═══ GRADE 0: Known high-risk diabetic foot ═══
    def _match_grade0(rel_path: str, fname: str) -> str | None:
        # dm_foot_grade0 — diabetes mellitus foot without ulcer
        if 'dm_foot_grade0' in rel_path:
            return 'dm_foot'
        # dermnet corn/callus/nail — Grade 0 signs
        if 'dermnet_grade0/corn-callus' in rel_path:
            return 'corn_callus'
        if any(kw in rel_path for kw in ['dermnet_grade0/nail', 'dermnet_grade0/onych',
                                           'dermnet_grade0/white-nail', 'dermnet_grade0/fungal-nail']):
            return 'nail_pathology'
        if 'dermnet_grade0/dry-skin' in rel_path:
            return 'dry_skin'
        # Any filename mentioning corn/callus/pre-ulcer
        if any(kw in fname.lower() for kw in ['corn', 'callus', 'callo', 'pre-ulcer', 'preulcer']):
            return 'filename_hint'
        return None
    rules.append({'label': 'grade0', 'confidence': 'high', 'match_fn': _match_grade0})

    # ═══ NORMAL: Known healthy foot/skin ═══
    def _match_normal(rel_path: str, fname: str) -> str | None:
        if 'normal_foot_control' in rel_path:
            return 'control_group'
        if 'Nomal' in rel_path:  # Mendeley typo
            return 'mendeley_normal'
        if 'Normal(Healthy' in rel_path or 'Normal (Healthy' in rel_path:
            return 'kaggle_normal_patch'
        if 'heel_dataset' in rel_path and '/normal/' in rel_path:
            return 'heel_xray_normal'
        return None
    rules.append({'label': 'normal', 'confidence': 'high', 'match_fn': _match_normal})

    # ═══ GRADE 1: Confirmed ulcer/wound (Wagner 1-3 pooled) ═══
    def _match_grade1(rel_path: str, fname: str) -> str | None:
        if 'wound_main' in rel_path:
            return 'mendeley_wound'
        if 'Abnormal(Ulcer)' in rel_path or 'Abnormal (Ulcer)' in rel_path:
            return 'kaggle_abnormal_patch'
        if 'hf_wound_classification' in rel_path and '/wound/' in rel_path:
            return 'hf_wound_cls'
        if 'wound_segmentation' in rel_path and 'train_images' in rel_path:
            return 'wound_seg_train'
        if 'wound_segmentation' in rel_path and 'test_images' in rel_path:
            return 'wound_seg_test'
        if 'wound_seg_callus/images' in rel_path:
            return 'wound_seg_callus'
        if 'dermnet_grade0/diabetic-foot-ulcer' in rel_path:
            return 'dermnet_dfu'
        if 'Wound Images' in rel_path or 'Wound Images2' in rel_path:
            return 'kaggle_tl_wound'
        if 'wikimedia_foot' in rel_path:
            return 'wikimedia'
        return None
    rules.append({'label': 'grade1', 'confidence': 'high', 'match_fn': _match_grade1})

    # ═══ NON-DFU: Other skin conditions from DermNet ═══
    def _match_non_dfu(rel_path: str, fname: str) -> str | None:
        dermnet_non_dfu = [
            'psoriasis', 'lichen-planus', 'granuloma-annulare', 'erythromelalgia',
            'hand-foot-and-mouth-disease', 'erysipelas', 'cellulitis',
            'tinea-pedis', 'athletes-foot', 'pitted-keratolysis',
            'juvenile-plantar-dermatosis', 'onychopapilloma',
        ]
        for kw in dermnet_non_dfu:
            if kw in rel_path:
                return kw
        return None
    rules.append({'label': 'non_dfu', 'confidence': 'high', 'match_fn': _match_non_dfu})

    # ═══ GRADE 1 (LOW CONFIDENCE): unlabeled / generic ═══
    def _match_grade1_low(rel_path: str, fname: str) -> str | None:
        if 'wound_unlabeled' in rel_path:
            return 'unlabeled'
        if 'kaggle_laithjj' in rel_path and '/TestSet/' in rel_path:
            return 'kaggle_testset'
        if 'kaggle_laithjj' in rel_path and '/Original Images/' in rel_path:
            return 'kaggle_original'
        if 'kaggle_laithjj' in rel_path and '/internetSet/' in rel_path:
            return 'kaggle_internet'
        if 'kaggle_laithjj' in rel_path and '/samples/' in rel_path:
            return 'kaggle_samples'
        return None
    rules.append({'label': 'grade1', 'confidence': 'low', 'match_fn': _match_grade1_low})

    # ═══ HEEL X-RAY — already excluded above, no-op here ═══

    return rules


# ─── A1: Build Manifest ──────────────────────────────────────────────

def build_manifest() -> list[dict]:
    """Scan all raw data and return labeled manifest."""
    rules = build_rules()
    manifest = []
    seen_hashes: dict[str, str] = {}  # md5 -> first label

    total_files = 0
    for src_file in sorted(RAW.rglob('*')):
        if not src_file.is_file():
            continue
        if src_file.suffix.lower() not in IMG_EXTS:
            continue

        total_files += 1
        rel_path = str(src_file.relative_to(RAW))
        fname = src_file.name

        # Compute hash for dedup info
        try:
            md5 = hashlib.md5(src_file.read_bytes()).hexdigest()
        except Exception:
            md5 = ""

        # Apply rules
        label = 'unknown'
        confidence = 'unknown'
        match_reason = 'no_match'

        for rule in rules:
            reason = rule['match_fn'](rel_path, fname)
            if reason is not None:
                label = rule['label']
                confidence = rule['confidence']
                match_reason = f"{rule['label']}:{reason}"
                break

        # Track duplicates
        is_dup = False
        if md5 and md5 in seen_hashes:
            is_dup = True

        if md5:
            seen_hashes[md5] = label

        manifest.append({
            'path': rel_path,
            'filename': fname,
            'label': label,
            'confidence': confidence,
            'match_reason': match_reason,
            'size_bytes': src_file.stat().st_size,
            'md5': md5[:12],
            'is_duplicate': is_dup,
        })

    print(f"Scanned {total_files} image files")
    return manifest


# ─── A2: Analyze ─────────────────────────────────────────────────────

def analyze(manifest: list[dict]):
    """Generate label distribution report."""
    labels = [m['label'] for m in manifest]
    confs = [m['confidence'] for m in manifest]
    reasons = [m['match_reason'] for m in manifest]

    label_counts = Counter(labels)
    conf_counts = Counter(confs)

    non_excluded = [m for m in manifest if m['label'] not in ('exclude', 'unknown')]
    trainable_labels = Counter(m['label'] for m in non_excluded)

    lines = []
    lines.append("=" * 70)
    lines.append("R2 Label Distribution Report")
    lines.append("=" * 70)
    lines.append(f"\nTotal images scanned: {len(manifest)}")
    lines.append(f"Excluded (masks/metadata): {label_counts.get('exclude', 0)}")
    lines.append(f"Trainable images: {len(non_excluded)}")

    lines.append(f"\n{'─' * 50}")
    lines.append(f"{'Label':<20} {'Count':>8} {'%':>8}  Status")
    lines.append(f"{'─' * 50}")

    for label in ['normal', 'grade0', 'grade1', 'grade2', 'grade3', 'grade4', 'grade5',
                   'non_dfu', 'exclude', 'unknown']:
        count = label_counts.get(label, 0)
        pct = 100 * count / len(manifest) if manifest else 0
        status = ''
        if label == 'normal':
            status = '✅ healthy control'
        elif label == 'grade0':
            status = '⚠️  target ≥500 (need R3 aug)'
        elif label == 'grade1':
            status = '⚠️  Wagner 1-3 pooled (sub-grade TBD)'
        elif label in ('grade2', 'grade3'):
            status = '📋 placeholder (empty, model-refined later)'
        elif label == 'grade4':
            status = '⚠️  Wagner 4-5 pooled (need more + R3 aug)'
        elif label == 'grade5':
            status = '📋 placeholder (empty, model-refined later)'
        elif label == 'non_dfu':
            status = 'ℹ️  excluded from training'
        elif label == 'exclude':
            status = '🗑️  masks / non-training'
        lines.append(f"{label:<20} {count:>8} {pct:>7.1f}%  {status}")

    # Confidence distribution
    lines.append(f"\n{'─' * 50}")
    lines.append("Confidence Distribution (trainable only):")
    for conf in ['high', 'medium', 'low']:
        count = sum(1 for m in non_excluded if m['confidence'] == conf)
        lines.append(f"  {conf}: {count}")

    # Data sources
    lines.append(f"\n{'─' * 50}")
    lines.append("Top data sources:")
    reason_counts = Counter(reasons)
    for reason, count in reason_counts.most_common(30):
        lines.append(f"  {reason}: {count}")

    # Warnings
    lines.append(f"\n{'─' * 50}")
    lines.append("Warnings & Recommendations:")
    g0 = label_counts.get('grade0', 0)
    gn = label_counts.get('grade4', 0) + label_counts.get('grade5', 0)
    unk = label_counts.get('unknown', 0)
    g1 = label_counts.get('grade1', 0)

    if g0 < 500:
        lines.append(f"  ⚠️  Grade 0: {g0} images — below 500 target (R3 aug will amplify)")
    else:
        lines.append(f"  ✅ Grade 0: {g0} images — meets 500 target")

    if gn < 50:
        lines.append(f"  ⚠️  Gangrene (grade4+grade5): only {gn} images — heavy augmentation needed in R3")
    else:
        lines.append(f"  ✅ Gangrene (grade4+grade5): {gn} images")

    if g1 > 0:
        lines.append(f"  💡 Grade1 images ({g1}) contain Wagner 1-3 pooled together —")
        lines.append(f"     sub-grades not in dataset metadata.")
        lines.append(f"     After R4 model training, use inference to auto-split into grade2/grade3.")
        lines.append(f"  📋 Grade2 & Grade3: empty placeholders (will be populated post-R4).")

    lines.append(f"  📋 Grade5: empty placeholder (will be populated post-R4).")

    if unk > 0:
        lines.append(f"  ⚠️  Unknown: {unk} images unmatched — review needed")

    # Duplicates
    dup_count = sum(1 for m in manifest if m['is_duplicate'])
    if dup_count:
        lines.append(f"  ℹ️  Duplicates (by MD5): {dup_count} — will be filtered in R2 split")

    lines.append(f"\n{'─' * 50}")
    lines.append("Next: R3 — Data Augmentation (RandAugment/MixUp/CutMix)")
    lines.append("      R4 — Two-Stage Model (binary + ordinal)")
    lines.append("=" * 70)

    report = '\n'.join(lines)
    print(report)
    REPORT.write_text(report)
    print(f"\n📄 Report saved to {REPORT}")

    return label_counts


# ─── A3: Stratified Split ────────────────────────────────────────────

def create_split(manifest: list[dict]):
    """Create stratified 70/15/15 train/val/test split and copy files."""

    # Only use trainable, high-confidence, non-duplicate images
    trainable = [
        m for m in manifest
        if m['label'] not in ('exclude', 'unknown', 'non_dfu')
        and m['confidence'] in ('high', 'medium')
        and not m['is_duplicate']
    ]

    print(f"\n📂 Creating stratified split from {len(trainable)} clean images...")

    # Group by label
    by_label = defaultdict(list)
    for m in trainable:
        by_label[m['label']].append(m)

    # 7-class schema: normal, grade0-5
    # grade2, grade3, grade5 are empty placeholders for future model-refinement
    ALL_LABELS = ['normal', 'grade0', 'grade1', 'grade2', 'grade3', 'grade4', 'grade5']
    print(f"\n  Class distribution:")
    for label in ALL_LABELS:
        count = len(by_label.get(label, []))
        marker = " (empty placeholder)" if count == 0 else ""
        print(f"    {label}: {count}{marker}")

    # Stratified split per label
    train_items, val_items, test_items = [], [], []

    for label, items in by_label.items():
        n = len(items)
        if n == 0:
            continue
        if n == 1:
            train_items.extend(items)
            continue
        if n == 2:
            train_items.append(items[0])
            val_items.append(items[1])
            continue

        # Split: 70/15/15
        train_val, test = train_test_split(
            items, test_size=0.15, random_state=RANDOM_SEED, stratify=None
        )
        train, val = train_test_split(
            train_val, test_size=0.15 / 0.85, random_state=RANDOM_SEED, stratify=None
        )
        train_items.extend(train)
        val_items.extend(val)
        test_items.extend(test)

    # Copy files + create empty placeholder directories for grade2, grade3, grade5
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)

    # Pre-create ALL 7 class directories in each split (including empty placeholders)
    for split_name in ['train', 'val', 'test']:
        for label in ALL_LABELS:
            (PROCESSED / split_name / label).mkdir(parents=True, exist_ok=True)

    split_map = {'train': train_items, 'val': val_items, 'test': test_items}
    copied = 0

    for split_name, items in split_map.items():
        for item in items:
            label = item['label']
            dst_dir = PROCESSED / split_name / label
            src = RAW / item['path']
            dst = dst_dir / item['filename']

            # Handle name collisions
            if dst.exists():
                stem, ext = os.path.splitext(item['filename'])
                dst = dst_dir / f"{stem}_{item['md5'][:6]}{ext}"

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"    ✗ {item['filename']}: {e}")

    print(f"\n  Copied {copied} files to {PROCESSED}")

    # Print final distribution
    print(f"\n  Final distribution:")
    for split_name in ['train', 'val', 'test']:
        split_dir = PROCESSED / split_name
        total = sum(1 for _ in split_dir.rglob('*') if _.is_file())
        print(f"    {split_name}: {total}")
        for label in ALL_LABELS:
            d = split_dir / label
            n = len(list(d.iterdir()))
            marker = " (empty placeholder)" if n == 0 else ""
            print(f"      {label}: {n}{marker}")


# ─── Write Manifest CSV ──────────────────────────────────────────────

def write_csv(manifest: list[dict]):
    """Write manifest to CSV."""
    fields = ['path', 'filename', 'label', 'confidence', 'match_reason',
              'size_bytes', 'md5', 'is_duplicate']
    with open(MANIFEST, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow({k: row[k] for k in fields})
    print(f"📄 Manifest saved to {MANIFEST} ({len(manifest)} rows)")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R2: Auto-Labeling Pipeline — DFU Wagner 0-5")
    print("=" * 70)

    # A1: Build manifest
    print("\n🔍 A1: Scanning all raw data...")
    manifest = build_manifest()
    write_csv(manifest)

    # A2: Analyze
    print("\n📊 A2: Label distribution analysis...")
    analyze(manifest)

    # A3: Split
    print("\n✂️  A3: Stratified train/val/test split...")
    create_split(manifest)

    print("\n" + "=" * 70)
    print("✅ R2 complete.")
    print(f"   Manifest: {MANIFEST}")
    print(f"   Report:   {REPORT}")
    print(f"   Dataset:  {PROCESSED}")
    print("=" * 70)


if __name__ == "__main__":
    main()
