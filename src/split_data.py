#!/usr/bin/env python3
"""
Data splitting script for DFU dataset.
Groups augmented images by original image ID, then performs
a stratified 70/15/15 train/val/test split to prevent data leakage.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split

random.seed(42)

ARCHIVE_DIR = Path("/root/dfu/archive")
OUTPUT_DIR = Path("/root/dfu/data")
SPLITS = ["train", "valid", "test"]
GRADES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]


def get_original_id(filename: str) -> str:
    """
    Extract the original image ID from a Roboflow-augmented filename.
    Pattern: <orig_id>_jpg.rf.<hash>.jpg  or  <orig_id>_png.rf.<hash>.jpg
    Returns everything before '.rf.'.
    """
    idx = filename.find(".rf.")
    if idx != -1:
        return filename[:idx]
    # Fallback: no .rf. marker — treat the whole stem as the ID
    return Path(filename).stem


def collect_files():
    """
    Scan archive directory and return a dict:
      { original_id: [list of (src_path, grade, split)] }
    """
    groups = defaultdict(list)
    total = 0
    for split in SPLITS:
        for grade in GRADES:
            dir_path = ARCHIVE_DIR / split / grade
            if not dir_path.exists():
                continue
            for f in dir_path.glob("*.jpg"):
                orig_id = get_original_id(f.name)
                groups[orig_id].append((str(f), grade, split))
                total += 1
    print(f"Collected {total} images from {len(groups)} original groups")
    return groups


def split_and_copy(groups: dict):
    """
    Split original groups into train/val/test (70/15/15),
    then copy all augmented variants of each group to the assigned split.
    """
    # Collect original IDs per grade (a group's grade is the majority grade of its files)
    ids_by_grade = defaultdict(list)
    for orig_id, files in groups.items():
        # Determine grade by majority vote
        grade_counts = defaultdict(int)
        for _, grade, _ in files:
            grade_counts[grade] += 1
        majority_grade = max(grade_counts, key=grade_counts.get)
        ids_by_grade[majority_grade].append(orig_id)

    # Split each grade independently
    train_ids, val_ids, test_ids = [], [], []
    for grade in GRADES:
        ids = sorted(ids_by_grade.get(grade, []))
        if not ids:
            continue
        n = len(ids)
        if n == 1:
            train_ids.extend(ids)
            print(f"  {grade}: only 1 group, placing in train")
            continue

        # First split off test (15%)
        train_val, te = train_test_split(
            ids, test_size=0.15, random_state=42
        )
        # Then split remaining into train/val (70:15 of total ≈ 82.35:17.65 of remaining)
        tr, va = train_test_split(
            train_val, test_size=0.1765, random_state=42
        )
        train_ids.extend(tr)
        val_ids.extend(va)
        test_ids.extend(te)
        print(f"  {grade}: {len(tr)} train / {len(va)} val / {len(te)} test groups")

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    # Copy files
    print("\nCopying files...")
    counts = {"train": 0, "val": 0, "test": 0}
    for orig_id, files in groups.items():
        if orig_id in train_set:
            target_split = "train"
        elif orig_id in val_set:
            target_split = "val"
        else:
            target_split = "test"

        for src_path, grade, _ in files:
            dst_dir = OUTPUT_DIR / target_split / grade
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_dir / Path(src_path).name)
            counts[target_split] += 1

    print(f"Done: {counts['train']} train / {counts['val']} val / {counts['test']} test")
    return counts


def main():
    # Clean existing output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    groups = collect_files()
    counts = split_and_copy(groups)

    # Summary
    print("\n=== Final Distribution ===")
    for split in ["train", "val", "test"]:
        print(f"\n{split}:")
        for grade in GRADES:
            d = OUTPUT_DIR / split / grade
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {grade}: {n}")


if __name__ == "__main__":
    main()
