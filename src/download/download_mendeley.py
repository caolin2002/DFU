#!/usr/bin/env python3
"""
Download the Mendeley "Lower Limb and Feet Wound Image Dataset" v2.

Dataset: https://data.mendeley.com/datasets/hsj38fwnvr/2
DOI: 10.17632/hsj38fwnvr.2

Contents:
  - 2,757 normal foot images (1,981 male, 776 female)
  - Wound images across multiple severity levels
  - 331×331 px JPEG, ~143 MB total

The download endpoints are public (no auth required), but we need the
file listing first. We use the public-files endpoint directly.

Strategy: Mendeley Data public downloads don't require OAuth for the
file_downloaded endpoint. We fetch file metadata from the dataset page.
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

# --- Config ---
DATASET_ID = "hsj38fwnvr"
DATASET_VERSION = "2"
BASE_URL = f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}"

# Direct archive download — Mendeley hosts a zip of the full dataset
ARCHIVE_URL = f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}/files/all_files/download"

# Fallback: individual file listing via API proxy
API_FILES_URL = f"https://api.mendeley.com/datasets/{DATASET_ID}/files?version={DATASET_VERSION}"

OUTPUT_DIR = Path("/root/dfu/data/raw/mendeley_wound")


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a single file with progress bar and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if dest.exists():
        headers["Range"] = f"bytes={dest.stat().st_size}-"

    with requests.get(url, stream=True, headers=headers, timeout=300) as r:
        if r.status_code == 416:  # Range not satisfiable → already complete
            print(f"  ✓ Already complete: {dest.name}")
            return dest

        if r.status_code not in (200, 206):
            print(f"  ✗ HTTP {r.status_code} for {url}")
            r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        mode = "ab" if r.status_code == 206 else "wb"
        initial = dest.stat().st_size if r.status_code == 206 else 0

        with tqdm(
            total=total + initial,
            initial=initial,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
        ) as pbar:
            with open(dest, mode) as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path, extract_to: Path) -> list[Path]:
    """Extract a zip file and return list of extracted files."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, extract_to)
    return list(extract_to.rglob("*"))


def classify_images(source_dir: Path) -> dict[str, list[Path]]:
    """
    Classify downloaded images into categories based on filename/annotation.

    The Mendeley dataset v2 has the following structure (based on published spec):
    - Filenames encode whether the image is 'normal' or has a wound
    - We parse metadata if available, otherwise use heuristics

    Returns: {"normal": [paths], "wound": [paths], "unknown": [paths]}
    """
    classified = {"normal": [], "wound": [], "unknown": []}

    for img_path in source_dir.rglob("*"):
        if not img_path.is_file():
            continue
        suffix = img_path.suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue

        fname = img_path.stem.lower()
        # Heuristic: filenames with 'normal' or 'healthy' → normal
        if any(kw in fname for kw in ["normal", "healthy", "control"]):
            classified["normal"].append(img_path)
        elif any(kw in fname for kw in ["wound", "ulcer", "dfu", "injury", "lesion"]):
            classified["wound"].append(img_path)
        else:
            classified["unknown"].append(img_path)

    return classified


def organize_by_class(classified: dict, target_dir: Path):
    """Symlink/copy classified images into target directory structure."""
    for category, paths in classified.items():
        if not paths:
            continue
        cat_dir = target_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            dest = cat_dir / p.name
            if not dest.exists():
                # Use relative symlink to save space
                try:
                    os.symlink(os.path.relpath(p, cat_dir), dest)
                except OSError:
                    import shutil
                    shutil.copy2(p, dest)
        print(f"  {category}: {len(paths)} images → {cat_dir}")


def main():
    print("=" * 60)
    print("Mendeley Wound & Normal Foot Dataset Downloader")
    print(f"Dataset: {DATASET_ID} (v{DATASET_VERSION})")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download the full archive ---
    zip_path = OUTPUT_DIR / f"mendeley_{DATASET_ID}_v{DATASET_VERSION}.zip"

    if zip_path.exists():
        print(f"\n[1/3] Archive already exists: {zip_path.name} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"\n[1/3] Downloading full dataset archive...")
        print(f"      URL: {ARCHIVE_URL}")
        try:
            download_file(ARCHIVE_URL, zip_path, desc="Downloading archive")
        except requests.HTTPError as e:
            print(f"  ✗ Archive download failed: {e}")
            print("  Trying alternative: individual file download...")
            download_individual_files()
            return

    # --- Step 2: Extract ---
    extract_dir = OUTPUT_DIR / "extracted"
    if extract_dir.exists() and any(extract_dir.rglob("*")):
        print(f"\n[2/3] Already extracted to: {extract_dir}")
    else:
        print(f"\n[2/3] Extracting archive...")
        extract_zip(zip_path, extract_dir)

    # --- Step 3: Classify and organize ---
    print(f"\n[3/3] Classifying images...")
    classified = classify_images(extract_dir)

    organized_dir = OUTPUT_DIR / "organized"
    organize_by_class(classified, organized_dir)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Download Complete — Summary:")
    total = sum(len(v) for v in classified.values())
    print(f"  Total images: {total}")
    for cat, paths in classified.items():
        if paths:
            print(f"  {cat}: {len(paths)}")
    print(f"  Output: {organized_dir}")
    print(f"{'=' * 60}")
    return classified


def download_individual_files():
    """Fallback: download files one by one via API listing."""
    print(f"  Fetching file list from API...")

    # Try public endpoint without auth first
    headers = {
        "Accept": "application/vnd.mendeley-public-dataset.1+json",
    }

    try:
        r = requests.get(API_FILES_URL, headers=headers, timeout=30)
        if r.status_code == 200:
            files = r.json()
        else:
            print(f"  API returned {r.status_code}, trying public-files...")
            files = []
    except Exception as e:
        print(f"  API error: {e}")
        files = []

    if not files:
        print("  ✗ Could not retrieve file list.")
        print("  Manual download URL:")
        print(f"    https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}")
        return

    files_dir = OUTPUT_DIR / "individual"
    files_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        file_id = f.get("id", f.get("file_id", ""))
        filename = f.get("filename", file_id)
        url = f"{BASE_URL}/files/{file_id}/file_downloaded"
        dest = files_dir / filename

        if dest.exists():
            print(f"  ✓ {filename}")
            continue

        print(f"  Downloading: {filename}")
        try:
            download_file(url, dest, desc=f"  {filename}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


if __name__ == "__main__":
    main()
