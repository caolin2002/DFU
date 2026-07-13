#!/usr/bin/env python3
"""
Download Kaggle DFU datasets.

Requires Kaggle API credentials. Set via environment variables or kaggle.json.

Datasets:
  1. laithjj/diabetic-foot-ulcer-dfu   — Binary healthy/ulcer (1,055 images)
  2. Additional DFU datasets if available

Setup:
  export KAGGLE_USERNAME="your_username"
  export KAGGLE_KEY="your_api_key"

  Or place kaggle.json in ~/.kaggle/
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path

OUTPUT_DIR = Path("/root/dfu/data/raw")


# ── Dataset registry ─────────────────────────────────────────────
DATASETS = [
    {
        "slug": "laithjj/diabetic-foot-ulcer-dfu",
        "name": "Kaggle Laithjj DFU",
        "target_dir": OUTPUT_DIR / "kaggle_laithjj",
        "description": "Binary healthy/ulcer classification, 1,055 images (224×224)",
        "classes": {"Healthy": "normal", "Ulcer": "wound"},
    },
    # Additional datasets can be added here as discovered
]


def check_kaggle_auth() -> bool:
    """Check if Kaggle API credentials are available."""
    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    # Check kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True

    return False


def setup_kaggle_auth():
    """Guide user to set up Kaggle authentication."""
    print("\n" + "!" * 60)
    print("  Kaggle API credentials NOT found!")
    print("!" * 60)
    print("""
To set up Kaggle API access:

1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section → click "Create New Token"
3. Download kaggle.json

Then either:
  A) Set environment variables:
     export KAGGLE_USERNAME="your_username"
     export KAGGLE_KEY="your_api_key"

  B) Place kaggle.json in ~/.kaggle/:
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
""")
    return False


def download_dataset(slug: str, target_dir: Path) -> bool:
    """Download a Kaggle dataset using the CLI."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if any(target_dir.iterdir()):
        existing = list(target_dir.rglob("*.jpg")) + list(target_dir.rglob("*.jpeg")) + list(target_dir.rglob("*.png"))
        if existing:
            print(f"  ✓ Already downloaded: {len(existing)} images in {target_dir}")
            return True

    print(f"  Downloading {slug}...")

    import subprocess

    # Use kaggle CLI
    cmd = [
        "kaggle", "datasets", "download",
        "-d", slug,
        "-p", str(target_dir),
        "--unzip",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  ✓ Download complete: {target_dir}")
            return True
        else:
            print(f"  ✗ CLI error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Download timed out (>10 min)")
        return False
    except FileNotFoundError:
        print(f"  ✗ 'kaggle' CLI not found. Install: pip install kaggle")
        return False


def try_direct_download(slug: str, target_dir: Path) -> bool:
    """
    Attempt direct download without Kaggle CLI (for public datasets).

    Kaggle datasets can sometimes be downloaded via their public URL pattern.
    This requires browser cookies for authenticated access to certain datasets.
    """
    import requests

    # Kaggle's public download URL pattern
    owner, dataset = slug.split("/")
    url = f"https://www.kaggle.com/api/v1/datasets/{owner}/{dataset}/download"

    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")

    if not username or not key:
        return False

    try:
        r = requests.get(url, auth=(username, key), stream=True, timeout=300)
        if r.status_code == 200:
            zip_path = target_dir / f"{dataset}.zip"
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
            zip_path.unlink()  # Remove zip
            return True
    except Exception as e:
        print(f"  Direct download failed: {e}")

    return False


def organize_kaggle_data(source_dir: Path):
    """Organize downloaded Kaggle data into class subdirectories."""
    # Many Kaggle DFU datasets already have class subdirectories
    # If not, organize by filename pattern
    jpgs = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.jpeg")) + list(source_dir.rglob("*.png"))

    if not jpgs:
        print(f"  No images found in {source_dir}")
        return

    # Check if already organized (has subdirectories)
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"  Already organized into {len(subdirs)} class directories")
        return

    print(f"  Found {len(jpgs)} images (flat), organizing...")


def main():
    print("=" * 60)
    print("Kaggle DFU Dataset Downloader")
    print("=" * 60)

    if not check_kaggle_auth():
        setup_kaggle_auth()

        # Ask user if they want to proceed with manual setup
        print("\nOptions:")
        print("  1. Set up Kaggle API now and retry")
        print("  2. Continue without Kaggle (skip these datasets)")
        print("  3. Provide credentials directly")

        choice = input("\nChoice [2]: ").strip() or "2"

        if choice == "1":
            setup_kaggle_auth()
            if not check_kaggle_auth():
                print("Still no credentials. Skipping Kaggle datasets.")
                return
        elif choice == "3":
            username = input("Kaggle username: ").strip()
            key = input("Kaggle API key: ").strip()
            if username and key:
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = key
            else:
                print("Invalid credentials. Skipping.")
                return
        else:
            print("Skipping Kaggle datasets.")
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for ds in DATASETS:
        print(f"\n[{ds['name']}]")
        print(f"  {ds['description']}")

        success = download_dataset(ds["slug"], ds["target_dir"])
        if not success:
            print(f"  Trying direct download...")
            success = try_direct_download(ds["slug"], ds["target_dir"])

        if success:
            organize_kaggle_data(ds["target_dir"])

        results[ds["slug"]] = success

    # Summary
    print(f"\n{'=' * 60}")
    print("Kaggle Download Summary:")
    for slug, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {slug}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
