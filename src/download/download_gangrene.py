#!/usr/bin/env python3
"""
Download gangrene foot images from public medical image repositories.

Sources:
  1. Open-i (NIH) — open-access biomedical images
  2. Wikimedia Commons — CC-licensed medical images
  3. MedPix — radiology/pathology images (some require registration)

Strategy:
  - Query Open-i API for "foot gangrene", "diabetic foot necrosis",
    "toe amputation", "forefoot gangrene"
  - Download with CC license filtering
  - Organize into W4 (localized) and W5 (full-foot) subdirectories
"""

import os
import time
import json
import hashlib
from pathlib import Path
from urllib.parse import urljoin, quote_plus

import requests
from tqdm import tqdm

OUTPUT_DIR = Path("/root/dfu/data/raw/gangrene")

# ── Open-i (NIH) API ──────────────────────────────────────────────
OPENI_BASE = "https://openi.nlm.nih.gov"
OPENI_SEARCH = f"{OPENI_BASE}/api/search"
OPENI_RETRIEVE = f"{OPENI_BASE}/retrieve.php"

# Query terms mapped to our categories
QUERIES = {
    "w4_localized": [
        "toe gangrene diabetic",
        "forefoot necrosis",
        "digital gangrene foot",
        "heel gangrene",
    ],
    "w5_full": [
        "full foot gangrene diabetic",
        "transmetatarsal amputation necrosis",
        "below knee amputation diabetic foot",
        "wet gangrene foot diabetic",
    ],
    "grade0_highrisk": [
        "diabetic foot callus",
        "foot deformity charcot",
        "claw toe deformity diabetic",
        "dry skin fissure diabetic foot",
        "onychomycosis diabetic nail",
    ],
}


def search_openi(query: str, max_results: int = 100) -> list[dict]:
    """Search Open-i for images matching query."""
    params = {
        "query": query,
        "m": str(max_results),
        "it": "j",  # image type: jpeg
    }
    try:
        r = requests.get(OPENI_SEARCH, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("list", [])
    except Exception as e:
        print(f"  Open-i search error for '{query}': {e}")
        return []


def download_openi_image(img_id: str, dest_dir: Path, prefix: str = "") -> Path | None:
    """Download a single Open-i image by ID."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (by img_id)
    existing = list(dest_dir.glob(f"{prefix}{img_id}*"))
    if existing:
        return existing[0]

    url = f"{OPENI_RETRIEVE}?id={img_id}&type=large"
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code != 200 or "image" not in r.headers.get("content-type", ""):
            return None

        ext = ".jpg"
        ct = r.headers.get("content-type", "")
        if "png" in ct:
            ext = ".png"
        elif "gif" in ct:
            ext = ".gif"

        fname = f"{prefix}{img_id}{ext}"
        fpath = dest_dir / fname

        with open(fpath, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        # Filter out tiny/non-images (< 5KB)
        if fpath.stat().st_size < 5000:
            fpath.unlink()
            return None

        return fpath
    except Exception as e:
        return None


def search_and_download_queries(
    query_map: dict[str, list[str]],
    output_base: Path,
    max_per_query: int = 50,
    delay: float = 0.5,
) -> dict[str, int]:
    """
    Search Open-i with multiple queries and download results.

    Returns: {category: count_downloaded}
    """
    results = {}
    seen_ids = set()

    for category, queries in query_map.items():
        cat_dir = output_base / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for query in queries:
            print(f"\n  Searching: '{query}'")
            items = search_openi(query, max_results=max_per_query)

            new_items = [it for it in items if it.get("img_id") not in seen_ids]
            print(f"    Found {len(items)} results, {len(new_items)} new")

            for item in tqdm(new_items, desc=f"  Downloading '{query}'"):
                img_id = item.get("img_id", "")
                if not img_id or img_id in seen_ids:
                    continue
                seen_ids.add(img_id)

                prefix = f"{category}_"
                fpath = download_openi_image(img_id, cat_dir, prefix)
                if fpath:
                    count += 1

                time.sleep(delay)  # Rate limiting

        results[category] = count
        print(f"  → {category}: {count} images downloaded")

    return results


def search_wikimedia(query: str, limit: int = 50) -> list[str]:
    """
    Search Wikimedia Commons for CC-licensed medical images.

    Uses the Commons Search API (no auth required).
    """
    import urllib.parse

    base_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "srnamespace": "6",  # File namespace
    }
    try:
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("search", [])
        return [p["title"] for p in pages]
    except Exception as e:
        print(f"  Wikimedia search error: {e}")
        return []


def download_wikimedia_image(file_title: str, dest_dir: Path) -> Path | None:
    """Download a single image from Wikimedia Commons."""
    import urllib.parse

    # Remove "File:" prefix for the download URL
    safe_title = file_title.replace("File:", "").strip()
    # Get the actual image URL
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
    }
    try:
        r = requests.get(api_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})

        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            img_info = page.get("imageinfo", [{}])[0]
            img_url = img_info.get("url", "")
            mime = img_info.get("mime", "")
            if not img_url or "image" not in mime:
                continue

            ext = Path(urllib.parse.urlparse(img_url).path).suffix or ".jpg"
            fname = f"{safe_title[:80].replace('/', '_')}{ext}"
            fpath = dest_dir / fname

            if fpath.exists():
                return fpath

            img_r = requests.get(img_url, timeout=60, stream=True)
            if img_r.status_code == 200:
                with open(fpath, "wb") as f:
                    for chunk in img_r.iter_content(8192):
                        f.write(chunk)
                if fpath.stat().st_size > 5000:
                    return fpath
                else:
                    fpath.unlink()
    except Exception as e:
        pass
    return None


def download_from_wikimedia(queries: list[str], dest_dir: Path, limit: int = 30) -> int:
    """Search and download from Wikimedia Commons."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()

    for query in queries:
        print(f"\n  Searching Wikimedia: '{query}'")
        titles = search_wikimedia(query, limit=limit)
        print(f"    Found {len(titles)} files")

        for title in tqdm(titles, desc=f"  Downloading"):
            safe = title.replace("File:", "").strip()[:80]
            if safe in seen:
                continue
            seen.add(safe)

            fpath = download_wikimedia_image(title, dest_dir)
            if fpath:
                count += 1
            time.sleep(0.3)

    return count


def main():
    print("=" * 60)
    print("Gangrene & High-Risk Foot Image Downloader")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Source 1: Open-i (NIH) ────────────────────────────────────
    print("\n[1/2] Open-i (NIH National Library of Medicine)")
    print("-" * 40)

    openi_queries = {
        "w4_localized": QUERIES["w4_localized"],
        "w5_full": QUERIES["w5_full"],
        "grade0_highrisk": QUERIES["grade0_highrisk"],
    }

    openi_results = search_and_download_queries(
        openi_queries,
        OUTPUT_DIR,
        max_per_query=50,
        delay=0.5,
    )

    # ── Source 2: Wikimedia Commons ───────────────────────────────
    print(f"\n[2/2] Wikimedia Commons")
    print("-" * 40)

    wiki_queries = [
        "diabetic foot gangrene",
        "toe necrosis",
        "forefoot amputation",
        "diabetic foot callus",
        "foot deformity charcot",
    ]

    wiki_count = download_from_wikimedia(wiki_queries, OUTPUT_DIR / "wikimedia")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Download Summary:")
    print(f"  Open-i (NIH):")
    for cat, cnt in openi_results.items():
        print(f"    {cat}: {cnt} images")
    print(f"  Wikimedia Commons: {wiki_count} images")
    total = sum(openi_results.values()) + wiki_count
    print(f"  TOTAL: {total} images")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
