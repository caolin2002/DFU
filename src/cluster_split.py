#!/usr/bin/env python3
"""
方案 A v2: 混合特征聚类拆分 grade1 → grade1 + grade2 + grade3

改进策略：
1. 传统 CV 特征 (HSV 颜色直方图 + GLCM 纹理) — 直接与创面深度相关
2. 深度特征 (ConvNeXt backbone) — 通用视觉特征
3. CORN 序数概率分布 — 模型的"倾向"
4. 多种策略对比：K-Means / DBSCAN / 概率阈值分割

Wagner 1-3 的临床视觉差异:
  Wagner 1 (浅表): 红色肉芽组织为主, 边界清晰
  Wagner 2 (深部): 黄色腐肉/渗出, 可见筋膜/肌腱
  Wagner 3 (骨髓炎): 白色骨暴露, 深部窦道, 红肿范围大
  → 颜色从红→黄→白/黑 是核心区分特征
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import get_convnext_tiny

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================================================
# Dataset
# ==========================================================================

class ImageFolderDataset(Dataset):
    """Load images from flat directory."""
    def __init__(self, root_dir: str, input_size: int = 224):
        self.root = Path(root_dir)
        self.paths = sorted([
            p for p in self.root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        ])
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, str(path), path.name
        except Exception:
            return torch.zeros(3, 224, 224), str(path), path.name


# ==========================================================================
# Traditional CV features — clinically relevant
# ==========================================================================

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB numpy array [H, W, 3] to HSV [H, W, 3] using pure numpy."""
    r, g, b = rgb[:, :, 0] / 255.0, rgb[:, :, 1] / 255.0, rgb[:, :, 2] / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask_r = cmax == r
    mask_g = cmax == g
    mask_b = cmax == b
    # Avoid division by zero (delta=0 in gray pixels)
    safe_delta = np.where(delta > 0, delta, 1.0)
    h = np.zeros_like(cmax)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / safe_delta[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / safe_delta[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / safe_delta[mask_b]) + 4)
    # Saturation
    s = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)
    # Value
    v = cmax

    return np.stack([h, s * 255, v * 255], axis=-1).astype(np.float32)


def compute_hsv_histogram(img: Image.Image, bins: int = 32) -> np.ndarray:
    """HSV color histogram — wound color correlates with severity (pure numpy)."""
    img_np = np.array(img.convert("RGB"))
    img_hsv = rgb_to_hsv_np(img_np)

    hist_h, _ = np.histogram(img_hsv[:, :, 0], bins=bins, range=(0, 360))
    hist_s, _ = np.histogram(img_hsv[:, :, 1], bins=bins, range=(0, 256))
    hist_v, _ = np.histogram(img_hsv[:, :, 2], bins=bins, range=(0, 256))
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    return hist


def compute_color_stats(img: Image.Image) -> np.ndarray:
    """Basic color statistics in RGB and HSV (pure numpy)."""
    img_np = np.array(img.convert("RGB")).astype(np.float32)
    img_hsv = rgb_to_hsv_np(np.array(img.convert("RGB")))

    features = []
    for channel_img in [img_np, img_hsv]:
        for c in range(3):
            ch = channel_img[:, :, c]
            features.extend([float(ch.mean()), float(ch.std()), float(np.median(ch)),
                             float(np.percentile(ch, 10)), float(np.percentile(ch, 90))])
    return np.array(features, dtype=np.float32)


def compute_texture_features(img: Image.Image) -> np.ndarray:
    """GLCM texture features — tissue texture changes with wound depth."""
    from skimage.feature import graycomatrix, graycoprops

    img_gray = np.array(img.convert("L"))
    # Quantize to 32 levels for manageable GLCM
    img_q = (img_gray // 8).astype(np.uint8)

    # GLCM at 4 angles, distance=3
    glcm = graycomatrix(img_q, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)

    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        vals = graycoprops(glcm, prop).flatten()
        features.extend([float(vals.mean()), float(vals.std())])
    return np.array(features, dtype=np.float32)


def compute_traditional_features(img_path: str, fast: bool = True) -> np.ndarray:
    """Compute traditional CV features for one image."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        # Return zero features for broken images
        return np.zeros(96 + 30 + 10, dtype=np.float32)

    # Resize for speed
    if fast:
        img = img.resize((224, 224))

    hsv_hist = compute_hsv_histogram(img, bins=32)  # 96-d
    color_stats = compute_color_stats(img)           # 30-d (3 RGB + 3 HSV) * 5 stats
    texture = compute_texture_features(img)           # 10-d (5 props * 2 stats each)

    return np.concatenate([hsv_hist, color_stats, texture])


# ==========================================================================
# Deep feature extraction
# ==========================================================================

@torch.no_grad()
def extract_deep_features(model, dataset, device, batch_size=128):
    """Extract ConvNeXt backbone features."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_paths = []
    all_filenames = []

    for tensors, paths, filenames in tqdm(loader, desc="Deep features"):
        tensors = tensors.to(device)
        features = model.backbone(tensors)
        all_features.append(features.cpu().numpy())
        all_paths.extend(paths)
        all_filenames.extend(filenames)

    return np.concatenate(all_features, axis=0), all_paths, all_filenames


@torch.no_grad()
def extract_ordinal_probs(model, dataset, device, batch_size=128):
    """Get CORN ordinal probability distribution for each image."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs = []
    all_thresh_probs = []

    for tensors, _, _ in tqdm(loader, desc="Ordinal probs"):
        tensors = tensors.to(device)
        features = model.backbone(tensors)
        ord_logits = model.ordinal_head(features)
        class_probs = model.ordinal_head.predict_proba(ord_logits)  # [B, 7]
        thresh_probs = torch.sigmoid(ord_logits)  # [B, 6] — P(>=class_k)
        all_probs.append(class_probs.cpu().numpy())
        all_thresh_probs.append(thresh_probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_thresh_probs, axis=0)


# ==========================================================================
# Clustering strategies
# ==========================================================================

def strategy_kmeans(features: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """K-Means clustering."""
    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(features)
    pca = PCA(n_components=0.95, random_state=42)
    feats_pca = pca.fit_transform(feats_norm)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(feats_pca)
    print(f"  K-Means: PCA {features.shape[1]}→{feats_pca.shape[1]}d, inertia={kmeans.inertia_:.1f}")
    return labels


def strategy_threshold_ranking(thresh_probs: np.ndarray) -> np.ndarray:
    """
    基于 CORN 序数阈值概率排序分割。

    思路：P(>=grade2) = sigmoid(logit_2) 即使对所有 grade1 图片模型都输出
    接近 0 的值，微小的差异也反映了模型感知到的"更像 grade2"的程度。
    按此概率排序后三等分：
      - 下 1/3: P(>=grade2) 最低 → 最不像 grade2 → Wagner 1
      - 中 1/3: 中间                   → Wagner 2
      - 上 1/3: P(>=grade2) 最高 → 最像 grade2 → Wagner 3
    """
    # P(>=grade2) is threshold index 2 (the 3rd threshold, 0-indexed)
    p_ge_grade2 = thresh_probs[:, 2]  # [N]

    # Sort by this probability
    order = np.argsort(p_ge_grade2)
    n = len(order)

    labels = np.zeros(n, dtype=int)
    split1 = n // 3
    split2 = 2 * n // 3
    labels[order[split1:split2]] = 1   # middle → grade2
    labels[order[split2:]] = 2          # top → grade3

    print(f"  Threshold ranking: P(>=grade2) range [{p_ge_grade2.min():.4f}, {p_ge_grade2.max():.4f}]")
    print(f"    Bottom 1/3 (grade1): P∈[{p_ge_grade2[order[0]]:.4f}, {p_ge_grade2[order[split1-1]]:.4f}]")
    print(f"    Middle 1/3 (grade2): P∈[{p_ge_grade2[order[split1]]:.4f}, {p_ge_grade2[order[split2-1]]:.4f}]")
    print(f"    Top 1/3    (grade3): P∈[{p_ge_grade2[order[split2]]:.4f}, {p_ge_grade2[order[-1]]:.4f}]")

    return labels


def strategy_feature_ranking(deep_feats: np.ndarray, thresh_probs: np.ndarray) -> np.ndarray:
    """
    混合策略：特征空间的质心距离 + 序数概率

    对 deep features 做 PCA 降维后计算每个样本到"平均 grade1 特征"的距离，
    越远的越异常 → 可能是 deeper wound。结合 P(>=grade2) 排序。
    """
    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(deep_feats)

    pca = PCA(n_components=50, random_state=42)
    feats_pca = pca.fit_transform(feats_norm)

    # Distance from centroid in PCA space
    centroid = feats_pca.mean(axis=0)
    distances = np.linalg.norm(feats_pca - centroid, axis=1)

    # Combine distance with ordinal probability
    p_ge_grade2 = thresh_probs[:, 2]
    combined = 0.5 * (distances / distances.std()) + 0.5 * (p_ge_grade2 / (p_ge_grade2.std() + 1e-8))

    order = np.argsort(combined)
    n = len(order)

    labels = np.zeros(n, dtype=int)
    split1 = n // 3
    split2 = 2 * n // 3
    labels[order[split1:split2]] = 1
    labels[order[split2:]] = 2

    print(f"  Feature+Prob ranking: distance range [{distances.min():.1f}, {distances.max():.1f}]")
    return labels


# ==========================================================================
# Evaluate clusters
# ==========================================================================

def evaluate_clusters(labels: np.ndarray, features: np.ndarray,
                      ordinal_probs: np.ndarray, thresh_probs: np.ndarray,
                      strategy_name: str) -> dict:
    """Evaluate cluster quality and interpretability."""
    n_clusters = len(np.unique(labels))

    # Silhouette score
    n_sample = min(3000, len(features))
    idx = np.random.RandomState(42).choice(len(features), n_sample, replace=False)
    sil = silhouette_score(features[idx], labels[idx]) if n_clusters > 1 else 0.0

    stats = {}
    for cid in range(n_clusters):
        mask = labels == cid
        c_probs = ordinal_probs[mask]
        c_thresh = thresh_probs[mask]

        # Mean ordinal probabilities per class
        mean_probs = c_probs.mean(axis=0)

        # Mean threshold probabilities
        mean_thresh = c_thresh.mean(axis=0)

        # Mean P(>=grade2) — key severity indicator
        p_ge_grade2 = c_thresh[:, 2]

        stats[int(cid)] = {
            "size": int(mask.sum()),
            "size_pct": round(100 * mask.sum() / len(labels), 1),
            "mean_class_probs": [round(float(p), 4) for p in mean_probs],
            "mean_thresh_probs": [round(float(p), 4) for p in mean_thresh],
            "mean_p_ge_grade2": round(float(p_ge_grade2.mean()), 6),
            "std_p_ge_grade2": round(float(p_ge_grade2.std()), 6),
            "p_ge_grade2_range": [round(float(p_ge_grade2.min()), 6),
                                  round(float(p_ge_grade2.max()), 6)],
        }

    # Sort clusters by mean P(>=grade2), map to Wagner
    sorted_cids = sorted(stats.keys(),
                         key=lambda c: stats[c]["mean_p_ge_grade2"])
    cluster_to_grade = {}
    grade_labels = {0: "grade1", 1: "grade2", 2: "grade3"}
    for rank, cid in enumerate(sorted_cids):
        cluster_to_grade[cid] = {
            "wagner_grade": rank + 2,
            "wagner_label": grade_labels[rank],
        }
        stats[cid]["assigned_grade"] = grade_labels[rank]

    report = {
        "strategy": strategy_name,
        "n_clusters": n_clusters,
        "silhouette_score": round(float(sil), 4),
        "cluster_stats": {int(k): v for k, v in stats.items()},
        "cluster_to_grade": {int(k): v for k, v in cluster_to_grade.items()},
    }

    # Print report
    print(f"\n  Silhouette: {sil:.4f}" + (" ✓" if sil > 0.25 else " ✗ (poor separation)"))
    print(f"  {'Cluster':<8} {'Size':<8} {'%':<7} {'P(>=grade2)':<16} {'→ Grade':<10}")
    print(f"  " + "-" * 55)
    for cid in sorted(stats.keys()):
        s = stats[cid]
        g = cluster_to_grade[cid]
        print(f"  C{cid:<7} {s['size']:<8} {s['size_pct']:<7} "
              f"{s['mean_p_ge_grade2']:<16.6f} → {g['wagner_label']:<10}")

    return report


# ==========================================================================
# Visualization
# ==========================================================================

def visualize_comparison(all_results: list, features_for_viz: np.ndarray,
                         output_dir: str):
    """Generate t-SNE with multiple strategy labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sample = min(2000, len(features_for_viz))
    idx = np.random.RandomState(42).choice(len(features_for_viz), n_sample, replace=False)

    # t-SNE on PCA-reduced features
    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(features_for_viz)
    pca = PCA(n_components=50, random_state=42)
    feats_pca = pca.fit_transform(feats_norm)

    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, n_jobs=1)
    feats_2d = tsne.fit_transform(feats_pca[idx])

    n_strategies = len(all_results)
    fig, axes = plt.subplots(1, n_strategies + 1, figsize=(6*(n_strategies+1), 5.5))

    # First plot: color by P(>=grade2) — ground truth severity proxy
    ax = axes[0]
    scatter = ax.scatter(feats_2d[:, 0], feats_2d[:, 1],
                         c=all_results[0].get("p_ge_grade2", np.zeros(n_sample))[idx],
                         cmap="RdYlGn_r", alpha=0.6, s=12)
    ax.set_title("P(>=grade2) heatmap\n(red=more severe)")
    plt.colorbar(scatter, ax=ax)

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red → grade1,2,3
    for i, result in enumerate(all_results):
        ax = axes[i + 1]
        labels = result["labels"][idx]
        grade_map = result["cluster_to_grade"]
        for cid in sorted(np.unique(labels)):
            mask = labels == cid
            lbl = grade_map.get(cid, {}).get("wagner_label", f"C{cid}")
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                       c=colors[cid % 3], alpha=0.5, s=8,
                       label=f"{lbl} (n={mask.sum()})")
        ax.set_title(result["strategy"])
        ax.legend(markerscale=3, fontsize=7)

    plt.tight_layout()
    vis_path = os.path.join(output_dir, "strategy_comparison_tsne.png")
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {vis_path}")


# ==========================================================================
# Apply split
# ==========================================================================

def apply_split(labels: np.ndarray, filenames: list, source_dir: str,
                output_base: str, cluster_to_grade: dict, mode: str = "copy"):
    """Copy/move images based on cluster assignment."""
    import shutil

    for cid, grade_info in cluster_to_grade.items():
        os.makedirs(os.path.join(output_base, grade_info["wagner_label"]), exist_ok=True)

    counts = Counter()
    for fname, cid in tqdm(zip(filenames, labels), total=len(filenames), desc=f"Applying ({mode})"):
        grade_label = cluster_to_grade[cid]["wagner_label"]
        src = os.path.join(source_dir, fname)
        dst = os.path.join(output_base, grade_label, fname)

        if os.path.abspath(src) == os.path.abspath(dst):
            # Source and destination are the same file — skip
            pass
        elif mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "move":
            shutil.move(src, dst)
        counts[grade_label] += 1

    print(f"\n  Split result: grade1={counts['grade1']}, "
          f"grade2={counts['grade2']}, grade3={counts['grade3']}")
    return dict(counts)


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="方案A v2: 混合特征+多策略聚类拆分 grade1→grade2/3"
    )
    parser.add_argument("--source", type=str,
                        default=str(PROJECT_ROOT / "data/processed/train/grade1"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "data/processed/train"))
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "models/best_model.pth"))
    parser.add_argument("--strategy", type=str,
                        choices=["all", "kmeans_deep", "kmeans_traditional",
                                 "kmeans_hybrid", "threshold_ranking",
                                 "feature_ranking"],
                        default="all")
    parser.add_argument("--mode", type=str,
                        choices=["copy", "move", "dry-run"], default="dry-run")
    parser.add_argument("--report-dir", type=str,
                        default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Source: {args.source}")

    # ── Load model ──
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    input_size = cfg.get("data", {}).get("input_size", 224)
    num_classes = cfg.get("model", {}).get("num_classes", 7)

    model = get_convnext_tiny(num_classes=num_classes, binary=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── Load dataset ──
    print(f"\nLoading {args.source}...")
    dataset = ImageFolderDataset(args.source, input_size=input_size)
    print(f"  {len(dataset)} images")

    if len(dataset) == 0:
        print("ERROR: No images found!"); return

    # ── Extract deep features ──
    print(f"\n{'='*60}")
    print("Extracting deep features...")
    deep_feats, paths, filenames = extract_deep_features(model, dataset, device, args.batch_size)
    print(f"  Shape: {deep_feats.shape}")

    # ── Extract ordinal probabilities ──
    print(f"\nExtracting ordinal probabilities...")
    ordinal_probs, thresh_probs = extract_ordinal_probs(model, dataset, device, args.batch_size)

    # Distribution of P(>=grade2) — key signal
    p_ge_grade2 = thresh_probs[:, 2]
    print(f"  P(>=grade2): min={p_ge_grade2.min():.6f}, max={p_ge_grade2.max():.6f}, "
          f"mean={p_ge_grade2.mean():.6f}, std={p_ge_grade2.std():.6f}")

    # ── Extract traditional features ──
    print(f"\nExtracting traditional CV features...")
    trad_feats = []
    for path in tqdm(paths, desc="CV features"):
        trad_feats.append(compute_traditional_features(path, fast=True))
    trad_feats = np.array(trad_feats)
    print(f"  Shape: {trad_feats.shape}")
    print(f"  Bins: HSV histogram(96) + Color stats(30) + GLCM texture(10) = {trad_feats.shape[1]}d")

    # ── Feature combinations ──
    print(f"\n{'='*60}")
    print("Running multiple clustering strategies...")
    print(f"{'='*60}")

    all_results = []

    # Feature matrices
    scaler = StandardScaler()
    deep_norm = scaler.fit_transform(deep_feats)
    trad_norm = scaler.fit_transform(trad_feats)
    # Handle NaN from GLCM on uniform images
    trad_norm = np.nan_to_num(trad_norm, nan=0.0, posinf=0.0, neginf=0.0)
    hybrid = np.concatenate([deep_norm, trad_norm, thresh_probs], axis=1)
    hybrid_norm = scaler.fit_transform(hybrid)

    strategies_to_run = args.strategy

    # Strategy 1: K-Means on deep features only
    if strategies_to_run in ("all", "kmeans_deep"):
        print(f"\n--- Strategy: K-Means on Deep Features ---")
        labels = strategy_kmeans(deep_feats, n_clusters=3)
        report = evaluate_clusters(labels, deep_feats, ordinal_probs, thresh_probs,
                                   "K-Means (Deep)")
        report["labels"] = labels
        all_results.append(report)

    # Strategy 2: K-Means on traditional CV features
    if strategies_to_run in ("all", "kmeans_traditional"):
        print(f"\n--- Strategy: K-Means on Traditional CV Features ---")
        labels = strategy_kmeans(trad_feats, n_clusters=3)
        report = evaluate_clusters(labels, trad_feats, ordinal_probs, thresh_probs,
                                   "K-Means (Trad CV)")
        report["labels"] = labels
        all_results.append(report)

    # Strategy 3: K-Means on hybrid features (deep + traditional + ordinal)
    if strategies_to_run in ("all", "kmeans_hybrid"):
        print(f"\n--- Strategy: K-Means on Hybrid Features ---")
        labels = strategy_kmeans(hybrid, n_clusters=3)
        report = evaluate_clusters(labels, hybrid, ordinal_probs, thresh_probs,
                                   "K-Means (Hybrid)")
        report["labels"] = labels
        all_results.append(report)

    # Strategy 4: Threshold ranking (P(>=grade2) split)
    if strategies_to_run in ("all", "threshold_ranking"):
        print(f"\n--- Strategy: CORN Threshold Ranking ---")
        labels = strategy_threshold_ranking(thresh_probs)
        report = evaluate_clusters(labels, deep_feats, ordinal_probs, thresh_probs,
                                   "Threshold Ranking")
        report["labels"] = labels
        all_results.append(report)

    # Strategy 5: Feature distance + ordinal ranking
    if strategies_to_run in ("all", "feature_ranking"):
        print(f"\n--- Strategy: Feature Distance + Ordinal Ranking ---")
        labels = strategy_feature_ranking(deep_feats, thresh_probs)
        report = evaluate_clusters(labels, deep_feats, ordinal_probs, thresh_probs,
                                   "Feature+Prob Ranking")
        report["labels"] = labels
        all_results.append(report)

    # ── Pick best strategy ──
    print(f"\n{'='*60}")
    print("Strategy Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Strategy':<30} {'Silhouette':<12} {'Separation OK?':<15}")
    print("-" * 60)

    best_result = None
    best_sil = -1
    for r in all_results:
        ok = "✓" if r["silhouette_score"] > 0.25 else "✗"
        print(f"  {r['strategy']:<30} {r['silhouette_score']:<12.4f} {ok:<15}")
        if r["silhouette_score"] > best_sil:
            best_sil = r["silhouette_score"]
            best_result = r

    print(f"\nBest strategy: {best_result['strategy']} (silhouette={best_sil:.4f})")

    # ── Save final report ──
    full_report = {
        "n_images": len(paths),
        "p_ge_grade2_stats": {
            "min": float(p_ge_grade2.min()),
            "max": float(p_ge_grade2.max()),
            "mean": float(p_ge_grade2.mean()),
            "std": float(p_ge_grade2.std()),
        },
        "all_strategies": [{k: v for k, v in r.items() if k != "labels"}
                          for r in all_results],
        "best_strategy": best_result["strategy"],
    }

    report_path = os.path.join(args.report_dir, "cluster_report_v2.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report: {report_path}")

    # ── Save assignments for best strategy ──
    csv_path = os.path.join(args.report_dir, "cluster_assignments_v2.csv")
    best_labels = best_result["labels"]
    best_ctg = best_result["cluster_to_grade"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cluster_id", "assigned_grade",
                         "p_ge_grade2", "p_ge_grade3", "p_ge_grade4"])
        for fname, cid in zip(filenames, best_labels):
            i = filenames.index(fname)
            writer.writerow([
                fname, cid, best_ctg[cid]["wagner_label"],
                round(float(thresh_probs[i, 2]), 6),
                round(float(thresh_probs[i, 3]), 6),
                round(float(thresh_probs[i, 4]), 6),
            ])
    print(f"Assignments CSV: {csv_path}")

    # ── Visualize ──
    if args.viz and len(all_results) > 0:
        # Attach p_ge_grade2 for visualization
        all_results[0]["p_ge_grade2"] = thresh_probs[:, 2]
        visualize_comparison(all_results, deep_feats, args.report_dir)

    # ── Apply split ──
    if args.mode != "dry-run":
        print(f"\n{'='*60}")
        print(f"Applying '{best_result['strategy']}' split (mode={args.mode})...")
        apply_split(best_labels, filenames, args.source, args.output_dir,
                    best_ctg, mode=args.mode)
    else:
        print(f"\n=== Dry Run — no files changed ===")
        counts = Counter()
        for cid in best_labels:
            lbl = best_ctg[cid]["wagner_label"]
            counts[lbl] += 1
        print(f"  Would assign: grade1={counts.get('grade1',0)}, "
              f"grade2={counts.get('grade2',0)}, grade3={counts.get('grade3',0)}")
        print(f"  To apply: --mode copy (or --mode move)")

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
