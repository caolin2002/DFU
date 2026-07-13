#!/usr/bin/env python3
"""
R3: 离线数据增广 — DFU 稀有类别定向扩增
=========================================

策略：
  - 只对 train/ 做增广，val/test 不动（杜绝数据泄漏）
  - grade0: 223 → ~2,000（每图 8 变体）
  - grade4:   1 →  200（单图 199 变体，极端增广）
  - normal/grade1 数量充足，不做离线增广
  - grade2/3/5 为空占位，不做增广

三条增广管线（全部基于 torchvision.transforms，零额外依赖）：
  轻度（30%）: 小角度旋转 + 轻微颜色抖动 + 水平翻转
  中度（50%）: 中角度旋转 + 颜色抖动 + 缩放 + 模糊 + 水平翻转
  重度（20%）: 大角度旋转 + 强颜色抖动 + 透视变换 + 仿射 + 锐化 + 翻转

文件命名：
  原图:  DM001_M_L.png
  变体:  DM001_M_L_aug001.jpg  ...  DM001_M_L_aug199.jpg
  get_original_id() 自动剥离 _augNNN 后缀
"""

import os
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

# ─── 配置 ──────────────────────────────────────────────────────────────
PROCESSED = Path("/root/dfu/data/processed")
TRAIN_DIR = PROCESSED / "train"
RANDOM_SEED = 42

# 需要增广的类别及其参数
AUG_CONFIG = {
    "grade0": {
        "variants_per_image": 8,
        "light_ratio": 0.3,
        "medium_ratio": 0.5,
        "heavy_ratio": 0.2,
    },
    "grade4": {
        "variants_per_image": 199,
        "light_ratio": 0.1,
        "medium_ratio": 0.5,
        "heavy_ratio": 0.4,  # 更激进的重度比例
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# ─── 三条增广管线 ─────────────────────────────────────────────────────

def build_light_pipeline() -> T.Compose:
    """轻度增广：小角度旋转 + 轻微颜色 + 水平翻转"""
    return T.Compose([
        T.RandomRotation(degrees=10, fill=0),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.RandomHorizontalFlip(p=0.5),
    ])


def build_medium_pipeline() -> T.Compose:
    """中度增广：中角度旋转 + 颜色 + 缩放 + 模糊 + 水平翻转"""
    return T.Compose([
        T.RandomRotation(degrees=25, fill=0),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=0, scale=(0.9, 1.1), fill=0),
        T.GaussianBlur(kernel_size=3),
    ])


def build_heavy_pipeline() -> T.Compose:
    """重度增广：大角度旋转 + 强颜色 + 透视 + 仿射 + 模糊 + 锐化 + 翻转"""
    return T.Compose([
        T.RandomRotation(degrees=45, fill=0),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
        T.RandomAffine(degrees=0, scale=(0.85, 1.15), shear=10, fill=0),
        T.GaussianBlur(kernel_size=5),
        T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
    ])


# ─── 主逻辑 ────────────────────────────────────────────────────────────

def collect_originals(class_dir: Path) -> list[Path]:
    """
    收集目录下所有原图（排除已有的增广变体）。
    通过文件名判断：带 _augNNN 后缀的是增广变体，其余为原图。
    """
    originals = []
    for fpath in sorted(class_dir.iterdir()):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in IMG_EXTS:
            continue
        # 排除已存在的增广变体（容错：脚本已运行过的情况）
        stem = fpath.stem
        if __import__('re').search(r'_aug\d{3,}$', stem):
            continue
        originals.append(fpath)
    return originals


def generate_variants(
    img: Image.Image,
    num_variants: int,
    light_ratio: float,
    medium_ratio: float,
    heavy_ratio: float,
) -> list[Image.Image]:
    """
    为一张原图生成 num_variants 个增广变体。
    按比例分配到轻/中/重三条管线。
    """
    # 计算各级别数量
    n_light = round(num_variants * light_ratio)
    n_medium = round(num_variants * medium_ratio)
    n_heavy = num_variants - n_light - n_medium

    pipelines = (
        [build_light_pipeline()] * n_light
        + [build_medium_pipeline()] * n_medium
        + [build_heavy_pipeline()] * n_heavy
    )
    random.shuffle(pipelines)  # 打乱顺序，避免同级别连续

    variants = []
    for pipe in pipelines:
        aug_img = pipe(img)  # PIL → Tensor after ToTensor in pipeline
        variants.append(aug_img)
    return variants


def augment_class(class_name: str, config: dict):
    """对单个类别的训练数据进行增广。"""
    class_dir = TRAIN_DIR / class_name
    if not class_dir.exists():
        print(f"  ⚠️  {class_dir} 不存在，跳过")
        return

    originals = collect_originals(class_dir)
    n_originals = len(originals)
    n_per = config["variants_per_image"]

    if n_originals == 0:
        print(f"  ⚠️  {class_name} 无原图，跳过")
        return

    total_new = n_originals * n_per
    print(f"\n  {class_name}: {n_originals} 张原图 × {n_per} = {total_new} 变体")
    print(f"    轻:中:重 = {config['light_ratio']:.0%}:{config['medium_ratio']:.0%}:{config['heavy_ratio']:.0%}")

    for i, fpath in enumerate(tqdm(originals, desc=f"  增广 {class_name}")):
        stem = fpath.stem
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"    ✗ 无法读取 {fpath.name}: {e}")
            continue

        variants = generate_variants(
            img,
            num_variants=n_per,
            light_ratio=config["light_ratio"],
            medium_ratio=config["medium_ratio"],
            heavy_ratio=config["heavy_ratio"],
        )

        for j, variant in enumerate(variants):
            aug_name = f"{stem}_aug{i * n_per + j + 1:04d}.jpg"
            aug_path = class_dir / aug_name
            variant.save(aug_path, quality=92)

    final_count = len(list(class_dir.iterdir()))
    print(f"    ✅ 完成，目录现有 {final_count} 个文件")


def main():
    print("=" * 60)
    print("R3: 离线数据增广 — DFU 稀有类别定向扩增")
    print("=" * 60)
    print(f"目标目录: {TRAIN_DIR}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 打印增广前分布
    print("\n📊 增广前训练集分布:")
    for class_name in ["normal", "grade0", "grade1", "grade2", "grade3", "grade4", "grade5"]:
        d = TRAIN_DIR / class_name
        count = len(list(d.iterdir())) if d.exists() else 0
        print(f"  {class_name}: {count}")

    # 执行增广
    for class_name, config in AUG_CONFIG.items():
        augment_class(class_name, config)

    # 打印增广后分布
    print("\n" + "=" * 60)
    print("📊 增广后训练集分布:")
    total = 0
    for class_name in ["normal", "grade0", "grade1", "grade2", "grade3", "grade4", "grade5"]:
        d = TRAIN_DIR / class_name
        count = len(list(d.iterdir())) if d.exists() else 0
        total += count
        marker = ""
        if class_name in AUG_CONFIG:
            marker = " ← R3 增广"
        elif count > 0 and class_name not in AUG_CONFIG:
            marker = " (未变动)"
        elif count == 0 and class_name in ("grade2", "grade3", "grade5"):
            marker = " (空占位)"
        print(f"  {class_name}: {count}{marker}")
    print(f"  总计: {total}")
    print(f"\n  val/ 和 test/ 未做任何增广 ✅")
    print("=" * 60)
    print("✅ R3 离线增广完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
