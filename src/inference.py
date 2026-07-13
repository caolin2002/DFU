#!/usr/bin/env python3
"""
Two-stage inference pipeline for DFU Wagner 0-5 grading.

Stage 1: Binary screening → benign (normal/grade0) or ulcer (grade1-5)
Stage 2: CORN ordinal regression → Wagner 0-5 detailed grade

Usage:
    # Single image
    python src/inference.py --image path/to/foot.jpg

    # Directory batch
    python src/inference.py --dir path/to/images/ --output results.json

    # JSON format output
    python src/inference.py --image path.jpg --json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

# ─── Add project root to path ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import IDX_TO_LABEL, LABEL_NAMES, LABEL_TO_IDX, NUM_CLASSES
from model import get_convnext_tiny, get_efficientnet_b0, get_resnet50
from tta import tta_predict_ordinal
from gradcam import GradCAM, denormalize, overlay_heatmap
from report import generate_html_report

# ─── ImageNet stats ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Clinical mappings ────────────────────────────────────────────────
BENIGN_LABELS = {0: "normal", 1: "grade0"}
ULCER_LABELS = {2: "grade1", 3: "grade2", 4: "grade3", 5: "grade4", 6: "grade5"}

# Clinical recommendations per Wagner grade
RECOMMENDATIONS = {
    "normal": {
        "summary": "健康足部，无 DFU 风险",
        "medical": ["常规足部护理", "保持足部卫生"],
        "lifestyle": ["每日足部自检", "穿合适鞋袜"],
        "urgency": "无需就医",
        "follow_up": "年度足部检查",
    },
    "grade0": {
        "summary": "Wagner 0 高危足 — 皮肤完整，无溃疡，但存在高危因素",
        "medical": [
            "建议前往足病专科做 Semmes-Weinstein 单丝检查",
            "振动觉阈值检测 + ABI（踝肱指数）",
            "评估足部畸形是否需要矫形鞋/鞋垫",
        ],
        "lifestyle": [
            "每日仔细检查足部（包括足底、趾间）",
            "温水洗脚，擦干后涂润肤霜（避开趾缝）",
            "禁止赤脚行走",
            "严格控制血糖（HbA1c < 7%）",
        ],
        "urgency": "门诊 1-2 周内就诊",
        "follow_up": "每 3 个月足病专科随访",
    },
    "grade1": {
        "summary": "Wagner 1 浅表溃疡 — 仅累及表皮/真皮，无感染征象",
        "medical": [
            "局部清创 + 无菌敷料覆盖",
            "如 1 周无好转，考虑培养 + 敏感抗生素",
            "评估是否需要减压支具（TCC / 减压鞋）",
        ],
        "lifestyle": [
            "严格避免患足负重",
            "抬高患足以减轻水肿",
            "严格控制血糖和血压",
        ],
        "urgency": "门诊 3-5 天内就诊",
        "follow_up": "每 1-2 周换药评估",
    },
    "grade2": {
        "summary": "Wagner 2 深部溃疡 — 累及皮下/筋膜/肌肉，伴软组织感染",
        "medical": [
            "彻底清创 + 分泌物培养 + 敏感抗生素",
            "X-ray 排除骨髓炎",
            "评估是否需要住院治疗",
            "多学科会诊（足病 + 感染 + 血管外科）",
        ],
        "lifestyle": [
            "严格卧床休息，患足完全免负重",
            "严格控制血糖",
        ],
        "urgency": "门诊 24-48 小时内就诊，必要时住院",
        "follow_up": "每周清创评估",
    },
    "grade3": {
        "summary": "Wagner 3 深部感染 — 肌腱/骨/关节受累，骨髓炎或深部脓肿",
        "medical": [
            "紧急住院治疗",
            "深部组织培养 + 静脉抗生素",
            "MRI 确认骨髓炎范围",
            "外科清创 ± 部分截肢",
            "血管评估 ± 血运重建",
        ],
        "lifestyle": [
            "住院治疗，严格卧床",
        ],
        "urgency": "急诊，立即住院",
        "follow_up": "住院期间每日评估",
    },
    "grade4": {
        "summary": "Wagner 4 局限性坏疽 — 足趾/足跟/前足局部坏死",
        "medical": [
            "紧急住院",
            "血管外科急会诊 ± 紧急血运重建",
            "外科清创/截肢 + 术中培养",
            "静脉抗生素",
            "评估截肢平面（尽量保留负重功能）",
        ],
        "lifestyle": [
            "住院治疗",
        ],
        "urgency": "急诊，24 小时内住院",
        "follow_up": "术后每日伤口评估",
    },
    "grade5": {
        "summary": "Wagner 5 全足坏疽 — 踝关节及小腿受累",
        "medical": [
            "紧急住院，多学科急会诊",
            "大截肢（膝下/膝上）评估",
            "围手术期静脉抗生素",
            "术后康复 + 假肢适配",
            "对侧足部严密保护",
        ],
        "lifestyle": [
            "住院治疗 + 术后康复",
        ],
        "urgency": "急诊，立即住院",
        "follow_up": "术后康复计划 + 假肢门诊",
    },
}


# ─── Image preprocessing ──────────────────────────────────────────────

def get_transform(input_size: int = 224):
    """Inference transform — resize + normalize (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Model loading ────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """
    Load trained DFU model from checkpoint.
    Supports both full checkpoint and weights-only formats.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        # Fallback: read from config.yaml
        with open(PROJECT_ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    binary_head = cfg["model"].get("binary_head", True)

    # Build model
    if model_name == "convnext_tiny":
        model = get_convnext_tiny(num_classes=num_classes, binary=binary_head)
    elif model_name == "resnet50":
        model = get_resnet50(num_classes=num_classes, binary=binary_head)
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_b0(num_classes=num_classes, binary=binary_head)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, cfg


# ─── Single-image inference ───────────────────────────────────────────

@torch.no_grad()
def predict_image(model, transform, image_path: str, device: torch.device,
                  binary_thresh: float = 0.5,
                  use_tta: bool = False, tta_views: int = 4,
                  gradcam: bool = False, gradcam_dir: str | None = None):
    """
    Run two-stage inference on a single image.

    Args:
        model: DFUModel
        transform: torchvision transform (Resize + ToTensor + Normalize)
        image_path: path to image file
        device: torch device
        binary_thresh: threshold for binary ulcer screening
        use_tta: enable Test-Time Augmentation (averages predictions across views)
        tta_views: number of TTA views (4-6, only used when use_tta=True)
        gradcam: generate Grad-CAM heatmap overlay
        gradcam_dir: directory to save Grad-CAM image (default: alongside report)

    Returns a dict with:
        - wagner_grade: integer 0-6
        - wagner_label: "normal" | "grade0" | ... | "grade5"
        - wagner_name: human-readable name
        - is_ulcer: bool
        - binary_prob: P(ulcer)
        - ordinal_probs: per-class probabilities [7]
        - confidence: max class probability
        - recommendation: clinical advice
        - caveats: data limitations
        - tta_used: bool
        - tta_views: int
        - gradcam_path: str or None
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    features = model.backbone(tensor)

    # Stage 1: Binary screening
    if model.has_binary:
        bin_logit = model.binary_head(features)
        bin_prob = torch.sigmoid(bin_logit).item()
        is_ulcer = bin_prob >= binary_thresh
    else:
        bin_prob = None
        is_ulcer = True

    # Stage 2: CORN ordinal grading (with optional TTA)
    if use_tta:
        tta_pred, tta_probs = tta_predict_ordinal(
            model, tensor.squeeze(0), device, n_views=tta_views
        )
        wagner_grade = tta_pred
        class_probs = tta_probs
        confidence = class_probs[wagner_grade].item()
    else:
        ord_logits = model.ordinal_head(features)
        wagner_grade = model.ordinal_head.predict(ord_logits).item()
        class_probs = model.ordinal_head.predict_proba(ord_logits).squeeze(0)
        confidence = class_probs[wagner_grade].item()

    wagner_label = IDX_TO_LABEL[wagner_grade]
    wagner_name = LABEL_NAMES[wagner_label]

    # Recommendations
    rec = RECOMMENDATIONS.get(wagner_label, RECOMMENDATIONS["normal"])

    # Caveats based on data limitations
    caveats = ["本报告为 AI 辅助评估，最终诊断请遵医嘱"]
    if wagner_label in ("grade2", "grade3", "grade5"):
        caveats.append(
            f"⚠️ {wagner_name} 当前训练数据为空占位，分级结果基于模型泛化，仅供参考"
        )
    if wagner_label == "grade4":
        caveats.append(
            "⚠️ 坏疽分类训练数据极少（1 张原图），分级置信度有限，建议结合影像学确认"
        )
    if wagner_label == "grade0":
        caveats.append(
            "⚠️ Wagner 0 高危足无法仅靠图像确诊，建议做神经/血管客观检查（单丝 + ABI）"
        )

    # Grad-CAM (needs gradients — must re-enable inside no_grad context)
    gradcam_path = None
    if gradcam:
        cam_extractor = GradCAM(model)
        try:
            with torch.enable_grad():
                heatmap, pred_cls, cam_conf = cam_extractor(
                    tensor, target_class=wagner_grade
                )
            img_np = denormalize(tensor.squeeze(0).cpu())
            overlaid = overlay_heatmap(img_np, heatmap)

            if gradcam_dir is None:
                gradcam_dir = os.path.join(
                    os.path.dirname(image_path), "gradcam"
                )
            os.makedirs(gradcam_dir, exist_ok=True)
            stem = Path(image_path).stem
            gradcam_path = os.path.join(gradcam_dir, f"{stem}_gradcam.jpg")
            Image.fromarray(overlaid).save(gradcam_path)
        finally:
            cam_extractor.remove_hooks()

    return {
        "wagner_grade": wagner_grade,
        "wagner_label": wagner_label,
        "wagner_name": wagner_name,
        "is_ulcer": is_ulcer,
        "binary_prob_ulcer": round(bin_prob, 4) if bin_prob is not None else None,
        "ordinal_probs": {IDX_TO_LABEL[i]: round(class_probs[i].item(), 4)
                          for i in range(len(class_probs))},
        "confidence": round(confidence, 4),
        "recommendation": rec,
        "caveats": caveats,
        "tta_used": use_tta,
        "tta_views": tta_views if use_tta else 0,
        "gradcam_path": gradcam_path,
    }


# ─── Batch inference ──────────────────────────────────────────────────

@torch.no_grad()
def predict_directory(model, transform, dir_path: str, device: torch.device,
                      binary_thresh: float = 0.5,
                      use_tta: bool = False, tta_views: int = 4,
                      gradcam: bool = False, gradcam_dir: str | None = None):
    """Run inference on all images in a directory."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    results = []

    paths = sorted([
        p for p in Path(dir_path).iterdir()
        if p.suffix.lower() in img_exts
    ])

    for img_path in paths:
        try:
            result = predict_image(
                model, transform, str(img_path), device, binary_thresh,
                use_tta=use_tta, tta_views=tta_views,
                gradcam=gradcam, gradcam_dir=gradcam_dir,
            )
            result["file"] = img_path.name
            results.append(result)
        except Exception as e:
            results.append({
                "file": img_path.name,
                "error": str(e),
            })

    return results


# ─── Formatting ───────────────────────────────────────────────────────

def format_result(result: dict) -> str:
    """Pretty-print a single inference result."""
    if "error" in result:
        return f"✗ {result['file']}: ERROR — {result['error']}"

    tta_info = ""
    if result.get("tta_used"):
        tta_info = f" (TTA ×{result.get('tta_views', 0)} views)"

    lines = [
        "=" * 64,
        f"File:        {result.get('file', 'N/A')}",
        f"Prediction:  {result['wagner_name']} (Wagner grade {result['wagner_grade']}){tta_info}",
        f"Confidence:  {result['confidence']:.2%}",
        f"Is Ulcer:    {'Yes ⚠️' if result['is_ulcer'] else 'No ✅'}",
    ]
    if result["binary_prob_ulcer"] is not None:
        lines.append(f"P(ulcer):    {result['binary_prob_ulcer']:.4f}")

    if result.get("gradcam_path"):
        lines.append(f"Grad-CAM:    {result['gradcam_path']}")

    lines.append("\nPer-Class Probabilities:")
    for label, prob in result["ordinal_probs"].items():
        bar = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))
        lines.append(f"  {label:<8}: {prob:.4f} {bar}")

    lines.append(f"\n📋 Recommendation:")
    lines.append(f"  {result['recommendation']['summary']}")
    lines.append(f"  Urgency: {result['recommendation']['urgency']}")
    lines.append(f"  Medical: {'; '.join(result['recommendation']['medical'])}")

    for caveat in result["caveats"]:
        lines.append(f"\n  {caveat}")

    lines.append("=" * 64)
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DFU Wagner 0-5 Two-Stage Inference Pipeline"
    )
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "models/best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--json", action="store_true",
                        help="Output in JSON format (single image)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binary classification threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    # TTA
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test-Time Augmentation (multi-view averaging)")
    parser.add_argument("--tta-views", type=int, default=4,
                        help="Number of TTA views (default: 4, max: 6)")

    # Grad-CAM
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM heatmap overlay")
    parser.add_argument("--gradcam-dir", type=str, default=None,
                        help="Directory to save Grad-CAM images")

    # Report
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML clinical report")
    parser.add_argument("--report-dir", type=str,
                        default="/root/dfu/models/corn_v2/reports",
                        help="Directory to save HTML reports")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, device)
    print(f"  Backbone: {cfg['model']['name']}")
    print(f"  Classes:  {cfg['model']['num_classes']}")
    print(f"  Binary head: {cfg['model'].get('binary_head', True)}")
    if args.tta:
        print(f"  TTA: enabled ({args.tta_views} views)")
    if args.gradcam:
        print(f"  Grad-CAM: enabled")

    transform = get_transform(cfg["data"]["input_size"])

    if args.image:
        result = predict_image(
            model, transform, args.image, device, args.threshold,
            use_tta=args.tta, tta_views=args.tta_views,
            gradcam=args.gradcam, gradcam_dir=args.gradcam_dir,
        )
        result["file"] = Path(args.image).name

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_result(result))

        if args.output:
            with open(args.output, "w") as f:
                json.dump([result], f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {args.output}")

        # Generate HTML report
        if args.report:
            report_path = generate_html_report(
                result, args.image,
                gradcam_path=result.get("gradcam_path"),
                output_dir=args.report_dir,
            )
            print(f"\n📄 Report saved to: {report_path}")

    elif args.dir:
        results = predict_directory(
            model, transform, args.dir, device, args.threshold,
            use_tta=args.tta, tta_views=args.tta_views,
            gradcam=args.gradcam, gradcam_dir=args.gradcam_dir,
        )

        # Summary
        n_success = sum(1 for r in results if "error" not in r)
        n_error = len(results) - n_success
        n_ulcer = sum(1 for r in results if r.get("is_ulcer", False))
        n_benign = n_success - n_ulcer

        print(f"\nBatch Summary: {n_success} success, {n_error} error")
        print(f"  Benign: {n_benign}  |  Ulcer: {n_ulcer}")
        if args.tta:
            print(f"  TTA: enabled ({args.tta_views} views)")

        for r in results:
            if "error" not in r:
                tta_tag = " [TTA]" if r.get("tta_used") else ""
                print(f"  {r['file']:<40} → {r['wagner_label']:<8} "
                      f"(conf={r['confidence']:.2%}){tta_tag}")
            else:
                print(f"  {r['file']:<40} → ERROR: {r['error']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {args.output}")

        # Generate reports for batch
        if args.report:
            report_count = 0
            for r in results:
                if "error" in r:
                    continue
                img_path = Path(args.dir) / r.get("file", "unknown.jpg")
                if not img_path.exists():
                    continue
                try:
                    rp = generate_html_report(
                        r, str(img_path),
                        gradcam_path=r.get("gradcam_path"),
                        output_dir=args.report_dir,
                    )
                    report_count += 1
                except Exception as e:
                    print(f"  ⚠️ Report failed for {r['file']}: {e}")
            print(f"\n📄 {report_count} reports saved to: {args.report_dir}")


if __name__ == "__main__":
    main()
