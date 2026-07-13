#!/usr/bin/env python3
"""
Clinical HTML report generator for DFU Wagner grading system.

Generates self-contained HTML reports (printable to PDF) — zero external dependencies.
All CSS inline, all images base64-embedded, works fully offline.

Usage:
    from report import generate_html_report
    report_path = generate_html_report(result_dict, "foot.jpg",
                                       gradcam_path="foot_gradcam.jpg",
                                       output_dir="/path/to/reports/")
"""

import base64
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

# ─── Add project root to path ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import IDX_TO_LABEL, LABEL_NAMES

# ─── Chinese labels for report display ────────────────────────────────

LABEL_NAMES_CN = {
    "normal":  "正常（健康足部）",
    "grade0":  "Wagner 0（高危足）",
    "grade1":  "Wagner 1（浅表溃疡）",
    "grade2":  "Wagner 2（深部溃疡）",
    "grade3":  "Wagner 3（深部感染）",
    "grade4":  "Wagner 4（局限性坏疽）",
    "grade5":  "Wagner 5（全足坏疽）",
}

SEVERITY_COLORS = {
    0: "#27ae60",  # normal — green
    1: "#f39c12",  # grade0 — amber
    2: "#e67e22",  # grade1 — orange
    3: "#d35400",  # grade2 — dark orange
    4: "#c0392b",  # grade3 — red
    5: "#8e44ad",  # grade4 — purple
    6: "#2c3e50",  # grade5 — dark
}

URGENCY_COLORS = {
    "无需就医":       "#27ae60",
    "门诊 1-2 周内就诊": "#f39c12",
    "门诊 3-5 天内就诊": "#e67e22",
    "门诊 24-48 小时内就诊": "#d35400",
    "急诊，立即住院":   "#c0392b",
    "急诊，24 小时内住院": "#8e44ad",
    "住院治疗":       "#2c3e50",
}


# ─── Image encoding ───────────────────────────────────────────────────

def image_to_base64(image_path: str, max_width: int = 600) -> str:
    """Load image from disk and encode as base64 JPEG data URI.

    Args:
        image_path: path to image file (JPEG/PNG/etc.)
        max_width: resize so width <= max_width (keeps aspect ratio)

    Returns:
        "data:image/jpeg;base64,..." string for inline HTML <img> src
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _prob_bar_html(label_cn: str, prob: float, is_predicted: bool,
                   color: str) -> str:
    """Generate a single probability bar row."""
    pct = prob * 100
    highlight = "prob-row-predicted" if is_predicted else ""
    return f"""
    <div class="prob-row {highlight}">
      <span class="prob-label">{label_cn}</span>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
      </div>
      <span class="prob-value">{pct:.1f}%</span>
    </div>"""


# ─── Main report generator ────────────────────────────────────────────

def generate_html_report(
    result: dict[str, Any],
    image_path: str,
    gradcam_path: str | None = None,
    output_dir: str | None = None,
    model_version: str = "corn_v2 / ConvNeXt-Tiny",
) -> str:
    """Generate a self-contained HTML clinical report.

    Args:
        result: prediction dict from inference.predict_image()
        image_path: path to the original foot image
        gradcam_path: path to Grad-CAM overlay image (optional)
        output_dir: directory to save report (auto-generates filename)
        model_version: model identifier shown in report header

    Returns:
        Absolute path to the saved HTML report file
    """
    image_path = Path(image_path)
    image_stem = image_path.stem
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Encode images
    original_b64 = image_to_base64(str(image_path))
    gradcam_b64 = image_to_base64(gradcam_path) if gradcam_path else None

    # Extract result fields
    wagner_grade: int = result["wagner_grade"]
    wagner_label: str = result["wagner_label"]
    wagner_name: str = result["wagner_name"]
    confidence: float = result["confidence"]
    is_ulcer: bool = result.get("is_ulcer", True)
    binary_prob: float | None = result.get("binary_prob_ulcer")
    ordinal_probs: dict[str, float] = result.get("ordinal_probs", {})
    recommendation: dict = result.get("recommendation", {})
    caveats: list[str] = result.get("caveats", [])
    tta_used: bool = result.get("tta_used", False)
    tta_views: int = result.get("tta_views", 0)

    # ── CSS ─────────────────────────────────────────────────────────
    css = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                   "PingFang SC", "Microsoft YaHei", "Noto Sans SC",
                   sans-serif;
      background: #f0f2f5; color: #2c3e50; line-height: 1.6;
    }
    .report { max-width: 900px; margin: 20px auto; background: #fff;
              border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
              overflow: hidden; }

    /* Header */
    .report-header {
      background: linear-gradient(135deg, #1a5276, #2980b9);
      color: #fff; padding: 28px 36px;
    }
    .report-header h1 { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
    .report-header .subtitle { font-size: 13px; opacity: 0.85; }
    .report-header .meta { margin-top: 10px; font-size: 12px; opacity: 0.7;
                           display: flex; gap: 24px; flex-wrap: wrap; }

    /* Sections */
    .section { padding: 24px 36px; border-bottom: 1px solid #e8ecf0; }
    .section:last-child { border-bottom: none; }
    .section h2 { font-size: 16px; color: #1a5276; margin-bottom: 14px;
                  padding-bottom: 6px; border-bottom: 2px solid #2980b9;
                  display: inline-block; }

    /* Images */
    .images-row { display: flex; gap: 20px; flex-wrap: wrap; }
    .image-box { flex: 1; min-width: 250px; text-align: center; }
    .image-box img { max-width: 100%; max-height: 320px; border-radius: 6px;
                     border: 1px solid #d5d8dc; }
    .image-box .caption { font-size: 12px; color: #7f8c8d; margin-top: 6px; }
    .image-placeholder { display: flex; align-items: center; justify-content: center;
                         height: 200px; background: #f7f9fa; border: 2px dashed #d5d8dc;
                         border-radius: 6px; color: #bdc3c7; font-size: 14px; }

    /* Diagnosis card */
    .diagnosis-card { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
    .grade-badge { display: flex; flex-direction: column; align-items: center;
                   justify-content: center; min-width: 140px; padding: 20px 24px;
                   border-radius: 12px; color: #fff; }
    .grade-badge .grade-label { font-size: 28px; font-weight: 800; }
    .grade-badge .grade-name { font-size: 13px; margin-top: 4px; opacity: 0.9; }
    .confidence-box { text-align: center; }
    .confidence-box .conf-number { font-size: 36px; font-weight: 700;
                                    color: #1a5276; }
    .confidence-box .conf-label { font-size: 12px; color: #7f8c8d; }
    .diag-info { font-size: 13px; color: #555; display: flex; flex-wrap: wrap;
                 gap: 16px; }
    .diag-info .info-item { display: flex; align-items: center; gap: 4px; }
    .tta-badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
                 font-size: 11px; font-weight: 600; }
    .tta-on { background: #d5f5e3; color: #1e8449; }
    .tta-off { background: #f2f3f4; color: #7f8c8d; }

    /* Probability bars */
    .prob-row { display: flex; align-items: center; gap: 10px; padding: 5px 0; }
    .prob-label { width: 160px; font-size: 13px; text-align: right;
                  flex-shrink: 0; color: #555; }
    .prob-bar-bg { flex: 1; height: 18px; background: #edf0f2; border-radius: 9px;
                   overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 9px; min-width: 2px;
                     transition: width 0.3s; }
    .prob-value { width: 50px; font-size: 12px; font-weight: 600;
                  text-align: right; color: #2c3e50; }
    .prob-row-predicted .prob-label { font-weight: 700; color: #1a5276; }
    .prob-row-predicted { background: #ebf5fb; border-radius: 6px;
                          padding: 5px 6px; margin: 2px -6px; }

    /* Recommendation */
    .rec-grid { display: grid; grid-template-columns: 1fr 1fr;
                gap: 14px 24px; font-size: 13px; }
    .rec-grid .rec-item { }
    .rec-grid .rec-item .rec-key { font-weight: 600; color: #1a5276;
                                   margin-bottom: 2px; }
    .rec-grid .rec-item .rec-val { color: #555; }
    .rec-list { margin: 6px 0 0 16px; color: #555; font-size: 13px; }
    .rec-list li { margin-bottom: 3px; }
    .urgency-badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
                     font-size: 12px; font-weight: 600; color: #fff; }

    /* Caveats */
    .caveats { background: #fef9e7; border-left: 4px solid #f1c40f;
               padding: 12px 16px; border-radius: 0 6px 6px 0; }
    .caveats .caveat { font-size: 12px; color: #7d6608; margin-bottom: 4px; }
    .caveats .caveat:last-child { margin-bottom: 0; }

    /* Footer */
    .report-footer { background: #f7f9fa; padding: 16px 36px; font-size: 11px;
                     color: #95a5a6; display: flex; justify-content: space-between;
                     flex-wrap: wrap; gap: 8px; }

    /* Print */
    @media print {
      body { background: #fff; }
      .report { box-shadow: none; margin: 0; max-width: 100%;
                border-radius: 0; }
      .report-header { background: #1a5276 !important;
                       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .grade-badge { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .prob-bar-fill { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .urgency-badge { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .caveats { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .report-footer { display: none; }
      @page { margin: 12mm; size: A4; }
    }
    """

    # ── Build HTML ──────────────────────────────────────────────────
    grade_color = SEVERITY_COLORS.get(wagner_grade, "#7f8c8d")

    # Probability bars
    prob_bars_rows = ""
    ordered_labels = ["normal", "grade0", "grade1", "grade2", "grade3", "grade4", "grade5"]
    for i, label in enumerate(ordered_labels):
        prob = ordinal_probs.get(label, 0.0)
        is_pred = (label == wagner_label)
        color = SEVERITY_COLORS.get(i, "#95a5a6")
        prob_bars_rows += _prob_bar_html(
            LABEL_NAMES_CN.get(label, label), prob, is_pred, color
        )

    # Recommendation
    urgency = recommendation.get("urgency", "请遵医嘱")
    urgency_color = URGENCY_COLORS.get(urgency, "#7f8c8d")

    medical_items = recommendation.get("medical", [])
    medical_html = "".join(f"<li>{m}</li>" for m in medical_items) if medical_items else ""

    lifestyle_items = recommendation.get("lifestyle", [])
    lifestyle_html = "".join(f"<li>{l}</li>" for l in lifestyle_items) if lifestyle_items else ""

    # Caveats
    caveats_html = ""
    if caveats:
        caveats_html = "".join(f'<div class="caveat">{c}</div>' for c in caveats)

    # TTA badge
    tta_html = ""
    if tta_used:
        tta_html = (f'<span class="tta-badge tta-on">'
                    f'TTA 已启用 ({tta_views} 视角)</span>')
    else:
        tta_html = '<span class="tta-badge tta-off">标准推理（无 TTA）</span>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DFU Wagner 分级报告 — {image_stem}</title>
<style>{css}</style>
</head>
<body>
<div class="report">

  <!-- Header -->
  <div class="report-header">
    <h1>糖尿病足 Wagner 分级 — AI 辅助评估报告</h1>
    <div class="subtitle">Diabetic Foot Ulcer Wagner Grading Report</div>
    <div class="meta">
      <span>📅 {now}</span>
      <span>🖥️ 模型：{model_version}</span>
      <span>📷 图像：{image_path.name}</span>
    </div>
  </div>

  <!-- Images -->
  <div class="section">
    <h2>影像资料</h2>
    <div class="images-row">
      <div class="image-box">
        <img src="{original_b64}" alt="原始图像">
        <div class="caption">原始足部图像</div>
      </div>
      <div class="image-box">
        {f'<img src="{gradcam_b64}" alt="Grad-CAM">'
         if gradcam_b64 else
         '<div class="image-placeholder">未生成 Grad-CAM 热力图</div>'}
        <div class="caption">Grad-CAM 热力图（模型关注区域）</div>
      </div>
    </div>
  </div>

  <!-- Diagnosis -->
  <div class="section">
    <h2>AI 诊断结论</h2>
    <div class="diagnosis-card">
      <div class="grade-badge" style="background:{grade_color}">
        <div class="grade-label">Wagner {wagner_grade}</div>
        <div class="grade-name">{LABEL_NAMES_CN.get(wagner_label, wagner_name)}</div>
      </div>
      <div class="confidence-box">
        <div class="conf-number">{confidence:.1%}</div>
        <div class="conf-label">模型置信度</div>
      </div>
      <div class="diag-info">
        <div>{tta_html}</div>
        <div class="info-item">
          筛查结果：{'⚠️ 溃疡' if is_ulcer else '✅ 良性'}
        </div>
        {f'<div class="info-item">P(溃疡) = {binary_prob:.3f}</div>'
         if binary_prob is not None else ''}
      </div>
    </div>
  </div>

  <!-- Probabilities -->
  <div class="section">
    <h2>各类别预测概率</h2>
    {prob_bars_rows}
  </div>

  <!-- Recommendations -->
  <div class="section">
    <h2>临床建议</h2>
    <div class="rec-grid">
      <div class="rec-item">
        <div class="rec-key">诊断摘要</div>
        <div class="rec-val">{recommendation.get('summary', '—')}</div>
      </div>
      <div class="rec-item">
        <div class="rec-key">紧急程度</div>
        <span class="urgency-badge" style="background:{urgency_color}">{urgency}</span>
      </div>
      <div class="rec-item">
        <div class="rec-key">医疗措施</div>
        {f'<ul class="rec-list">{medical_html}</ul>' if medical_html else '<div class="rec-val">请遵医嘱</div>'}
      </div>
      <div class="rec-item">
        <div class="rec-key">生活建议</div>
        {f'<ul class="rec-list">{lifestyle_html}</ul>' if lifestyle_html else '<div class="rec-val">请遵医嘱</div>'}
      </div>
      <div class="rec-item">
        <div class="rec-key">随访建议</div>
        <div class="rec-val">{recommendation.get('follow_up', '请遵医嘱')}</div>
      </div>
    </div>
  </div>

  <!-- Caveats -->
  {f'''
  <div class="section">
    <h2>⚠️ 注意事项与免责声明</h2>
    <div class="caveats">{caveats_html}</div>
  </div>''' if caveats_html else ''}

  <!-- Footer -->
  <div class="report-footer">
    <span>本报告由 DFU Wagner 分级系统 v1.0 自动生成 | AI 辅助评估，最终诊断以医师判断为准</span>
    <span>Generated: {now}</span>
  </div>

</div>
</body>
</html>"""

    # ── Save ────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = Path("/root/dfu/models/corn_v2/reports")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_stem}_report.html"
    output_path.write_text(html, encoding="utf-8")

    return str(output_path.resolve())


# ─── CLI (standalone) ────────────────────────────────────────────────

def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Generate DFU clinical HTML report from inference result"
    )
    parser.add_argument("--result", type=str, required=True,
                        help="Path to JSON inference result (from inference.py --json)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to original image file")
    parser.add_argument("--gradcam", type=str, default=None,
                        help="Path to Grad-CAM overlay image")
    parser.add_argument("--output-dir", type=str,
                        default="/root/dfu/models/corn_v2/reports",
                        help="Output directory for HTML report")
    parser.add_argument("--model-version", type=str,
                        default="corn_v2 / ConvNeXt-Tiny",
                        help="Model version string for report header")
    args = parser.parse_args()

    # Load result JSON (supports both single dict and list)
    with open(args.result, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # Batch: generate one report per result
        for r in data:
            if "error" in r:
                print(f"  ⚠️ Skipping {r['file']} — inference error: {r['error']}")
                continue
            img_path = Path(args.image).parent / r.get("file", "unknown.jpg")
            if not img_path.exists():
                img_path = args.image  # fallback to --image arg
            path = generate_html_report(
                r, str(img_path),
                gradcam_path=r.get("gradcam_path"),
                output_dir=args.output_dir,
                model_version=args.model_version,
            )
            print(f"  ✅ {path}")
    else:
        path = generate_html_report(
            data, args.image,
            gradcam_path=args.gradcam,
            output_dir=args.output_dir,
            model_version=args.model_version,
        )
        print(f"✅ Report saved to: {path}")


if __name__ == "__main__":
    main()
