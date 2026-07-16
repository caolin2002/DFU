# DFU 糖尿病足 Wagner 0-5 智能分级系统 — 实现计划书 v3

> **当前状态**: v4 已完成（2026-07-16）。实际实现以 CORN 有序回归 + 两阶段推理为主线，与下述原始计划有较大差异。详见 [docs/DFU_技术架构总览.md](docs/DFU_技术架构总览.md)。

## 一、项目目标

构建 Wagner 0-5 六级糖尿病足智能分级系统：

| 级别 | 临床定义 | 核心特征 |
|:--|:--|:--|
| **正常** | 健康足部 | 无任何异常 |
| **Grade 0** | 高危足（无开放性溃疡） | 胼胝、畸形、干燥皲裂、趾甲病变、发凉发麻 |
| **Grade 1** | 浅表溃疡 | 仅表皮/真皮，无感染 |
| **Grade 2** | 深部溃疡 | 皮下/筋膜/肌肉，软组织感染，无骨受累 |
| **Grade 3** | 深部感染 | 肌腱/骨/关节受累，骨髓炎或深部脓肿 |
| **Grade 4** | 局限性坏疽 | 足趾/足跟/前足局部坏死 |
| **Grade 5** | 全足坏疽 | 踝关节及小腿受累 |

### 分级策略

```
          ┌──────────────────────────────┐
Image ──→ │ Stage-1: Binary (正常/异常)   │
          └──────────┬───────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
      正常                      异常
      结束                  ┌──────────┐
                            │ Stage-2  │
                            │ Wagner   │
                            │ 多任务分级│
                            └──────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                Grade 0    Grade 1-3    Grade 4-5
                多模态       图像分级      图像分级
                (图像+文字)   (主目标≥80%)  (声明准确性限制)
```

### Grade 0 特殊处理

Grade 0 高危足皮肤完整，无法仅靠图像确诊 → 多模态方案：
- 输入：足部图片 + 用户填写的症状文字（发凉 Y/N、发麻 Y/N、茧子 Y/N 等）
- 输出："疑诊 Grade 0（高危足），建议前往医院做 Semmes-Weinstein 单丝 + 振动觉 + ABI 检查"

### 准确率目标

| 类别 | 目标 | 策略 |
|:--|:--:|:--|
| 二分类 (Normal/Abnormal) | ≥92% | 全量数据训练 |
| Grade 1-3 | **≥80%** 🎯 | 多任务 + 医学预训练 + 强增强 |
| Grade 0 | ≥70% | 多模态，声明"疑诊" |
| Grade 4-5 | 独立报告 | 声明"参考分级，需结合影像学" |

---

## 二、分工

| 角色 | 职责 |
|:--|:--|
| **AI (Claude)** | 全部代码实现：数据下载、自动标注、增强管线、模型架构、训练、评估、系统集成 |
| **用户** | 最终验证：确认 Grade 1-3 准确率 ≥80%，审查输出质量 |

> 全流程零人工标注。数据重标注使用规则映射 + 弱标签自动推导。

---

## 三、数据获取

### 3.1 现有数据

| 来源 | 内容 | 数量 |
|:--|:--|:--|
| ADPM V3.3 | Grade 1-4 伤口（已废弃预增强，仅用原图） | ~1,373 原图 |
| Kaggle Laithjj DFU | 二分类 健康/溃疡 | 1,055 张 |

### 3.2 需获取数据

| 编号 | 任务 | 目标 |
|:--|:--|:--|
| D1 | 正常足部图片 | ≥500 张（多角度、多肤色） |
| D2 | Grade 0 高危足图片 | ≥300 张 |
| D3 | Kaggle Laithjj DFU 下载 | 1,055 张 |
| D4 | Grade 4/5 坏疽图片搜索 | 尽可能多 |

### 3.3 自动标注管线

```
ADPM Grade 1 ──→ Wagner 1   (直接映射)
ADPM Grade 2 ──→ Wagner 2   (直接映射)
ADPM Grade 3 ──→ Wagner 3   (直接映射)
ADPM Grade 4 ──→ Wagner 4/5 (GPT-4V 零样本分类：局限性 vs 全足坏疽)
```

子任务标签自动推导：
```
Wagner → 深度             → 感染       → 缺血
─────────────────────────────────────────────
W0     → 无                → 无         → 可能 (视PAD)
W1     → 表皮/真皮         → 无/轻      → 无
W2     → 皮下/筋膜         → 中         → 可能
W3     → 骨/关节           → 重         → 可能
W4     → 全层              → 重         → 是
W5     → 全层              → 重         → 是
```

---

## 四、数据增强体系

### 完全废弃 Roboflow 预增强

原 ADPM 数据集 1,373 张 → Roboflow 固定增强 → 10,062 张（严重过拟合来源）。
改为：仅用原图 + 在线强增强，每个 epoch 生成不同变体。

### 增强方案

| 方法 | 参数 | 作用 |
|:--|:--|:--|
| RandAugment | magnitude=15, num_ops=3 | 基线增强，替代固定几何变换 |
| MixUp | α=0.2 | 样本间插值，平滑决策边界 |
| CutMix | α=1.0 | 区域替换，提升局部特征识别 |
| RandomErasing | p=0.3 | 模拟伤口被敷料/坏死组织遮挡 |
| AugMix | width=3, depth=-1 | 多链混合，域泛化 |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1 | 光照/设备差异鲁棒 |

### 数据划分

按原始图片 ID 分组后 70/15/15 分层划分（已有代码），确保同一伤口的不同增强不出现在多个集合中。

---

## 五、模型架构

### 5.1 Backbone 选型

| 候选 | 预训练 | 优势 |
|:--|:--|:--|
| ResNet-50 | ImageNet | 文献验证充分，基线可靠 |
| MedImageInsight | 医学影像 | 专为医学图像预训练 |
| RAD-DINO | 医学影像 | 放射学视觉特征 |
| BioMedCLIP | 医学图文 | 图文对齐，适合多模态 |

> 最终选择：对比实验后确定（优先测试 MedImageInsight vs ResNet-50 ImageNet）

### 5.2 Stage-1：二分类模型

```
Input (224×224×3)
    ↓
[Shared Backbone] (同 Stage-2)
    ↓
[GAP] → [FC 256] → [ReLU] → [Dropout 0.3] → [FC 2]
    ↓
Normal / Abnormal
```

### 5.3 Stage-2：Wagner 多任务分级模型

```
                         ┌─→ [Head 1] Wagner 0-5 (6类)    loss_weight=1.0  ← 主任务
                        │
Input → [Shared Backbone]─┼─→ [Head 2] 感染 (4类)         loss_weight=0.3  ← 辅助
                        │    无/轻/中/重
                        │
                        ├─→ [Head 3] 缺血 (2类)           loss_weight=0.2  ← 辅助
                        │    无/有
                        │
                        └─→ [Head 4] 深度 (4类)           loss_weight=0.3  ← 辅助
                             无/表皮-真皮/皮下-筋膜/骨-关节
```

每个 Head 结构：`GAP → FC 128 → ReLU → Dropout 0.2 → FC (num_class)`

总损失：`L = L_wagner + 0.3×L_infection + 0.2×L_ischemia + 0.3×L_depth`

多任务学习提升机制：
- 子任务强制 backbone 学习感染(红肿/脓液)、缺血(坏死/苍白)、深度(层次)特征
- 各任务梯度相互制约，减少虚假相关 → 降低过拟合
- 每个样本贡献 4 倍标注信息，数据效率更高

### 5.4 Grade 0 多模态模块

```
图像特征 (2048-d) ─┐
                    ├─→ [Concat] → [FC → FC] → Grade 0 Score
症状编码 (n-d)  ───┘
```

症状输入：用户勾选/填写的结构化文本（发凉/发麻/畸形/茧子/趾甲异常等）→ 简单 MLP 编码。

---

## 六、训练策略

### 6.1 Stage-1（二分类）

| 参数 | 值 |
|:--|:--|
| Epochs | 30 |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T0=10) |
| Loss | CrossEntropy |
| AMP | ✅ |

### 6.2 Stage-2（多任务分级）

| 参数 | 值 |
|:--|:--|
| Epochs | 80 |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T0=15, Tmult=2) |
| Backbone LR | 1e-5（冻结 backbone 5 epoch warmup 后解冻） |
| Loss (主) | Focal Loss (γ=2) 或 Label Smoothing CE (α=0.1) |
| Loss (辅) | CrossEntropy × 权重系数 |
| AMP | ✅ |
| Early Stopping | patience=20, monitor=val_wagner_acc |
| Weight Decay | 1e-4 |

### 6.3 长尾处理（Grade 4/5）

| 方法 | 说明 |
|:--|:--|
| Class-Balanced Loss (β=0.999) | 按有效样本数重加权 |
| 少数类过采样 | 训练时重复采样 Grade 4/5 |
| 合成数据 | Stable Diffusion/Medfusion 生成坏疽变体（可选） |

---

## 七、评估方案

### 7.1 主要指标

| 指标 | 用途 |
|:--|:--|
| Per-class Accuracy | 各级别独立评估 |
| Macro F1-Score | 类别不均衡下的综合指标 |
| Cohen's Kappa | 与临床评估一致性 |
| Confusion Matrix | 邻级混淆分析 |

### 7.2 可解释性

| 方法 | 目的 |
|:--|:--|
| Grad-CAM | 验证关注区域是否符合临床（溃疡灶 vs 背景） |
| 混淆矩阵分析 | 识别最易混淆的级别对（Grade 2↔3, W4↔W5） |
| 消融实验 | 逐个去掉子任务 Head，量化多任务增益 |

### 7.3 Grade 4/5 特殊评估

单独报告 ± 置信度区间，标注"建议结合影像学（X-ray/MRI）确认骨/关节受累范围"。

---

## 八、系统集成

### 8.1 推理 Pipeline

```
Input Image (+ Text for Grade 0)
    │
    ▼
[Stage-1: Binary] ──→ Normal ──→ "健康足部" → 结束
    │
    ▼ Abnormal
[Stage-2: Multi-Task]
    │
    ├── Wagner Grade (0-5)
    ├── 感染程度
    ├── 缺血评估
    └── 深度评估
    │
    ▼
[建议生成] → JSON 报告
```

### 8.2 报告格式

```json
{
  "prediction": "Grade 2",
  "confidence": 0.87,
  "binary_result": "Abnormal",
  "sub_tasks": {
    "infection": {"level": "moderate", "confidence": 0.72},
    "ischemia": {"present": false, "confidence": 0.91},
    "depth": {"level": "subcutaneous", "confidence": 0.79}
  },
  "recommendation": {
    "medical": ["深度清创处理", "分泌物培养 + 敏感抗生素", "转诊足病专科"],
    "lifestyle": ["避免患足负重", "严格控制血糖", "每日足部检查"],
    "urgency": "门诊 1-2 周随访"
  },
  "caveats": "本报告为 AI 辅助评估，最终诊断请遵医嘱",
  "grade_0_note": null
}
```

---

## 九、执行顺序

| 阶段 | 内容 | 状态 |
|:--|:--|:--|
| **R1** | 数据下载脚本 (D1-D5) | ✅ 已完成 |
| **R2** | 自动标注 + 数据集划分 (A1-A5) | ✅ 已完成 |
| **R3** | 增强管线重构 (E1-E6) | ✅ 已完成 |
| **R4** | 模型架构 (M1-M4) | ✅ 已完成（改用 CORN 有序回归，非原始多任务设计） |
| **R5** | 训练策略 (T1-T5) | ✅ 已完成（corn_v4, 80 epoch joint training） |
| **R6** | 评估与可解释性 (V1-V4) | ✅ 已完成（Grad-CAM + HTML 报告 + 独立测试） |
| **R7** | 系统集成 (S1-S4) | ✅ 已完成 |
| **验收** | 二分类 F1 99.36% ✅ · Grade 1-3 F1 38-57%（原始目标 ≥80% 未达成）· Grade 5 从零到 22% F1 🆕 | ✅ 已完成 |

---

## 十、风险与应对

| 风险 | 应对 |
|:--|:--|
| Grade 0 数据不足或无公开数据集 | 从正常足部中筛选有高危特征(胼胝/畸形)的图片 + 增强模拟 |
| Grade 4/5 样本极少 (<100) | 合成数据 + 重度过采样 + Class-Balanced Loss |
| 医学预训练模型提升有限 | 回退 ImageNet ResNet-50 作为可靠基线 |
| Grade 2↔3 邻级混淆 | 深度子任务 Head 提供额外梯度，强化层次区分 |
| 训练资源不足 (GPU) | 降低 batch_size + AMP + Gradient Accumulation |

---

## 十一、项目结构

```
/root/dfu/
├── data/                        # 数据目录
│   ├── raw/                     # 原始下载数据
│   │   ├── adpm/                #   ADPM 原图
│   │   ├── normal_feet/         #   正常足部
│   │   ├── grade0/              #   高危足
│   │   ├── kaggle_laithjj/      #   Kaggle DFU
│   │   └── gangrene/            #   坏疽 Grade 4/5
│   └── processed/               # 处理后数据
│       ├── train/
│       ├── val/
│       └── test/
├── models/                      # 模型保存
├── src/
│   ├── download/                # 数据下载脚本
│   ├── label/                   # 自动标注脚本
│   ├── dataset.py               # Dataset + 增强
│   ├── model.py                 # 模型定义
│   ├── train_stage1.py          # Stage-1 二分类训练
│   ├── train_stage2.py          # Stage-2 多任务训练
│   ├── evaluate.py              # 评估脚本
│   ├── gradcam_viz.py           # Grad-CAM 可视化
│   ├── inference.py             # 推理 Pipeline
│   └── recommend.py             # 建议生成
├── config.yaml                  # 配置文件
├── requirements.txt
└── PLAN.md                      # 本文件
```
