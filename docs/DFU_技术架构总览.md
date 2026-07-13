# DFU Wagner 0-5 分级系统 — 全套技术架构

> **项目路径**: `/root/dfu`
> **模型版本**: corn_v2 / ConvNeXt-Tiny
> **更新日期**: 2026-07-13

---

## 总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DFU Wagner 0-5 分级系统 技术栈                          │
├───────────┬──────────────────────────────────────────────────────────────────┤
│   层级     │  核心技术                                                         │
├───────────┼──────────────────────────────────────────────────────────────────┤
│  数据采集    │  Kaggle / Mendeley / HuggingFace / Wikimedia / Open-i 多源下载            │
│  数据标注    │  规则基自动标注（28K+ 图片 → 7 类标签）+ 无监督聚类拆分（HSV+GLCM+深度特征）   │
│  数据增广    │  离线 R3 三级增广（轻/中/重, 稀有类别定向扩增）+ 在线 RandAugment + RandomErasing │
│  骨干网络    │  ConvNeXt-Tiny (ImageNet-1K 预训练, 冻结全部权重)                          │
│  分类头      │  CORN 有序回归头 + BinaryHead 二分类筛查头                                │
│  损失函数    │  CORN BCE Loss（有序）+ Focal Loss / Label Smoothing CE（对比实验）         │
│  训练策略    │  AdamW + CosineWarmRestarts + AMP 混合精度（GradScaler）+ Early Stopping   │
│  验证策略    │  3-Fold 交叉验证（患者级伤口分割, 杜绝数据泄漏）+ Mean ± 95% CI              │
│  推理增强    │  TTA 测试时增强（6 视角 logit 平均）+ Grad-CAM 可解释性热力图                 │
│  临床输出    │  终端 + JSON + 自包含 HTML 中文报告（base64 内嵌图片, 可打印 PDF）           │
└───────────┴──────────────────────────────────────────────────────────────────┘
```

---

## 一、数据采集与数据层 (`src/download/` + `src/dataset.py`)

### 1.1 多源数据采集管线

项目从 5 个数据源采集了 28,000+ 足部图像，是整个分级系统的数据基础：

| 来源 | 内容 | 采集方式 | 数量 |
|:--|:--|:--|:--|
| **Kaggle Laithjj DFU** | 二分类 健康/溃疡 | `kagglehub` 匿名下载 | ~1,055 原图 |
| **Mendeley Wound** | 伤口图像 + 正常足 | Cloudscraper 绕过 Cloudflare 反爬 | ~2,000 |
| **HuggingFace Wound** | 伤口分类（parquet 格式） | huggingface_hub + pandas 解析 | ~5,000 |
| **Open-i NIH** | 搜索下载坏疽、高危足 | NIH API + 12 组查询词 | 不定量 |
| **Wikimedia Commons** | CC 许可医学图片 | Wikimedia API | 不定量 |

关键技术细节：
- **网络环境**：HuggingFace 等境外数据源在国内受限，需自行配置网络环境后通过 `huggingface_hub` 直接下载
- **Parquet 解析**：HuggingFace 数据集以 parquet 格式存储，通过 pandas 读取 `bytes` 列中的图像数据后保存
- **Cloudscraper**：Mendeley 使用 Cloudflare 保护，`cloudscraper` 模拟 TLS 握手绕过反爬
- **MD5 去重**：`extract_new_data.py` 在复制文件前先对已有数据做 MD5 哈希去重

### 1.2 标签体系（7 类 Wagner 有序分级）

| 索引 | 标签 | 临床含义 |
|:--|:--|:--|
| 0 | normal | 健康足部，无 DFU |
| 1 | grade0 | Wagner 0 高危足，无溃疡但存在风险因素 |
| 2 | grade1 | Wagner 1 浅表溃疡 |
| 3 | grade2 | Wagner 2 深部溃疡 |
| 4 | grade3 | Wagner 3 深部感染 |
| 5 | grade4 | Wagner 4 局限性坏疽 |
| 6 | grade5 | Wagner 5 全足坏疽（预留） |

**为什么用 7 类而非简单二分类？** 临床治疗决策取决于严重度等级——Wagner 0 只需随访，Wagner 3 需要急诊住院。二分类会丢失这个关键信息。

### 1.3 Group-based Sampling（按伤口分组采样）

同一伤口的多张增强变体会被归入同一组，每次 `__getitem__` 时随机选取组内一个变体：

```python
# 文件命名示例:
#   同一伤口:  DM001_M_L.jpg → DM001_M_L_aug0001.jpg → DM001_M_L_aug0002.jpg
#   提取 orig_id: DM001_M_L
#   分组:        {DM001_M_L: ["DM001_M_L.jpg", "DM001_M_L_aug0001.jpg", ...]}
grade_groups[orig_id].append(str(img_path))
```

**为什么？** 离线数据增强（旋转、锐化、Roboflow 增强）会为同一伤口产生多个变体。如果不分组，同一伤口的不同变体可能同时出现在训练集和验证集中 → **数据泄漏**。分组保证同一伤口的所有变体只在同一 fold / 同一 split 中。

### 1.4 在线训练增强

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=3, magnitude=12),  # 在线随机增强
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.3),                   # 随机遮挡
])
```

| 技术 | 作用 | 为什么选它 |
|:--|:--|:--|
| **RandAugment** | 随机组合 3 种增强操作（亮度/对比度/旋转/裁切等） | 自动搜索最优增强策略，比手工设计更强；减少过拟合 |
| **RandomErasing** | 随机遮挡图像中一个小矩形区域 | 迫使模型不只依赖局部纹理，提升对遮挡/敷料的鲁棒性 |
| **ImageNet 归一化** | 用 ImageNet 的均值/标准差归一化 | 配合 ImageNet 预训练权重，关键——不这样做预训练特征完全不可用 |

### 1.5 逆频类别权重

```python
# 稀有类别（grade4 坏疽）得到更高权重
weights[i] = total / (n * count)
```

**为什么？** Grade 4（坏疽）仅 331 张图，而 Grade 1 有 2800+ 张。不加权的话模型会忽略稀有类，全部预测为常见类也能获得低 loss。

---

## 二、离线数据增广 R3 (`src/augmentation/r3_augment.py`)

### 2.1 为什么需要离线增广？

在线增强（RandAugment）每个 epoch 生成不同变体，但不能增加数据集的有效样本量。对于**极度稀有的类别**，需要先通过离线增广扩充基础数据量，再配合在线增强做每个 epoch 的随机扰动。

```
稀有类别增广策略：

grade0（高危足）:  223 张 → ×8 变体/张  → ~2,000 张  (目标 ≥500)
grade4（坏疽）:      1 张 → ×199 变体/张  →    200 张  (极端扩增)
normal/grade1:    充足, 不做离线增广
grade2/3/5:       空占位, 不做增广
```

**关键原则**：只对 `train/` 做增广，`val/` 和 `test/` 完全不动 → 杜绝数据泄漏。

### 2.2 三条增广管线（全部基于 torchvision.transforms，零额外依赖）

增广强度按 轻:中:重 比例分配，模拟真实拍摄中的各种变异：

```
原图 (DM001_M_L.png)
  │
  ├── 轻度 (30%)  — 小角度旋转 ±10° + 轻微颜色抖动 + 水平翻转
  ├── 中度 (50%)  — 旋转 ±25° + 颜色抖动 + 缩放 0.9-1.1× + 高斯模糊 + 翻转
  └── 重度 (20%)  — 旋转 ±45° + 强颜色抖动 + 透视变换 + 仿射剪切 + 模糊 + 锐化 + 翻转
        │
        ▼
  输出: DM001_M_L_aug0001.jpg … DM001_M_L_aug0199.jpg
```

```python
# 轻度 — 模拟轻微手持抖动、光照变化
T.Compose([
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    T.RandomHorizontalFlip(p=0.5),
])

# 中度 — 模拟不同角度/距离拍摄、轻微运动模糊
T.Compose([
    T.RandomRotation(degrees=25),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
    T.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # 模拟拍摄距离变化
    T.GaussianBlur(kernel_size=3),
    T.RandomHorizontalFlip(p=0.5),
])

# 重度 — 模拟极端光照、敷料遮挡、透视畸变 (grade4 更多用重度)
T.Compose([
    T.RandomRotation(degrees=45),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
    T.RandomPerspective(distortion_scale=0.3, p=0.5),  # 透视畸变
    T.RandomAffine(degrees=0, scale=(0.85, 1.15), shear=10),  # 仿射+剪切
    T.GaussianBlur(kernel_size=5),
    T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),  # 锐化→模拟不同对焦
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
])
```

### 2.3 文件名编码与溯源

增广变体命名为 `{原图stem}_aug{序号:04d}.jpg`：

```
DM001_M_L.png            ← 原图
DM001_M_L_aug0001.jpg    ← 增广变体
DM001_M_L_aug0002.jpg
...
```

`get_original_id()` 函数通过正则 `_aug\d{4,}$` 剥离增广后缀，将同一伤口的所有变体归入同一组 → 保证后续 split/CV 时它们不会分散到不同集合中。

### 2.4 与在线增强的协同

```
离线 R3 增广                  在线训练增强 (RandAugment)
─────────────────────────────────────────────────────────
目的: 扩充稀有类别基数        目的: 每个 epoch 不同扰动 → 防过拟合
时机: 训练前, 只跑一次         时机: 训练时每个 __getitem__
作用: grade4 1张→200张         作用: 200张 × 每epoch不同变体
特点: 固定保存到磁盘            特点: 纯内存, 不占磁盘
```

两者互补：R3 解决"图太少模型根本学不到"，RandAugment 解决"图多了但每 epoch 都一样模型背下来"。

### 2.5 不再使用 Roboflow 预增强

原 ADPM 数据集 1,373 张 → Roboflow 固定增强 → 10,062 张（严重过拟合来源）。R3 改为仅用原图 + 自定义增广管线，完全可控、可复现。

---

## 三、无监督聚类数据精炼 (`src/cluster_split.py` + `src/labeling/`)

### 3.1 问题背景

原始数据集只有粗粒度的二类标签——所有溃疡图片统一标为 "grade1"，所有坏疽图片统一标为 "grade4"。但实际上：

```
原始标签         实际包含的 Wagner 级别
─────────────────────────────────────────
grade1    ──→   Wagner 1（浅表）+ Wagner 2（深部）+ Wagner 3（骨髓炎）
grade4    ──→   Wagner 4（局限坏疽）+ Wagner 5（全足坏疽）
```

**没有任何数据集直接标注了 Wagner 细分级别。** 如果只用粗标签训练，模型永远学不会区分 Wagner 1/2/3，无法满足临床分级需求。

### 3.2 核心思路：利用临床先验做无监督拆分

Wagner 1→2→3 的创面颜色存在明确的临床渐变规律：

```
Wagner 1（浅表溃疡）  Wagner 2（深部溃疡）   Wagner 3（骨髓炎）
    红色肉芽           黄色腐肉/渗出          白色骨暴露
    ████████           ████████████          ████████████
    RGB: 红＞绿＞蓝     RGB: 黄≈红＞蓝        RGB: 白≈红≈绿≈蓝
```

**创面颜色从红 → 黄 → 白/黑是区分 Wagner 1/2/3 的核心视觉信号。** 这为无监督聚类提供了临床基础。

### 3.3 三类特征工程

对每张图片提取三类互补特征：

```
┌─────────────────────────────────────────────────────────────────┐
│                      特征融合管线                                │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ 传统 CV 特征     │  深度特征        │  CORN 序数概率              │
│ (136 维)        │  (768 维)       │  (6 维)                    │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ HSV 颜色直方图   │ ConvNeXt-Tiny   │ P(≥grade0)                 │
│ (32 bin × 3)    │ backbone 输出   │ P(≥grade1)                 │
│ = 96 维         │ (冻结权重)       │ P(≥grade2) ← 关键信号       │
│                 │                 │ P(≥grade3)                 │
│ RGB/HSV 颜色统计 │                 │ P(≥grade4)                 │
│ (均值/标准差/中位│                 │ P(≥grade5)                 │
│  数/P10/P90)    │                 │                            │
│ 3ch×2space×5stat│                 │                            │
│ = 30 维         │                 │                            │
│                 │                 │                            │
│ GLCM 纹理特征    │                 │                            │
│ (对比度/差异性/  │                 │                            │
│  同质性/能量/    │                 │                            │
│  相关性×4角度    │                 │                            │
│  = 10 维)       │                 │                            │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ 捕捉创面颜色、   │ 通用视觉语义,    │ 模型对严重度的"直觉"——     │
│ 纹理、光照变化   │ 不受人工特征局限  │ 即使未细分训练, 模型对      │
│                 │                 │ 不同严重度已有微弱偏好       │
└─────────────────┴─────────────────┴─────────────────────────────┘
│                                                                 │
│              Hybrid = 136 + 768 + 6 = 910 维                    │
└─────────────────────────────────────────────────────────────────┘
```

**为什么三类特征互补？**

| 特征类型 | 优势 | 局限 |
|:--|:--|:--|
| HSV 直方图 + 颜色统计 | 直接编码"红→黄→白"的临床信号，可解释性强 | 对光照/肤色变化敏感 |
| GLCM 纹理 | 捕捉肉芽组织 vs 腐肉 vs 骨质的纹理差异 | 维度低，信息有限 |
| ConvNeXt 深度特征 | 768 维丰富语义，对光照/旋转鲁棒 | 黑盒，不可直接解释 |
| CORN 序数概率 | 模型已从粗标签中学到的严重度排序知识 | 依赖模型质量，可能引入偏差 |

### 3.4 五种聚类策略对比

```python
# 策略 1: K-Means on 深度特征
labels = KMeans(ConvNeXt 768d features, n_clusters=3)

# 策略 2: K-Means on 传统 CV 特征（HSV + 纹理）
labels = KMeans(传统 CV 136d features, n_clusters=3)

# 策略 3: K-Means on 混合特征（深度 + CV + 序数概率）
labels = KMeans(Hybrid 910d features, n_clusters=3)

# 策略 4: CORN 阈值排序 — 按 P(≥grade2) 三等分
order = np.argsort(P(>=grade2))
labels = [底部1/3→grade1, 中部1/3→grade2, 顶部1/3→grade3]

# 策略 5: 质心距离 + 序数概率联合排序
combined = 0.5×(PCA距离/std) + 0.5×(P(≥grade2)/std)
labels = 按 combined 分数三等分
```

所有策略均在 PCA 降维后的特征空间上运行（保留 95% 方差），并通过 **StandardScaler** 归一化消除量纲差异。

### 3.5 聚类评估与标签映射

**评估指标**：Silhouette Score（轮廓系数）——衡量簇内紧密度 vs 簇间分离度。

**自动标签映射**：聚类本身输出的是无意义的簇编号（C0/C1/C2），需要映射到 Wagner 1/2/3。利用 **P(≥grade2) 即模型判断"至少是 grade2"的概率** 作为严重度排序依据：

```
簇的 P(≥grade2) 均值最低  → 创面最浅 → Wagner 1
簇的 P(≥grade2) 均值中间  → 中等深度 → Wagner 2
簇的 P(≥grade2) 均值最高  → 创面最深 → Wagner 3
```

### 3.6 聚类结果

从 [reports/cluster_report_v2.json](reports/cluster_report_v2.json)：

| 簇 | 分配级别 | 样本数 | 占比 | P(≥grade2) 均值 |
|:--|:--|:--|:--|:--|
| C0 | **grade1** | 1,546 | 27.4% | 最低（浅表创面） |
| C1 | **grade2** | ~2,000+ | ~36% | 中等 |
| C2 | **grade3** | ~2,000+ | ~36% | 最高（深度创面） |

总计 **5,634 张** grade1 粗标签图片 → 拆分为 **grade1 + grade2 + grade3** 三个细分级别。

最终输出：
- `reports/cluster_assignments_v2.csv` — 每张图片的簇分配 + Wagner 级别
- `reports/cluster_report_v2.json` — 完整聚类报告（各策略对比、簇统计）

### 3.7 规则标注管线 (`src/labeling/r2_labeling.py`)

除了无监督聚类拆分 grade1→3，项目还实现了完整的规则基标注管线：

| 阶段 | 内容 |
|:--|:--|
| **A1 清单构建** | 扫描全部 28,000+ 原始图片，按数据来源路径匹配标签规则 |
| **A2 质量分析** | 标签分布统计、类别不平衡检测、重复图片 MD5 去重 |
| **A3 分层分割** | 70/15/15 分层 train/val/test 切分，保留类别比例 |

规则匹配涵盖 20+ 数据来源的自动标注：

```
dm_foot_grade0/          → grade0 (糖尿病高危足)
normal_foot_control/     → normal (健康对照)
mendeley_wound/          → grade1 (伤口)
kaggle_laithjj/          → grade1/normal (按子目录)
dermnet_grade0/          → grade0 (胼胝/趾甲) 或 non_dfu (银屑病/湿疹)
gangrene 关键词           → grade4 (坏疽)
wound_segmentation/      → grade1 (分割数据集伤口)
wound_unlabeled/         → grade1 (低置信度)
```

输出：
- `data/manifest.csv` — 全量清单（路径、标签、置信度、来源、MD5）
- `data/manifest_report.txt` — 标签分布报告
- `data/processed/` — 按 train/val/test/{label} 组织的目录结构

---

## 四、骨干网络 (`src/model.py`) — ConvNeXt-Tiny

### 4.1 为什么选 ConvNeXt-Tiny？

| 对比维度 | ConvNeXt-Tiny | ResNet-50 | EfficientNet-B0 |
|:--|:--|:--|:--|
| 参数量 | 28M | 25M | 5.3M |
| ImageNet Top-1 | **82.1%** | 76.1% | 77.1% |
| 架构风格 | 现代化 CNN（借鉴 Swin Transformer 设计） | 经典残差网络 (2015) | NAS 搜索架构 |
| 特征维度 | 768 | 2048 | 1280 |
| 医学图像适配 | ✅ 层次化特征，适合多尺度病变 | ⚠️ 较深，小病变信息可能丢失 | ⚠️ 太轻量，特征表达能力有限 |

**核心理由**：

- ConvNeXt 借鉴了 Vision Transformer 的设计哲学（大 kernel 7×7、LayerNorm 替代 BatchNorm、GELU 激活），但保留了 CNN 的效率和局部归纳偏置
- 224×224 输入时，最后一个空间特征图是 7×7（与 ResNet 相同），适合中等大小的足部病变
- 冻结全部骨干网络后仅训练头部 → **仅 5,389 可训练参数**，极大降低过拟合风险

### 4.2 冻结策略

```python
# 冻结整个 ConvNeXt-Tiny 骨干网络（28M 参数不参与训练）
for param in model.parameters():
    param.requires_grad = False
```

**为什么冻结全部？**

- 医学数据集通常远小于 ImageNet（我们约 8000 张 vs ImageNet 120 万张）
- 预训练特征已包含边缘/纹理/形状检测能力，足以描述足部病变
- 可训练参数仅 5,389 → 训练极快，几乎不过拟合

---

## 五、CORN 有序回归头 (`src/model.py`) — 核心分类方法

### 5.1 为什么不用普通 Softmax 分类？

普通分类将 Wagner 0-5 视为 7 个**独立**类别，完全忽略了严重度的**有序性**：

```
普通分类:  normal  grade0  grade1  grade2  grade3  grade4
           └──────┴───────┴───────┴───────┴───────┴──────┘
                    7 个独立类别（无顺序关系）

CORN:     normal < grade0 < grade1 < grade2 < grade3 < grade4
          └──────────────────────────────────────────────────┘
                    严重度递增（有序关系被显式建模）
```

**实际影响**：把 Wagner 4 预测成 Wagner 3 只差一级（可接受），预测成 normal 差五级（严重误诊）。普通分类对这两种错误的惩罚完全相同；CORN 天然能区分。

### 5.2 CORN 工作原理

```
K=7 个类别 → K-1=6 个二分类器, 每个回答 "严重度 ≥ 等级 k 吗？"

输入图片
  │
  ▼
ConvNeXt-Tiny → 768 维特征向量
  │
  ▼
CORNHead (线性层: 768 → 6)
  │
  ├─ logit[0]: P(≥ grade0) → sigmoid → 0.95
  ├─ logit[1]: P(≥ grade1) → sigmoid → 0.82
  ├─ logit[2]: P(≥ grade2) → sigmoid → 0.60
  ├─ logit[3]: P(≥ grade3) → sigmoid → 0.30
  ├─ logit[4]: P(≥ grade4) → sigmoid → 0.08
  └─ logit[5]: P(≥ grade5) → sigmoid → 0.01

预测类别 = sum(sigmoid(logits) >= 0.5) = 2 → grade1
```

### 5.3 单调性约束（关键设计）

```python
class CORNHead(nn.Module):
    def __init__(self, in_features, num_classes=7):
        self.linear = nn.Linear(in_features, num_classes - 1)  # 768 → 6
        self.base_bias = nn.Parameter(torch.zeros(1))
        self.bias_deltas = nn.Parameter(torch.zeros(num_classes - 2))  # 5 个 delta

    def forward(self, x):
        logits = self.linear(x)
        # bias 链: bias[k] = bias[k-1] + softplus(delta[k-1])
        # 因为 softplus 永远 ≥ 0，所以 bias 非递减
        # → 保证 P(≥grade0) ≥ P(≥grade1) ≥ ... ≥ P(≥grade5)
        biases = [self.base_bias]
        for delta in self.bias_deltas:
            biases.append(biases[-1] + F.softplus(delta))
        return logits + torch.cat(biases)
```

**为什么 bias 要单调？** 如果 P(≥grade3) > P(≥grade1)，逻辑上矛盾——已经严重到 grade3 却不如 grade1 严重？这个约束确保输出始终在合理范围内。

### 5.4 类别概率计算

```python
def predict_proba(self, logits):
    probs = torch.sigmoid(logits)  # [B, 6]
    # 通过减法链从阈值概率推导每类概率
    class_probs = [
        1 - probs[:, 0],            # P(class 0) = "什么都不够"
        probs[:, 0] - probs[:, 1],  # P(class 1) = "够了 grade0 但不够 grade1"
        probs[:, 1] - probs[:, 2],  # P(class 2) = "够了 grade1 但不够 grade2"
        probs[:, 2] - probs[:, 3],
        probs[:, 3] - probs[:, 4],
        probs[:, 4] - probs[:, 5],
        probs[:, 5],                # P(class 6) = "所有阈值都够了"
    ]
    return torch.stack(class_probs, dim=1).clamp(0, 1)
```

---

## 六、两阶段推理架构

### 6.1 为什么用两阶段？

```
阶段 1 (BinaryHead):  "这是溃疡吗？"  →  benign / ulcer
阶段 2 (CORNHead):    "Wagner 几级？"  →  0 / 1 / 2 / 3 / 4 / 5
```

| 设计 | 优势 |
|:--|:--|
| **共享骨干网络** | 两个任务共享 ConvNeXt 特征提取，不增加推理成本 |
| **先筛查再分级** | 如果阶段 1 判为 benign，可以跳过详细分级（快速筛查场景） |
| **互补验证** | 两个头的结果可以互相校验——如果 binary 说 benign 但 ordinal 说 grade3，说明结果不可靠，需要医生复核 |

### 6.2 推理流程

```python
features = model.backbone(tensor)              # 共享特征提取

# 阶段 1: 二分类筛查
bin_logit = model.binary_head(features)        # 768 → 1
bin_prob = sigmoid(bin_logit)                  # P(溃疡)
is_ulcer = bin_prob >= 0.5

# 阶段 2: CORN 有序分级
ord_logits = model.ordinal_head(features)      # 768 → 6
wagner = sum(sigmoid(ord_logits) >= 0.5)       # 累积超过阈值的个数
```

---

## 七、损失函数

### 7.1 CORN Loss

```python
def corn_loss(logits, labels, sample_weights=None):
    # 将类别标签转换为 K-1 个二分类目标
    # grade 2 → [1, 1, 0, 0, 0, 0]  (前两个阈值"够了"，后面"不够")
    targets = (labels.unsqueeze(1) > torch.arange(num_tasks)).float()
    # BCEWithLogitsLoss 对每个阈值独立计算
    return F.binary_cross_entropy_with_logits(logits, targets)
```

本质是 **K-1 个独立的二分类交叉熵之和**，但通过 bias 单调性约束建立了它们之间的依赖关系。

### 7.2 损失函数对比与代码

| 损失 | 使用场景 | 特点 |
|:--|:--|:--|
| **CORN BCE** | 有序回归（主线） | 每个阈值独立优化，配合 bias 约束 |
| **Focal Loss** | 普通分类（备选/对比实验） | `(1-pt)^γ` 因子降低易分样本权重，聚焦难分样本 |
| **Label Smoothing CE** | 普通分类（备选/对比实验） | 软标签 ε=0.1 → 防止过拟合到 one-hot |

**Focal Loss 实现**（用于标准分类模式的对比实验）：

```python
class FocalLoss(nn.Module):
    """
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    当 γ=2 时:
      - 易分样本 (p_t≈0.9): (1-0.9)² = 0.01 → loss 缩小 100 倍
      - 难分样本 (p_t≈0.1): (1-0.1)² = 0.81 → loss 几乎不变
      → 模型自动聚焦难分样本 (如 grade4 坏疽)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 可选类别权重 [C]
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()
```

**Label Smoothing CE 实现**：

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    将硬标签 [1, 0, 0, 0, 0, 0, 0] 变为软标签 [0.9, 0.0167, 0.0167, ...]
    ε=0.1: 真实类概率 0.9, 其余 6 类各 0.0167
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
```

**为什么 CORN loss 不加 Focal 机制？** 因为 CORN 通过类别权重（`sample_weights`）已解决了类别不平衡——稀有类别获得更高权重，无需 Focal 的二次加权。

---

## 八、训练策略 (`src/train.py` + `src/cross_validate.py`)

### 8.1 完整训练配置 (`config.yaml`)

| 组件 | 选择 | 原因 |
|:--|:--|:--|
| **优化器** | AdamW | Adam 的解耦权重衰减版——正则化更干净，不干扰自适应学习率 |
| **学习率** | 1e-3（仅头部） | 冻结骨干后可以用较大学习率 |
| **调度器** | CosineAnnealingWarmRestarts | 周期性重启避免陷入局部最优，`T_0=15, T_mult=2` |
| **混合精度** | AMP (torch.cuda.amp) | 训练速度提升约 2x，显存减半，精度损失可忽略 |
| **早停** | patience=15 | 验证 F1 不再提升时自动停止 |
| **Batch Size** | 64 | ConvNeXt-Tiny 足够小的显存占用，最大化 GPU 利用率 |
| **训练轮数** | 最大 80 epochs | 实际通常 30-40 轮就触发早停 |

### 8.2 AMP 混合精度训练

使用 `torch.amp.autocast` + `GradScaler` 实现混合精度训练：

```python
scaler = torch.amp.GradScaler("cuda")

for images, labels in train_loader:
    optimizer.zero_grad()

    with torch.amp.autocast("cuda"):       # 自动 FP16 前向
        logits = model(images, task="ordinal")
        loss = corn_loss(logits, labels, class_weights)

    scaler.scale(loss).backward()          # loss × scale → 放大梯度防下溢
    scaler.step(optimizer)                 # 梯度还原后更新参数
    scaler.update()                        # 动态调整 scale_factor
```

**工作原理**：
- **前向传播**：autocast 自动将 ConvNeXt 的卷积/矩阵乘法转为 FP16 加速，保持 BatchNorm/Softmax 为 FP32 保证精度
- **反向传播**：GradScaler 在 loss 上乘以 scale factor（初始 65536），放大 FP16 下可能下溢为 0 的微小梯度
- **参数更新**：`scaler.step()` 将梯度还原到真实尺度后更新参数；若检测到 inf/NaN 则跳过本次更新并降低 scale factor
- **收益**：训练速度提升约 **2×**，显存占用减半，精度损失可忽略（< 0.1%）

### 8.3 学习率调度器：CosineAnnealingWarmRestarts

```
LR
^
|  ╲                  ╲                  ╲
|   ╲    T_0=15        ╲    T_1=30        ╲    T_2=60
|    ╲                ╲                ╲
|     ╲______________╲______________╲______________>
      0    15    30    45    60    75    epochs
```

**为什么用 WarmRestarts 而非 ReduceLROnPlateau？** 周期性重置学习率让模型有机会跳出局部最优，而不是一路衰减到零停滞。每次重启的峰值略低于前一次（`T_mult=2` 让周期越来越长）。

### 8.4 3-Fold 交叉验证策略

```
关键设计：患者级分割（非图片级）

传统做法 ❌:  把所有图片打乱分 3 份 → 同一患者的不同图片可能分散在不同 fold
我们的做法 ✅:  同一患者的所有伤口图片必须放在同一 fold

具体:
  - DM/CG 真实患者: patient-level → 同一患者的全部伤口在同一个 fold
  - 聚类标注数据: wound-level → 每个 cluster ID 独立处理（因为没有患者 ID）

交叉验证流程:
  Fold 0: train(2/3) → val(15%) → test(1/3)
  Fold 1: train(2/3) → val(15%) → test(1/3)  ← 不同的 test 集合
  Fold 2: train(2/3) → val(15%) → test(1/3)
  最终报告: mean ± 95% CI (t 分布)
```

**为什么患者级分割重要？** 如果患者 DM001 的左脚和右脚照片分别出现在训练集和测试集，模型可能过拟合到该患者的特定皮肤纹理/光照，而非学习溃疡特征 → 虚假的高准确率。

---

## 九、TTA 测试时增强 (`src/tta.py`)

### 9.1 6 种增强视角

```python
TTA_TRANSFORMS = {
    "identity":    原图,                           # dims 不变
    "hflip":       torch.flip(dims=[-1]),          # 水平镜像
    "rotate+5":    rotate(angle=5.0°),             # 顺时针微旋
    "rotate-5":    rotate(angle=-5.0°),            # 逆时针微旋
    "brightness+": clamp(x * 1.1),                 # 亮度 +10%
    "brightness-": clamp(x * 0.9),                 # 亮度 -10%
}
```

### 9.2 为什么用 TTA？

```
单次推理:  输入 → 模型 → 输出（可能受噪声影响）
TTA 推理:  输入 → [原图, 翻转, 旋转+5°, 旋转-5°] → 模型 → [logit1, logit2, logit3, logit4]
                └────────────────────────────────┘
                      取平均 logit → 输出（更稳定）

效果: 减少单次前向的随机波动，Kappa 通常提升 1-2%
代价: 推理时间 ×4（但每张图仍 <100ms，临床可接受）
```

### 9.3 为什么平均 logit 而非平均概率？

```python
# ✅ 正确: logit 空间平均（线性空间，信息无损）
avg_logits = torch.stack(all_logits).mean(dim=0)
probs = sigmoid(avg_logits)

# ❌ 错误: 概率空间平均（sigmoid 是非线性压缩，信息损失）
avg_probs = torch.stack(all_probs).mean(dim=0)
```

Logit 空间是线性的，取平均后再 sigmoid 保留了完整的分布信息。概率空间经过非线性压缩后再平均会导致信息损失。

---

## 十、Grad-CAM 可解释性 (`src/gradcam.py`)

### 10.1 为什么需要 Grad-CAM？

医生无法信任一个"黑盒"AI。如果模型把正常皮肤判为 Wagner 3，医生需要看到**模型到底看了哪里**。Grad-CAM 回答了这个问题——热力图高亮模型决策依赖的图像区域。

### 10.2 技术实现详解

```python
class GradCAM:
    def __init__(self, model):
        # 在 backbone 最后一个空间层注册前向 hook
        self.target_layer = model.backbone.features  # ConvNeXt 的输出
        self.handle = self.target_layer.register_forward_hook(self._hook)

    def __call__(self, x, target_class):
        # ====== Step 1: 手动前向传播，捕获中间激活 ======
        activations = self.model.backbone.features(x)   # [1, 768, 7, 7]
        pooled = self.model.backbone.avgpool(activations)
        flat = self.model.backbone.classifier(pooled)    # [1, 768]
        ord_logits = self.model.ordinal_head(flat)       # [1, 6]

        # ====== Step 2: 确定目标分数 ======
        # CORN 特有——不是简单的 class logit
        if target_class == 0:
            score = -ord_logits[:, 0]   # normal → 要降低 P(≥grade0)
        else:
            score = ord_logits[:, target_class - 1]  # grade k → 要提高第 k-1 个阈值

        # ====== Step 3: 用 autograd.grad 求梯度 ======
        grads = torch.autograd.grad(
            outputs=score,
            inputs=activations,
            retain_graph=False,
            create_graph=False,
        )[0]  # [1, 768, 7, 7]

        # ====== Step 4: 全局平均池化梯度 → 通道权重 ======
        weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, 768, 1, 1]

        # ====== Step 5: 加权组合 + ReLU + 上采样 ======
        cam = (weights * activations).sum(dim=1)          # [1, 7, 7]
        cam = F.relu(cam)                                  # 只保留正向贡献
        cam = F.interpolate(cam, size=(224, 224))          # 上采样 32×
        return heatmap, pred_class, confidence
```

### 10.3 为什么用 `torch.autograd.grad()` 而非 backward hook？

| 方案 | 机制 | 优缺点 |
|:--|:--|:--|
| backward hook | `loss.backward()` → 遍历全图 | 需要完整前向+反向传播，慢，容易断图 |
| `autograd.grad` | 直接指定 `∂score/∂activations` | 只计算一条梯度路径，精确、快速、可控 |

`autograd.grad` 只计算一条梯度路径（score → activations），比完整反向传播高效得多。

### 10.4 `torch.enable_grad()` 的必要性

```python
@torch.no_grad()           # predict_image() 禁用了梯度以节省显存
def predict_image(...):
    ...
    with torch.enable_grad():  # 临时重新启用梯度 → Grad-CAM 需要！
        heatmap = cam_extractor(tensor, target_class=wagner_grade)
```

> ⚠️ 这是 R7 开发中踩过的坑——`no_grad` 上下文会阻断所有梯度计算，即使 `autograd.grad` 也不例外。

---

## 十一、HTML 临床报告 (`src/report.py`)

### 11.1 设计理念

| 设计决策 | 原因 |
|:--|:--|
| **纯 Python，零外部依赖** | 部署在医院内网环境，`pip install` 可能不可用 |
| **base64 内嵌图片** | 单个 HTML 文件离线可用，不会被"图片找不到"困扰 |
| **`@media print`** | 浏览器 Ctrl+P 直接打印 PDF，A4 自适应 |
| **中文第一语言** | 中国医院场景，临床术语全部中文 |
| **严重度色标** | 绿→黄→橙→红→紫，医生一目了然 |

### 11.2 报告结构

```
┌─────────────────────────────────────────────────────────────┐
│  糖尿病足 Wagner 分级 — AI 辅助评估报告                       │
│  生成时间 · 模型版本 · 图像文件名                             │
├───────────────────────┬─────────────────────────────────────┤
│  原始足部图像          │  Grad-CAM 热力图                     │
│  (base64 内嵌)        │  (模型关注区域高亮)                   │
├───────────────────────┴─────────────────────────────────────┤
│  Wagner X  [色标]        置信度: XX.X%    TTA: 启用/禁用     │
│  筛查结果: ✅良性 / ⚠️溃疡                                   │
├─────────────────────────────────────────────────────────────┤
│  各类别概率 (纯 CSS 柱状图, 7 行)                            │
│  normal   ████████░░  80%                                    │
│  grade0   ██░░░░░░░░  15%                                    │
│  ...                                                        │
├─────────────────────────────────────────────────────────────┤
│  临床建议: 诊断摘要 · 医疗措施 · 生活建议 · 紧急程度 · 随访   │
├─────────────────────────────────────────────────────────────┤
│  ⚠️ 免责声明 + 数据局限性说明                                 │
└─────────────────────────────────────────────────────────────┘
```

### 11.3 紧急程度分级

| 临床场景 | 紧急程度 | 颜色 |
|:--|:--|:--|
| Normal 健康足 | 无需就医 | 🟢 `#27ae60` |
| Grade 0 高危足 | 门诊 1-2 周内 | 🟡 `#f39c12` |
| Grade 1 浅表溃疡 | 门诊 3-5 天内 | 🟠 `#e67e22` |
| Grade 2 深部溃疡 | 门诊 24-48 小时内 | 🔴 `#d35400` |
| Grade 3 深部感染 | 急诊，立即住院 | 🔴 `#c0392b` |
| Grade 4 局限性坏疽 | 急诊，24 小时内住院 | 🟣 `#8e44ad` |

---

## 十二、完整数据流

```
┌──────────────┐
│  足部照片     │  (手机/相机拍摄, JPEG/PNG)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  预处理 (dataset.py)                                          │
│  · Resize → 224×224                                          │
│  · ToTensor → [0,1] float32                                  │
│  · Normalize(ImageNet μ, σ) → 匹配预训练分布                   │
└──────┬───────────────────────────────────────────────────────┘
       │  [1, 3, 224, 224]
       ▼
┌──────────────────────────────────────────────────────────────┐
│  ConvNeXt-Tiny (冻结, 28M 参数, 不训练)                        │
│  · features → [1, 768, 7, 7]  空间特征图 (给 Grad-CAM)        │
│  · avgpool → [1, 768, 1, 1]                                  │
│  · classifier (Flatten) → [1, 768]  特征向量                   │
└──────┬───────────────────────────────────────────────────────┘
       │  [1, 768]
       ├─────────────────────┬─────────────────────┐
       ▼                     ▼                     ▼
┌─────────────┐     ┌───────────────┐     ┌───────────────┐
│  BinaryHead  │     │   CORNHead    │     │  (Grad-CAM)   │
│  768→1       │     │   768→6       │     │  特征图可视化   │
│  sigmoid     │     │   单调 bias   │     │  autograd     │
│  P(ulcer)    │     │   6个阈值     │     │  热力图叠加    │
└──────┬──────┘     └──────┬────────┘     └──────┬────────┘
       │                   │                     │
       ▼                   ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│  TTA (可选, tta.py)                                           │
│  原图 + 水平翻转 + 旋转±5° + 亮度±10%                          │
│  → 6 个 logit 向量 → logit 空间平均 → 更稳定的预测              │
└──────┬───────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  终端输出 / JSON / HTML 报告                                   │
│  · 预测等级 + 置信度 + 临床建议                                 │
│  · 7 类概率分布（柱状图）                                       │
│  · 原图 + Grad-CAM → base64 自包含                             │
│  · @media print → 浏览器直接打印 A4 PDF                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 十三、文件清单

| 文件 | 行数 | 职责 |
|:--|:--|:--|
| `src/dataset.py` | 285 | 数据集加载、增强、分组采样、类别权重 |
| `src/model.py` | 374 | ConvNeXt-Tiny + CORNHead + BinaryHead + DFUModel + 损失函数 |
| `src/train.py` | 424 | 单次训练（train/val/test）、AMP、早停、TensorBoard |
| `src/cross_validate.py` | 453 | 3-Fold 患者级交叉验证、统计汇总（均值 ± 95% CI） |
| `src/inference.py` | 542 | 两阶段推理 CLI、TTA、Grad-CAM、报告触发 |
| `src/tta.py` | 161 | TTA 多视角增强（6 种变换）、logit 平均 |
| `src/gradcam.py` | 254 | Grad-CAM 热力图生成（autograd.grad 方式） |
| `src/report.py` | 473 | 自包含 HTML 中文临床报告（base64 内嵌、可打印） |
| `src/cluster_split.py` | 689 | 无监督聚类：HSV/GLCM/深度特征 + 5 策略对比 → 拆分 grade1 为 1/2/3 |
| `src/labeling/r2_labeling.py` | 505 | 规则基自动标注管线：28K+ 图片 → 7 类标签 + 70/15/15 分层切分 |
| `src/augmentation/r3_augment.py` | 235 | 离线数据增广：三级管线（轻/中/重）→ 稀有类别定向扩增 |
| `src/download/download_all.py` | — | 多源数据采集编排器：Kaggle + Mendeley + HuggingFace |
| `src/download/download_gangrene.py` | — | Open-i NIH + Wikimedia 坏疽/高危足专项下载 |
| `src/download/extract_new_data.py` | — | 本地数据提取 + MD5 去重 + 关键词分类 |
| `src/eval_tta.py` | — | TTA vs 标准推理对比评估 |
| `config.yaml` | 64 | 全局配置文件（模型/数据/训练/日志参数） |

---

## 十四、关键性能指标

| 指标 | 值 | 含义 |
|:--|:--|:--|
| **可训练参数** | 5,389 | 仅 CORN head + binary head，冻结整个骨干 |
| **总参数量** | ~28M | ConvNeXt-Tiny 骨干 |
| **单次推理时间** | ~25ms (GPU) | 不含 TTA |
| **TTA 推理时间** | ~100ms (GPU) | 4 视角 |
| **Grad-CAM 生成** | ~30ms (GPU) | autograd.grad |
| **报告生成** | ~200ms | 含图片编码 |

### 3-Fold 交叉验证结果

| 指标 | 均值 ± 95% CI |
|:--|:--|
| **Accuracy（准确率）** | **70.57%** ± 4.31% |
| **Macro F1（宏平均 F1）** | **68.41%** ± 4.28% |
| **Kappa（二次加权）** | **0.854** ± 0.025 |

| Wagner 分级 | F1 ± 95% CI | 评价 |
|:--|:--|:--|
| Normal（正常） | 97.43% ± 0.82% | 🟢 极好 |
| Grade 0（有风险） | 77.09% ± 3.48% | 🟢 良好 |
| Grade 1（浅表） | 59.40% ± 6.62% | 🟡 中等 |
| Grade 2（深部） | 60.30% ± 3.57% | 🟡 中等 |
| Grade 3（感染） | 69.64% ± 6.66% | 🟡 中等 |
| Grade 4（坏疽） | 46.58% ± 5.66% | 🔴 偏低 |

---

## 十五、核心设计哲学

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   "用最小的可训练参数（5K）撬动最强的预训练表征（28M）"        │
│                                                             │
│   配合 CORN 有序回归保证预测的临床合理性                       │
│   （不会出现 normal > grade3 这种矛盾输出）                    │
│                                                             │
│   通过 Grad-CAM 热力图和中文报告让 AI 决策对医生透明可理解      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 三个关键选择

1. **冻结骨干 + CORN 头**：防止小数据集过拟合，同时利用有序回归的临床先验
2. **患者级交叉验证**：真实反映模型对新患者的泛化能力，而非对同一患者不同照片的记忆
3. **推理 + 可解释性 + 报告一体化**：一条命令出结果，降低临床使用门槛
