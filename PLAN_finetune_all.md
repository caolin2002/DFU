# 全参数解冻微调计划 — DFU Wagner 分级系统

> **目标**: 解冻 ConvNeXt-Tiny 全部参数，用差异化小学习率微调 20 轮，验证是否能提升分级准确率（尤其是 Grade 1-3 的 F1）。
>
> **日期**: 2026-07-22

---

## 一、背景分析

### 当前状态

| 项目 | 现状 |
|:--|:--|
| 骨干网络 | ConvNeXt-Tiny (ImageNet-1K 预训练) |
| 冻结策略 | **全部冻结** — `freeze_backbone=True`，仅训练 CORNHead + BinaryHead |
| 可训练参数量 | ~1.5K / 28.6M（约 0.005%） |
| 当前学习率 | `lr=1e-3`（仅作用于 Head） |
| 训练轮数 | 80 epochs（联合训练 binary + CORN ordinal） |
| 最佳模型 | `models/corn_v4/best_model.pth` |

### 问题诊断

1. **Backbone 特征可能不是最优的** — ConvNeXt-Tiny 在 ImageNet-1K 上预训练，其底层特征（边缘、纹理）具有通用性，但高层语义特征偏向自然物体分类，与医学伤口图像存在分布差异（domain gap）
2. **仅训练分类头限制了模型容量** — Head 只有 1.5K 参数，模型的表达能力完全依赖固定的 backbone 特征。如果 backbone 提取的特征对 DFU 分级不是最优的，那么无论 Head 怎么训练都无法补偿
3. **Grade 1-3 的准确率瓶颈** — v4 报告中 Grade 1-3 F1 仅 38-57%，这些中间级别的视觉差异较微妙（浅表溃疡 vs 深部溃疡 vs 深部感染），可能需要 backbone 学习更细粒度的特征

### 为什么全参数微调可能有效

- 让 backbone 的高层卷积核适应伤口图像的纹理、颜色、边界特征
- 让中间层学习到 DFU 特定的层次化特征（表皮→真皮→皮下→骨）
- 差异化学习率可以保护底层通用特征不被破坏，同时让高层语义特征向目标任务对齐

---

## 二、技术方案

### 2.1 差异化学习率策略

```
Backbone (ConvNeXt-Tiny)  →  lr = 1e-5   (小 100 倍)
CORNHead + BinaryHead     →  lr = 1e-4   (中等)
```

**原理**: 较低的 backbone 学习率保护预训练权重不被大幅破坏，同时允许高层特征缓慢适应；Head 保持较高学习率以快速收敛。

参考实践：
- BERT 微调常用 `lr=2e-5` for backbone, `lr=1e-4` for classifier
- 计算机视觉迁移学习中，backbone 微调通常取分类头学习率的 1/10 ~ 1/100

### 2.2 训练配置

| 参数 | 原值 (v4) | 新值 (finetune) | 变更原因 |
|:--|:--|:--|:--|
| `freeze_backbone` | `true` | **`false`** | 核心变更 |
| `epochs` | 80 | **20** | 全参数微调收敛更快，避免过拟合 |
| `learning_rate` (head) | `1e-3` | **`1e-4`** | 降低以避免覆盖 backbone 梯度 |
| `learning_rate` (backbone) | N/A | **`1e-5`** | 新增，保护预训练权重 |
| `lr_t_0` | 15 | **10** | 适配更短的训练周期 |
| `early_stopping_patience` | 15 | **8** | 适配更短的训练周期 |
| `weight_decay` | `1e-4` | **`5e-5`** (backbone), `1e-4` (head) | backbone 用更小的正则化 |
| `checkpoint_dir` | `models/corn_v4` | **`models/corn_v4_finetune`** | 隔离输出，不覆盖原模型 |
| 其他 | 不变 | 不变 | — |

### 2.3 训练策略

1. **从 v4 最佳 checkpoint 热启动** — 加载 `models/corn_v4/best_model.pth`，而不是从头训练。这样 Head 已经有了良好的初始权重，微调过程直接专注于 backbone 适应
2. **解冻全部参数** — `freeze_backbone=False`
3. **差异化学习率** — 用 `torch.optim.AdamW` 的 `param_groups` 实现
4. **CosineAnnealingWarmRestarts** — T0=10, Tmult=2
5. **保持联合训练模式** — Binary + CORN ordinal 同时训练

---

## 三、需要修改/新建的文件

| 文件 | 操作 | 说明 |
|:--|:--|:--|
| `config_finetune_all.yaml` | **新建** | 微调专用配置，与 config.yaml 隔离 |
| `src/train_finetune.py` | **新建** | 微调训练脚本，支持差异化 LR + 热启动 |
| `src/model.py` | **不改** | 已有 `freeze_backbone` 参数，设为 False 即可 |

### 3.1 `config_finetune_all.yaml` 设计

```yaml
# DFU Wagner 0-5 Grading — Full Fine-tuning Configuration
# 解冻全部参数，差异化学习率微调

model:
  name: convnext_tiny
  num_classes: 7
  pretrained: true
  binary_head: true
  freeze_backbone: false              # ← 核心变更：解冻 backbone

data:
  data_dir: /root/dfu/data/processed
  input_size: 224
  binary: false

training:
  batch_size: 64                      # 不改（与 v4 一致）
  epochs: 20                          # ← 降低：80 → 20
  learning_rate: 1.0e-4              # ← 降低：Head 基础学习率
  learning_rate_backbone: 1.0e-5     # ← 新增：Backbone 学习率
  weight_decay: 1.0e-4               # Head weight decay
  weight_decay_backbone: 5.0e-5      # ← 新增：Backbone weight decay

  optimizer: adamw
  lr_scheduler: cosine_warm_restart
  lr_t_0: 10                         # ← 降低：15 → 10
  lr_t_mult: 2
  lr_min: 1.0e-6

  early_stopping_patience: 8         # ← 降低：15 → 8

  use_amp: true
  use_corn: true
  focal_gamma: 2.0
  use_class_weights: true
  joint_training: true
  binary_loss_weight: 0.5

  # 热启动
  resume_checkpoint: /root/dfu/models/corn_v4/best_model.pth  # ← 新增

logging:
  log_interval: 20
  checkpoint_dir: /root/dfu/models/corn_v4_finetune            # ← 新目录
  tensorboard_dir: /root/dfu/models/runs_finetune              # ← 新目录
  csv_log: /root/dfu/models/corn_v4_finetune/training_log.csv

seed: 42
device: cuda
```

### 3.2 `src/train_finetune.py` 核心逻辑

与 `src/train.py` 的主要差异：

```
1. 加载 v4 checkpoint 热启动
   checkpoint = torch.load(resume_checkpoint)
   model.load_state_dict(checkpoint["model_state_dict"])
   # 不加载 optimizer state（因为参数组结构变了）

2. 构建差异化 optimizer param_groups
   backbone_params = []
   head_params = []
   for name, param in model.named_parameters():
       if param.requires_grad:
           if name.startswith("backbone."):
               backbone_params.append(param)
           else:
               head_params.append(param)
   
   optimizer = AdamW([
       {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone},
       {"params": head_params, "lr": lr_head, "weight_decay": wd_head},
   ])

3. 训练循环、验证、测试与 train.py 相同
   （复用 train_epoch / validate_epoch 函数）
```

---

## 四、预期结果与风险评估

### 4.1 预期提升

| 指标 | v4 (冻结) | 期望 (微调后) | 理由 |
|:--|:--|:--|:--|
| Test Accuracy | ~60-65% | **65-72%** | Backbone 适应 DFU 特征 |
| Test Macro F1 | ~35-45% | **42-55%** | 中间级别区分度提升 |
| Grade 1-3 F1 | 38-57% | **48-65%** | 高层特征学习细粒度差异 |
| Grade 4/5 F1 | 20-40% | **30-50%** | 更好的坏死组织特征表示 |
| Binary Accuracy | 99.08% | **≥99%** | 保持或略微提升 |

### 4.2 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|:--|:--|:--|:--|
| **过拟合** — 全参数微调在 ~10K 张图上容易过拟合 | 中 | 中 | 差异化 LR + 早停 patience=8 + weight decay |
| **灾难性遗忘** — backbone 丢失 ImageNet 通用特征 | 低 | 高 | `lr_backbone=1e-5` 很小，缓慢适应 |
| **无显著提升** — backbone 特征已经足够好 | 中 | 低 | 20 轮后对比 v4 baseline，若未提升则回退 |
| **显存不足** — 全参数训练 + AMP 可能 OOM | 低 | 高 | batch_size=64 + AMP 应该安全；若 OOM 降至 32 |

### 4.3 成功标准

- **显著提升**：Test Macro F1 相对提升 ≥ 5 个百分点 → 合并为新的默认方案
- **小幅提升**：Test Macro F1 相对提升 1-5 个百分点 → 保留为可选方案，更新文档
- **无提升/下降**：回退到冻结策略，可能考虑其他方案（如更换更大 backbone、增加数据等）

---

## 五、执行步骤

| 步骤 | 内容 | 预计时间 |
|:--|:--|:--|
| 1 | 创建 `config_finetune_all.yaml` | 即时 |
| 2 | 创建 `src/train_finetune.py` | 即时 |
| 3 | 确认 GPU 可用 + 数据路径正确 | 1 min |
| 4 | 运行训练 `python src/train_finetune.py` | ~30-60 min (20 epochs, batch_size=64, ~10K 训练图片) |
| 5 | 对比 v4 baseline 评估结果 | 5 min |
| 6 | 若有效：更新 config.yaml 默认配置 | 即时 |

---

## 六、后续迭代方向（若本次微调有效）

1. **渐进式解冻** — 先解冻最后 N 层，逐步解冻更多层
2. **更大的 backbone** — ConvNeXt-Small / ConvNeXt-Base
3. **更强的增强** — MixUp/CutMix 在线增强
4. **更长训练** — 微调 40-60 epochs 看是否继续提升

---

*本计划由 Claude 基于 v4 代码分析生成，待用户审阅确认后执行。*
