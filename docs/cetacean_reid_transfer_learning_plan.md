# 鲸豚个体识别：迁移学习与双主线研究方案

> 整理日期：2026-08-11（v2，方向调整后重写）
>
> 旧版（2026-08-07）假设"外部鲸豚 → 本土 supervised fine-tune"为主路线。
> 经核实本土数据**不具备可靠全局 individual_id**（详见 `docs/anchor_pool_semantics.md`），
> 因此迁移学习路线重构为两条独立主线（见下文）。

---

## 1. 核心结论（方向调整）

- 本土数据：`folder ≠ individual_id`。历史整理只挑选了代表照片（Anchor），其余照片未严格归档、混入公共池（Pool）。
- 本土数据**不承担监督训练**，只承担：真实目标域 zero-shot 检索、案例展示、定性分析。
- 所有定量实验（Recall@K / mAP / 消融）放在**有可靠 individual_id 的公开鲸豚数据**上完成。
- 迁移学习的价值是：公开数据上学到的表征，能否 zero-shot 迁移到本土真实场景。

---

## 2. 主路线 A：公开可靠数据上的定量研究

### 2.1 核心问题

> 不同预训练视觉表征以及鲸豚领域迁移学习，对鲸豚个体 Re-ID 的效果有什么差异？

### 2.2 优先比较的模型

| 编号 | 模型 | 类型 |
|---|---|---|
| A | DINOv2 | 通用自监督视觉表征（对照组） |
| B | MegaDescriptor（WildlifeTools） | Wildlife Re-ID 预训练 |
| C | MiewID | Wildlife Re-ID（WildMe） |
| D | 鲸豚数据进一步训练后的模型 | cetacean-domain adaptation |

### 2.3 实验设计（逐级递进）

```text
A. 通用预训练模型 zero-shot
B. Wildlife Re-ID 模型 zero-shot
C. 通用模型 → 公开鲸豚数据 fine-tune
D. Wildlife Re-ID 模型 → 公开鲸豚数据 fine-tune
E. 多鲸豚数据训练 → 不同物种测试
```

### 2.4 数据划分（防泄漏）

- 禁止 random image split（连拍近重复帧会造成虚假高指标）；
- 划分层级优先：`encounter-level > date-level > sequence/burst-level > image-level`；
- 公开数据若无 encounter 信息，必须在实验报告中明确说明限制。

### 2.5 指标

```text
Recall@1 / Recall@5 / Recall@10 / mAP
```

### 2.6 统一数据格式（identity 必须 namespace）

```csv
image_path,identity,species,source_dataset,encounter_id,date,viewpoint,split
```

```text
happywhale__id_001
ndd20__id_014
beluga__id_003
```

禁止不同数据源 ID 冲突。

---

## 3. 主路线 B：中华白海豚真实无标签场景验证

### 3.1 流程

```text
历史代表照片 / Anchor
        ↓
pretrained embedding model（zero-shot，不训练）
        ↓
embedding + L2 normalize
        ↓
cosine similarity
        ↓
unresolved image pool
        ↓
Top-K candidates（供人工审核）
```

### 3.2 输出约束

- 只输出 `Candidate Retrieval Results`；
- **不得自动宣称** `candidate == Anchor`；
- 允许展示：`source_path / source_group / quality_group / similarity`；
- 禁止给出：`individual_id`、`same_individual=true`。

### 3.3 本土实验定位

```text
qualitative evaluation
real-world case study
deployment demonstration
```

无可靠专家 Ground Truth 前，**不得计算并宣称本土识别准确率**。

---

## 4. 迁移学习的新定位

不再默认：

```text
公开鲸豚 → 本土 supervised fine-tune
```

改为：

```text
General Vision Model
        ↓
Wildlife Re-ID Pretraining
        ↓
Cetacean Public Dataset
        ↓
cetacean-domain embedding
        ↓
中华白海豚 zero-shot retrieval（仅测试，不训练）
```

外部公开数据负责让模型学习：dorsal fin / body markings / scars / pigmentation / shape / notches / cetacean appearance。
本土照片只负责测试：这种表征能否迁移到真实历史数据。

---

## 5. 本土数据禁止事项

在无可靠专家身份标签之前：

- 不得把 Source Group 当作 individual_id；
- 不得把 Anchor folder 当作 individual class；
- 不得把 70-79 / 80+ 当作 class；
- 不得进行 CrossEntropy / ArcFace / Triplet / Contrastive 监督训练；
- 不得「模型预测 → 伪标签 → 重新训练 → 继续自动扩展」（confirmation bias 风险）。

---

## 6. 聚类与伪标签降级（Optional / Future）

- HDBSCAN / Candidate Cluster / 伪标签迭代 / Catalogue 重建 → 全部降级为 Optional / Future Work；
- 仅保留探索用途：embedding structure、potential groups、outlier discovery；
- **不得作为 Ground Truth 产生器**。

---

## 7. 公开数据集优先级与下载现状（2026-08-11 实测）

| 优先级 | 数据集 | 状态 | 备注 |
|---|---|---|---|
| 1 | Happywhale（Kaggle） | ❌ Kaggle 手机号验证失败，官方渠道不可用 | 备选：hf-mirror 社区镜像（40GB，含 individual_name/species_name，分片可下） |
| 2 | NDD20 | ⚠️ data.ncl.ac.uk 为 JS 渲染，curl 拿不到直链，待解析 | CC BY-NC-SA 4.0，背鳍场景最接近 |
| 3 | NOAA Choctawhatchee Dolphin | ⚠️ NCEI 302 重定向待解析 | 背鳍 + notch 相关 |
| 4 | BelugaID 2022（LILA） | ✅ 直链可用（122MB，Azure Blob） | 纹理型 Re-ID，encounter 级划分；许可待确认 |

下载与实验挂待办，路径待确认（候选：`D:\dolphin_data\`）。

---

## 8. 参考仓库（SOURCE_MAP 索引）

| 仓库 | 用途 | 状态 |
|---|---|---|
| WildlifeDatasets（wildlife-datasets） | 统一数据接口 / benchmark | 待接入 |
| WildlifeTools（wildlife-tools） | MegaDescriptor / pipeline / evaluation | 待接入（adapter 封装） |
| MiewID（wbia-plugin-miew-id） | wildlife embedding baseline | 参考，adapter 封装 |
| DINOv2 | 通用表征 baseline | 已列入实验 A |
| Happywhale 1st place（knshnb） | 预处理 / ArcFace / 检索参考 | 仅借鉴思路 |
| CetaMatch | dorsal-fin 预处理 / 检索 | 重点借鉴，不复制训练假设 |
| Faiss | 大规模 Top-K | 接口预留（当前 NumPy 足够） |
| PyTorch Metric Learning | 公开数据上的 metric loss | 仅公开数据使用 |

---

## 9. 推荐实验目录结构

```text
configs/            # 实验配置（yaml）
data/               # 数据（公开数据不复制进仓库，放独立目录）
src/reid/
├── dataset/        # Dataset 接口 + adapters
├── embedding/      # Embedding 接口 + adapters（DINOv2 / MegaDescriptor / MiewID）
├── retrieval/      # 检索接口（cosine Top-K，Faiss 可替换）
└── evaluation/     # 评估接口（Recall@K / mAP，仅公开数据）
scripts/            # 实验入口
outputs/            # 实验产物（可追溯）
```

---

## 10. 当前优先级

```text
可靠公开数据 Benchmark
> 统一 embedding / retrieval pipeline
> 鲸豚领域迁移实验
> 中华白海豚 zero-shot retrieval
> 定性案例与失败分析
> Open-set
> 本土监督训练
> 聚类 / 伪标签
> 复杂模型创新
```

核心原则：

```text
Use reliable labels where reliable labels exist.
Do not invent labels where they do not.
```
