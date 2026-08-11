# 鲸豚个体识别：跨物种迁移学习与少样本本土适配方案

> 目的：在本土鲸豚个体识别数据量较少的情况下，利用公开鲸豚 / 动物 Re-ID 数据学习通用视觉表示，再用本土数据微调；重点验证“源域主要依赖背鳍轮廓，而目标域更依赖斑点 / 色素纹理”时，迁移学习是否仍然有效。
>
> 整理日期：2026-08-07

---

## 1. 当前问题

现有本土数据量较少，直接从头训练个体识别模型容易出现：

- 过拟合；
- 模型记住背景、拍摄时间、相机风格，而不是鲸豚本体；
- 每个个体样本过少，普通分类器泛化困难；
- 对新个体（open-set）识别能力弱；
- 同一次 encounter 连拍照片若随机划分，会造成严重数据泄漏。

一个可行方向是：

1. 先使用较大规模的外部鲸豚数据学习通用的鲸豚视觉表示；
2. 学习背鳍定位、轮廓、缺口、伤疤、姿态、海面环境等通用信息；
3. 再使用本土数据微调，让模型学习目标物种特有的斑点 / 色素纹理；
4. 最终以 Re-ID / Metric Learning，而不是普通封闭集分类，作为主要任务形式。

---

# 2. 核心判断

## 2.1 外部鲸豚没有斑点，并不意味着不能迁移

外部纯色鲸豚仍然能够提供以下信息：

- dorsal fin 的基本形态；
- 前缘 / 后缘；
- notch / nick / 缺口；
- 伤疤；
- fin 与海水、浪花、天空的分离；
- 拍摄尺度变化；
- 侧视、斜视等姿态变化；
- 遮挡；
- 不同光照条件；
- “同一个个体的不同图片应该映射到接近的 embedding”这一 Re-ID 能力。

因此应该迁移的是：

**cetacean / dorsal-fin representation**

而不是：

**某一种源物种的个体分类规则。**

---

## 2.2 最大风险：Negative Transfer

如果：

- 外部数据主要通过背鳍轮廓识别；
- 本土物种主要通过斑点 / pigmentation 区分；

那么直接执行：

```text
外部鲸豚 individual classification
        ↓
替换 classifier
        ↓
本土 individual classification
```

可能让模型过度依赖 shape，并忽略 texture。

因此需要用消融实验验证：

> 外部鲸豚数据本身有没有价值？
>
> 如果有价值，什么训练方式最适合迁移？

---

# 3. 推荐任务定义

不要优先定义成：

> 输入图片 → 直接预测 whale_001 / whale_002 / whale_003

更推荐定义为：

> 输入鲸豚照片 → 提取 embedding → 与个体数据库计算相似度 → 返回最相似个体 / 判断为新个体

即：

```text
image
  ↓
backbone
  ↓
embedding
  ↓
cosine similarity
  ↓
gallery retrieval
  ↓
Top-K candidates / new individual
```

这是典型：

- Wildlife Re-Identification
- Metric Learning
- Fine-grained Recognition
- Open-set Recognition

---

# 4. 第一阶段不要做太复杂：先完成 4 组核心实验

最重要的是先回答：

> 外部鲸豚数据到底能不能帮助本土数据？

建议先做下面四组。

| 实验 | 初始化 / 预训练 | 本土训练 | 目的 |
|---|---|---|---|
| A | 通用 ImageNet / DINOv2 / MegaDescriptor | 本土数据 | Baseline |
| B | Happywhale supervised Re-ID | 本土数据 | 验证监督式鲸豚迁移 |
| C | Happywhale / NDD20 self-supervised 或 metric pretrain | 本土数据 | 验证减少 source cue bias 后的迁移 |
| D | 多源鲸豚数据（Happywhale + NDD20 + NOAA 等） | 本土数据 | 验证跨物种通用 representation |

如果资源有限，优先顺序：

```text
A → B → D → C
```

因为 A/B 最容易实现。

---

# 5. 一个更现实的模型 Baseline

推荐不要从零写网络。

## Baseline 1：MegaDescriptor

直接使用动物 Re-ID 预训练模型：

```python
import timm

model = timm.create_model(
    "hf-hub:BVRA/MegaDescriptor-T-224",
    pretrained=True,
    num_classes=0,
)
```

如果显存允许，可以继续尝试：

```text
MegaDescriptor-S-224
MegaDescriptor-B-224
MegaDescriptor-L-384
MegaDescriptor-DINOv2-518
```

优点：

- 本身就是 Animal Re-ID foundation model；
- 比直接从 ImageNet CNN 开始更贴近任务；
- 可以非常快地得到 zero-shot / fine-tune baseline；
- WildlifeTools 已经提供 feature extraction、similarity、retrieval 和 fine-tuning 工具。

模型页面：

https://huggingface.co/BVRA/MegaDescriptor-T-224

https://huggingface.co/BVRA/MegaDescriptor-S-224

https://huggingface.co/BVRA/MegaDescriptor-B-224

https://huggingface.co/BVRA/MegaDescriptor-L-384

---

# 6. Loss 建议

第一版优先：

```text
ArcFace
```

或者：

```text
Sub-center ArcFace
```

原因：

- Happywhale 顶级方案大量使用 ArcFace 系方法；
- Where's Whale-do? 获奖方案也大量使用 ArcFace / Sub-center ArcFace；
- 适合 individual ID 数量多、每类样本较少的 embedding 学习。

之后再消融：

```text
Triplet Loss
Supervised Contrastive Loss
ArcFace + Triplet
```

不要第一版同时上四五种 loss。

---

# 7. 数据划分：这是整个实验最容易踩坑的地方

## 禁止随机按图片划 train / val / test

例如同一次拍摄：

```text
whale_001_001.jpg
whale_001_002.jpg
whale_001_003.jpg
```

随机划分后：

```text
001 → train
002 → train
003 → test
```

模型可能只是记住：

- 海浪；
- 光照；
- 相机位置；
- 拍摄角度；
- 连续帧外观。

结果会虚高。

---

## 应优先按以下层级划分

如果数据有 encounter：

```text
encounter-level split
```

否则：

```text
date-level split
```

再不行：

```text
sequence / burst-level split
```

同一 encounter 只能存在于一个 split。

---

# 8. Closed-set 与 Open-set 最好分开评估

## Closed-set

测试个体训练阶段已经见过：

```text
Known Individual
      ↓
Known Individual
```

指标：

```text
Top-1 Accuracy
Top-5 Accuracy
Recall@1
Recall@5
mAP
```

---

## Open-set

测试中出现训练阶段没有出现过的新个体：

```text
Query
  ↓
known individual?
  ├─ yes → retrieve ID
  └─ no  → new_individual
```

可以根据 cosine similarity 设置 threshold。

建议记录：

```text
mAP
Recall@K
AUROC（known vs unknown）
F1（new individual detection）
```

---

# 9. 第二阶段：如果 Baseline 证明迁移有效，再做 Shape + Texture 双分支

只有第一阶段结果证明“值得继续”之后，再做：

```text
                    ┌→ RGB / Texture branch
RGB image ──────────┤
                    │
                    └→ segmentation
                           ↓
                       silhouette
                           ↓
                     Shape branch
                           ↓
               feature fusion
                           ↓
                       embedding
```

### Shape branch

重点学习：

- 背鳍轮廓；
- notch；
- trailing edge；
- fin shape；
- scar geometry。

比较适合用：

```text
NDD20
NOAA bottlenose dolphin photo-ID
Happywhale dorsal-fin subset
```

---

### Texture branch

重点学习：

- spots；
- pigmentation；
- scars；
- skin pattern；
- 局部色素变化。

主要依赖：

```text
本土数据
Beluga ID（scar / natural markings）
Happywhale 中有明显 body markings 的样本
```

如果后期需要做 pattern-transfer 消融，也可以考虑非鲸豚的 pattern-based wildlife Re-ID 数据，但它们不应成为主训练源。

---

# 10. 最值得下载的外部数据集

---

## P0：Happywhale - Whale and Dolphin Identification

### 推荐程度

**最高。先下载这个。**

### 数据特点

公开 Kaggle 鲸豚个体识别比赛。

训练部分约：

```text
51,033 images
15,587 individuals
30 species
```

任务直接是 whale / dolphin individual Re-ID。

图像包含：

- dorsal fin；
- lateral body；
- tail / fluke；
- natural markings；
- 多物种；
- 大量现实环境变化。

对当前课题最有价值的是：

> 可以过滤出包含 dorsal fin 的物种 / 图像作为 source domain。

### 数据

https://www.kaggle.com/competitions/happy-whale-and-dolphin

### 第一名代码

https://github.com/knshnb/kaggle-happywhale-1st-place

https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq

### 一个更容易阅读的 Top-100 方案

https://github.com/vadimtimakin/kaggle-happy-whale

这个 repo 明确做了：

```text
YOLOv5 crop
dorsal-fin crop
full-body crop
```

对当前项目非常有借鉴价值。

---

# 11. P0：NDD20 - Northumberland Dolphin Dataset 2020

这是第二推荐的数据。

### 为什么非常适合本课题

它直接包含：

```text
above-water dolphin images
below-water dolphin images
individual IDs
dorsal-fin based identification
segmentation annotations（部分）
```

约：

```text
4,402 images
2,201 above-water
2,201 below-water
```

above-water 个体识别本身就是根据 dorsal fin structure 进行。

部分 above-water 图片还有 fin segmentation mask，可用于：

```text
dorsal fin detector
segmentation
shape branch
```

### 官方数据 DOI

https://doi.org/10.25405/data.ncl.c.4982342

### 数据页面

https://data.ncl.ac.uk/articles/dataset/NDD20_zip/12357383

### 论文

https://arxiv.org/abs/2005.13359

### License

```text
CC BY-NC-SA 4.0
```

因此如果未来不是纯科研用途，要重新确认授权条件。

---

# 12. P0：NOAA Choctawhatchee Bay Bottlenose Dolphin Photo-ID

这个数据集规模不如 Happywhale，但它和“背鳍”高度相关。

### 特点

NOAA 的 common bottlenose dolphin photo-identification 数据。

包含：

- dorsal fin photographs；
- unique dolphin ID；
- sighting history；
- fin distinctiveness；
- photo quality；
- notch / non-trailing-edge characteristics；
- encounter 信息。

记录中有约：

```text
188 unique identified dolphins
```

完整 archive 约：

```text
219 MB
```

### 官方 landing page

https://www.fisheries.noaa.gov/inport/item/26481

### 直接 archive

https://www.ncei.noaa.gov/archive/accession/0237742/1.1/

### metadata

https://www.fisheries.noaa.gov/inport/item/67423

这个数据尤其适合：

```text
Shape pretraining
Dorsal fin crop
Notch / edge representation
Encounter-safe split experiments
```

---

# 13. P1：Beluga ID 2022 / Where's Whale-do?

注意：

> Beluga 没有典型 dorsal fin。

所以它不是 Shape branch 最好的数据。

但它非常适合验证：

```text
scar / marking / texture based Re-ID
```

### 数据规模

约：

```text
5,902 photos
788 individual belugas
1,617 encounters
```

包含：

```text
individual ID
viewpoint: top / left / right
encounter metadata
```

专家主要通过：

```text
scarring
natural visual markings
```

进行识别。

### 数据

https://lila.science/datasets/beluga-id-2022/

### DrivenData 比赛

https://www.drivendata.org/competitions/96/beluga-whales/

### 获奖模型代码

https://github.com/drivendataorg/wheres-whale-do

这个 winning repo 很值得看。

获奖方案主要包含：

```text
EfficientNet / ConvNeXt
ArcFace
Sub-center ArcFace
k-fold
embedding retrieval
re-ranking
horizontal flip TTA
```

这和本项目的目标非常接近。

---

# 14. P1：Humpback Whale Identification

这是较早的 Kaggle whale Re-ID 比赛。

### 数据

https://www.kaggle.com/competitions/humpback-whale-identification

它主要依赖 tail / fluke，不如 Happywhale 和 NDD20 贴近 dorsal-fin 场景。

但是可以用于：

```text
Re-ID pipeline 验证
metric learning
ArcFace / retrieval 代码参考
general whale-domain representation
```

优先级低于：

```text
Happywhale
NDD20
NOAA dorsal-fin
BelugaID
```

---

# 15. P2：Happywhale / OBIS-SEAMAP 的持续数据

如果后续需要更多鲸豚图片做：

```text
self-supervised pretraining
domain pretraining
unlabelled feature learning
```

可以考虑 Happywhale 的公开数据工具。

### Happywhale download 工具文档

https://happywhale.openoceans.xyz/projects/download/

### Search / export

https://happywhale.openoceans.xyz/projects/search-export/

可以按：

```text
species
date
geographic area
```

搜索 / 下载。

注意：

> 这部分一定要逐条检查 image license / dataset license。
>
> 不要默认“网上能下载 = 可以随意用于论文训练或公开再分发”。

---

# 16. 数据集总入口：WildlifeDatasets

强烈建议装。

GitHub：

https://github.com/WildlifeDatasets/wildlife-datasets

文档：

https://wildlifedatasets.github.io/wildlife-datasets/datasets/

当前工具集中已经整理了大量 Wildlife Re-ID 数据，包括：

```text
HappyWhale
BelugaID
NDD20
HumpbackWhaleID
NOAARightWhale
SealID
SeaTurtleID
WhaleSharkID
...
```

主要价值：

```text
统一 dataset API
统一 metadata
统一下载
统一 split
统一 evaluation
```

示例：

```python
from wildlife_datasets import datasets

# 具体类名以当前版本文档为准
dataset = datasets.SomeDataset("data/SomeDataset")
print(dataset.df)
```

---

# 17. 最值得直接借鉴的代码仓库

下面这些可以直接交给 Claude Code / Codex 阅读。

---

## 17.1 WildlifeDatasets

```text
https://github.com/WildlifeDatasets/wildlife-datasets
```

用途：

```text
dataset zoo
metadata
download
split
evaluation
```

---

## 17.2 WildlifeTools

```text
https://github.com/WildlifeDatasets/wildlife-tools
```

这是当前项目非常值得优先借鉴的 repo。

已经支持：

```text
training
feature extraction
cosine similarity
retrieval
classification
MegaDescriptor
WildFusion
local feature matching
```

非常适合作为本项目 Re-ID pipeline 的基础。

---

## 17.3 Happywhale Kaggle 第一名

```text
https://github.com/knshnb/kaggle-happywhale-1st-place
```

重点看：

```text
dorsal-fin preprocessing
species handling
ArcFace / metric learning
cropping
embedding inference
```

---

## 17.4 Happywhale 第一名另一位队员代码

```text
https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq
```

用于对照第一名完整实现。

---

## 17.5 一个结构较容易理解的 Happywhale 方案

```text
https://github.com/vadimtimakin/kaggle-happy-whale
```

特别值得看它的：

```text
YOLOv5 dorsal fin crop
full-body vs dorsal-fin preprocessing
```

---

## 17.6 Where's Whale-do? 全部获奖方案

```text
https://github.com/drivendataorg/wheres-whale-do
```

非常适合当前项目的 pigmentation / scar 路线。

README 已经汇总 1~4 名方案。

主要技术：

```text
EfficientNet
ConvNeXt
ArcFace
Sub-center ArcFace
adaptive margin
embedding retrieval
re-ranking
TTA
```

---

## 17.7 MiewID

```text
https://github.com/WildMeOrg/wbia-plugin-miew-id
```

WildMe 的动物 Re-ID embedding 系统。

重点借鉴：

```text
embedding
similarity matching
wildlife identification pipeline
```

不一定需要直接把整个 WBIA 环境搬进项目，可以只研究它的设计。

---

## 17.8 LightGlue wildlife matching plugin

```text
https://github.com/WildMeOrg/wbia-plugin-lightglue
```

后期如果想做：

```text
global embedding
       +
local feature matching
```

非常值得参考。

例如：

```text
MegaDescriptor / MiewID global similarity
             +
ALIKED / DISK / SuperPoint
             +
LightGlue local matching
```

这种方法特别适合：

```text
斑点
疤痕
局部纹理
局部 notch
```

---

## 17.9 Wildbook

```text
https://github.com/WildMeOrg/Wildbook
```

它不是用来直接抄训练代码的。

主要用于理解完整 wildlife photo-ID 系统：

```text
image
↓
annotation
↓
feature / embedding
↓
matching
↓
candidate review
↓
individual database
```

如果后期项目需要做 Demo / 管理个体库，可以参考。

---

## 17.10 Animal Re-ID 数据集索引

```text
https://github.com/DariaKern/IndividualAnimalRe-IDDatasets
```

当你继续找数据时，这个 repo 很方便。

---

# 18. 推荐项目目录

建议最后整理成：

```text
project/
├── configs/
│   ├── baseline_local.yaml
│   ├── happywhale_pretrain.yaml
│   ├── ndd20_pretrain.yaml
│   └── multisource_pretrain.yaml
│
├── data/
│   ├── local/
│   ├── happywhale/
│   ├── ndd20/
│   ├── noaa_dolphin/
│   └── beluga/
│
├── datasets/
│   ├── base.py
│   ├── local.py
│   ├── happywhale.py
│   ├── ndd20.py
│   ├── noaa.py
│   └── beluga.py
│
├── models/
│   ├── backbone.py
│   ├── embedding.py
│   └── losses.py
│
├── scripts/
│   ├── prepare_happywhale.py
│   ├── prepare_ndd20.py
│   ├── prepare_noaa.py
│   ├── train.py
│   ├── evaluate.py
│   └── extract_embeddings.py
│
├── splits/
│   ├── local_train.csv
│   ├── local_val.csv
│   └── local_test.csv
│
└── results/
    └── experiments.csv
```

---

# 19. 建议统一数据格式

所有 source dataset 都转成一份 metadata：

```csv
image_path,identity,species,source_dataset,encounter_id,date,viewpoint,split
```

例：

```csv
data/happywhale/001.jpg,id_001,bottlenose_dolphin,happywhale,E001,2020-04-03,left,train
data/ndd20/002.jpg,id_014,white_beaked_dolphin,ndd20,E551,2018-07-12,right,train
```

如果某字段没有：

```text
NULL
```

不要编造。

---

# 20. 建议先跑的模型

## Experiment A：MegaDescriptor Zero-shot

甚至先不要训练。

```text
local images
↓
MegaDescriptor
↓
embedding
↓
cosine similarity
↓
Recall@1 / Recall@5 / mAP
```

这能快速判断：

> 一个通用 animal Re-ID 模型在本土鲸豚上已经能做到什么程度？

---

## Experiment B：MegaDescriptor → local fine-tune

```text
MegaDescriptor
↓
local ArcFace fine-tune
↓
evaluation
```

这是最重要 baseline。

---

## Experiment C：MegaDescriptor → Happywhale → local

```text
MegaDescriptor
↓
Happywhale whale/dolphin fine-tune
↓
local fine-tune
↓
evaluation
```

若：

```text
C > B
```

说明跨物种鲸豚 supervision 有帮助。

若：

```text
C < B
```

可能存在 source cue bias / negative transfer。

---

## Experiment D：MegaDescriptor → multi-source cetacean → local

Source：

```text
Happywhale
+
NDD20
+
NOAA dolphin
+
BelugaID
```

注意 identity label 必须 namespace：

错误：

```text
id_001
```

因为不同 dataset 可能重复。

正确：

```text
happywhale__id_001
ndd20__id_001
beluga__id_001
```

---

# 21. 一个非常有价值的额外消融：Shape-only vs RGB

如果能够得到 dorsal-fin mask，可以构造：

```text
Original RGB
```

和：

```text
Silhouette / edge image
```

分别训练。

实验：

| 模型 | 输入 |
|---|---|
| RGB | 原图 |
| Shape | binary silhouette / contour |
| RGB+Shape | feature fusion |

如果：

```text
外部鲸豚 Shape 很强
本土 Shape 一般
本土 RGB 明显更强
```

就能直接支持论文中的核心判断：

> Source species and target species rely on different discriminative visual cues.

这个结果本身很有研究价值。

---

# 22. Data Augmentation 注意事项

目标域依赖 spots / pigmentation 时，不要使用过强的：

```text
ColorJitter
Grayscale
Solarize
Posterize
```

否则可能主动破坏身份特征。

优先：

```text
RandomResizedCrop（幅度保守）
HorizontalFlip（先确认左右侧是否语义一致）
small Rotation
mild Brightness / Contrast
Blur（小概率）
Random Erasing（谨慎）
```

特别注意：

如果鲸豚左侧和右侧斑点不同，则：

```text
HorizontalFlip
```

可能改变生物学意义。

最好把：

```text
left / right
```

保留成 metadata，并单独做实验。

---

# 23. 论文 / 毕设最漂亮的研究问题

可以将研究问题定义成：

> Can cross-species cetacean pretraining improve individual photo-identification when source and target species rely on different visual identity cues?

中文：

> 当源物种与目标物种依赖不同个体判别特征时，跨物种鲸豚预训练能否提升少样本个体照片识别性能？

核心 hypothesis：

### H1

跨物种鲸豚数据可以提升本土少样本 Re-ID。

### H2

直接 supervised source-ID pretraining 可能因为 source cue bias 出现负迁移。

### H3

更通用的 metric / self-supervised representation 或多源训练，比单一源物种 individual classification 更适合迁移。

### H4

目标物种的 RGB pigmentation 特征与通用 dorsal-fin shape 特征存在互补性。

---

# 24. 最小可完成版本（本科项目强烈推荐）

不要一开始做：

```text
segmentation
+
dual branch
+
contrastive
+
ArcFace
+
LightGlue
+
open set
+
web demo
```

这样很容易做不完。

第一版只做：

```text
1. 整理本土 metadata
2. 做 encounter-safe split
3. MegaDescriptor zero-shot
4. MegaDescriptor local fine-tune
5. Happywhale → local
6. multi-source cetacean → local
7. Recall@1 / Recall@5 / mAP
8. 写迁移效果分析
```

能完成这 8 个步骤，已经足够形成一个完整、可解释的实验章节。

---

# 25. 给 Claude Code / Codex 的直接提示词

下面可以整段复制给 CC。

```text
我正在做一个鲸豚个体照片识别（wildlife re-identification）项目。

背景：
我的本土目标物种数据量较少，而且目标物种可能主要依赖背鳍/身体上的斑点、色素纹理、伤疤等特征区分个体；外部鲸豚数据中很多物种主要依靠 dorsal-fin shape、notches 和 scars 区分。

我希望验证跨物种迁移学习是否有效，而不是直接假设外部数据一定有帮助。

请先阅读并借鉴以下仓库，不要直接复制成一个巨大系统：

https://github.com/WildlifeDatasets/wildlife-datasets
https://github.com/WildlifeDatasets/wildlife-tools
https://github.com/knshnb/kaggle-happywhale-1st-place
https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq
https://github.com/vadimtimakin/kaggle-happy-whale
https://github.com/drivendataorg/wheres-whale-do
https://github.com/WildMeOrg/wbia-plugin-miew-id
https://github.com/WildMeOrg/wbia-plugin-lightglue
https://github.com/WildMeOrg/Wildbook
https://github.com/DariaKern/IndividualAnimalRe-IDDatasets

优先外部数据：

1. Happywhale
https://www.kaggle.com/competitions/happy-whale-and-dolphin

2. NDD20
https://doi.org/10.25405/data.ncl.c.4982342
https://data.ncl.ac.uk/articles/dataset/NDD20_zip/12357383

3. NOAA Choctawhatchee Bay bottlenose dolphin photo-ID
https://www.fisheries.noaa.gov/inport/item/26481
https://www.ncei.noaa.gov/archive/accession/0237742/1.1/

4. Beluga ID 2022
https://lila.science/datasets/beluga-id-2022/

5. Where's Whale-do challenge
https://www.drivendata.org/competitions/96/beluga-whales/

推荐 backbone：
https://huggingface.co/BVRA/MegaDescriptor-T-224

请首先检查我当前项目的代码结构和已有本土数据格式，然后以“最小可验证实验”为目标实现，不要一开始做复杂双分支模型。

第一阶段需要完成：

A. 建立统一数据 metadata schema：

image_path
identity
species
source_dataset
encounter_id
date
viewpoint
split

所有不同数据集的 identity 必须加 namespace，例如：

happywhale__xxx
ndd20__xxx
beluga__xxx

禁止不同数据源的 ID 冲突。

B. 实现 data adapters：

LocalDataset
HappyWhaleDataset
NDD20Dataset
NOAADolphinDataset
BelugaDataset

要求 adapter 最终输出统一 metadata dataframe。

C. 实现防止数据泄漏的数据划分。

优先级：

encounter-level split
> date-level split
> sequence-level split
> image-level split

同一 encounter 的图片绝对不能同时出现在 train 和 val/test。

D. 先实现 MegaDescriptor zero-shot baseline。

使用：

hf-hub:BVRA/MegaDescriptor-T-224

流程：

image
→ embedding
→ L2 normalize
→ cosine similarity
→ gallery retrieval

输出：

Recall@1
Recall@5
Top-1
Top-5
mAP

E. 再实现 local fine-tuning baseline。

优先使用：

ArcFace

模型训练后仍然输出 embedding，不要只做 softmax classifier。

F. 配置化运行以下实验：

exp_A:
MegaDescriptor zero-shot → local evaluation

exp_B:
MegaDescriptor → local ArcFace fine-tune

exp_C:
MegaDescriptor → Happywhale pretrain → local fine-tune

exp_D:
MegaDescriptor → Happywhale + NDD20 + NOAA + Beluga multi-source pretrain → local fine-tune

目标是比较：

A vs B vs C vs D

判断外部 cetacean data 是否真的带来收益。

G. 所有实验结果写入：

results/experiments.csv

至少包含：

experiment
pretrain_dataset
backbone
loss
split_strategy
recall_at_1
recall_at_5
map
notes

H. 暂时不要实现：

复杂 GUI
完整 Wildbook
复杂双分支
LightGlue fusion
大规模 hyperparameter search

这些全部放到 baseline 跑通以后。

I. 在修改代码前先：
1. 阅读当前 repo
2. 告诉我现有代码哪些可以复用
3. 给出最小改动计划
4. 然后再开始修改

工程目标：
先得到一个结构清楚、实验可复现、能回答“跨物种鲸豚迁移到底有没有用”的 baseline。
```

---

# 26. 建议的数据下载顺序

如果硬盘和网络有限：

```text
1. NDD20
2. NOAA Choctawhatchee
3. Beluga ID
4. Happywhale
```

原因：

NDD20 / NOAA 规模较小，能先把 pipeline 跑通。

如果硬盘充足：

```text
1. Happywhale
2. NDD20
3. NOAA
4. Beluga ID
```

然后统一转换 metadata。

---

# 27. 我目前最推荐的执行顺序

```text
Step 1
现有本土数据 audit

↓

Step 2
MegaDescriptor zero-shot

↓

Step 3
MegaDescriptor + local fine-tune

↓

Step 4
下载 NDD20
测试 NDD20 → local

↓

Step 5
下载 Happywhale
测试 Happywhale → local

↓

Step 6
multi-source cetacean

↓

Step 7
如果结果证明 external pretraining 有价值
再开发 Shape + Texture 双分支

↓

Step 8
最后考虑 local feature matching / LightGlue / open-set threshold
```

---

# 28. 数据许可提醒

以下数据的许可条件并不完全相同。

例如：

```text
NDD20:
CC BY-NC-SA 4.0

MegaDescriptor 模型：
具体模型页面标注 CC-BY-NC-4.0

WildlifeDatasets：
代码 license 与各数据集 license 是两回事
```

因此：

- 学术研究通常问题较少；
- 论文中要正确 citation；
- 不要随意把第三方图片重新打包上传到自己的 GitHub；
- 数据集 README / license 应单独保留；
- 如果未来商业化，要重新逐项检查授权。

---

# 29. 最终建议

当前最值得做的不是马上设计新网络，而是先验证：

```text
外部鲸豚 domain knowledge
是否真的能够帮助本土数据？
```

最有信息量的比较是：

```text
MegaDescriptor
vs
MegaDescriptor + local
vs
MegaDescriptor + Happywhale + local
vs
MegaDescriptor + multi-source cetacean + local
```

如果外部迁移有效，再继续研究：

```text
shape vs pigmentation
```

这样实验逻辑清楚、风险低，也更容易形成完整论文故事。
