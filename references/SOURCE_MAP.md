# SOURCE_MAP：外部参考与数据源记录

> 记录所有外部仓库、数据源、镜像的引用方式与来源，保证可追溯。
> 更新日期：2026-08-25

---

## 一、参考代码仓库

| 仓库 | URL | License | 用途 | 状态 |
|---|---|---|---|---|
| WildlifeDatasets | https://github.com/WildlifeDatasets/wildlife-datasets | GPL-3.0 | 统一数据接口 / benchmark | 待接入 |
| WildlifeTools | https://github.com/WildlifeDatasets/wildlife-tools | GPL-3.0 | MegaDescriptor / pipeline / evaluation | 待接入（adapter 封装） |
| MiewID | https://github.com/WildMeOrg/wbia-plugin-miew-id | 无 LICENSE | wildlife embedding baseline | 参考，adapter 封装 |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Apache-2.0 | 通用视觉表征 | 计划实验 A |
| Happywhale 1st place | https://github.com/knshnb/kaggle-happywhale-1st-place | 无明确 | 预处理 / ArcFace / 检索参考 | 仅借鉴思路 |
| CetaMatch | https://github.com/The-Dolphin-Project/cetamatch | MIT | dorsal-fin 预处理 / 检索 | 重点借鉴，不复制训练假设 |
| CurvRank | https://github.com/WildMeOrg/wbia-plugin-curvrank | 无明确 | fin contour / notch | 后期视需要 |
| Faiss | https://github.com/facebookresearch/faiss | MIT | 大规模 Top-K | 接口预留（当前 NumPy） |
| PyTorch Metric Learning | https://github.com/KevinMusgrave/pytorch-metric-learning | MIT | metric loss | 仅公开数据使用 |
| Lightly | https://github.com/lightly-ai/lightly | MIT | 自监督 | 非第一阶段 |
| FiftyOne | https://github.com/voxel51/fiftyone | Apache-2.0 | 人工审核 | 非第一阶段 |
| HDBSCAN | https://github.com/scikit-learn-contrib/hdbscan | BSD-3-Clause | 辅助聚类 | 降级 Optional |
| CVAT | https://github.com/cvat-ai/cvat | MIT | 标注 | 非第一阶段 |
| Detectron2 | https://github.com/facebookresearch/detectron2 | Apache-2.0 | 检测 | 非第一阶段 |

---

## 二、公开数据集

| 数据集 | 官方地址 | License | 本地路径 | 状态 |
|---|---|---|---|---|
| Happywhale | https://www.kaggle.com/competitions/happy-whale-and-dolphin | Kaggle 竞赛条款（不可再分发） | `D:\dolphin_data\happywhale\` | ✅ 分片0 已下（1.96GB parquet，2247 图）；adapter 已接入 `experiments/pub_reid/dataset/happywhale.py` |
| NDD20 | https://doi.org/10.25405/data.ncl.c.4982342 | CC BY-NC-SA 4.0 | 未下载 | data.ncl.ac.uk JS 渲染，待解析 |
| NOAA Choctawhatchee | https://www.fisheries.noaa.gov/inport/item/26481 | NOAA 条款 | 未下载 | 302 重定向待解析 |
| BelugaID 2022 | https://lila.science/datasets/beluga-id-2022/ | LILA 条款（需逐项确认） | `D:\dolphin_data\beluga\` | ✅ test.zip + coco.tar.gz 已下（3402 图/978 身份/10 scenario）；adapter 已接入 `experiments/pub_reid/dataset/beluga.py` |

---

## 三、镜像与下载通道（2026-08-11 实测）

### Happywhale（Kaggle 官方不可用，改用 HF 镜像）

- 镜像仓库：`GATE-engine/happy-whale-dolphin-classification`（huggingface.co）
- 镜像地址（经 hf-mirror）：`https://hf-mirror.com/datasets/GATE-engine/happy-whale-dolphin-classification/resolve/main/data/{split}-*.parquet`
- 内容：`image / species / species_name / individual / individual_name`（含 individual_name！）
- 规模：train 42,678 张 / val 2,088 / test 6,267，全量 40GB
- 使用：分片下载（~1.9GB/片），首次只下 1 片跑通 pipeline
- **注意**：镜像无 license 标注，仅供内部研究，不可再分发

### BelugaID（LILA 官方，可直连）

- 容器枚举：`https://lilawildlife.blob.core.windows.net/lila-wildlife?restype=container&comp=list&prefix=wild-me`
- 测试集：`wild-me/beluga-id-test.zip`（122MB，3401 图 + private_test_labels.csv + private_train_metadata.csv）
  - ⚠️ zip 内只有 **test 图片**（无 train 图）；`private_test_metadata.csv` 无 whale_id
- 完整标注：`wild-me/beluga.coco.tar.gz`（590MB，COCO 格式，含 whale_id / encounter_id / viewpoint / date）
  - ✅ 已下载完（2026-08-25）；adapter 按官方 Re-ID benchmark 语义接入
  - ⚠️ original_whale_id 是竞赛未公开身份，只用于内部研究评估，不可再分发
- 官方入口：https://lila.science/datasets/beluga-id-2022/
- 备选镜像：storage.googleapis.com / s3 us-west-2

### 网络环境

- huggingface.co 不可达（SSL 中断）；hf-mirror.com 可达
- Kaggle（kaggle.com + kagglehub）不可用（手机号验证失败）
- data.ncl.ac.uk / NCEI 需浏览器级请求（JS 渲染 / 重定向链）

---

## 四、模型权重通道

| 模型 | 来源 | 通道 | 状态 |
|---|---|---|---|
| MegaDescriptor-T-224 | hf-hub:BVRA/MegaDescriptor-T-224 | hf-mirror（HF_ENDPOINT） | ✅ 已用 |
| DINOv2 ViT-B/14 | facebookresearch/dinov2 官方权重 | 本地 .pth：`D:\dolphin_data\dinov2_weights\dinov2_vitb14_pretrain.pth`（官方 pth 与 timm 键 174/174 匹配，仅多 mask_token，加载时剔除） | ✅ 已用（离线加载） |

### 网络通道实测（2026-08-12）

- **HF 系域名全部 TLS 握手失败**：huggingface.co / hf-mirror.com / cdn-lfs.huggingface.co（直连 + 代理 127.0.0.1:7890 均 exit 35）
- **可用通道**：
  - GitHub（经代理 7890）✅
  - `dl.fbaipublicfiles.com`（Facebook 官方 CDN，经代理 7890）✅ 用于下载 DINOv2 官方权重
  - modelscope.cn ✅（备用，未用）
- timm 权重加载需离线模式：`HF_HUB_OFFLINE=1`（缓存完整时跳过 HEAD 校验）

---

## 五、使用规则

1. 外部代码优先直接依赖成熟库，通过 adapter 封装；
2. 不复制硬编码路径和错误的数据假设；
3. 参考代码后补充本项目测试；
4. 所有处理保留 image_id / source_path / source_group / quality_group 追溯；
5. 公开数据不复制进项目仓库（放 D:\dolphin_data\ 独立目录）。
