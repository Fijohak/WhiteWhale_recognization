# WhiteWhale_recognization — 中华白海豚个体识别

基于野外调查影像，研究可靠的中华白海豚**个体识别方法**（背鳍检测 → 特征学习 → 批内分组聚类 → 跨时间匹配 → 新个体发现），并将方法**落地为辅助归档工具**：输入一批新照片自动按个体分组、每簇取最清晰的一张代表图归档，减少工作人员逐张手工整理的工作量，同时为"这只海豚见过没"提供候选匹配。模型结果一律是 **Candidate（候选）**，正式个体身份须经人工核验（human-in-the-loop）。

组员可先阅读 [系统架构与协作指南](docs/系统架构与组员协作指南.md) 快速了解网站、服务器、数据库和 4060 Laptop Worker 如何配合。协作平台的完整实施边界见 [PLAN.md](PLAN.md)；分阶段说明见 [M1 控制面](docs/platform_m1.md)、[M2 归档](docs/platform_m2.md)、[M3 关系与身份更正](docs/platform_m3.md)、[M4 训练与模型](docs/platform_m4.md)及 [M5 部署交付](docs/platform_m5.md)。

## 1. 核心语义

| 概念 | 说明 |
|---|---|
| 调查批次 | 一次野外调查、日期或航次 |
| 拍摄序列 | 短时间内连续拍摄的一组照片（连拍）；训练/评估按序列划分，不拆分 |
| 批次内已确认个体 | 高分目录数字子文件夹中的 `individual_id`；身份在该调查批次内已确认，但不是跨批次全局 ID |
| Unresolved Image Pool | `70-79` 散图 = 未归属照片池，作检索 Gallery |
| 候选分组 | 原数据中人工或程序初步整理出的图片集合（Candidate Cluster） |
| Candidate ≠ Confirmed | 聚类簇、检索结果都只是候选；人工审核（确认/不确定/拒绝）后才是个体身份 |
| 个体 ID | 经人工核验后确认的白海豚身份（Confirmed Individual） |
| 图像质量 | 图像清晰度、目标大小或人工评分（`70-79` / `80 and above` 等评分区间） |
| 关系备注 | 文件夹名称中记录的候选关联信息 |
| 批内（同一天） | 封闭集归档：检测 → 聚类 → 每簇选代表图 → 人工审核归档 |
| 跨时间（不同批次） | 开放集匹配：新批次与历史个体库匹配，低于未校准阈值标记"疑似新个体"；结果仅作候选，不能自动确认身份 |
| 左右侧分离 | 背鳍两侧特征不同，照片记录朝向（left/right/unknown），左右侧分别比较 |
| 原始数据只读 | 原图在 `src_dataset/` 只读；所有生成物写 `outputs/`，可追溯到原始路径 |

数据集结构、规模、已确认事实与待确认假设见 [§2 数据](#2-数据)。历史实验结果与结论见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 2. 数据

原始数据（原图）不入库，位于 `src_dataset/`。本节是数据理解的参考手册：结构、规模、已确认事实与待确认假设；日常跑系统不需要逐条核对。

### 2.1 目录结构与命名规范

> **命名规范**：数据根目录名 = `YYYYMMDD NN`（年月日 + 空格 + 群体编号，编号仅当日唯一），session 标识 = 完整目录名。当前共 **9 个批次**：20140806 01/03（历史库）、20140418 01、20140419 02/04/05/06、20151017 02/03。结构精简：仅 `70-79` / `80 and above`（部分批次含 `nn relationship`，非必有；数据提供方要求忽略低分数据）。

```text
src_dataset/20140806 01（示例批次，历史库；调查 20140417W01，地点代码 SZi，批次 01）
├── 70-79/                      # 70-79 分组
│   ├── 05/ 11/ 12/ 13/ 14/ …   # 数字子文件夹 = 批次内已确认个体（非全局 ID）
│   └── *.JPG                   # 散图，Unresolved Image Pool（人工跟进中）
├── 80 and above/               # 80 分及以上
│   └── 01/ … 10/               # 数字子文件夹 = 批次内已确认个体
├── 50-59/ 60-69/ below 50/     # 低于 70 分，暂不用于训练
├── MO/  miscellaneous/  nn relationship/   # 拍摄者 / 其他目录
├── 20140417W01.txt             # 调查记录
└── individual.docx             # 人工对照材料

其余批次同构；原始压缩包（只读）：01.zip / processed.zip / Processed 2.zip
```

### 2.2 数据规模（最新 scan 统计）

| 项 | 数量 |
|---|---|
| 总批次 / 总图片数 | 9 批次 / 1040 张（全可读） |
| 已分组个体（labeled） | 75 个体 / 352 张（其中历史库 20140806：43 组 202 张） |
| 散图（loose_known） | 322 张 |
| 忽略（ignored：低分/拍摄者目录等） | 366 张 |
| 质量区间 70-79 / 80 and above | 348 / 326 张 |

> 批次分布：20140806 03（507）、20140806 01（155）、20140419 02（126）、20151017 02（90）、20140419 06（88）、20151017 03（29）、20140419 05（28）、20140419 04（10）、20140418 01（7）。个体数按标签 `{session}_{group_id}` 计，跨调查同名编号未合并；单图个体 22 个，多图个体 53 个，最大个体 27 张（连拍）。

### 2.3 已确认事实

* **评分区间 = 图片分级**：W01 txt 计数（50-59:14, 60-69:100, 70-79:64, 80+:28）与批次目录图片数逐项一致；
* **MO / RAY / DEREK = 拍摄者代码**：txt 人员计数明确；
* **文件名双模式**：RAY 拍摄 `0145_20140417_SZi_01_RAY_0632.JPG`（编号_日期_地点_批次_人员_连拍号）；MO 拍摄 `RES20001.JPG`（无日期字段）；
* **01 = W01（SZi），03 = W03（HBi）**，同一天两个调查批次；两批的分组编号各自独立。`01_02` 与 `03_02` 只表示两个批次内 ID，是否对应同一真实个体尚未对齐；跨批次照片既不能默认同体，也不能默认异体；
* **`nn relationship` 目录 = 疑似亲缘关系个体样本**（2026-08-19 数据提供方确认，**非每个批次必有**）：一图两鳍（同框多头），不参与个体分组，scan_dataset 标记 relation_note=nn_relationship；亲缘关系仍需人工确认（同框 ≠ 有亲缘）。

### 2.4 待确认假设（A1–A13）

| # | 假设 | 状态 |
|---|---|---|
| A1 | 评分区间为质量/匹配置信度分级 | 基本证实（W01 计数一致） |
| A2 | MO/RAY/DEREK 为拍摄者代码 | 已证实 |
| A3 | SZi/HBi 为调查地点代码 | 推测 |
| A4 | 70-79 / 80 and above 数字分组为批次内已确认个体，但编号不是跨批次全局 ID | 已确认（数据提供方） |
| A5 | `sj of 01` = 伴随/相关个体关系 | 推测 |
| A6 | `nn relationship` = 疑似亲缘关系个体样本目录；同框是否确有亲缘仍待人工确认 | 目录语义已确认，具体关系待核验 |
| A7 | processed / Processed 2 为不同批次 | 推测（SHA256 无重叠） |
| A8 | txt 计数为全量，目录为整理后子集 | 推测 |
| A9 | 文件名连拍号 = 连续拍摄序列 | 推测，须抽样核验后用于划分 |
| A10 | RES+数字（MO 拍摄）无连拍信息 | 推测 |
| A11 | 跨时间重复拍摄到同一群/同一只海豚（无全局编号） | **推定**（数据提供方，2026-08-17） |
| A12 | 个体特征跨时间会变化（成长、意外事件致疤痕/轮廓改变） | **推定**（鲸豚领域已知现象） |
| A13 | 批内散图池照片全部属于本批已确认个体（无新个体混入） | 批内既定；跨批不保证 |
| A14 | 背鳍剪影左右近似对称：单侧个体可镜像检索跨侧同体剪影（斑点/伤口等表面特征不对称，须忽略） | 推测（鲸豚领域生物学依据；须轮廓特征支持，见待办 4.3） |

> 所有假设在确认前：不得作为训练标签、不得作为数据划分依据、Manifest 中只能作为带 `_guess` 后缀的启发式字段；推定（Assumed）与既定（Confirmed）必须区分记录（A11/A12 未经数据验证，不依赖其成立即可运行批内任务）。

### 2.5 数据使用要点

> 概念定义见 §1 核心语义表。

* 当前使用全部 9 个批次的 `70-79`、`80 and above` 两个区间（≥70 分）数据；历史库（gallery）= 20140806 01/03 的批次内已确认个体（43 个体、202 张）；
* 这两个区间下的数字子文件夹是批次内已确认个体，可用作监督标签；`individual_id={session}_{group_id}` 通过 session 隔离同名编号，不代表跨批次身份已经对齐；
* 多图个体常来自同一次连拍或相邻时段，时间间隔短。训练和评估必须整串隔离；即使指标良好，也只能说明短期批内辨识能力，不能外推为跨年能力；
* 高分照片可作为该个体的**质量上界先验**（模糊图较难进入高分段），散图收集时可作质量参照（先验非事实，使用须标注）；
* 散图（322 张 loose_known）属于 Unresolved Image Pool：同调查内可能属于某个已选代表照片的个体，但具体归属未确认，只能作为检索 Gallery 的候选，不可作标签；
* 分组编号只在同一调查内有效，跨调查同名编号是否对应同一只海豚需人工核验（当前跨调查对应关系尚未核验）；
* 背鳍有左右两面，现有多图个体仍以短时连拍为主，双侧与长时间跨度覆盖不足；照片须记录朝向（left / right / unknown），左右侧分别比较。

### 2.6 任务定位与数据语义（2026-08-17 与数据提供方对齐）

任务按时间尺度分层，批内与跨时间的语义、任务与可靠性不同：

| 尺度 | 数据语义 | 任务 | 可靠性 |
|---|---|---|---|
| 批内（同一天调查） | 每个已确认个体可视为独立；散图池照片**有主**（属于本批某已确认个体，A13） | 归档分配、池找回（封闭集） | 特征短期稳定，可靠 |
| 跨时间（不同日期批次） | 可能重复拍摄到同一群，个体被单独标记但无全局编号；跨时间同体对为**推定**（A11） | 与历史个体库匹配，发现疑似新个体（开放集） | 特征可能漂移（A12），结果仅为候选，须人工核验 |

要点：

* **项目目标（两者并存）**：研究个体识别方法 + 落地为辅助归档工具（见开头简介）；"新个体识别"不是独立目的，而是低置信度输出的自然副产品；
* **批内主路径 = 自动归档**：检测背鳍 → 裁剪 → embedding → 批内聚类（Candidate Cluster）→ 每簇选最清晰代表图 → 人工审核确认归档；检索用于"与历史库匹配"（这只见过没），聚类用于"这一批有几只"；
* 项目负责人期望：辨别新个体的出现（开放集）；跨时间 Re-ID（个体再识别）不作为当前目标；
* "新个体"判定依赖"认出旧个体"的能力：特征漂移的已知个体（成长、伤病致疤痕/轮廓改变）是最主要的"疑似新个体"假阳性来源，通过多特征通道互补 + 置信度输出 + 人工核验兜底；
* 跨时间匹配结果默认落在 possibly_same 关系，个体档案记录照片时间戳与特征变化历史，支持专家核验；
* 跨时间同体对会随批次积累逐步被发现（匹配 + 人工审核），是"新个体发现"阈值标定的依据；在积累之前，阈值保守、多报候选、人工兜底（宁可拆分不可错并）。

### 2.7 数据使用原则

* **原始数据只读**：所有清洗、裁剪、重命名和整理结果保存在独立目录中，避免破坏原始材料；
* **分层信任目录标签**：数字子文件夹中的 `individual_id` 是批次内已确认个体，可作批内监督标签；分组编号只在同调查内成立，跨批次同名编号不得自动合并；`70-79` 散图归属未定；低于 70 分的区间及 `MO`/`RAY`/`DEREK`/`miscellaneous`/`nn relationship` 等目录暂不用于个体标签；
* **避免重复数据**：数据索引阶段进行文件哈希或图像相似度检查，避免重复统计和重复训练；
* **可追溯**：每张处理后的图片都能追溯到原始文件路径、调查日期、拍摄批次、连续拍摄序列、原始候选分组、图像质量区间、处理方式、人工核验状态。

## 3. 环境

- Python 3.13+，依赖见 `requirements.txt`（torch、timm、fastapi、pandas、numpy、hdbscan、ultralytics、PyYAML）——**新环境先执行 `pip install -r requirements.txt`（网络慢可加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）；运行报 `ModuleNotFoundError` 时请先检查是否已按此安装依赖，再排查其他问题**
- 模型权重与原始数据不入库（`models/`、`*.pt`、`outputs/*` 已被 git 忽略，**请勿直接 commit 权重文件**）：检测器 `models/detectors/yolov8n_dorsalfin.pt`，r4 特征权重 `outputs/metric_learning/r4/best.pt`，原图 `src_dataset/`
- **权重获取方式**：自训练权重（检测器 + r3/r4 + 训练记录）打包为 `whitewhale_weights_2026-08-26.zip`（约 200MB，含包内说明 `README.md`）——从网盘分享或项目 Releases 附件下载，**解压到仓库根目录即可**（路径与 `configs/pipeline.yaml` 自动对齐，无需改配置）；MegaDescriptor-T-224 特征模型首次运行自动下载，无需手动获取；ultralytics 官方预训练等仅重训检测器时才需要
- 所有入口默认值从 `configs/*.yaml` 读取，可用 CLI 参数覆盖

## 4. 正式入口

所有命令在仓库根目录执行。七个正式入口（4.1–4.7），**一个功能一个入口**：

### 4.1 数据盘点 `scripts/prepare_data.py`

扫描原始目录生成数据清单（manifest）：

```bash
python scripts/prepare_data.py scan                # 扫描 configs/pipeline.yaml 的 data_roots
python scripts/prepare_data.py scan --sha256      # 附加 SHA-256 完全重复检测
python scripts/prepare_data.py build-pilot        # 由 manifest 生成批次内已确认个体库
```

`build-pilot` 使用保留来源标签的 `source_label_preserving_v2` ID 规则，例如来源目录 `05` 保持为
`20140806 01_05`，不再经数值解析变成 `5.0`。2026-08-29 的一次性迁移映射和
旧表备份分别保存在 `outputs/pilot/individual_id_migration_v1_to_v2.csv` 与
`outputs/pilot/pilot_set_legacy_ids_20260829.csv`；历史审核包里的旧 ID 须经映射表解析，
不得按字符串直接与当前 ID 拼接。

### 4.2 批内归档管线 `scripts/run_pipeline.py`

新批次全流程：YOLO 背鳍检测裁剪 → r4 特征 → HDBSCAN 批内候选聚类 → 子簇化 → 簇级多帧投票匹配历史库 → 代表图 + 候选簇拼图（人工审核材料）。

```bash
# 散图池验证模式（复用已预提取的池特征产物）
python scripts/run_pipeline.py --pool

# 从全量 manifest 中明确选择一个新批次
python scripts/run_pipeline.py --input-manifest outputs/index/dataset_manifest.csv \
    --session "20140419 02" --batch-name "20140419 02" --sheets
```

| 关键参数 | 默认 | 说明 |
|---|---|---|
| `--pool` / `--input-manifest` | 二选一必填 | 散图池验证 / 新批次清单 |
| `--session` | 多批次 manifest 时必填 | 只处理指定 `session_id`，禁止把全量清单静默当成一个批次 |
| `--gallery-embeddings` | `outputs/artifacts/r4_yolocrop_v3/gallery/embeddings.npy` | 当前活动历史个体库特征（r4 + YOLO 裁剪，严格 provenance） |
| `--threshold-cluster` | 0.58 | 簇级候选阈值（provisional，历史 E5 参考值，未完成当前开放集校准） |
| `--threshold-image` | 0.50 | 单图候选阈值（provisional；旧 E4 跨批次负例假设无效，不能据此解释 FA） |
| `--out` | `outputs/cluster_archival` | 输出目录（内部按 batch_name 分目录） |
| `--sheets` | 关 | 生成候选簇拼图 |

### 4.3 人工审核网页 `scripts/launch_review.py`

审核批内归档候选簇：逐簇确认（confirmed）/ 不确定（uncertain）/ 拒绝（reject）。
多人模式按 `image_id + reviewer` 保存独立原始票，网页只显示当前审核人的判断；
正式导出默认至少 3 人且必须完全一致，人数不足或任意分歧均不进入确认库。

```bash
python scripts/launch_review.py --clusters "outputs/cluster_archival/20140419 02/clusters.csv" \
    --reviewer reviewer_a --port 8001
python scripts/launch_review.py --clusters "outputs/cluster_archival/20140419 02/clusters.csv" \
    --reviewer reviewer_b --port 8002
python scripts/launch_review.py --clusters "outputs/cluster_archival/20140419 02/clusters.csv" \
    --reviewer reviewer_c --port 8003
python scripts/launch_review.py --clusters "outputs/cluster_archival/20140419 02/clusters.csv" \
    --export --min-reviewers 3
```

审核结果在 `outputs/review/`（`review_annotations.csv` 原始票、
`review_vote_summary.csv` 裁决证据、`confirmed_individuals.csv` 一致确认结果）。

### 4.4 个体查询客户端 `scripts/launch_query.py`

上传一张背鳍照片 → YOLO 检测裁剪（未检出回退整图）→ r4 特征 → 全库 Top-K 检索，输出二态候选提示：

- `known`：最高相似度 ≥ 阈值，展示 Top-K 候选（仍需人工核验）；
- `unknown`：最高相似度 < 阈值，提示"疑似未知个体（可能新个体）"，仍返回 Top-K 供参考。

```bash
python scripts/launch_query.py                    # http://127.0.0.1:8000
```

可双击 `start_query_app.bat` 一键启动。默认阈值 0.55 的状态为
`provisional_unvalidated`：旧 E4 跨批次负例假设无效，因此该值不能解释为已校准的
错误接受率。正式使用前必须用当前模型、裁剪方式和独立确认集重新标定。查询端会校验
embedding、meta 与权重的 SHA-256，并要求查询模型和 gallery config 的模型、裁剪、
预处理及 checkpoint 哈希一致；缺少 provenance 或跨版本混用会直接拒绝启动。

### 4.5 特征训练 `scripts/train_reid.py`

用批次内已确认的个体标签训练特征模型（ArcFace 度量学习两阶段：冻结 backbone 训 head → 解冻微调）：

```bash
python scripts/train_reid.py --out outputs/metric_learning/r6_candidate \
    --test-session "20140419 02" --extract
python scripts/train_reid.py --out outputs/metric_learning/r6_ce \
    --test-session "20140419 02" --no-hard-negative
```

`--out` 必须显式指定新版本目录；目录已有 `best.pt/history.csv/metrics.json` 时默认
拒绝覆盖。只有明确要复跑同版本才使用高风险参数 `--overwrite`。候选模型完成独立评估
前不要修改生产 `reid_checkpoint`；当前生产模型仍是 r4。

`r5_candidate` 是一次协议纠正后的历史候选，但其训练缓存缩放和 Triplet 的整串
身份排除仍有缺口，因此不能作为最终纠正版本。`r6_candidate` 补齐这两项：训练缓存
等比缩放至短边 256，且某身份只要曾与 anchor 同串共现，就不能作为该 anchor 的
Triplet 负类；`20140419 02` 整个 session（46 张 / 7 个体）也被独立留出，不参与
训练或 checkpoint 选择。

r6 的小验证集 best R@1 为 0.953（43 个有效 query，stage 2 epoch 19），独立测试
R@1 为 0.500（预训练基线 0.478）；两者只用于训练诊断，不能代替固定全量评估。在
75 个批次内身份的同协议全量跨串评估（97 query / 255 gallery）中，r4/r5/r6 的
R@1/mAP 分别为 **0.567/0.707**、0.412/0.580、0.454/0.636。E5 使用 session-local
候选集并逐 query 排除完整同串：保守口径的簇级 R@1 / MRR@10 分别为 r4 0.581/0.710、
r5 0.488/0.616、r6 0.558/0.687；串抽样口径为 r4 0.628/0.719、r5 0.395/0.553、
r6 0.512/0.639。r6 比 r5 有所回升，但仍未超过 r4，因此生产模型不切换。旧跨
session E5 数字无效，且单真值簇的 reciprocal-rank 指标应称 MRR@10，不是 mAP。
所有评估仍是当前短时间间隔内的批次内跨串评估，不构成跨月或跨年结论。详见
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 的 r5/r6 实跑记录。

### 4.6 评估 `scripts/evaluate.py`

在已提取特征上评估批次内跨串检索指标（R@1 / mAP），或生成批次内同/异体
相似度诊断。两种模式都剔除完整同串；跨批次身份未对齐，不进入正式正负例：

```bash
python scripts/evaluate.py --embeddings outputs/artifacts/r4_yolocrop_v3/gallery/embeddings.npy \
    --meta outputs/artifacts/r4_yolocrop_v3/gallery/embeddings_meta.csv \
    --mode retrieval --out outputs/verification/r4_yolocrop_v3
python scripts/evaluate.py --embeddings outputs/artifacts/r4_yolocrop_v3/gallery/embeddings.npy \
    --meta outputs/artifacts/r4_yolocrop_v3/gallery/embeddings_meta.csv \
    --mode pairs --out outputs/verification/r4_yolocrop_v3
```

`pairs` 只报告诊断分布（`diagnostic_only_not_open_set_calibration`），不能据此给出
开放集 FA 或生产阈值；正式阈值需要跨批次人工对齐后的 confirmed same/different/unknown
独立标定集。

### 4.7 跨时间批次驱动 `scripts/run_cross_time_batch.py`

历史库（20140806 01/03 labeled）→ YOLO 裁剪 + r4 特征 → 逐个新批次跑批内归档管线并匹配历史库（实验 E7 验证的真实流程）：

```bash
python scripts/run_cross_time_batch.py                       # 全流程（7 个新批次，E7 验证）
python scripts/run_cross_time_batch.py --sessions "20140419 02"   # 只跑指定批次
python scripts/run_cross_time_batch.py --only-gallery        # 只读严格校验当前历史库
```

该入口禁止原地重建或覆盖活动 gallery。需要重建时，使用
`scripts/rebuild_r4_artifacts.py --out outputs/artifacts/<新版本目录>` 先发布不可覆盖的新版本，
再显式修改 `configs/pipeline.yaml` 切换活动产物。

### 4.8 其他工具

| 入口 | 用途 |
|---|---|
| `scripts/assign_pool.py` | 同群散图划分：散图对同群已确认个体 Top-K 候选（低分标记疑似新个体） |
| `scripts/train_detector.py` | 训练 YOLO 背鳍检测器（数据由 `scripts/build_yolo_det_dataset.py` + SAM 预标注构建） |
| `scripts/contact_sheets.py` | 候选簇拼图（已被审核网页取代，备用） |

### 4.9 数据管理工具（3.2/3.3/3.6/1.9 支撑）

| 入口 | 用途 |
|---|---|
| `scripts/finalize_history_verify.py` | 历史库核验回填：汇总表 → 可信基准 `history_verified_individuals.csv` + pilot_set 置 verified（改前备份；结论=通过且组内无不确定/排除才登记，组名不在 pilot_set 拒绝执行）。见 [docs/history_verify_crossyear.md](docs/history_verify_crossyear.md) §一 1.5 |
| `scripts/group_sequences.py` | 散图连拍串分组：按文件名连拍号分串（322 张散图 → 78 串）+ A9 抽样核验清单；核验通过前不用于划归 |
| `scripts/build_eval_set.py` | 人工评估集划分草案：确认个体 → 按 Sequence 划分 query/gallery（同序列不拆分防泄漏）；草案须人工确认 |
| `scripts/export_relations.py` | 确认关系表导出：confirmed_same 对 + confirmed_different / possibly_same 空表结构（数据源待 3.8） |

## 5. 配置

`configs/` 三个 YAML，所有路径相对仓库根；数据盘路径只在 `pipeline.yaml` 出现一次：

| 文件 | 内容 |
|---|---|
| `pipeline.yaml` | 数据根、检测器/ReID 权重、裁剪参数（pad_x/pad_up/pad_down）、聚类与检索阈值、review/query 端口 |
| `reid.yaml` | 训练超参（epochs/lr/batch）、hard_negative 开关、HN 挖掘参数 |
| `detector.yaml` | YOLO 训练超参（epochs/imgsz/device） |

## 6. 目录结构

```text
WhiteWhale_recognization/
├── README.md                  # 本文档：项目总纲（读一次即可上手）
├── scripts/                   # 正式入口（薄 wrapper，实现全在 src/whitewhale/）
│   ├── prepare_data.py        #   1. 数据盘点：scan 生成 manifest / build-pilot
│   ├── run_pipeline.py        #   2. 批内归档管线（检测→聚类→子簇→匹配→审核材料）
│   ├── launch_review.py       #   3. 人工审核网页（:8001）
│   ├── launch_query.py        #   4. 个体查询客户端（:8000）
│   ├── train_reid.py          #   5. 特征训练（r6 候选已实跑，正式特征源仍为 r4）
│   ├── evaluate.py            #   6. 检索指标 / 对级分布评估
│   ├── run_cross_time_batch.py#   7. 跨时间批次驱动
│   ├── finalize_history_verify.py#  8. 历史库核验回填（3.6 步骤4 自动化）
│   ├── group_sequences.py       #    9. 散图连拍串分组（1.9 准备）
│   ├── build_eval_set.py        #   10. 评估集划分草案（3.2）
│   ├── export_relations.py      #   11. 确认关系表导出（3.3）
│   ├── assign_pool.py         #   同群散图划分（辅助工具）
│   ├── train_detector.py      #   YOLO 背鳍检测器训练（辅助工具）
│   ├── build_yolo_det_dataset.py、annotate_sam.py   # 检测数据构建（辅助工具）
│   └── contact_sheets.py      #   候选簇拼图（备用）
├── src/whitewhale/            # 正式模块（唯一实现，一个功能一个入口）
│   ├── pipeline/              #   archival（批内归档）/ cross_time（跨时间）/ assign_pool
│   ├── detection/             #   YOLO 检测 + 非均匀扩展裁剪（detector.py）
│   ├── reid/                  #   embedding（模型族+统一提取）/ training / evaluation / retrieval
│   ├── review/                #   审核网页 app + 拼图 contact_sheets
│   ├── data/                  #   manifest（扫描+清单）/ dataset（审核数据集）/
│   │                          #   数据工具（评估集划分 eval_set / 分串 sequence_groups /
│   │                          #   关系表 relations / 核验回填 history_verify）
│   ├── query.py               #   查询客户端应用
│   └── config.py              #   yaml 统一加载（load_config）
├── configs/                   # 统一配置：pipeline.yaml / reid.yaml / detector.yaml
├── docs/                      # 操作手册（历史库核验与跨年匹配重建）
├── experiments/               # 一次性科研实验（benchmark/评估预演；可追溯，不参与正式流程）
├── tests/                     # pytest 回归测试（见 §7）
├── outputs/                   # 生成物（不入库，可重新生成）
│   ├── embeddings/            #   特征库（*.npy + meta.csv + config.json）
│   ├── cluster_archival/      #   归档管线产物（每批次一个子目录：clusters/ 代表图/ 拼图）
│   ├── review/                #   人工审核标注与确认个体表
│   ├── metric_learning/       #   训练产物（best.pt / metrics.json）
│   └── index/ pilot/          #   manifest、pilot 清单
├── models/                    # 模型权重（不入库）：detectors/（YOLO）
├── start_query_app.bat        # 双击一键启动查询客户端
└── src_dataset/                # 原始数据（只读，不入库，configs/pipeline.yaml 中配置）
```

## 7. 测试

```bash
python -m pytest tests/ -v
```

覆盖：检索指标（R@k/mAP）、query/gallery 划分防泄漏、拼图输出、审核数据可追溯、查询二态提示与模型匹配防护、yaml 配置加载、数据工具（3.6 回填/1.9 分串/3.2 划分/3.3 关系表）。

## 8. 待办（当前）

> 只列待办；已完成任务的记录在 `EXPERIMENT_LOG.md` 与本文档。状态：`[ ]` 待办 / `[~]` 进行中。

**数据与语义**

| # | 任务 | 状态 |
|---|---|---|
| 0.8 | 确认数据授权（是否允许训练、发布模型或公开派生数据） | [ ] 需数据提供方 |
| 1.9 | 散图连拍整串划归：散图池按拍摄序列分组，任一帧匹配上某个体则整串划归候选（A9 抽样核验后执行；一次确认一串，大幅提升收集效率）——**分组脚本已就绪 `scripts/group_sequences.py`**（真实数据 322 张散图 → 78 串，含 A9 抽样核验清单 `outputs/index/sequence_sample_checklist.csv`），A9 核验通过即可使用 | [ ] |
| 1.8 | 散图分拣（人工待办）：322 张散图（loose_known，9 批次）人工分配到对应个体文件夹 | [ ] 组员 |

**人工基准集**

| # | 任务 | 状态 |
|---|---|---|
| 3.1 | 人工核验候选簇：初审已完成（首批 135 张/31 个体 + 跨时间 7 批次 270 张全过审：65 组确认 / 68 不确定 / 7 排除）；正式核验待做（多人复核投票，见 3.7） | [~] |
| 3.2 | 建立人工评估集（多个体 × 多日期 × 多角度，按 Sequence 划分）——**自动划分脚本已就绪 `scripts/build_eval_set.py`**（真实数据草案：123 张 / 28 个体，query 28 / gallery 95，同序列不拆分），草案待人工确认 | [ ] |
| 3.3 | 建立确认关系表（confirmed_same / confirmed_different / possibly_same）——**表结构与导出代码已就绪 `scripts/export_relations.py`**（confirmed_same 有数据，另两张空表结构 + 数据源说明） | [ ] |
| 3.4 | 确定可接受错误合并率（优先控制错误合并风险，种群统计低估） | [ ] 科研决策 |
| 3.5 | Pilot 独立核验集（~20-30 个体，每个体选 2-4 张不同序列/时段照片） | [ ] |
| 3.6 | 历史库二次一致性核验：20140806 的 43 个批次内已确认个体、202 张照片，复核组内异常及形成独立可信基准（包已就绪：batches_history/history_verify.csv，202/202 带特征辅助）——**操作手册见 [docs/history_verify_crossyear.md](docs/history_verify_crossyear.md) §一**；**核验回填脚本已就绪 `scripts/finalize_history_verify.py`**（汇总表 → 可信基准 + pilot_set 置 verified，含自洽校验与组名防错） | [ ] 组员执行 |
| 3.7 | 6 张撤回照片多人复核投票重审（E9 撤回；关键判定须至少 3 人独立标注、完全一致才确认）——**6.11 工具已完成**，操作手册见 [docs/history_verify_crossyear.md](docs/history_verify_crossyear.md) §二 | [ ] 组员执行 |
| 3.8 | 跨年匹配重建（依赖 3.6 + 3.7 通过后）——**操作手册见 [docs/history_verify_crossyear.md](docs/history_verify_crossyear.md) §三**（前置条件/命令/审核/成功标准已写明） | [ ] |

**自监督与轮廓特征**

| # | 任务 | 状态 |
|---|---|---|
| 4.1 | 自监督微调（DINOv2 / SimCLR / MoCo / BYOL / MAE 之一） | [ ] |
| 4.2 | 增强策略设计——**镜像鲁棒性实测完成（E11）**：水平翻转对 r3 特征影响小（同体相似度 −0.030，翻转 query 检索 R@1 −0.038），**翻转增强基本安全**，可纳入训练（小概率）；正式训练增强配置待定 | [~] |
| 4.3 | 背鳍轮廓特征（CurvRank 风格）——**原型实验完成（E10）**：A14 几何对称性成立（同一轮廓镜像后特征不变，对称化 1.000）；但 Otsu 分割不稳定使特征区分度不足（同体≈跨体，R@1 0.087 vs 外观 0.728）；**自由边比例扫描（0.45/0.55/0.65）确认瓶颈在分割本身而非轮廓范围**（R@1 不升反降），改善分割需更强方法（SAM 等，需标注辅助） | [~] |
| 4.4 | 特征对比——轮廓 vs 外观已测（E10：轮廓 R@1 0.087 vs 外观 0.728）；融合待做（依赖分割改善） | [ ] |
| 4.5 | 特征可视化与错误分析——E10 完成基础版（外观 R@1 失败 50/184，分散 19 个个体，无集中病灶）；可视化报告待完善 | [~] |

**确认标签与度量学习**

| # | 任务 | 状态 |
|---|---|---|
| 5.1 | 将模型候选簇经人工审核转为补充训练标签（只纳入 confirmed，独立版本化；不得把候选结果直接当标签） | [~] |
| 5.2 | 度量学习训练（ArcFace + 修正 batch-hard）——r6 补齐 r5 的等比缓存与整串共现身份排除，并以 `20140419 02` 为独立测试；小验证集 0.953、独立测试 0.500 不能替代固定全量结论。全量同协议图级 R@1/mAP：r4 0.567/0.707，r5 0.412/0.580，r6 0.454/0.636。E5 session-local 的 r6 两种簇级 R@1/MRR@10 为 0.558/0.687、0.512/0.639，仍低于 r4；生产继续使用 r4，短间隔数据不能外推跨年 | [~] |
| 5.4 | 难例主动审核（低置信边界样本） | [ ] |

**自动化系统**

| # | 任务 | 状态 |
|---|---|---|
| 6.2 | 自动判断左右侧与质量：质量判定规则已用（E9：未检出背鳍 / det_conf<0.3 / Laplacian 方差<p10）；模型化与左右侧判断未做 | [~] |
| 6.10 | 多头同框检测（NN relationship 候选 + 多归属归档）：YOLO 取全部框 + IoU 去重；**语义红线：同框多头 ≠ 亲缘关系**，只能标记疑似供人工判断 | [ ] |

**文档与工程基建**

| # | 任务 | 状态 |
|---|---|---|
| D.2 | 参考仓库 SOURCE_MAP（[references/SOURCE_MAP.md](references/SOURCE_MAP.md)）：CetaMatch(MIT)、MiewID(无LICENSE)、DINOv2(Apache-2.0)、WildlifeDatasets/WildlifeTools、Happywhale-1st、Faiss、PyTorch Metric Learning；公开数据集：Happywhale(Kaggle)、NDD20、NOAA Choctawhatchee、BelugaID——**beluga + happywhale 已接入 `experiments/pub_reid/`，SOURCE_MAP 已同步（2026-08-25）** | [~] |
| D.3 | 数据伦理与合规（不公开敏感地点/坐标/未经授权影像）——**已检查（2026-08-25）**：代码无密钥/token/.env；git 历史无敏感文件；文档无真实坐标（仅数据内部地点代码 SZi/HBi，已声明）；生成物与原始数据均不入库 | [~] |

## 9. 科研边界

- 一次性实验脚本在 `experiments/`，正式功能一律走 `src/whitewhale/` + 上述入口；
- 实验结果只追加记录于 `EXPERIMENT_LOG.md`（配置、结果、结论），删除脚本不删除实验记录；
- 所有输出保留 `image_id` / `relative_path` 可追溯字段；
- 聚类与检索结果 = Candidate，人工确认后才能叫个体；`uncertain` / `reject` 是合法审核结论，不强行归档。
