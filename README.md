# WhiteWhale_recognization — 中华白海豚个体识别

基于野外调查影像，研究可靠的中华白海豚**个体识别方法**（背鳍检测 → 特征学习 → 批内分组聚类 → 跨时间匹配 → 新个体发现），并将方法**落地为辅助归档工具**：输入一批新照片自动按个体分组、每簇取最清晰的一张代表图归档，减少工作人员逐张手工整理的工作量，同时为"这只海豚见过没"提供候选匹配。模型结果一律是 **Candidate（候选）**，正式个体身份须经人工核验（human-in-the-loop）。

## 1. 核心语义

| 概念 | 说明 |
|---|---|
| 调查批次 | 一次野外调查、日期或航次 |
| 拍摄序列 | 短时间内连续拍摄的一组照片（连拍）；训练/评估按序列划分，不拆分 |
| Anchor（代表照片） | 高分目录数字子文件夹 = 历史挑选的个体代表照片，作检索查询；未经确认前不代表身份（非全局个体 ID） |
| Unresolved Image Pool | `70-79` 散图 = 未归属照片池，作检索 Gallery |
| 候选分组 | 原数据中人工或程序初步整理出的图片集合（Candidate Cluster） |
| Candidate ≠ Confirmed | 聚类簇、检索结果都只是候选；人工审核（确认/不确定/拒绝）后才是个体身份 |
| 个体 ID | 经人工核验后确认的白海豚身份（Confirmed Individual） |
| 图像质量 | 图像清晰度、目标大小或人工评分（`70-79` / `80 and above` 等评分区间） |
| 关系备注 | 文件夹名称中记录的候选关联信息 |
| 批内（同一天） | 封闭集归档：检测 → 聚类 → 每簇选代表图 → 人工审核归档 |
| 跨时间（不同批次） | 开放集匹配：新批次与历史个体库匹配，低于阈值标记"疑似新个体"；结果仅作候选 |
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
│   ├── 05/ 11/ 12/ 13/ 14/ …   # 数字子文件夹 = 代表照片 Anchor（组名非全局 ID）
│   └── *.JPG                   # 散图，Unresolved Image Pool（人工跟进中）
├── 80 and above/               # 80 分及以上
│   └── 01/ … 10/               # 数字子文件夹 = 代表照片 Anchor
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
* **01 = W01（SZi），03 = W03（HBi）**，同一天两群；01 与 03 是两个独立海豚群，分组编号在群内各自独立（01_02 与 03_02 不是同一只），跨群照片默认视为不同个体，除非未来人工确认；
* **`nn relationship` 目录 = 疑似亲缘关系个体样本**（2026-08-19 数据提供方确认，**非每个批次必有**）：一图两鳍（同框多头），不参与个体分组，scan_dataset 标记 relation_note=nn_relationship；亲缘关系仍需人工确认（同框 ≠ 有亲缘）。

### 2.4 待确认假设（A1–A13）

| # | 假设 | 状态 |
|---|---|---|
| A1 | 评分区间为质量/匹配置信度分级 | 基本证实（W01 计数一致） |
| A2 | MO/RAY/DEREK 为拍摄者代码 | 已证实 |
| A3 | SZi/HBi 为调查地点代码 | 推测 |
| A4 | 80 and above 数字分组为代表照片（Anchor）而非个体 ID | 已确认（历史整理方式） |
| A5 | `sj of 01` = 伴随/相关个体关系 | 推测 |
| A6 | `nn relationship` = 相邻关系候选组 | 推测（txt 字段为空） |
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

* 当前使用全部 9 个批次的 `70-79`、`80 and above` 两个区间（≥70 分）数据；历史库（gallery）= 20140806 01/03 的 labeled（43 组 202 张，Candidate 级）；
* 这两个区间下的数字子文件夹 = 历史挑选的代表照片（Anchor），不代表全局个体 ID，可作 Anchor 检索的查询集，不可直接作监督标签；
* 每个 Anchor 基本只有一张照片（极少数含连拍多帧），属于极端少样本的个体识别问题；
* 单图 Anchor 可视为该个体的**质量上界先验**（高分分段挑选出的代表照片，模糊图进不了该分段），散图收集时可作质量参照（先验非事实，使用须标注）；
* 散图（322 张 loose_known）属于 Unresolved Image Pool：同调查内可能属于某个已选代表照片的个体，但具体归属未确认，只能作为检索 Gallery 的候选，不可作标签；
* 分组编号只在同一调查内有效，跨调查同名编号是否对应同一只海豚需人工核验（当前跨调查对应关系尚未核验）；
* 背鳍有左右两面，但当前每个个体只有一张照片、尚无双侧样本；照片仍须记录朝向（left / right / unknown），左右侧分别比较。

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
* **分层信任目录标签**：数字子文件夹（Anchor）未经人工核验前不代表个体身份，不可作监督分类标签；分组编号只在同调查内成立；`70-79` 散图仅作检索 Gallery；低于 70 分的区间及 `MO`/`RAY`/`DEREK`/`miscellaneous`/`nn relationship` 等目录暂不用于个体标签；
* **避免重复数据**：数据索引阶段进行文件哈希或图像相似度检查，避免重复统计和重复训练；
* **可追溯**：每张处理后的图片都能追溯到原始文件路径、调查日期、拍摄批次、连续拍摄序列、原始候选分组、图像质量区间、处理方式、人工核验状态。

## 3. 环境

- Python 3.13+，依赖见 `requirements.txt`（torch、timm、fastapi、pandas、numpy、hdbscan、ultralytics、PyYAML）——**新环境先执行 `pip install -r requirements.txt`（网络慢可加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）；运行报 `ModuleNotFoundError` 时请先检查是否已按此安装依赖，再排查其他问题**
- 模型权重与原始数据不入库（`models/`、`*.pt`、`outputs/*` 已被 git 忽略，**请勿直接 commit 权重文件**）：检测器 `models/detectors/yolov8n_dorsalfin.pt`，r3 特征权重 `outputs/metric_learning/r3/best.pt`，原图 `src_dataset/`
- **权重获取方式**：自训练权重（检测器 + r3/r4 + 训练记录）打包为 `whitewhale_weights_2026-08-26.zip`（约 200MB，含包内说明 `README.md`）——从网盘分享或项目 Releases 附件下载，**解压到仓库根目录即可**（路径与 `configs/pipeline.yaml` 自动对齐，无需改配置）；MegaDescriptor-T-224 特征模型首次运行自动下载，无需手动获取；ultralytics 官方预训练等仅重训检测器时才需要
- 所有入口默认值从 `configs/*.yaml` 读取，可用 CLI 参数覆盖

## 4. 正式入口

所有命令在仓库根目录执行。七个正式入口（4.1–4.7），**一个功能一个入口**：

### 4.1 数据盘点 `scripts/prepare_data.py`

扫描原始目录生成数据清单（manifest）：

```bash
python scripts/prepare_data.py scan                # 扫描 configs/pipeline.yaml 的 data_roots
python scripts/prepare_data.py scan --sha256      # 附加 SHA-256 完全重复检测
python scripts/prepare_data.py build-pilot        # 由 manifest 生成 pilot_set.csv（高分 Anchor 照片表）
```

### 4.2 批内归档管线 `scripts/run_pipeline.py`

新批次全流程：YOLO 背鳍检测裁剪 → r3 特征 → HDBSCAN 批内候选聚类 → 子簇化 → 簇级多帧投票匹配历史库 → 代表图 + 候选簇拼图（人工审核材料）。

```bash
# 散图池验证模式（复用已预提取的池特征产物）
python scripts/run_pipeline.py --pool

# 任意新批次（输入 manifest）
python scripts/run_pipeline.py --input-manifest outputs/index/dataset_manifest.csv \
    --batch-name "20240419 02" --sheets
```

| 关键参数 | 默认 | 说明 |
|---|---|---|
| `--pool` / `--input-manifest` | 二选一必填 | 散图池验证 / 新批次清单 |
| `--gallery-embeddings` | `outputs/embeddings/embeddings_metric_r3_yolocrop.npy` | 历史个体库特征（r3 + YOLO 裁剪） |
| `--threshold-cluster` | 0.58 | 簇级匹配阈值（E5 标定 FA≤5%，语义"宁可多标疑似"） |
| `--threshold-image` | 0.50 | 单图投票阈值下限（E4 标定） |
| `--out` | `outputs/cluster_archival` | 输出目录（内部按 batch_name 分目录） |
| `--sheets` | 关 | 生成候选簇拼图 |

### 4.3 人工审核网页 `scripts/launch_review.py`

审核批内归档候选簇：逐簇确认（confirmed）/ 不确定（uncertain）/ 拒绝（reject），记录审核人与时间，可追溯。

```bash
python scripts/launch_review.py                          # 打开审核页（http://127.0.0.1:8001）
python scripts/launch_review.py --export                 # 导出审核结果（confirmed 个体表）后退出
```

审核结果在 `outputs/review/`（annotations.csv + confirmed_individuals.csv）。

### 4.4 个体查询客户端 `scripts/launch_query.py`

上传一张背鳍照片 → YOLO 检测裁剪（未检出回退整图）→ r3 特征 → 全库 Top-K 检索，输出三态判定：

- `known`：最高相似度 ≥ 阈值，展示 Top-K 候选（仍需人工核验）；
- `unknown`：最高相似度 < 阈值，提示"疑似未知个体（可能新个体）"，仍返回 Top-K 供参考。

```bash
python scripts/launch_query.py                    # http://127.0.0.1:8000
```

可双击 `start_query_app.bat` 一键启动。默认阈值 0.55（E4 标定 FA≤5% 区间中值）。查询模型必须与 gallery 特征同模型（读取 embedding 旁 config 自动匹配，显式指定时校验拒绝跨模型）。

### 4.5 特征训练 `scripts/train_reid.py`

用人工确认的伪标签训练特征模型（ArcFace 度量学习两阶段：冻结 backbone 训 head → 解冻微调）：

```bash
python scripts/train_reid.py                          # r3：跨群 hard negative 微调（默认，正式特征源）
python scripts/train_reid.py --no-hard-negative       # r1/r2 历史链路（纯 ArcFace CE）
python scripts/train_reid.py --extract                # 训练后重新提取特征
```

### 4.6 评估 `scripts/evaluate.py`

在已提取特征上评估检索指标（个体级 R@1 / mAP）或对级相似度分布（FA5% 阈值建议）：

```bash
python scripts/evaluate.py --embeddings outputs/embeddings/embeddings_metric_r3_yolocrop.npy \
    --meta outputs/embeddings/embeddings_metric_r3_yolocrop_meta.csv --mode retrieval
python scripts/evaluate.py --mode pairs   # 同/跨个体余弦分布 + 阈值建议
```

### 4.7 跨时间批次驱动 `scripts/run_cross_time_batch.py`

历史库（20140806 01/03 labeled）→ YOLO 裁剪 + r3 特征 → 逐个新批次跑批内归档管线并匹配历史库（实验 E7 验证的真实流程）：

```bash
python scripts/run_cross_time_batch.py                       # 全流程（7 个新批次，E7 验证）
python scripts/run_cross_time_batch.py --sessions "20140419 02"   # 只跑指定批次
python scripts/run_cross_time_batch.py --only-gallery        # 只构建历史库特征
```

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
│   ├── train_reid.py          #   5. 特征训练（r3 正式链路）
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
├── tests/                     # pytest 接口测试（43 个，见 §7）
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

覆盖：检索指标（R@k/mAP）、query/gallery 划分防泄漏、拼图输出、审核数据可追溯、查询三态判定与模型匹配防护、yaml 配置加载、数据工具（3.6 回填/1.9 分串/3.2 划分/3.3 关系表）。

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
| 3.5 | Pilot 人工确认集（~20-30 个 Anchor 的同个体照片 2-4 张） | [ ] |
| 3.6 | 历史库核验：20140806 43 组 202 张 Candidate 级标签人工核验（包已就绪：batches_history/history_verify.csv，202/202 带特征辅助）——**操作手册见 [docs/history_verify_crossyear.md](docs/history_verify_crossyear.md) §一**；**核验回填脚本已就绪 `scripts/finalize_history_verify.py`**（汇总表 → 可信基准 + pilot_set 置 verified，含自洽校验与组名防错） | [ ] 组员执行 |
| 3.7 | 6 张撤回照片多人复核投票重审（E9 撤回；关键判定须多人独立标注 + 投票裁决）——**操作手册见 [docs/history_verify_crossyear.md](docs/history_verify_crossyear.md) §二**（6 张照片清单与流程已写明） | [ ] 组员执行（依赖 6.11 工具） |
| 3.8 | 跨年匹配重建（依赖 3.6 + 3.7 通过后）——**操作手册见 [docs/history_verify_crossyear.md](docs/history_verify_crossyear.md) §三**（前置条件/命令/审核/成功标准已写明） | [ ] |

**自监督与轮廓特征**

| # | 任务 | 状态 |
|---|---|---|
| 4.1 | 自监督微调（DINOv2 / SimCLR / MoCo / BYOL / MAE 之一） | [ ] |
| 4.2 | 增强策略设计——**镜像鲁棒性实测完成（E11）**：水平翻转对 r3 特征影响小（同体相似度 −0.030，翻转 query 检索 R@1 −0.038），**翻转增强基本安全**，可纳入训练（小概率）；正式训练增强配置待定 | [~] |
| 4.3 | 背鳍轮廓特征（CurvRank 风格）——**原型实验完成（E10）**：A14 几何对称性成立（同一轮廓镜像后特征不变，对称化 1.000）；但 Otsu 分割不稳定使特征区分度不足（同体≈跨体，R@1 0.087 vs 外观 0.728）；**自由边比例扫描（0.45/0.55/0.65）确认瓶颈在分割本身而非轮廓范围**（R@1 不升反降），改善分割需更强方法（SAM 等，需标注辅助） | [~] |
| 4.4 | 特征对比——轮廓 vs 外观已测（E10：轮廓 R@1 0.087 vs 外观 0.728）；融合待做（依赖分割改善） | [ ] |
| 4.5 | 特征可视化与错误分析——E10 完成基础版（外观 R@1 失败 50/184，分散 19 个个体，无集中病灶）；可视化报告待完善 | [~] |

**伪标签与度量学习**

| # | 任务 | 状态 |
|---|---|---|
| 5.1 | 生成伪标签（仅高可信确认簇，独立版本化；当前为 Candidate 级初审标签） | [~] |
| 5.2 | 度量学习训练（ArcFace / Triplet / 对比学习）——**r4 重训完成（E5.2，2026-08-25）**：individual_id 作标签（候选级），训练/评估个体彻底隔离，保守口径新批次 R@1 0.238→0.381（+14.3pp）；**生产 `reid_checkpoint` 仍为 r3，切换 r4 待确认**；Triplet 其余变体未试 | [~] |
| 5.4 | 难例主动审核（低置信边界样本） | [ ] |

**自动化系统**

| # | 任务 | 状态 |
|---|---|---|
| 6.2 | 自动判断左右侧与质量：质量判定规则已用（E9：未检出背鳍 / det_conf<0.3 / Laplacian 方差<p10）；模型化与左右侧判断未做 | [~] |
| 6.10 | 多头同框检测（NN relationship 候选 + 多归属归档）：YOLO 取全部框 + IoU 去重；**语义红线：同框多头 ≠ 亲缘关系**，只能标记疑似供人工判断 | [ ] |
| 6.11 | 审核网页支持多人复核投票：多审核人独立标注 + 汇总投票裁决（配合 3.7） | [ ] 组员开发 |

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
