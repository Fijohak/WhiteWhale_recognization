# WhiteWhale_recognization — 中华白海豚个体识别

基于野外调查影像，研究可靠的中华白海豚**个体识别方法**（背鳍检测 → 特征学习 → 批内分组聚类 → 跨时间匹配 → 新个体发现），并将方法**落地为辅助归档工具**：输入一批新照片自动按个体分组、每簇取最清晰的一张代表图归档，减少工作人员逐张手工整理的工作量，同时为"这只海豚见过没"提供候选匹配。模型结果一律是 **Candidate（候选）**，正式个体身份须经人工核验（human-in-the-loop）。

## 1. 核心语义

| 概念 | 说明 |
|---|---|
| Anchor / Pool | 高分目录数字子文件夹 = 历史挑选的代表照片（Anchor，非全局个体 ID）；`70-79` 散图 = Unresolved Image Pool（未归属照片池） |
| Candidate ≠ Confirmed | 聚类簇、检索结果都只是候选；人工审核（确认/不确定/拒绝）后才是个体身份 |
| 批内（同一天） | 封闭集归档：检测 → 聚类 → 每簇选代表图 → 人工审核归档 |
| 跨时间（不同批次） | 开放集匹配：新批次与历史个体库匹配，低于阈值标记"疑似新个体"；结果仅作候选 |
| 左右侧分离 | 背鳍两侧特征不同，照片记录朝向（left/right/unknown），左右侧分别比较 |
| 连拍不泄漏 | 连续拍摄序列不可拆分，训练/评估按序列划分 |
| 原始数据只读 | 原图在 `I:/` 只读；所有生成物写 `outputs/`，可追溯到原始路径 |

数据集结构、已确认事实与假设详见 [docs/DATASET.md](docs/DATASET.md)。历史实验结果与结论见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 2. 环境

- Python 3.13+，依赖见 `requirements.txt`（torch、timm、fastapi、pandas、numpy、hdbscan、ultralytics、PyYAML）
- 模型权重与原始数据不入库：检测器 `models/detectors/yolov8n_dorsalfin.pt`，r3 特征权重 `outputs/metric_learning/r3/best.pt`，原图 `I:/`
- 所有入口默认值从 `configs/*.yaml` 读取，可用 CLI 参数覆盖

## 3. 正式入口

所有命令在仓库根目录执行。七个正式入口（3.1–3.7），**一个功能一个入口**：

### 3.1 数据盘点 `scripts/prepare_data.py`

扫描原始目录生成数据清单（manifest）：

```bash
python scripts/prepare_data.py scan                # 扫描 configs/pipeline.yaml 的 data_roots
python scripts/prepare_data.py scan --sha256      # 附加 SHA-256 完全重复检测
python scripts/prepare_data.py build-pilot        # 由 manifest 生成 pilot_set.csv（高分 Anchor 照片表）
```

### 3.2 批内归档管线 `scripts/run_pipeline.py`

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

### 3.3 人工审核网页 `scripts/launch_review.py`

审核批内归档候选簇：逐簇确认（confirmed）/ 不确定（uncertain）/ 拒绝（reject），记录审核人与时间，可追溯。

```bash
python scripts/launch_review.py                          # 打开审核页（http://127.0.0.1:8001）
python scripts/launch_review.py --export                 # 导出审核结果（confirmed 个体表）后退出
```

审核结果在 `outputs/review/`（annotations.csv + confirmed_individuals.csv）。

### 3.4 个体查询客户端 `scripts/launch_query.py`

上传一张背鳍照片 → YOLO 检测裁剪（未检出回退整图）→ r3 特征 → 全库 Top-K 检索，输出三态判定：

- `known`：最高相似度 ≥ 阈值，展示 Top-K 候选（仍需人工核验）；
- `unknown`：最高相似度 < 阈值，提示"疑似未知个体（可能新个体）"，仍返回 Top-K 供参考。

```bash
python scripts/launch_query.py                    # http://127.0.0.1:8000
```

可双击 `start_query_app.bat` 一键启动。默认阈值 0.55（E4 标定 FA≤5% 区间中值）。查询模型必须与 gallery 特征同模型（读取 embedding 旁 config 自动匹配，显式指定时校验拒绝跨模型）。

### 3.5 特征训练 `scripts/train_reid.py`

用人工确认的伪标签训练特征模型（ArcFace 度量学习两阶段：冻结 backbone 训 head → 解冻微调）：

```bash
python scripts/train_reid.py                          # r3：跨群 hard negative 微调（默认，正式特征源）
python scripts/train_reid.py --no-hard-negative       # r1/r2 历史链路（纯 ArcFace CE）
python scripts/train_reid.py --extract                # 训练后重新提取特征
```

### 3.6 评估 `scripts/evaluate.py`

在已提取特征上评估检索指标（个体级 R@1 / mAP）或对级相似度分布（FA5% 阈值建议）：

```bash
python scripts/evaluate.py --embeddings outputs/embeddings/embeddings_metric_r3_yolocrop.npy \
    --meta outputs/embeddings/embeddings_metric_r3_yolocrop_meta.csv --mode retrieval
python scripts/evaluate.py --mode pairs   # 同/跨个体余弦分布 + 阈值建议
```

### 3.7 跨时间批次驱动 `scripts/run_cross_time_batch.py`

历史库（20140806 01/03 labeled）→ YOLO 裁剪 + r3 特征 → 逐个新批次跑批内归档管线并匹配历史库（实验 E7 验证的真实流程）：

```bash
python scripts/run_cross_time_batch.py                       # 全流程（8 个新批次）
python scripts/run_cross_time_batch.py --sessions "20140419 02"   # 只跑指定批次
python scripts/run_cross_time_batch.py --only-gallery        # 只构建历史库特征
```

### 3.8 其他工具

| 入口 | 用途 |
|---|---|
| `scripts/assign_pool.py` | 同群散图划分：散图对同群已确认个体 Top-K 候选（低分标记疑似新个体） |
| `scripts/train_detector.py` | 训练 YOLO 背鳍检测器（数据由 `scripts/build_yolo_det_dataset.py` + SAM 预标注构建） |
| `scripts/contact_sheets.py` | 候选簇拼图（已被审核网页取代，备用） |

## 4. 配置

`configs/` 三个 YAML，所有路径相对仓库根；数据盘路径只在 `pipeline.yaml` 出现一次：

| 文件 | 内容 |
|---|---|
| `pipeline.yaml` | 数据根、检测器/ReID 权重、裁剪参数（pad_x/pad_up/pad_down）、聚类与检索阈值、review/query 端口 |
| `reid.yaml` | 训练超参（epochs/lr/batch）、hard_negative 开关、HN 挖掘参数 |
| `detector.yaml` | YOLO 训练超参（epochs/imgsz/device） |

## 5. 目录结构

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
│   ├── assign_pool.py         #   同群散图划分（辅助工具）
│   ├── train_detector.py      #   YOLO 背鳍检测器训练（辅助工具）
│   ├── build_yolo_det_dataset.py、annotate_sam.py   # 检测数据构建（辅助工具）
│   └── contact_sheets.py      #   候选簇拼图（备用）
├── src/whitewhale/            # 正式模块（唯一实现，一个功能一个入口）
│   ├── pipeline/              #   archival（批内归档）/ cross_time（跨时间）/ assign_pool
│   ├── detection/             #   YOLO 检测 + 非均匀扩展裁剪（detector.py）
│   ├── reid/                  #   embedding（模型族+统一提取）/ training / evaluation / retrieval
│   ├── review/                #   审核网页 app + 拼图 contact_sheets
│   ├── data/                  #   manifest（扫描+清单）/ dataset（审核数据集）
│   ├── query.py               #   查询客户端应用
│   └── config.py              #   yaml 统一加载（load_config）
├── configs/                   # 统一配置：pipeline.yaml / reid.yaml / detector.yaml
├── experiments/               # 一次性科研实验（benchmark/评估预演；可追溯，不参与正式流程）
├── tests/                     # pytest 接口测试（31 个，见 §7）
├── docs/                      # 深挖资料（见下方"文档导航"）
├── outputs/                   # 生成物（不入库，可重新生成）
│   ├── embeddings/            #   特征库（*.npy + meta.csv + config.json）
│   ├── cluster_archival/      #   归档管线产物（每批次一个子目录：clusters/ 代表图/ 拼图）
│   ├── review/                #   人工审核标注与确认个体表
│   ├── metric_learning/       #   训练产物（best.pt / metrics.json）
│   └── index/ pilot/          #   manifest、pilot 清单
├── models/                    # 模型权重（不入库）：detectors/（YOLO）
├── start_query_app.bat        # 双击一键启动查询客户端
└── I:/                        # 原始数据（只读，不入库，configs/pipeline.yaml 中配置）
```

### 文档导航（读一次 README 之后，按需深挖）

| 文档 | 内容 | 何时看 |
|---|---|---|
| `docs/DATASET.md` | 数据结构、规模、已确认事实、假设 A1–A13、数据使用规则 | 需要理解数据语义时 |
| `EXPERIMENT_LOG.md` | E1–E9 实验记录（配置/结果/结论），只追加不修改 | 查历史实验结论时 |
| `docs/anchor_pool_semantics.md` | Anchor/Pool 语义专项说明（方向调整背景） | 回顾项目方向时 |
| `docs/cetacean_reid_transfer_learning_plan.md` | 跨物种迁移学习科研计划 | 科研设计时 |
| `scripts/README.md` | scripts/ 快速导航（本文档 §3 的精简版） | 找脚本时 |

## 6. 测试

```bash
python -m pytest tests/ -v
```

覆盖：检索指标（R@k/mAP）、query/gallery 划分防泄漏、拼图输出、审核数据可追溯、查询三态判定与模型匹配防护、yaml 配置加载。

## 7. 待办（当前）

> 只列待办；已完成任务的记录在 `EXPERIMENT_LOG.md` 与本文档。状态：`[ ]` 待办 / `[~]` 进行中。

**数据与语义**

| # | 任务 | 状态 |
|---|---|---|
| 0.8 | 确认数据授权（是否允许训练、发布模型或公开派生数据） | [ ] |
| 1.8 | 散图分拣（人工待办）：70-79 的 207 张散图人工分配到对应个体文件夹 | [ ] |

**人工基准集**

| # | 任务 | 状态 |
|---|---|---|
| 3.1 | 人工核验首批候选簇（确认/拆分/合并/标记未知；首批初审 135 张/31 个体已完成，正式核验待做） | [~] |
| 3.2 | 建立人工评估集（多个体 × 多日期 × 多角度，按 Sequence 划分） | [ ] |
| 3.3 | 建立确认关系表（confirmed_same / confirmed_different / possibly_same） | [ ] |
| 3.4 | 确定可接受错误合并率（优先控制错误合并风险，种群统计低估） | [ ] |
| 3.5 | Pilot 人工确认集（~20-30 个 Anchor 的同个体照片 2-4 张） | [ ] |

**自监督与轮廓特征**

| # | 任务 | 状态 |
|---|---|---|
| 4.1 | 自监督微调（DINOv2 / SimCLR / MoCo / BYOL / MAE 之一） | [ ] |
| 4.2 | 增强策略设计（谨慎水平翻转，左右侧） | [ ] |
| 4.3 | 背鳍轮廓特征（CurvRank 风格：缺口、凹陷、曲率） | [ ] |
| 4.4 | 特征对比（外观/轮廓/融合） | [ ] |
| 4.5 | 特征可视化与错误分析 | [ ] |

**伪标签与度量学习**

| # | 任务 | 状态 |
|---|---|---|
| 5.1 | 生成伪标签（仅高可信确认簇，独立版本化；当前为 Candidate 级初审标签） | [~] |
| 5.2 | 度量学习训练（ArcFace / Triplet / 对比学习） | [~] |
| 5.4 | 难例主动审核（低置信边界样本） | [ ] |

**自动化系统**

| # | 任务 | 状态 |
|---|---|---|
| 6.2 | 自动判断左右侧与质量（模型化人工标记） | [ ] |
| 6.10 | 多头同框检测（NN relationship 候选 + 多归属归档）：YOLO 取全部框 + IoU 去重；**语义红线：同框多头 ≠ 亲缘关系**，只能标记疑似供人工判断 | [ ] |

**文档与工程基建**

| # | 任务 | 状态 |
|---|---|---|
| D.2 | 参考仓库 SOURCE_MAP（CetaMatch(MIT)/MiewID(无LICENSE)/DINOv2(Apache-2.0) 等） | [~] |
| D.3 | 数据伦理与合规（不公开敏感地点/坐标/未经授权影像） | [ ] |

## 8. 科研边界

- 一次性实验脚本在 `experiments/`，正式功能一律走 `src/whitewhale/` + 上述入口；
- 实验结果只追加记录于 `EXPERIMENT_LOG.md`（配置、结果、结论），删除脚本不删除实验记录；
- 所有输出保留 `image_id` / `relative_path` 可追溯字段；
- 聚类与检索结果 = Candidate，人工确认后才能叫个体；`uncertain` / `reject` 是合法审核结论，不强行归档。
