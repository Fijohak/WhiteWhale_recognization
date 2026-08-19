# 任务清单（中华白海豚个体识别）

> 本清单是项目的执行索引，主干采用阶段 0–6 结构。
> 所有阶段必须遵守的**全局约束**见文末「全局约束」一节。
> 状态：`[ ]` 待办 / `[x]` 已完成 / `[~]` 进行中
>
> **方向调整（2026-08-11）**：原始文件夹结构**不直接等于个体身份**。历史数据整理时工作人员从每个体挑选一张代表照片（Anchor），其余照片混入公共照片池（Pool）。项目第一目标从"训练分类模型"调整为**"Anchor-based 检索 + 人工审核，恢复可信个体目录（Confirmed Individual Catalogue）"**。详见 `docs/anchor_pool_semantics.md`。
>
> **任务定位补充（2026-08-17，与数据提供方对齐）**：任务按时间尺度分层——**批内**（同一天）：个体独立、散图池照片有主，做归档分配与池找回（封闭集）；**跨时间**（不同批次）：无全局编号、同体对为**推定**，做历史个体库匹配与**疑似新个体发现**（开放集），结果仅作候选须人工核验。**项目负责人期望：辨别新个体的出现**；跨时间 Re-ID 不作为当前目标。个体特征跨时间会因成长/意外事件漂移（README §5.3 A11–A13、§5.5）。

---

## 阶段 0：语义确认

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| 0.1 | 阅读原始 `.txt` 调查记录文件 | 解析日期、位置、航次、人员、个体记录信息 | 字段含义记录 | [x] |
| 0.2 | 解析 `individual.docx` | 确认是否包含人工个体对照结果 | 解析结果记录 | [~] |
| 0.3 | 确认评分区间含义 | `70-79` / `80 and above` = 历史评分筛选结果，**不代表个体身份** | 结论记录 | [x] |
| 0.4 | 确认代码含义 | `MO` / `RAY` / `DEREK` = 拍摄者代码 | 结论记录 | [x] |
| 0.5 | 确认关系目录含义 | `nn relationship`、`sj of 01` 的定义 | 结论记录 | [~] |
| 0.6 | 确认目录结构 | 01/03 = 同一天两群；processed/Processed 2 已收进 zip | 结论记录 | [x] |
| 0.7 | 确认数字文件夹编号语义 | **子文件夹 = 历史挑选的代表照片（Anchor），不代表全局个体 ID**；跨调查未核验 | 结论记录 | [x] |
| 0.8 | 确认数据授权 | 是否允许训练、发布模型或公开派生数据 | 结论记录 | [ ] |
| 0.9 | 编写数据语义与假设文档 | 已确认事实与推测并入 README §5.2–5.3 | `README.md` §5 | [x] |
| 0.10 | 更新 Anchor / Pool 语义 | 高分目录子文件夹 = Anchor；`70-79` 等公共池 = Unresolved Image Pool | `docs/anchor_pool_semantics.md` | [x] |

---

## 阶段 1：数据索引

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| 1.1 | 搭建项目目录结构 | data / outputs / models / configs / scripts / src / docs | 目录骨架 | [x] |
| 1.2 | 编写数据扫描脚本 | 多根支持（01/03）、跳过 ZIP/视频、解析 session/评分区间/分组、candidate-label、SHA-256、读尺寸、记损坏 | `scripts/scan_dataset.py` | [x] |
| 1.3 | 运行扫描 | 生成 Manifest、统计 JSON、目录树、异常文件表 | `outputs/index/` 四件套 | [x] |
| 1.4 | 精确重复检测 | SHA-256 全量比对（已运行） | `dataset_stats.json` 重复字段 | [x] |
| 1.5 | 第一轮结果质量检查 | 抽查路径、标签、candidate_groups、sequence_guess | 检查记录 | [x] |
| 1.6 | 统计分析 | 批次数、图片数、格式与分辨率分布、质量区间分布、Anchor 数 | `dataset_stats.json` | [x] |
| 1.7 | 汇总数据盘点报告 | 数据规模、已确认事实、假设清单（并入 README §5） | `README.md` §5 | [x] |
| 1.8 | 散图分拣（人工待办） | 把 70-79 的 207 张散图人工分配到对应个体文件夹，更新 Manifest | 更新后 Manifest | [ ] |

---

## 阶段 2：Anchor-based 基线（方向调整后）

> 原「Pilot Set 与最小基线」改为 Anchor-based 流程：**高分目录子文件夹 = Anchor，公共池 = Gallery**。

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| 2.1 | 选择 Pilot Set | 高分 43 组代表照片（Anchor，199 张）+ 207 张散图（Pool） | 清单文件 | [x] |
| 2.2 | 基线 embedding 提取 | 预训练模型（MegaDescriptor-T-224 / DINOv2 / MiewID）直接提特征，**不裁剪** | `outputs/embeddings/` | [x] |
| 2.3 | Anchor Top-K 检索 | **query=Anchor，gallery=Pool（+ 同调查 Anchor 集）**，报告 Recall@1/5/10、mAP | `outputs/nearest_neighbors/` | [ ] |
| 2.4 | 候选拼图 | 每个 Anchor 的 Top-K 候选照片、相似度、来源批次 | `outputs/contact_sheets/` | [ ] |
| 2.5 | 基线诊断 | 特征是否区分个体、是否主要按背景/拍摄者聚类 | 基线实验报告 | [ ] |
| 2.6 | HDBSCAN 降为辅助 | 仅用于无 Anchor 照片 / 发现潜在新个体 / 辅助候选；结果只能叫 Candidate Cluster | 结论记录 | [ ] |
| 2.7 | 同群散图划分（Pool Assignment） | **query=散图（207 张），gallery=同群已确认个体（133 张）**，中心裁剪 0.55 + 预训练特征，群内 Top-K 候选 + 低分疑似新个体标记；候选需人工审核（2026-08-14 完成） | `scripts/assign_pool.py` + `outputs/pool_assignment/` | [x] |

---

## 阶段 3：人工基准集

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| 3.1 | 人工核验首批候选簇 | 确认 / 拆分 / 合并 / 标记未知；记录操作者、时间、依据 | `data/reviewed/` | [ ] |
| 3.2 | 建立人工评估集 | 多个体 × 多日期 × 多角度；按 Sequence 划分，训练评估不重叠 | `data/evaluation/` | [ ] |
| 3.3 | 建立确认关系表 | confirmed_same / confirmed_different / possibly_same | `relations.csv` | [ ] |
| 3.4 | 确定可接受错误合并率 | 优先控制错误合并风险（种群统计低估） | 结论记录 | [ ] |
| 3.5 | Pilot 人工确认集 | ~20–30 个 Anchor，人工确认其少量同个体照片（2–4 张），用于比较 retrieval baseline | `data/reviewed/pilot/` | [ ] |

---

## 阶段 4：自监督与轮廓特征

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| 4.1 | 自监督微调 | 用全部合格图片微调 DINOv2 / SimCLR / MoCo / BYOL / MAE 之一 | `models/self_supervised/` | [ ] |
| 4.2 | 增强策略设计 | 轻微缩放/旋转/亮度/对比度；谨慎使用水平翻转（左右侧） | 配置文档 | [ ] |
| 4.3 | 背鳍轮廓特征 | CurvRank 风格：后缘缺口、凹陷、曲率、多尺度轮廓 | `models/contour/` | [ ] |
| 4.4 | 特征对比 | 外观 embedding / 轮廓 embedding / 融合特征对比 | 对比报告 | [ ] |
| 4.5 | 特征可视化与错误分析 | 检查模型关注区域、错误案例归因 | `outputs/visualizations/` | [ ] |

---

## 阶段 5：伪标签与度量学习

> 仅当人工确认数据逐渐积累后才进入本阶段（方向调整 §4）。

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| 5.1 | 生成伪标签 | 仅用人工确认或高可信约束支持的簇；独立版本化（当前：人工初审标签，Candidate 级，未复核） | `data/reviewed/` 版本 | [~] |
| 5.2 | 度量学习训练 | ArcFace / Triplet / 对比学习（每类 1 张时注意少样本限制） | `models/metric_learning/` | [~] |
| 5.3 | 重新提取特征与聚类 | 用新模型重跑 embedding → 检索 → 聚类 | `outputs/` 更新 | [x] |
| 5.4 | 难例主动审核 | 低置信边界样本主动学习式人工审核 | 审核记录 | [ ] |
| 5.5 | 迭代评估 | 微调前后对比检索与聚类指标 | 评价报告 | [x] |
| 5.6 | 后续升级（2026-08-14 记入） | ① 微调模型 + 跨群 hard negative **已完成**（2026-08-17，r3，实验 E4：验证个体 R@1 0.706→0.765；A 协议 mAP 0.435→0.533；E3 预演 known recall 36-47%→52-61%，三分布分离首次成立）；② 阈值标定需基于微调特征重做——已由 E3/E4 预演完成（FA≤5% 阈值 0.5-0.6） | `scripts/train_metric_learning_hn.py` + `outputs/metric_learning/r3/` | [x] |

---

## 阶段 6：自动化系统

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| 6.1 | 背鳍自动检测 | 已落地：SAM vit_b 辅助预标注 199 张 → 人工剔除 30 张 → YOLOv8n 训练（134/33 按 Sequence 划分，fliplr=0，imgsz=1024，早停 62 轮）→ val mAP50=0.635 / R=0.581（2026-08-17，实验 E1） | `models/detectors/yolov8n_dorsalfin.pt` + `scripts/annotate_sam.py` + `scripts/build_yolo_det_dataset.py` + `scripts/train_yolo_detector.py` | [x] |
| 6.1b | 检测裁剪评估 | 特写 benchmark（B 口径）打平无提升（YOLO 0.310/0.364 vs 中心裁剪 0.304/0.372）；散图 202 张检出 183（90.6%），非特写图泛化可用 → 支撑散图自动归档工作流（实验 E1，结论入 EXPERIMENT_LOG） | `outputs/reports/benchmark_yolo_crop/` + `outputs/crops_yolo/` + `outputs/crops_yolo_pool/` | [x] |
| 6.1c | 散图归档场景检索对比 | 已落地：202 张散图，中心 0.55 vs YOLO 裁剪；Top1 归档把握度 YOLO 显著更高（中位 0.866 vs 0.812，Wilcoxon p=3.4e-16，低置信 1.0% vs 3.5%），相邻帧互检无显著差异（2026-08-17，实验 E2） | `scripts/eval_pool_archival.py` + `outputs/reports/pool_archival/` | [x] |
| 6.2 | 自动判断左右侧与质量 | 模型化人工标记 | 模块 | [ ] |
| 6.3 | 自动特征提取与 Top-K 检索 | 新图 → 检测 → 裁剪 → embedding → 检索 | 检索模块 | [ ] |
| 6.4 | 未知个体标记 | 低于阈值标记"疑似新个体/需人工判断" | 阈值配置 | [ ] |
| 6.5 | 人工核验界面 | 确认 / 合并 / 拆分 / 拒绝 / 新个体登记 + 审计 | `src/` 核验界面 | [ ] |
| 6.6 | 个体目录版本管理 | 可持续更新的个体数据库与历史记录 | 数据库 + 流程文档 | [ ] |
| 6.7 | 开放集评价 | 第一步已完成预演（2026-08-17，实验 E3）：01↔03 互检 + 阈值标定；预训练特征无拒识能力（FA≤5% 时 known recall 仅 16-38%），微调特征可用但 known recall 仅 36-47%；正式评价待跨批数据 | `outputs/reports/openset_preview_*/` + `scripts/eval_openset_preview.py` | [~] |
| 6.8 | 簇级检索（多帧投票） | 评估已完成（2026-08-17，实验 E5：探针/库拆分，多帧投票 vs 单图——库内簇级 R@1 稳定提升 8-12pp，03 群 0.727 vs 0.647，n=22；拒识方向一致但 known 侧 n=6 过薄；发现 known+ 双峰结构）。**真实流程接入已完成**（2026-08-17，实验 E6）：`scripts/pipeline_archival.py` 全流程（检测 → r3 特征 → HDBSCAN → 簇级匹配 → 审核清单/代表图/拼图），散图池验证通过（8 簇 + 89 噪声，2 簇 match、6 簇 suspected，连拍聚簇语义正确，噪声逐图不合并）；散图簇级分数普遍 0.44-0.55 属特征空间现状，阈值 0.58 语义为"宁可多标疑似" | `scripts/eval_cluster_retrieval.py` + `scripts/pipeline_archival.py` + `outputs/reports/cluster_retrieval/` + `outputs/cluster_archival/` | [x] |
| 6.9 | 工具链接入 | 已落地（2026-08-17）：query_app / assign_pool 统一升级为 r3 跨群 HN 微调特征 + YOLO 检测裁剪（E1/E2/E4 结论落地）；gallery 与散图池特征由 `scripts/extract_r3_yolocrop.py` 预提取；query_app 上传图默认走检测裁剪（--no-detect 可关）、阈值 0.55（E4 标定区间中值）；assign_pool 群内 leave-one-out R@1 01:0.842 / 03:0.912；r3 分数空间整体下移，散图低分占比升高属特征现状，阈值待数据标定 | `src/query_app.py` + `scripts/assign_pool.py` + `scripts/extract_r3_yolocrop.py` + `outputs/embeddings/embeddings_*_r3_yolocrop.npy` | [x] |
| 6.10 | 多头同框检测（NN relationship 候选 + 多归属归档） | 用户提议（2026-08-18），技术可行已确认：YOLOv8 天然支持多目标，现代码只取 `boxes[0]`（最高置信框），改为取全部框即可。功能：① 画面存在多头海豚时输出**同框候选清单**（NN candidate，弱线索）；② 每个背鳍框各裁一张、各走个体匹配、图片可多归属归档（符合 A13 散图有主语义）。**语义红线：同框多头 ≠ 亲缘关系**（同群≠有亲缘），只能标记"疑似同框关系"供人工判断，绝不自动标亲缘；与 TASKS 0.5（nn relationship 目录含义确认）相关联。**当前数据多为单背鳍特写，价值待真实海上批次验证，暂不实现**；实现时：`detect_and_crop.py` / `pipeline_archival.py` 的 `_detect_all` 加多框支持 + IoU 去重 + `nn_candidates.csv` 输出 | 待办（未实现） | [ ] |

---

## 文档与工程基建（贯穿全程）

| # | 任务 | 说明 | 交付物 | 状态 |
|---|------|------|--------|------|
| D.1 | 跨物种迁移学习方案 | 源域轮廓 vs 目标域纹理的负迁移风险与消融实验设计 | `docs/cetacean_reid_transfer_learning_plan.md` | [x] |
| D.2 | 参考仓库 SOURCE_MAP | 已核查：CetaMatch(MIT)/MiewID(无LICENSE)/DINOv2(Apache-2.0) 等 | `references/SOURCE_MAP.md` | [~] |
| D.3 | 数据伦理与合规 | 不公开敏感地点/坐标/未经授权影像；发布前权限检查 | 文档 | [ ] |
| D.4 | 实验日志 | 每次实验记录配置、结果、结论（E1：YOLOv8 检测器已记录） | `EXPERIMENT_LOG.md` | [x] |
| D.5 | Anchor/Pool 语义文档 | 方向调整后的数据语义（见阶段 0.10） | `docs/anchor_pool_semantics.md` | [x] |

---

## 全局约束（验收前提）

1. **Anchor / Pool 语义**：高分目录子文件夹 = 代表照片（Anchor），公共池 = Unresolved Image Pool；**目录不直接等于个体 ID**，不作全局 ID。
2. **原始数据只读**：所有处理写入独立目录；处理样本可追溯到原始路径。
3. **连拍不泄漏**：连续拍摄序列不可拆分，训练/评估按 Sequence 划分。
4. **左右侧分离**：记录朝向（left/right/unknown），左右侧分别比较，不默认水平翻转。
5. **保守聚类**：允许噪声与低置信样本，宁可拆分不可错并（影响种群统计）。
6. **人工核验闭环**：模型结果只是候选，正式身份须人工确认并留审计。
7. **未知个体支持**：开放集检索，不强制分类。
8. **伪标签纯净**：仅高可信确认结果进入训练，逐轮版本化。
9. **不提前训练**：在可信身份数据建立以前，不得训练个体分类器 / ArcFace / 制造伪标签；模型仅作预训练特征提取器。
