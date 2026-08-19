# scripts/ 目录导航

按数据处理流程排序，每个脚本可独立运行（`python scripts/xxx.py --help` 查看参数）。

## 数据索引
- `scan_dataset.py` — 扫描原始目录，生成数据清单（manifest）与统计
- `build_pilot_set.py` — 构建 pilot 数据集（高分 Anchor 照片表 pilot_set.csv）

## 特征与模型
- `extract_embeddings.py` — 用预训练模型（MegaDescriptor / DINOv2）批量提取特征
- `extract_r3_yolocrop.py` — 用 r3 模型提取 YOLO 检测裁剪图特征（工具链 gallery / 散图池两份，2026-08-17）
- `train_metric_learning.py` — 伪标签 ArcFace 度量学习训练；`--extract` 用微调模型重新提取特征
- `train_metric_learning_hn.py` — 跨群 hard negative 微调（r3，batch 内动态挖掘）
- `confidence_check.py` — 微调前后置信度体检（对级相似度分布 / 阈值判定 / P@K）

## 检索与聚类（候选，非身份）
- `local_reid_benchmark.py` — 本地弱标签检索评估（A 代表图 / B leave-one-out / C 跨序列）
- `pub_reid_benchmark.py` — 公开数据（Beluga / HappyWhale）检索基线
- `hdbscan_cluster.py` — HDBSCAN 候选聚类（-1 是合法噪声，结果只能叫 Candidate Cluster）
- `eval_cluster_retrieval.py` — 簇级检索评估（多帧投票 vs 单图，实验 E5）
- `eval_openset_preview.py` — 跨群未知个体预演（开放集拒识，实验 E3/E4 复跑）
- `eval_pool_archival.py` — 散图归档场景检索对比（中心 vs YOLO 裁剪，实验 E2）

## 簇级归档管线（真实流程，实验 E6）
- `pipeline_archival.py` — 新批次全流程：YOLO 检测裁剪 → r3 特征 → HDBSCAN 批内候选聚类 → 簇级多帧投票匹配历史库 → 审核清单 + 代表图 + 候选簇拼图；`--pool` 复用散图池预提取产物验证；`--input-manifest 清单.csv` 跑任意新批次

## 散图归档（工具链）
- `assign_pool.py` — 同群散图划分：散图 r3+YOLO 特征 → 同群已确认个体 Top-K 候选（2026-08-17 起用 r3+YOLO 链路）

## 背鳍检测（工具链）
- `annotate_sam.py` — SAM vit_b 辅助预标注（人工剔除后进 YOLO 训练）
- `build_yolo_det_dataset.py` — 构建 YOLO 检测数据集（按 Sequence 划分）
- `train_yolo_detector.py` — 训练 YOLOv8 背鳍检测器（权重 models/detectors/）
- `detect_and_crop.py` — YOLO 检测 + 非均匀扩展裁剪（未检出回退中心 0.45 窗）

## 人工审核（已完成，保留复盘）
- `review_app.py` + `review_app.html` — 自建中文审核网页（端口 8001），审核结果在 `outputs/review/`
- `fiftyone_review.py` — FiftyOne 审核流程（曾被 review_app 取代）

## 辅助
- `contact_sheets.py` — 候选簇拼图（已被审核网页取代，保留备用）

## 约定
- 所有脚本禁止硬编码本地绝对路径；原始数据（`I:/`）只读；
- 输出一律带 `image_id` / `relative_path` 可追溯字段；
- 聚类与检索结果 = Candidate，人工确认后才能叫个体。
