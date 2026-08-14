# scripts/ 目录导航

按数据处理流程排序，每个脚本可独立运行（`python scripts/xxx.py --help` 查看参数）。

## 数据索引
- `scan_dataset.py` — 扫描原始目录，生成数据清单（manifest）与统计
- `build_pilot_set.py` — 构建 pilot 数据集（高分 Anchor 照片表 pilot_set.csv）

## 特征与模型
- `extract_embeddings.py` — 用预训练模型（MegaDescriptor / DINOv2）批量提取特征
- `train_metric_learning.py` — 伪标签 ArcFace 度量学习训练；`--extract` 用微调模型重新提取特征
- `confidence_check.py` — 微调前后置信度体检（对级相似度分布 / 阈值判定 / P@K）

## 检索与聚类（候选，非身份）
- `local_reid_benchmark.py` — 本地弱标签检索评估（A 代表图 / B leave-one-out / C 跨序列）
- `pub_reid_benchmark.py` — 公开数据（Beluga / HappyWhale）检索基线
- `hdbscan_cluster.py` — HDBSCAN 候选聚类（-1 是合法噪声，结果只能叫 Candidate Cluster）

## 人工审核（已完成，保留复盘）
- `review_app.py` + `review_app.html` — 自建中文审核网页（端口 8001），审核结果在 `outputs/review/`
- `fiftyone_review.py` — FiftyOne 审核流程（曾被 review_app 取代）

## 辅助
- `contact_sheets.py` — 候选簇拼图（已被审核网页取代，保留备用）

## 约定
- 所有脚本禁止硬编码本地绝对路径；原始数据（`I:/`）只读；
- 输出一律带 `image_id` / `relative_path` 可追溯字段；
- 聚类与检索结果 = Candidate，人工确认后才能叫个体。
