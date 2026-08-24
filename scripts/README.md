# scripts/ 目录导航

正式入口脚本（薄 wrapper），实现逻辑都在 `src/whitewhale/`，默认参数从 `configs/*.yaml` 读取。
完整用法见根目录 `README.md`（每个脚本可 `python scripts/xxx.py --help` 查看参数）。

## 主流程入口

| 脚本 | 功能 |
|---|---|
| `prepare_data.py` | 数据盘点：`scan` 扫描原始目录生成 manifest；`build-pilot` 生成 pilot_set.csv |
| `run_pipeline.py` | 批内归档管线：YOLO 检测裁剪 → r3 特征 → HDBSCAN → 子簇化 → 簇级投票匹配历史库 → 审核材料（`--pool` 验证模式 / `--input-manifest` 新批次） |
| `launch_review.py` | 人工审核网页（端口 8001）；`--export` 导出审核结果 |
| `launch_query.py` | 个体查询客户端（端口 8000）：上传照片 → 检测裁剪 → 检索 → 三态判定 |
| `train_reid.py` | 特征模型训练（ArcFace 两阶段；`--hard-negative` 默认开 = r3 正式链路；`--extract` 重提特征） |
| `evaluate.py` | 特征评估（`--mode retrieval` 个体级 R@1/mAP；`--mode pairs` 同/跨个体分布 + FA5% 阈值建议） |
| `run_cross_time_batch.py` | 跨时间批次驱动：历史库特征 + 逐批次跑批内管线并匹配历史库 |

## 工具与训练辅助

| 脚本 | 功能 |
|---|---|
| `assign_pool.py` | 同群散图划分：散图 → 同群已确认个体 Top-K 候选（低分标记疑似新个体） |
| `contact_sheets.py` | 候选簇拼图（已被审核网页取代，备用） |
| `train_detector.py` | 训练 YOLO 背鳍检测器（数据由 `build_yolo_det_dataset.py` 构建） |
| `build_yolo_det_dataset.py` | 构建 YOLO 检测数据集（按 Sequence 划分） |
| `annotate_sam.py` | SAM vit_b 辅助预标注（人工剔除后进 YOLO 训练） |

## 约定

- 脚本不硬编码本地绝对路径；原始数据（`I:/`）只读；
- 输出一律带 `image_id` / `relative_path` 可追溯字段；
- 聚类与检索结果 = Candidate，人工确认后才能叫个体；
- 一次性实验脚本在 `experiments/`，正式功能不在 scripts 内堆实验代码。
