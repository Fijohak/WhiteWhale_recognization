# M4 训练与模型生命周期

M4 把训练从服务器进程中彻底分离：Ubuntu 只冻结数据、签出租约、验证产物并控制上线门禁；带 GPU 的成员电脑执行 Detector/Re-ID 训练、固定评估和 Catalog 重建。Worker 不连接 PostgreSQL，也不会得到租约之外的图片、Crop 或权重。

## 数据与 Split

- Dataset Version 是不可变快照，必须同时具有 `train`、`val`、`calibration`、`test`。
- 只有 `provider_confirmed`、`project_verified`、`high_trust_pseudo_label` 可以成为监督标签。
- Sequence、Encounter、近重复组和同一原图派生 Crop 均不得跨 Split。
- 训练 Manifest 只含 train/val/calibration；Re-ID 与 Detector Worker 都会在下载前再次拒绝 test。
- calibration 不参与梯度更新，只用于上线前阈值标定；固定 test 只在 Evaluation Job 中授权。

## GPU Worker

Re-ID 训练直接使用服务器冻结的 train/val 分配，不在 Worker 端重新切分。每个 epoch 的 `last.pt` 会立即作为 Artifact 上传并登记为带 stage/epoch/step 的已验证 Checkpoint；恢复任务只能引用相同 Dataset、任务类型、模型族和配置的恢复点，并恢复模型、优化器和历史记录。

Detector 训练把租约中的原图和已审核 bbox 转成临时 YOLO 数据集，多目标原图写入多条标注。calibration 不进训练。Worker 需显式提供本地基础权重：

```bash
python scripts/run_worker.py run \
  --api https://<server>/ \
  --token-file worker-token.json \
  --detector-training-base /path/to/local-yolo-base.pt \
  --device 0
```

归档能力仍使用 `--detector-weights` 与 `--reid-checkpoint`；这些权重只存在于 Worker 本机或服务器文件库，不进入 Git。

## 固定评估与上线

- Re-ID：calibration 生成 accept/uncertain 相似度阈值；test 计算跨 Sequence Rank-1 和 mAP。
- Detector：calibration 基于 IoU 0.5 搜索 F1 最优置信度；test 计算 precision、recall、F1。
- 若已有同模型族 Production，Worker 使用同一个固定 test 同时计算候选与现网指标及差值。
- 评估指标只能从租约 Worker 上传的 JSON Artifact 登记，API 不能改写报告内容。
- 权重文件进入 `models/` 前再次校验 SHA-256，并绑定完整 Model Manifest。
- Detector 在 Reviewer 发起且全部评估门禁通过后切换对应 Production 指针。
- Re-ID 还必须为所有 active Observation 重算 Embedding，生成行绑定的 `embeddings.npy` 和 `IndexFlatIP`。服务器复核模型、维度、Observation 顺序和 Faiss 内容，先激活兼容 Catalog，再切换 Production Model 指针。

上线失败不会改变现有 Production Model 或 active Catalog；所有申请、切换和上一版本均记录在追加式事件中。
