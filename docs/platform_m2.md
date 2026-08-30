# 协作平台 M2 归档闭环运行手册

M2 将现有 YOLO、Re-ID 和 HDBSCAN 算法接入租约 Worker。服务器仍是唯一事实源：Worker 不连接 PostgreSQL、不决定正式身份，也不能提交历史匹配结论；历史 Top-K 由服务器使用 active Catalog 重新计算。

## 1. 派发批次归档任务

普通或 iDolphin 文件夹上传并导入 Batch 后，operator/admin 在网页或 API 选择已登记的模型、检测器和预处理协议，调用：

```text
POST /api/batches/{batch_id}/archive-jobs
```

服务器从 `images` 表生成不可伪造的图片 ID、原始相对路径、大小和 SHA-256 清单，并创建 `batch_archival` Job。重复提交相同 Batch、模型和 Pipeline Config 会返回同一个幂等 Job。

示例请求体：

```json
{
  "model_version": "reid-r4",
  "detector_version": "yolov8-dorsalfin-v1",
  "preprocess_id": "megadescriptor-center224-v1",
  "pipeline_config": {
    "min_cluster_size": 3,
    "det_conf": 0.25
  },
  "required_vram_mb": 4096,
  "max_attempts": 3
}
```

## 2. 启动 4060 Laptop Worker

成员电脑安装完整 `requirements.txt`，取得与服务器登记信息一致的检测和 Re-ID 权重，然后运行：

```bash
python scripts/run_worker.py register \
  --api 'https://服务器地址/' \
  --registration-code '一次性登记码' \
  --token-file "$HOME/.config/whitewhale/worker.json" \
  --gpu-model 'RTX 4060 Laptop' \
  --vram-mb 8192 \
  --cuda-version '12.6' \
  --capabilities 'batch_archival' \
  --model-versions 'reid-r4'

python scripts/run_worker.py run \
  --api 'https://服务器地址/' \
  --token-file "$HOME/.config/whitewhale/worker.json" \
  --detector-weights models/detectors/yolov8n_dorsalfin.pt \
  --reid-checkpoint outputs/metric_learning/r4/best.pt \
  --device cuda \
  --heartbeat-seconds 60
```

运行期间 Worker 每 60 秒续租。它只可下载当前 Job 所属 Batch 的原图；下载后逐图复验 SHA-256。租约失效、设备不匹配或访问其他 Batch 时，服务器拒绝请求。

## 3. 算法与产物语义

Worker 的生产 Handler 依次执行：

1. YOLO 背鳍检测与 Crop；当前 M2 每张图产生一个主 Crop，多目标 N-Crop 在 M3 接入。
2. 指定 Checkpoint 的 Re-ID Embedding，结果重新 L2 normalize 并拒绝 NaN、Inf 和零向量。
3. HDBSCAN 批内聚类；正常簇保留成员概率，`-1` 噪声逐图形成独立候选，绝不把所有噪声合并成一个个体。
4. 生成 ZIP Artifact，包含 `manifest.json`、`embeddings.npy` 和 Crop 文件；Artifact Header 同时绑定模型、检测器、预处理、Pipeline digest 和 Embedding 行序摘要。

服务器再次验证 ZIP 路径、解压上限、Artifact SHA、Schema、Batch、模型协议、Crop 清单、Embedding 维度/有限性和行绑定。Worker 返回的任何历史匹配字段均被拒绝。

## 4. 两轮审核与发布

归档 Artifact 投影后：

```text
registered
→ candidate_ready
→ 单人批内簇纯度审核
→ under_review
→ 三人独立历史身份审核
→ approved
→ catalog_staged
→ published
```

- 普通簇纯度由 1 名 reviewer 确认，只表示批内候选可用。
- Existing 必须 3 人选择同一 UUID；New 至少 2 票，冲突规则由服务器固定。
- 审核前 `confirmed_individuals` 和 `observations` 不会增加。
- 审核后服务器从 Observation 对应的已验证 Embedding 构建不可变 Faiss `IndexFlatIP`。
- 激活前复验索引类型、维度、行数、向量摘要、Membership digest 和文件 SHA；失败不改变现有 active Catalog。
- reviewer 可发起激活或回滚，但不能直接修改活动指针。

## 5. 网页页面

React/TypeScript 网页现包含：文件夹上传、批次阶段、独立盲审、正式个体目录和 Catalog 版本。审核页只显示当前审核人的票；其他人的未完成票不会返回客户端。Catalog 和查询结果始终展示 Model Version、Catalog Version 与校准状态；`provisional_unvalidated` 不得解释为开放集阈值已经标定。

## 6. 当前验证证据

- 服务器派发清单只来自 Batch/Image 正式事实，且幂等。
- Worker 下载哈希不一致时，在运行检测前失败。
- 真实生产 Handler 默认调用现有 YOLO、Re-ID、HDBSCAN 模块；测试通过注入轻量替身验证编排协议。
- 长任务心跳发生在 Artifact 上传之前，避免 5 分钟租约在 GPU 推理中自然过期。
- Worker 归档包到审核、正式身份、Catalog 发布和查询已有 PostgreSQL 端到端测试。
- 全仓库回归：299 个测试、85 个子测试通过；前端生产构建和高危依赖审计通过。
