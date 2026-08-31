# Plan

本计划把现有中华白海豚检测、Re-ID、批内聚类、人工审核和单图查询代码，演进为一个以 TypeScript 网页为主要入口、Ubuntu 服务器为唯一事实源、多台成员 GPU 电脑提供计算能力的内部协作平台。系统坚持“算法只产生候选、人工确认正式身份”，并把图片、身份、审核、模型、训练数据集和 Faiss 目录全部做成可追溯、可版本化对象。

> 执行状态（2026-08-31）：M1–M4 已通过分层验收，M5 交付强化实施中。控制面、上传、GPU Worker 租约、两轮归档、三人盲审、正式身份、关系证据、不可变 Catalog，以及 Detector/Re-ID 的 Dataset、训练、Checkpoint 恢复、固定评估和模型上线门禁均已落地。

## Scope

- **In**：文件夹上传与导入、批内归档、历史身份匹配、多目标共现、疑似关系集合、多人审核、正式个体目录、分布式 GPU Worker、训练与评估、模型/目录发布、查询与可视化、离线部署、三分钟开发分支自动更新。
- **Out**：公开科普网站、自动确认身份或亲缘、直接暴露 PostgreSQL、服务器本机 GPU 训练、依赖公网云服务、首阶段自动定时备份、首阶段自定义 `.top` 公网域名。

## 1. 已确认的产品决策

1. 第一版是课题组内部工作平台，不是公开网站。
2. 网页使用 React、TypeScript 和 Vite；模型和数据处理继续使用 Python。
3. 后端使用 FastAPI，PostgreSQL 是唯一正式业务数据库，大文件保存在服务器本机文件库。
4. Ubuntu 服务器不承担 GPU 训练或批量推理；成员的 4060 Laptop 均可作为 Worker 抢占任务。
5. Worker 不复制正式数据库，也不直接连接 PostgreSQL；它只下载当前租约授权的数据并回传产物。
6. 单批约 3000 张图片、总量约 2–20 GB，网页必须支持文件夹选择、分片上传、断点续传、进度显示和哈希去重。
7. 导入同时支持标准 iDolphin 目录和任意图片目录；二者最终生成同一种不可变 Manifest。
8. 用户通过局域网或 Tailscale 访问；Tailscale 是远程组网方式，不是核心系统依赖。
9. Tailscale 成员仍需应用内登录和角色授权；Tailscale 不能代替业务权限。
10. 当前只有开发环境。管理员在 Ubuntu 上选择部署分支，服务器每三分钟检查并部署远端新提交。
11. 首阶段不启用自动定时备份，但必须提供一致性手动导出和恢复校验能力；自动增量备份列入后续强化。
12. 第一版包含检测器训练、Re-ID 训练、评估、Checkpoint、模型比较、上线门禁和 Faiss 重建，但计算仍发生在 GPU Worker。

## 2. 架构

### 2.1 运行拓扑

```text
成员浏览器
  │
  ├─ 局域网 HTTP/HTTPS
  └─ Tailscale Serve HTTPS (*.ts.net)
           │
           ▼
         Caddy
  ├─ /          React + TypeScript 静态构建产物
  ├─ /api/*     FastAPI
  └─ /media/*   受权限保护的图片与产物
           │
           ├─ PostgreSQL：正式元数据与状态
           ├─ 宿主机文件库：原图、Crop、Embedding、模型、目录和导出
           └─ PostgreSQL 任务租约
                         │
                         ▼
              多台 Python GPU Worker
```

### 2.2 组件边界

- **Web**：上传、导入向导、任务进度、审核、个体档案、关系集合、训练、模型、目录版本和系统管理。
- **API**：身份认证、角色授权、状态机、审核政策、任务调度、文件授权、产物校验、目录/模型发布门禁。
- **领域层**：Batch、Job、Candidate、Review、Individual、Catalog、Dataset、Model 等规则；不得依赖网页状态。
- **算法内核**：复用现有 `src/whitewhale/` 中的检测、Embedding、聚类、检索、训练和评估代码。
- **Worker**：执行检测、Crop、Embedding、聚类辅助、训练、评估和索引构建；不持有正式目录状态。
- **PostgreSQL**：唯一事实源；第一版直接实现租约队列，不引入 Redis/Celery。
- **文件库**：保存大文件；数据库只保存由服务端生成的安全相对路径、哈希、大小和 Manifest。

### 2.3 技术选择

- 前端：React、TypeScript、Vite。
- API：FastAPI，OpenAPI 自动生成 TypeScript Client 和类型。
- 数据访问与迁移：SQLAlchemy 2、Psycopg 3、Alembic。
- 反向代理与静态资源：Caddy。
- 数据库：PostgreSQL。
- 精确向量检索：Faiss `IndexFlatIP`，输入先 L2 normalize。
- 服务编排：Docker Compose；正式数据全部挂载宿主机 Volume。
- 后台运行与自动部署：systemd service + systemd timer。

## 3. 固定数据语义

| 对象 | 含义 |
|---|---|
| Image | 一张原始图片，原文件只读 |
| Crop | Image 中一个检测目标；一个 Image 可有 1..N 个 Crop |
| Sequence | 连续拍摄序列，训练和评估不可拆分 |
| Encounter | 同次相遇或调查单元，训练和评估不可拆分 |
| Source Group | 原始文件夹、提供方分组等弱来源信息 |
| Candidate Cluster | 算法产生的批内候选簇 |
| Confirmed Individual | 人工确认的正式个体，使用 UUID |
| Observation | Confirmed Individual 的一次图像观测，关联到 Crop |
| Catalog Version | 某一时刻不可变的正式个体目录快照 |
| Dataset Version | 某次训练使用的不可变样本、标签和 Split 快照 |

不可突破的红线：

- `01`、`02` 等目录名不是全局个体 ID。
- Source Group 只能作为来源或别名，不能成为监督身份标签。
- Candidate Cluster、Top-1 和高相似度都不能自动写成 Confirmed Individual。
- HDBSCAN `-1` 是合法噪声。
- `unknown`、`uncertain`、`unusable` 和 `disputed` 都是合法结果。
- 原图只读；所有结果保留 `image_id`、`crop_id`、`source_path`、来源哈希和处理版本。
- 未审核或未发布批次不能进入正式历史 Gallery。
- 当前 r4 权重和阈值作为旧生产产物导入时，必须保留 `provisional_unvalidated` 校准状态，不能宣称已完成开放集阈值标定。

## 4. 状态模型

### 4.1 Batch 业务阶段

Batch 只记录已经成功完成的业务阶段；Job 失败时 Batch 停留在最后成功阶段。

```text
registered
→ candidate_ready
→ under_review
→ approved
→ catalog_staged
→ published
→ archived
```

### 4.2 Job 状态

```text
queued
→ leased
→ running
→ uploading
→ validating
→ succeeded
```

异常终态或 Attempt 结果：

```text
failed
cancelled
lease_expired
```

### 4.3 候选与审核状态

候选分流：

```text
existing_match_candidate
suspected_unknown
needs_review
unusable
```

审核结论：

```text
confirm_existing
confirm_new
uncertain
reject
split_required
```

### 4.4 Catalog 与 Model 状态

Catalog：

```text
staged → active → retired
```

Model：

```text
training → candidate → evaluating → validated → production → retired
                                            └→ rejected
```

训练完成不能自动成为 `production`。

## 5. 上传与导入

### 5.1 分片上传

- 浏览器使用文件夹选择能力保留相对路径。
- 上传会话记录文件清单、总大小、相对路径、分片状态和最终 SHA-256。
- 默认分片大小为 32 MiB；失败后只重传缺失分片。
- 所有文件先进入 `data/staging/`，完成哈希、格式和路径校验后再原子登记。
- 服务端拒绝绝对路径、`..`、路径越界、同名冲突和未声明文件。
- 不完整上传由可配置保留期清理；清理只作用于已确认没有活跃会话引用的 staging 对象。

### 5.2 标准 iDolphin 目录

- 解析批次名、质量目录、数字子目录、散图池、原始相对路径和关系备注。
- 已有数字目录标签只表示批次内来源身份，不自动跨批次合并。
- `nn relationship` 原始目录图片自动加入系统 `nn_relationship` 虚拟集合：
  - `assignment_source = original_folder`
  - `membership_status = candidate`
  - 只表示疑似关系研究样本，不表示已确认亲缘。

### 5.3 任意图片目录

- 无法识别标准结构时进入导入向导。
- 至少补充批次名称和拍摄日期；地点、Sequence、Encounter、侧别和来源组可补充或明确记为 `unknown`。
- 普通目录和标准目录最终生成完全相同的 Manifest，后续算法不再依赖原目录格式。

## 6. 两轮归档业务流

### 6.1 第一轮：批内归档

```text
批次登记与原图哈希
→ 检测并为每个目标生成 Crop
→ Crop 质量检查
→ Embedding
→ Sequence / Encounter 聚合
→ HDBSCAN Candidate Cluster
→ 代表图与审核材料
→ 普通簇纯度审核
```

普通簇审核支持：确认候选组、拆簇、排除 Crop、标记不可用、请求重新检测/裁剪和标记不确定。单人确认只表示“批内候选审核结果”，不是跨时间身份确认。

### 6.2 第二轮：历史身份匹配

```text
已审核的本批个体实例
→ 查询当前 active Catalog 的 Faiss Top-K
→ 展示代表图、逐图分数、投票、质量、侧别和时间
→ 多人独立盲审
→ 关联 Existing 或创建 New
→ staged Catalog
→ Faiss 重建与完整性校验
→ 原子激活
```

任何阈值都绑定 Model Version、Detector Version、Crop Config、Preprocess ID 和 Pipeline Config digest。

## 7. 多目标、共现和疑似关系

### 7.1 多目标图片

- 支持一张 Image 检出 1、2、3 或 N 个目标。
- 每个检测目标拥有独立 `crop_id`。
- 多目标检测创建一个 `cooccurrence_event` 和 N 条 `cooccurrence_member`，不得使用固定 `individual_a/individual_b` 列。
- 身份确认前，成员的 `individual_id` 允许为空。
- 散图池多目标图片自动进入 `nn_relationship` 的 `review_pending` 队列。
- 重复框或错误 Crop 进入 `detection_conflict` 或 `rejected`。

### 7.2 疑似关系

- 系统不产生“已确认亲缘关系”。
- 界面、导出和论文数据只允许使用“疑似亲缘”“证据不足”“存在争议”或“已拒绝”等表述。
- N 个身份确认后可生成所有无序个体对作为疑似关系证据，但它们仍共享原始 Cooccurrence Event。
- 关系假设支持 `co_occurrence`、`repeated_association` 和 `suspected_kinship`，不支持 `confirmed_kinship`。

## 8. 审核政策矩阵

审核规则由服务端按任务类型固定，客户端不能降低人数或阈值。

| 审核任务 | 审核人数 | 通过规则 |
|---|---:|---|
| 普通簇纯度 | 1 | 单人确认批内候选质量 |
| 多目标真实性 | 3 | 三人完成后至少 2 人确认多目标 |
| 跨时间 Existing Match | 3 | 3 人选择同一个 Existing UUID |
| 创建 New Individual | 3 | 至少 2 人选择 New |
| 疑似亲缘 | 3 | 只能形成 suspected，不能形成 confirmed |
| 身份合并 | 3 | 3 人选择同一合并方案 |
| 身份拆分 | 3 | 3 人选择同一拆分方案 |
| 撤回照片重审 | 3 | 3 人选择同一结论 |
| Catalog 激活/回滚 | Reviewer 可发起 | 服务器门禁通过后执行 |
| Model 上线 | Reviewer 可发起 | 服务器门禁通过后执行 |

新个体票型规则：

| 票型 | 结果 |
|---|---|
| New / New / New | 创建新 UUID |
| New / New / Existing | 创建新 UUID，标记 `possible_duplicate` |
| New / New / Uncertain | 创建新 UUID，标记低共识 |
| Existing A / Existing A / Existing A | 关联 Existing A |
| Existing A / Existing A / New | 不自动合并，进入 conflict |
| Existing A / Existing B / New | uncertain/conflict |
| 三票均不一致 | uncertain/conflict |

审核人彼此不可见尚未完成的票。原始票永久追加保存；重新投票产生新事件，不覆盖旧票。

## 9. PostgreSQL 领域模型

### 9.1 用户与设备

- `users`
- `roles`
- `user_roles`
- `worker_devices`
- `worker_tokens`
- `worker_heartbeats`

### 9.2 采集事实

- `batches`
- `images`
- `crops`
- `sequences`
- `encounters`
- `source_groups`

### 9.3 任务与产物

- `jobs`
- `job_dependencies`
- `job_attempts`
- `job_leases`
- `job_events`
- `artifacts`
- `artifact_manifests`
- `detector_versions`
- `pipeline_configs`

### 9.4 候选与审核

- `candidate_clusters`
- `candidate_cluster_members`
- `match_candidates`
- `candidate_events`
- `review_tasks`
- `review_events`
- `reviewer_rosters`
- `review_consensus`
- `review_conflicts`

### 9.5 正式身份与目录

- `confirmed_individuals`
- `individual_aliases`
- `observations`
- `identity_events`
- `catalog_versions`
- `catalog_memberships`
- `active_catalog_pointer`

### 9.6 集合、共现与关系

- `collections`
- `collection_memberships`
- `cooccurrence_events`
- `cooccurrence_members`
- `relationship_hypotheses`
- `relationship_evidence`
- `relationship_events`

### 9.7 训练与评估

- `dataset_versions`
- `dataset_memberships`
- `dataset_splits`
- `training_runs`
- `training_checkpoints`
- `evaluation_runs`
- `evaluation_results`
- `model_versions`
- `model_promotion_events`

### 9.8 关键约束

- 正式实体使用 UUID。
- Crop 必须属于一个 Image。
- 同一 Catalog Version 中，一个 Crop 最多属于一个 Confirmed Individual。
- 同一时刻只有一个 active Catalog Version。
- Source Group 不能成为正式身份。
- Review Event、Identity Event、已发布 Artifact 和 Catalog Version 采用追加或不可变语义。
- 合并、拆分、撤回和更正通过新 Identity Event 和新 Catalog Version 表达，不物理删除已发布历史。
- 当前页面状态使用投影表加速读取，事件表保留完整证据；系统不采用复杂的纯事件溯源架构。

## 10. 文件与 Artifact 管理

```text
/srv/whitewhale/data/
├── raw/                 原图，只读
├── working/             非正式工作文件
├── artifacts/           Crop、Embedding、报告与日志
├── models/              权重与 Model Manifest
├── catalog_versions/    Catalog 快照和 Faiss 索引
├── exports/             手动导出与离线包
└── staging/             未完成上传和待校验产物
```

- 禁止把正式文件写入 Docker 容器层。
- 数据库只保存安全相对路径；路径由服务端生成，客户端不能直接拼接。
- `/media/*` 每次校验用户权限或 Worker 租约。
- 文件先写 staging，校验通过后原子移动，再由数据库事务发布引用。
- Worker 可只读下载原图、Crop、训练包、权重和配置；任务结束后必须清理本地缓存。
- 原始 EXIF 先登记入库；GPS、拍摄时间和相机信息按角色单独授权。

每个 Artifact Manifest 至少记录：

- Artifact、Job、Attempt、Batch ID。
- Artifact 类型、Schema Version、安全相对路径、SHA-256 和大小。
- Image/Crop 行绑定摘要。
- Producer Worker。
- Model、Detector、Preprocess 和 Pipeline Config 版本。
- 创建时间和幂等键。

## 11. GPU Worker 协议

### 11.1 注册与能力

Worker 上报：设备 ID、GPU 型号、显存、CUDA、Python/Worker 版本、支持的任务类型、已安装模型版本和当前容量。管理员使用一次性登记码签发可撤销、可轮换的设备令牌。

### 11.2 原子租约

PostgreSQL 使用 `SELECT ... FOR UPDATE SKIP LOCKED` 原子领取兼容任务。租约至少包含：

- `leased_by_device`
- `lease_token`
- `lease_expires_at`
- `last_heartbeat_at`
- `attempt_number`
- `idempotency_key`

规则：

- 同一任务同一时刻只能有一个有效租约。
- Worker 定期发送心跳；过期后任务可重新分配。
- 旧 Worker 使用失效 lease token 上传的迟到结果必须被拒绝。
- 重复完成请求不得产生重复产物。
- 达到最大尝试次数后进入人工处理队列。
- Worker 只能报告进度、成功或失败，最终 Job 状态由服务器决定。

### 11.3 Worker API

```text
POST /api/workers/register
POST /api/workers/heartbeat
POST /api/tasks/lease
POST /api/tasks/{task_id}/heartbeat
GET  /api/tasks/{task_id}/inputs
POST /api/tasks/{task_id}/artifacts
POST /api/tasks/{task_id}/complete
POST /api/tasks/{task_id}/fail
```

服务器派发前检查 Worker 能力和模型兼容性；回传后重新校验权重哈希、特征维度、检测/Crop/预处理配置、输入输出行绑定、NaN/Inf 和文件哈希。

## 12. 训练、评估与模型发布

### 12.1 Dataset Version

Dataset Version 保存 Image/Crop 成员、身份来源、Sequence、Encounter、Split、数据授权、Catalog Version、成员摘要和创建时间。

可作为监督身份标签的来源：

```text
provider_confirmed
project_verified
high_trust_pseudo_label
```

禁止作为监督身份标签的来源：

```text
source_group
candidate_cluster
suspected_kinship
unreviewed_match_candidate
```

### 12.2 Split 门禁

- 同一 Sequence 不跨 train/val/calibration/test。
- 同一 Encounter 不跨集合。
- 近重复图片不跨集合。
- 同一原图派生 Crop 跟随原图。
- 冻结测试集不能被训练 Worker 下载。
- 未知身份协议和已知身份更新协议分别保存。

### 12.3 Training Job

训练任务记录任务类型、Dataset Version、Model Family、Base Checkpoint、配置、Seed、最低显存、最长时间、Checkpoint 周期、恢复点和输出 Model Version。Worker 持续回传日志、epoch/step、阶段 Checkpoint 和心跳，并支持从已验证 Checkpoint 恢复。

### 12.4 Model Manifest

每套权重至少记录：

```yaml
model_id:
model_family:
version:
file_name:
sha256:
feature_dim:
preprocess_id:
checkpoint_source:
license:
compatible_detector_version:
compatible_crop_config:
compatible_index_schema:
```

权重不进入 Git。

### 12.5 Production 门禁

模型上线前必须满足：权重哈希通过、固定测试协议完成、评估报告存在、完成与当前 Production 的比较、重新标定阈值、Reviewer 发起上线、服务器门禁通过。模型切换后按新特征协议重建对应 Catalog Faiss 索引。

## 13. Catalog 与 Faiss

发布流程：

1. Batch 审核达到 `approved`。
2. 生成不可变 `staged` Catalog 快照。
3. 使用该快照构建 Faiss `IndexFlatIP`。
4. 校验 Catalog Membership digest、Observation 行顺序、模型版本、特征维度和索引 SHA-256。
5. 服务器原子切换 `active_catalog_pointer`。
6. 原活动版本变为 `retired`。
7. Batch 变为 `published`。

要求：

- staged Catalog 不能用于正式识别。
- 发布失败不影响当前 active Catalog。
- Reviewer 可发起激活或回滚，但不能直接修改活动指针。
- 可回滚到上一有效版本。
- 每个 Confirmed Individual 保留多个 Observation，并支持侧别、质量和时间元数据。
- Top-K 先按 Observation 检索，再按个体去重展示。
- Catalog 规模实际需要后才考虑 IVF/HNSW。

## 14. Web 页面与 API

### 14.1 页面

- 总览：批次阶段、任务积压、失败、Worker 在线状态和待审核事项。
- 批次：文件夹上传、导入向导、Manifest、Job/Attempt 和产物。
- 审核中心：普通簇、多目标、历史身份、合并/拆分/撤回。
- 个体目录：代表图、Observation、侧别、质量、时间线、别名和 Identity Event。
- 关系集合：`nn_relationship`、共现事件和疑似关系证据。
- 查询：单图和整批查询、Top-K、分数、支持帧、模型/Catalog/校准状态。
- 数据集与训练：Dataset、Split 校验、日志、Checkpoint 和恢复。
- 模型：评估对比、上线申请、阈值状态和模型回滚。
- Catalog：构建、验证、激活和回滚。
- Worker：设备能力、令牌、心跳、当前租约和任务历史。
- 系统：账号、角色、部署版本、审计和手动导出/恢复。

### 14.2 人用 API 模块

```text
/api/auth
/api/users
/api/uploads
/api/batches
/api/jobs
/api/artifacts
/api/candidates
/api/reviews
/api/individuals
/api/collections
/api/relationships
/api/datasets
/api/training-runs
/api/evaluations
/api/models
/api/catalog
/api/query
/api/media
```

查询结果必须返回状态、Top-K、分数、代表图、支持帧数、是否跨侧、质量、模型版本、Catalog Version、校准状态和人工审核状态。

前端不能伪造最终状态：任务完成、审核裁决、模型上线和目录切换均以服务器事件和版本号为准。实时通道只刷新进度；断线后使用 REST 恢复完整状态。

## 15. 权限与安全

角色：

- `admin`：用户、角色、设备令牌和系统配置；不能绕过审核门禁。
- `operator`：上传、补元数据、启动/重试任务、创建 Dataset 和训练任务。
- `reviewer`：独立审核，发起 Catalog/Model 激活、回滚和身份变更提案。
- `viewer`：只读授权数据。
- `worker`：设备主体，只能访问租约授权的数据和接口。

安全要求：

- 本地账号使用强密码哈希；浏览器使用 HttpOnly、Secure、SameSite Session Cookie。
- 所有写操作做 CSRF 防护和服务端角色检查。
- Worker 使用 Bearer 设备令牌，不使用普通用户 Cookie。
- PostgreSQL 不向局域网或 Tailscale 用户直接开放。
- `/media/*` 按用户或租约逐次授权。
- 登录、下载、读取 EXIF、审核、发布、回滚和令牌撤销记录不可修改审计事件。
- 默认不监听公网，不使用外部遥测，不自动上传数据。
- 设备丢失后管理员可立即撤销令牌。

## 16. 部署与三分钟自动更新

### 16.1 Docker Compose

服务器 Compose 包含 Caddy、API 和 PostgreSQL。GPU Worker 在成员电脑独立运行。健康端点：

- `/health`：只检查 API 进程是否存活。
- `/ready`：检查数据库、Migration、文件库、必要配置和活动 Catalog。

### 16.2 自动部署

管理员在 `/etc/whitewhale/deploy.env` 设置：

```text
DEPLOY_BRANCH=<管理员选择的分支>
```

systemd timer 每三分钟：

1. 使用部署锁防止并发。
2. `git fetch` 并比较远端 commit；无变化立即退出。
3. 在 `/srv/whitewhale/releases/<commit>/` 创建独立 release。
4. 使用锁定依赖和本地缓存构建前端与容器镜像。
5. 运行 Migration 预检、文档化的最小测试和 `/ready` 检查。
6. 全部通过后原子切换 `/srv/whitewhale/current` 并重启服务。
7. 失败时保留上一 release 和当前服务。
8. 页面显示分支、commit、部署时间和失败原因。
9. 只保留少量最近成功 release，避免无界磁盘增长。

自动部署只允许向后兼容的扩展式数据库 Migration。删除列、不可逆改写或会破坏旧代码兼容性的 Migration 必须阻止自动发布并提示管理员手动处理。

### 16.3 离线交付

```text
docker-images/
python-wheels/
frontend-build/
database-migrations/
models/
model-manifests/
configs/
scripts/
docs/
checksums.sha256
```

核心系统不能依赖 GitHub、Hugging Face、外部 CDN、云数据库或公网 API 才能运行。完全断网时使用局域网地址；Tailscale 不可用不影响本地运行。断网期间自动 Git 更新暂停，但当前 release 继续服务。

### 16.4 备份决定

首阶段不启用自动定时备份。必须实现手动一致性导出和恢复校验，导出同时记录 PostgreSQL 快照、文件库清单、Active Catalog、Model Manifest 和 Faiss 索引。自动 PostgreSQL + 文件库增量备份作为后续强化项，不阻塞第一版。

## 17. 现有仓库迁移策略

- 现有 `src/whitewhale/` 算法模块继续作为 Worker 计算内核，不重写已验证的检测、Embedding、训练和检索算法。
- 现有 `scripts/run_pipeline.py`、`launch_review.py`、`launch_query.py`、`train_reid.py` 等 CLI 在网页闭环稳定前保留为兼容入口和验证基线。
- 新增一次性导入器，把现有 Manifest、审核 CSV、Pilot Set、Embedding Meta、Artifact Manifest 和权重元数据导入 PostgreSQL/文件库。
- 导入时不根据字符串猜测跨批次身份；旧目录编号仍按批次隔离。
- 当前 r4 权重注册为初始 Model Version；当前 Gallery 注册为初始 Catalog Version；保留权重、裁剪、预处理、Checkpoint 哈希和阈值未校准状态。
- 旧 CSV/NPY 在核对摘要和行绑定后变为只读 Legacy Artifact，不直接删除。
- 新系统端到端通过验收后，再决定是否弃用旧 HTML 审核页和单图查询页。

## 18. 验收测试

至少覆盖：

1. 两台 Worker 并发时只有一台能有效领取同一 Job。
2. 租约过期后可安全重试，旧 Worker 迟到结果被拒绝。
3. 重复 `complete` 不生成重复 Artifact。
4. SHA-256、模型版本、特征维度或行绑定不一致时不能成功。
5. 原图未被修改，所有结果可追溯到 Image、Crop、Job、Worker、模型和配置。
6. 原 `nn relationship` 目录图片自动进入候选集合。
7. 散图池三目标图片生成三个 Crop 和一个 Cooccurrence Event。
8. 三人中至少两人确认多目标后成为 `confirmed_member`。
9. 多目标 Crop 可分别归属不同个体。
10. 普通 Candidate Cluster 可由一人完成纯度审核。
11. Candidate、Top-1 和 HDBSCAN 噪声不会被强制写入正式身份。
12. `New/New/Existing` 创建新 UUID 且标记 `possible_duplicate`。
13. `Existing A/Existing A/New` 不自动合并。
14. 疑似亲缘永远不会进入 confirmed kinship。
15. 身份合并/拆分未达到 3/3 时保持现状。
16. Reviewer 能发起 Catalog/Model 操作，但不能绕过服务器门禁。
17. staged Catalog 或 Faiss 构建失败不影响 active Catalog。
18. Catalog、Model 和代码 release 可以分别回滚。
19. 未授权用户不能读取 `/media/*`；Worker 不能访问数据库或非租约输入。
20. 服务重启后 Job、租约、审核和事件记录不丢失。
21. 同一 Sequence、Encounter、近重复图片和原图派生 Crop 不跨训练 Split。
22. Candidate Model 未完成评估、比较和阈值标定时不能成为 Production。
23. 完全断网、仅局域网环境可完成上传、推理、审核、查询和 Worker 任务。
24. 三分钟部署失败时旧 release 继续可用，数据目录不受分支切换影响。

## 19. Action items

- [x] **建立基础工程与数据边界**：新增 TypeScript Web、FastAPI API/领域层、Worker 包、Docker Compose、Caddy、Alembic 和宿主机目录初始化；保持现有算法模块与 CLI 可运行。
- [x] **实现 PostgreSQL 领域模型与迁移**：按采集、任务、候选、审核、身份、Catalog、集合/共现、训练/模型分组建表，加入 UUID、唯一性、不可变事件和活动指针约束。
- [x] **实现账号、角色、文件库与分片上传**：完成应用登录、设备令牌、媒体授权、32 MiB 分片续传、标准/任意目录导入和 Manifest 校验。
- [x] **实现分布式 Job/Worker 控制面**：完成注册、能力匹配、原子租约、心跳、重试、幂等完成、迟到结果拒绝、Artifact staging 与完整性验证。
- [x] **接入批内归档与两阶段审核**：封装现有检测、Crop、Embedding、HDBSCAN 和历史 Top-K；实现普通簇、多目标和跨时间身份审核政策。
- [x] **实现正式个体、共现、疑似关系与不可变 Catalog**：完成 Observation、Identity Event、`nn_relationship`、N 目标关系、Faiss 构建、原子激活和回滚。
- [x] **实现训练与模型生命周期**：完成 Dataset Version/Split 门禁、Detector/Re-ID Training Job、Checkpoint 恢复、评估比较、阈值标定和 Production Promotion。
- [ ] **迁移并核验现有产物**：导入当前 Manifest、审核 CSV、r4 权重、Gallery 和 Artifact provenance；保持阈值未校准警告，不猜测跨批身份。
- [ ] **完成部署、离线交付与安全更新**：提供 Compose、LAN/Tailscale 配置、systemd 三分钟分支部署、健康检查、release 回退、离线安装包和手动一致性导出/恢复。
- [ ] **执行分层验收**：运行数据库约束、Worker 并发/租约、审核票型、Catalog 原子切换、训练 Split、防越权、断网运行和部署失败回退测试，记录每阶段验收证据。

## 20. 实施里程碑

### M1：基础控制面

完成数据库、账号、文件库、Batch/Job、Worker 租约和分片上传。验收目标是：一个标准或普通目录可以可靠上传，任意空闲 4060 Worker 可以领取测试任务并回传通过校验的 Artifact。

### M2：归档闭环

完成检测、Crop、Embedding、Candidate Cluster、普通簇审核、历史匹配审核、Confirmed Individual、Catalog、Faiss 和查询。验收目标是：一个新批次能从上传走到可回滚的正式目录版本。

### M3：扩展语义

完成 `nn_relationship`、N 目标共现、疑似关系、身份合并/拆分/撤回及其审核。验收目标是：多目标和关系证据不污染正式身份语义。

### M4：训练闭环

完成 Dataset Version、Split 门禁、Detector/Re-ID 训练、Checkpoint、评估、模型比较、阈值标定和上线。验收目标是：候选模型只能在全部服务器门禁通过后成为 Production，并触发兼容 Catalog 索引重建。

### M5：交付强化

完成多 Worker 压力测试、局域网/Tailscale 双模式、离线包、三分钟自动部署、监控、手动导出和恢复演练。自动定时备份仍为后续可选增强，不纳入本阶段完成条件。

## 21. 完成标准

当 M1–M5 的对应验收测试全部有可复查证据，且系统能够在完全断网局域网中完成上传、分布式计算、审核、正式目录发布、查询、训练与模型上线时，第一版视为完成。任何未通过的审核、校验、兼容性或完整性门禁都必须保持候选、冲突或失败状态，不能通过人工改数据库绕过。
