# M3 扩展语义实现说明

M3 把多目标共现、疑似关系和身份更正接入正式控制面。所有算法结果仍是候选或证据，不能自动成为正式身份或已确认亲缘。

## 多目标与关系证据

- Worker 保留 YOLO NMS 后的全部检测框，一张 Image 可以产生 N 个独立 Crop。
- 服务器为同图多 Crop 创建一个 `cooccurrence_event` 和 N 个 `cooccurrence_member`。
- 多目标真实性固定由 3 名 reviewer 盲审；至少 2 人确认才成为 `confirmed_member`。
- 原始 `nn relationship` 目录和新检测到的多目标图片进入同一个系统 Collection；Collection 只表达研究候选，不表达亲缘结论。
- 只有共现事件已确认、每个 Crop 都有 active Observation、且至少涉及两个不同正式个体时，才生成所有无序个体对。
- 关系类型数据库约束只允许 `co_occurrence`、`repeated_association` 和 `suspected_kinship`；状态只允许 suspected、证据不足、争议或拒绝，不存在 confirmed kinship。

网页审核中心会按任务类型展示 Crop 或不可变更正方案；“关系证据”页面始终显示“不是亲缘结论”的提示。

## 身份更正

合并、拆分和照片撤回先创建带 SHA-256 摘要的不可变 Proposal，再建立固定 3 人审核任务。只有三票全部批准同一 Proposal 才能应用；任何分歧都记录为 disputed，当前身份保持不变。

- 合并：保留目标 UUID，把 Observation 当前投影移向目标；其他来源身份标记为 merged，并把旧展示名保存为 alias。
- 拆分：方案必须完整覆盖来源身份的全部 active Observation，至少分为两组；通过后创建新的 UUID，原 UUID 标记为 split。
- 撤回：Observation 标记为 withdrawn，原行、Crop、审核来源和历史事件都保留，不物理删除。
- 每个 Proposal 只允许一个终态 `identity_change_event`，重复应用是幂等读取。

身份更正后，Catalog 查询以 Observation 当前状态和当前 UUID 为准：withdrawn 行会被跳过，合并/拆分后的旧 Catalog 行会映射到当前 active 身份。新 Catalog 的 staging 门禁拒绝 withdrawn Observation 或非 active 身份。受更正影响的关系证据先降为 `evidence_insufficient`，只有仍满足完整共现条件时才重新投影为 suspected。

## 主要接口

- `GET /api/cooccurrences/{event_id}`
- `GET /api/relationships`
- `POST /api/cooccurrences`
- `POST /api/reviews/tasks/{task_id}/apply-multi-target`
- `POST /api/cooccurrences/{event_id}/project-relationships`
- `POST /api/identity-changes/merge`
- `POST /api/identity-changes/split`
- `POST /api/identity-changes/withdrawal`
- `GET /api/identity-changes/{proposal_id}`
- `POST /api/reviews/tasks/{task_id}/apply-identity-change`

浏览器写接口继续要求登录 Cookie、CSRF 和角色检查；Worker 不能访问这些人工裁决接口或 PostgreSQL。

## 验收证据

针对性测试覆盖：N-Crop 检测、归档 Artifact 行绑定、多目标审核、Collection 状态同步、关系无序对、关系撤回失效、合并 3/3、拆分完整覆盖、撤回不删除、重复应用幂等、API 登录保护、Catalog 当前身份重映射和 Alembic 无漂移。生产前仍应运行完整回归与前端生产构建。
