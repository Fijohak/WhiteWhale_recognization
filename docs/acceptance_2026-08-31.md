# 平台第一版验收证据（2026-08-31）

## 自动化结果

- 后端完整回归：336 passed，85 subtests passed；仅保留 Starlette TestClient/httpx 的上游弃用警告。
- M5 分层入口：34 passed。
- React/TypeScript：`tsc -b && vite build` 通过，生产 JS 187.61 kB（gzip 59.11 kB）。
- npm 高危审计：0 vulnerabilities。
- PostgreSQL：从空 schema 连续升级到 `f22b7a9d5e04` 成功，开发库与空库 `alembic check` 均无漂移；审计表 UPDATE/DELETE 均被数据库触发器拒绝。
- 新增 M5 Migration 通过 expand-only 自动部署门禁。
- Compose 配置解析、所有部署 shell 的 `bash -n`、三分钟 systemd calendar 均通过。

覆盖的关键行为包括并发 `SKIP LOCKED` 租约、租约过期与迟到产物、Artifact 哈希/行绑定、32 MiB 续传、防路径越界、Job/Attempt/Artifact 详情、单图/整批查询、查询图片租约下载、一图多目标、紧凑 ZIP/NPY Embedding、固定 Catalog Top-K、固定审核票型、正式身份投影、N 目标共现、关系语义、身份合并/拆分/撤回、Catalog 原子切换、Dataset 防泄漏、test 下载隔离、Checkpoint 恢复、固定评估/生产比较/阈值详情、Production 门禁、兼容 Catalog 重建和受 Catalog 兼容门禁保护的模型回滚。

## 本机集成结果

- `whitewhale_dev` 数据库迁移到 `f22b7a9d5e04`。
- 开发文件库可读写；`GET /ready` 返回 database/storage/migrations/active_catalog 全部通过。
- API 运行于 `127.0.0.1:8000`，Vite 运行于 `127.0.0.1:5173`。
- 实际现有产物已登记 5 项：Dataset Manifest、r4 权重、r4 Gallery Embedding、Gallery Meta、Embedding Config。
- r4 与 Gallery 明确保留 `provisional_unvalidated`；实际摘要与大小已核验。
- r4 已投影为初始 Production Model，Gallery 已投影为 active Catalog：43 个按 session 隔离的 UUID 身份、202 条 Observation、768 维/202 行 Faiss；首条向量回查分数 1.0。旧训练成员不可恢复的事实被显式记录，Gallery 没有被冒充训练集。
- 一致性导出包完成成员与 SHA-256 校验，并成功恢复到隔离数据库 `whitewhale_restore_m5`：Alembic revision 为 `d20f4ea783b2`、Legacy Artifact 为 5 项、文件库恢复 5 个文件。
- 演练产生的临时 tar 与文件副本已移入桌面回收站，可恢复；隔离恢复数据库保留供管理员抽查。

`f22` 当前快照已重新完成 336 项全量回归、85 项子测试、34 项 M5 分层验收和前端生产构建。查询专项 12 项测试覆盖服务、HTTP、Worker 注册/租约下载和算法产物合同。浏览器自动化未执行：项目虚拟环境未安装 Playwright，未为验收临时修改依赖。最终交付前仍需完成约 3000 张真实批次的断网端到端验收。

## 尚需管理员完成的一次性外部动作

Tailscale 守护进程已在线，节点 DNS 为 `cancade.tail39defd.ts.net`。当前 tailnet 尚未启用 Serve；官方控制台要求账号本人登录后确认一次。启用后运行：

```bash
tailscale serve --bg --yes http://127.0.0.1:5173
tailscale serve status
```

随后应从另一台 tailnet 成员设备打开 `https://cancade.tail39defd.ts.net`，验证登录页和 `/ready`。此动作不影响 LAN/localhost，且不会启用公网 Funnel。
