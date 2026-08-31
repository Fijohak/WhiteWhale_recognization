# M5 部署、离线交付与恢复

M5 保持开发期“成员只管 push，服务器自动更新”的使用方式，同时把失败回退、离线运行和手动一致性导出做成可复查流程。PostgreSQL 始终只在 Compose 内网，浏览器与 Worker 不能直接连接数据库。

## 当前本机开发实例

- API：`http://127.0.0.1:8000`
- Vite：`http://127.0.0.1:5173`
- 开发数据库：容器 `whitewhale-platform-test-postgres` 内的 `whitewhale_dev`
- 文件库：`/home/cancade/.local/share/whitewhale-dev/data`
- `/ready` 已验证数据库、Migration 和文件库可用。

当前 Manifest、Pilot Set、r4 权重、r4 Gallery Embedding、Gallery Meta 与特征配置已登记到 `legacy_artifacts`。r4 同时作为带有迁移例外事件的初始 Production Model，202 条 Gallery 作为 active Catalog；二者始终保留 `provisional_unvalidated`，不会被描述为已经完成阈值标定。43 个旧身份只按 `session + 原数字组` 隔离投影为 UUID，旧浮点化字符串仅作为 alias，不猜测跨批身份。原始源文件不修改，平台文件库保存按 SHA-256 登记的只读副本。当前工作区没有最终审核 CSV 与身份迁移表，系统不会按文档描述虚构这些文件。

一次性导入使用 `scripts/import_legacy_release.py`。它会校验 Meta/Embedding/Config 行数、特征维度、有限值、原图/Crop 路径和摘要，再原子创建旧训练血缘、Model、Observation 与 Catalog。旧训练成员无法复原时会显式记录 `legacy_training_membership_unavailable`，绝不把 Gallery 冒充训练集。

## 三分钟分支部署

1. 将仓库放到 `/srv/whitewhale/repo`，复制 `deploy/deploy.env.example` 为 `/etc/whitewhale/deploy.env`，管理员只修改 `DEPLOY_BRANCH`。
2. `/etc/whitewhale/platform.env` 保存 PostgreSQL 密码、宿主机数据目录、LAN 站点等 Compose 参数，权限应为 `0600`，不得提交 Git。
3. 安装 `deploy/whitewhale-deploy.service` 与 `.timer`，执行 `systemctl enable --now whitewhale-deploy.timer`。
4. `systemctl list-timers whitewhale-deploy.timer` 应显示三分钟周期。

每次运行先取得 `flock`，fetch 所选远端分支并比较 commit。新 release 在独立目录构建，新增 Migration 先经过 `scripts/check_expand_only_migrations.py`；自动部署拒绝 drop/alter/raw destructive SQL。镜像、编译和 Migration 成功后才原子切换 `current`。新 `/ready` 连续失败时，脚本恢复旧软链接和旧 commit 镜像。保留最近五个 release，当前与上一个 release 不会被清理。

网页登录后显示当前分支、commit 和部署时间。`whitewhale-ready.timer` 每分钟检查一次 `/ready`，失败会进入 systemd 日志和 failed unit，可由系统现有告警工具采集。

部署脚本现在会运行镜像内的最小领域/Migration/Faiss 契约测试，而不是只做 `compileall`。成功或任一步失败都会原子写入 `data/working/deploy-status.json`；系统页显示状态和失败命令/Readiness 原因。自动部署的 Migration 检查只允许可静态审查的字面量安全 DDL，动态 SQL 与破坏性变更继续被拒绝。

## 管理面与审计

Web 已提供控制面总览、最近 Job/Attempt/Artifact 数量、批次 Manifest 与 Job 明细、Worker GPU/心跳/当前租约/历史次数、账号角色和审计事件。管理员可撤销设备令牌；服务器仍不向 Worker 暴露 PostgreSQL。

`audit_events` 记录登录成功/失败、退出、用户与 Worker 下载、EXIF 读取、审核票、Catalog 发布/激活/回滚、模型上线以及 Worker 登记和令牌撤销。PostgreSQL 触发器拒绝对该表执行 UPDATE 或 DELETE；开发库和空库 Migration 演练都已实际验证拒绝生效。

## LAN 与 Tailscale

LAN 使用 Caddy 内部证书站点；Tailscale 只给 tailnet 成员提供网络入口，应用登录仍然必需。当前安装版本验证过的命令是：

```bash
tailscale serve --bg --yes http://127.0.0.1:8080
tailscale serve status
```

首次使用必须由 tailnet 管理员在 Tailscale 官方控制台启用 Serve。不要使用 Funnel；Funnel 会发布到公网，不在本项目范围内。Tailscale 不可用时，LAN 地址和服务器本机仍可工作。

## 一致性导出与恢复演练

首阶段不设自动备份。管理员按需运行：

```bash
sudo -E /srv/whitewhale/current/deploy/manual-export.sh \
  /srv/whitewhale/exports/whitewhale-YYYYMMDD.tar
python /srv/whitewhale/current/scripts/verify_export.py \
  /srv/whitewhale/exports/whitewhale-YYYYMMDD.tar
```

导出期间 API 暂停写入，PostgreSQL custom dump、完整文件库、活动 Catalog/Production 指针、release commit 与 SHA-256 清单进入同一个原子生成的 tar。trap 会在失败时重新启动 API。

恢复先做隔离演练，不直接覆盖生产：

```bash
sudo -E /srv/whitewhale/current/deploy/restore-drill.sh \
  /srv/whitewhale/exports/whitewhale-YYYYMMDD.tar \
  /srv/whitewhale/restore-check/whitewhale-YYYYMMDD
```

脚本先校验所有摘要和 tar 路径，再恢复到 `whitewhale_restore_check` 数据库与一个原本不存在的文件目录。确认 Alembic revision、活动指针和抽样文件后，生产替换必须另行安排维护窗口，不能由自动部署执行。

## 离线包

`deploy/build-offline-bundle.sh /absolute/new-directory` 会准备锁定 release、平台 wheels、API/Web/PostgreSQL 镜像，以及 `frontend-build/`、`database-migrations/`、`models/`、`model-manifests/`、`configs/`、`scripts/`、`docs/` 和总 SHA-256 清单。通过 `WHITEWHALE_OFFLINE_MODELS_DIR` 指向生产模型目录；未设置时读取 `WHITEWHALE_DATA_ROOT/models`。镜像与 wheels 可能占用数 GB；运行前先估算目标盘空间，完成后保留一份经过校验的包即可。离线机先 `docker load`，再从本地 wheels 安装 Worker；运行时不依赖 GitHub、CDN、Hugging Face 或云数据库。

## 分层验收

```bash
WHITEWHALE_TEST_DATABASE_URL=postgresql+psycopg://... \
WHITEWHALE_PYTHON=/path/to/python \
scripts/run_platform_acceptance.sh
```

该入口覆盖 PostgreSQL 并发租约、迟到 Worker、Artifact 完整性、上传防越权、归档审核、Catalog 原子切换、身份更正、Split、Checkpoint、模型门禁和交付脚本。完整回归仍使用仓库 pytest 全量与 `web/npm run build`。
