# 协作平台 M1 运行手册

这一阶段已经把“Ubuntu 服务器是唯一事实源、成员 4060 Laptop 只做计算”落实为可运行的控制面。Worker 不需要 PostgreSQL，也不应复制正式数据库；它只通过 HTTPS 领取租约、读取本次任务输入并上传带哈希和版本信息的产物。

## 1. 服务器首次启动

Ubuntu 需要 Docker Engine、Docker Compose v2、Git 和 Tailscale（仅远程访问时需要）。在仓库根目录执行：

```bash
cp .env.example .env
# 编辑 .env，至少替换 POSTGRES_PASSWORD 和 WHITEWHALE_HTTPS_SITE

sudo install -d -o 10001 -g 10001 /srv/whitewhale/data
sudo install -d /srv/whitewhale/postgres
docker compose up -d --build
docker compose ps
```

PostgreSQL 只在 Compose 内部网络开放；浏览器和 Worker 都不能直连。API 容器启动前自动执行 Alembic Migration，失败时 API 不会进入健康状态。

检查：

```bash
curl -k https://localhost/health
curl -k https://localhost/ready
```

`/health` 只表示进程存活；`/ready` 同时检查 PostgreSQL、Migration、文件库分区和当前阶段要求的 Catalog 配置。

## 2. 初始化首个管理员

初始化接口只允许在数据库还没有用户时成功一次：

```bash
curl -k https://localhost/api/auth/bootstrap \
  -H 'Content-Type: application/json' \
  -d '{"username":"owner","password":"请替换为至少12位强密码"}'
```

之后在网页登录。浏览器会使用 `HttpOnly + Secure + SameSite=Strict` 会话 Cookie，写操作还必须携带 CSRF Token。Tailscale 身份不替代应用账号和角色。

## 3. 局域网与 Tailscale 地址

- 局域网：给 `.env` 中的 `WHITEWHALE_HTTPS_SITE` 配置一个组内可解析的主机名，并在成员电脑安装 Caddy 内部 CA；然后访问该 HTTPS 主机名。
- Tailscale：服务器执行 `sudo tailscale serve --bg http://127.0.0.1:8080`。成员访问 Tailscale 提供的 `https://<机器名>.<tailnet>.ts.net`。
- 完全断网时 Tailscale 可以不可用，局域网 HTTPS 和现有 Compose 服务仍可运行。

不要把 PostgreSQL 端口映射到局域网，也不要把 Vite 的 `5173` 或 FastAPI 的 `8000` 直接作为协作入口。正式入口统一经过 Caddy。

## 4. 文件夹上传与恢复

网页支持标准 iDolphin 目录和普通图片目录。上传过程是：

1. 浏览器逐文件流式计算 SHA-256，不一次性读入整个 20GB 批次。
2. 服务器建立不可变 Manifest，默认分片为 32 MiB。
3. 中断或刷新后，浏览器查询服务器已收到的分片，只补传缺失部分。
4. 服务端流式合并并复验整文件大小和 SHA-256。
5. 图片格式、相对路径和目录语义检查通过后，原子登记为 Batch/Image。

服务端拒绝绝对路径、Windows 盘符、任何 `..`、路径越界、清单同名冲突、未声明分片和内容不一致的重复分片。iDolphin 的 `05` 等数字目录只登记为批次内 Source Group，不会成为跨批次正式个体 ID。

## 5. 登记 4060 Worker

管理员先通过 `/api/workers/registration-codes` 生成 15 分钟有效的一次性登记码。成员电脑克隆同一代码版本并执行：

```bash
python scripts/run_worker.py register \
  --api 'https://服务器地址/' \
  --registration-code '管理员给的一次性登记码' \
  --token-file "$HOME/.config/whitewhale/worker.json" \
  --gpu-model 'RTX 4060 Laptop' \
  --vram-mb 8192 \
  --cuda-version '12.6' \
  --capabilities 'test_echo,detect,embedding' \
  --model-versions 'reid-r4'

python scripts/run_worker.py run \
  --api 'https://服务器地址/' \
  --token-file "$HOME/.config/whitewhale/worker.json"
```

若使用局域网内部 CA，额外传入 `--ca-file /path/to/caddy-root.crt`。设备令牌文件以 `0600` 创建；设备丢失后管理员应立即撤销。Worker 不读取 `.env` 中的数据库地址，也不保存正式目录状态。

M1 内置 `test_echo` 用来验证完整控制面。M2 已接入 `batch_archival` 的检测、Embedding 和聚类生产 Handler，详见 [platform_m2.md](platform_m2.md)。

## 6. 开发运行

复用现有 Python 环境启动 API：

```bash
export WHITEWHALE_DATABASE_URL='postgresql+psycopg://whitewhale:密码@127.0.0.1:5432/whitewhale'
export WHITEWHALE_DATA_ROOT="$PWD/.local-data"
PYTHONPATH=src python -m alembic upgrade head
PYTHONPATH=src uvicorn whitewhale.platform.main:app --reload
```

另一个终端启动 TypeScript 前端：

```bash
cd web
npm ci
npm run dev
```

Vite 只用于本机开发；协作服务器使用 `npm run build` 后由 Caddy 提供静态文件。

## 7. 当前验收证据

- 真实 PostgreSQL：并发 Worker 只能有一个领取同一 Job。
- 租约过期：任务可重新排队，旧租约 Token 被拒绝。
- 产物：大小、SHA-256、模型版本和行绑定不匹配时拒绝；重复完成不重复建 Artifact。
- 上传：路径越界、冲突分片、缺失分片和整文件哈希错误均不能导入。
- 安全：浏览器写操作要求登录与 CSRF；原图只能按 Image UUID 通过授权接口读取；Worker Token 可撤销。
- 部署：前端生产构建、依赖审计、Compose 配置、API Migration、Caddy HTTP/HTTPS 反向代理和 `/ready` 已做临时栈冒烟测试。
