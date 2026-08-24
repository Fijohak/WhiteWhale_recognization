# 一键启动中华白海豚个体查询客户端（本地离线小工具）。
#
# 功能：后台启动 query_app 服务（最小化窗口）→ 轮询端口就绪 →
#       自动打开浏览器 http://127.0.0.1:{Port}。
# 用法：双击仓库根目录 start_query_app.bat，或：
#       powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_query_app.ps1
# 退出：关闭最小化的服务窗口即停止服务。

$ErrorActionPreference = "Stop"

# ---- 可调参数（端口与默认启动命令一致） ----
$Port = 8000

$root = Split-Path -Parent $PSScriptRoot   # 仓库根目录（相对定位，不硬编码路径）
Set-Location $root

# 1) python 检查
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "[错误] 未找到 python，请先安装 Python 并加入 PATH。" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

# 2) 后台启动服务（最小化窗口，日志留在该窗口）
Write-Host "[1/2] 启动查询服务：python scripts/launch_query.py --port $Port ..."
$proc = Start-Process python -ArgumentList "scripts/launch_query.py", "--port", "$Port" `
    -WorkingDirectory $root -WindowStyle Minimized -PassThru

# 3) 轮询端口就绪（最多 60 秒，模型首次加载需数秒）
$ok = $false
for ($i = 0; $i -lt 120 -and -not $proc.HasExited; $i++) {
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $Port)
        $c.Close()
        $ok = $true
        break
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if ($ok) {
    Write-Host "[2/2] 服务已就绪，正在打开浏览器 http://127.0.0.1:$Port ..." -ForegroundColor Green
    Start-Process "http://127.0.0.1:$Port"
    Write-Host "服务在最小化窗口运行（任务栏可恢复）。关闭该窗口即停止服务。"
} else {
    Write-Host "[错误] 服务未在 60 秒内就绪，请检查 python / 模型环境，并查看服务窗口日志。" -ForegroundColor Red
}
