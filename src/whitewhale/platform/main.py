"""Uvicorn 生产入口：whitewhale.platform.main:app。"""
from .runtime import PlatformSettings, build_runtime


runtime = build_runtime(PlatformSettings())
app = runtime.app
