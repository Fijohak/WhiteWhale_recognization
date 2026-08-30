#!/usr/bin/env python3
"""注册或运行不持有数据库副本的 WhiteWhale GPU Worker。"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.worker.client import (  # noqa: E402
    HttpWorkerApi,
    WorkerRunner,
    test_echo_handler,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WhiteWhale GPU Worker")
    subcommands = parser.add_subparsers(dest="command", required=True)

    register = subcommands.add_parser("register")
    register.add_argument("--api", required=True)
    register.add_argument("--registration-code", required=True)
    register.add_argument("--token-file", type=Path, required=True)
    register.add_argument("--name", default=socket.gethostname())
    register.add_argument("--gpu-model", required=True)
    register.add_argument("--vram-mb", type=int, required=True)
    register.add_argument("--cuda-version", required=True)
    register.add_argument("--worker-version", default="0.1.0")
    register.add_argument("--capabilities", default="test_echo")
    register.add_argument("--model-versions", default="")
    register.add_argument("--ca-file")

    run = subcommands.add_parser("run")
    run.add_argument("--api", required=True)
    run.add_argument("--token-file", type=Path, required=True)
    run.add_argument("--ca-file")
    run.add_argument("--idle-seconds", type=float, default=5.0)
    return parser


def _write_secret(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "register":
        grant = HttpWorkerApi.register(args.api, {
            "registration_code": args.registration_code,
            "name": args.name,
            "gpu_model": args.gpu_model,
            "vram_mb": args.vram_mb,
            "cuda_version": args.cuda_version,
            "worker_version": args.worker_version,
            "capabilities": [
                item for item in args.capabilities.split(",") if item],
            "model_versions": [
                item for item in args.model_versions.split(",") if item],
            "capacity": 1,
        }, ca_file=args.ca_file)
        _write_secret(args.token_file, grant)
        print(f"Worker 已登记: {grant['device_id']}")
        return

    credentials = json.loads(args.token_file.read_text(encoding="utf-8"))
    api = HttpWorkerApi(
        args.api, credentials["device_token"], ca_file=args.ca_file)
    WorkerRunner(api, handlers={
        "test_echo": test_echo_handler,
    }).run_forever(idle_seconds=args.idle_seconds)


if __name__ == "__main__":
    main()
