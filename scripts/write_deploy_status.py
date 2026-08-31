#!/usr/bin/env python3
"""原子写入供系统页读取的部署结果；避免 shell 拼接 JSON。"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def write_status(
    output: Path,
    *,
    status: str,
    branch: str,
    commit: str,
    deployed_at: str,
    failure_reason: str | None,
) -> None:
    if status not in {"deployed", "failed"}:
        raise ValueError("status 必须是 deployed 或 failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump({
                "status": status,
                "branch": branch,
                "commit": commit,
                "deployed_at": deployed_at,
                "failure_reason": failure_reason,
            }, target, ensure_ascii=False, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--status", required=True, choices=["deployed", "failed"])
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--deployed-at", required=True)
    parser.add_argument("--failure-reason")
    args = parser.parse_args()
    write_status(
        args.output,
        status=args.status,
        branch=args.branch,
        commit=args.commit,
        deployed_at=args.deployed_at,
        failure_reason=args.failure_reason,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
