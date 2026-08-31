#!/usr/bin/env python3
"""Release 镜像内执行的无外部状态最小契约测试。"""
from __future__ import annotations

import numpy as np
from alembic.config import Config
from alembic.script import ScriptDirectory

from whitewhale.platform.app import Readiness
from whitewhale.platform.catalogs import build_flat_ip_index
from whitewhale.platform.models import Base


def main() -> int:
    required_tables = {
        "users", "batches", "jobs", "artifacts", "review_tasks",
        "confirmed_individuals", "catalog_versions", "model_versions",
        "audit_events",
    }
    missing = required_tables - set(Base.metadata.tables)
    if missing:
        raise RuntimeError(f"领域模型缺表: {sorted(missing)}")
    readiness = Readiness(True, True, True, True)
    if not readiness.ready:
        raise RuntimeError("Readiness 聚合契约失败")
    index = build_flat_ip_index(np.eye(2, dtype=np.float32))
    if not index:
        raise RuntimeError("Faiss IndexFlatIP 构建失败")
    heads = ScriptDirectory.from_config(Config("/app/alembic.ini")).get_heads()
    if len(heads) != 1:
        raise RuntimeError(f"Migration 必须只有一个 head: {heads}")
    print(f"release smoke passed: migration={heads[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
