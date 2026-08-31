#!/usr/bin/env python3
"""按显式清单登记现有产物；不会猜测跨批次身份。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.legacy import (  # noqa: E402
    LegacyArtifactSpec,
    LegacyImportService,
)
from whitewhale.platform.storage import StorageLayout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("inventory 必须是非空 JSON 数组")
    if not args.dry_run and (not args.database_url or not args.data_root):
        parser.error("正式登记需要 --database-url 和 --data-root")
    if args.dry_run:
        engine = create_engine("sqlite://")
        service = LegacyImportService(
            sessionmaker(engine), StorageLayout(args.data_root or Path("/tmp")))
    else:
        engine = create_engine(args.database_url, pool_pre_ping=True)
        storage = StorageLayout(args.data_root)
        storage.initialize()
        service = LegacyImportService(sessionmaker(engine), storage)
    results = []
    try:
        for item in inventory:
            spec = LegacyArtifactSpec(
                artifact_kind=item["artifact_kind"],
                source_path=Path(item["source_path"]),
                calibration_status=item.get(
                    "calibration_status", "not_applicable"),
                metadata=item.get("metadata", {}),
            )
            inspected = service.inspect(spec)
            inspected["legacy_artifact_id"] = (
                None if args.dry_run else str(service.register(spec)))
            results.append(inspected)
    finally:
        engine.dispose()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
