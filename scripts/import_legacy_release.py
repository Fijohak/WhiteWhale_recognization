#!/usr/bin/env python3
"""把经管理员明确指定的旧 Re-ID 权重与 Gallery 导入正式数据模型。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.legacy import LegacyImportService  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "导入旧 Re-ID release。身份只按 metadata 中精确的 "
            "session/quality/numeric-group 解析，不进行跨批次猜测。"
        ))
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-source-root", required=True, type=Path)
    parser.add_argument("--crop-source-root", required=True, type=Path)
    args = parser.parse_args()

    engine = create_engine(args.database_url, pool_pre_ping=True)
    storage = StorageLayout(args.data_root)
    storage.initialize()
    try:
        result = LegacyImportService(
            sessionmaker(engine, expire_on_commit=False), storage,
        ).import_initial_reid_release(
            model_version=args.model_version,
            weights_path=args.weights,
            embeddings_path=args.embeddings,
            metadata_path=args.metadata,
            config_path=args.config,
            raw_source_root=args.raw_source_root,
            crop_source_root=args.crop_source_root,
        )
    finally:
        engine.dispose()
    print(json.dumps({
        "model_version_id": str(result.model_version_id),
        "catalog_id": str(result.catalog_id),
        "observation_count": result.observation_count,
        "individual_count": result.individual_count,
        "copied_raw_files": result.copied_raw_files,
        "copied_crop_files": result.copied_crop_files,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
