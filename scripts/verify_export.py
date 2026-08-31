#!/usr/bin/env python3
"""离线校验手动导出包的成员、路径和 SHA-256。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


REQUIRED = {
    "database.dump", "data.tar", "active-pointers.json",
    "export-manifest.json", "checksums.sha256",
}


def verify(bundle: Path) -> dict:
    with tarfile.open(bundle, "r:") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        if names != REQUIRED:
            raise ValueError(f"导出包成员不符: {sorted(names)}")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts \
                    or not member.isfile():
                raise ValueError(f"导出包包含不安全成员: {member.name}")
        checksums = archive.extractfile("checksums.sha256")
        assert checksums is not None
        expected = {}
        for line in checksums.read().decode("utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip(" *")] = digest
        for name in REQUIRED - {"checksums.sha256"}:
            source = archive.extractfile(name)
            assert source is not None
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected.get(name):
                raise ValueError(f"{name} SHA-256 不一致")
        nested_source = archive.extractfile("data.tar")
        assert nested_source is not None
        try:
            with tarfile.open(fileobj=nested_source, mode="r|") as nested:
                for member in nested:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts \
                            or not (member.isfile() or member.isdir()):
                        raise ValueError(
                            f"data.tar 包含不安全成员: {member.name}")
        except tarfile.TarError as exc:
            raise ValueError("data.tar 不是合法 tar") from exc
        manifest_source = archive.extractfile("export-manifest.json")
        pointers_source = archive.extractfile("active-pointers.json")
        assert manifest_source is not None and pointers_source is not None
        manifest = json.load(io.TextIOWrapper(manifest_source, "utf-8"))
        pointers = json.load(io.TextIOWrapper(pointers_source, "utf-8"))
        if not manifest.get("created_at") or not manifest.get("release_commit"):
            raise ValueError("export manifest 缺少版本信息")
        if "active_catalog_id" not in pointers \
                or "production_models" not in pointers:
            raise ValueError("active pointer 快照不完整")
        return {"manifest": manifest, "pointers": pointers}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
