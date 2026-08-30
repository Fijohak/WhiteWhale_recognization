"""宿主机文件库的固定分区和跨平台路径越界防护。"""
from __future__ import annotations

from pathlib import Path, PureWindowsPath


class StorageLayout:
    PARTITIONS = frozenset({
        "raw", "working", "artifacts", "models",
        "catalog_versions", "exports", "staging",
    })

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def initialize(self) -> None:
        root = self.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for name in sorted(self.PARTITIONS):
            (root / name).mkdir(exist_ok=True)

    def resolve(self, partition: str, relative_path: Path | str) -> Path:
        if partition not in self.PARTITIONS:
            raise ValueError(f"未知文件分区: {partition}")
        raw = str(relative_path)
        posix_path = Path(raw)
        windows_path = PureWindowsPath(raw)
        if posix_path.anchor or windows_path.drive or windows_path.root \
                or ".." in posix_path.parts or ".." in windows_path.parts:
            raise ValueError(f"路径越界: {raw} 不是安全相对路径")

        partition_root = (self.root.resolve() / partition).resolve()
        resolved = (partition_root / posix_path).resolve()
        try:
            resolved.relative_to(partition_root)
        except ValueError as exc:
            raise ValueError(f"路径越界: {raw} 超出 {partition} 分区") from exc
        return resolved
