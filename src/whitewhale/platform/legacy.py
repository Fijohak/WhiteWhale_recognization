"""现有 Manifest、审核表、权重和 Gallery 的只读登记。"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .models import LegacyArtifact
from .storage import StorageLayout


@dataclass(frozen=True)
class LegacyArtifactSpec:
    artifact_kind: str
    source_path: Path
    calibration_status: str = "not_applicable"
    metadata: dict | None = None


class LegacyImportService:
    ALLOWED_KINDS = {
        "dataset_manifest", "review_csv", "model_weights",
        "gallery_embeddings", "gallery_metadata", "artifact_manifest",
    }

    def __init__(self, sessions: sessionmaker[Session], storage: StorageLayout):
        self._sessions = sessions
        self._storage = storage

    def inspect(self, spec: LegacyArtifactSpec) -> dict:
        source = spec.source_path.resolve()
        if spec.artifact_kind not in self.ALLOWED_KINDS:
            raise ValueError("Legacy Artifact 类型无效")
        if not source.is_file():
            raise ValueError(f"Legacy 文件不存在: {source}")
        if spec.calibration_status not in {
                "not_applicable", "provisional_unvalidated", "calibrated"}:
            raise ValueError("Legacy 校准状态无效")
        if spec.artifact_kind in {"model_weights", "gallery_embeddings"} \
                and spec.calibration_status == "calibrated" \
                and not (spec.metadata or {}).get("calibration_report_sha256"):
            raise ValueError("Legacy 模型/图库不能无报告声明已校准")
        return {
            "artifact_kind": spec.artifact_kind,
            "source_name": source.name,
            "source_path": str(source),
            "sha256": self._sha256(source),
            "size_bytes": source.stat().st_size,
            "calibration_status": spec.calibration_status,
            "metadata": spec.metadata or {},
        }

    def register(self, spec: LegacyArtifactSpec) -> uuid.UUID:
        inspected = self.inspect(spec)
        with self._sessions() as db:
            existing = db.scalar(select(LegacyArtifact).where(
                LegacyArtifact.artifact_kind == inspected["artifact_kind"],
                LegacyArtifact.sha256 == inspected["sha256"],
            ))
            if existing is not None:
                return existing.id
        artifact_id = uuid.uuid4()
        relative = Path("legacy") / str(artifact_id) / inspected["source_name"]
        target = self._storage.resolve("artifacts", relative)
        self._atomic_copy(Path(inspected["source_path"]), target)
        try:
            with self._sessions.begin() as db:
                db.add(LegacyArtifact(
                    id=artifact_id,
                    artifact_kind=inspected["artifact_kind"],
                    source_name=inspected["source_name"],
                    relative_path=relative.as_posix(),
                    sha256=inspected["sha256"],
                    size_bytes=inspected["size_bytes"],
                    calibration_status=inspected["calibration_status"],
                    metadata_json=inspected["metadata"],
                ))
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return artifact_id

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as value:
                shutil.copyfileobj(value, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
