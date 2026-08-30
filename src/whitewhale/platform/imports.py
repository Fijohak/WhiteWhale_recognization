"""把完成上传转成不可变 Batch、Image 和批内来源记录。"""
from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image as PillowImage
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .models import Batch, Image, SourceGroup, UploadFile, UploadSession
from .storage import StorageLayout


_IMAGE_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
})
_QUALITY_BANDS = frozenset({
    "below 50", "50-59", "60-69", "70-79", "80 and above",
})


@dataclass(frozen=True)
class _SourceSemantics:
    name: str
    kind: str
    quality_band: str | None = None
    relation_note: str | None = None


class BatchImportService:
    def __init__(self, sessions: sessionmaker[Session], storage: StorageLayout):
        self._sessions = sessions
        self._storage = storage

    def import_upload(
        self,
        upload_session_id: uuid.UUID,
        *,
        captured_on: date | None = None,
    ) -> uuid.UUID:
        with self._sessions() as db:
            existing = db.scalar(select(Batch.id).where(
                Batch.upload_session_id == upload_session_id))
            if existing is not None:
                return existing
            upload = db.get(UploadSession, upload_session_id)
            if upload is None or upload.state != "complete":
                raise ValueError("上传会话尚未完成")
            if upload.source_format == "generic" and captured_on is None:
                raise ValueError("普通目录必须明确填写拍摄日期")
            files = list(db.scalars(
                select(UploadFile)
                .where(UploadFile.upload_session_id == upload_session_id)
                .order_by(UploadFile.relative_path)
            ))
            file_records = [
                self._inspect_uploaded_file(upload_file)
                for upload_file in files
                if Path(upload_file.relative_path).suffix.lower()
                in _IMAGE_SUFFIXES
            ]
            if not file_records:
                raise ValueError("上传清单中没有受支持的图片")
            upload_details = (
                upload.batch_name,
                upload.source_format,
                upload.manifest_digest,
            )

        batch_id = uuid.uuid4()
        moved: list[tuple[Path, Path]] = []
        try:
            with self._sessions.begin() as db:
                existing = db.scalar(select(Batch.id).where(
                    Batch.upload_session_id == upload_session_id).with_for_update())
                if existing is not None:
                    return existing
                batch = Batch(
                    id=batch_id,
                    upload_session_id=upload_session_id,
                    name=upload_details[0],
                    source_format=upload_details[1],
                    manifest_sha256=upload_details[2],
                    metadata_json={
                        "captured_on": captured_on.isoformat()
                        if captured_on else self._date_from_batch_name(
                            upload_details[0]),
                    },
                )
                db.add(batch)
                groups: dict[tuple[str, str], SourceGroup] = {}
                for upload_file, staged_path, actual_hash, exif in file_records:
                    semantics = self._classify(
                        upload_details[1], upload_file.relative_path)
                    group_key = (semantics.kind, semantics.name)
                    group = groups.get(group_key)
                    if group is None:
                        group = SourceGroup(
                            batch_id=batch_id,
                            name=semantics.name,
                            kind=semantics.kind,
                            metadata_json={},
                        )
                        db.add(group)
                        db.flush()
                        groups[group_key] = group

                    raw_relative = (
                        Path(str(batch_id)) / upload_file.relative_path
                    )
                    target = self._storage.resolve("raw", raw_relative)
                    if target.exists():
                        raise ValueError(
                            f"原图目标路径已经存在: {upload_file.relative_path}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged_path, target)
                    moved.append((target, staged_path))
                    db.add(Image(
                        batch_id=batch_id,
                        source_group_id=group.id,
                        source_path=raw_relative.as_posix(),
                        original_relative_path=upload_file.relative_path,
                        source_sha256=actual_hash,
                        size_bytes=upload_file.size_bytes,
                        quality_band=semantics.quality_band,
                        relation_note=semantics.relation_note,
                        exif_json=exif,
                    ))
            return batch_id
        except BaseException:
            for target, staged_path in reversed(moved):
                if target.exists() and not staged_path.exists():
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, staged_path)
            raise

    def _inspect_uploaded_file(
        self,
        upload_file: UploadFile,
    ) -> tuple[UploadFile, Path, str, dict[str, object]]:
        if upload_file.state != "complete" or not upload_file.completed_path:
            raise ValueError(f"文件尚未完成: {upload_file.relative_path}")
        staged_path = self._storage.resolve(
            "staging", upload_file.completed_path)
        digest = hashlib.sha256()
        size = 0
        with staged_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        actual_hash = digest.hexdigest()
        if size != upload_file.size_bytes or actual_hash != upload_file.sha256:
            raise ValueError(f"文件完整性校验失败: {upload_file.relative_path}")
        try:
            with PillowImage.open(staged_path) as image:
                image.verify()
            with PillowImage.open(staged_path) as image:
                exif = {
                    "format": image.format,
                    "width": image.width,
                    "height": image.height,
                }
        except Exception as exc:
            raise ValueError(f"图片格式校验失败: {upload_file.relative_path}") from exc
        return upload_file, staged_path, actual_hash, exif

    @staticmethod
    def _classify(source_format: str, relative_path: str) -> _SourceSemantics:
        parts = Path(relative_path).parts
        lowered = tuple(part.casefold() for part in parts)
        if source_format != "idolphin":
            parent = parts[-2] if len(parts) > 1 else "root"
            return _SourceSemantics(parent, "generic_folder")

        if "nn relationship" in lowered:
            return _SourceSemantics(
                "nn_relationship", "relationship_candidate",
                relation_note="nn_relationship",
            )
        quality_band = next(
            (part for part in lowered if part in _QUALITY_BANDS), None)
        quality_index = lowered.index(quality_band) if quality_band else -1
        if quality_index >= 0 and len(parts) > quality_index + 2:
            group_name = parts[quality_index + 1]
            if group_name.isdigit():
                return _SourceSemantics(
                    group_name,
                    "batch_local_individual",
                    quality_band=quality_band,
                )
        if quality_band:
            return _SourceSemantics(
                "unresolved_pool", "unresolved_pool",
                quality_band=quality_band,
            )
        parent = parts[-2] if len(parts) > 1 else "root"
        return _SourceSemantics(parent, "idolphin_other")

    @staticmethod
    def _date_from_batch_name(batch_name: str) -> str | None:
        prefix = batch_name.strip().split(maxsplit=1)[0]
        if len(prefix) != 8 or not prefix.isdigit():
            return None
        try:
            return date(
                int(prefix[:4]), int(prefix[4:6]), int(prefix[6:8])
            ).isoformat()
        except ValueError:
            return None
