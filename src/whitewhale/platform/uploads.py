"""大文件清单上传、分片幂等写入与流式完整性校验。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .models import UploadFile, UploadPart, UploadSession
from .storage import StorageLayout


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class UploadConflict(RuntimeError):
    """上传状态或内容与已登记清单冲突。"""


@dataclass(frozen=True)
class UploadFileSpec:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class UploadFileGrant:
    file_id: uuid.UUID
    relative_path: str


@dataclass(frozen=True)
class UploadGrant:
    session_id: uuid.UUID
    chunk_size: int
    files: tuple[UploadFileGrant, ...]


@dataclass(frozen=True)
class UploadFileStatus:
    file_id: uuid.UUID
    relative_path: str
    state: str
    received_parts: tuple[int, ...]
    missing_parts: tuple[int, ...]


@dataclass(frozen=True)
class UploadStatus:
    session_id: uuid.UUID
    owner_user_id: uuid.UUID
    state: str
    chunk_size: int
    files: tuple[UploadFileStatus, ...]


class UploadService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        storage: StorageLayout,
        *,
        chunk_size: int = 32 * 1024 * 1024,
        session_ttl: timedelta = timedelta(days=7),
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        self._sessions = sessions
        self._storage = storage
        self._chunk_size = chunk_size
        self._session_ttl = session_ttl

    def create_session(
        self,
        *,
        owner_user_id: uuid.UUID,
        batch_name: str,
        source_format: str,
        files: Iterable[UploadFileSpec],
    ) -> UploadGrant:
        specs = tuple(files)
        normalized_paths: set[str] = set()
        manifest: list[dict[str, object]] = []
        for spec in specs:
            self._validate_spec(spec)
            self._storage.resolve("staging", spec.relative_path)
            collision_key = spec.relative_path.casefold()
            if collision_key in normalized_paths:
                raise ValueError(f"清单包含重复路径: {spec.relative_path}")
            normalized_paths.add(collision_key)
            manifest.append({
                "relative_path": spec.relative_path,
                "size_bytes": spec.size_bytes,
                "sha256": spec.sha256.lower(),
            })

        manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        now = datetime.now(UTC)
        upload = UploadSession(
            owner_user_id=owner_user_id,
            batch_name=batch_name,
            source_format=source_format,
            chunk_size=self._chunk_size,
            manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(),
            total_bytes=sum(spec.size_bytes for spec in specs),
            expires_at=now + self._session_ttl,
        )
        rows: list[UploadFile] = []
        with self._sessions.begin() as db:
            db.add(upload)
            db.flush()
            for spec in specs:
                row = UploadFile(
                    upload_session_id=upload.id,
                    relative_path=spec.relative_path,
                    size_bytes=spec.size_bytes,
                    sha256=spec.sha256.lower(),
                )
                db.add(row)
                rows.append(row)
            db.flush()

        return UploadGrant(
            session_id=upload.id,
            chunk_size=self._chunk_size,
            files=tuple(UploadFileGrant(row.id, row.relative_path) for row in rows),
        )

    def status(self, session_id: uuid.UUID) -> UploadStatus:
        with self._sessions() as db:
            upload = db.get(UploadSession, session_id)
            if upload is None:
                raise UploadConflict("上传会话不存在")
            files = list(db.scalars(
                select(UploadFile)
                .where(UploadFile.upload_session_id == session_id)
                .order_by(UploadFile.relative_path)
            ))
            statuses: list[UploadFileStatus] = []
            for upload_file in files:
                received = tuple(db.scalars(
                    select(UploadPart.part_number)
                    .where(UploadPart.upload_file_id == upload_file.id)
                    .order_by(UploadPart.part_number)
                ))
                expected_count = self._expected_part_count(
                    upload_file.size_bytes, upload.chunk_size)
                received_set = set(received)
                statuses.append(UploadFileStatus(
                    file_id=upload_file.id,
                    relative_path=upload_file.relative_path,
                    state=upload_file.state,
                    received_parts=received,
                    missing_parts=tuple(
                        part_number for part_number in range(expected_count)
                        if part_number not in received_set
                    ),
                ))
            return UploadStatus(
                session_id=upload.id,
                owner_user_id=upload.owner_user_id,
                state=upload.state,
                chunk_size=upload.chunk_size,
                files=tuple(statuses),
            )

    def put_part(
        self,
        session_id: uuid.UUID,
        file_id: uuid.UUID,
        part_number: int,
        data: bytes,
        sha256: str,
    ) -> Path:
        normalized_hash = self._validate_hash(sha256)
        computed_hash = hashlib.sha256(data).hexdigest()
        if computed_hash != normalized_hash:
            raise UploadConflict("分片 SHA-256 校验失败")

        with self._sessions.begin() as db:
            upload_file = db.scalar(
                select(UploadFile)
                .where(
                    UploadFile.id == file_id,
                    UploadFile.upload_session_id == session_id,
                )
                .with_for_update()
            )
            if upload_file is None:
                raise UploadConflict("上传文件不属于当前会话")
            upload = db.get(UploadSession, session_id)
            if upload is None or upload.state != "uploading":
                raise UploadConflict("上传会话当前不可写入")
            if upload.expires_at <= datetime.now(UTC):
                raise UploadConflict("上传会话已过期")

            expected_parts = self._expected_part_count(
                upload_file.size_bytes, upload.chunk_size)
            if part_number < 0 or part_number >= expected_parts:
                raise UploadConflict(f"非法分片编号: {part_number}")
            expected_size = min(
                upload.chunk_size,
                upload_file.size_bytes - part_number * upload.chunk_size,
            )
            if len(data) != expected_size:
                raise UploadConflict(
                    f"分片大小错误: 期望 {expected_size}，实际 {len(data)}")

            existing = db.scalar(select(UploadPart).where(
                UploadPart.upload_file_id == file_id,
                UploadPart.part_number == part_number,
            ))
            if existing is not None:
                if (existing.sha256 != normalized_hash
                        or existing.size_bytes != len(data)):
                    raise UploadConflict("分片冲突：相同编号已有不同内容")
                return self._storage.resolve("staging", existing.staging_path)

            relative = Path("uploads") / str(session_id) / str(file_id) / "parts" / f"{part_number:08d}.part"
            target = self._storage.resolve("staging", relative)
            self._atomic_write(target, data)
            db.add(UploadPart(
                upload_file_id=file_id,
                part_number=part_number,
                size_bytes=len(data),
                sha256=normalized_hash,
                staging_path=relative.as_posix(),
            ))
            upload_file.state = "uploading"
            return target

    def complete_file(
        self,
        session_id: uuid.UUID,
        file_id: uuid.UUID,
    ) -> Path:
        with self._sessions.begin() as db:
            upload_file = db.scalar(
                select(UploadFile)
                .where(
                    UploadFile.id == file_id,
                    UploadFile.upload_session_id == session_id,
                )
                .with_for_update()
            )
            if upload_file is None:
                raise UploadConflict("上传文件不属于当前会话")
            upload = db.get(UploadSession, session_id)
            if upload is None or upload.state != "uploading":
                raise UploadConflict("上传会话当前不可完成")
            if upload_file.state == "complete" and upload_file.completed_path:
                return self._storage.resolve("staging", upload_file.completed_path)

            expected_count = self._expected_part_count(
                upload_file.size_bytes, upload.chunk_size)
            parts = list(db.scalars(
                select(UploadPart)
                .where(UploadPart.upload_file_id == file_id)
                .order_by(UploadPart.part_number)
            ))
            if [part.part_number for part in parts] != list(range(expected_count)):
                raise UploadConflict("文件分片不完整，不能合并")

            expected_size = upload_file.size_bytes
            expected_hash = upload_file.sha256
            source_paths = [
                self._storage.resolve("staging", part.staging_path)
                for part in parts
            ]
            relative = Path("uploads") / str(session_id) / str(file_id) / "assembled.bin"
            target = self._storage.resolve("staging", relative)
            self._assemble_and_verify(
                source_paths, target,
                expected_size=expected_size,
                expected_hash=expected_hash,
            )
            upload_file.state = "complete"
            upload_file.completed_path = relative.as_posix()
            upload_file.completed_at = datetime.now(UTC)
            return target

    def complete_session(self, session_id: uuid.UUID) -> str:
        with self._sessions.begin() as db:
            upload = db.get(UploadSession, session_id, with_for_update=True)
            if upload is None:
                raise UploadConflict("上传会话不存在")
            if upload.state == "complete":
                return upload.state
            incomplete = db.scalar(
                select(UploadFile.id)
                .where(
                    UploadFile.upload_session_id == session_id,
                    UploadFile.state != "complete",
                )
                .limit(1)
            )
            if incomplete is not None:
                raise UploadConflict("仍有文件未完成，不能结束上传会话")
            upload.state = "complete"
            upload.completed_at = datetime.now(UTC)
            return upload.state

    @staticmethod
    def _validate_hash(value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("SHA-256 必须是 64 位十六进制字符串")
        return value.lower()

    def _validate_spec(self, spec: UploadFileSpec) -> None:
        if not spec.relative_path:
            raise ValueError("清单文件路径不能为空")
        if spec.size_bytes < 0:
            raise ValueError("文件大小不能为负数")
        self._validate_hash(spec.sha256)

    @staticmethod
    def _expected_part_count(size_bytes: int, chunk_size: int) -> int:
        return (size_bytes + chunk_size - 1) // chunk_size

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".part-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_name, target)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _assemble_and_verify(
        source_paths: list[Path],
        target: Path,
        *,
        expected_size: int,
        expected_hash: str,
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".assembled-", dir=target.parent)
        digest = hashlib.sha256()
        total_size = 0
        try:
            with os.fdopen(fd, "wb") as output:
                for source_path in source_paths:
                    with source_path.open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                            total_size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total_size != expected_size or digest.hexdigest() != expected_hash:
                raise UploadConflict("整文件 SHA-256 或大小校验失败")
            os.replace(temp_name, target)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
