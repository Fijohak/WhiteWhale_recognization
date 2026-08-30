"""一次性 Worker 登记码与可撤销设备 Bearer Token。"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    WorkerDevice,
    WorkerRegistrationCode,
    WorkerToken,
)


class InvalidWorkerCredential(ValueError):
    pass


@dataclass(frozen=True)
class WorkerRegistration:
    name: str
    gpu_model: str
    vram_mb: int
    cuda_version: str
    worker_version: str
    capabilities: tuple[str, ...]
    model_versions: tuple[str, ...]
    capacity: int = 1


@dataclass(frozen=True)
class WorkerGrant:
    device_id: uuid.UUID
    device_token: str
    token_expires_at: datetime


@dataclass(frozen=True)
class WorkerPrincipal:
    device_id: uuid.UUID
    name: str
    capabilities: frozenset[str]
    model_versions: frozenset[str]


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


class WorkerAuthService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        registration_ttl: timedelta = timedelta(minutes=15),
        token_ttl: timedelta = timedelta(days=90),
    ) -> None:
        if registration_ttl <= timedelta(0) or token_ttl <= timedelta(0):
            raise ValueError("凭据有效期必须为正数")
        self._sessions = sessions
        self._registration_ttl = registration_ttl
        self._token_ttl = token_ttl

    def create_registration_code(self, created_by_user_id: uuid.UUID) -> str:
        code = secrets.token_urlsafe(32)
        with self._sessions.begin() as db:
            db.add(WorkerRegistrationCode(
                code_digest=_digest(code),
                created_by_user_id=created_by_user_id,
                expires_at=datetime.now(UTC) + self._registration_ttl,
            ))
        return code

    def register(
        self,
        code: str,
        registration: WorkerRegistration,
    ) -> WorkerGrant:
        self._validate_registration(registration)
        now = datetime.now(UTC)
        with self._sessions.begin() as db:
            registration_code = db.scalar(
                select(WorkerRegistrationCode)
                .where(WorkerRegistrationCode.code_digest == _digest(code))
                .with_for_update()
            )
            if registration_code is None \
                    or registration_code.used_at is not None \
                    or registration_code.expires_at <= now:
                raise InvalidWorkerCredential("Worker 登记码无效、已使用或已过期")

            device = WorkerDevice(
                name=registration.name.strip(),
                gpu_model=registration.gpu_model.strip(),
                vram_mb=registration.vram_mb,
                cuda_version=registration.cuda_version.strip(),
                worker_version=registration.worker_version.strip(),
                capabilities=sorted(set(registration.capabilities)),
                model_versions=sorted(set(registration.model_versions)),
                capacity=registration.capacity,
            )
            db.add(device)
            db.flush()
            token = secrets.token_urlsafe(48)
            token_expires_at = now + self._token_ttl
            db.add(WorkerToken(
                device_id=device.id,
                token_digest=_digest(token),
                expires_at=token_expires_at,
            ))
            registration_code.used_at = now
            registration_code.used_by_device_id = device.id
            return WorkerGrant(device.id, token, token_expires_at)

    def resolve_token(self, token: str) -> WorkerPrincipal:
        now = datetime.now(UTC)
        with self._sessions() as db:
            worker_token = db.scalar(select(WorkerToken).where(
                WorkerToken.token_digest == _digest(token),
                WorkerToken.revoked_at.is_(None),
                WorkerToken.expires_at > now,
            ))
            if worker_token is None:
                raise InvalidWorkerCredential("设备令牌无效、已撤销或已过期")
            device = db.get(WorkerDevice, worker_token.device_id)
            if device is None or not device.is_active:
                raise InvalidWorkerCredential("Worker 设备已停用")
            return WorkerPrincipal(
                device_id=device.id,
                name=device.name,
                capabilities=frozenset(device.capabilities),
                model_versions=frozenset(device.model_versions),
            )

    def revoke_device(self, device_id: uuid.UUID) -> None:
        now = datetime.now(UTC)
        with self._sessions.begin() as db:
            device = db.get(WorkerDevice, device_id, with_for_update=True)
            if device is None:
                raise InvalidWorkerCredential("Worker 设备不存在")
            device.is_active = False
            db.execute(
                update(WorkerToken)
                .where(
                    WorkerToken.device_id == device_id,
                    WorkerToken.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )

    @staticmethod
    def _validate_registration(registration: WorkerRegistration) -> None:
        if not registration.name.strip() or len(registration.name) > 128:
            raise ValueError("Worker 名称长度必须为 1–128")
        if not registration.gpu_model.strip():
            raise ValueError("必须上报 GPU 型号")
        if registration.vram_mb <= 0:
            raise ValueError("显存必须大于 0")
        if registration.capacity <= 0:
            raise ValueError("容量必须大于 0")
        if not registration.capabilities:
            raise ValueError("至少需要一种任务能力")
