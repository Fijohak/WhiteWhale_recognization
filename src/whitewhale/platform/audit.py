"""数据库持久化的只追加安全审计。"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .models import AuditEvent


class AuditService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def record(
        self,
        event_type: str,
        *,
        actor_type: str,
        actor_user_id: uuid.UUID | None = None,
        actor_worker_id: uuid.UUID | None = None,
        target_type: str | None = None,
        target_id: str | uuid.UUID | None = None,
        detail: dict | None = None,
    ) -> uuid.UUID:
        if actor_type not in {"user", "worker", "system", "anonymous"}:
            raise ValueError("审计 Actor 类型无效")
        if not event_type.strip() or len(event_type) > 128:
            raise ValueError("审计事件类型无效")
        if actor_user_id is not None and actor_worker_id is not None:
            raise ValueError("审计事件不能同时绑定用户与 Worker")
        with self._sessions.begin() as db:
            event = AuditEvent(
                actor_type=actor_type,
                actor_user_id=actor_user_id,
                actor_worker_id=actor_worker_id,
                event_type=event_type,
                target_type=target_type,
                target_id=str(target_id) if target_id is not None else None,
                detail=detail or {},
            )
            db.add(event)
            db.flush()
            return event.id

    def recent(self, *, limit: int = 100) -> list[dict]:
        if limit <= 0 or limit > 1000:
            raise ValueError("审计查询 limit 必须为 1–1000")
        with self._sessions() as db:
            events = list(db.scalars(
                select(AuditEvent)
                .order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc())
                .limit(limit)
            ))
            return [{
                "audit_event_id": str(event.id),
                "actor_type": event.actor_type,
                "actor_user_id": str(event.actor_user_id)
                if event.actor_user_id else None,
                "actor_worker_id": str(event.actor_worker_id)
                if event.actor_worker_id else None,
                "event_type": event.event_type,
                "target_type": event.target_type,
                "target_id": event.target_id,
                "detail": event.detail,
                "occurred_at": event.occurred_at.isoformat(),
            } for event in events]
