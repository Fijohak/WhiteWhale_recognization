"""应用用户认证与可撤销数据库会话。"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Role, User, UserRole, UserSession


class BootstrapClosed(RuntimeError):
    pass


class InvalidCredentials(ValueError):
    pass


class InvalidSession(ValueError):
    pass


@dataclass(frozen=True)
class LoginGrant:
    session_token: str
    csrf_token: str
    expires_at: datetime


@dataclass(frozen=True)
class Principal:
    session_id: uuid.UUID
    user_id: uuid.UUID
    username: str
    roles: frozenset[str]


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _username(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if not normalized or len(normalized) > 128:
        raise ValueError("用户名长度必须为 1–128 个字符")
    return normalized


class AuthService:
    ROLE_NAMES = ("admin", "operator", "reviewer", "viewer")

    def __init__(self, sessions: sessionmaker[Session], *,
                 session_duration: timedelta = timedelta(hours=12)):
        if session_duration <= timedelta(0):
            raise ValueError("session_duration 必须为正数")
        self._sessions = sessions
        self._session_duration = session_duration
        self._passwords = PasswordHasher()

    def bootstrap_admin(self, username: str, password: str) -> User:
        canonical = _username(username)
        if len(password) < 12:
            raise ValueError("密码至少需要 12 个字符")

        with self._sessions.begin() as session:
            session.execute(text("SELECT pg_advisory_xact_lock(91524001)"))
            if self._has_admin(session):
                raise BootstrapClosed("管理员初始化已关闭")

            roles = {
                role.name: role for role in session.scalars(
                    select(Role).where(Role.name.in_(self.ROLE_NAMES)))
            }
            missing = [
                Role(name=name) for name in self.ROLE_NAMES if name not in roles
            ]
            session.add_all(missing)
            session.flush()
            roles.update({role.name: role for role in missing})
            user = User(
                username=canonical,
                password_hash=self._passwords.hash(password),
            )
            session.add(user)
            session.flush()
            session.add(UserRole(user_id=user.id, role_id=roles["admin"].id))
            return user

    def bootstrap_open(self) -> bool:
        with self._sessions() as session:
            return not self._has_admin(session)

    def login(self, username: str, password: str) -> LoginGrant:
        canonical = _username(username)
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            user = session.scalar(
                select(User).where(
                    User.username == canonical,
                    User.is_active.is_(True),
                )
            )
            if user is None:
                raise InvalidCredentials("用户名或密码错误")
            try:
                self._passwords.verify(user.password_hash, password)
            except (VerifyMismatchError, InvalidHashError) as exc:
                raise InvalidCredentials("用户名或密码错误") from exc

            session_token = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
            expires_at = now + self._session_duration
            session.add(UserSession(
                user_id=user.id,
                token_digest=_digest(session_token),
                csrf_digest=_digest(csrf_token),
                expires_at=expires_at,
                last_seen_at=now,
            ))
            return LoginGrant(session_token, csrf_token, expires_at)

    def create_user(
        self,
        username: str,
        password: str,
        *,
        roles: set[str],
    ) -> User:
        canonical = _username(username)
        if len(password) < 12:
            raise ValueError("密码至少需要 12 个字符")
        if not roles or not roles.issubset(set(self.ROLE_NAMES)):
            raise ValueError("角色集合无效")
        with self._sessions.begin() as session:
            if session.scalar(select(User.id).where(
                    User.username == canonical)) is not None:
                raise ValueError("用户名已经存在")
            stored_roles = {
                role.name: role for role in session.scalars(
                    select(Role).where(Role.name.in_(roles)))
            }
            if set(stored_roles) != roles:
                raise ValueError("角色尚未初始化")
            user = User(
                username=canonical,
                password_hash=self._passwords.hash(password),
            )
            session.add(user)
            session.flush()
            session.add_all([
                UserRole(user_id=user.id, role_id=stored_roles[name].id)
                for name in sorted(roles)
            ])
            return user

    def resolve_session(self, session_token: str) -> Principal:
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            user_session = session.scalar(
                select(UserSession).where(
                    UserSession.token_digest == _digest(session_token),
                    UserSession.revoked_at.is_(None),
                    UserSession.expires_at > now,
                )
            )
            if user_session is None:
                raise InvalidSession("会话无效或已过期")
            user = session.get(User, user_session.user_id)
            if user is None or not user.is_active:
                raise InvalidSession("会话用户不可用")
            roles = frozenset(session.scalars(
                select(Role.name)
                .join(UserRole, UserRole.role_id == Role.id)
                .where(UserRole.user_id == user.id)
            ))
            user_session.last_seen_at = now
            return Principal(
                session_id=user_session.id,
                user_id=user.id,
                username=user.username,
                roles=roles,
            )

    def verify_csrf(self, session_token: str, csrf_token: str) -> None:
        now = datetime.now(UTC)
        with self._sessions() as session:
            user_session = session.scalar(
                select(UserSession).where(
                    UserSession.token_digest == _digest(session_token),
                    UserSession.revoked_at.is_(None),
                    UserSession.expires_at > now,
                )
            )
            if user_session is None or not hmac.compare_digest(
                    user_session.csrf_digest, _digest(csrf_token)):
                raise InvalidSession("CSRF 令牌无效")

    def logout(self, session_token: str) -> None:
        with self._sessions.begin() as session:
            user_session = session.scalar(
                select(UserSession)
                .where(UserSession.token_digest == _digest(session_token))
                .with_for_update()
            )
            if user_session is not None and user_session.revoked_at is None:
                user_session.revoked_at = datetime.now(UTC)

    @staticmethod
    def _has_admin(session: Session) -> bool:
        return session.scalar(
            select(UserRole.user_id)
            .join(Role, Role.id == UserRole.role_id)
            .where(Role.name == "admin")
            .limit(1)
        ) is not None
