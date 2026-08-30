"""由环境变量装配可部署的协作平台运行时。"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import sessionmaker

from .app import PlatformServices, Readiness, create_app
from .artifacts import WorkerResultService
from .auth import AuthService
from .imports import BatchImportService
from .jobs import JobQueueService, LeaseService
from .media import MediaService
from .reviews import ReviewService
from .storage import StorageLayout
from .uploads import UploadService
from .worker_auth import WorkerAuthService


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHITEWHALE_",
        env_file=".env",
        extra="ignore",
    )

    database_url: str
    data_root: Path = Path("/srv/whitewhale/data")
    alembic_ini: Path = _PROJECT_ROOT / "alembic.ini"
    require_active_catalog: bool = False
    upload_chunk_size: int = Field(default=32 * 1024 * 1024, gt=0)
    lease_seconds: int = Field(default=300, gt=0)


class PlatformRuntime:
    def __init__(self, settings: PlatformSettings):
        self.settings = settings
        self.engine: Engine = create_engine(
            settings.database_url, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self.storage = StorageLayout(settings.data_root)
        self.storage.initialize()
        self.services = PlatformServices(
            auth=AuthService(self.sessions),
            uploads=UploadService(
                self.sessions,
                self.storage,
                chunk_size=settings.upload_chunk_size,
            ),
            imports=BatchImportService(self.sessions, self.storage),
            worker_auth=WorkerAuthService(self.sessions),
            jobs=JobQueueService(self.sessions),
            leases=LeaseService(
                self.sessions,
                lease_duration=timedelta(seconds=settings.lease_seconds),
            ),
            results=WorkerResultService(self.sessions, self.storage),
            media=MediaService(self.sessions, self.storage),
            reviews=ReviewService(self.sessions),
        )
        self.app = create_app(
            readiness_probe=self.readiness,
            services=self.services,
        )

    def readiness(self) -> Readiness:
        details: list[str] = []
        database = False
        migrations = False
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                database = True
                current_revision = MigrationContext.configure(
                    connection).get_current_revision()
            config = Config(str(self.settings.alembic_ini))
            expected_revision = ScriptDirectory.from_config(
                config).get_current_head()
            migrations = (
                current_revision is not None
                and current_revision == expected_revision
            )
            if not migrations:
                details.append(
                    f"migration={current_revision or 'none'}, "
                    f"expected={expected_revision}")
        except Exception as exc:
            details.append(f"database/migration: {exc}")

        storage = all(
            (self.storage.root / partition).is_dir()
            and os.access(self.storage.root / partition, os.R_OK | os.W_OK)
            for partition in self.storage.PARTITIONS
        )
        if not storage:
            details.append("文件库分区不存在或不可读写")

        active_catalog = not self.settings.require_active_catalog
        if not active_catalog:
            details.append("尚未配置 active Catalog")
        return Readiness(
            database=database,
            storage=storage,
            migrations=migrations,
            active_catalog=active_catalog,
            detail="; ".join(details),
        )


def build_runtime(settings: PlatformSettings) -> PlatformRuntime:
    return PlatformRuntime(settings)
