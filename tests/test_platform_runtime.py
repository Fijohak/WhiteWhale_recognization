"""生产运行时的数据库、迁移和文件库 readiness 契约。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.runtime import (  # noqa: E402
    PlatformSettings,
    build_runtime,
)
from whitewhale.platform.models import Base  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestPlatformRuntime(unittest.TestCase):
    def test_ready_requires_the_database_to_be_at_alembic_head(self):
        engine = create_engine(TEST_DATABASE_URL)
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()

        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = build_runtime(PlatformSettings(
                database_url=TEST_DATABASE_URL,
                data_root=Path(tmp_dir),
                alembic_ini=ROOT / "alembic.ini",
                require_active_catalog=False,
            ))
            before = runtime.readiness()
            self.assertTrue(before.database)
            self.assertTrue(before.storage)
            self.assertFalse(before.migrations)
            self.assertFalse(before.ready)

            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
            command.upgrade(config, "head")
            after = runtime.readiness()
            self.assertTrue(after.ready, after.detail)
            runtime.engine.dispose()

    def test_required_catalog_readiness_tracks_the_active_pointer(self):
        engine = create_engine(TEST_DATABASE_URL)
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        engine.dispose()
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime = build_runtime(PlatformSettings(
                database_url=TEST_DATABASE_URL,
                data_root=Path(tmp_dir),
                alembic_ini=ROOT / "alembic.ini",
                require_active_catalog=True,
            ))
            result = runtime.readiness()
            self.assertFalse(result.active_catalog)
            self.assertIn("active Catalog", result.detail)
            runtime.engine.dispose()


if __name__ == "__main__":
    unittest.main()
