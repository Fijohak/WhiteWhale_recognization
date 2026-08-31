"""现有产物只读登记与未校准门禁。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.legacy import (  # noqa: E402
    LegacyArtifactSpec,
    LegacyImportService,
)
from whitewhale.platform.models import Base, LegacyArtifact  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestLegacyImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
        cls.sessions = sessionmaker(cls.engine, expire_on_commit=False)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.temp = tempfile.TemporaryDirectory()
        self.storage = StorageLayout(Path(self.temp.name) / "data")
        self.storage.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def test_registers_immutable_copy_and_is_idempotent(self):
        source = Path(self.temp.name) / "best.pt"
        source.write_bytes(b"legacy-r4")
        service = LegacyImportService(self.sessions, self.storage)
        spec = LegacyArtifactSpec(
            artifact_kind="model_weights", source_path=source,
            calibration_status="provisional_unvalidated",
            metadata={"version": "r4"})
        first = service.register(spec)
        second = service.register(spec)
        self.assertEqual(first, second)
        with self.sessions() as db:
            record = db.get(LegacyArtifact, first)
            stored = self.storage.resolve("artifacts", record.relative_path)
            self.assertEqual(stored.read_bytes(), b"legacy-r4")
            self.assertEqual(record.calibration_status,
                             "provisional_unvalidated")
        source.write_bytes(b"changed-source")
        self.assertEqual(stored.read_bytes(), b"legacy-r4")

    def test_rejects_unproven_calibrated_gallery(self):
        source = Path(self.temp.name) / "gallery.npy"
        source.write_bytes(b"npy")
        service = LegacyImportService(self.sessions, self.storage)
        with self.assertRaisesRegex(ValueError, "无报告"):
            service.inspect(LegacyArtifactSpec(
                artifact_kind="gallery_embeddings", source_path=source,
                calibration_status="calibrated"))


if __name__ == "__main__":
    unittest.main()
