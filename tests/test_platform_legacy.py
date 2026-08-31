"""现有产物只读登记与未校准门禁。"""
from __future__ import annotations

import os
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image as PillowImage
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.legacy import (  # noqa: E402
    LegacyArtifactSpec,
    LegacyImportService,
)
from whitewhale.platform.models import (  # noqa: E402
    ActiveCatalogPointer,
    Base,
    CatalogVersion,
    ConfirmedIndividual,
    IndividualAlias,
    LegacyArtifact,
    ModelVersion,
    ProductionModelPointer,
)
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

    def test_imports_r4_as_provisional_model_and_active_catalog(self):
        source_root = Path(self.temp.name) / "raw-source"
        crop_root = Path(self.temp.name) / "crop-source"
        source_root.mkdir()
        crop_root.mkdir()
        rows = []
        source_hashes = {}
        for index in range(2):
            image_id = f"IMG_{index}"
            group = "05 （provider note）" if index == 1 else "05"
            relative = Path("20140806 01") / "80 and above" / group \
                / f"image-{index}.jpg"
            source = source_root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            PillowImage.new("RGB", (20, 20), (index * 20, 40, 80)).save(source)
            source_hashes[relative.as_posix()] = hashlib.sha256(
                source.read_bytes()).hexdigest()
            PillowImage.new("RGB", (10, 10), (index * 20, 40, 80)).save(
                crop_root / f"{image_id}.jpg")
            rows.append({
                "image_id": image_id,
                "relative_path": relative.as_posix(),
                "session_id": "20140806 01",
                "x": "2", "y": "3", "w": "10", "h": "10",
                "confirmed_identity": "20140806 01_5.0",
            })
        meta = Path(self.temp.name) / "gallery_meta.csv"
        import csv
        with meta.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        vectors = np.array([
            [1, 0, 0, 0], [0.9, 0.1, 0, 0],
        ], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        embeddings = Path(self.temp.name) / "gallery.npy"
        np.save(embeddings, vectors, allow_pickle=False)
        config = Path(self.temp.name) / "config.json"
        config.write_text(json.dumps({
            "model": "megadescriptor-metric-learning-r4",
            "crop": "yolo",
            "preprocess": "Resize256+CenterCrop224",
            "feat_dim": 4,
            "n": 2,
        }), encoding="utf-8")
        weights = Path(self.temp.name) / "best.pt"
        weights.write_bytes(b"r4-weights")

        service = LegacyImportService(self.sessions, self.storage)
        result = service.import_initial_reid_release(
            model_version="megadescriptor-metric-learning-r4",
            weights_path=weights,
            embeddings_path=embeddings,
            metadata_path=meta,
            config_path=config,
            raw_source_root=source_root,
            crop_source_root=crop_root,
        )
        with self.sessions() as db:
            model = db.get(ModelVersion, result.model_version_id)
            catalog = db.get(CatalogVersion, result.catalog_id)
            pointer = db.get(ProductionModelPointer, "metric-learning")
            active = db.get(ActiveCatalogPointer, 1)
            individuals = list(db.scalars(select(ConfirmedIndividual)))
            aliases = list(db.scalars(select(IndividualAlias)))
            self.assertEqual(model.status, "production")
            self.assertEqual(pointer.model_version_id, model.id)
            self.assertEqual(catalog.status, "active")
            self.assertEqual(catalog.calibration_status,
                             "provisional_unvalidated")
            self.assertEqual(active.catalog_id, catalog.id)
            self.assertEqual(catalog.row_count, 2)
            self.assertEqual(len(individuals), 1)
            self.assertIn("20140806 01_05",
                          {item.alias for item in aliases})
            self.assertIn("20140806 01_5.0",
                          {item.alias for item in aliases})
        self.assertEqual(result.copied_raw_files, 2)
        self.assertEqual({
            item["relative_path"]: hashlib.sha256(
                (source_root / item["relative_path"]).read_bytes()
            ).hexdigest()
            for item in rows
        }, source_hashes)


if __name__ == "__main__":
    unittest.main()
