"""已完成上传导入标准/普通目录的契约。"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from PIL import Image as PillowImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.imports import BatchImportService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Base,
    Batch,
    Collection,
    CollectionMembership,
    Image,
    SourceGroup,
    User,
)
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import UploadFileSpec, UploadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


def _jpeg_bytes() -> bytes:
    output = io.BytesIO()
    PillowImage.new("RGB", (4, 3), "white").save(output, format="JPEG")
    return output.getvalue()


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestBatchImports(unittest.TestCase):
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
        with self.sessions.begin() as session:
            owner = User(username="operator", password_hash="$argon2id$test")
            session.add(owner)
            session.flush()
            self.owner_id = owner.id

    def _completed_upload(self, layout, relative_path, source_format):
        payload = _jpeg_bytes()
        uploads = UploadService(self.sessions, layout, chunk_size=1024)
        grant = uploads.create_session(
            owner_user_id=self.owner_id,
            batch_name="20140419 02",
            source_format=source_format,
            files=[UploadFileSpec(
                relative_path=relative_path,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )],
        )
        file_id = grant.files[0].file_id
        uploads.put_part(
            grant.session_id, file_id, 0, payload,
            hashlib.sha256(payload).hexdigest())
        uploads.complete_file(grant.session_id, file_id)
        uploads.complete_session(grant.session_id)
        return grant.session_id, payload

    def test_idolphin_source_group_is_batch_scoped_not_a_global_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            upload_id, payload = self._completed_upload(
                layout, "70-79/05/a.jpg", "idolphin")

            batch_id = BatchImportService(
                self.sessions, layout).import_upload(upload_id)

            with self.sessions() as session:
                batch = session.get(Batch, batch_id)
                image = session.scalar(select(Image))
                group = session.get(SourceGroup, image.source_group_id)
                self.assertEqual(batch.source_format, "idolphin")
                self.assertEqual(group.name, "05")
                self.assertEqual(group.kind, "batch_local_individual")
                self.assertEqual(image.quality_band, "70-79")
                self.assertEqual(image.original_relative_path,
                                 "70-79/05/a.jpg")
                raw_path = layout.resolve("raw", image.source_path)
                self.assertEqual(raw_path.read_bytes(), payload)

            self.assertEqual(
                BatchImportService(self.sessions, layout)
                .import_upload(upload_id),
                batch_id,
            )

    def test_generic_import_requires_an_explicit_capture_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            upload_id, _ = self._completed_upload(
                layout, "camera/a.jpg", "generic")
            service = BatchImportService(self.sessions, layout)

            with self.assertRaisesRegex(ValueError, "拍摄日期"):
                service.import_upload(upload_id)
            batch_id = service.import_upload(
                upload_id, captured_on=date(2014, 4, 19))
            with self.sessions() as session:
                batch = session.get(Batch, batch_id)
                self.assertEqual(batch.metadata_json["captured_on"],
                                 "2014-04-19")

    def test_nn_relationship_is_added_to_candidate_collection_not_identity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            upload_id, _ = self._completed_upload(
                layout, "nn relationship/pair.jpg", "idolphin")
            batch_id = BatchImportService(
                self.sessions, layout).import_upload(upload_id)
            with self.sessions() as session:
                image = session.scalar(select(Image).where(
                    Image.batch_id == batch_id))
                collection = session.scalar(select(Collection).where(
                    Collection.system_key == "nn_relationship"))
                membership = session.scalar(select(CollectionMembership).where(
                    CollectionMembership.collection_id == collection.id,
                    CollectionMembership.image_id == image.id,
                ))
                self.assertIsNotNone(membership)
                self.assertEqual(membership.membership_status, "candidate")
                self.assertEqual(
                    membership.assignment_source, "original_folder")


if __name__ == "__main__":
    unittest.main()
