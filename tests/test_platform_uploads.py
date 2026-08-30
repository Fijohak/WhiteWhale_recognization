"""PostgreSQL + 宿主机 staging 上的可续传上传契约。"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.models import Base, User  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import (  # noqa: E402
    UploadConflict,
    UploadFileSpec,
    UploadService,
)


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestResumableUploads(unittest.TestCase):
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

    def test_parts_are_resumable_and_complete_only_after_sha256_validation(self):
        payload = b"abcdef"
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            service = UploadService(self.sessions, layout, chunk_size=4)
            grant = service.create_session(
                owner_user_id=self.owner_id,
                batch_name="20140419 02",
                source_format="idolphin",
                files=[UploadFileSpec(
                    relative_path="70-79/05/a.jpg",
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )],
            )
            file_id = grant.files[0].file_id

            first_hash = hashlib.sha256(payload[:4]).hexdigest()
            second_hash = hashlib.sha256(payload[4:]).hexdigest()
            service.put_part(grant.session_id, file_id, 0,
                             payload[:4], first_hash)
            service.put_part(grant.session_id, file_id, 0,
                             payload[:4], first_hash)
            status = service.status(grant.session_id)
            self.assertEqual(status.state, "uploading")
            self.assertEqual(status.files[0].received_parts, (0,))
            self.assertEqual(status.files[0].missing_parts, (1,))
            service.put_part(grant.session_id, file_id, 1,
                             payload[4:], second_hash)

            assembled = service.complete_file(grant.session_id, file_id)
            self.assertEqual(assembled.read_bytes(), payload)
            self.assertEqual(service.complete_session(grant.session_id),
                             "complete")

    def test_rejects_unsafe_manifest_paths_and_conflicting_part_replays(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            service = UploadService(self.sessions, layout, chunk_size=4)

            with self.assertRaisesRegex(ValueError, "路径越界"):
                service.create_session(
                    owner_user_id=self.owner_id,
                    batch_name="unsafe",
                    source_format="generic",
                    files=[UploadFileSpec(
                        relative_path="../secret.jpg",
                        size_bytes=4,
                        sha256=hashlib.sha256(b"safe").hexdigest(),
                    )],
                )

            grant = service.create_session(
                owner_user_id=self.owner_id,
                batch_name="safe",
                source_format="generic",
                files=[UploadFileSpec(
                    relative_path="photo.jpg",
                    size_bytes=4,
                    sha256=hashlib.sha256(b"safe").hexdigest(),
                )],
            )
            file_id = grant.files[0].file_id
            service.put_part(
                grant.session_id, file_id, 0, b"safe",
                hashlib.sha256(b"safe").hexdigest())
            with self.assertRaisesRegex(UploadConflict, "分片冲突"):
                service.put_part(
                    grant.session_id, file_id, 0, b"evil",
                    hashlib.sha256(b"evil").hexdigest())


if __name__ == "__main__":
    unittest.main()
