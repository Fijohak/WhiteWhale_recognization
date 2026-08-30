"""一次性登记码、设备令牌和撤销契约。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.models import Base, WorkerToken  # noqa: E402
from whitewhale.platform.worker_auth import (  # noqa: E402
    InvalidWorkerCredential,
    WorkerAuthService,
    WorkerRegistration,
)


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestWorkerAuth(unittest.TestCase):
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
        self.admin = AuthService(self.sessions).bootstrap_admin(
            "owner", "correct horse battery staple")

    def test_registration_code_is_single_use_and_token_is_revocable(self):
        service = WorkerAuthService(self.sessions)
        code = service.create_registration_code(self.admin.id)
        registration = WorkerRegistration(
            name="laptop-4060-a",
            gpu_model="RTX 4060 Laptop",
            vram_mb=8192,
            cuda_version="12.6",
            worker_version="0.1.0",
            capabilities=("test_echo", "embedding"),
            model_versions=("test-model-v1",),
            capacity=1,
        )

        grant = service.register(code, registration)
        with self.assertRaisesRegex(InvalidWorkerCredential, "登记码"):
            service.register(code, registration)

        principal = service.resolve_token(grant.device_token)
        self.assertEqual(principal.device_id, grant.device_id)
        self.assertIn("test_echo", principal.capabilities)
        with self.sessions() as session:
            stored = session.scalar(select(WorkerToken))
            self.assertNotEqual(stored.token_digest, grant.device_token)

        service.revoke_device(grant.device_id)
        with self.assertRaises(InvalidWorkerCredential):
            service.resolve_token(grant.device_token)


if __name__ == "__main__":
    unittest.main()
