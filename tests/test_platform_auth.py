"""真实 PostgreSQL 上的账号、角色和可撤销会话契约。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.auth import (  # noqa: E402
    AuthService,
    BootstrapClosed,
    InvalidCredentials,
    InvalidSession,
)
from whitewhale.platform.models import Base, Role, User  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestPlatformAuth(unittest.TestCase):
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

    def test_admin_bootstrap_is_one_time_and_never_stores_plaintext(self):
        service = AuthService(self.sessions)
        password = "correct horse battery staple"

        admin = service.bootstrap_admin("Owner", password)

        with self.sessions() as session:
            stored = session.get(User, admin.id)
            self.assertNotEqual(stored.password_hash, password)
            self.assertTrue(stored.password_hash.startswith("$argon2"))
            self.assertEqual(
                set(session.scalars(select(Role.name)).all()),
                {"admin", "operator", "reviewer", "viewer"},
            )

        with self.assertRaisesRegex(BootstrapClosed, "初始化已关闭"):
            service.bootstrap_admin("second", "another strong password")

    def test_login_resolves_roles_checks_csrf_and_revokes_session(self):
        service = AuthService(self.sessions)
        service.bootstrap_admin("owner", "correct horse battery staple")

        with self.assertRaises(InvalidCredentials):
            service.login("owner", "wrong password")

        login = service.login("OWNER", "correct horse battery staple")
        principal = service.resolve_session(login.session_token)
        self.assertEqual(principal.username, "owner")
        self.assertEqual(principal.roles, frozenset({"admin"}))
        service.verify_csrf(login.session_token, login.csrf_token)
        with self.assertRaisesRegex(InvalidSession, "CSRF"):
            service.verify_csrf(login.session_token, "wrong")

        service.logout(login.session_token)
        with self.assertRaises(InvalidSession):
            service.resolve_session(login.session_token)

    def test_admin_can_create_a_reviewer_account_with_server_validated_roles(self):
        service = AuthService(self.sessions)
        service.bootstrap_admin("owner", "correct horse battery staple")
        reviewer = service.create_user(
            "reviewer-a",
            "another correct horse battery",
            roles={"reviewer"},
        )
        login = service.login("reviewer-a", "another correct horse battery")
        principal = service.resolve_session(login.session_token)
        self.assertEqual(principal.user_id, reviewer.id)
        self.assertEqual(principal.roles, frozenset({"reviewer"}))
        with self.assertRaisesRegex(ValueError, "角色"):
            service.create_user(
                "invalid-role",
                "another correct horse battery",
                roles={"worker"},
            )


if __name__ == "__main__":
    unittest.main()
