"""M5 部署、扩展式迁移与一致性导出包门禁。"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MIGRATIONS = _load(
    "check_expand_only_migrations",
    ROOT / "scripts" / "check_expand_only_migrations.py")
EXPORT = _load("verify_export", ROOT / "scripts" / "verify_export.py")
DEPLOY_STATUS = _load(
    "write_deploy_status", ROOT / "scripts" / "write_deploy_status.py")


class TestM5Delivery(unittest.TestCase):
    def test_deployment_status_is_written_atomically_with_failure_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "working" / "deploy-status.json"
            DEPLOY_STATUS.write_status(
                target, status="failed", branch="dev", commit="a" * 40,
                deployed_at="2026-08-31T00:00:00Z",
                failure_reason="readiness failed")
            value = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["failure_reason"], "readiness failed")
            self.assertFalse(any(target.parent.glob("*.tmp")))

    def test_expand_only_checker_rejects_destructive_alembic(self):
        with tempfile.TemporaryDirectory() as temp:
            safe = Path(temp) / "safe.py"
            safe.write_text(
                "from alembic import op\n"
                "def upgrade():\n    op.add_column('x', None)\n",
                encoding="utf-8")
            unsafe = Path(temp) / "unsafe.py"
            unsafe.write_text(
                "from alembic import op\n"
                "def upgrade():\n    op.drop_table('users')\n",
                encoding="utf-8")
            self.assertEqual(MIGRATIONS.validate(safe), [])
            self.assertIn("drop_table", MIGRATIONS.validate(unsafe)[0])

    def test_expand_only_checker_allows_append_only_trigger_not_raw_update(self):
        with tempfile.TemporaryDirectory() as temp:
            trigger = Path(temp) / "trigger.py"
            trigger.write_text(
                "from alembic import op\n"
                "def upgrade():\n"
                "    op.execute('CREATE FUNCTION guard() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION ''no''; END; $$')\n"
                "    op.execute('CREATE TRIGGER guard BEFORE UPDATE OR DELETE "
                "ON audit_events FOR EACH ROW EXECUTE FUNCTION guard()')\n",
                encoding="utf-8")
            raw_update = Path(temp) / "raw_update.py"
            raw_update.write_text(
                "from alembic import op\n"
                "def upgrade():\n    op.execute('UPDATE users SET is_active=false')\n",
                encoding="utf-8")
            dynamic = Path(temp) / "dynamic.py"
            dynamic.write_text(
                "from alembic import op\n"
                "def upgrade():\n    sql='SELECT 1'\n    op.execute(sql)\n",
                encoding="utf-8")
            self.assertEqual(MIGRATIONS.validate(trigger), [])
            self.assertTrue(any("破坏性 SQL" in item
                                for item in MIGRATIONS.validate(raw_update)))
            self.assertTrue(any("字面量 SQL" in item
                                for item in MIGRATIONS.validate(dynamic)))

    def test_export_verifier_checks_every_payload_digest(self):
        nested = io.BytesIO()
        with tarfile.open(fileobj=nested, mode="w") as data_archive:
            content = b"legacy"
            info = tarfile.TarInfo("artifacts/legacy.bin")
            info.size = len(content)
            data_archive.addfile(info, io.BytesIO(content))
        payloads = {
            "database.dump": b"postgres",
            "data.tar": nested.getvalue(),
            "active-pointers.json": json.dumps({
                "active_catalog_id": None,
                "production_models": {},
            }).encode(),
            "export-manifest.json": json.dumps({
                "created_at": "2026-08-31T00:00:00Z",
                "release_commit": "a" * 40,
            }).encode(),
        }
        checksums = "".join(
            f"{hashlib.sha256(value).hexdigest()}  {name}\n"
            for name, value in payloads.items()).encode()
        payloads["checksums.sha256"] = checksums
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "export.tar"
            with tarfile.open(bundle, "w") as archive:
                for name, value in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
            result = EXPORT.verify(bundle)
            self.assertEqual(
                result["manifest"]["release_commit"], "a" * 40)

    def test_delivery_shell_and_three_minute_timer_are_valid(self):
        scripts = [
            ROOT / "deploy" / "deploy-release.sh",
            ROOT / "deploy" / "manual-export.sh",
            ROOT / "deploy" / "restore-drill.sh",
            ROOT / "deploy" / "build-offline-bundle.sh",
            ROOT / "scripts" / "run_platform_acceptance.sh",
        ]
        subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
        timer = (ROOT / "deploy" / "whitewhale-deploy.timer").read_text(
            encoding="utf-8")
        self.assertIn("OnCalendar=*-*-* *:0/3:00", timer)
        deploy = scripts[0].read_text(encoding="utf-8")
        self.assertIn("flock -n", deploy)
        self.assertIn("previous release restored", deploy)
        self.assertIn("release_smoke.py", deploy)
        self.assertIn("write_deploy_status failed", deploy)
        bundle = scripts[3].read_text(encoding="utf-8")
        for required in (
            "frontend-build", "database-migrations", "models",
            "model-manifests", "configs", "scripts", "docs",
        ):
            self.assertIn(required, bundle)


if __name__ == "__main__":
    unittest.main()
