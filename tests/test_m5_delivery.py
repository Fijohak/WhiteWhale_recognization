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


class TestM5Delivery(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
