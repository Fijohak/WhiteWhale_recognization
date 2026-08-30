"""GPU Worker 进程只通过租约 API 执行任务。"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.worker.client import (  # noqa: E402
    ArtifactOutput,
    TaskLease,
    WorkerRunner,
)


class _FakeApi:
    def __init__(self):
        self.calls = []
        self.leases = [TaskLease(
            job_id="job-1",
            lease_token="lease-secret",
            task_type="test_echo",
            input_manifest={"items": ["image-1"]},
            required_model_version="test-model-v1",
        )]

    def lease(self):
        self.calls.append(("lease",))
        return self.leases.pop(0) if self.leases else None

    def start(self, lease):
        self.calls.append(("start", lease.job_id))

    def submit(self, lease, artifact):
        self.calls.append(("submit", lease.job_id, artifact.data))
        return "artifact-1"

    def complete(self, lease):
        self.calls.append(("complete", lease.job_id))

    def heartbeat(self, lease):
        self.calls.append(("heartbeat", lease.job_id))

    def fail(self, lease, detail):
        self.calls.append(("fail", lease.job_id, detail))


class TestWorkerRunner(unittest.TestCase):
    def test_run_once_uses_the_leased_manifest_and_completes(self):
        api = _FakeApi()
        runner = WorkerRunner(api, handlers={
            "test_echo": lambda lease: ArtifactOutput(
                artifact_type="test_result",
                data=str(lease.input_manifest).encode(),
                schema_version=1,
                pipeline_config_digest="a" * 64,
                model_version=lease.required_model_version,
            ),
        })

        self.assertTrue(runner.run_once())
        self.assertEqual(
            [call[0] for call in api.calls],
            ["lease", "start", "submit", "complete"],
        )
        self.assertFalse(runner.run_once())

    def test_unknown_task_is_reported_as_failure_not_executed(self):
        api = _FakeApi()
        api.leases[0] = TaskLease(
            job_id="job-2",
            lease_token="secret",
            task_type="unknown",
            input_manifest={},
            required_model_version=None,
        )
        self.assertTrue(WorkerRunner(api, handlers={}).run_once())
        self.assertEqual(api.calls[-1][0], "fail")

    def test_long_handler_renews_lease_until_artifact_upload_finishes(self):
        api = _FakeApi()

        def slow_handler(lease):
            time.sleep(0.04)
            return ArtifactOutput(
                artifact_type="result", data=b"ok", schema_version=1,
                pipeline_config_digest="a" * 64,
                model_version=lease.required_model_version,
            )

        runner = WorkerRunner(
            api,
            handlers={"test_echo": slow_handler},
            heartbeat_seconds=0.01,
        )
        self.assertTrue(runner.run_once())
        names = [call[0] for call in api.calls]
        self.assertIn("heartbeat", names)
        self.assertLess(names.index("heartbeat"), names.index("submit"))


if __name__ == "__main__":
    unittest.main()
