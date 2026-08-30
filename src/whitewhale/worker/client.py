"""Worker HTTP 客户端与可插拔任务执行循环。"""
from __future__ import annotations

import hashlib
import json
import ssl
import time
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class TaskLease:
    job_id: str
    lease_token: str
    task_type: str
    input_manifest: dict
    required_model_version: str | None


@dataclass(frozen=True)
class ArtifactOutput:
    artifact_type: str
    data: bytes
    schema_version: int
    pipeline_config_digest: str
    model_version: str | None = None
    row_binding_digest: str | None = None
    detector_version: str | None = None
    preprocess_id: str | None = None


class WorkerApi(Protocol):
    def lease(self) -> TaskLease | None: ...
    def start(self, lease: TaskLease) -> None: ...
    def submit(self, lease: TaskLease, artifact: ArtifactOutput) -> str: ...
    def complete(self, lease: TaskLease) -> None: ...
    def fail(self, lease: TaskLease, detail: str) -> None: ...


Handler = Callable[[TaskLease], ArtifactOutput]


class WorkerRunner:
    def __init__(self, api: WorkerApi, *, handlers: dict[str, Handler]):
        self._api = api
        self._handlers = handlers

    def run_once(self) -> bool:
        lease = self._api.lease()
        if lease is None:
            return False
        handler = self._handlers.get(lease.task_type)
        if handler is None:
            self._api.fail(lease, f"不支持的任务类型: {lease.task_type}")
            return True
        try:
            self._api.start(lease)
            artifact = handler(lease)
            self._api.submit(lease, artifact)
            self._api.complete(lease)
        except Exception as exc:
            self._api.fail(lease, f"{type(exc).__name__}: {exc}")
        return True

    def run_forever(self, *, idle_seconds: float = 5.0) -> None:
        while True:
            if not self.run_once():
                time.sleep(idle_seconds)


class HttpWorkerApi:
    def __init__(
        self,
        base_url: str,
        device_token: str,
        *,
        ca_file: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._device_token = device_token
        self._timeout = timeout
        self._ssl_context = ssl.create_default_context(cafile=ca_file)

    @classmethod
    def register(
        cls,
        base_url: str,
        registration: dict,
        *,
        ca_file: str | None = None,
    ) -> dict:
        client = cls(base_url, "registration", ca_file=ca_file)
        return client._request(
            "POST", "api/workers/register", json_body=registration,
            authenticated=False)

    def lease(self) -> TaskLease | None:
        result = self._request("POST", "api/tasks/lease")
        if result is None:
            return None
        return TaskLease(
            job_id=result["job_id"],
            lease_token=result["lease_token"],
            task_type=result["task_type"],
            input_manifest=result["input_manifest"],
            required_model_version=result.get("required_model_version"),
        )

    def start(self, lease: TaskLease) -> None:
        self._request(
            "POST", f"api/tasks/{lease.job_id}/start",
            headers={"X-Lease-Token": lease.lease_token})

    def submit(self, lease: TaskLease, artifact: ArtifactOutput) -> str:
        digest = hashlib.sha256(artifact.data).hexdigest()
        headers = {
            "X-Lease-Token": lease.lease_token,
            "X-Artifact-Type": artifact.artifact_type,
            "X-Content-SHA256": digest,
            "X-Artifact-Size": str(len(artifact.data)),
            "X-Schema-Version": str(artifact.schema_version),
            "X-Pipeline-Config-Digest": artifact.pipeline_config_digest,
            "Content-Type": "application/octet-stream",
        }
        for name, value in (
            ("X-Model-Version", artifact.model_version),
            ("X-Row-Binding-Digest", artifact.row_binding_digest),
            ("X-Detector-Version", artifact.detector_version),
            ("X-Preprocess-Id", artifact.preprocess_id),
        ):
            if value is not None:
                headers[name] = value
        result = self._request(
            "POST", f"api/tasks/{lease.job_id}/artifacts",
            data=artifact.data, headers=headers)
        return result["artifact_id"]

    def complete(self, lease: TaskLease) -> None:
        self._request(
            "POST", f"api/tasks/{lease.job_id}/complete",
            headers={"X-Lease-Token": lease.lease_token})

    def fail(self, lease: TaskLease, detail: str) -> None:
        self._request(
            "POST", f"api/tasks/{lease.job_id}/fail",
            json_body={"detail": detail[:8000]},
            headers={"X-Lease-Token": lease.lease_token})

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        authenticated: bool = True,
    ):
        request_headers = dict(headers or {})
        if authenticated:
            request_headers["Authorization"] = f"Bearer {self._device_token}"
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request = Request(
            urljoin(self._base_url, path),
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(
                request, timeout=self._timeout,
                context=self._ssl_context,
            ) as response:
                payload = response.read()
                if response.status == 204 or not payload:
                    return None
                return json.loads(payload)
        except HTTPError as exc:
            payload = exc.read()
            try:
                detail = json.loads(payload).get("detail")
            except Exception:
                detail = payload.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Worker API {exc.code}: {detail or exc.reason}") from exc


def test_echo_handler(lease: TaskLease) -> ArtifactOutput:
    data = json.dumps(
        lease.input_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ArtifactOutput(
        artifact_type="test_result",
        data=data,
        schema_version=1,
        pipeline_config_digest=hashlib.sha256(
            b"whitewhale.worker.test_echo.v1").hexdigest(),
        model_version=lease.required_model_version,
    )
