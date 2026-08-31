"""按数据库对象授权解析媒体，不接受客户端文件系统路径。"""
from __future__ import annotations

import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from .models import Artifact, Crop, Image, Job
from .storage import StorageLayout


class MediaNotFound(ValueError):
    pass


@dataclass(frozen=True)
class MediaFile:
    path: Path
    media_type: str
    original_name: str


class MediaService:
    def __init__(self, sessions: sessionmaker[Session], storage: StorageLayout):
        self._sessions = sessions
        self._storage = storage

    def image(self, image_id: uuid.UUID) -> MediaFile:
        with self._sessions() as db:
            image = db.get(Image, image_id)
            if image is None:
                raise MediaNotFound("图片不存在")
            path = self._storage.resolve("raw", image.source_path)
            original_name = Path(image.original_relative_path).name
        if not path.is_file():
            raise MediaNotFound("图片文件不存在")
        media_type = mimetypes.guess_type(original_name)[0] or \
            "application/octet-stream"
        return MediaFile(path, media_type, original_name)

    def crop(self, crop_id: uuid.UUID) -> MediaFile:
        with self._sessions() as db:
            crop = db.get(Crop, crop_id)
            if crop is None:
                raise MediaNotFound("Crop 不存在")
            path = self._storage.resolve("artifacts", crop.artifact_path)
            original_name = path.name
        if not path.is_file():
            raise MediaNotFound("Crop 文件不存在")
        media_type = mimetypes.guess_type(original_name)[0] or \
            "application/octet-stream"
        return MediaFile(path, media_type, original_name)

    def leased_image(self, job_id: uuid.UUID, image_id: uuid.UUID) -> MediaFile:
        with self._sessions() as db:
            job = db.get(Job, job_id)
            image = db.get(Image, image_id)
            if job is None or image is None:
                raise MediaNotFound("图片不属于该租约任务输入")
            allowed = {
                uuid.UUID(item["image_id"])
                for item in job.input_manifest.get("samples", [])
                if item.get("image_id")
            }
            if not ((job.batch_id is not None and image.batch_id == job.batch_id)
                    or image_id in allowed):
                raise MediaNotFound("图片不属于该租约任务的 Batch")
        return self.image(image_id)

    def leased_crop(self, job_id: uuid.UUID, crop_id: uuid.UUID) -> MediaFile:
        with self._sessions() as db:
            job = db.get(Job, job_id)
            if job is None:
                raise MediaNotFound("租约任务不存在")
            manifest = job.input_manifest
            allowed = {
                uuid.UUID(item["crop_id"])
                for key in ("samples", "observations")
                for item in manifest.get(key, [])
                if item.get("crop_id")
            }
            if crop_id not in allowed:
                raise MediaNotFound("Crop 不属于该租约任务输入清单")
            if job.task_type in {"detector_training", "reid_training"}:
                split_by_crop = {
                    uuid.UUID(item["crop_id"]): item.get("split")
                    for item in manifest.get("samples", [])
                    if item.get("crop_id")
                }
                if split_by_crop.get(crop_id) == "test":
                    raise MediaNotFound("训练 Worker 不能下载冻结 test Crop")
        return self.crop(crop_id)

    def leased_artifact(
        self, job_id: uuid.UUID, artifact_id: uuid.UUID,
    ) -> MediaFile:
        with self._sessions() as db:
            job = db.get(Job, job_id)
            artifact = db.get(Artifact, artifact_id)
            if job is None or artifact is None:
                raise MediaNotFound("Artifact 不存在")
            manifest = job.input_manifest
            allowed = {
                value for value in (
                    manifest.get("weight_artifact_id"),
                    (manifest.get("resume") or {}).get("artifact_id"),
                    (manifest.get("production_model") or {}).get(
                        "weight_artifact_id"),
                ) if value
            }
            if str(artifact_id) not in allowed:
                raise MediaNotFound("Artifact 不属于该租约任务输入清单")
            path = self._storage.resolve("artifacts", artifact.relative_path)
        if not path.is_file():
            raise MediaNotFound("Artifact 文件不存在")
        return MediaFile(path, "application/octet-stream", path.name)
