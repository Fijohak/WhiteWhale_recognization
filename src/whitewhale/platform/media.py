"""按数据库对象授权解析媒体，不接受客户端文件系统路径。"""
from __future__ import annotations

import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from .models import Crop, Image, Job
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
            if job is None or image is None or job.batch_id is None \
                    or image.batch_id != job.batch_id:
                raise MediaNotFound("图片不属于该租约任务的 Batch")
        return self.image(image_id)
