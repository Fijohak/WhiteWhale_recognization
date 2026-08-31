"""M1 PostgreSQL 领域模型。"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    ForeignKeyConstraint,
    Float,
    Integer,
    JSON,
    LargeBinary,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .states import BatchStage, JobState


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)


class UserRole(Base):
    __tablename__ = "user_roles"
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)


class UserSession(TimestampMixin, Base):
    __tablename__ = "user_sessions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_digest: Mapped[bytes] = mapped_column(
        LargeBinary(32), unique=True, nullable=False)
    csrf_digest: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    """安全与裁决动作的只追加记录；数据库 Migration 禁止 UPDATE/DELETE。"""

    __tablename__ = "audit_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"))
    actor_worker_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("worker_devices.id", ondelete="RESTRICT"))
    event_type: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True)
    target_type: Mapped[str | None] = mapped_column(String(64))
    target_id: Mapped[str | None] = mapped_column(String(256))
    detail: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
        index=True)
    __table_args__ = (
        CheckConstraint(
            "actor_type IN ('user', 'worker', 'system', 'anonymous')",
            name="valid_actor_type"),
        CheckConstraint(
            "actor_user_id IS NULL OR actor_worker_id IS NULL",
            name="at_most_one_actor"),
    )


class WorkerDevice(TimestampMixin, Base):
    __tablename__ = "worker_devices"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    gpu_model: Mapped[str] = mapped_column(String(256), nullable=False)
    vram_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    cuda_version: Mapped[str] = mapped_column(String(64), nullable=False)
    worker_version: Mapped[str] = mapped_column(String(64), nullable=False)
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    model_versions: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    __table_args__ = (
        CheckConstraint("vram_mb > 0", name="positive_vram_mb"),
        CheckConstraint("capacity > 0", name="positive_capacity"),
    )


class WorkerToken(TimestampMixin, Base):
    __tablename__ = "worker_tokens"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    device_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("worker_devices.id", ondelete="CASCADE"), nullable=False)
    token_digest: Mapped[bytes] = mapped_column(
        LargeBinary(32), unique=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorkerRegistrationCode(TimestampMixin, Base):
    __tablename__ = "worker_registration_codes"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    code_digest: Mapped[bytes] = mapped_column(
        LargeBinary(32), unique=True, nullable=False)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    used_by_device_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("worker_devices.id", ondelete="RESTRICT"))


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    device_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("worker_devices.id", ondelete="CASCADE"), nullable=False)
    reported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    available_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    detail: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        CheckConstraint("available_capacity >= 0", name="nonnegative_capacity"),
    )


class Batch(TimestampMixin, Base):
    __tablename__ = "batches"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    upload_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("upload_sessions.id", ondelete="RESTRICT"),
        unique=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    stage: Mapped[BatchStage] = mapped_column(
        SqlEnum(
            BatchStage,
            values_callable=lambda enum: [member.value for member in enum],
            native_enum=False,
            create_constraint=False,
        ),
        default=BatchStage.REGISTERED, nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_format: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        CheckConstraint(
            "stage IN ('registered', 'candidate_ready', 'under_review', "
            "'approved', 'catalog_staged', 'published', 'archived')",
            name="valid_stage",
        ),
    )


class UploadSession(TimestampMixin, Base):
    __tablename__ = "upload_sessions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    batch_name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_format: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), default="uploading", nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("chunk_size > 0", name="positive_chunk_size"),
        CheckConstraint("total_bytes >= 0", name="nonnegative_total_bytes"),
    )


class UploadFile(TimestampMixin, Base):
    __tablename__ = "upload_files"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    upload_session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("upload_sessions.id", ondelete="CASCADE"), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False)
    completed_path: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        UniqueConstraint("upload_session_id", "relative_path"),
        CheckConstraint("size_bytes >= 0", name="nonnegative_size_bytes"),
    )


class UploadPart(TimestampMixin, Base):
    __tablename__ = "upload_parts"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    upload_file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("upload_files.id", ondelete="CASCADE"), nullable=False)
    part_number: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    staging_path: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        UniqueConstraint("upload_file_id", "part_number"),
        CheckConstraint("part_number >= 0", name="nonnegative_part_number"),
        CheckConstraint("size_bytes >= 0", name="nonnegative_size_bytes"),
    )


class QueryRequest(TimestampMixin, Base):
    __tablename__ = "query_requests"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    upload_session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("upload_sessions.id", ondelete="RESTRICT"),
        unique=True, nullable=False)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="RESTRICT"), unique=True)
    catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    detector_version: Mapped[str] = mapped_column(String(128), nullable=False)
    preprocess_id: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), default="queued", nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    input_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint(
            "state IN ('queued', 'running', 'succeeded', 'failed')",
            name="valid_state"),
        CheckConstraint("top_k > 0 AND top_k <= 100", name="valid_top_k"),
    )


class QueryImage(TimestampMixin, Base):
    __tablename__ = "query_images"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    query_request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("query_requests.id", ondelete="CASCADE"), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    original_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "query_request_id", "row_index",
            name="query_images_request_row"),
        UniqueConstraint(
            "query_request_id", "original_relative_path",
            name="query_images_request_path"),
        CheckConstraint("row_index >= 0", name="nonnegative_row_index"),
        CheckConstraint("size_bytes >= 0", name="nonnegative_size_bytes"),
    )


class SourceGroup(TimestampMixin, Base):
    __tablename__ = "source_groups"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        UniqueConstraint("batch_id", "kind", "name"),
    )


class Image(TimestampMixin, Base):
    __tablename__ = "images"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("batches.id", ondelete="RESTRICT"), nullable=False)
    source_group_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("source_groups.id", ondelete="RESTRICT"))
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    original_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    quality_band: Mapped[str | None] = mapped_column(String(64))
    relation_note: Mapped[str | None] = mapped_column(String(128))
    exif_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        UniqueConstraint("batch_id", "source_path"),
        CheckConstraint("size_bytes >= 0", name="nonnegative_size_bytes"),
    )


class Crop(TimestampMixin, Base):
    __tablename__ = "crops"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("images.id", ondelete="RESTRICT"), nullable=False)
    crop_index: Mapped[int] = mapped_column(Integer, nullable=False)
    x: Mapped[int] = mapped_column(Integer, nullable=False)
    y: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    detector_version: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        UniqueConstraint("image_id", "crop_index"),
        CheckConstraint("crop_index >= 0", name="nonnegative_crop_index"),
        CheckConstraint("width > 0 AND height > 0", name="positive_dimensions"),
    )


class CropEmbedding(TimestampMixin, Base):
    __tablename__ = "crop_embeddings"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    crop_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("crops.id", ondelete="RESTRICT"), nullable=False)
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    preprocess_id: Mapped[str] = mapped_column(String(128), nullable=False)
    pipeline_config_digest: Mapped[str] = mapped_column(
        String(64), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "crop_id", "model_version", "preprocess_id",
            name="crop_embeddings_crop_model_preprocess",
        ),
        UniqueConstraint(
            "artifact_id", "row_index",
            name="crop_embeddings_artifact_row",
        ),
        CheckConstraint("row_index >= 0", name="nonnegative_row_index"),
        CheckConstraint("feature_dim > 0", name="positive_feature_dim"),
    )


class Collection(TimestampMixin, Base):
    __tablename__ = "collections"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    system_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)


class CollectionMembership(TimestampMixin, Base):
    __tablename__ = "collection_memberships"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"))
    crop_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("crops.id", ondelete="CASCADE"))
    assignment_source: Mapped[str] = mapped_column(String(64), nullable=False)
    membership_status: Mapped[str] = mapped_column(
        String(32), default="candidate", nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "collection_id", "image_id",
            name="collection_memberships_collection_image",
        ),
        UniqueConstraint(
            "collection_id", "crop_id",
            name="collection_memberships_collection_crop",
        ),
        CheckConstraint(
            "(image_id IS NULL) <> (crop_id IS NULL)",
            name="exactly_one_subject",
        ),
        CheckConstraint(
            "membership_status IN ('candidate', 'review_pending', "
            "'confirmed_member', 'rejected')",
            name="valid_status",
        ),
    )


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("batches.id", ondelete="RESTRICT"))
    task_type: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[JobState] = mapped_column(
        SqlEnum(
            JobState,
            values_callable=lambda enum: [member.value for member in enum],
            native_enum=False,
            create_constraint=False,
        ),
        default=JobState.QUEUED, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    required_vram_mb: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    required_model_version: Mapped[str | None] = mapped_column(String(128))
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False)
    input_manifest: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        CheckConstraint(
            "state IN ('queued', 'leased', 'running', 'uploading', "
            "'validating', 'succeeded', 'failed', 'cancelled', "
            "'lease_expired')",
            name="valid_state",
        ),
        CheckConstraint("required_vram_mb >= 0", name="nonnegative_vram"),
        CheckConstraint("max_attempts > 0", name="positive_max_attempts"),
    )


class JobAttempt(TimestampMixin, Base):
    __tablename__ = "job_attempts"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_device_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("worker_devices.id", ondelete="RESTRICT"))
    lease_token_digest: Mapped[bytes | None] = mapped_column(LargeBinary(32))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str | None] = mapped_column(String(64))
    error_detail: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        UniqueConstraint("job_id", "attempt_number"),
        CheckConstraint("attempt_number > 0", name="positive_attempt_number"),
    )


class JobLease(Base):
    __tablename__ = "job_leases"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), unique=True, nullable=False)
    attempt_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("job_attempts.id", ondelete="CASCADE"), unique=True,
        nullable=False)
    device_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("worker_devices.id", ondelete="RESTRICT"), nullable=False)
    lease_token_digest: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    leased_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False)
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("job_attempts.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class Artifact(TimestampMixin, Base):
    __tablename__ = "artifacts"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="RESTRICT"), nullable=False)
    attempt_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("job_attempts.id", ondelete="RESTRICT"), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    producer_device_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("worker_devices.id", ondelete="RESTRICT"), nullable=False)
    __table_args__ = (
        UniqueConstraint("job_id", "artifact_type", "sha256"),
        CheckConstraint("size_bytes >= 0", name="nonnegative_size_bytes"),
    )


class ArtifactManifest(Base):
    __tablename__ = "artifact_manifests"
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifacts.id", ondelete="CASCADE"), primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    row_binding_digest: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(128))
    detector_version: Mapped[str | None] = mapped_column(String(128))
    preprocess_id: Mapped[str | None] = mapped_column(String(128))
    pipeline_config_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        CheckConstraint("schema_version > 0", name="positive_schema_version"),
    )


class LegacyArtifact(TimestampMixin, Base):
    __tablename__ = "legacy_artifacts"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    artifact_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    source_name: Mapped[str] = mapped_column(String(512), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    calibration_status: Mapped[str] = mapped_column(
        String(64), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict,
                                                nullable=False)
    __table_args__ = (
        UniqueConstraint("artifact_kind", "sha256"),
        CheckConstraint("size_bytes >= 0", name="nonnegative_size"),
        CheckConstraint(
            "calibration_status IN ('not_applicable', "
            "'provisional_unvalidated', 'calibrated')",
            name="valid_calibration_status"),
    )


class CandidateCluster(TimestampMixin, Base):
    __tablename__ = "candidate_clusters"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("batches.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(128), nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), default="needs_review", nullable=False)
    representative_crop_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("crops.id", ondelete="RESTRICT"))
    provenance_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        UniqueConstraint("batch_id", "label"),
    )


class CandidateClusterMember(Base):
    __tablename__ = "candidate_cluster_members"
    cluster_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidate_clusters.id", ondelete="CASCADE"),
        primary_key=True)
    crop_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("crops.id", ondelete="RESTRICT"), primary_key=True)
    membership_score: Mapped[float | None] = mapped_column(Float)
    is_excluded: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False)


class MatchCandidate(TimestampMixin, Base):
    __tablename__ = "match_candidates"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    cluster_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidate_clusters.id", ondelete="CASCADE"), nullable=False)
    catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"), nullable=False)
    individual_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"),
        nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    support_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="needs_review", nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "cluster_id", "rank", name="match_candidates_cluster_rank"),
        UniqueConstraint(
            "cluster_id", "individual_id",
            name="match_candidates_cluster_individual",
        ),
        CheckConstraint("rank > 0", name="positive_rank"),
        CheckConstraint("support_frames > 0", name="positive_support_frames"),
    )


class CandidateEvent(Base):
    __tablename__ = "candidate_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cluster_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidate_clusters.id", ondelete="CASCADE"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"))
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class ReviewTask(TimestampMixin, Base):
    __tablename__ = "review_tasks"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="open", nullable=False)
    required_reviewers: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (
        CheckConstraint("required_reviewers > 0", name="positive_reviewers"),
    )


class ReviewerRoster(Base):
    __tablename__ = "reviewer_rosters"
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="CASCADE"), primary_key=True)
    reviewer_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), primary_key=True)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class ReviewEvent(Base):
    __tablename__ = "review_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="CASCADE"), nullable=False)
    reviewer_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    __table_args__ = (
        UniqueConstraint("task_id", "reviewer_user_id", "revision"),
        CheckConstraint("revision > 0", name="positive_revision"),
    )


class ReviewConsensus(Base):
    __tablename__ = "review_consensus"
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    conclusion: Mapped[str | None] = mapped_column(String(64))
    individual_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class ReviewConflict(Base):
    __tablename__ = "review_conflicts"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="CASCADE"),
        unique=True, nullable=False)
    reason: Mapped[str] = mapped_column(String(256), nullable=False)
    detail: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class ConfirmedIndividual(TimestampMixin, Base):
    __tablename__ = "confirmed_individuals"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    display_name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    state: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)


class IndividualAlias(TimestampMixin, Base):
    __tablename__ = "individual_aliases"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    individual_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="CASCADE"),
        nullable=False)
    alias: Mapped[str] = mapped_column(String(256), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (
        UniqueConstraint("individual_id", "alias", "source"),
    )


class Observation(TimestampMixin, Base):
    __tablename__ = "observations"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    individual_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"),
        nullable=False)
    crop_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("crops.id", ondelete="RESTRICT"), unique=True, nullable=False)
    side: Mapped[str] = mapped_column(String(16), default="unknown", nullable=False)
    quality: Mapped[float | None] = mapped_column(Float)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_review_task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="RESTRICT"), nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), default="active", nullable=False)
    __table_args__ = (
        CheckConstraint(
            "state IN ('active', 'withdrawn')", name="valid_state"),
    )


class IdentityEvent(Base):
    __tablename__ = "identity_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    individual_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"),
        nullable=False)
    review_task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="RESTRICT"),
        unique=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class IdentityChangeProposal(TimestampMixin, Base):
    __tablename__ = "identity_change_proposals"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    change_type: Mapped[str] = mapped_column(String(32), nullable=False)
    plan: Mapped[dict] = mapped_column(JSON, nullable=False)
    plan_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="review_pending", nullable=False)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint(
            "change_type IN ('merge', 'split', 'withdrawal')",
            name="valid_change_type",
        ),
        CheckConstraint(
            "status IN ('review_pending', 'applied', 'rejected', 'disputed')",
            name="valid_status",
        ),
    )


class IdentityChangeEvent(Base):
    __tablename__ = "identity_change_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    proposal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("identity_change_proposals.id", ondelete="RESTRICT"),
        unique=True, nullable=False)
    review_task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="RESTRICT"),
        unique=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class CooccurrenceEvent(TimestampMixin, Base):
    __tablename__ = "cooccurrence_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("images.id", ondelete="RESTRICT"), unique=True,
        nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="review_pending", nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    provenance_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"))
    __table_args__ = (
        CheckConstraint(
            "status IN ('review_pending', 'confirmed', 'rejected', 'disputed')",
            name="valid_status",
        ),
    )


class CooccurrenceMember(Base):
    __tablename__ = "cooccurrence_members"
    event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cooccurrence_events.id", ondelete="CASCADE"),
        primary_key=True)
    crop_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("crops.id", ondelete="RESTRICT"), primary_key=True)
    individual_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"))
    membership_status: Mapped[str] = mapped_column(
        String(32), default="candidate", nullable=False)
    source_review_task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("review_tasks.id", ondelete="RESTRICT"))
    __table_args__ = (
        CheckConstraint(
            "membership_status IN ('candidate', 'confirmed_member', 'rejected')",
            name="valid_status",
        ),
    )


class RelationshipHypothesis(TimestampMixin, Base):
    __tablename__ = "relationship_hypotheses"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    individual_low_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"),
        nullable=False)
    individual_high_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"),
        nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="suspected", nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "individual_low_id", "individual_high_id", "relationship_type",
            name="relationship_hypotheses_pair_type",
        ),
        CheckConstraint(
            "individual_low_id <> individual_high_id", name="distinct_pair"),
        CheckConstraint(
            "relationship_type IN ('co_occurrence', 'repeated_association', "
            "'suspected_kinship')",
            name="valid_type",
        ),
        CheckConstraint(
            "status IN ('suspected', 'evidence_insufficient', 'disputed', "
            "'rejected')",
            name="valid_status",
        ),
    )


class RelationshipEvidence(TimestampMixin, Base):
    __tablename__ = "relationship_evidence"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("relationship_hypotheses.id", ondelete="CASCADE"),
        nullable=False)
    cooccurrence_event_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("cooccurrence_events.id", ondelete="RESTRICT"),
        nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        UniqueConstraint("hypothesis_id", "cooccurrence_event_id"),
    )


class RelationshipEvent(Base):
    __tablename__ = "relationship_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("relationship_hypotheses.id", ondelete="RESTRICT"),
        nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"))
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class DatasetVersion(TimestampMixin, Base):
    __tablename__ = "dataset_versions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    protocol: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="frozen", nullable=False)
    catalog_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"))
    membership_digest: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False)
    rights_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    __table_args__ = (
        CheckConstraint(
            "protocol IN ('known_identity_update', 'open_set_unknown')",
            name="valid_protocol",
        ),
        CheckConstraint(
            "status IN ('frozen', 'retired')", name="valid_status"),
    )


class DatasetMembership(Base):
    __tablename__ = "dataset_memberships"
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        primary_key=True)
    observation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("observations.id", ondelete="RESTRICT"), primary_key=True)
    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("images.id", ondelete="RESTRICT"), nullable=False)
    crop_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("crops.id", ondelete="RESTRICT"), nullable=False)
    individual_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"),
        nullable=False)
    label_source: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence_key: Mapped[str] = mapped_column(String(256), nullable=False)
    encounter_key: Mapped[str] = mapped_column(String(256), nullable=False)
    duplicate_group: Mapped[str] = mapped_column(String(256), nullable=False)
    data_license: Mapped[str] = mapped_column(String(128), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id", "crop_id",
            name="dataset_memberships_dataset_crop",
        ),
        CheckConstraint(
            "label_source IN ('provider_confirmed', 'project_verified', "
            "'high_trust_pseudo_label')",
            name="valid_label_source",
        ),
    )


class DatasetSplit(Base):
    __tablename__ = "dataset_splits"
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True)
    observation_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    split: Mapped[str] = mapped_column(String(32), nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_version_id", "observation_id"],
            ["dataset_memberships.dataset_version_id",
             "dataset_memberships.observation_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "split IN ('train', 'val', 'calibration', 'test')",
            name="valid_split",
        ),
    )


class TrainingRun(TimestampMixin, Base):
    __tablename__ = "training_runs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="RESTRICT"), unique=True,
        nullable=False)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_family: Mapped[str] = mapped_column(String(128), nullable=False)
    base_model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "model_versions.id", ondelete="RESTRICT", use_alter=True,
            name="fk_training_runs_base_model_version_id_model_versions",
        ))
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    required_vram_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    max_runtime_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_interval_steps: Mapped[int] = mapped_column(
        Integer, nullable=False)
    resume_checkpoint_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    state: Mapped[str] = mapped_column(
        String(32), default="queued", nullable=False)
    __table_args__ = (
        CheckConstraint(
            "task_type IN ('detector_training', 'reid_training')",
            name="valid_task_type",
        ),
        CheckConstraint(
            "state IN ('queued', 'running', 'completed', 'failed')",
            name="valid_state",
        ),
        CheckConstraint(
            "required_vram_mb > 0 AND max_runtime_seconds > 0 "
            "AND checkpoint_interval_steps > 0",
            name="positive_limits",
        ),
    )


class TrainingCheckpoint(TimestampMixin, Base):
    __tablename__ = "training_checkpoints"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    training_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False)
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"), unique=True,
        nullable=False)
    stage: Mapped[int] = mapped_column(Integer, nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict,
                                                nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "training_run_id", "stage", "epoch", "step",
            name="training_checkpoints_run_stage_epoch_step",
        ),
        CheckConstraint(
            "stage >= 0 AND epoch >= 0 AND step >= 0",
            name="nonnegative_step"),
    )


class ModelVersion(TimestampMixin, Base):
    __tablename__ = "model_versions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    training_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("training_runs.id", ondelete="RESTRICT"), unique=True,
        nullable=False)
    weight_artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"), unique=True,
        nullable=False)
    model_family: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="candidate", nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    weight_path: Mapped[str] = mapped_column(Text, nullable=False)
    feature_dim: Mapped[int | None] = mapped_column(Integer)
    preprocess_id: Mapped[str] = mapped_column(String(128), nullable=False)
    checkpoint_source: Mapped[str] = mapped_column(String(256), nullable=False)
    license: Mapped[str] = mapped_column(String(256), nullable=False)
    compatible_detector_version: Mapped[str | None] = mapped_column(
        String(128))
    compatible_crop_config: Mapped[str] = mapped_column(String(128),
                                                        nullable=False)
    compatible_index_schema: Mapped[int] = mapped_column(Integer,
                                                          nullable=False)
    calibrated_thresholds: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False)
    __table_args__ = (
        CheckConstraint(
            "status IN ('candidate', 'promotion_pending', 'production', "
            "'retired', 'rejected')",
            name="valid_status",
        ),
        CheckConstraint(
            "feature_dim IS NULL OR feature_dim > 0", name="positive_dim"),
        CheckConstraint(
            "compatible_index_schema > 0", name="positive_index_schema"),
    )


class EvaluationRun(TimestampMixin, Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="RESTRICT"), unique=True,
        nullable=False)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False)
    protocol: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="queued", nullable=False)
    report_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="RESTRICT"))
    comparison: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    calibrated_thresholds: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False)
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')",
            name="valid_status",
        ),
    )


class EvaluationResult(TimestampMixin, Base):
    __tablename__ = "evaluation_results"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    split: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "evaluation_run_id", "metric_name", "split",
            name="evaluation_results_run_metric_split",
        ),
    )


class ModelPromotionEvent(Base):
    __tablename__ = "model_promotion_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    catalog_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"))
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)


class ProductionModelPointer(Base):
    __tablename__ = "production_model_pointer"
    model_family: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        unique=True, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False)


class CatalogVersion(TimestampMixin, Base):
    __tablename__ = "catalog_versions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(32), default="staged", nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    calibration_status: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    membership_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    index_path: Mapped[str] = mapped_column(Text, nullable=False)
    index_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_catalog_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"))
    source_batch_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("batches.id", ondelete="RESTRICT"))
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("feature_dim > 0", name="positive_feature_dim"),
        CheckConstraint("row_count > 0", name="positive_row_count"),
        CheckConstraint(
            "status IN ('staged', 'active', 'retired')",
            name="valid_status"),
    )


class CatalogMembership(Base):
    __tablename__ = "catalog_memberships"
    catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="CASCADE"), primary_key=True)
    observation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("observations.id", ondelete="RESTRICT"), primary_key=True)
    individual_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("confirmed_individuals.id", ondelete="RESTRICT"),
        nullable=False)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (
        UniqueConstraint("catalog_id", "row_index"),
        CheckConstraint("row_index >= 0", name="nonnegative_row_index"),
    )


class ActiveCatalogPointer(Base):
    __tablename__ = "active_catalog_pointer"
    singleton_id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"),
        unique=True, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False)
    __table_args__ = (
        CheckConstraint("singleton_id = 1", name="singleton"),
    )


class CatalogEvent(Base):
    __tablename__ = "catalog_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    catalog_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_catalog_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("catalog_versions.id", ondelete="RESTRICT"))
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
