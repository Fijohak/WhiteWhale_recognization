"""把 GPU 归档产物投影为候选、审核、正式身份与 Catalog。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import uuid
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .catalogs import CatalogEntry, CatalogService, build_flat_ip_index
from .identities import IdentityService
from .models import (
    ActiveCatalogPointer,
    Artifact,
    ArtifactManifest,
    Batch,
    CandidateCluster,
    CandidateClusterMember,
    CandidateEvent,
    CatalogVersion,
    ConfirmedIndividual,
    Crop,
    CropEmbedding,
    Image,
    Job,
    MatchCandidate,
    Observation,
    ReviewConsensus,
    ReviewerRoster,
    ReviewTask,
    Role,
    UserRole,
)
from .states import BatchStage, JobState, advance_batch_stage
from .storage import StorageLayout


_MAX_ARCHIVE_MEMBERS = 20_000
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
_POLICY_VERSION = "review-policy-v1"


def archival_row_binding_digest(crop_keys: list[str]) -> str:
    payload = json.dumps(
        crop_keys, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ArchivalWorkflowService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        storage: StorageLayout,
        *,
        catalogs: CatalogService,
    ) -> None:
        self._sessions = sessions
        self._storage = storage
        self._catalogs = catalogs

    def ingest_artifact(
        self,
        artifact_id: uuid.UUID,
        *,
        purity_reviewer_id: uuid.UUID,
    ) -> list[uuid.UUID]:
        existing = self._existing_clusters(artifact_id)
        if existing:
            return existing
        artifact, artifact_manifest, job = self._artifact_snapshot(artifact_id)
        payload = self._storage.resolve(
            "artifacts", artifact.relative_path).read_bytes()
        if len(payload) != artifact.size_bytes \
                or hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("归档 Artifact 文件大小或 SHA-256 不一致")
        manifest, vectors, crop_files = self._read_archive(payload)
        self._validate_manifest(
            manifest, vectors, artifact_manifest, job, crop_files)
        crop_keys = [str(item["key"]) for item in manifest["crops"]]
        normalized = self._normalize(vectors)
        written_paths: list[Path] = []
        extracted_paths: dict[str, str] = {}
        try:
            for crop in manifest["crops"]:
                key = str(crop["key"])
                suffix = Path(str(crop["path"])).suffix.lower() or ".jpg"
                relative = Path("extracted") / str(artifact_id) / f"{key}{suffix}"
                target = self._storage.resolve("artifacts", relative)
                self._atomic_write(target, crop_files[str(crop["path"])])
                written_paths.append(target)
                extracted_paths[key] = relative.as_posix()

            with self._sessions.begin() as db:
                artifact_row = db.get(Artifact, artifact_id)
                batch = db.get(Batch, job.batch_id, with_for_update=True)
                if artifact_row is None or batch is None:
                    raise ValueError("归档 Artifact 或 Batch 不存在")
                if batch.stage != BatchStage.REGISTERED:
                    raise ValueError("只有 registered Batch 可导入归档产物")
                self._require_reviewer(db, purity_reviewer_id)
                images = {
                    image.id: image
                    for image in db.scalars(select(Image).where(
                        Image.batch_id == batch.id,
                        Image.id.in_([
                            uuid.UUID(str(item["image_id"]))
                            for item in manifest["crops"]
                        ]),
                    ))
                }
                crop_rows: dict[str, Crop] = {}
                for row_index, crop_data in enumerate(manifest["crops"]):
                    key = crop_keys[row_index]
                    image_id = uuid.UUID(str(crop_data["image_id"]))
                    if image_id not in images:
                        raise ValueError("Crop 引用了当前 Batch 之外的 Image")
                    x, y, width, height = self._bbox(crop_data["bbox"])
                    crop = Crop(
                        image_id=image_id,
                        crop_index=int(crop_data["crop_index"]),
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        detector_version=manifest["detector_version"],
                        artifact_path=extracted_paths[key],
                    )
                    db.add(crop)
                    db.flush()
                    vector = normalized[row_index]
                    db.add(CropEmbedding(
                        crop_id=crop.id,
                        artifact_id=artifact_id,
                        row_index=row_index,
                        feature_dim=normalized.shape[1],
                        embedding_sha256=hashlib.sha256(
                            vector.tobytes()).hexdigest(),
                        model_version=manifest["model_version"],
                        preprocess_id=manifest["preprocess_id"],
                        pipeline_config_digest=manifest[
                            "pipeline_config_digest"],
                    ))
                    crop_rows[key] = crop

                active_catalog_id = db.scalar(select(
                    ActiveCatalogPointer.catalog_id).where(
                        ActiveCatalogPointer.singleton_id == 1))
                cluster_ids: list[uuid.UUID] = []
                for cluster_data in manifest["clusters"]:
                    member_keys = [str(value)
                                   for value in cluster_data["member_keys"]]
                    scores = cluster_data.get(
                        "membership_scores", [None] * len(member_keys))
                    if len(scores) != len(member_keys):
                        raise ValueError("候选簇成员分数数量不一致")
                    cluster = CandidateCluster(
                        batch_id=batch.id,
                        label=str(cluster_data["label"]),
                        algorithm="hdbscan",
                        algorithm_version="platform-v1",
                        representative_crop_id=crop_rows[
                            str(cluster_data["representative_key"])].id,
                        provenance_artifact_id=artifact_id,
                        metadata_json={
                            "model_version": manifest["model_version"],
                            "preprocess_id": manifest["preprocess_id"],
                            "pipeline_config_digest": manifest[
                                "pipeline_config_digest"],
                        },
                    )
                    db.add(cluster)
                    db.flush()
                    cluster_ids.append(cluster.id)
                    db.add_all([
                        CandidateClusterMember(
                            cluster_id=cluster.id,
                            crop_id=crop_rows[key].id,
                            membership_score=None if score is None else float(score),
                        )
                        for key, score in zip(member_keys, scores, strict=True)
                    ])
                    self._add_matches(
                        db,
                        cluster,
                        cluster_data.get("matches", []),
                        active_catalog_id,
                        manifest["model_version"],
                    )
                    db.add(CandidateEvent(
                        cluster_id=cluster.id,
                        event_type="worker_candidate_ingested",
                        payload={"artifact_id": str(artifact_id)},
                    ))
                    self._add_review_task(
                        db,
                        task_type="cluster_purity",
                        subject_id=cluster.id,
                        reviewer_ids=[purity_reviewer_id],
                    )
                batch.stage = advance_batch_stage(
                    batch.stage, BatchStage.CANDIDATE_READY)
                return cluster_ids
        except BaseException:
            for target in written_paths:
                target.unlink(missing_ok=True)
            raise

    def advance_after_purity(
        self,
        cluster_id: uuid.UUID,
        *,
        identity_reviewer_ids: list[uuid.UUID],
    ) -> uuid.UUID:
        with self._sessions.begin() as db:
            cluster = db.get(CandidateCluster, cluster_id, with_for_update=True)
            if cluster is None:
                raise ValueError("候选簇不存在")
            existing = db.scalar(select(ReviewTask).where(
                ReviewTask.task_type == "identity_match",
                ReviewTask.subject_type == "candidate_cluster",
                ReviewTask.subject_id == cluster_id,
            ))
            if existing is not None:
                return existing.id
            purity_task = db.scalar(select(ReviewTask).where(
                ReviewTask.task_type == "cluster_purity",
                ReviewTask.subject_type == "candidate_cluster",
                ReviewTask.subject_id == cluster_id,
            ))
            consensus = None if purity_task is None else db.get(
                ReviewConsensus, purity_task.id)
            if consensus is None or consensus.status != "resolved":
                raise ValueError("普通簇纯度审核尚未完成")
            if consensus.conclusion != "batch_cluster_confirmed":
                cluster.state = consensus.conclusion or "uncertain"
                raise ValueError("候选簇未通过纯度审核")
            cluster.state = "existing_match_candidate"
            task_id = self._add_review_task(
                db,
                task_type="identity_match",
                subject_id=cluster.id,
                reviewer_ids=identity_reviewer_ids,
            )
            batch = db.get(Batch, cluster.batch_id, with_for_update=True)
            if batch is None:
                raise ValueError("候选簇 Batch 不存在")
            if batch.stage == BatchStage.CANDIDATE_READY:
                batch.stage = advance_batch_stage(
                    batch.stage, BatchStage.UNDER_REVIEW)
            return task_id

    def apply_identity_review(self, task_id: uuid.UUID) -> uuid.UUID:
        individual_id = IdentityService(self._sessions).apply_review(task_id)
        with self._sessions.begin() as db:
            task = db.get(ReviewTask, task_id)
            cluster = None if task is None else db.get(
                CandidateCluster, task.subject_id)
            if cluster is None:
                raise ValueError("身份审核引用的候选簇不存在")
            batch = db.get(Batch, cluster.batch_id, with_for_update=True)
            states = set(db.scalars(select(CandidateCluster.state).where(
                CandidateCluster.batch_id == cluster.batch_id)))
            if states and states.issubset({
                "identity_confirmed", "rejected", "unusable",
            }):
                batch.stage = advance_batch_stage(
                    batch.stage, BatchStage.APPROVED)
        return individual_id

    def stage_catalog(
        self,
        batch_id: uuid.UUID,
        *,
        model_version: str,
        calibration_status: str = "provisional_unvalidated",
    ) -> uuid.UUID:
        with self._sessions() as db:
            batch = db.get(Batch, batch_id)
            if batch is None or batch.stage != BatchStage.APPROVED:
                raise ValueError("只有 approved Batch 可构建 Catalog")
            rows = list(db.execute(
                select(Observation.id, CropEmbedding)
                .join(CropEmbedding, CropEmbedding.crop_id == Observation.crop_id)
                .where(CropEmbedding.model_version == model_version)
                .order_by(Observation.id)
            ))
            if not rows:
                raise ValueError("没有可用于 Catalog 的已确认 Embedding")
            artifact_paths = {
                artifact_id: relative_path
                for artifact_id, relative_path in db.execute(
                    select(Artifact.id, Artifact.relative_path).where(
                        Artifact.id.in_({row[1].artifact_id for row in rows}))
                )
            }
        matrices: dict[uuid.UUID, np.ndarray] = {}
        entries: list[CatalogEntry] = []
        for observation_id, embedding in rows:
            if embedding.artifact_id not in matrices:
                path = self._storage.resolve(
                    "artifacts", artifact_paths[embedding.artifact_id])
                with zipfile.ZipFile(path) as archive:
                    matrices[embedding.artifact_id] = np.load(
                        io.BytesIO(archive.read("embeddings.npy")),
                        allow_pickle=False,
                    )
            vector = self._normalize(
                matrices[embedding.artifact_id][embedding.row_index:embedding.row_index + 1]
            )[0]
            if hashlib.sha256(vector.tobytes()).hexdigest() \
                    != embedding.embedding_sha256:
                raise ValueError("Catalog 源 Embedding 摘要不一致")
            entries.append(CatalogEntry(observation_id, vector))
        matrix = np.stack([entry.embedding for entry in entries])
        catalog_id = self._catalogs.stage(
            entries,
            model_version=model_version,
            calibration_status=calibration_status,
            index_bytes=build_flat_ip_index(matrix),
            source_batch_id=batch_id,
        )
        with self._sessions.begin() as db:
            batch = db.get(Batch, batch_id, with_for_update=True)
            if batch is None:
                raise ValueError("Batch 不存在")
            batch.stage = advance_batch_stage(
                batch.stage, BatchStage.CATALOG_STAGED)
        return catalog_id

    def publish_catalog(
        self, batch_id: uuid.UUID, catalog_id: uuid.UUID,
    ) -> None:
        with self._sessions() as db:
            batch = db.get(Batch, batch_id)
            catalog = db.get(CatalogVersion, catalog_id)
            if batch is None or batch.stage != BatchStage.CATALOG_STAGED:
                raise ValueError("Batch 尚未进入 catalog_staged")
            if catalog is None or catalog.source_batch_id != batch_id:
                raise ValueError("Catalog 不属于指定 Batch")
        self._catalogs.activate(catalog_id)
        with self._sessions.begin() as db:
            batch = db.get(Batch, batch_id, with_for_update=True)
            batch.stage = advance_batch_stage(
                batch.stage, BatchStage.PUBLISHED)

    def _existing_clusters(self, artifact_id: uuid.UUID) -> list[uuid.UUID]:
        with self._sessions() as db:
            return list(db.scalars(
                select(CandidateCluster.id)
                .where(CandidateCluster.provenance_artifact_id == artifact_id)
                .order_by(CandidateCluster.label)
            ))

    def _artifact_snapshot(
        self, artifact_id: uuid.UUID,
    ) -> tuple[Artifact, ArtifactManifest, Job]:
        with self._sessions() as db:
            artifact = db.get(Artifact, artifact_id)
            manifest = db.get(ArtifactManifest, artifact_id)
            job = None if artifact is None else db.get(Job, artifact.job_id)
            if artifact is None or manifest is None or job is None:
                raise ValueError("归档 Artifact、Manifest 或 Job 不存在")
            if artifact.artifact_type != "batch_archival" \
                    or job.task_type != "batch_archival" \
                    or job.state != JobState.SUCCEEDED \
                    or job.batch_id is None:
                raise ValueError("Artifact 不是已成功的批次归档产物")
            db.expunge(artifact)
            db.expunge(manifest)
            db.expunge(job)
            return artifact, manifest, job

    @staticmethod
    def _read_archive(
        payload: bytes,
    ) -> tuple[dict, np.ndarray, dict[str, bytes]]:
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_ARCHIVE_MEMBERS:
                    raise ValueError("归档文件成员数量超限")
                if sum(item.file_size for item in infos) \
                        > _MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("归档文件解压后大小超限")
                for info in infos:
                    path = PurePosixPath(info.filename)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("归档文件包含不安全路径")
                manifest = json.loads(archive.read("manifest.json"))
                vectors = np.load(
                    io.BytesIO(archive.read("embeddings.npy")),
                    allow_pickle=False,
                )
                crop_paths = [str(item["path"])
                              for item in manifest.get("crops", [])]
                crop_files = {path: archive.read(path) for path in crop_paths}
        except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise ValueError("归档 Artifact 结构无效") from exc
        return manifest, vectors, crop_files

    @staticmethod
    def _validate_manifest(
        manifest: dict,
        vectors: np.ndarray,
        stored: ArtifactManifest,
        job: Job,
        crop_files: dict[str, bytes],
    ) -> None:
        required = {
            "schema_version", "batch_id", "model_version",
            "detector_version", "preprocess_id", "pipeline_config_digest",
            "row_binding_digest", "crops", "clusters",
        }
        if not isinstance(manifest, dict) or required - set(manifest):
            raise ValueError("归档 Manifest 缺少必需字段")
        if manifest["schema_version"] != 1 or stored.schema_version != 1:
            raise ValueError("归档 Manifest Schema Version 不兼容")
        if uuid.UUID(str(manifest["batch_id"])) != job.batch_id:
            raise ValueError("归档 Manifest Batch 不匹配")
        for key, expected in (
            ("model_version", stored.model_version),
            ("detector_version", stored.detector_version),
            ("preprocess_id", stored.preprocess_id),
            ("pipeline_config_digest", stored.pipeline_config_digest),
            ("row_binding_digest", stored.row_binding_digest),
        ):
            if not expected or manifest[key] != expected:
                raise ValueError(f"归档 Manifest {key} 与 Artifact Manifest 不匹配")
        crops = manifest["crops"]
        if not isinstance(crops, list) or not crops:
            raise ValueError("归档 Manifest 没有 Crop")
        keys = [str(item.get("key", "")) for item in crops]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Crop key 为空或重复")
        expected_binding = archival_row_binding_digest(keys)
        if manifest["row_binding_digest"] != expected_binding:
            raise ValueError("归档 Embedding 行绑定摘要不一致")
        if vectors.ndim != 2 or vectors.shape[0] != len(crops) \
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("归档 Embedding 维度、行数或有限性无效")
        if set(crop_files) != {str(item.get("path", "")) for item in crops}:
            raise ValueError("归档 Crop 文件清单不一致")
        counts: Counter[str] = Counter()
        for cluster in manifest["clusters"]:
            member_keys = [str(value) for value in cluster.get("member_keys", [])]
            if not member_keys or any(key not in set(keys) for key in member_keys):
                raise ValueError("候选簇引用未知或空 Crop")
            if str(cluster.get("representative_key")) not in member_keys:
                raise ValueError("候选簇代表 Crop 不属于该簇")
            counts.update(member_keys)
        if counts != Counter({key: 1 for key in keys}):
            raise ValueError("每个 Crop 必须且只能属于一个候选簇")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Embedding 必须是有限二维数组")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Embedding 包含零向量")
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)

    @staticmethod
    def _bbox(value) -> tuple[int, int, int, int]:
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("Crop bbox 必须是 [x, y, width, height]")
        x, y, width, height = (int(item) for item in value)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Crop bbox 范围无效")
        return x, y, width, height

    @staticmethod
    def _require_reviewer(db: Session, reviewer_id: uuid.UUID) -> None:
        eligible = db.scalar(
            select(UserRole.user_id)
            .join(Role, Role.id == UserRole.role_id)
            .where(Role.name == "reviewer", UserRole.user_id == reviewer_id)
        )
        if eligible is None:
            raise ValueError("审核名单包含没有 reviewer 角色的用户")

    def _add_review_task(
        self,
        db: Session,
        *,
        task_type: str,
        subject_id: uuid.UUID,
        reviewer_ids: list[uuid.UUID],
    ) -> uuid.UUID:
        required = 1 if task_type == "cluster_purity" else 3
        unique_reviewers = list(dict.fromkeys(reviewer_ids))
        if len(unique_reviewers) != required:
            raise ValueError(f"{task_type} 固定需要 {required} 名审核人")
        for reviewer_id in unique_reviewers:
            self._require_reviewer(db, reviewer_id)
        task = ReviewTask(
            task_type=task_type,
            subject_type="candidate_cluster",
            subject_id=subject_id,
            required_reviewers=required,
            policy_version=_POLICY_VERSION,
        )
        db.add(task)
        db.flush()
        db.add_all([
            ReviewerRoster(
                task_id=task.id,
                reviewer_user_id=reviewer_id,
            )
            for reviewer_id in unique_reviewers
        ])
        return task.id

    @staticmethod
    def _add_matches(
        db: Session,
        cluster: CandidateCluster,
        matches: list[dict],
        active_catalog_id: uuid.UUID | None,
        model_version: str,
    ) -> None:
        if not matches:
            return
        if active_catalog_id is None:
            raise ValueError("Worker 返回历史匹配，但服务器没有 active Catalog")
        for rank, match in enumerate(matches, start=1):
            if int(match.get("rank", rank)) != rank:
                raise ValueError("历史匹配 rank 必须从 1 连续递增")
            individual_id = uuid.UUID(str(match["individual_id"]))
            if db.get(ConfirmedIndividual, individual_id) is None:
                raise ValueError("历史匹配引用不存在的正式个体")
            db.add(MatchCandidate(
                cluster_id=cluster.id,
                catalog_id=active_catalog_id,
                individual_id=individual_id,
                rank=rank,
                score=float(match["score"]),
                support_frames=int(match["support_frames"]),
                model_version=model_version,
            ))

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".crop-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
