"""不可变 Faiss IndexFlatIP Catalog 的校验、激活、回滚与查询。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    ActiveCatalogPointer,
    CatalogEvent,
    CatalogMembership,
    CatalogVersion,
    ConfirmedIndividual,
    Observation,
)
from .storage import StorageLayout


class CatalogValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    observation_id: uuid.UUID
    embedding: np.ndarray


@dataclass(frozen=True)
class CatalogSearchResult:
    individual_id: uuid.UUID
    observation_id: uuid.UUID
    score: float
    support_frames: int
    catalog_id: uuid.UUID
    model_version: str
    calibration_status: str


@dataclass(frozen=True)
class CatalogSnapshot:
    catalog_id: uuid.UUID
    status: str
    model_version: str
    calibration_status: str
    feature_dim: int
    row_count: int
    parent_catalog_id: uuid.UUID | None
    source_batch_id: uuid.UUID | None
    activated_at: datetime | None
    created_at: datetime


def _normalized_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise CatalogValidationError("Embedding 矩阵必须是非空二维数组")
    if not np.isfinite(matrix).all():
        raise CatalogValidationError("Embedding 包含 NaN/Inf")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise CatalogValidationError("Embedding 包含零向量")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def build_flat_ip_index(vectors: np.ndarray) -> bytes:
    matrix = _normalized_matrix(vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return faiss.serialize_index(index).tobytes()


class CatalogService:
    def __init__(self, sessions: sessionmaker[Session], storage: StorageLayout):
        self._sessions = sessions
        self._storage = storage

    def stage(
        self,
        entries: list[CatalogEntry],
        *,
        model_version: str,
        calibration_status: str,
        index_bytes: bytes,
        source_batch_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        if not model_version:
            raise CatalogValidationError("Catalog 必须绑定 Model Version")
        if not entries:
            raise CatalogValidationError("Catalog 至少需要一个 Observation")
        observation_ids = [entry.observation_id for entry in entries]
        if len(set(observation_ids)) != len(observation_ids):
            raise CatalogValidationError("Catalog 包含重复 Observation")
        matrix = _normalized_matrix(np.stack([
            np.asarray(entry.embedding, dtype=np.float32) for entry in entries
        ]))
        index = self._deserialize_and_validate(index_bytes, matrix)
        del index

        with self._sessions() as db:
            observations = {
                item.id: item for item in db.scalars(
                    select(Observation)
                    .join(
                        ConfirmedIndividual,
                        ConfirmedIndividual.id == Observation.individual_id,
                    )
                    .where(
                        Observation.id.in_(observation_ids),
                        Observation.state == "active",
                        ConfirmedIndividual.state == "active",
                    ))
            }
        if set(observations) != set(observation_ids):
            raise CatalogValidationError(
                "Catalog 只能引用存在且归属 active 身份的 active Observation")

        catalog_id = uuid.uuid4()
        memberships = []
        digest_rows = []
        for row_index, (entry, vector) in enumerate(zip(entries, matrix, strict=True)):
            vector_sha = hashlib.sha256(vector.tobytes()).hexdigest()
            individual_id = observations[entry.observation_id].individual_id
            memberships.append(CatalogMembership(
                catalog_id=catalog_id,
                observation_id=entry.observation_id,
                individual_id=individual_id,
                row_index=row_index,
                embedding_sha256=vector_sha,
            ))
            digest_rows.append({
                "row_index": row_index,
                "observation_id": str(entry.observation_id),
                "individual_id": str(individual_id),
                "embedding_sha256": vector_sha,
            })
        membership_digest = hashlib.sha256(json.dumps(
            digest_rows, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        index_sha = hashlib.sha256(index_bytes).hexdigest()
        relative = Path(str(catalog_id)) / "index.faiss"
        target = self._storage.resolve("catalog_versions", relative)
        self._atomic_write(target, index_bytes)
        try:
            with self._sessions.begin() as db:
                parent_id = db.scalar(select(ActiveCatalogPointer.catalog_id).where(
                    ActiveCatalogPointer.singleton_id == 1))
                version = CatalogVersion(
                    id=catalog_id,
                    model_version=model_version,
                    calibration_status=calibration_status,
                    feature_dim=matrix.shape[1],
                    row_count=matrix.shape[0],
                    membership_digest=membership_digest,
                    index_path=relative.as_posix(),
                    index_sha256=index_sha,
                    parent_catalog_id=parent_id,
                    source_batch_id=source_batch_id,
                )
                db.add(version)
                db.flush()
                db.add_all(memberships)
                db.flush()
                db.add(CatalogEvent(
                    catalog_id=catalog_id,
                    event_type="staged",
                    previous_catalog_id=parent_id,
                    payload={"membership_digest": membership_digest},
                ))
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return catalog_id

    def activate(self, catalog_id: uuid.UUID) -> None:
        with self._sessions.begin() as db:
            db.execute(text("SELECT pg_advisory_xact_lock(91524002)"))
            target = db.get(CatalogVersion, catalog_id, with_for_update=True)
            if target is None or target.status not in {"staged", "retired"}:
                raise CatalogValidationError("Catalog 不处于可激活状态")
            memberships = list(db.scalars(
                select(CatalogMembership)
                .where(CatalogMembership.catalog_id == catalog_id)
                .order_by(CatalogMembership.row_index)
            ))
            index_path = self._storage.resolve(
                "catalog_versions", target.index_path)
            try:
                index_bytes = index_path.read_bytes()
            except FileNotFoundError as exc:
                raise CatalogValidationError("Catalog Faiss 文件不存在") from exc
            if hashlib.sha256(index_bytes).hexdigest() != target.index_sha256:
                raise CatalogValidationError("Catalog Faiss SHA-256 不一致")
            self._validate_stored_snapshot(target, memberships, index_bytes)

            pointer = db.get(
                ActiveCatalogPointer, 1, with_for_update=True)
            previous_id = pointer.catalog_id if pointer else None
            if previous_id == catalog_id:
                return
            if previous_id is not None:
                previous = db.get(
                    CatalogVersion, previous_id, with_for_update=True)
                if previous is not None:
                    previous.status = "retired"
            target.status = "active"
            target.activated_at = datetime.now(UTC)
            if pointer is None:
                db.add(ActiveCatalogPointer(
                    singleton_id=1, catalog_id=catalog_id))
            else:
                pointer.catalog_id = catalog_id
                pointer.updated_at = datetime.now(UTC)
            db.add(CatalogEvent(
                catalog_id=catalog_id,
                event_type="activated" if target.parent_catalog_id == previous_id
                else "rollback_activated",
                previous_catalog_id=previous_id,
                payload={},
            ))

    def active_catalog_id(self) -> uuid.UUID | None:
        with self._sessions() as db:
            return db.scalar(select(ActiveCatalogPointer.catalog_id).where(
                ActiveCatalogPointer.singleton_id == 1))

    def list_versions(self) -> list[CatalogSnapshot]:
        with self._sessions() as db:
            versions = list(db.scalars(
                select(CatalogVersion).order_by(CatalogVersion.created_at.desc())
            ))
            return [self._snapshot(version) for version in versions]

    def active_version(self, *, validate_index: bool = True) -> CatalogSnapshot:
        with self._sessions() as db:
            catalog_id = db.scalar(select(ActiveCatalogPointer.catalog_id).where(
                ActiveCatalogPointer.singleton_id == 1))
            if catalog_id is None:
                raise CatalogValidationError("尚无 active Catalog")
            catalog = db.get(CatalogVersion, catalog_id)
            if catalog is None or catalog.status != "active":
                raise CatalogValidationError("active Catalog 指针无效")
            memberships = list(db.scalars(
                select(CatalogMembership)
                .where(CatalogMembership.catalog_id == catalog_id)
                .order_by(CatalogMembership.row_index)
            ))
            snapshot = self._snapshot(catalog)
            index_path = catalog.index_path
            expected_sha = catalog.index_sha256
        if validate_index:
            try:
                index_bytes = self._storage.resolve(
                    "catalog_versions", index_path).read_bytes()
            except FileNotFoundError as exc:
                raise CatalogValidationError("Catalog Faiss 文件不存在") from exc
            if hashlib.sha256(index_bytes).hexdigest() != expected_sha:
                raise CatalogValidationError("Catalog Faiss SHA-256 不一致")
            with self._sessions() as db:
                catalog = db.get(CatalogVersion, snapshot.catalog_id)
                if catalog is None:
                    raise CatalogValidationError("active Catalog 指针无效")
                self._validate_stored_snapshot(catalog, memberships, index_bytes)
        return snapshot

    def search(self, probe: np.ndarray, *, k: int = 5) -> list[CatalogSearchResult]:
        catalog_id = self.active_catalog_id()
        if catalog_id is None:
            raise CatalogValidationError("尚无 active Catalog")
        return self.search_at(catalog_id, probe, k=k)

    def search_at(
        self,
        catalog_id: uuid.UUID,
        probe: np.ndarray,
        *,
        k: int = 5,
    ) -> list[CatalogSearchResult]:
        if k <= 0:
            raise ValueError("k 必须大于 0")
        probe_matrix = _normalized_matrix(np.asarray(probe).reshape(1, -1))
        with self._sessions() as db:
            catalog = db.get(CatalogVersion, catalog_id)
            if catalog is None or catalog.status not in {"active", "retired"}:
                raise CatalogValidationError("查询绑定的 Catalog 不可用")
            memberships = list(db.scalars(
                select(CatalogMembership)
                .where(CatalogMembership.catalog_id == catalog_id)
                .order_by(CatalogMembership.row_index)
            ))
            current_individuals = {
                observation_id: individual_id
                for observation_id, individual_id in db.execute(
                    select(Observation.id, Observation.individual_id)
                    .join(
                        ConfirmedIndividual,
                        ConfirmedIndividual.id == Observation.individual_id,
                    )
                    .where(
                        Observation.id.in_([
                            item.observation_id for item in memberships]),
                        Observation.state == "active",
                        ConfirmedIndividual.state == "active",
                    )
                )
            }
            snapshot = (
                catalog.id, catalog.model_version,
                catalog.calibration_status, catalog.feature_dim,
                catalog.index_path, catalog.index_sha256,
            )
        if probe_matrix.shape[1] != snapshot[3]:
            raise CatalogValidationError("查询特征维度与 Catalog 不一致")
        index_bytes = self._storage.resolve(
            "catalog_versions", snapshot[4]).read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != snapshot[5]:
            raise CatalogValidationError("Catalog Faiss SHA-256 不一致")
        with self._sessions() as db:
            stored = db.get(CatalogVersion, catalog_id)
            if stored is None:
                raise CatalogValidationError("查询绑定的 Catalog 不存在")
            self._validate_stored_snapshot(stored, memberships, index_bytes)
        index = faiss.deserialize_index(
            np.frombuffer(index_bytes, dtype=np.uint8).copy())
        scores, rows = index.search(probe_matrix, len(memberships))
        grouped: dict[uuid.UUID, dict] = {}
        for score, row_index in zip(scores[0], rows[0], strict=True):
            if row_index < 0:
                continue
            membership = memberships[int(row_index)]
            individual_id = current_individuals.get(membership.observation_id)
            if individual_id is None:
                continue
            value = grouped.setdefault(individual_id, {
                "score": float(score),
                "observation_id": membership.observation_id,
                "support_frames": 0,
            })
            value["support_frames"] += 1
            if float(score) > value["score"]:
                value["score"] = float(score)
                value["observation_id"] = membership.observation_id
        ordered = sorted(grouped.items(), key=lambda item: item[1]["score"],
                         reverse=True)[:k]
        return [CatalogSearchResult(
            individual_id=individual_id,
            observation_id=value["observation_id"],
            score=value["score"],
            support_frames=value["support_frames"],
            catalog_id=snapshot[0],
            model_version=snapshot[1],
            calibration_status=snapshot[2],
        ) for individual_id, value in ordered]

    @staticmethod
    def _snapshot(catalog: CatalogVersion) -> CatalogSnapshot:
        return CatalogSnapshot(
            catalog_id=catalog.id,
            status=catalog.status,
            model_version=catalog.model_version,
            calibration_status=catalog.calibration_status,
            feature_dim=catalog.feature_dim,
            row_count=catalog.row_count,
            parent_catalog_id=catalog.parent_catalog_id,
            source_batch_id=catalog.source_batch_id,
            activated_at=catalog.activated_at,
            created_at=catalog.created_at,
        )

    @staticmethod
    def _deserialize_and_validate(index_bytes: bytes, expected: np.ndarray):
        try:
            index = faiss.deserialize_index(
                np.frombuffer(index_bytes, dtype=np.uint8).copy())
        except Exception as exc:
            raise CatalogValidationError("无法解析 Faiss 索引") from exc
        if not isinstance(index, faiss.IndexFlatIP) \
                or index.metric_type != faiss.METRIC_INNER_PRODUCT \
                or index.d != expected.shape[1] \
                or index.ntotal != expected.shape[0]:
            raise CatalogValidationError("Faiss 类型、维度或行数不一致")
        reconstructed = index.reconstruct_n(0, index.ntotal)
        if not np.allclose(reconstructed, expected, rtol=1e-5, atol=1e-6):
            raise CatalogValidationError("Faiss 行顺序或向量内容不一致")
        return index

    def _validate_stored_snapshot(
        self,
        catalog: CatalogVersion,
        memberships: list[CatalogMembership],
        index_bytes: bytes,
    ) -> None:
        if len(memberships) != catalog.row_count:
            raise CatalogValidationError("Catalog Membership 行数不一致")
        try:
            index = faiss.deserialize_index(
                np.frombuffer(index_bytes, dtype=np.uint8).copy())
        except Exception as exc:
            raise CatalogValidationError("无法解析 Faiss 索引") from exc
        if not isinstance(index, faiss.IndexFlatIP) \
                or index.metric_type != faiss.METRIC_INNER_PRODUCT \
                or index.d != catalog.feature_dim \
                or index.ntotal != catalog.row_count:
            raise CatalogValidationError("Faiss 类型、维度或行数不一致")
        rows = []
        for membership in memberships:
            vector = np.asarray(index.reconstruct(membership.row_index),
                                dtype=np.float32)
            vector_sha = hashlib.sha256(vector.tobytes()).hexdigest()
            if vector_sha != membership.embedding_sha256:
                raise CatalogValidationError("Faiss 向量行摘要不一致")
            rows.append({
                "row_index": membership.row_index,
                "observation_id": str(membership.observation_id),
                "individual_id": str(membership.individual_id),
                "embedding_sha256": membership.embedding_sha256,
            })
        digest = hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if digest != catalog.membership_digest:
            raise CatalogValidationError("Catalog Membership digest 不一致")

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".catalog-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_name, target)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
