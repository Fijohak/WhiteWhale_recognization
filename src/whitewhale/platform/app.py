"""协作平台 FastAPI 入口。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
import uuid

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import (
    ArtifactSpec,
    ArtifactValidationError,
    WorkerResultService,
)
from .archival_workflow import ArchivalWorkflowService
from .archival_dispatch import ArchivalDispatchRequest, ArchivalDispatchService
from .auth import (
    AuthService,
    BootstrapClosed,
    InvalidCredentials,
    InvalidSession,
    Principal,
)
from .catalogs import CatalogService, CatalogSnapshot, CatalogValidationError
from .cooccurrence import CooccurrenceService
from .imports import BatchImportService
from .identity_changes import IdentityChangeService
from .jobs import InvalidLease, JobConflict, JobQueueService, LeaseService
from .media import MediaNotFound, MediaService
from .review_policy import ReviewVote
from .reviews import ReviewService
from .uploads import (
    UploadConflict,
    UploadFileSpec,
    UploadService,
)
from .views import ArchiveReadService
from .worker_auth import (
    InvalidWorkerCredential,
    WorkerAuthService,
    WorkerPrincipal,
    WorkerRegistration,
)


@dataclass(frozen=True)
class Readiness:
    database: bool
    storage: bool
    migrations: bool
    active_catalog: bool
    detail: str = ""

    @property
    def ready(self) -> bool:
        return all((self.database, self.storage,
                    self.migrations, self.active_catalog))


ReadinessProbe = Callable[[], Readiness]


@dataclass(frozen=True)
class PlatformServices:
    auth: AuthService
    uploads: UploadService
    imports: BatchImportService
    worker_auth: WorkerAuthService | None = None
    jobs: JobQueueService | None = None
    leases: LeaseService | None = None
    results: WorkerResultService | None = None
    media: MediaService | None = None
    reviews: ReviewService | None = None
    catalogs: CatalogService | None = None
    archival: ArchivalWorkflowService | None = None
    views: ArchiveReadService | None = None
    archival_dispatch: ArchivalDispatchService | None = None
    cooccurrence: CooccurrenceService | None = None
    identity_changes: IdentityChangeService | None = None


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Credentials(_StrictModel):
    username: str
    password: str


class UserCreateRequest(Credentials):
    roles: list[str]


class UploadFileRequest(_StrictModel):
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str


class UploadRequest(_StrictModel):
    batch_name: str = Field(min_length=1, max_length=256)
    source_format: str = Field(pattern="^(idolphin|generic)$")
    files: list[UploadFileRequest]


class WorkerRegistrationRequest(_StrictModel):
    registration_code: str
    name: str
    gpu_model: str
    vram_mb: int = Field(gt=0)
    cuda_version: str
    worker_version: str
    capabilities: list[str]
    model_versions: list[str]
    capacity: int = Field(default=1, gt=0)


class WorkerHeartbeatRequest(_StrictModel):
    available_capacity: int = Field(ge=0)
    detail: dict = Field(default_factory=dict)


class JobRequest(_StrictModel):
    task_type: str
    required_vram_mb: int = Field(ge=0)
    required_model_version: str | None = None
    max_attempts: int = Field(default=3, gt=0)
    idempotency_key: str
    input_manifest: dict = Field(default_factory=dict)
    priority: int = 0
    batch_id: uuid.UUID | None = None


class TaskFailureRequest(_StrictModel):
    detail: str = Field(min_length=1, max_length=8000)


class ReviewTaskCreateRequest(_StrictModel):
    task_type: str
    subject_type: str
    subject_id: uuid.UUID
    reviewer_ids: list[uuid.UUID]


class ReviewVoteRequest(_StrictModel):
    choice: str
    individual_id: uuid.UUID | None = None


class EmbeddingQueryRequest(_StrictModel):
    embedding: list[float] = Field(min_length=1)
    k: int = Field(default=5, gt=0, le=100)


class ArchivalIngestRequest(_StrictModel):
    purity_reviewer_id: uuid.UUID
    multi_target_reviewer_ids: list[uuid.UUID] = Field(default_factory=list)


class PurityAdvanceRequest(_StrictModel):
    identity_reviewer_ids: list[uuid.UUID]


class CatalogStageRequest(_StrictModel):
    model_version: str = Field(min_length=1, max_length=128)
    calibration_status: str = Field(
        default="provisional_unvalidated", min_length=1, max_length=64)


class ArchivalJobRequest(_StrictModel):
    model_version: str = Field(min_length=1, max_length=128)
    detector_version: str = Field(min_length=1, max_length=128)
    preprocess_id: str = Field(min_length=1, max_length=128)
    pipeline_config: dict = Field(default_factory=dict)
    required_vram_mb: int = Field(default=4096, gt=0)
    max_attempts: int = Field(default=3, gt=0)


class CooccurrenceCreateRequest(_StrictModel):
    image_id: uuid.UUID
    crop_ids: list[uuid.UUID] = Field(min_length=2)
    reviewer_ids: list[uuid.UUID]
    provenance_artifact_id: uuid.UUID | None = None


class IdentityMergeRequest(_StrictModel):
    source_individual_ids: list[uuid.UUID] = Field(min_length=2)
    target_individual_id: uuid.UUID
    reviewer_ids: list[uuid.UUID]


class IdentitySplitAssignmentRequest(_StrictModel):
    observation_id: uuid.UUID
    group: str = Field(min_length=1, max_length=128)


class IdentitySplitRequest(_StrictModel):
    source_individual_id: uuid.UUID
    assignments: list[IdentitySplitAssignmentRequest] = Field(min_length=2)
    reviewer_ids: list[uuid.UUID]


class ObservationWithdrawalRequest(_StrictModel):
    observation_ids: list[uuid.UUID] = Field(min_length=1)
    reviewer_ids: list[uuid.UUID]


def _default_readiness_probe() -> Readiness:
    return Readiness(
        database=False,
        storage=False,
        migrations=False,
        active_catalog=False,
        detail="platform dependencies are not configured",
    )


def create_app(
    readiness_probe: ReadinessProbe | None = None,
    *,
    services: PlatformServices | None = None,
) -> FastAPI:
    probe = readiness_probe or _default_readiness_probe
    app = FastAPI(title="中华白海豚识别归档平台")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        result = probe()
        body = {
            "status": "ready" if result.ready else "not_ready",
            "checks": {
                key: value
                for key, value in asdict(result).items()
                if key != "detail"
            },
            "detail": result.detail,
        }
        if result.ready:
            return body
        return JSONResponse(body, status_code=503)

    if services is not None:
        _mount_human_api(app, services)

    return app


def _mount_human_api(app: FastAPI, services: PlatformServices) -> None:
    session_cookie = "whitewhale_session"

    def current_principal(request: Request) -> Principal:
        token = request.cookies.get(session_cookie)
        if not token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "需要登录")
        try:
            return services.auth.resolve_session(token)
        except InvalidSession as exc:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    def protected_write(
        request: Request,
        csrf_token: str | None = Header(None, alias="X-CSRF-Token"),
    ) -> Principal:
        principal = current_principal(request)
        session_token = request.cookies.get(session_cookie)
        if csrf_token is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "缺少 CSRF 令牌")
        try:
            services.auth.verify_csrf(session_token or "", csrf_token)
        except InvalidSession as exc:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, str(exc)) from exc
        return principal

    def require_upload_role(principal: Principal) -> None:
        if principal.roles.isdisjoint({"admin", "operator"}):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "没有上传权限")

    def require_admin(principal: Principal) -> None:
        if "admin" not in principal.roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")

    def require_reviewer(principal: Principal) -> None:
        if "reviewer" not in principal.roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "需要审核人权限")

    def require_workflow_role(principal: Principal) -> None:
        if principal.roles.isdisjoint({"admin", "operator", "reviewer"}):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "没有归档操作权限")

    @app.post("/api/auth/bootstrap", status_code=status.HTTP_201_CREATED)
    def bootstrap(credentials: Credentials):
        try:
            user = services.auth.bootstrap_admin(
                credentials.username, credentials.password)
        except BootstrapClosed as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return {"user_id": str(user.id), "username": user.username}

    @app.post("/api/auth/login")
    def login(credentials: Credentials, response: Response):
        try:
            grant = services.auth.login(
                credentials.username, credentials.password)
        except (InvalidCredentials, ValueError) as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
        response.set_cookie(
            session_cookie,
            grant.session_token,
            expires=grant.expires_at,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
        )
        return {
            "csrf_token": grant.csrf_token,
            "expires_at": grant.expires_at.isoformat(),
        }

    @app.post("/api/users", status_code=status.HTTP_201_CREATED)
    def create_user(
        body: UserCreateRequest,
        principal: Principal = Depends(protected_write),
    ):
        require_admin(principal)
        try:
            user = services.auth.create_user(
                body.username, body.password, roles=set(body.roles))
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return {"user_id": str(user.id), "username": user.username}

    @app.get("/api/auth/me")
    def me(request: Request):
        principal = current_principal(request)
        return {
            "user_id": str(principal.user_id),
            "username": principal.username,
            "roles": sorted(principal.roles),
        }

    @app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(request: Request, response: Response,
               principal: Principal = Depends(protected_write)):
        del principal
        token = request.cookies.get(session_cookie)
        if token:
            services.auth.logout(token)
        response.delete_cookie(
            session_cookie, path="/", secure=True, httponly=True,
            samesite="strict")

    @app.post("/api/uploads", status_code=status.HTTP_201_CREATED)
    def create_upload(
        body: UploadRequest,
        principal: Principal = Depends(protected_write),
    ):
        require_upload_role(principal)
        try:
            grant = services.uploads.create_session(
                owner_user_id=principal.user_id,
                batch_name=body.batch_name,
                source_format=body.source_format,
                files=[UploadFileSpec(
                    relative_path=item.relative_path,
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                ) for item in body.files],
            )
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return {
            "session_id": str(grant.session_id),
            "chunk_size": grant.chunk_size,
            "files": [{
                "file_id": str(item.file_id),
                "relative_path": item.relative_path,
            } for item in grant.files],
        }

    @app.get("/api/uploads/{session_id}")
    def get_upload_status(session_id: uuid.UUID, request: Request):
        principal = current_principal(request)
        try:
            upload_status = services.uploads.status(session_id)
        except UploadConflict as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        if upload_status.owner_user_id != principal.user_id \
                and "admin" not in principal.roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "不能查看该上传会话")
        return {
            "session_id": str(upload_status.session_id),
            "state": upload_status.state,
            "chunk_size": upload_status.chunk_size,
            "files": [{
                "file_id": str(item.file_id),
                "relative_path": item.relative_path,
                "state": item.state,
                "received_parts": list(item.received_parts),
                "missing_parts": list(item.missing_parts),
            } for item in upload_status.files],
        }

    @app.put(
        "/api/uploads/{session_id}/files/{file_id}/parts/{part_number}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def put_upload_part(
        session_id: uuid.UUID,
        file_id: uuid.UUID,
        part_number: int,
        request: Request,
        content_sha256: str = Header(alias="X-Content-SHA256"),
        principal: Principal = Depends(protected_write),
    ):
        require_upload_role(principal)
        try:
            services.uploads.put_part(
                session_id, file_id, part_number,
                await request.body(), content_sha256)
        except UploadConflict as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    @app.post("/api/uploads/{session_id}/files/{file_id}/complete")
    def complete_upload_file(
        session_id: uuid.UUID,
        file_id: uuid.UUID,
        principal: Principal = Depends(protected_write),
    ):
        require_upload_role(principal)
        try:
            services.uploads.complete_file(session_id, file_id)
        except UploadConflict as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"status": "complete"}

    @app.post("/api/uploads/{session_id}/complete")
    def complete_upload_session(
        session_id: uuid.UUID,
        principal: Principal = Depends(protected_write),
    ):
        require_upload_role(principal)
        try:
            result = services.uploads.complete_session(session_id)
        except UploadConflict as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"status": result}

    @app.post(
        "/api/uploads/{session_id}/import",
        status_code=status.HTTP_201_CREATED,
    )
    def import_upload(
        session_id: uuid.UUID,
        captured_on: date | None = None,
        principal: Principal = Depends(protected_write),
    ):
        require_upload_role(principal)
        try:
            batch_id = services.imports.import_upload(
                session_id, captured_on=captured_on)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return {"batch_id": str(batch_id)}

    if services.media is not None:
        @app.get("/api/media/images/{image_id}")
        def get_image(image_id: uuid.UUID, request: Request):
            current_principal(request)
            try:
                media = services.media.image(image_id)
            except MediaNotFound as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc
            return FileResponse(media.path, media_type=media.media_type)

        @app.get("/api/media/crops/{crop_id}")
        def get_crop(crop_id: uuid.UUID, request: Request):
            current_principal(request)
            try:
                media = services.media.crop(crop_id)
            except MediaNotFound as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc
            return FileResponse(media.path, media_type=media.media_type)

    if services.views is not None:
        @app.get("/api/batches")
        def list_batches(request: Request):
            current_principal(request)
            return services.views.batches()

        @app.get("/api/reviews/inbox")
        def review_inbox(request: Request):
            principal = current_principal(request)
            require_reviewer(principal)
            return services.views.review_inbox(principal.user_id)

        @app.get("/api/candidates/{cluster_id}")
        def get_candidate(cluster_id: uuid.UUID, request: Request):
            current_principal(request)
            try:
                return services.views.candidate(cluster_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc

        @app.get("/api/cooccurrences/{event_id}")
        def get_cooccurrence(event_id: uuid.UUID, request: Request):
            current_principal(request)
            try:
                return services.views.cooccurrence(event_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc

        @app.get("/api/relationships")
        def list_relationships(request: Request):
            current_principal(request)
            return services.views.relationships()

        @app.get("/api/identity-changes/{proposal_id}")
        def get_identity_change(proposal_id: uuid.UUID, request: Request):
            current_principal(request)
            try:
                return services.views.identity_change(proposal_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc

        @app.get("/api/individuals")
        def list_individuals(request: Request):
            current_principal(request)
            return services.views.individuals()

        @app.get("/api/individuals/{individual_id}")
        def get_individual(individual_id: uuid.UUID, request: Request):
            current_principal(request)
            try:
                return services.views.individual(individual_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc

    if services.reviews is not None:
        @app.post("/api/reviews/tasks", status_code=status.HTTP_201_CREATED)
        def create_review_task(
            body: ReviewTaskCreateRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_upload_role(principal)
            try:
                task_id = services.reviews.create_task(
                    task_type=body.task_type,
                    subject_type=body.subject_type,
                    subject_id=body.subject_id,
                    reviewer_ids=body.reviewer_ids,
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"task_id": str(task_id)}

        @app.get("/api/reviews/tasks/{task_id}")
        def get_review_task(task_id: uuid.UUID, request: Request):
            principal = current_principal(request)
            if "reviewer" not in principal.roles:
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN, "需要审核人权限")
            try:
                view = services.reviews.view_for_reviewer(
                    task_id, principal.user_id)
            except ValueError as exc:
                raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
            return _review_view_body(view)

        @app.post("/api/reviews/tasks/{task_id}/votes")
        def submit_review_vote(
            task_id: uuid.UUID,
            body: ReviewVoteRequest,
            principal: Principal = Depends(protected_write),
        ):
            if "reviewer" not in principal.roles:
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN, "需要审核人权限")
            try:
                decision = services.reviews.submit_vote(
                    task_id,
                    principal.user_id,
                    ReviewVote(
                        choice=body.choice,
                        individual_id=body.individual_id,
                    ),
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return _review_decision_body(decision)

    if services.catalogs is not None:
        @app.get("/api/catalogs")
        def list_catalogs(request: Request):
            current_principal(request)
            return [
                _catalog_snapshot_body(item)
                for item in services.catalogs.list_versions()
            ]

        @app.get("/api/catalogs/active")
        def get_active_catalog(request: Request):
            current_principal(request)
            try:
                snapshot = services.catalogs.active_version()
            except CatalogValidationError as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc
            return _catalog_snapshot_body(snapshot)

        @app.post("/api/catalogs/{catalog_id}/activate")
        def activate_catalog(
            catalog_id: uuid.UUID,
            principal: Principal = Depends(protected_write),
        ):
            require_reviewer(principal)
            try:
                services.catalogs.activate(catalog_id)
                snapshot = services.catalogs.active_version()
            except CatalogValidationError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return _catalog_snapshot_body(snapshot)

        @app.post("/api/query/embedding")
        def query_embedding(body: EmbeddingQueryRequest, request: Request):
            current_principal(request)
            try:
                matches = services.catalogs.search(
                    np.asarray(body.embedding, dtype=np.float32), k=body.k)
            except CatalogValidationError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            if not matches:
                snapshot = services.catalogs.active_version()
                return {
                    "catalog_id": str(snapshot.catalog_id),
                    "model_version": snapshot.model_version,
                    "calibration_status": snapshot.calibration_status,
                    "matches": [],
                }
            first = matches[0]
            return {
                "catalog_id": str(first.catalog_id),
                "model_version": first.model_version,
                "calibration_status": first.calibration_status,
                "matches": [{
                    "individual_id": str(item.individual_id),
                    "observation_id": str(item.observation_id),
                    "score": item.score,
                    "support_frames": item.support_frames,
                } for item in matches],
            }

    if services.archival is not None:
        @app.post("/api/artifacts/{artifact_id}/ingest-archival")
        def ingest_archival_artifact(
            artifact_id: uuid.UUID,
            body: ArchivalIngestRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_upload_role(principal)
            try:
                cluster_ids = services.archival.ingest_artifact(
                    artifact_id,
                    purity_reviewer_id=body.purity_reviewer_id,
                    multi_target_reviewer_ids=body.multi_target_reviewer_ids,
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"cluster_ids": [str(value) for value in cluster_ids]}

        @app.post("/api/candidates/{cluster_id}/advance-purity")
        def advance_candidate_purity(
            cluster_id: uuid.UUID,
            body: PurityAdvanceRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                task_id = services.archival.advance_after_purity(
                    cluster_id,
                    identity_reviewer_ids=body.identity_reviewer_ids,
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"task_id": str(task_id)}

        @app.post("/api/reviews/tasks/{task_id}/apply-identity")
        def apply_identity_review(
            task_id: uuid.UUID,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                individual_id = services.archival.apply_identity_review(task_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"individual_id": str(individual_id)}

        @app.post("/api/batches/{batch_id}/catalogs/stage")
        def stage_batch_catalog(
            batch_id: uuid.UUID,
            body: CatalogStageRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_upload_role(principal)
            try:
                catalog_id = services.archival.stage_catalog(
                    batch_id,
                    model_version=body.model_version,
                    calibration_status=body.calibration_status,
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"catalog_id": str(catalog_id)}

        @app.post("/api/batches/{batch_id}/catalogs/{catalog_id}/publish")
        def publish_batch_catalog(
            batch_id: uuid.UUID,
            catalog_id: uuid.UUID,
            principal: Principal = Depends(protected_write),
        ):
            require_reviewer(principal)
            try:
                services.archival.publish_catalog(batch_id, catalog_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"status": "published", "catalog_id": str(catalog_id)}

    if services.archival_dispatch is not None:
        @app.post(
            "/api/batches/{batch_id}/archive-jobs",
            status_code=status.HTTP_201_CREATED,
        )
        def dispatch_archival_job(
            batch_id: uuid.UUID,
            body: ArchivalJobRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_upload_role(principal)
            try:
                job_id = services.archival_dispatch.dispatch(
                    batch_id,
                    ArchivalDispatchRequest(
                        model_version=body.model_version,
                        detector_version=body.detector_version,
                        preprocess_id=body.preprocess_id,
                        pipeline_config=body.pipeline_config,
                        required_vram_mb=body.required_vram_mb,
                        max_attempts=body.max_attempts,
                    ),
                )
            except (ValueError, JobConflict) as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"job_id": str(job_id)}

    if services.cooccurrence is not None:
        @app.post("/api/cooccurrences", status_code=status.HTTP_201_CREATED)
        def create_cooccurrence(
            body: CooccurrenceCreateRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_upload_role(principal)
            try:
                event_id, task_id = services.cooccurrence.create_event(
                    body.image_id,
                    body.crop_ids,
                    reviewer_ids=body.reviewer_ids,
                    provenance_artifact_id=body.provenance_artifact_id,
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"event_id": str(event_id), "task_id": str(task_id)}

        @app.post("/api/reviews/tasks/{task_id}/apply-multi-target")
        def apply_multi_target_review(
            task_id: uuid.UUID,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                event_id = services.cooccurrence.apply_review(task_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"event_id": str(event_id)}

        @app.post("/api/cooccurrences/{event_id}/project-relationships")
        def project_cooccurrence_relationships(
            event_id: uuid.UUID,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                hypothesis_ids = services.cooccurrence.project_relationships(
                    event_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"hypothesis_ids": [str(value) for value in hypothesis_ids]}

    if services.identity_changes is not None:
        @app.post(
            "/api/identity-changes/merge",
            status_code=status.HTTP_201_CREATED,
        )
        def create_identity_merge(
            body: IdentityMergeRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                proposal_id, task_id = services.identity_changes.create_merge(
                    body.source_individual_ids,
                    target_individual_id=body.target_individual_id,
                    reviewer_ids=body.reviewer_ids,
                    actor_user_id=principal.user_id,
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"proposal_id": str(proposal_id), "task_id": str(task_id)}

        @app.post(
            "/api/identity-changes/split",
            status_code=status.HTTP_201_CREATED,
        )
        def create_identity_split(
            body: IdentitySplitRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                assignments = {
                    item.observation_id: item.group for item in body.assignments
                }
                if len(assignments) != len(body.assignments):
                    raise ValueError("拆分方案包含重复 Observation")
                proposal_id, task_id = services.identity_changes.create_split(
                    body.source_individual_id,
                    assignments=assignments,
                    reviewer_ids=body.reviewer_ids,
                    actor_user_id=principal.user_id,
                )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"proposal_id": str(proposal_id), "task_id": str(task_id)}

        @app.post(
            "/api/identity-changes/withdrawal",
            status_code=status.HTTP_201_CREATED,
        )
        def create_observation_withdrawal(
            body: ObservationWithdrawalRequest,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                proposal_id, task_id = \
                    services.identity_changes.create_withdrawal(
                        body.observation_ids,
                        reviewer_ids=body.reviewer_ids,
                        actor_user_id=principal.user_id,
                    )
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {"proposal_id": str(proposal_id), "task_id": str(task_id)}

        @app.post("/api/reviews/tasks/{task_id}/apply-identity-change")
        def apply_identity_change(
            task_id: uuid.UUID,
            principal: Principal = Depends(protected_write),
        ):
            require_workflow_role(principal)
            try:
                result = services.identity_changes.apply_review(task_id)
            except ValueError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
            return {
                "proposal_id": str(result.proposal_id),
                "status": result.status,
                "created_individual_ids": [
                    str(value) for value in result.created_individual_ids],
                "affected_observation_ids": [
                    str(value) for value in result.affected_observation_ids],
            }

    if all((services.worker_auth, services.jobs,
            services.leases, services.results)):
        _mount_worker_api(
            app,
            services,
            protected_write=protected_write,
            require_admin=require_admin,
            require_upload_role=require_upload_role,
        )


def _mount_worker_api(
    app: FastAPI,
    services: PlatformServices,
    *,
    protected_write,
    require_admin,
    require_upload_role,
) -> None:
    worker_auth = services.worker_auth
    jobs = services.jobs
    leases = services.leases
    results = services.results
    assert worker_auth is not None
    assert jobs is not None
    assert leases is not None
    assert results is not None

    def current_worker(
        authorization: str | None = Header(None, alias="Authorization"),
    ) -> WorkerPrincipal:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, "需要 Worker Bearer Token")
        try:
            return worker_auth.resolve_token(authorization[len(prefix):])
        except InvalidWorkerCredential as exc:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    @app.post(
        "/api/workers/registration-codes",
        status_code=status.HTTP_201_CREATED,
    )
    def create_worker_registration_code(
        principal: Principal = Depends(protected_write),
    ):
        require_admin(principal)
        code = worker_auth.create_registration_code(principal.user_id)
        return {"registration_code": code}

    @app.post("/api/workers/register", status_code=status.HTTP_201_CREATED)
    def register_worker(body: WorkerRegistrationRequest):
        try:
            grant = worker_auth.register(
                body.registration_code,
                WorkerRegistration(
                    name=body.name,
                    gpu_model=body.gpu_model,
                    vram_mb=body.vram_mb,
                    cuda_version=body.cuda_version,
                    worker_version=body.worker_version,
                    capabilities=tuple(body.capabilities),
                    model_versions=tuple(body.model_versions),
                    capacity=body.capacity,
                ),
            )
        except InvalidWorkerCredential as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return {
            "device_id": str(grant.device_id),
            "device_token": grant.device_token,
            "token_expires_at": grant.token_expires_at.isoformat(),
        }

    @app.post("/api/workers/heartbeat", status_code=status.HTTP_204_NO_CONTENT)
    def worker_heartbeat(
        body: WorkerHeartbeatRequest,
        worker: WorkerPrincipal = Depends(current_worker),
    ):
        jobs.record_worker_heartbeat(
            worker.device_id,
            available_capacity=body.available_capacity,
            detail=body.detail,
        )

    @app.post("/api/jobs", status_code=status.HTTP_201_CREATED)
    def create_job(
        body: JobRequest,
        principal: Principal = Depends(protected_write),
    ):
        require_upload_role(principal)
        try:
            job_id = jobs.create(
                task_type=body.task_type,
                required_vram_mb=body.required_vram_mb,
                required_model_version=body.required_model_version,
                max_attempts=body.max_attempts,
                idempotency_key=body.idempotency_key,
                input_manifest=body.input_manifest,
                priority=body.priority,
                batch_id=body.batch_id,
            )
        except JobConflict as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        return {"job_id": str(job_id)}

    @app.post("/api/tasks/lease")
    def lease_task(worker: WorkerPrincipal = Depends(current_worker)):
        grant = leases.lease_next(worker.device_id)
        if grant is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        snapshot = jobs.get(grant.job_id)
        return {
            "job_id": str(grant.job_id),
            "attempt_id": str(grant.attempt_id),
            "attempt_number": grant.attempt_number,
            "lease_token": grant.lease_token,
            "lease_expires_at": grant.lease_expires_at.isoformat(),
            "task_type": snapshot.task_type,
            "input_manifest": snapshot.input_manifest,
            "required_model_version": snapshot.required_model_version,
        }

    @app.get("/api/tasks/{job_id}/inputs")
    def task_inputs(
        job_id: uuid.UUID,
        lease_token: str = Header(alias="X-Lease-Token"),
        worker: WorkerPrincipal = Depends(current_worker),
    ):
        try:
            leases.validate(
                job_id, lease_token, device_id=worker.device_id)
        except InvalidLease as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        snapshot = jobs.get(job_id)
        return {
            "job_id": str(job_id),
            "task_type": snapshot.task_type,
            "input_manifest": snapshot.input_manifest,
            "required_model_version": snapshot.required_model_version,
        }

    if services.media is not None:
        @app.get("/api/tasks/{job_id}/inputs/images/{image_id}")
        def task_input_image(
            job_id: uuid.UUID,
            image_id: uuid.UUID,
            lease_token: str = Header(alias="X-Lease-Token"),
            worker: WorkerPrincipal = Depends(current_worker),
        ):
            try:
                leases.validate(
                    job_id, lease_token, device_id=worker.device_id)
                media = services.media.leased_image(job_id, image_id)
            except InvalidLease as exc:
                raise HTTPException(
                    status.HTTP_409_CONFLICT, str(exc)) from exc
            except MediaNotFound as exc:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, str(exc)) from exc
            return FileResponse(media.path, media_type=media.media_type)

    @app.post(
        "/api/tasks/{job_id}/start",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def start_task(
        job_id: uuid.UUID,
        lease_token: str = Header(alias="X-Lease-Token"),
        worker: WorkerPrincipal = Depends(current_worker),
    ):
        _ensure_worker_lease(jobs, job_id, worker.device_id)
        try:
            results.start(job_id, lease_token)
        except (InvalidLease, ArtifactValidationError) as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    @app.post("/api/tasks/{job_id}/heartbeat")
    def task_heartbeat(
        job_id: uuid.UUID,
        lease_token: str = Header(alias="X-Lease-Token"),
        worker: WorkerPrincipal = Depends(current_worker),
    ):
        _ensure_worker_lease(jobs, job_id, worker.device_id)
        try:
            expires_at = leases.heartbeat(job_id, lease_token)
        except InvalidLease as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"lease_expires_at": expires_at.isoformat()}

    @app.post(
        "/api/tasks/{job_id}/artifacts",
        status_code=status.HTTP_201_CREATED,
    )
    async def submit_artifact(
        job_id: uuid.UUID,
        request: Request,
        lease_token: str = Header(alias="X-Lease-Token"),
        artifact_type: str = Header(alias="X-Artifact-Type"),
        content_sha256: str = Header(alias="X-Content-SHA256"),
        artifact_size: int = Header(alias="X-Artifact-Size"),
        schema_version: int = Header(alias="X-Schema-Version"),
        pipeline_config_digest: str = Header(
            alias="X-Pipeline-Config-Digest"),
        model_version: str | None = Header(None, alias="X-Model-Version"),
        row_binding_digest: str | None = Header(
            None, alias="X-Row-Binding-Digest"),
        detector_version: str | None = Header(
            None, alias="X-Detector-Version"),
        preprocess_id: str | None = Header(
            None, alias="X-Preprocess-Id"),
        worker: WorkerPrincipal = Depends(current_worker),
    ):
        _ensure_worker_lease(jobs, job_id, worker.device_id)
        try:
            artifact_id = results.submit(
                job_id,
                lease_token,
                ArtifactSpec(
                    artifact_type=artifact_type,
                    sha256=content_sha256,
                    size_bytes=artifact_size,
                    schema_version=schema_version,
                    pipeline_config_digest=pipeline_config_digest,
                    row_binding_digest=row_binding_digest,
                    model_version=model_version,
                    detector_version=detector_version,
                    preprocess_id=preprocess_id,
                ),
                await request.body(),
            )
        except (InvalidLease, ArtifactValidationError) as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"artifact_id": str(artifact_id)}

    @app.post("/api/tasks/{job_id}/complete")
    def complete_task(
        job_id: uuid.UUID,
        lease_token: str = Header(alias="X-Lease-Token"),
        worker: WorkerPrincipal = Depends(current_worker),
    ):
        try:
            jobs.assert_lease_device(job_id, worker.device_id)
        except InvalidLease:
            # 幂等完成时有效租约已经删除，由完成令牌摘要继续鉴权。
            try:
                jobs.assert_latest_attempt_device(job_id, worker.device_id)
            except InvalidLease as exc:
                raise HTTPException(
                    status.HTTP_409_CONFLICT, str(exc)) from exc
        try:
            artifact_id = results.complete(job_id, lease_token)
        except (InvalidLease, ArtifactValidationError) as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return {"artifact_id": str(artifact_id), "status": "succeeded"}

    @app.post(
        "/api/tasks/{job_id}/fail",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def fail_task(
        job_id: uuid.UUID,
        body: TaskFailureRequest,
        lease_token: str = Header(alias="X-Lease-Token"),
        worker: WorkerPrincipal = Depends(current_worker),
    ):
        _ensure_worker_lease(jobs, job_id, worker.device_id)
        try:
            results.fail(job_id, lease_token, body.detail)
        except (InvalidLease, ArtifactValidationError) as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def _ensure_worker_lease(
    jobs: JobQueueService,
    job_id: uuid.UUID,
    device_id: uuid.UUID,
) -> None:
    try:
        jobs.assert_lease_device(job_id, device_id)
    except InvalidLease as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


def _review_decision_body(decision) -> dict:
    return {
        "status": decision.status,
        "conclusion": decision.conclusion,
        "individual_id": str(decision.individual_id)
        if decision.individual_id else None,
        "flags": sorted(decision.flags),
    }


def _review_view_body(view) -> dict:
    return {
        "task_id": str(view.task_id),
        "task_type": view.task_type,
        "subject_type": view.subject_type,
        "subject_id": str(view.subject_id),
        "status": view.status,
        "own_votes": [{
            "choice": vote.choice,
            "individual_id": str(vote.individual_id)
            if vote.individual_id else None,
        } for vote in view.own_votes],
        "consensus": _review_decision_body(view.consensus)
        if view.consensus else None,
    }


def _catalog_snapshot_body(snapshot: CatalogSnapshot) -> dict:
    return {
        "catalog_id": str(snapshot.catalog_id),
        "status": snapshot.status,
        "model_version": snapshot.model_version,
        "calibration_status": snapshot.calibration_status,
        "feature_dim": snapshot.feature_dim,
        "row_count": snapshot.row_count,
        "parent_catalog_id": str(snapshot.parent_catalog_id)
        if snapshot.parent_catalog_id else None,
        "source_batch_id": str(snapshot.source_batch_id)
        if snapshot.source_batch_id else None,
        "activated_at": snapshot.activated_at.isoformat()
        if snapshot.activated_at else None,
        "created_at": snapshot.created_at.isoformat(),
    }
