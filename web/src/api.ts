export type User = {
  user_id: string;
  username: string;
  roles: string[];
};

export type Deployment = {
  branch: string;
  commit: string;
  deployed_at: string;
  status?: string;
  failure_reason?: string | null;
};

export type Dashboard = {
  batch_counts: Record<string, number>;
  job_counts: Record<string, number>;
  worker_counts: { total: number; online: number };
  pending_reviews: number;
  failed_jobs: number;
  queued_jobs: number;
};

export type ManifestFile = {
  relative_path: string;
  size_bytes: number;
  sha256: string;
};

export type BatchSummary = {
  batch_id: string; name: string; stage: string; source_format: string;
  image_count: number; crop_count: number; cluster_count: number; created_at: string;
};

export type BatchDetail = BatchSummary & {
  manifest_sha256: string;
  metadata: Record<string, unknown>;
  jobs: Array<{ job_id: string; task_type: string; state: string; created_at: string }>;
};

export type JobSummary = {
  job_id: string; batch_id: string | null; task_type: string; state: string;
  priority: number; required_vram_mb: number; required_model_version: string | null;
  attempt_count: number; artifact_count: number; current_worker_id: string | null;
  lease_expires_at: string | null; last_error: string | null; created_at: string;
};

export type WorkerSummary = {
  device_id: string; name: string; gpu_model: string; vram_mb: number;
  cuda_version: string; worker_version: string; capabilities: string[];
  model_versions: string[]; capacity: number; is_active: boolean; is_online: boolean;
  last_heartbeat_at: string | null; available_capacity: number | null;
  current_job_id: string | null; attempt_count: number;
};

export type UserSummary = {
  user_id: string; username: string; is_active: boolean;
  roles: string[]; created_at: string;
};

export type AuditEvent = {
  audit_event_id: string; actor_type: string; actor_user_id: string | null;
  actor_worker_id: string | null; event_type: string; target_type: string | null;
  target_id: string | null; detail: Record<string, unknown>; occurred_at: string;
};

export type ReviewTask = {
  task_id: string; task_type: string; subject_type: string;
  subject_id: string; status: string; created_at: string;
};

export type Candidate = {
  cluster_id: string; batch_id: string; label: string; state: string;
  metadata: Record<string, unknown>;
  crops: Array<{ crop_id: string; media_url: string; membership_score: number | null; is_excluded: boolean }>;
  matches: Array<{ individual_id: string; rank: number; score: number; support_frames: number; model_version: string }>;
};

export type Cooccurrence = {
  event_id: string; image_id: string; image_media_url: string;
  status: string; source: string;
  crops: Array<{
    crop_id: string; crop_index: number; media_url: string;
    membership_status: string; individual_id: string | null;
    individual_name: string | null;
  }>;
};

export type Relationship = {
  hypothesis_id: string;
  individual_low_id: string; individual_low_name: string;
  individual_high_id: string; individual_high_name: string;
  relationship_type: "co_occurrence" | "repeated_association" | "suspected_kinship";
  status: "suspected" | "evidence_insufficient" | "disputed" | "rejected";
  evidence_count: number; created_at: string;
};

export type ReviewDecision = {
  status: "pending" | "resolved" | "conflict";
  conclusion: string | null;
  individual_id: string | null;
  flags: string[];
};

export type IdentityChange = {
  proposal_id: string;
  change_type: "merge" | "split" | "withdrawal";
  status: string; plan: Record<string, unknown>; plan_digest: string;
  created_by_user_id: string; created_at: string; applied_at: string | null;
};

export type DatasetSummary = {
  dataset_version_id: string; name: string; protocol: string; status: string;
  membership_digest: string; sample_count: number;
  split_counts: Record<string, number>; created_at: string;
};

export type TrainingRunSummary = {
  training_run_id: string; job_id: string; dataset_version_id: string;
  task_type: string; model_family: string; state: string; job_state: string;
  seed: number; created_at: string;
};

export type ModelSummary = {
  model_version_id: string; model_family: string; version: string;
  status: string; sha256: string; feature_dim: number | null;
  preprocess_id: string; calibrated_thresholds: Record<string, number>;
  completed_evaluations: number; created_at: string;
};

export type Individual = {
  individual_id: string; display_name: string; state: string;
  flags: string[]; observation_count: number;
};

export type Catalog = {
  catalog_id: string; status: string; model_version: string;
  calibration_status: string; feature_dim: number; row_count: number;
  source_batch_id: string | null; created_at: string;
};

export type QueryMatch = {
  individual_id: string; individual_name: string; observation_id: string;
  representative_media_url: string; score: number; support_frames: number;
  side: string; cross_side: boolean; quality: number;
  catalog_id: string; model_version: string;
  calibration_status: string;
};

export type QueryResult = {
  query_request_id: string;
  status: "queued" | "running" | "failed" | "succeeded";
  job_id?: string; error?: string | null; catalog_id?: string;
  model_version?: string; calibration_status?: string;
  human_review_status?: "candidate_only";
  images?: Array<{
    query_image_id: string; original_relative_path: string;
  }>;
  detections?: Array<{
    query_image_id: string; crop_index: number;
    bbox: [number, number, number, number]; quality: number;
    matches: QueryMatch[];
  }>;
};

type UploadStatus = {
  session_id: string;
  state: string;
  chunk_size: number;
  files: Array<{
    file_id: string;
    relative_path: string;
    state: string;
    received_parts: number[];
    missing_parts: number[];
  }>;
};

const CSRF_KEY = "whitewhale-csrf";

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json() as { detail?: string };
      message = body.detail ?? message;
    } catch {
      // Keep the HTTP status when the response body is not JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

function writeHeaders(): HeadersInit {
  const csrf = sessionStorage.getItem(CSRF_KEY);
  return csrf ? { "X-CSRF-Token": csrf } : {};
}

export async function login(username: string, password: string): Promise<User> {
  const result = await parse<{ csrf_token: string }>(await fetch("/api/auth/login", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  }));
  sessionStorage.setItem(CSRF_KEY, result.csrf_token);
  return me();
}

export async function bootstrapStatus(): Promise<{ open: boolean }> {
  return parse(await fetch("/api/auth/bootstrap-status"));
}

export async function bootstrapAdmin(username: string, password: string): Promise<void> {
  await parse(await fetch("/api/auth/bootstrap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  }));
}

export async function me(): Promise<User> {
  return parse(await fetch("/api/auth/me", { credentials: "include" }));
}

export async function deployment(): Promise<Deployment> {
  return parse(await fetch("/api/system/deployment", { credentials: "include" }));
}

export async function logout(): Promise<void> {
  await parse(await fetch("/api/auth/logout", {
    method: "POST",
    credentials: "include",
    headers: writeHeaders()
  }));
  sessionStorage.removeItem(CSRF_KEY);
}

export async function createUpload(
  batchName: string,
  sourceFormat: "idolphin" | "generic",
  files: ManifestFile[]
): Promise<{ session_id: string }> {
  return parse(await fetch("/api/uploads", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", ...writeHeaders() },
    body: JSON.stringify({ batch_name: batchName, source_format: sourceFormat, files })
  }));
}

export async function getUploadStatus(sessionId: string): Promise<UploadStatus> {
  return parse(await fetch(`/api/uploads/${sessionId}`, { credentials: "include" }));
}

export async function putPart(
  sessionId: string,
  fileId: string,
  partNumber: number,
  data: Blob,
  sha256: string
): Promise<void> {
  await parse(await fetch(
    `/api/uploads/${sessionId}/files/${fileId}/parts/${partNumber}`,
    {
      method: "PUT",
      credentials: "include",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Content-SHA256": sha256,
        ...writeHeaders()
      },
      body: data
    }
  ));
}

export async function completeFile(sessionId: string, fileId: string): Promise<void> {
  await parse(await fetch(`/api/uploads/${sessionId}/files/${fileId}/complete`, {
    method: "POST",
    credentials: "include",
    headers: writeHeaders()
  }));
}

export async function completeSession(sessionId: string): Promise<void> {
  await parse(await fetch(`/api/uploads/${sessionId}/complete`, {
    method: "POST",
    credentials: "include",
    headers: writeHeaders()
  }));
}

export async function importSession(
  sessionId: string,
  capturedOn?: string
): Promise<{ batch_id: string }> {
  const query = capturedOn ? `?captured_on=${encodeURIComponent(capturedOn)}` : "";
  return parse(await fetch(`/api/uploads/${sessionId}/import${query}`, {
    method: "POST",
    credentials: "include",
    headers: writeHeaders()
  }));
}

export async function dispatchQuery(
  sessionId: string,
  request: {
    k: number; detector_version: string; required_vram_mb: number;
  }
): Promise<{ query_request_id: string }> {
  return parse(await fetch(`/api/uploads/${sessionId}/query`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", ...writeHeaders() },
    body: JSON.stringify(request)
  }));
}

export async function getQueryResult(queryId: string): Promise<QueryResult> {
  return parse(await fetch(`/api/query/requests/${queryId}`, {
    credentials: "include"
  }));
}

export async function listBatches(): Promise<BatchSummary[]> {
  return parse(await fetch("/api/batches", { credentials: "include" }));
}

export async function getDashboard(): Promise<Dashboard> {
  return parse(await fetch("/api/dashboard", { credentials: "include" }));
}

export async function getBatch(batchId: string): Promise<BatchDetail> {
  return parse(await fetch(`/api/batches/${batchId}`, { credentials: "include" }));
}

export async function dispatchArchivalJob(
  batchId: string,
  request: {
    model_version: string;
    detector_version: string;
    preprocess_id: string;
    pipeline_config: Record<string, unknown>;
    required_vram_mb: number;
    max_attempts: number;
  }
): Promise<{ job_id: string }> {
  return parse(await fetch(`/api/batches/${batchId}/archive-jobs`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", ...writeHeaders() },
    body: JSON.stringify(request)
  }));
}

export async function listJobs(): Promise<JobSummary[]> {
  return parse(await fetch("/api/jobs", { credentials: "include" }));
}

export async function listWorkers(): Promise<WorkerSummary[]> {
  return parse(await fetch("/api/workers", { credentials: "include" }));
}

export async function revokeWorker(deviceId: string): Promise<void> {
  await parse(await fetch(`/api/workers/${deviceId}/revoke`, {
    method: "POST", credentials: "include", headers: writeHeaders()
  }));
}

export async function listUsers(): Promise<UserSummary[]> {
  return parse(await fetch("/api/users", { credentials: "include" }));
}

export async function listAuditEvents(): Promise<AuditEvent[]> {
  return parse(await fetch("/api/system/audit", { credentials: "include" }));
}

export async function reviewInbox(): Promise<ReviewTask[]> {
  return parse(await fetch("/api/reviews/inbox", { credentials: "include" }));
}

export async function getCandidate(clusterId: string): Promise<Candidate> {
  return parse(await fetch(`/api/candidates/${clusterId}`, { credentials: "include" }));
}

export async function getCooccurrence(eventId: string): Promise<Cooccurrence> {
  return parse(await fetch(`/api/cooccurrences/${eventId}`, { credentials: "include" }));
}

export async function getIdentityChange(proposalId: string): Promise<IdentityChange> {
  return parse(await fetch(`/api/identity-changes/${proposalId}`, {
    credentials: "include"
  }));
}

export async function submitVote(
  taskId: string, choice: string, individualId?: string
): Promise<ReviewDecision> {
  return parse(await fetch(`/api/reviews/tasks/${taskId}/votes`, {
    method: "POST", credentials: "include",
    headers: { "Content-Type": "application/json", ...writeHeaders() },
    body: JSON.stringify({ choice, individual_id: individualId || null })
  }));
}

export async function applyMultiTargetReview(taskId: string): Promise<{ event_id: string }> {
  return parse(await fetch(`/api/reviews/tasks/${taskId}/apply-multi-target`, {
    method: "POST", credentials: "include", headers: writeHeaders()
  }));
}

export async function applyIdentityChange(taskId: string): Promise<{ proposal_id: string; status: string }> {
  return parse(await fetch(`/api/reviews/tasks/${taskId}/apply-identity-change`, {
    method: "POST", credentials: "include", headers: writeHeaders()
  }));
}

export async function listIndividuals(): Promise<Individual[]> {
  return parse(await fetch("/api/individuals", { credentials: "include" }));
}

export async function listRelationships(): Promise<Relationship[]> {
  return parse(await fetch("/api/relationships", { credentials: "include" }));
}

export async function listDatasets(): Promise<DatasetSummary[]> {
  return parse(await fetch("/api/datasets", { credentials: "include" }));
}

export async function listTrainingRuns(): Promise<TrainingRunSummary[]> {
  return parse(await fetch("/api/training-runs", { credentials: "include" }));
}

export async function listModels(): Promise<ModelSummary[]> {
  return parse(await fetch("/api/models", { credentials: "include" }));
}

export async function requestModelPromotion(modelId: string): Promise<{ catalog_rebuild_job_id: string | null }> {
  return parse(await fetch(`/api/models/${modelId}/request-promotion`, {
    method: "POST", credentials: "include", headers: writeHeaders()
  }));
}

export async function listCatalogs(): Promise<Catalog[]> {
  return parse(await fetch("/api/catalogs", { credentials: "include" }));
}

export async function activateCatalog(catalogId: string): Promise<Catalog> {
  return parse(await fetch(`/api/catalogs/${catalogId}/activate`, {
    method: "POST", credentials: "include", headers: writeHeaders()
  }));
}
