import { FormEvent, useEffect, useMemo, useState } from "react";
import { sha256 } from "@noble/hashes/sha256";
import { bytesToHex } from "@noble/hashes/utils";
import {
  completeFile,
  completeSession,
  bootstrapAdmin,
  bootstrapStatus,
  createUpload,
  deployment,
  applyMultiTargetReview,
  applyIdentityChange,
  getUploadStatus,
  getDashboard,
  getBatch,
  getJob,
  dispatchArchivalJob,
  dispatchQuery,
  importSession,
  activateCatalog,
  getCandidate,
  getCooccurrence,
  getIdentityChange,
  getModel,
  getQueryResult,
  listBatches,
  listCatalogs,
  listIndividuals,
  listDatasets,
  listModels,
  listJobs,
  listWorkers,
  listUsers,
  listAuditEvents,
  listRelationships,
  listTrainingRuns,
  login,
  logout,
  me,
  putPart,
  requestModelPromotion,
  rollbackModel,
  revokeWorker,
  reviewInbox,
  submitVote,
  type BatchSummary,
  type BatchDetail,
  type Dashboard,
  type JobSummary,
  type JobDetail,
  type WorkerSummary,
  type UserSummary,
  type AuditEvent,
  type Candidate,
  type Catalog,
  type Cooccurrence,
  type DatasetSummary,
  type Deployment,
  type Individual,
  type IdentityChange,
  type ManifestFile,
  type ModelSummary,
  type ModelDetail,
  type QueryResult,
  type Relationship,
  type ReviewTask,
  type TrainingRunSummary,
  type User
} from "./api";

type Page = "overview" | "query" | "upload" | "batches" | "reviews" | "individuals"
  | "relationships" | "training" | "catalogs" | "workers" | "system";

type Progress = {
  stage: string;
  doneBytes: number;
  totalBytes: number;
};

async function hashBlob(blob: Blob): Promise<string> {
  const hasher = sha256.create();
  const step = 8 * 1024 * 1024;
  for (let offset = 0; offset < blob.size; offset += step) {
    hasher.update(new Uint8Array(await blob.slice(offset, offset + step).arrayBuffer()));
  }
  return bytesToHex(hasher.digest());
}

async function resumeKey(batchName: string, manifest: ManifestFile[]): Promise<string> {
  const text = JSON.stringify({ batchName, manifest });
  return `whitewhale-upload-${bytesToHex(sha256(new TextEncoder().encode(text)))}`;
}

async function uploadFiles(
  files: File[],
  batchName: string,
  sourceFormat: "idolphin" | "generic",
  setProgress: (progress: Progress) => void
): Promise<{ sessionId: string; storageKey: string }> {
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  const manifest: ManifestFile[] = [];
  let hashed = 0;
  for (const file of files) {
    setProgress({
      stage: `正在计算哈希：${file.name}`,
      doneBytes: hashed,
      totalBytes
    });
    manifest.push({
      relative_path: file.webkitRelativePath || file.name,
      size_bytes: file.size,
      sha256: await hashBlob(file)
    });
    hashed += file.size;
  }
  const storageKey = await resumeKey(batchName, manifest);
  let sessionId = localStorage.getItem(storageKey);
  if (sessionId) {
    try {
      await getUploadStatus(sessionId);
    } catch {
      localStorage.removeItem(storageKey);
      sessionId = null;
    }
  }
  if (!sessionId) {
    sessionId = (await createUpload(batchName, sourceFormat, manifest)).session_id;
    localStorage.setItem(storageKey, sessionId);
  }

  let uploaded = 0;
  const status = await getUploadStatus(sessionId);
  const byPath = new Map(status.files.map((file) => [file.relative_path, file]));
  for (const browserFile of files) {
    const path = browserFile.webkitRelativePath || browserFile.name;
    const remote = byPath.get(path);
    if (!remote) throw new Error(`服务器清单中缺少 ${path}`);
    for (const partNumber of remote.missing_parts) {
      const startOffset = partNumber * status.chunk_size;
      const part = browserFile.slice(startOffset, startOffset + status.chunk_size);
      setProgress({ stage: `正在上传：${path}`, doneBytes: uploaded, totalBytes });
      await putPart(sessionId, remote.file_id, partNumber, part, await hashBlob(part));
      uploaded += part.size;
    }
    if (remote.state !== "complete") await completeFile(sessionId, remote.file_id);
  }
  await completeSession(sessionId);
  return { sessionId, storageKey };
}

function Login({ onLogin }: { onLogin: (user: User) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [bootstrapOpen, setBootstrapOpen] = useState(false);
  useEffect(() => {
    bootstrapStatus().then((value) => setBootstrapOpen(value.open)).catch(() => undefined);
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (bootstrapOpen) await bootstrapAdmin(username, password);
      onLogin(await login(username, password));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登录失败");
    } finally {
      setBusy(false);
    }
  }

  return <main className="login-shell">
    <section className="login-card">
      <div className="mark">WW</div>
      <p className="eyebrow">RESEARCH CONTROL PLANE</p>
      <h1>中华白海豚<br />识别归档平台</h1>
      <p className="muted">{bootstrapOpen ? "尚无管理员：本次将创建唯一的首位管理员。密码至少 12 个字符。" : "数据留在课题组服务器，算法结果保持候选状态，正式身份由人工审核决定。"}</p>
      <form onSubmit={submit}>
        <label>用户名<input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" /></label>
        <label>密码<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></label>
        {error && <p className="error">{error}</p>}
        <button disabled={busy}>{busy ? (bootstrapOpen ? "正在初始化…" : "正在登录…") : (bootstrapOpen ? "创建管理员并登录" : "登录")}</button>
      </form>
    </section>
  </main>;
}

function UploadPanel() {
  const [files, setFiles] = useState<File[]>([]);
  const [batchName, setBatchName] = useState("");
  const [sourceFormat, setSourceFormat] = useState<"idolphin" | "generic">("idolphin");
  const [capturedOn, setCapturedOn] = useState("");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const totalBytes = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files]);

  async function start() {
    if (!files.length || !batchName.trim()) return;
    if (sourceFormat === "generic" && !capturedOn) {
      setError("普通目录必须填写拍摄日期");
      return;
    }
    setError("");
    setMessage("");
    try {
      const { sessionId, storageKey } = await uploadFiles(
        files, batchName, sourceFormat, setProgress);
      const imported = await importSession(
        sessionId, sourceFormat === "generic" ? capturedOn : undefined);
      localStorage.removeItem(storageKey);
      setProgress({ stage: "导入完成", doneBytes: totalBytes, totalBytes });
      setMessage(`批次已登记：${imported.batch_id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "上传失败");
    }
  }

  const ratio = progress && progress.totalBytes
    ? Math.round(progress.doneBytes / progress.totalBytes * 100)
    : 0;

  return <section className="panel upload-panel">
    <div className="panel-heading">
      <div><p className="eyebrow">BATCH INGEST</p><h2>上传调查目录</h2></div>
      <span className="pill">支持断点续传</span>
    </div>
    <div className="form-grid">
      <label>批次名称<input value={batchName} onChange={(e) => setBatchName(e.target.value)} placeholder="例如 20140419 02" /></label>
      <label>目录结构<select value={sourceFormat} onChange={(e) => setSourceFormat(e.target.value as "idolphin" | "generic")}><option value="idolphin">标准 iDolphin</option><option value="generic">普通图片目录</option></select></label>
      {sourceFormat === "generic" && <label>拍摄日期<input type="date" value={capturedOn} onChange={(e) => setCapturedOn(e.target.value)} /></label>}
    </div>
    <label className="dropzone">
      <input type="file" multiple {...{ webkitdirectory: "", directory: "" }} onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
      <strong>{files.length ? `已选择 ${files.length} 个文件` : "选择一个完整文件夹"}</strong>
      <span>{files.length ? `${(totalBytes / 1024 / 1024).toFixed(1)} MiB` : "保留原始相对路径，不会修改源文件"}</span>
    </label>
    {progress && <div className="progress"><div><span>{progress.stage}</span><strong>{ratio}%</strong></div><progress max="100" value={ratio} /></div>}
    {message && <p className="success">{message}</p>}
    {error && <p className="error">{error}</p>}
    <button onClick={start} disabled={!files.length || !batchName || Boolean(progress && ratio < 100)}>校验并上传</button>
  </section>;
}

function QueryPanel() {
  const [files, setFiles] = useState<File[]>([]);
  const [batchName, setBatchName] = useState("");
  const [topK, setTopK] = useState(5);
  const [detectorVersion, setDetectorVersion] = useState("legacy-yolo-v2");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [previews, setPreviews] = useState<Map<string, string>>(new Map());
  const totalBytes = useMemo(
    () => files.reduce((sum, file) => sum + file.size, 0), [files]);

  useEffect(() => {
    const next = new Map<string, string>();
    for (const file of files) {
      next.set(file.webkitRelativePath || file.name, URL.createObjectURL(file));
    }
    setPreviews(next);
    return () => next.forEach((url) => URL.revokeObjectURL(url));
  }, [files]);

  async function waitForResult(queryId: string): Promise<void> {
    for (;;) {
      const next = await getQueryResult(queryId);
      setResult(next);
      if (next.status === "succeeded" || next.status === "failed") return;
      setProgress({
        stage: next.status === "queued" ? "等待可用 GPU Worker" : "GPU 正在检测与提取特征",
        doneBytes: totalBytes,
        totalBytes
      });
      await new Promise((resolve) => window.setTimeout(resolve, 3000));
    }
  }

  async function start() {
    if (!files.length || !batchName.trim()) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const { sessionId, storageKey } = await uploadFiles(
        files, batchName, "generic", setProgress);
      const dispatched = await dispatchQuery(sessionId, {
        k: topK,
        detector_version: detectorVersion,
        required_vram_mb: 4096
      });
      localStorage.removeItem(storageKey);
      await waitForResult(dispatched.query_request_id);
      setProgress({ stage: "查询完成", doneBytes: totalBytes, totalBytes });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "查询失败");
    } finally {
      setBusy(false);
    }
  }

  const imagePaths = new Map(
    (result?.images ?? []).map((item) => [
      item.query_image_id, item.original_relative_path
    ]));
  const ratio = progress && progress.totalBytes
    ? Math.round(progress.doneBytes / progress.totalBytes * 100)
    : 0;

  return <section className="panel query-panel">
    <div className="panel-heading">
      <div><p className="eyebrow">BATCH QUERY</p><h2>单图 / 文件夹识别</h2></div>
      <span className="pill">批内检测 → 历史库 Top-K</span>
    </div>
    <div className="notice inline-notice"><strong>结果仅为候选</strong><span>系统不会自动合并身份；跨时间个体归档仍需三名审核人独立一致。</span></div>
    <div className="form-grid">
      <label>查询名称<input value={batchName} onChange={(event) => setBatchName(event.target.value)} placeholder="例如 2026-08-31 航次查询" /></label>
      <label>返回候选数<input type="number" min="1" max="100" value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></label>
      <label>Detector 版本<input value={detectorVersion} onChange={(event) => setDetectorVersion(event.target.value)} /></label>
    </div>
    <div className="query-pickers">
      <label className="dropzone">
        <input type="file" accept="image/*" multiple onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
        <strong>选择一张或多张图片</strong><span>适合临时快速识别</span>
      </label>
      <label className="dropzone">
        <input type="file" accept="image/*" multiple {...{ webkitdirectory: "", directory: "" }} onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
        <strong>选择完整文件夹</strong><span>保留原始数据库相对路径</span>
      </label>
    </div>
    {files.length > 0 && <p className="muted">已选择 {files.length} 张图片 · {(totalBytes / 1024 / 1024).toFixed(1)} MiB</p>}
    {progress && <div className="progress"><div><span>{progress.stage}</span><strong>{ratio}%</strong></div><progress max="100" value={ratio} /></div>}
    {error && <p className="error">{error}</p>}
    {result?.status === "failed" && <p className="error">{result.error || "GPU 查询任务失败"}</p>}
    <button onClick={start} disabled={busy || !files.length || !batchName.trim() || topK < 1 || topK > 100}>{busy ? "正在处理…" : "上传并开始识别"}</button>

    {result?.status === "succeeded" && <div className="query-results">
      <div className="query-summary"><strong>{result.detections?.length ?? 0} 个目标</strong><span>Model {result.model_version} · Catalog {result.catalog_id?.slice(0, 12)}… · {result.calibration_status}</span></div>
      {(result.detections?.length ?? 0) === 0 && <p className="muted">本批图片中没有检测到可查询的海豚目标。</p>}
      {result.detections?.map((detection) => {
        const path = imagePaths.get(detection.query_image_id) ?? detection.query_image_id;
        return <article className="query-detection" key={`${detection.query_image_id}-${detection.crop_index}`}>
          <div className="query-source">
            {previews.get(path) && <img src={previews.get(path)} alt={path} />}
            <div><strong>{path}</strong><span>目标 #{detection.crop_index + 1} · 检测置信度 {detection.quality.toFixed(3)}</span></div>
          </div>
          <div className="match-grid">{detection.matches.map((match, index) => <div className="match-card" key={match.observation_id}>
            <img src={match.representative_media_url} alt={match.individual_name} />
            <div><strong>#{index + 1} {match.individual_name}</strong><span>相似度 {match.score.toFixed(4)} · {match.support_frames} 帧支持</span><small>{match.side || "侧别未知"}{match.cross_side ? " · 跨侧证据" : ""} · 质量 {match.quality.toFixed(2)}</small></div>
          </div>)}</div>
          {detection.matches.length === 0 && <p className="muted">历史 Catalog 中没有可返回的候选。</p>}
        </article>;
      })}
    </div>}
  </section>;
}

function OverviewPanel() {
  const [summary, setSummary] = useState<Dashboard | null>(null);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([getDashboard(), listJobs()])
      .then(([dashboard, rows]) => { setSummary(dashboard); setJobs(rows.slice(0, 8)); })
      .catch((reason) => setError(String(reason)));
  }, []);
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">OPERATIONS OVERVIEW</p><h2>控制面总览</h2></div></div>
    {error && <p className="error">{error}</p>}
    {summary && <div className="card-grid overview-grid">
      <article className="individual-card"><strong>{Object.values(summary.batch_counts).reduce((a, b) => a + b, 0)}</strong><span>批次总数</span></article>
      <article className="individual-card"><strong>{summary.queued_jobs}</strong><span>排队任务</span></article>
      <article className="individual-card"><strong>{summary.pending_reviews}</strong><span>待审核</span></article>
      <article className="individual-card"><strong>{summary.worker_counts.online}/{summary.worker_counts.total}</strong><span>Worker 在线</span></article>
      <article className="individual-card"><strong>{summary.failed_jobs}</strong><span>异常终态任务</span></article>
    </div>}
    <h3 className="subheading">最近任务</h3>
    <div className="data-list clickable-list">{jobs.map((job) => <article key={job.job_id} onClick={() => getJob(job.job_id).then(setJobDetail).catch((reason) => setError(String(reason)))}><div><strong>{job.task_type}</strong><span>{job.job_id.slice(0, 12)}… · {new Date(job.created_at).toLocaleString()}</span>{job.last_error && <small className="error">{job.last_error}</small>}</div><div className="metrics"><b>{job.attempt_count} 次尝试</b><b>{job.artifact_count} 个产物</b><span className="pill">{job.state}</span></div></article>)}</div>
    {jobDetail && <div className="review-detail detail-block"><div className="panel-heading"><h3>Job {jobDetail.job_id}</h3><span className="pill">{jobDetail.state}</span></div><p className="muted">{jobDetail.task_type} · 最低显存 {jobDetail.required_vram_mb} MiB · 模型 {jobDetail.required_model_version ?? "不限定"}</p><h4>Attempt</h4><div className="data-list compact">{jobDetail.attempts.map((attempt) => <article key={attempt.attempt_id}><div><strong>#{attempt.attempt_number} · {attempt.outcome ?? "pending"}</strong><span>Worker {attempt.worker_device_id ?? "未分配"}</span>{attempt.error_detail && <small className="error">{attempt.error_detail}</small>}</div></article>)}</div><h4>Artifact</h4><div className="data-list compact">{jobDetail.artifacts.map((artifact) => <article key={artifact.artifact_id}><div><strong>{artifact.artifact_type}</strong><span>{(artifact.size_bytes / 1024 / 1024).toFixed(2)} MiB · SHA-256 {artifact.sha256.slice(0, 16)}…</span><small>{artifact.model_version ?? "无模型版本"} · schema {artifact.schema_version ?? "-"}</small></div></article>)}</div><h4>输入 Manifest</h4><pre className="plan-preview">{JSON.stringify(jobDetail.input_manifest, null, 2)}</pre></div>}
  </section>;
}

function BatchesPanel({ canDispatch }: { canDispatch: boolean }) {
  const [items, setItems] = useState<BatchSummary[]>([]);
  const [detail, setDetail] = useState<BatchDetail | null>(null);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelVersion, setModelVersion] = useState("");
  const [detectorVersion, setDetectorVersion] = useState("legacy-yolo-v2");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    listBatches().then(setItems).catch((e) => setError(String(e)));
    listModels().then((rows) => {
      const reid = rows.filter((item) => item.feature_dim !== null);
      setModels(reid);
      const production = reid.find((item) => item.status === "production") ?? reid[0];
      if (production) setModelVersion(production.version);
    }).catch((e) => setError(String(e)));
  }, []);
  async function dispatch() {
    if (!detail || !modelVersion) return;
    const model = models.find((item) => item.version === modelVersion);
    try {
      const result = await dispatchArchivalJob(detail.batch_id, {
        model_version: modelVersion,
        detector_version: detectorVersion,
        preprocess_id: model?.preprocess_id ?? "unknown",
        pipeline_config: { source: "web", identity_policy: "human_review_required" },
        required_vram_mb: 4096,
        max_attempts: 3
      });
      setMessage(`GPU 归档任务已排队：${result.job_id}`);
      setDetail(await getBatch(detail.batch_id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务创建失败");
    }
  }
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">BATCHES</p><h2>批次进度</h2></div></div>
    {error && <p className="error">{error}</p>}
    {message && <p className="success">{message}</p>}
    <div className="data-list">{items.map((item) => <article key={item.batch_id} onClick={() => getBatch(item.batch_id).then(setDetail).catch((e) => setError(String(e)))}><div><strong>{item.name}</strong><span>{item.source_format} · {new Date(item.created_at).toLocaleString()}</span></div><div className="metrics"><b>{item.image_count} 图</b><b>{item.crop_count} Crop</b><b>{item.cluster_count} 簇</b><span className="pill">{item.stage}</span></div></article>)}</div>
    {detail && <div className="review-detail"><h3>{detail.name} · Manifest</h3><p className="muted">SHA-256 {detail.manifest_sha256}</p><pre className="plan-preview">{JSON.stringify(detail.metadata, null, 2)}</pre>{canDispatch && detail.stage === "registered" && <><h3 className="subheading">提交到成员 GPU</h3><div className="form-grid"><label>Re-ID 模型<select value={modelVersion} onChange={(e) => setModelVersion(e.target.value)}>{models.map((item) => <option key={item.model_version_id} value={item.version}>{item.version} · {item.status}</option>)}</select></label><label>Detector 版本<input value={detectorVersion} onChange={(e) => setDetectorVersion(e.target.value)} /></label></div><button onClick={dispatch} disabled={!modelVersion || !detectorVersion}>创建批内归档任务</button></>}<h3 className="subheading">Job / Attempt 入口</h3><div className="data-list compact">{detail.jobs.map((job) => <article key={job.job_id}><div><strong>{job.task_type}</strong><span>{job.job_id}</span></div><span className="pill">{job.state}</span></article>)}</div></div>}
  </section>;
}

function WorkersPanel({ canManage }: { canManage: boolean }) {
  const [items, setItems] = useState<WorkerSummary[]>([]);
  const [message, setMessage] = useState("");
  const refresh = () => listWorkers().then(setItems).catch((reason) => setMessage(String(reason)));
  useEffect(() => { void refresh(); }, []);
  async function revoke(deviceId: string) {
    await revokeWorker(deviceId);
    setMessage("设备令牌已撤销；该 Worker 不能再领取任务。");
    await refresh();
  }
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">GPU WORKERS</p><h2>计算设备</h2></div><span className="pill">训练不在服务器运行</span></div>{message && <p className={message.includes("撤销") ? "success" : "error"}>{message}</p>}<div className="data-list">{items.map((item) => <article key={item.device_id}><div><strong>{item.name}</strong><span>{item.gpu_model} · {item.vram_mb} MiB · CUDA {item.cuda_version}</span><small>{item.capabilities.join(" · ")} · {item.attempt_count} 次任务</small></div><div className="metrics"><b>{item.current_job_id ? `任务 ${item.current_job_id.slice(0, 8)}…` : "空闲"}</b><span className="pill">{item.is_online ? "online" : item.is_active ? "offline" : "revoked"}</span>{canManage && item.is_active && <button className="danger" onClick={() => revoke(item.device_id)}>撤销令牌</button>}</div></article>)}</div></section>;
}

function SystemPanel({ release }: { release: Deployment | null }) {
  const [users, setUsers] = useState<UserSummary[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([listUsers(), listAuditEvents()])
      .then(([userRows, eventRows]) => { setUsers(userRows); setEvents(eventRows); })
      .catch((reason) => setError(String(reason)));
  }, []);
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">SYSTEM & AUDIT</p><h2>系统管理</h2></div></div>{error && <p className="error">{error}</p>}
    {release && <div className="notice inline-notice"><strong>{release.status ?? "deployed"}</strong><span>{release.branch} · {release.commit} · {release.deployed_at}{release.failure_reason ? ` · ${release.failure_reason}` : ""}</span></div>}
    <h3 className="subheading">账号与角色</h3><div className="data-list compact">{users.map((item) => <article key={item.user_id}><div><strong>{item.username}</strong><span>{item.roles.join(" · ")}</span></div><span className="pill">{item.is_active ? "active" : "disabled"}</span></article>)}</div>
    <h3 className="subheading">不可变审计</h3><div className="data-list compact">{events.map((item) => <article key={item.audit_event_id}><div><strong>{item.event_type}</strong><span>{item.actor_type} · {item.target_type ?? "-"} {item.target_id?.slice(0, 18) ?? ""}</span></div><small>{new Date(item.occurred_at).toLocaleString()}</small></article>)}</div>
  </section>;
}

function ReviewPanel() {
  const [tasks, setTasks] = useState<ReviewTask[]>([]);
  const [candidate, setCandidate] = useState<Candidate | null>(null);
  const [cooccurrence, setCooccurrence] = useState<Cooccurrence | null>(null);
  const [identityChange, setIdentityChange] = useState<IdentityChange | null>(null);
  const [selected, setSelected] = useState<ReviewTask | null>(null);
  const [existingId, setExistingId] = useState("");
  const [message, setMessage] = useState("");
  const refresh = () => reviewInbox().then(setTasks).catch((e) => setMessage(String(e)));
  useEffect(() => { void refresh(); }, []);
  async function open(task: ReviewTask) {
    setSelected(task);
    setCandidate(null);
    setCooccurrence(null);
    setIdentityChange(null);
    if (task.subject_type === "cooccurrence_event") {
      setCooccurrence(await getCooccurrence(task.subject_id));
    } else if (task.subject_type === "identity_change_proposal") {
      setIdentityChange(await getIdentityChange(task.subject_id));
    } else {
      setCandidate(await getCandidate(task.subject_id));
    }
  }
  async function vote(choice: string) {
    if (!selected) return;
    const decision = await submitVote(
      selected.task_id, choice, choice === "existing" ? existingId : undefined);
    if (selected.task_type === "multi_target" && decision.status !== "pending") {
      await applyMultiTargetReview(selected.task_id);
    } else if (selected.subject_type === "identity_change_proposal"
        && decision.status !== "pending") {
      await applyIdentityChange(selected.task_id);
    }
    setMessage("本次投票已追加保存；其他审核人的未完成票仍不可见。");
    setSelected(null); setCandidate(null); setCooccurrence(null);
    setIdentityChange(null); await refresh();
  }
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">BLIND REVIEW</p><h2>独立审核中心</h2></div><span className="pill">{tasks.length} 待处理</span></div>
    {message && <p className="success">{message}</p>}
    <div className="review-layout"><div className="data-list compact">{tasks.map((task) => <button className="task-row" key={task.task_id} onClick={() => open(task)}><strong>{task.task_type === "cluster_purity" ? "批内簇纯度" : task.task_type === "multi_target" ? "多目标真实性" : task.task_type === "identity_merge" ? "身份合并" : task.task_type === "identity_split" ? "身份拆分" : task.task_type === "observation_withdrawal" ? "撤回照片" : "历史身份匹配"}</strong><span>{task.subject_id.slice(0, 12)}…</span></button>)}</div>
      {candidate && selected && <div className="review-detail"><h3>{candidate.label}</h3><div className="crop-grid">{candidate.crops.map((crop) => <img key={crop.crop_id} src={crop.media_url} alt="海豚 Crop" />)}</div>
        {candidate.matches.length > 0 && <div className="matches">{candidate.matches.map((match) => <label key={match.individual_id}><input type="radio" name="existing" onChange={() => setExistingId(match.individual_id)} />#{match.rank} {match.individual_id.slice(0, 8)} · {match.score.toFixed(3)} · {match.support_frames} 帧</label>)}</div>}
        <div className="actions">{selected.task_type === "cluster_purity" ? <><button onClick={() => vote("confirm_cluster")}>确认候选组</button><button className="secondary" onClick={() => vote("split_required")}>需要拆簇</button><button className="danger" onClick={() => vote("unusable")}>不可用</button></> : <><button disabled={!existingId} onClick={() => vote("existing")}>选择 Existing</button><button className="secondary" onClick={() => vote("new")}>选择 New</button><button className="quiet-dark" onClick={() => vote("uncertain")}>不确定</button></>}</div>
      </div>}
      {cooccurrence && selected && <div className="review-detail"><h3>是否确为同一张原图中的多个个体？</h3><p className="muted">只确认多目标事实，不判断亲缘关系。三个审核人的投票彼此独立。</p><div className="crop-grid">{cooccurrence.crops.map((crop) => <img key={crop.crop_id} src={crop.media_url} alt={`目标 ${crop.crop_index + 1}`} />)}</div><div className="actions"><button onClick={() => vote("confirm_multi_target")}>确认多目标</button><button className="danger" onClick={() => vote("reject")}>驳回</button><button className="quiet-dark" onClick={() => vote("uncertain")}>不确定</button></div></div>}
      {identityChange && selected && <div className="review-detail"><h3>{identityChange.change_type === "merge" ? "身份合并方案" : identityChange.change_type === "split" ? "身份拆分方案" : "照片撤回方案"}</h3><p className="muted">方案摘要 {identityChange.plan_digest.slice(0, 16)}…；必须三人选择完全相同的结论，任何分歧都不会修改正式身份。</p><pre className="plan-preview">{JSON.stringify(identityChange.plan, null, 2)}</pre><div className="actions"><button onClick={() => vote("approve_change")}>批准此方案</button><button className="danger" onClick={() => vote("reject")}>驳回</button><button className="quiet-dark" onClick={() => vote("uncertain")}>不确定</button></div></div>}</div>
  </section>;
}

function IndividualsPanel() {
  const [items, setItems] = useState<Individual[]>([]);
  useEffect(() => { void listIndividuals().then(setItems); }, []);
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">CONFIRMED INDIVIDUALS</p><h2>正式个体目录</h2></div><span className="pill">{items.length} 个体</span></div><div className="card-grid">{items.map((item) => <article className="individual-card" key={item.individual_id}><div className="avatar">{item.display_name.slice(-2)}</div><strong>{item.display_name}</strong><span>{item.observation_count} 次观测 · {item.state}</span>{item.flags.map((flag) => <small key={flag}>{flag}</small>)}</article>)}</div></section>;
}

function CatalogPanel() {
  const [items, setItems] = useState<Catalog[]>([]);
  const refresh = () => listCatalogs().then(setItems);
  useEffect(() => { void refresh(); }, []);
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">IMMUTABLE CATALOG</p><h2>目录版本</h2></div></div><div className="data-list">{items.map((item) => <article key={item.catalog_id}><div><strong>{item.catalog_id.slice(0, 12)}…</strong><span>{item.model_version} · {item.feature_dim} 维 · {item.row_count} 行</span></div><div className="metrics"><span className="pill">{item.status}</span><small>{item.calibration_status}</small>{item.status !== "active" && <button onClick={() => activateCatalog(item.catalog_id).then(refresh)}>校验并激活</button>}</div></article>)}</div></section>;
}

function RelationshipsPanel() {
  const [items, setItems] = useState<Relationship[]>([]);
  const [error, setError] = useState("");
  useEffect(() => { listRelationships().then(setItems).catch((e) => setError(String(e))); }, []);
  const typeName = (value: Relationship["relationship_type"]) => ({
    co_occurrence: "同框共现", repeated_association: "重复伴随", suspected_kinship: "疑似亲缘"
  })[value];
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">RELATIONSHIP EVIDENCE</p><h2>关系证据集合</h2></div><span className="pill">{items.length} 条假设</span></div><div className="notice inline-notice"><strong>不是亲缘结论</strong><span>这里仅展示经审核的同框证据与待验证假设；系统不会自动写入“已确认亲缘”。</span></div>{error && <p className="error">{error}</p>}<div className="data-list">{items.map((item) => <article key={item.hypothesis_id}><div><strong>{item.individual_low_name} ↔ {item.individual_high_name}</strong><span>{typeName(item.relationship_type)} · {new Date(item.created_at).toLocaleString()}</span></div><div className="metrics"><b>{item.evidence_count} 条证据</b><span className="pill">{item.status}</span></div></article>)}</div></section>;
}

function TrainingPanel() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelDetail, setModelDetail] = useState<ModelDetail | null>(null);
  const [message, setMessage] = useState("");
  const refresh = () => Promise.all([
    listDatasets().then(setDatasets),
    listTrainingRuns().then(setRuns),
    listModels().then(setModels)
  ]).catch((reason) => setMessage(String(reason)));
  useEffect(() => { void refresh(); }, []);
  async function promote(modelId: string) {
    try {
      const result = await requestModelPromotion(modelId);
      setMessage(result.catalog_rebuild_job_id
        ? `上线门禁已通过，Catalog 重建任务：${result.catalog_rebuild_job_id}`
        : "上线门禁已通过，Detector Production 指针已切换。");
      await refresh();
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "上线请求失败");
    }
  }
  async function rollback(modelId: string) {
    try {
      await rollbackModel(modelId);
      setMessage("模型已回滚为 Production；操作已写入不可变事件和审计。");
      await refresh();
      setModelDetail(await getModel(modelId));
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "模型回滚失败");
    }
  }
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">TRAINING LIFECYCLE</p><h2>数据集、训练与模型</h2></div><span className="pill">服务器只调度 · GPU 在 Worker</span></div><div className="notice inline-notice"><strong>测试集冻结</strong><span>训练 Worker 只能下载 train / val / calibration；Production 必须完成固定 test、生产比较、阈值标定和兼容 Catalog 重建。</span></div>{message && <p className={message.includes("失败") || message.includes("尚未") || message.includes("必须") ? "error" : "success"}>{message}</p>}<h3 className="subheading">Dataset Versions</h3><div className="data-list">{datasets.map((item) => <article key={item.dataset_version_id}><div><strong>{item.name}</strong><span>{item.protocol} · {item.membership_digest.slice(0, 12)}…</span></div><div className="metrics"><b>{item.sample_count} 样本</b>{Object.entries(item.split_counts).map(([split, count]) => <small key={split}>{split}: {count}</small>)}<span className="pill">{item.status}</span></div></article>)}</div><h3 className="subheading">Training Runs</h3><div className="data-list">{runs.map((item) => <article key={item.training_run_id}><div><strong>{item.model_family} · {item.task_type}</strong><span>seed {item.seed} · {new Date(item.created_at).toLocaleString()}</span></div><div className="metrics"><span className="pill">{item.job_state}</span></div></article>)}</div><h3 className="subheading">Model Versions</h3><div className="data-list clickable-list">{models.map((item) => <article key={item.model_version_id} onClick={() => getModel(item.model_version_id).then(setModelDetail).catch((reason) => setMessage(String(reason)))}><div><strong>{item.version}</strong><span>{item.model_family} · {item.feature_dim ?? "-"} 维 · {item.completed_evaluations} 次固定评估</span></div><div className="metrics"><span className="pill">{item.status}</span>{item.status === "candidate" && <button onClick={(event) => { event.stopPropagation(); void promote(item.model_version_id); }}>请求上线门禁</button>}{item.status === "retired" && <button className="danger" onClick={(event) => { event.stopPropagation(); void rollback(item.model_version_id); }}>回滚为 Production</button>}</div></article>)}</div>{modelDetail && <div className="review-detail detail-block"><div className="panel-heading"><div><h3>{modelDetail.version}</h3><p className="muted">{modelDetail.model_family} · {modelDetail.sha256}</p></div><span className="pill">{modelDetail.is_production ? "Production" : modelDetail.status}</span></div><div className="manifest-grid"><span>Feature Dim<strong>{modelDetail.feature_dim ?? "-"}</strong></span><span>Preprocess<strong>{modelDetail.preprocess_id}</strong></span><span>Index Schema<strong>{modelDetail.compatible_index_schema}</strong></span><span>License<strong>{modelDetail.license}</strong></span></div><h4>固定评估与阈值</h4>{modelDetail.evaluations.length === 0 ? <p className="muted">尚无固定评估。</p> : modelDetail.evaluations.map((evaluation) => <div className="evaluation-card" key={evaluation.evaluation_run_id}><div className="panel-heading"><strong>{evaluation.protocol}</strong><span className="pill">{evaluation.status} / {evaluation.job_state}</span></div><div className="metrics-table">{Object.entries(evaluation.metrics).map(([name, value]) => <span key={name}>{name}<strong>{value.toFixed(4)}</strong></span>)}</div><p className="muted">阈值：{JSON.stringify(evaluation.calibrated_thresholds)}</p><pre className="plan-preview">{JSON.stringify(evaluation.comparison, null, 2)}</pre></div>)}<h4>上线 / 回滚事件</h4><div className="data-list compact">{modelDetail.promotion_events.map((event) => <article key={event.event_id}><div><strong>{event.event_type}</strong><span>{new Date(event.created_at).toLocaleString()} · Catalog {event.catalog_id?.slice(0, 12) ?? "-"}</span></div></article>)}</div></div>}</section>;
}

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState<Page>("overview");
  const [release, setRelease] = useState<Deployment | null>(null);

  useEffect(() => {
    me().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false));
  }, []);
  useEffect(() => {
    if (user) void deployment().then(setRelease).catch(() => setRelease(null));
  }, [user]);

  if (loading) return <main className="loading">正在连接控制面…</main>;
  if (!user) return <Login onLogin={setUser} />;
  const isAdmin = user.roles.includes("admin");
  const canOperate = isAdmin || user.roles.includes("operator");
  const canSeeWorkers = canOperate;
  const navigation: Array<[Page, string]> = [
    ["overview", "总览"], ["query", "查询识别"],
    ["upload", "批次上传"], ["batches", "批次进度"],
    ["reviews", "审核中心"], ["individuals", "个体目录"],
    ["relationships", "关系证据"], ["training", "训练与模型"],
    ["catalogs", "Catalog"],
  ];
  if (canSeeWorkers) navigation.push(["workers", "GPU Worker"]);
  if (isAdmin) navigation.push(["system", "系统管理"]);

  return <div className="app-shell">
    <aside>
      <div className="brand"><div className="mark">WW</div><div><strong>WhiteWhale</strong><span>Control Plane</span></div></div>
      <nav>{navigation.map(([key, label]) => <button key={key} className={page === key ? "active" : ""} onClick={() => setPage(key)}>{label}</button>)}</nav>
      <div className="account"><span>{user.username}</span><small>{user.roles.join(" · ")}</small><button className="quiet" onClick={() => logout().finally(() => setUser(null))}>退出</button></div>
    </aside>
    <main className="workspace">
      <header><div><p className="eyebrow">M5 · DELIVERY</p><h1>海豚归档控制台</h1>{release && <small>{release.branch} · {release.commit.slice(0, 12)} · {release.deployed_at}</small>}</div><div className="status-dot">控制面在线</div></header>
      <div className="notice"><strong>候选不等于正式身份</strong><span>上传完成后，模型只生成候选结果；任何跨时间个体合并均需独立人工审核。</span></div>
      {page === "overview" && <OverviewPanel />}
      {page === "query" && <QueryPanel />}
      {page === "upload" && <UploadPanel />}
      {page === "batches" && <BatchesPanel canDispatch={canOperate} />}
      {page === "reviews" && <ReviewPanel />}
      {page === "individuals" && <IndividualsPanel />}
      {page === "relationships" && <RelationshipsPanel />}
      {page === "training" && <TrainingPanel />}
      {page === "catalogs" && <CatalogPanel />}
      {page === "workers" && <WorkersPanel canManage={isAdmin} />}
      {page === "system" && <SystemPanel release={release} />}
    </main>
  </div>;
}
