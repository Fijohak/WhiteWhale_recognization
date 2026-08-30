import { FormEvent, useEffect, useMemo, useState } from "react";
import { sha256 } from "@noble/hashes/sha256";
import { bytesToHex } from "@noble/hashes/utils";
import {
  completeFile,
  completeSession,
  createUpload,
  getUploadStatus,
  importSession,
  activateCatalog,
  getCandidate,
  listBatches,
  listCatalogs,
  listIndividuals,
  login,
  logout,
  me,
  putPart,
  reviewInbox,
  submitVote,
  type BatchSummary,
  type Candidate,
  type Catalog,
  type Individual,
  type ManifestFile,
  type ReviewTask,
  type User
} from "./api";

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

function Login({ onLogin }: { onLogin: (user: User) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
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
      <p className="muted">数据留在课题组服务器，算法结果保持候选状态，正式身份由人工审核决定。</p>
      <form onSubmit={submit}>
        <label>用户名<input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" /></label>
        <label>密码<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></label>
        {error && <p className="error">{error}</p>}
        <button disabled={busy}>{busy ? "正在登录…" : "登录"}</button>
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
      const manifest: ManifestFile[] = [];
      let hashed = 0;
      for (const file of files) {
        setProgress({ stage: `正在计算哈希：${file.name}`, doneBytes: hashed, totalBytes });
        manifest.push({
          relative_path: file.webkitRelativePath || file.name,
          size_bytes: file.size,
          sha256: await hashBlob(file)
        });
        hashed += file.size;
      }
      const key = await resumeKey(batchName, manifest);
      let sessionId = localStorage.getItem(key);
      if (sessionId) {
        try {
          await getUploadStatus(sessionId);
        } catch {
          localStorage.removeItem(key);
          sessionId = null;
        }
      }
      if (!sessionId) {
        sessionId = (await createUpload(batchName, sourceFormat, manifest)).session_id;
        localStorage.setItem(key, sessionId);
      }

      let uploaded = 0;
      let status = await getUploadStatus(sessionId);
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
      const imported = await importSession(
        sessionId, sourceFormat === "generic" ? capturedOn : undefined);
      localStorage.removeItem(key);
      setProgress({ stage: "导入完成", doneBytes: totalBytes, totalBytes });
      setMessage(`批次已登记：${imported.batch_id}`);
      status = await getUploadStatus(sessionId);
      void status;
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

function BatchesPanel() {
  const [items, setItems] = useState<BatchSummary[]>([]);
  const [error, setError] = useState("");
  useEffect(() => { listBatches().then(setItems).catch((e) => setError(String(e))); }, []);
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">BATCHES</p><h2>批次进度</h2></div></div>
    {error && <p className="error">{error}</p>}
    <div className="data-list">{items.map((item) => <article key={item.batch_id}><div><strong>{item.name}</strong><span>{item.source_format} · {new Date(item.created_at).toLocaleString()}</span></div><div className="metrics"><b>{item.image_count} 图</b><b>{item.crop_count} Crop</b><b>{item.cluster_count} 簇</b><span className="pill">{item.stage}</span></div></article>)}</div>
  </section>;
}

function ReviewPanel() {
  const [tasks, setTasks] = useState<ReviewTask[]>([]);
  const [candidate, setCandidate] = useState<Candidate | null>(null);
  const [selected, setSelected] = useState<ReviewTask | null>(null);
  const [existingId, setExistingId] = useState("");
  const [message, setMessage] = useState("");
  const refresh = () => reviewInbox().then(setTasks).catch((e) => setMessage(String(e)));
  useEffect(() => { void refresh(); }, []);
  async function open(task: ReviewTask) { setSelected(task); setCandidate(await getCandidate(task.subject_id)); }
  async function vote(choice: string) {
    if (!selected) return;
    await submitVote(selected.task_id, choice, choice === "existing" ? existingId : undefined);
    setMessage("本次投票已追加保存；其他审核人的未完成票仍不可见。");
    setSelected(null); setCandidate(null); await refresh();
  }
  return <section className="panel"><div className="panel-heading"><div><p className="eyebrow">BLIND REVIEW</p><h2>独立审核中心</h2></div><span className="pill">{tasks.length} 待处理</span></div>
    {message && <p className="success">{message}</p>}
    <div className="review-layout"><div className="data-list compact">{tasks.map((task) => <button className="task-row" key={task.task_id} onClick={() => open(task)}><strong>{task.task_type === "cluster_purity" ? "批内簇纯度" : "历史身份匹配"}</strong><span>{task.subject_id.slice(0, 12)}…</span></button>)}</div>
      {candidate && selected && <div className="review-detail"><h3>{candidate.label}</h3><div className="crop-grid">{candidate.crops.map((crop) => <img key={crop.crop_id} src={crop.media_url} alt="海豚 Crop" />)}</div>
        {candidate.matches.length > 0 && <div className="matches">{candidate.matches.map((match) => <label key={match.individual_id}><input type="radio" name="existing" onChange={() => setExistingId(match.individual_id)} />#{match.rank} {match.individual_id.slice(0, 8)} · {match.score.toFixed(3)} · {match.support_frames} 帧</label>)}</div>}
        <div className="actions">{selected.task_type === "cluster_purity" ? <><button onClick={() => vote("confirm_cluster")}>确认候选组</button><button className="secondary" onClick={() => vote("split_required")}>需要拆簇</button><button className="danger" onClick={() => vote("unusable")}>不可用</button></> : <><button disabled={!existingId} onClick={() => vote("existing")}>选择 Existing</button><button className="secondary" onClick={() => vote("new")}>选择 New</button><button className="quiet-dark" onClick={() => vote("uncertain")}>不确定</button></>}</div>
      </div>}</div>
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

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState<"upload" | "batches" | "reviews" | "individuals" | "catalogs">("upload");

  useEffect(() => {
    me().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false));
  }, []);

  if (loading) return <main className="loading">正在连接控制面…</main>;
  if (!user) return <Login onLogin={setUser} />;

  return <div className="app-shell">
    <aside>
      <div className="brand"><div className="mark">WW</div><div><strong>WhiteWhale</strong><span>Control Plane</span></div></div>
      <nav>{[["upload", "批次上传"], ["batches", "批次进度"], ["reviews", "审核中心"], ["individuals", "个体目录"], ["catalogs", "Catalog"]].map(([key, label]) => <button key={key} className={page === key ? "active" : ""} onClick={() => setPage(key as typeof page)}>{label}</button>)}</nav>
      <div className="account"><span>{user.username}</span><small>{user.roles.join(" · ")}</small><button className="quiet" onClick={() => logout().finally(() => setUser(null))}>退出</button></div>
    </aside>
    <main className="workspace">
      <header><div><p className="eyebrow">M2 · ARCHIVAL LOOP</p><h1>海豚归档控制台</h1></div><div className="status-dot">控制面在线</div></header>
      <div className="notice"><strong>候选不等于正式身份</strong><span>上传完成后，模型只生成候选结果；任何跨时间个体合并均需独立人工审核。</span></div>
      {page === "upload" && <UploadPanel />}
      {page === "batches" && <BatchesPanel />}
      {page === "reviews" && <ReviewPanel />}
      {page === "individuals" && <IndividualsPanel />}
      {page === "catalogs" && <CatalogPanel />}
    </main>
  </div>;
}
