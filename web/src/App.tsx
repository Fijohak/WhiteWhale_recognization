import { FormEvent, useEffect, useMemo, useState } from "react";
import { sha256 } from "@noble/hashes/sha256";
import { bytesToHex } from "@noble/hashes/utils";
import {
  completeFile,
  completeSession,
  createUpload,
  getUploadStatus,
  importSession,
  login,
  logout,
  me,
  putPart,
  type ManifestFile,
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

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    me().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false));
  }, []);

  if (loading) return <main className="loading">正在连接控制面…</main>;
  if (!user) return <Login onLogin={setUser} />;

  return <div className="app-shell">
    <aside>
      <div className="brand"><div className="mark">WW</div><div><strong>WhiteWhale</strong><span>Control Plane</span></div></div>
      <nav><a className="active">批次上传</a><a>任务队列</a><a>审核中心</a><a>个体目录</a><a>Worker</a></nav>
      <div className="account"><span>{user.username}</span><small>{user.roles.join(" · ")}</small><button className="quiet" onClick={() => logout().finally(() => setUser(null))}>退出</button></div>
    </aside>
    <main className="workspace">
      <header><div><p className="eyebrow">M1 · FOUNDATION</p><h1>批次控制台</h1></div><div className="status-dot">控制面在线</div></header>
      <div className="notice"><strong>候选不等于正式身份</strong><span>上传完成后，模型只生成候选结果；任何跨时间个体合并均需独立人工审核。</span></div>
      <UploadPanel />
    </main>
  </div>;
}
