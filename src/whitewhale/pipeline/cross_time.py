"""
跨时间批次管线驱动（E7 首跑验证，正式流程）。

流程：
1. 历史库 gallery：配置指定批次的 labeled → YOLO 裁剪 → r4 特征；
   个体标识 = 已确认的 individual_id；
2. 新批次逐个（按 session）跑批内归档管线（见 whitewhale.pipeline.archival）：
   检测裁剪 → r4 特征 → HDBSCAN 批内候选聚类 → 子簇化 →
   簇级多帧投票匹配历史库 → 代表图 + 候选簇拼图（人工审核材料）。

输出：
- pipeline.cross_time.gallery_crops 指向的历史库裁剪图
- pipeline.query.embeddings/meta 指向的历史库特征
- outputs/cluster_archival/cross_time/<session>/  每批结果

CLI 入口见 scripts/run_cross_time_batch.py。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd

from whitewhale.config import load_config
from whitewhale.data.manifest import compute_sha256
from whitewhale.detection.detector import yolo_crop_provenance
from whitewhale.pipeline.archival import (
    _validate_args,
    _validate_crop_artifact,
    _validate_metadata,
    load_gallery,
    run as run_archival,
)
from whitewhale.reid.embedding import (
    MegaDescriptorMetricAdapter,
    read_metadata_csv,
    require_compatible_embedding_configs,
)

BASE = Path(__file__).resolve().parents[3]


def _repo_path(value: str | Path) -> Path:
    """把 pipeline.yaml 中相对仓库根的路径解析为稳定绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else BASE / path


CFG = load_config("pipeline")
OUTPUT_ROOT = _repo_path(CFG.get("output_root", "outputs"))
QUERY_CFG = CFG.get("query", {})
CROSS_CFG = CFG.get("cross_time", {})
RETR_CFG = CFG.get("retrieval", {})
CLUSTER_CFG = CFG.get("clustering", {})
CROP_CFG = CFG.get("crop", {})
DET_CFG = CFG.get("detector", {})

DATA_ROOT = _repo_path(CFG.get("data_root", "src_dataset"))
MANIFEST = OUTPUT_ROOT / "index" / "dataset_manifest.csv"
CKPT = _repo_path(CFG.get("reid_checkpoint", "outputs/metric_learning/r4/best.pt"))
DET_WEIGHTS = _repo_path(
    CFG.get("detector_checkpoint", "models/detectors/yolov8n_dorsalfin.pt"))
GALLERY_SESSIONS = [str(s) for s in CROSS_CFG.get(
    "gallery_sessions", ["20140806 01", "20140806 03"])]
OUT_ROOT = OUTPUT_ROOT / CROSS_CFG.get("out_dir", "cluster_archival/cross_time")
GALLERY_CROPS = OUTPUT_ROOT / CROSS_CFG.get(
    "gallery_crops", "artifacts/r4_yolocrop_v3/gallery/crops")
GAL_NPY = OUTPUT_ROOT / QUERY_CFG.get(
    "embeddings", "artifacts/r4_yolocrop_v3/gallery/embeddings.npy")
GAL_META = OUTPUT_ROOT / QUERY_CFG.get(
    "meta", "artifacts/r4_yolocrop_v3/gallery/embeddings_meta.csv")

DET_CONF = float(DET_CFG.get("conf", 0.25))
DET_IMGSZ = int(DET_CFG.get("imgsz", 1024))
DET_DEVICE = DET_CFG.get("device", "auto")
PAD_X = float(CROP_CFG.get("pad_x", 0.30))
PAD_UP = float(CROP_CFG.get("pad_up", 0.15))
PAD_DOWN = float(CROP_CFG.get("pad_down", 0.60))

_SAFE_SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,127}\Z", re.ASCII)
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

BATCH_MANIFEST_NAME = "input_manifest.csv"
COMMIT_MARKER_NAME = "_COMMITTED.json"
BATCH_COMMIT_SCHEMA_VERSION = 2
DIRECTORY_DIGEST_ALGORITHM = "sha256-v1:ordered-relative-path+file-sha256"


def _require_within(root: Path, candidate: Path, label: str) -> Path:
    """解析并确认输出路径仍位于约定根目录内。"""
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} 越过输出根目录: {candidate}") from exc
    return candidate


def _validate_session_leaf(session: str) -> str:
    """session 必须可安全作为 Windows 目录名和 CSV basename。"""
    if not isinstance(session, str) or not _SAFE_SESSION.fullmatch(session):
        raise ValueError(
            f"不安全的 session {session!r}；只允许 ASCII 字母、数字、空格、点、"
            "下划线和连字符，且必须是单个非空名称")
    if session.endswith((".", " ")):
        raise ValueError(f"session {session!r} 不能以点或空格结尾")
    device_stem = session.split(".", 1)[0].rstrip(" ").upper()
    if device_stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"session {session!r} 是 Windows 保留名")
    return session


def validate_sessions(sessions: list[str], manifest: pd.DataFrame,
                      *, allow_gallery: bool = False) -> list[str]:
    """校验目标 session 非空、安全、真实存在且不会重复执行。"""
    if "session_id" not in manifest.columns:
        raise ValueError("dataset manifest 缺少 session_id")
    available = set(manifest["session_id"].fillna("").astype(str)) - {""}
    if not sessions:
        raise ValueError("至少需要一个非空 session")

    validated: list[str] = []
    seen: set[str] = set()
    for raw in sessions:
        session = _validate_session_leaf(raw)
        if session not in available:
            raise ValueError(
                f"session {session!r} 不在 dataset manifest 中；"
                f"可用 session: {sorted(available)}")
        if not allow_gallery and session in GALLERY_SESSIONS:
            raise ValueError(f"session {session!r} 是历史库批次，不能作为跨时间目标批次")
        key = session.casefold()
        if key in seen:
            raise ValueError(f"session {session!r} 重复")
        seen.add(key)
        validated.append(session)
    return validated


def _session_paths(session: str) -> tuple[Path, Path]:
    """返回受 containment 保护的 query manifest 与批次输出路径。"""
    session = _validate_session_leaf(session)
    out_root = _require_within(OUTPUT_ROOT, OUT_ROOT, "cross_time.out_dir")
    manifests_root = _require_within(out_root, out_root / "manifests", "manifest 目录")
    manifest_path = _require_within(
        manifests_root, manifests_root / f"{session}.csv", "session manifest")
    output_path = _require_within(out_root, out_root / session, "session 输出")
    if manifest_path.parent != manifests_root or output_path.parent != out_root:
        raise ValueError("session 输出必须是约定根目录的直属子项")
    return manifest_path, output_path


_REBUILD_HINT = (
    "跨时间入口禁止原地重建或覆盖当前活动 gallery；"
    "请使用 scripts/rebuild_r4_artifacts.py --out 新版本目录")


def _configured_gallery_sessions() -> list[str]:
    """校验并返回配置中的唯一 gallery session 列表。"""
    if not GALLERY_SESSIONS:
        raise ValueError("pipeline.cross_time.gallery_sessions 为空")
    sessions = [_validate_session_leaf(value) for value in GALLERY_SESSIONS]
    folded = [value.casefold() for value in sessions]
    if len(set(folded)) != len(folded):
        raise ValueError("pipeline.cross_time.gallery_sessions 含重复 session")
    return sessions


def _expected_runtime_embedding_config() -> dict:
    """根据当前运行配置构造 gallery 兼容性期望值。"""
    _validate_args(argparse.Namespace(
        pool=False,
        min_cluster_size=int(CLUSTER_CFG.get("min_cluster_size", 3)),
        subcluster_min_size=int(CLUSTER_CFG.get("subcluster_min_size", 4)),
        topk=int(RETR_CFG.get("topk", 3)),
        threshold_cluster=float(RETR_CFG.get("threshold_cluster", 0.58)),
        threshold_image=float(RETR_CFG.get("threshold_image", 0.50)),
        max_sheets=int(CROSS_CFG.get("max_sheets", 200)),
        det_conf=DET_CONF,
        det_imgsz=DET_IMGSZ,
        det_pad_x=PAD_X,
        det_pad_up=PAD_UP,
        det_pad_down=PAD_DOWN,
    ))
    if not CKPT.is_file():
        raise FileNotFoundError(f"ReID checkpoint 不存在: {CKPT}")
    expected = {
        "model": f"megadescriptor-metric-learning-{CKPT.parent.name}",
        "preprocess": MegaDescriptorMetricAdapter.preprocess_id,
        "checkpoint_sha256": compute_sha256(CKPT),
    }
    expected.update(yolo_crop_provenance(
        DET_WEIGHTS, DET_CONF, DET_IMGSZ, PAD_X, PAD_UP, PAD_DOWN))
    return expected


def validate_active_gallery(target_sessions: list[str] | None = None) -> dict:
    """
    只读校验活动 gallery 的批次、crop 行绑定和运行配置。

    ``target_sessions`` 非空时额外拒绝把 gallery 批次当作本次查询。
    """
    _require_within(OUTPUT_ROOT, GALLERY_CROPS, "gallery crops")
    _require_within(OUTPUT_ROOT, GAL_NPY, "gallery embedding")
    _require_within(OUTPUT_ROOT, GAL_META, "gallery meta")
    _, _, info, config = load_gallery(GAL_NPY, GAL_META)
    expected_sessions = set(_configured_gallery_sessions())
    actual_sessions = set(info["session_id"].astype(str))
    if actual_sessions != expected_sessions:
        raise ValueError(
            "活动 gallery session 集合与配置不一致："
            f"actual={sorted(actual_sessions)}, expected={sorted(expected_sessions)}")
    targets = set(target_sessions or [])
    overlap = sorted(actual_sessions & targets)
    if overlap:
        raise ValueError(f"目标 session 不得出现在活动 gallery: {overlap}")

    _validate_crop_artifact(info, GALLERY_CROPS, "gallery", config)
    require_compatible_embedding_configs(
        config, _expected_runtime_embedding_config(),
        left_name="gallery", right_name="pipeline runtime")
    print(
        f"[gallery] 只读严格校验通过：{len(info)} 张 / "
        f"{len(actual_sessions)} 批（未重建任何产物） → {GAL_NPY}")
    return config


def build_gallery(*_args, **_kwargs) -> None:
    """保留旧 API 的 fail-closed 门禁；活动 gallery 只能只读复用。"""
    raise RuntimeError(_REBUILD_HINT)


def build_query_manifest(session: str, m: pd.DataFrame,
                         destination: Path) -> Path:
    """将新批次清单写入明确的暂存路径。"""
    validate_sessions([session], m)
    required = {"image_id", "relative_path", "session_id", "label_status"}
    missing = sorted(required - set(m.columns))
    if missing:
        raise ValueError(f"dataset manifest 缺少必需列: {missing}")
    q = m[m["session_id"] == session]
    q = q[q["label_status"].isin(["labeled", "loose_known"])]
    if q.empty:
        raise ValueError(f"session {session!r} 没有 labeled/loose_known 图片")
    manifests_root = _session_paths(session)[0].parent
    p = _require_within(manifests_root, destination, "session manifest staging")
    p.parent.mkdir(parents=True, exist_ok=True)
    # relative_path 需含 session 前缀（archival 直接拼 images_root 读图），
    # 但已带前缀的全局清单不可重复拼接。
    relative = q["relative_path"].fillna("").astype(str).str.replace(
        "\\", "/", regex=False).str.strip()
    prefix = f"{session}/"
    q = q.assign(relative_path=[
        value if value.startswith(prefix) else prefix + value
        for value in relative
    ])
    q = _validate_metadata(q, "session query")
    q[["image_id", "relative_path", "session_id"]].to_csv(
        p, index=False, encoding="utf-8-sig")
    print(f"[query] {session}: {len(q)} 张"
          f"（labeled {int((q['label_status'] == 'labeled').sum())} / "
          f"散图 {int((q['label_status'] == 'loose_known').sum())}）")
    return p


def _directory_content_digest(directory: Path) -> dict[str, str | int]:
    """摘要目录内全部普通文件；代表图目录只允许扁平、无链接内容。"""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"批次缺少普通目录: {directory.name}")
    digest = hashlib.sha256()
    digest.update(DIRECTORY_DIGEST_ALGORITHM.encode("ascii") + b"\0")
    files = sorted(directory.iterdir(), key=lambda path: path.name)
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"批次目录含非普通代表图: {path}")
        encoded_name = path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(compute_sha256(path)))
    return {
        "algorithm": DIRECTORY_DIGEST_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def _batch_output_digests(batch_out: Path) -> dict[str, object]:
    """计算 commit marker 必须绑定的正式输出内容摘要。"""
    clusters = batch_out / "clusters.csv"
    summary = batch_out / "summary.json"
    for path in (clusters, summary):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"批次缺少普通完整性文件: {path.name}")
    return {
        "clusters.csv": compute_sha256(clusters),
        "summary.json": compute_sha256(summary),
        "representatives": _directory_content_digest(
            batch_out / "representatives"),
    }


def _validate_published_batch(
    session: str,
    batch_out: Path,
) -> tuple[Path, str, dict[str, object]]:
    """校验可恢复批次的完整输出和内置 manifest snapshot。"""
    required = ("clusters.csv", "summary.json", BATCH_MANIFEST_NAME)
    missing = [
        name for name in required
        if (not (batch_out / name).is_file()
            or (batch_out / name).is_symlink())
    ]
    if missing:
        raise ValueError(f"批次目录缺少完整性文件: {missing}")
    try:
        summary = json.loads((batch_out / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("批次 summary.json 无法校验") from exc
    try:
        n_images = int(summary.get("n_images", 0)) if isinstance(summary, dict) else 0
    except (TypeError, ValueError) as exc:
        raise ValueError("批次 summary.json 的 n_images 无效") from exc
    if n_images <= 0:
        raise ValueError("批次 summary.json 缺少有效 n_images")

    snapshot = batch_out / BATCH_MANIFEST_NAME
    frame = _validate_metadata(read_metadata_csv(snapshot), "batch manifest snapshot")
    sessions = set(frame["session_id"])
    if sessions != {session}:
        raise ValueError(
            f"batch manifest snapshot session 不一致: {sorted(sessions)} / {session!r}")
    if n_images != len(frame):
        raise ValueError("batch summary.n_images 与 manifest snapshot 行数不一致")

    clusters = _validate_metadata(
        read_metadata_csv(batch_out / "clusters.csv"), "batch clusters")
    if set(clusters["image_id"]) != set(frame["image_id"]):
        raise ValueError("batch clusters 与 manifest snapshot 的 image_id 集合不一致")
    expected_rows = frame.set_index("image_id")[["relative_path", "session_id"]].sort_index()
    actual_rows = clusters.set_index("image_id")[["relative_path", "session_id"]].sort_index()
    if not expected_rows.equals(actual_rows):
        raise ValueError("batch clusters 与 manifest snapshot 的追溯行不一致")
    return snapshot, compute_sha256(snapshot), _batch_output_digests(batch_out)


def _isolate_recovery_path(path: Path, root: Path, reason: str) -> Path:
    """将遗留临时路径原子改名隔离，保留现场且不递归删除。"""
    root = root.resolve()
    path = _require_within(root, path, "恢复遗留路径")
    destination = _require_within(
        root,
        root / f".orphan-{reason}-{path.name}-{uuid.uuid4().hex}",
        "恢复隔离路径",
    )
    path.rename(destination)
    print(f"[recovery] 遗留 {reason} 已隔离至 {destination}")
    return destination


def _isolate_restart_leftovers(
    session: str,
    active_manifest: Path,
    batch_out: Path,
) -> list[Path]:
    """启动时识别上次崩溃留下的 publish/staging/marker tmp 并隔离。"""
    out_root = batch_out.parent.resolve()
    candidates = [
        *active_manifest.parent.glob(f".{session}.staging-*.csv"),
        *active_manifest.parent.glob(f".{session}.publish-*.csv"),
        *out_root.glob(f".{session}.staging-*"),
    ]
    marker_temporaries: list[Path] = []
    if batch_out.is_dir():
        marker_temporaries = list(
            batch_out.glob(f".{COMMIT_MARKER_NAME}.tmp-*"))
        candidates.extend(marker_temporaries)
    isolated: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        isolated.append(_isolate_recovery_path(path, out_root, "restart-temp"))
    if marker_temporaries and batch_out.is_dir():
        remaining = list(batch_out.iterdir())
        if not remaining or all(
                entry.name.startswith(".orphan-restart-temp-")
                for entry in remaining):
            isolated.append(_isolate_recovery_path(
                batch_out, out_root, "restart-empty-batch"))
    return isolated


def _publish_manifest_snapshot(snapshot: Path, active_manifest: Path) -> None:
    """通过同目录暂存文件原子发布批次 manifest。"""
    active_manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_staging = tempfile.mkstemp(
        prefix=f".{active_manifest.stem}.publish-", suffix=".csv",
        dir=active_manifest.parent)
    os.close(fd)
    staging = Path(raw_staging)
    try:
        shutil.copyfile(snapshot, staging)
        os.replace(staging, active_manifest)
    finally:
        if staging.exists():
            _isolate_recovery_path(
                staging, active_manifest.parent.parent, "publish-failed")


def _write_commit_marker(session: str, batch_out: Path,
                         manifest_sha256: str,
                         output_digests: dict[str, object]) -> None:
    """在批次目录内原子写入最终提交标记。"""
    marker = batch_out / COMMIT_MARKER_NAME
    temporary = batch_out / f".{COMMIT_MARKER_NAME}.tmp-{uuid.uuid4().hex}"
    payload = {
        "schema_version": BATCH_COMMIT_SCHEMA_VERSION,
        "state": "committed",
        "session_id": session,
        "manifest_snapshot": BATCH_MANIFEST_NAME,
        "manifest_sha256": manifest_sha256,
        "output_digests": output_digests,
    }
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, marker)
    finally:
        if temporary.exists():
            _isolate_recovery_path(
                temporary, batch_out.parent, "marker-write-failed")


def _quarantine_incomplete_batch(batch_out: Path, reason: Exception) -> Path:
    """将无法安全恢复的管线批次隔离，不删除现场。"""
    quarantine = batch_out.with_name(
        f".{batch_out.name}.incomplete-{uuid.uuid4().hex}")
    batch_out.rename(quarantine)
    print(f"[recovery] 半提交批次无法安全完成，已隔离至 {quarantine}: {reason}")
    return quarantine


def _recover_existing_batch(session: str, active_manifest: Path,
                            batch_out: Path) -> bool:
    """
    恢复管线留下的半提交批次。

    返回 ``True`` 表示本次已自动完成提交；已正常提交的批次仍拒绝
    重跑，无管线特征的用户目录不会被移动。
    """
    if not batch_out.exists():
        return False
    if not batch_out.is_dir():
        raise FileExistsError(f"session 输出路径已存在且不是目录: {batch_out}")

    snapshot_path = batch_out / BATCH_MANIFEST_NAME
    marker_path = batch_out / COMMIT_MARKER_NAME
    has_legacy_outputs = (
        (batch_out / "summary.json").exists()
        and (batch_out / "clusters.csv").exists()
    )
    if (has_legacy_outputs
            and not snapshot_path.exists()
            and not marker_path.exists()):
        raise FileExistsError(
            "session 旧版结果已存在且缺少提交快照/标记，按只读结果保留，"
            f"禁止自动隔离、移动或重跑: {batch_out}")

    managed = (
        snapshot_path.exists()
        or marker_path.exists()
        or ((batch_out / "summary.json").exists()
            and (batch_out / "clusters.csv").exists())
    )
    if not managed:
        raise FileExistsError(f"session 正式结果已存在，拒绝覆盖: {batch_out}")

    try:
        snapshot, snapshot_hash, output_digests = _validate_published_batch(
            session, batch_out)
        if (marker_path.exists() or marker_path.is_symlink()) and (
                not marker_path.is_file() or marker_path.is_symlink()):
            raise ValueError("批次 commit marker 不是普通文件")
        was_committed = marker_path.is_file()
        if was_committed:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                raise ValueError("批次 commit marker 必须是 JSON 对象")
            if (marker.get("schema_version") != BATCH_COMMIT_SCHEMA_VERSION
                    or marker.get("state") != "committed"
                    or marker.get("session_id") != session
                    or marker.get("manifest_snapshot") != BATCH_MANIFEST_NAME
                    or marker.get("manifest_sha256") != snapshot_hash
                    or marker.get("output_digests") != output_digests):
                raise ValueError("批次 commit marker 与 snapshot/输出内容不一致")
    except (ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        _quarantine_incomplete_batch(batch_out, exc)
        return False

    # 发布阶段的 I/O 错误不是产物损坏：保留原批次并向上抛出，
    # 避免把可恢复的结果隔离后又重跑昂贵特征提取。
    active_hash = (
        compute_sha256(active_manifest) if active_manifest.is_file() else None)
    if active_hash != snapshot_hash:
        _publish_manifest_snapshot(snapshot, active_manifest)
    if not was_committed:
        _write_commit_marker(
            session, batch_out, snapshot_hash, output_digests)
        print(f"[recovery] {session} 半提交批次已自动完成")
        return True

    raise FileExistsError(f"session 正式结果已存在，拒绝覆盖: {batch_out}")


def run_batch(session: str, m: pd.DataFrame) -> None:
    active_manifest, batch_out = _session_paths(session)
    _isolate_restart_leftovers(session, active_manifest, batch_out)
    if _recover_existing_batch(session, active_manifest, batch_out):
        return
    active_manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_staging = tempfile.mkstemp(
        prefix=f".{session}.staging-", suffix=".csv",
        dir=active_manifest.parent)
    os.close(fd)
    staged_manifest = Path(raw_staging)
    completed = False
    try:
        man = build_query_manifest(session, m, staged_manifest)
        args = argparse.Namespace(
            pool=False, input_manifest=man,
            input_manifest_data=read_metadata_csv(man),
            input_manifest_snapshot=BATCH_MANIFEST_NAME,
            images_root=DATA_ROOT, ckpt=CKPT,
            gallery_embeddings=GAL_NPY, gallery_meta=GAL_META,
            min_cluster_size=int(CLUSTER_CFG.get("min_cluster_size", 3)),
            subcluster_min_size=int(CLUSTER_CFG.get("subcluster_min_size", 4)),
            topk=int(RETR_CFG.get("topk", 3)),
            threshold_cluster=float(RETR_CFG.get("threshold_cluster", 0.58)),
            threshold_image=float(RETR_CFG.get("threshold_image", 0.50)),
            out=batch_out, det_weights=DET_WEIGHTS,
            det_conf=DET_CONF, det_imgsz=DET_IMGSZ, det_device=DET_DEVICE,
            det_pad_x=PAD_X, det_pad_up=PAD_UP, det_pad_down=PAD_DOWN,
            sheets=bool(CROSS_CFG.get("sheets", True)),
            max_sheets=int(CROSS_CFG.get("max_sheets", 200)),
        )
        run_archival(args)
        # 批次目录先携带 manifest snapshot 原子发布；随后发布外部
        # manifest，最后写 commit marker。任一时刻崩溃均可由下次启动完成。
        snapshot, snapshot_hash, output_digests = _validate_published_batch(
            session, batch_out)
        _publish_manifest_snapshot(snapshot, active_manifest)
        _write_commit_marker(
            session, batch_out, snapshot_hash, output_digests)
        completed = True
    finally:
        if staged_manifest.exists():
            if completed:
                staged_manifest.unlink()
            else:
                _isolate_recovery_path(
                    staged_manifest, batch_out.parent, "query-staging-failed")
    print(f"[pipeline] {session} 完成 → {batch_out}")


def main():
    ap = argparse.ArgumentParser(description="跨时间批次管线驱动")
    ap.add_argument("--only-gallery", action="store_true",
                    help="只读严格校验当前配置的历史库")
    ap.add_argument("--skip-gallery", action="store_true",
                    help="兼容旧参数；当前仍会只读严格校验 gallery")
    ap.add_argument(
        "--rebuild-gallery", "--overwrite-gallery", dest="rebuild_gallery",
        action="store_true",
        help="已禁用；请用 rebuild_r4_artifacts.py 发布新版本")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="只跑指定 session（默认全部新批次）")
    args = ap.parse_args()

    if args.rebuild_gallery:
        ap.error(_REBUILD_HINT)
    if args.only_gallery and args.sessions is not None:
        ap.error("--only-gallery 不能同时指定 --sessions")
    if args.sessions is not None and not args.sessions:
        ap.error("--sessions 后至少需要一个 session")

    if args.only_gallery:
        try:
            validate_active_gallery()
        except (OSError, ValueError, SystemExit) as exc:
            ap.error(str(exc))
        return

    m = read_metadata_csv(MANIFEST)
    try:
        validate_sessions(GALLERY_SESSIONS, m, allow_gallery=True)
        sessions = [] if args.only_gallery else validate_sessions(
            args.sessions if args.sessions is not None else sorted(
                str(s) for s in m["session_id"].dropna().unique()
                if str(s) not in GALLERY_SESSIONS),
            m,
        )
    except ValueError as exc:
        ap.error(str(exc))

    try:
        validate_active_gallery(sessions)
    except (OSError, ValueError, SystemExit) as exc:
        ap.error(str(exc))
    for s in sessions:
        run_batch(s, m)


if __name__ == "__main__":
    main()
