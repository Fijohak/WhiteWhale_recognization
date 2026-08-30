"""
批内簇级归档管线（正式流程）：新批次 → 检测 → 特征 → 候选聚类 → 簇级匹配 → 审核清单。

场景：项目最终目的 = 加速数据处理。新批次（或散图池）到达时自动走：
  YOLO 背鳍检测裁剪 → r4 特征 → HDBSCAN 批内候选聚类 → 大簇内子簇化 →
  每子簇与历史库（已确认个体）多帧投票匹配 → 输出归档候选 + 疑似新个体候选 + 噪声。

输出语义（项目红线）：
- HDBSCAN 簇 = Candidate Cluster，不是个体；-1 噪声合法，不强制分配；
- 匹配结果 = Candidate（候选划归），人工确认后才能叫个体/入库；
- 所有行保留 image_id / relative_path / session_id，可追溯到原图；
- 每簇选代表图（与簇均值特征最接近的一帧，归档用）。

两档阈值（历史实验参考值，非当前模型的独立标定结论）：
- 簇级（多帧投票）：0.58，来自 E5；
- 单图（噪声/孤图退化）：0.50，来自 E4。
更换模型、裁剪方式或独立评估集后必须重新标定，不能直接沿用历史 FA 结论。

用法（CLI 入口 scripts/run_pipeline.py）：
    python scripts/run_pipeline.py --pool                       # 散图池验证（复用预提取产物）
    python scripts/run_pipeline.py --input-manifest 清单.csv     # 新批次完整流程
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from whitewhale.data.image_store import validate_safe_image_ids
from whitewhale.data.manifest import compute_sha256
from whitewhale.detection.detector import detect_and_crop, yolo_crop_provenance
from whitewhale.reid.embedding import (
    load_verified_embedding_artifact,
    read_metadata_csv,
    require_compatible_embedding_configs,
    require_generated_artifact_provenance,
)


IdentityKey = tuple[str, str]
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
CROP_BUNDLE_DIGEST_ALGORITHM = "sha256-v1:ordered-image-id+file-sha256"


def _validate_relative_paths(values: pd.Series, label: str) -> list[str]:
    """校验清单路径是数据根内的非空相对路径。"""
    normalized: list[str] = []
    for position, raw in enumerate(values):
        value = "" if raw is None else str(raw).strip().replace("\\", "/")
        path = PurePosixPath(value)
        if (not value or value.endswith("/") or value.startswith("/")
                or _WINDOWS_DRIVE.match(value)
                or ".." in path.parts or path.name in {"", ".", ".."}):
            raise ValueError(
                f"{label}.relative_path 必须是数据根内的非空相对路径"
                f"（第 {position + 1} 行）: {raw!r}")
        normalized.append(path.as_posix())
    return normalized


def _validate_metadata(
    meta: pd.DataFrame,
    label: str,
    *,
    required: tuple[str, ...] = ("image_id", "relative_path", "session_id"),
) -> pd.DataFrame:
    """校验归档输入 meta 的非空性、追溯列和安全标识。"""
    missing = [column for column in required if column not in meta.columns]
    if missing:
        raise ValueError(f"{label} meta 缺少必需列: {missing}")
    if meta.empty:
        raise ValueError(f"{label} meta 为空，拒绝归档")

    checked = meta.copy()
    validate_safe_image_ids(checked["image_id"])
    checked["image_id"] = checked["image_id"].astype(str)
    checked["relative_path"] = _validate_relative_paths(
        checked["relative_path"], label)
    sessions = checked["session_id"].fillna("").astype(str).str.strip()
    if (sessions == "").any():
        raise ValueError(f"{label} meta 存在空 session_id，拒绝正式归档")
    checked["session_id"] = sessions
    return checked


def crop_bundle_provenance(meta: pd.DataFrame, crops_dir: Path) -> dict[str, str]:
    """计算按 meta 行序绑定的 crop JPEG 内容摘要与 manifest 摘要。"""
    checked = _validate_metadata(meta, "crop digest")
    crops_dir = Path(crops_dir)
    digest = hashlib.sha256()
    digest.update(CROP_BUNDLE_DIGEST_ALGORITHM.encode("ascii") + b"\0")
    for image_id in checked["image_id"]:
        crop = crops_dir / f"{image_id}.jpg"
        if not crop.is_file() or crop.is_symlink() or crop.stat().st_size <= 0:
            raise ValueError(f"crop 不是完整普通文件: {crop}")
        encoded_id = image_id.encode("ascii")
        digest.update(len(encoded_id).to_bytes(4, "big"))
        digest.update(encoded_id)
        digest.update(bytes.fromhex(compute_sha256(crop)))
    manifest = crops_dir / "crops_manifest.csv"
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(f"crop manifest 不存在或不是普通文件: {manifest}")
    return {
        "crop_bundle_digest_algorithm": CROP_BUNDLE_DIGEST_ALGORITHM,
        "crop_bundle_sha256": digest.hexdigest(),
        "crop_manifest_sha256": compute_sha256(manifest),
    }


def _validate_crop_artifact(meta: pd.DataFrame, crops_dir: Path,
                            label: str, artifact_config: dict) -> None:
    """严格校验 crop manifest、meta 行和 ``image_id.jpg`` 内容集。"""
    if not isinstance(artifact_config, dict):
        raise ValueError(f"{label} embedding config 必须是对象")
    checked_meta = _validate_metadata(meta, label)
    crops_dir = Path(crops_dir)
    if not crops_dir.is_dir():
        raise FileNotFoundError(f"{label} crops 目录不存在: {crops_dir}")
    crop_manifest_path = crops_dir / "crops_manifest.csv"
    if not crop_manifest_path.is_file():
        raise FileNotFoundError(
            f"{label} crops 缺少行绑定清单: {crop_manifest_path}")

    crop_manifest = _validate_metadata(
        read_metadata_csv(crop_manifest_path), f"{label} crop manifest")
    for column in ("image_id", "relative_path", "session_id"):
        if crop_manifest[column].tolist() != checked_meta[column].tolist():
            raise ValueError(
                f"{label} crops_manifest.{column} 与 embedding meta 行绑定不一致")

    expected_names = {
        "crops_manifest.csv",
        *(f"{image_id}.jpg" for image_id in checked_meta["image_id"]),
    }
    actual_entries = list(crops_dir.iterdir())
    actual_names = {entry.name for entry in actual_entries}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"{label} crops 内容集与 meta 不一致："
            f"missing={missing[:5]}, unexpected={unexpected[:5]}")
    for image_id in checked_meta["image_id"]:
        crop = crops_dir / f"{image_id}.jpg"
        if not crop.is_file() or crop.is_symlink() or crop.stat().st_size <= 0:
            raise ValueError(f"{label} crop 不是完整普通文件: {crop}")

    expected = crop_bundle_provenance(checked_meta, crops_dir)
    for field, value in expected.items():
        trusted = artifact_config.get(field)
        if not isinstance(trusted, str) or not trusted:
            raise ValueError(f"{label} embedding config 缺少可信 crop 摘要字段: {field}")
        if trusted != value:
            raise ValueError(f"{label} crop 内容摘要不一致: {field}")


def _validate_args(args) -> None:
    """在创建 staging 前校验聚类、阈值与检测参数范围。"""
    integer_ranges = {
        "min_cluster_size": (2, None),
        "subcluster_min_size": (2, None),
        "topk": (1, None),
        "max_sheets": (1, None),
    }
    for name, (minimum, maximum) in integer_ranges.items():
        value = getattr(args, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} 必须是整数")
        if value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{name} 超出允许范围")

    for name in ("threshold_cluster", "threshold_image"):
        value = float(getattr(args, name, float("nan")))
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须是 [-1, 1] 内的有限数")

    if not args.pool:
        conf = float(getattr(args, "det_conf", float("nan")))
        imgsz = getattr(args, "det_imgsz", None)
        if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise ValueError("det_conf 必须是 [0, 1] 内的有限数")
        if isinstance(imgsz, bool) or not isinstance(imgsz, (int, np.integer)) or imgsz < 1:
            raise ValueError("det_imgsz 必须是正整数")
        for name in ("det_pad_x", "det_pad_up", "det_pad_down"):
            value = float(getattr(args, name, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是非负有限数")


def _as_identity_key(value: object) -> IdentityKey:
    """内部候选键始终保留 session 与原始 ID，旧字符串仅作测试兼容。"""
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        return str(value[0]), str(value[1])
    return "", str(value)


def _identity_display(key: IdentityKey) -> str:
    """展示值同时保留 session 与不透明原 ID。"""
    session, identity = key
    return f"{session}::{identity}" if session else identity


def _score_feature_by_identity(
        feature: np.ndarray, gallery_emb: np.ndarray,
        gallery_ind: list[IdentityKey] | np.ndarray) -> dict[IdentityKey, float]:
    """单图对批次内身份取 max 图像分，不合并跨 session 同号。"""
    if len(gallery_emb) != len(gallery_ind):
        raise ValueError("gallery embedding 与身份键数量不一致")
    similarities = feature @ gallery_emb.T
    scores: dict[IdentityKey, float] = {}
    for value, score in zip(gallery_ind, similarities):
        key = _as_identity_key(value)
        scores[key] = max(scores.get(key, -np.inf), float(score))
    return scores


def _fit_hdbscan(features: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """运行 HDBSCAN，并统一返回标签和成员概率。"""
    try:
        import hdbscan
    except ImportError as exc:
        raise SystemExit(f"缺少 hdbscan 依赖: {exc}") from exc
    clusterer = hdbscan.HDBSCAN(**kwargs)
    labels = clusterer.fit_predict(features)
    return labels, np.asarray(clusterer.probabilities_)


def load_gallery(emb_path: Path, meta_path: Path):
    """历史库：已确认个体的 r4+YOLO 裁剪特征与溯源配置。"""
    emb, meta, config = load_verified_embedding_artifact(
        emb_path, meta_path, require_hashes=True)
    require_generated_artifact_provenance(config)
    info = _validate_metadata(
        meta, "gallery",
        required=("image_id", "relative_path", "session_id", "confirmed_identity"))
    identities = info["confirmed_identity"].fillna("").astype(str).str.strip()
    if (identities == "").any():
        raise ValueError("gallery 含未确认身份行，拒绝从正式库静默过滤")
    info["confirmed_identity"] = identities
    ind = [
        (str(session).strip(), str(identity).strip())
        for session, identity in zip(
            info["session_id"], info["confirmed_identity"])
    ]
    if len(ind) == 0:
        raise SystemExit("[pipeline] 历史库为空（无已确认个体），无法进行匹配。请先构建历史库特征。")
    return emb, ind, info, config


def match_single_feature(feature: np.ndarray, gallery_emb: np.ndarray,
                         gallery_ind: list[IdentityKey] | np.ndarray,
                         threshold: float) -> tuple[str, float, str]:
    """单图退化匹配；无效特征绝不能伪造任意 Top-1 候选。"""
    if not np.isfinite(feature).all():
        return "", float("nan"), "invalid_feature"
    scores = _score_feature_by_identity(feature, gallery_emb, gallery_ind)
    if not scores:
        return "", float("nan"), "no_gallery"
    top1 = max(scores, key=scores.get)
    score = float(scores[top1])
    status = "noise" if score < threshold else "noise_match_candidate"
    return _identity_display(top1), score, status


def cluster_embeddings(emb: np.ndarray,
                       min_cluster_size: int) -> tuple[np.ndarray, np.ndarray]:
    """聚类有限特征；不足最小簇大小时直接保留为噪声。"""
    nan_mask = ~np.isfinite(emb).all(axis=1)
    labels = np.full(len(emb), -1, dtype=int)
    probs = np.zeros(len(emb))
    if int((~nan_mask).sum()) < min_cluster_size:
        return labels, probs
    sub_labels, sub_probs = _fit_hdbscan(
        emb[~nan_mask], min_cluster_size=min_cluster_size)
    labels[~nan_mask] = sub_labels
    probs[~nan_mask] = sub_probs
    return labels, probs


def cluster_embeddings_by_session(
    emb: np.ndarray,
    sessions: pd.Series,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """逐 session 独立聚类，并把局部簇号映射为全批次唯一簇号。"""
    if len(emb) != len(sessions):
        raise ValueError("embedding 与 session 行数不一致")
    labels = np.full(len(emb), -1, dtype=int)
    probabilities = np.zeros(len(emb), dtype=float)
    next_cluster = 0
    session_values = sessions.astype(str).to_numpy()
    for session in dict.fromkeys(session_values):
        positions = np.flatnonzero(session_values == session)
        local_labels, local_probabilities = cluster_embeddings(
            emb[positions], min_cluster_size)
        for local_label in sorted(set(local_labels) - {-1}):
            labels[positions[local_labels == local_label]] = next_cluster
            next_cluster += 1
        probabilities[positions] = local_probabilities
    return labels, probabilities


def _scoped_gallery(
    query_rows: pd.DataFrame,
    gallery_emb: np.ndarray,
    gallery_ind: list[IdentityKey] | np.ndarray,
    gallery_info: pd.DataFrame,
    *,
    pool_mode: bool,
) -> tuple[np.ndarray, list[IdentityKey], int, str | None]:
    """为一个 query/簇选择 gallery；pool 严格限制同 session 且排除同串。"""
    if len(gallery_emb) != len(gallery_ind) or len(gallery_emb) != len(gallery_info):
        raise ValueError("gallery embedding、身份键与 meta 行数不一致")
    if not pool_mode:
        return gallery_emb, [_as_identity_key(value) for value in gallery_ind], 0, None

    for name, frame in (("pool", query_rows), ("gallery", gallery_info)):
        if "series_id" not in frame.columns:
            raise ValueError(f"{name} meta 缺少全量 manifest 派生的 series_id")
    sessions = query_rows["session_id"].astype(str).str.strip().unique().tolist()
    if len(sessions) != 1:
        raise ValueError(f"pool 候选簇跨越多个 session: {sessions}")

    session = sessions[0]
    mask = gallery_info["session_id"].astype(str).str.strip().eq(
        session).to_numpy(copy=True)
    if not mask.any():
        return gallery_emb[:0], [], 0, "no_gallery"

    query_ids = set(query_rows["image_id"].astype(str))
    if query_ids:
        mask &= ~gallery_info["image_id"].astype(str).isin(query_ids).to_numpy()
    before_series = int(mask.sum())
    query_series: set[str] = set()
    for value in query_rows["series_id"]:
        if pd.isna(value):
            continue
        normalized = str(value).strip()
        if normalized:
            query_series.add(normalized)
    if query_series:
        gallery_series = gallery_info["series_id"].fillna("").astype(str).str.strip()
        mask &= ~gallery_series.isin(query_series).to_numpy()
    excluded_same_series = before_series - int(mask.sum())
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return gallery_emb[:0], [], excluded_same_series, "no_cross_series_candidate"
    return (
        gallery_emb[indices],
        [_as_identity_key(gallery_ind[index]) for index in indices],
        excluded_same_series,
        None,
    )


def run(args) -> None:
    """在同盘暂存目录运行完整归档，成功后原子发布新批次。"""
    out_dir = Path(args.out).resolve()
    if out_dir.exists():
        raise FileExistsError(f"正式批次输出已存在，拒绝覆盖: {out_dir}")
    _validate_args(args)

    # 历史库是整条管线的只读前置条件。先验证哈希、行绑定、
    # session 与确认身份，失败时连 staging 都不创建。
    gallery = load_gallery(args.gallery_embeddings, args.gallery_meta)

    pool_artifact = None
    if args.pool:
        pool_artifact = load_verified_embedding_artifact(
            args.pool_embeddings, args.pool_meta,
            require_hashes=True, allow_nonfinite=True)
        require_generated_artifact_provenance(pool_artifact[2])
        require_compatible_embedding_configs(
            pool_artifact[2], gallery[3],
            left_name="batch", right_name="gallery")
        pool_meta = _validate_metadata(
            pool_artifact[1], "pool",
            required=("image_id", "relative_path", "session_id", "series_id"))
        gallery_info = gallery[2]
        if "series_id" not in gallery_info.columns:
            raise ValueError("gallery meta 缺少全量 manifest 派生的 series_id")
        pool_artifact = (pool_artifact[0], pool_meta, pool_artifact[2])
        _validate_crop_artifact(
            pool_meta, args.pool_crops, "pool", pool_artifact[2])

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))
    try:
        _run_staged(args, staging, gallery, pool_artifact)
        if out_dir.exists():
            raise FileExistsError(f"发布前发现正式批次输出已存在: {out_dir}")
        staging.rename(out_dir)
    except BaseException:
        # 不递归删除现场：既避免误删，也保留失败调查所需中间件。
        print(f"[pipeline] 归档失败，未发布正式批次；暂存现场保留于 {staging}")
        raise
    print(f"[pipeline] 批次已原子发布 → {out_dir}")


def _run_staged(args, out_dir: Path, gallery, pool_artifact) -> None:
    """只在新建空 staging 中执行归档，不接触任何旧批次目录。"""

    # ---------- 阶段 1-2：检测裁剪 + r4 特征（新批次）或复用散图池产物 ----------
    crops_dir = None
    if args.pool:
        pool_crops = args.pool_crops
        print("[pipeline] 散图池模式：复用预提取 r4+YOLO 特征")
        emb, meta, batch_config = pool_artifact
    else:
        # 新批次：检测裁剪 → 特征
        from whitewhale.reid.embedding import extract_embeddings, make_embedder

        supplied = getattr(args, "input_manifest_data", None)
        man = (supplied.copy() if supplied is not None
               else read_metadata_csv(args.input_manifest))
        man = _validate_metadata(man, "批次输入")
        snapshot_name = getattr(args, "input_manifest_snapshot", None)
        if snapshot_name:
            man.to_csv(out_dir / snapshot_name, index=False, encoding="utf-8-sig")
        crops_dir = out_dir / "crops"
        det_rows = detect_and_crop(man, args.images_root, crops_dir,
                                   args.det_weights, args.det_conf,
                                   args.det_imgsz, args.det_device,
                                   args.det_pad_x, args.det_pad_up,
                                   args.det_pad_down, preview=False)
        emb_path = out_dir / "embeddings.npy"
        model = make_embedder("metric-learning", metric_ckpt=args.ckpt)
        extract_embeddings(
            det_rows, model, crops_dir=crops_dir, out_path=emb_path,
            merge_from=man, missing="nan",
            model_cfg={
                "model": model.name,
                "ckpt": str(args.ckpt),
                "preprocess": model.preprocess_id,
                **yolo_crop_provenance(
                    args.det_weights, args.det_conf, args.det_imgsz,
                    args.det_pad_x, args.det_pad_up, args.det_pad_down),
            })
        meta_path = emb_path.with_name(emb_path.stem + "_meta.csv")

        emb, meta, batch_config = load_verified_embedding_artifact(
            emb_path, meta_path, require_hashes=True, allow_nonfinite=True)
        require_generated_artifact_provenance(batch_config)
        require_compatible_embedding_configs(
            batch_config, gallery[3],
            left_name="batch", right_name="gallery")

    meta = _validate_metadata(meta, "批次特征").reset_index(drop=True)
    meta["_input_order"] = np.arange(len(meta), dtype=int)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    meta["ind"] = [str(x) for x in meta["image_id"]]
    print(f"[pipeline] 批次 {len(meta)} 张（session: {meta['session_id'].value_counts().to_dict()}）")

    # ---------- 阶段 3：HDBSCAN 候选聚类（-1 = 合法噪声） ----------
    # NaN 特征行（缺失裁剪图等，missing="nan"）不进聚类，直接标噪声，
    # 避免 HDBSCAN 因 NaN 崩溃或产出异常簇；噪声行后续逐图退化处理。
    if args.pool:
        labels, probs = cluster_embeddings_by_session(
            emb, meta["session_id"], args.min_cluster_size)
    else:
        labels, probs = cluster_embeddings(emb, args.min_cluster_size)
    nan_mask = ~np.isfinite(emb).all(axis=1)
    meta["cluster"] = labels
    meta["cluster_probability"] = probs
    n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)  # 真实簇数
    n_clustered_images = int((labels >= 0).sum())
    n_noise = int((labels == -1).sum())
    print(f"[pipeline] HDBSCAN: {len(meta)} 张 → {n_clusters} "
          f"候选簇 + 噪声 {n_noise} 张（{n_noise / len(meta):.1%}）")

    # ---------- 阶段 3.5：大簇内再聚类（子簇化） ----------
    # 大簇常为混簇（多只并在一起）。对 >= subcluster_min_size 的簇内部再聚一次，
    # 拆出"纯子簇 + 残噪"，人工审核以子簇为单元（更小更纯，一键判定可行）。
    # 语义：subcluster >= 0 = 纯子簇；-1 = 残噪（逐图退化，同噪声级）。
    meta["subcluster"] = -1
    for c in set(labels):
        if c == -1:
            continue
        grp_idx = np.where(labels == c)[0]
        if len(grp_idx) >= getattr(args, "subcluster_min_size", 4):
            try:
                sub, _ = _fit_hdbscan(
                    emb[grp_idx], min_cluster_size=2, min_samples=1)
                meta.loc[meta["cluster"] == c, "subcluster"] = sub
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"簇 {c} 子簇化失败，已停止本批写出: {e}") from e
        else:
            meta.loc[meta["cluster"] == c, "subcluster"] = 0
    n_sub = int((meta["subcluster"] >= 0).sum())
    n_residual = int((meta["subcluster"] == -1).sum())
    print(f"[pipeline] 子簇化: 纯子簇单元 {n_sub} 张，残噪 {n_residual} 张"
          f"（{n_residual / len(meta):.1%}）")

    # ---------- 阶段 4：子簇级匹配历史库（多帧投票） ----------
    # 审核单元 = (cluster, subcluster)：cluster=-1 噪声 / subcluster=-1 残噪 → 逐图退化；
    # 纯子簇（subcluster>=0）→ 子簇内多帧投票（更纯，投票更可靠）。
    gal_emb, gal_ind, gal_info, _ = gallery

    # staging 必为新建空目录，代表图不会混入旧批次残留。
    rep_dst = out_dir / "representatives"
    rep_dst.mkdir(exist_ok=False)

    rows = []          # 逐图
    cluster_rows = []  # 逐子簇汇总（仅纯子簇）
    for c in sorted(set(labels)):
        for sc in sorted(meta.loc[meta["cluster"] == c, "subcluster"].unique()):
            sub = meta[(meta["cluster"] == c) & (meta["subcluster"] == sc)]
            if c == -1 or sc == -1:
                # 噪声/残噪：点互不相似，不能合并成"簇"；逐图独立匹配（单图退化）
                for _, r in sub.iterrows():
                    scoped_emb, scoped_ind, excluded_n, empty_status = _scoped_gallery(
                        r.to_frame().T, gal_emb, gal_ind, gal_info,
                        pool_mode=args.pool)
                    if empty_status is not None:
                        t1, s1, status = "", float("nan"), empty_status
                    else:
                        t1, s1, status = match_single_feature(
                            emb[r.name], scoped_emb, scoped_ind,
                            args.threshold_image)
                    rows.append({
                        "image_id": r["image_id"], "relative_path": r["relative_path"],
                        "session_id": r["session_id"], "cluster": int(c),
                        "subcluster": int(sc),
                        "series_id": r.get("series_id", ""),
                        "cluster_probability": r["cluster_probability"],
                        "top1": t1,
                        "top1_score": round(s1, 4) if np.isfinite(s1) else np.nan,
                        "vote1_ratio": 1.0 if np.isfinite(s1) else np.nan,
                        "candidate_gallery_n": int(len(scoped_emb)),
                        "excluded_same_series_n": excluded_n,
                        "status": status, "_input_order": int(r["_input_order"]),
                    })
                continue
            # 纯子簇：图-个体分数 = max → 簇内 mean（多帧投票）
            scoped_emb, scoped_ind, excluded_n, empty_status = _scoped_gallery(
                sub, gal_emb, gal_ind, gal_info, pool_mode=args.pool)
            top: list[IdentityKey] = []
            if empty_status is not None:
                t1, s1, vote1, status = (
                    "", float("nan"), float("nan"), empty_status)
            else:
                per_img = [
                    _score_feature_by_identity(emb[i], scoped_emb, scoped_ind)
                    for i in sub.index
                ]
                all_g = sorted(per_img[0].keys())
                agg = {
                    identity: float(np.mean([scores[identity] for scores in per_img]))
                    for identity in all_g
                }
                top = sorted(agg, key=agg.get, reverse=True)[: args.topk]
                t1_key = top[0]
                t1 = _identity_display(t1_key)
                s1 = agg[t1_key]
                vote1 = float(np.mean([
                    max(scores, key=scores.get) == t1_key for scores in per_img
                ]))
                status = ("match" if s1 >= args.threshold_cluster
                          else "suspected_new")
            # 代表图：与子簇均值特征最接近的一帧（归档用）
            mean_feat = np.mean(emb[sub.index], axis=0)
            rep_i = int((emb[sub.index] @ mean_feat).argmax())
            rep = sub.iloc[rep_i]
            # 复制代表图到 representatives/（新批次模式下裁剪图就在 out_dir 内）
            if args.pool:
                src = pool_crops / f"{rep['image_id']}.jpg"
            else:
                src = crops_dir / f"{rep['image_id']}.jpg"
            if not src.is_file():
                raise FileNotFoundError(f"代表图缺失，拒绝发布本批: {src}")
            shutil.copy2(src, rep_dst / f"cluster_{c:03d}_sub{sc}.jpg")

            for _, r in sub.iterrows():
                rows.append({
                    "image_id": r["image_id"], "relative_path": r["relative_path"],
                    "session_id": r["session_id"], "cluster": int(c),
                    "subcluster": int(sc),
                    "series_id": r.get("series_id", ""),
                    "cluster_probability": r["cluster_probability"],
                    "top1": t1,
                    "top1_score": round(s1, 4) if np.isfinite(s1) else np.nan,
                    "vote1_ratio": round(vote1, 2) if np.isfinite(vote1) else np.nan,
                    "candidate_gallery_n": int(len(scoped_emb)),
                    "excluded_same_series_n": excluded_n,
                    "status": status, "_input_order": int(r["_input_order"]),
                })
            cluster_rows.append({
                "session_id": str(sub.iloc[0]["session_id"]),
                "cluster": int(c), "subcluster": int(sc), "n_members": len(sub),
                "members": "; ".join(sub["image_id"]),
                "rep_image_id": rep["image_id"], "rep_relative_path": rep["relative_path"],
                "top1": t1,
                "top1_score": round(s1, 4) if np.isfinite(s1) else np.nan,
                "top2": (_identity_display(top[1]) if len(top) > 1 else ""),
                "top3": (_identity_display(top[2]) if len(top) > 2 else ""),
                "vote1_ratio": round(vote1, 2) if np.isfinite(vote1) else np.nan,
                "candidate_gallery_n": int(len(scoped_emb)),
                "excluded_same_series_n": excluded_n,
                "status": status,
            })

    # ---------- 阶段 5：汇总与落盘 ----------
    if not rows:
        raise SystemExit("[pipeline] 无任何可处理图片，请检查输入清单")
    out_img = pd.DataFrame(rows).sort_values("_input_order", kind="stable")
    out_img = out_img.drop(columns="_input_order").reset_index(drop=True)
    out_img.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8-sig")
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out_dir / "cluster_matches.csv",
                                          index=False, encoding="utf-8-sig")
    cm = pd.DataFrame(cluster_rows)
    n_pure = len(cluster_rows)
    summary = {
        "n_images": int(len(meta)), "n_clusters": n_clusters,
        "n_clustered_images": n_clustered_images, "n_noise": n_noise,
        "n_pure_subclusters": n_pure,
        "noise_ratio": round(n_noise / len(meta), 3),
        # status_counts 口径：有纯子簇时统计纯子簇状态（match/suspected_new，
        # 噪声/残噪逐图状态见 clusters.csv）；全噪声批次（无纯子簇）才统计逐图状态
        "status_counts": (cm["status"].value_counts().to_dict() if cluster_rows
                          else out_img["status"].value_counts().to_dict()),
        "subcluster_size": ({"min": int(cm["n_members"].min()),
                             "median": int(cm["n_members"].median()),
                             "max": int(cm["n_members"].max())} if cluster_rows else {}),
        "threshold": {"cluster": args.threshold_cluster, "image": args.threshold_image},
        "note": "簇 = Candidate Cluster（-1 噪声合法）；子簇 = 大簇内再聚类（subcluster>=0 纯子簇，"
                "-1 残噪逐图）；match/suspected_new 均为候选，须人工核验后才能叫个体。"
                "阈值为历史实验参考值，正式使用前需在当前模型和独立集上重标定。",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"[pipeline] 状态分布: {summary['status_counts']}")
    print(f"[pipeline] 暂存结果已完整生成 → {out_dir}")

    # ---------- 阶段 6（可选）：候选簇拼图（人工逐簇审核用） ----------
    if args.sheets:
        from whitewhale.review.contact_sheets import build_cluster_contact_sheets

        build_cluster_contact_sheets(out_dir / "clusters.csv",
                                     out_dir / "contact_sheets",
                                     args.images_root, max_sheets=args.max_sheets)
