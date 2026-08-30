"""
批内归档管线（正式入口 2/7）：新批次 → 检测裁剪 → r4 特征 → 候选聚类 →
子簇化 → 簇级多帧投票匹配历史库 → 审核清单。

用法：
    python scripts/run_pipeline.py --pool                      # 散图池验证（复用预提取产物）
    python scripts/run_pipeline.py --input-manifest 清单.csv \
        --session 批次名 --batch-name 批次名                    # 从全量清单选择新批次

输出（out/batch_name/ 下）：clusters.csv（逐图）/ cluster_matches.csv（簇级）/
representatives/（代表图）/ summary.json / contact_sheets/（--sheets）。

数据语义：簇 = Candidate Cluster（-1 噪声合法）；match/suspected_new 均为
候选，人工核验后才能叫个体。默认阈值沿用历史实验参考值（E5 簇级 0.58 /
E4 单图 0.50）；正式使用前必须在当前模型和独立评估集上重新标定。
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.data.image_store import validate_safe_image_ids  # noqa: E402
from whitewhale.pipeline.archival import run  # noqa: E402


_SAFE_BATCH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,127}\Z", re.ASCII)
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _repo_path(path: Path) -> Path:
    """CLI 相对路径统一按仓库根解析，不依赖启动时工作目录。"""
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_batch_output(out_root: Path, batch_name: str) -> Path:
    """校验 Windows 安全目录名，并保证批次输出仍是 out_root 的直属子目录。"""
    batch_path = Path(batch_name)
    if batch_path.anchor or not _SAFE_BATCH_NAME.fullmatch(batch_name):
        raise ValueError(
            "--batch-name 只允许 ASCII 字母、数字、空格、点、下划线和连字符，"
            "且必须是单个目录名")
    if batch_name.endswith((".", " ")):
        raise ValueError("--batch-name 不能以点或空格结尾（Windows 路径不安全）")
    device_stem = batch_name.split(".", 1)[0].rstrip(" ").upper()
    if device_stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"--batch-name {batch_name!r} 是 Windows 保留名")

    root = out_root.resolve()
    candidate = (root / batch_name).resolve()
    if candidate.parent != root:
        raise ValueError("--batch-name 导致输出越过 --out 根目录")
    return candidate


def select_manifest_session(manifest_path: Path, session: str | None):
    """校验清单批次；显式指定 session 时只返回该批次的行。"""
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if manifest.empty:
        raise ValueError(f"输入 manifest 为空: {manifest_path}")
    if "image_id" not in manifest.columns:
        raise ValueError("输入 manifest 缺少 image_id")

    requested = session.strip() if session else ""
    if session is not None and not requested:
        raise ValueError("--session 不能为空")
    if "session_id" not in manifest.columns:
        raise ValueError("输入 manifest 缺少 session_id，正式批次拒绝继续")

    session_values = manifest["session_id"].astype(str).str.strip()
    available = sorted(value for value in session_values.unique() if value)
    has_blank = bool((session_values == "").any())
    if not available:
        raise ValueError("输入 manifest 的 session_id 全为空，正式批次拒绝继续")
    if requested:
        if requested not in available:
            raise ValueError(
                f"--session {requested!r} 不在 manifest 中；可用批次: {available}")
        selected = manifest.loc[session_values == requested].copy()
        validate_safe_image_ids(selected["image_id"])
        return selected, requested

    if has_blank and available:
        raise ValueError("manifest 的 session_id 部分为空；请修正清单或显式指定 --session")
    if len(available) > 1:
        raise ValueError(
            f"manifest 含多个批次 {available}；必须用 --session 明确选择一个批次")
    validate_safe_image_ids(manifest["image_id"])
    return manifest, available[0] if available else None


def main():
    cfg = load_config("pipeline")
    output_root = Path(cfg.get("output_root", "outputs"))
    base = output_root if output_root.is_absolute() else REPO_ROOT / output_root
    retr = cfg.get("retrieval", {})
    clut = cfg.get("clustering", {})
    crop = cfg.get("crop", {})
    det = cfg.get("detector", {})
    query_cfg = cfg.get("query", {})
    pool_cfg = cfg.get("pool", {})
    data_root = Path(cfg.get("data_root", "src_dataset"))
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    parser = argparse.ArgumentParser(description="批内簇级归档管线")
    parser.add_argument("--pool", action="store_true",
                        help="散图池模式：复用 pipeline.yaml 指定的 r4+YOLO 产物")
    parser.add_argument("--input-manifest", type=Path, default=None,
                        help="新批次图片清单（image_id, relative_path[, session_id]）")
    parser.add_argument("--session", default=None,
                        help="从多批次 manifest 中显式选择一个 session_id（新批次模式）")
    parser.add_argument("--batch-name", default="batch",
                        help="批次标识（输出目录名）")
    parser.add_argument("--images-root", type=Path,
                        default=data_root,
                        help="图片根目录（只读）")
    parser.add_argument("--ckpt", type=Path,
                        default=REPO_ROOT / cfg.get("reid_checkpoint",
                                                    "outputs/metric_learning/r4/best.pt"),
                        help="r4 微调权重（新批次模式）")
    parser.add_argument("--gallery-embeddings", type=Path,
                        default=base / query_cfg.get(
                            "embeddings",
                            "artifacts/r4_yolocrop_v3/gallery/embeddings.npy"),
                        help="历史库特征（已确认个体）")
    parser.add_argument("--gallery-meta", type=Path,
                        default=base / query_cfg.get(
                            "meta",
                            "artifacts/r4_yolocrop_v3/gallery/embeddings_meta.csv"))
    parser.add_argument("--pool-embeddings", type=Path,
                        default=base / pool_cfg.get(
                            "embeddings",
                            "artifacts/r4_yolocrop_v3/pool/embeddings.npy"),
                        help="散图池模式复用的特征")
    parser.add_argument("--pool-meta", type=Path,
                        default=base / pool_cfg.get(
                            "meta",
                            "artifacts/r4_yolocrop_v3/pool/embeddings_meta.csv"))
    parser.add_argument("--pool-crops", type=Path,
                        default=base / pool_cfg.get(
                            "crops", "artifacts/r4_yolocrop_v3/pool/crops"),
                        help="散图池模式复制代表图时使用的裁剪目录")
    parser.add_argument("--min-cluster-size", type=int, default=clut.get("min_cluster_size", 3))
    parser.add_argument("--subcluster-min-size", type=int,
                        default=clut.get("subcluster_min_size", 4),
                        help="大于等于此成员的簇才做内部子簇化")
    parser.add_argument("--topk", type=int, default=retr.get("topk", 3))
    parser.add_argument("--threshold-cluster", type=float,
                        default=retr.get("threshold_cluster", 0.58),
                        help="簇级多帧投票阈值（历史 E5 参考值；需在当前模型上重标定）")
    parser.add_argument("--threshold-image", type=float,
                        default=retr.get("threshold_image", 0.50),
                        help="单图阈值（历史 E4 参考值；噪声/残噪退化用，需重标定）")
    parser.add_argument("--out", type=Path, default=base / "cluster_archival")
    parser.add_argument("--sheets", action="store_true", help="生成候选簇拼图（人工审核）")
    parser.add_argument("--max-sheets", type=int, default=200)
    parser.add_argument("--det-weights", type=Path,
                        default=REPO_ROOT / cfg.get("detector_checkpoint",
                                                    "models/detectors/yolov8n_dorsalfin.pt"))
    parser.add_argument("--det-conf", type=float, default=det.get("conf", 0.25))
    parser.add_argument("--det-imgsz", type=int, default=det.get("imgsz", 1024))
    parser.add_argument("--det-device", default=det.get("device", "auto"),
                        help="检测设备：auto / cpu / cuda / CUDA 编号（默认 auto）")
    parser.add_argument("--det-pad-x", type=float, default=crop.get("pad_x", 0.30))
    parser.add_argument("--det-pad-up", type=float, default=crop.get("pad_up", 0.15))
    parser.add_argument("--det-pad-down", type=float, default=crop.get("pad_down", 0.60))
    args = parser.parse_args()

    if args.pool == bool(args.input_manifest):
        parser.error("必须且只能指定其一：--pool（散图池）或 --input-manifest（新批次）")
    if args.pool and args.session is not None:
        parser.error("--session 只适用于 --input-manifest 模式")

    for field in (
            "images_root", "ckpt", "gallery_embeddings", "gallery_meta",
            "pool_embeddings", "pool_meta", "pool_crops", "out", "det_weights"):
        setattr(args, field, _repo_path(Path(getattr(args, field))))
    if args.input_manifest is not None:
        args.input_manifest = _repo_path(args.input_manifest)

    # 输出目录 = out/batch_name（archival.run 约定 out 为完整输出目录）
    try:
        args.out = resolve_batch_output(args.out, args.batch_name)
    except ValueError as exc:
        parser.error(str(exc))
    if args.out.exists():
        parser.error(f"正式批次输出已存在，拒绝覆盖: {args.out}")
    if args.input_manifest is not None:
        try:
            selected, _ = select_manifest_session(args.input_manifest, args.session)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            parser.error(str(exc))
        # 筛选结果只在内存中传给归档管线；正式目录由管线在
        # 同盘 staging 完整成功后一次性发布，避免预先落下半成品清单。
        args.input_manifest_data = selected
        args.input_manifest_snapshot = (
            "input_manifest_selected.csv" if args.session is not None else None)
    run(args)


if __name__ == "__main__":
    main()
