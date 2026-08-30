"""
r4 正式检索产物重建入口。

从完整 dataset manifest 一次性生成稳定连拍串，再用同一 YOLO 配置和同一 r4
checkpoint 构建 gallery/pool。全部结果先写入同盘暂存目录，验证通过后才发布为一个
不可覆盖的版本化目录，避免 embedding、meta、config 或裁剪图新旧混用。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.data.image_store import validate_safe_image_ids  # noqa: E402
from whitewhale.data.manifest import compute_sha256  # noqa: E402
from whitewhale.data.sequence_groups import annotate_series  # noqa: E402
from whitewhale.detection.detector import (  # noqa: E402
    detect_and_crop,
    yolo_crop_provenance,
)
from whitewhale.pipeline.archival import crop_bundle_provenance  # noqa: E402
from whitewhale.reid.embedding import (  # noqa: E402
    extract_embeddings,
    load_verified_embedding_artifact,
    make_embedder,
    read_metadata_csv,
    require_compatible_embedding_configs,
    require_generated_artifact_provenance,
)


def _repo_path(path: Path) -> Path:
    """把相对路径稳定解析到仓库根目录。"""
    return path if path.is_absolute() else REPO_ROOT / path


def _require_within(root: Path, candidate: Path, label: str) -> Path:
    """限制生成产物只能位于配置的 output_root 内。"""
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} 必须位于 output_root 内: {candidate}") from exc
    return candidate


def prepare_sources(manifest_path: Path, pilot_path: Path,
                    gallery_sessions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从全量 manifest 派生稳定 series，并返回 gallery/pool 两份可追溯清单。"""
    full = read_metadata_csv(manifest_path)
    required = {"image_id", "relative_path", "filename", "session_id", "label_status"}
    missing = sorted(required - set(full.columns))
    if missing:
        raise ValueError(f"dataset manifest 缺少必需列: {missing}")
    if full.empty:
        raise ValueError("dataset manifest 为空")
    validate_safe_image_ids(full["image_id"])
    annotate_series(full)

    pilot = read_metadata_csv(pilot_path)
    if "image_id" not in pilot.columns:
        raise ValueError("pilot 缺少 image_id")
    if pilot["image_id"].duplicated().any():
        raise ValueError("pilot.image_id 重复，无法建立唯一身份映射")
    identity_col = (
        "confirmed_identity" if "confirmed_identity" in pilot.columns
        else "individual_id" if "individual_id" in pilot.columns
        else None
    )
    if identity_col is None:
        raise ValueError("pilot 缺少 confirmed_identity/individual_id")
    identities = pilot.set_index("image_id")[identity_col].astype(str)
    full["confirmed_identity"] = full["image_id"].map(identities).fillna("")
    full["individual_id"] = full["confirmed_identity"]

    expected_labeled = set(full.loc[full["label_status"] == "labeled", "image_id"])
    pilot_images = set(pilot["image_id"])
    if pilot_images != expected_labeled:
        raise ValueError(
            "pilot 与完整 manifest 的 labeled 图片不一致："
            f"pilot_only={len(pilot_images - expected_labeled)}, "
            f"manifest_only={len(expected_labeled - pilot_images)}")

    known_sessions = set(full["session_id"].astype(str))
    absent = [session for session in gallery_sessions if session not in known_sessions]
    if absent:
        raise ValueError(f"gallery session 不在 manifest 中: {absent}")
    gallery = full[
        full["session_id"].isin(gallery_sessions)
        & full["label_status"].eq("labeled")
    ].copy()
    pool = full[full["label_status"].eq("loose_known")].copy()
    if gallery.empty or pool.empty:
        raise ValueError(
            f"gallery/pool 不得为空：gallery={len(gallery)}, pool={len(pool)}")
    if gallery["confirmed_identity"].str.strip().eq("").any():
        raise ValueError("gallery 含未映射 confirmed_identity，拒绝生成不完整个体库")

    # 原始 manifest 的 relative_path 是 session 内路径；正式 ImageStore 根目录位于其上一级。
    for frame in (gallery, pool):
        prefixed = frame["session_id"].astype(str) + "/"
        relative = frame["relative_path"].astype(str).str.replace("\\", "/", regex=False)
        already_prefixed = pd.Series(
            [value.startswith(prefix) for value, prefix in zip(relative, prefixed)],
            index=frame.index,
        )
        frame["relative_path"] = relative.where(
            already_prefixed, prefixed + relative)
        frame.reset_index(drop=True, inplace=True)
    return gallery, pool


def _build_subset(name: str, source: pd.DataFrame, stage: Path,
                  *, images_root: Path, detector_weights: Path,
                  checkpoint: Path, detector_cfg: dict, crop_cfg: dict,
                  model) -> tuple[Path, Path, dict]:
    """在版本包暂存目录中生成一个 gallery 或 pool 子产物。"""
    subset_dir = stage / name
    subset_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = subset_dir / "source_manifest.csv"
    source.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    crops_dir = subset_dir / "crops"
    detected = detect_and_crop(
        source, images_root, crops_dir, detector_weights,
        conf=float(detector_cfg["conf"]),
        imgsz=int(detector_cfg["imgsz"]),
        device=detector_cfg.get("device", "auto"),
        pad_x=float(crop_cfg["pad_x"]),
        pad_up=float(crop_cfg["pad_up"]),
        pad_down=float(crop_cfg["pad_down"]),
        preview=False,
    )
    embeddings_path = subset_dir / "embeddings.npy"
    extract_embeddings(
        detected,
        model,
        crops_dir=crops_dir,
        out_path=embeddings_path,
        merge_from=source,
        missing="error",
        model_cfg={
            "model": model.name,
            "ckpt": str(checkpoint),
            "preprocess": model.preprocess_id,
            **yolo_crop_provenance(
                detector_weights,
                float(detector_cfg["conf"]),
                int(detector_cfg["imgsz"]),
                float(crop_cfg["pad_x"]),
                float(crop_cfg["pad_up"]),
                float(crop_cfg["pad_down"]),
            ),
        },
    )
    meta_path = subset_dir / "embeddings_meta.csv"
    embeddings, meta, config = load_verified_embedding_artifact(
        embeddings_path, meta_path, require_hashes=True)
    require_generated_artifact_provenance(config)
    if len(embeddings) != len(source) or len(meta) != len(source):
        raise RuntimeError(f"{name} 产物行数与输入不一致")
    if meta["image_id"].tolist() != source["image_id"].tolist():
        raise RuntimeError(f"{name} 产物行顺序与输入 image_id 不一致")
    config.update(crop_bundle_provenance(meta, crops_dir))
    config_path = embeddings_path.with_name(
        f"{embeddings_path.stem}_config.json")
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return embeddings_path, meta_path, config


def build_bundle(args: argparse.Namespace) -> Path:
    """构建并验证完整版本包，成功后原子发布目录。"""
    pipeline_cfg = load_config("pipeline")
    output_root = _repo_path(Path(pipeline_cfg.get("output_root", "outputs"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = _require_within(output_root, args.out, "--out")
    if destination.exists():
        raise FileExistsError(
            f"版本化产物目录已存在，拒绝覆盖: {destination}；请更换 --out 版本名")
    destination.parent.mkdir(parents=True, exist_ok=True)

    detector_cfg = pipeline_cfg.get("detector", {})
    crop_cfg = pipeline_cfg.get("crop", {})
    for section, values, required in (
        ("detector", detector_cfg, ("conf", "imgsz")),
        ("crop", crop_cfg, ("pad_x", "pad_up", "pad_down")),
    ):
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"pipeline.{section} 缺少参数: {missing}")

    gallery_sessions = [str(value) for value in pipeline_cfg.get(
        "cross_time", {}).get("gallery_sessions", [])]
    if not gallery_sessions:
        raise ValueError("pipeline.cross_time.gallery_sessions 为空")
    gallery, pool = prepare_sources(
        args.manifest, args.pilot, gallery_sessions)
    model = make_embedder("metric-learning", metric_ckpt=args.checkpoint)

    with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent) as temporary:
        stage = Path(temporary)
        _, _, gallery_config = _build_subset(
            "gallery", gallery, stage,
            images_root=args.images_root,
            detector_weights=args.detector_weights,
            checkpoint=args.checkpoint,
            detector_cfg=detector_cfg,
            crop_cfg=crop_cfg,
            model=model,
        )
        _, _, pool_config = _build_subset(
            "pool", pool, stage,
            images_root=args.images_root,
            detector_weights=args.detector_weights,
            checkpoint=args.checkpoint,
            detector_cfg=detector_cfg,
            crop_cfg=crop_cfg,
            model=model,
        )
        require_compatible_embedding_configs(
            pool_config, gallery_config, left_name="pool", right_name="gallery")
        summary = {
            "artifact_version": destination.name,
            "gallery_sessions": gallery_sessions,
            "gallery_images": int(len(gallery)),
            "gallery_identities": int(gallery[[
                "session_id", "confirmed_identity"]].drop_duplicates().shape[0]),
            "gallery_series": int(gallery["series_id"].replace("", pd.NA).nunique()),
            "pool_images": int(len(pool)),
            "pool_series": int(pool["series_id"].replace("", pd.NA).nunique()),
            "manifest_sha256": compute_sha256(args.manifest),
            "pilot_sha256": compute_sha256(args.pilot),
            "checkpoint_sha256": gallery_config["checkpoint_sha256"],
            "detector_checkpoint_sha256": gallery_config[
                "detector_checkpoint_sha256"],
            "gallery_crop_bundle_sha256": gallery_config["crop_bundle_sha256"],
            "pool_crop_bundle_sha256": pool_config["crop_bundle_sha256"],
            "provenance": "generated_with_row_binding",
        }
        (stage / "bundle.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """创建 CLI 参数解析器。"""
    pipeline_cfg = load_config("pipeline")
    output_root = _repo_path(Path(pipeline_cfg.get("output_root", "outputs")))
    parser = argparse.ArgumentParser(description="重建严格溯源的 r4 gallery/pool 产物包")
    parser.add_argument(
        "--manifest", type=Path,
        default=output_root / "index" / "dataset_manifest.csv")
    parser.add_argument(
        "--pilot", type=Path,
        default=output_root / "pilot" / "pilot_set.csv")
    parser.add_argument(
        "--images-root", type=Path,
        default=_repo_path(Path(pipeline_cfg.get("data_root", "src_dataset"))))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=_repo_path(Path(pipeline_cfg["reid_checkpoint"])))
    parser.add_argument(
        "--detector-weights", type=Path,
        default=_repo_path(Path(pipeline_cfg["detector_checkpoint"])))
    parser.add_argument(
        "--out", type=Path,
        default=output_root / "artifacts" / "r4_yolocrop_v3")
    return parser


def main() -> None:
    """解析路径并执行不可覆盖的版本化重建。"""
    args = build_parser().parse_args()
    for field in ("manifest", "pilot", "images_root", "checkpoint",
                  "detector_weights", "out"):
        setattr(args, field, _repo_path(Path(getattr(args, field))))
    try:
        destination = build_bundle(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(f"[artifact] r4 gallery/pool 严格产物已发布: {destination}")


if __name__ == "__main__":
    main()
