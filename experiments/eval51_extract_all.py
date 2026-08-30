"""
E5.1 全量特征提取：9 批次 1040 张全部 YOLO 裁剪 + 度量学习特征。

正式链路（cross_time.build_gallery）只处理 20140806 两群，本脚本把它
泛化到全量 manifest，供 E5.1 全库同体检索评估使用。

流程：manifest 全量 → YOLO 检测裁剪（detect_and_crop，未检出回退中心窗）
→ embedding（extract_embeddings）→ 输出 npy + meta（merge pilot 个体标签）。

同协议比较已有模型时，可用 --reuse-crops-from 显式复用一套已生成特征产物
所绑定的裁剪目录。复用前会校验源产物哈希、裁剪参数、清单行序、检测框、
标签/分串信息、裁剪图集合与修改时间，并把复用来源和裁剪内容摘要写入新 config。

--ckpt 指定权重（r3 默认；r4 重训后传 outputs/metric_learning/r4/best.pt），
--out 指定输出文件（默认 embeddings_eval51_all.npy，r4 用别名避免覆盖）。
"""
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.data.manifest import compute_sha256  # noqa: E402
from whitewhale.data.sequence_groups import annotate_series  # noqa: E402
from whitewhale.detection.detector import (  # noqa: E402
    detect_and_crop, yolo_crop_provenance)
from whitewhale.reid.embedding import (  # noqa: E402
    embedding_config_path,
    extract_embeddings,
    load_verified_embedding_artifact,
    make_embedder,
    read_metadata_csv,
    require_generated_artifact_provenance,
)

BASE = REPO_ROOT
MANIFEST = BASE / "outputs" / "index" / "dataset_manifest.csv"
PILOT = BASE / "outputs" / "pilot" / "pilot_set.csv"
DET = BASE / "models" / "detectors" / "yolov8n_dorsalfin.pt"
CROPS = BASE / "outputs" / "crops_yolo_eval51"
DEFAULT_OUT = BASE / "outputs" / "embeddings" / "embeddings_eval51_all.npy"

YOLO_EXACT_FIELDS = (
    "crop", "crop_schema_version", "detector_checkpoint_sha256",
    "detector_fallback_policy",
)
YOLO_NUMERIC_FIELDS = (
    "detector_conf", "detector_imgsz", "detector_pad_x",
    "detector_pad_up", "detector_pad_down",
)


def resolve_data_root(config: dict | None = None) -> Path:
    """读取统一配置的数据根；相对路径始终锚定仓库根。"""
    cfg = load_config("pipeline") if config is None else config
    root = Path(cfg.get("data_root", "src_dataset"))
    return root if root.is_absolute() else REPO_ROOT / root


def _require_same_yolo_provenance(source: dict, expected: dict) -> None:
    """确认复用裁剪的检测器、阈值、尺寸和扩框参数与当前协议完全一致。"""
    for key in YOLO_EXACT_FIELDS:
        if source.get(key) in (None, ""):
            raise ValueError(f"裁剪源 config 缺少 {key}")
        if str(source[key]) != str(expected[key]):
            raise ValueError(
                f"裁剪源与当前协议不一致：{key}="
                f"{source[key]!r} / {expected[key]!r}")
    for key in YOLO_NUMERIC_FIELDS:
        if source.get(key) in (None, ""):
            raise ValueError(f"裁剪源 config 缺少 {key}")
        try:
            matches = np.isclose(
                float(source[key]), float(expected[key]), rtol=0.0, atol=1e-12)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"裁剪源 config 的 {key} 不是数值") from exc
        if not matches:
            raise ValueError(
                f"裁剪源与当前协议不一致：{key}="
                f"{source[key]!r} / {expected[key]!r}")


def _normalized_bool(values: pd.Series) -> list[bool]:
    """把 CSV 中的布尔字段规范化，拒绝含糊或非法值。"""
    mapping = {"true": True, "1": True, "false": False, "0": False}
    normalized = []
    for value in values:
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError(f"裁剪清单 fallback 含非法布尔值: {value!r}")
        normalized.append(mapping[key])
    return normalized


def _require_equal_columns(left: pd.DataFrame, right: pd.DataFrame,
                           columns: tuple[str, ...], label: str) -> None:
    """按行比较溯源字段；数值允许 CSV 往返产生的类型差异。"""
    if len(left) != len(right):
        raise ValueError(f"{label} 行数不一致: {len(left)} / {len(right)}")
    missing = [column for column in columns
               if column not in left.columns or column not in right.columns]
    if missing:
        raise ValueError(f"{label} 缺少字段: {missing}")
    numeric = {"x", "y", "w", "h", "det_conf", "frame"}
    for column in columns:
        if column == "fallback":
            matches = (_normalized_bool(left[column]) ==
                       _normalized_bool(right[column]))
        elif column in numeric:
            left_values = pd.to_numeric(left[column], errors="coerce").to_numpy()
            right_values = pd.to_numeric(right[column], errors="coerce").to_numpy()
            matches = bool(np.allclose(
                left_values, right_values, rtol=0.0, atol=1e-12,
                equal_nan=True))
        else:
            matches = (left[column].fillna("").astype(str).tolist() ==
                       right[column].fillna("").astype(str).tolist())
        if not matches:
            raise ValueError(f"{label} 的 {column} 与源产物不一致")


def _crop_bundle_digest(crops_dir: Path, image_ids: list[str],
                        manifest_hash: str) -> str:
    """计算裁剪清单和全部 JPG 的稳定组合摘要，绑定本次实际输入。"""
    digest = hashlib.sha256(b"whitewhale-eval-crop-bundle-v1\0")
    digest.update(manifest_hash.encode("ascii"))
    for image_id in image_ids:
        digest.update(b"\0")
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(compute_sha256(crops_dir / f"{image_id}.jpg").encode("ascii"))
    return digest.hexdigest()


def validate_reusable_crops(
    crops_dir: Path,
    expected_manifest: pd.DataFrame,
    source_embeddings: Path,
    expected_crop_config: dict,
) -> tuple[pd.DataFrame, dict]:
    """验证固定裁剪可安全复用，并返回裁剪清单及待写入的新产物溯源。"""
    crops_dir = Path(crops_dir)
    source_embeddings = Path(source_embeddings)
    crop_manifest_path = crops_dir / "crops_manifest.csv"
    if not crops_dir.is_dir() or not crop_manifest_path.is_file():
        raise FileNotFoundError(
            f"复用裁剪目录缺少 crops_manifest.csv: {crops_dir}")

    _, source_meta, source_config = load_verified_embedding_artifact(
        source_embeddings, require_hashes=True)
    require_generated_artifact_provenance(source_config)
    _require_same_yolo_provenance(source_config, expected_crop_config)

    crop_manifest = read_metadata_csv(crop_manifest_path)
    if crop_manifest.empty or crop_manifest["image_id"].duplicated().any():
        raise ValueError("复用 crops_manifest.image_id 为空或重复")
    if len(crop_manifest) != len(expected_manifest):
        raise ValueError(
            f"复用裁剪数量 {len(crop_manifest)} ≠ 当前 manifest "
            f"{len(expected_manifest)}")

    base_columns = (
        "image_id", "relative_path", "session_id", "x", "y", "w", "h",
        "det_conf", "fallback",
    )
    _require_equal_columns(
        crop_manifest, source_meta, base_columns, "裁剪清单/源特征 meta")
    _require_equal_columns(
        crop_manifest, expected_manifest,
        ("image_id", "relative_path", "session_id"),
        "裁剪清单/当前完整 manifest")
    _require_equal_columns(
        source_meta, expected_manifest,
        ("image_id", "relative_path", "session_id", "confirmed_identity",
         "individual_id", "series_id", "sequence_key", "frame"),
        "源特征 meta/当前评测标签与分串")

    image_ids = crop_manifest["image_id"].astype(str).tolist()
    expected_names = {f"{image_id}.jpg" for image_id in image_ids}
    actual_names = {path.name for path in crops_dir.glob("*.jpg")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"复用裁剪图集合不一致：缺失 {len(missing)}，多余 {len(extra)}")

    created_at = source_config.get("created_at_utc")
    try:
        source_created_ts = datetime.fromisoformat(str(created_at)).timestamp()
    except (TypeError, ValueError) as exc:
        raise ValueError("裁剪源 config.created_at_utc 非法") from exc
    for row in crop_manifest.itertuples(index=False):
        crop_path = crops_dir / f"{row.image_id}.jpg"
        if crop_path.stat().st_mtime > source_created_ts:
            raise ValueError(
                f"裁剪图晚于源特征产物生成时间，可能已被修改: {crop_path.name}")
        with Image.open(crop_path) as image:
            if image.size != (int(row.w), int(row.h)):
                raise ValueError(
                    f"裁剪图尺寸与清单不一致: {crop_path.name} "
                    f"{image.size} != {(int(row.w), int(row.h))}")

    crop_manifest_hash = compute_sha256(crop_manifest_path)
    source_config_path = embedding_config_path(source_embeddings)
    reuse_provenance = {
        "crop_reuse_mode": "validated_existing_directory",
        "crop_source_embedding_file": str(source_embeddings.resolve()),
        "crop_source_embedding_sha256": source_config["embedding_sha256"],
        "crop_source_meta_sha256": source_config["meta_sha256"],
        "crop_source_config_file": str(source_config_path.resolve()),
        "crop_source_config_sha256": compute_sha256(source_config_path),
        "crop_source_created_at_utc": source_config["created_at_utc"],
        "crop_manifest_file": str(crop_manifest_path.resolve()),
        "crop_manifest_sha256": crop_manifest_hash,
        "crop_bundle_digest_algorithm": (
            "sha256(manifest_sha256 + ordered(image_id, jpg_sha256))"),
        "crop_bundle_sha256": _crop_bundle_digest(
            crops_dir, image_ids, crop_manifest_hash),
    }
    return crop_manifest, reuse_provenance


def main():
    ap = argparse.ArgumentParser(description="E5.1 全量特征提取")
    ap.add_argument("--ckpt", type=Path,
                    default=BASE / "outputs" / "metric_learning" / "r3" / "best.pt")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--crops", type=Path, default=CROPS,
                    help="版本化裁剪图目录；默认要求不存在")
    ap.add_argument(
        "--reuse-crops-from", type=Path,
        help=("显式复用 --crops，并传入原先使用该目录生成的 embedding .npy；"
              "会严格校验三件套、裁剪清单、完整 manifest 和修改时间"))
    args = ap.parse_args()
    CKPT, OUT, crops_dir = args.ckpt, args.out, args.crops
    protected = [
        OUT,
        OUT.with_name(f"{OUT.stem}_meta.csv"),
        OUT.with_name(f"{OUT.stem}_config.json"),
    ]
    if args.reuse_crops_from is None:
        protected.append(crops_dir)
    existing = [path for path in protected if path.exists()]
    if existing:
        raise SystemExit(
            f"FATAL: 评测产物已存在，拒绝覆盖或混用: {existing[0]}；"
            "请使用新的 --out/--crops 版本名")

    pipeline_cfg = load_config("pipeline")
    detector_cfg = pipeline_cfg.get("detector", {})
    crop_cfg = pipeline_cfg.get("crop", {})
    det_conf = float(detector_cfg.get("conf", 0.25))
    det_imgsz = int(detector_cfg.get("imgsz", 1024))
    det_device = detector_cfg.get("device", "auto")
    pad_x = float(crop_cfg.get("pad_x", 0.30))
    pad_up = float(crop_cfg.get("pad_up", 0.15))
    pad_down = float(crop_cfg.get("pad_down", 0.60))

    m = read_metadata_csv(MANIFEST)
    annotate_series(m)
    pilot = read_metadata_csv(PILOT)
    identity_col = ("confirmed_identity" if "confirmed_identity" in pilot.columns
                    else "individual_id")
    if pilot["image_id"].duplicated().any():
        raise ValueError("pilot.image_id 重复，无法唯一回填身份")
    identities = pilot.set_index("image_id")[identity_col].astype(str)
    m["individual_id"] = m["image_id"].map(identities).fillna("")
    m["confirmed_identity"] = m["individual_id"]
    # relative_path 需含 session 前缀（与 pilot_set / cross_time 约定一致）
    m = m.assign(relative_path=m["session_id"] + "/" + m["relative_path"])
    print(f"[extract] 全量清单 {len(m)} 张（9 批次）| 权重 {CKPT}")

    crop_provenance = yolo_crop_provenance(
        DET, det_conf, det_imgsz, pad_x, pad_up, pad_down)
    reuse_provenance = {}
    if args.reuse_crops_from is None:
        man = detect_and_crop(
            m, resolve_data_root(pipeline_cfg), crops_dir, DET,
            conf=det_conf, imgsz=det_imgsz, device=det_device,
            pad_x=pad_x, pad_up=pad_up, pad_down=pad_down, preview=False)
    else:
        man, reuse_provenance = validate_reusable_crops(
            crops_dir, m, args.reuse_crops_from, crop_provenance)
        print(
            f"[extract] 已验证并复用固定裁剪 {len(man)} 张 | "
            f"来源 {args.reuse_crops_from}")
    model = make_embedder("metric-learning", metric_ckpt=CKPT)
    extract_embeddings(
        man, model, crops_dir=crops_dir, out_path=OUT, missing="error",
        merge_from=m,
        model_cfg={
            "model": model.name,
            "ckpt": str(CKPT),
            "preprocess": model.preprocess_id,
            **crop_provenance,
            **reuse_provenance,
        })

    # 个体标签：labeled 才有（loose_known / ignored 无归属 → NaN，仅作干扰项）。
    # 标签在统一提取函数写 meta/config 哈希前按 image_id 合并，避免哈希随即失效。
    _, meta, config = load_verified_embedding_artifact(
        OUT, OUT.with_name(OUT.stem + "_meta.csv"), require_hashes=True)
    require_generated_artifact_provenance(config)
    print(f"[done] 特征 {len(meta)} 张 → {OUT}")
    labeled = meta["individual_id"].astype(str).str.strip().ne("")
    print(f"[done] 带个体标签 {int(labeled.sum())} 张")


if __name__ == "__main__":
    main()
