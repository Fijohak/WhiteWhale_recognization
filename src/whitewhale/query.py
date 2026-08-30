"""
本地图库个体查询客户端（Web）。

功能（A+B 二态候选提示）：
- 上传单张照片 → YOLO 背鳍检测裁剪（未检出回退整图）→ r4 特征 → 全库 Top-K 检索；
- 二态提示（单阈值）：
    known   → 最高相似度 ≥ 阈值，展示 Top-K 候选；
    unknown → 最高相似度 < 阈值，提示"疑似未知个体（可能新个体）"，
              仍返回 Top-K 供人工参考；
- 候选展示：缩略图 + 相似度 + 来源（source_group / session / quality / cluster）；
- 所有候选一律标注"待人工核验"，不自动判定身份。

链路（实验 E1/E2/E4 结论）：gallery 与查询统一为 "YOLO 检测裁剪 + r4
微调特征"（散图场景检测裁剪显著更优、特写图打平，避免混合语义）。
阈值默认 0.55 来自历史 E4 实验，仅作当前产物的参考值；更换模型、裁剪方式或
独立评估集后必须重新标定，不能把历史 FA 结果直接外推到新数据。

语义约束：
- 目录名/源分组只是弱信息（source_group），不作为身份；
- 簇号是 Candidate Cluster，人工确认前不叫个体；
- 查询模型必须与 gallery 特征同模型（同族同权重），维度不一致时报错提示。

CLI 入口见 scripts/launch_query.py。
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

from whitewhale.data.image_store import ImageStore
from whitewhale.data.manifest import compute_sha256
from whitewhale.detection.detector import (center_fallback_box, expand_box,
                                           resolve_yolo_device,
                                           yolo_crop_provenance)
from whitewhale.reid.embedding import (DINOv2Adapter, MegaDescriptorAdapter,
                                       MegaDescriptorMetricAdapter,
                                       embedding_config_path,
                                       load_verified_embedding_artifact,
                                       require_generated_artifact_provenance)
from whitewhale.reid.retrieval import cosine_topk

MODEL_PREPROCESS = {
    "metric-learning": "Resize256+CenterCrop224",
    "megadescriptor": "Resize224",
    "dinov2": "Resize518",
}


def load_gallery(embeddings_path: Path, meta_path: Path, pilot_path: Path):
    """加载 gallery：embedding + meta + pilot 追溯字段，返回 (emb, info_df)。

    pilot 清单提供 source_group/cluster 等追溯字段；文件不存在时降级
    （只用 meta 自带字段），不阻塞查询。
    """
    emb, meta, _ = load_verified_embedding_artifact(
        embeddings_path, meta_path, require_hashes=True)
    df = meta
    if pilot_path is not None and Path(pilot_path).exists():
        pilot = pd.read_csv(pilot_path, dtype=str, keep_default_na=False)
        df = meta.merge(pilot, on="image_id", how="left", suffixes=("", "_pilot"))
        if len(df) != len(meta):
            raise ValueError("pilot.image_id 不唯一，merge 后行数变化，拒绝加载查询图库")
    if "confirmed_identity" not in df.columns:
        raise ValueError("gallery 缺少 confirmed_identity，无法生成个体级候选")
    invalid_identity = (
        df["confirmed_identity"].isna()
        | (df["confirmed_identity"].astype(str).str.strip() == "")
    )
    if invalid_identity.any():
        raise ValueError(
            f"gallery 有 {int(invalid_identity.sum())} 张图缺少 confirmed_identity，"
            "拒绝生成不可追溯的个体候选"
        )
    if "session_id" not in df.columns:
        raise ValueError("gallery 缺少 session_id，无法隔离批次内身份")
    invalid_session = (
        df["session_id"].isna()
        | (df["session_id"].astype(str).str.strip() == "")
    )
    if invalid_session.any():
        raise ValueError(
            f"gallery 有 {int(invalid_session.sum())} 张图缺少 session_id，"
            "拒绝聚合可能碰撞的批次内身份"
        )
    # 归一化兜底（gallery 应已 L2 归一化）
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb, df


def _gallery_model(embeddings_path: Path) -> str:
    """读取 gallery 特征模型来源。

    优先读 {stem}_config.json（统一提取接口的产物），回退旧固定名
    embedding_config.json（早期脚本产物），都没有 → "unknown"。
    """
    cfg_path = embedding_config_path(Path(embeddings_path))
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8")).get("model", "unknown")
        except Exception:
            pass
    return "unknown"


def _gallery_config(embeddings_path: Path) -> dict:
    """读取已由 artifact loader 校验过的 gallery config。"""
    config_path = embedding_config_path(Path(embeddings_path))
    if not config_path.is_file():
        raise SystemExit(f"gallery config 不存在: {config_path}")
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"gallery config 无法读取: {exc}") from exc


def _validate_metric_checkpoint(args, gallery_model: str) -> None:
    """metric gallery 必须与查询权重版本和哈希完全一致。"""
    if "metric-learning" not in gallery_model:
        return
    checkpoint = Path(args.metric_ckpt) if args.metric_ckpt is not None else None
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit(f"查询权重不存在，无法验证 gallery 模型: {checkpoint}")
    query_model = f"megadescriptor-metric-learning-{checkpoint.parent.name}"
    if gallery_model != query_model:
        raise SystemExit(
            f"模型版本不匹配：gallery={gallery_model}，查询权重={query_model}")
    config_path = embedding_config_path(Path(args.embeddings))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = config.get("checkpoint_sha256")
    if not expected_hash:
        raise SystemExit("gallery config 缺少 checkpoint_sha256，拒绝仅凭版本名比较")
    if compute_sha256(checkpoint) != expected_hash:
        raise SystemExit("查询权重 SHA-256 与 gallery 提取权重不一致，拒绝跨权重比较")


def _model_family_matches(query: str, gallery: str) -> bool:
    """查询模型与 gallery 特征是否同源（同族同权重）。

    微调特征（metric-learning）与预训练特征分布不同，不可混配；
    megadescriptor 用精确匹配避免被 metric-learning 前缀误配。
    """
    if query == "metric-learning":
        return "metric-learning" in gallery
    if query == "dinov2":
        return "dinov2" in gallery
    if query == "megadescriptor":
        return gallery in ("megadescriptor", "hf-hub:BVRA/MegaDescriptor-T-224")
    return False


def resolve_model(args, emb_dim: int) -> str:
    """决定查询模型名，返回 "megadescriptor" / "dinov2" / "metric-learning"。

    默认自动匹配 gallery 特征模型（读 embedding 旁 config）；
    --model 显式指定时校验必须与 gallery 同源，否则拒绝（防错配）。
    """
    gallery_model = _gallery_model(args.embeddings)

    if args.model is not None:
        if not _model_family_matches(args.model, gallery_model):
            raise SystemExit(
                f"模型不匹配：gallery 特征由 [{gallery_model}] 提取，"
                f"查询用 [{args.model}] 特征分布不同，不可比较。")
        selected = args.model
        if selected == "metric-learning":
            _validate_metric_checkpoint(args, gallery_model)
        return selected
    for candidate in ("metric-learning", "dinov2", "megadescriptor"):
        if _model_family_matches(candidate, gallery_model):
            if candidate == "metric-learning":
                _validate_metric_checkpoint(args, gallery_model)
            return candidate
    raise SystemExit(
        f"无法确定查询模型：gallery 特征由 [{gallery_model}] 提取（--model 可显式指定）。")


def _validate_query_pipeline(args, model_name: str, config: dict) -> None:
    """验证查询裁剪、预处理及本地权重与 gallery 生成链路完全兼容。"""
    require_generated_artifact_provenance(config)
    crop = str(config.get("crop", "")).strip().lower()
    preprocess = str(config.get("preprocess", "")).strip()
    if not crop or not preprocess:
        raise SystemExit("gallery config 缺少 crop/preprocess，无法验证查询链路")
    expected_crop = "yolo" if args.detect else "whole"
    if crop != expected_crop:
        raise SystemExit(
            f"查询裁剪与 gallery 不兼容：gallery crop={crop!r}，"
            f"当前查询 crop={expected_crop!r}")
    expected_preprocess = MODEL_PREPROCESS[model_name]
    if preprocess != expected_preprocess:
        raise SystemExit(
            f"查询预处理与 gallery 不兼容：gallery preprocess={preprocess!r}，"
            f"当前模型 preprocess={expected_preprocess!r}")

    if crop == "yolo":
        expected_crop_config = yolo_crop_provenance(
            args.det_weights, args.det_conf, args.det_imgsz,
            args.det_pad_x, args.det_pad_up, args.det_pad_down)
        for key, expected in expected_crop_config.items():
            if key == "detector_checkpoint_file":
                continue
            actual = config.get(key)
            if actual is None:
                raise SystemExit(f"gallery config 缺少 {key}，无法验证检测裁剪链路")
            if isinstance(expected, float):
                matches = np.isclose(float(actual), expected, rtol=0.0, atol=1e-12)
            else:
                matches = str(actual) == str(expected)
            if not matches:
                raise SystemExit(
                    f"查询检测裁剪与 gallery 不兼容：{key}={actual!r} / {expected!r}")

    if model_name != "dinov2":
        return
    expected_hash = config.get("checkpoint_sha256")
    weight_value = getattr(args, "dinov2_weight", None)
    if expected_hash:
        weight = Path(weight_value) if weight_value else None
        if weight is None or not weight.is_file():
            raise SystemExit("DINOv2 gallery 记录了本地权重哈希，查询必须提供同一权重文件")
        if compute_sha256(weight) != str(expected_hash):
            raise SystemExit("DINOv2 查询权重 SHA-256 与 gallery 不一致")
    elif weight_value:
        raise SystemExit("gallery 未记录权重哈希，拒绝改用无法核对的自定义 DINOv2 权重")


def build_app(args, embedder=None, detector=None) -> FastAPI:
    """构建 FastAPI 应用（延迟构建，便于测试传参数）。

    embedder: 可选注入的特征提取器（测试用 mock，避免加载真实模型）。
    detector: 可选注入的检测器（测试用 mock；None 且 args.detect 时懒加载 YOLO）。
    """
    if int(args.k) < 1:
        raise ValueError("k 必须至少为 1")
    if not np.isfinite(float(args.threshold)) or not -1.0 <= float(args.threshold) <= 1.0:
        raise ValueError("threshold 必须是 [-1, 1] 内的有限余弦值")
    emb, info = load_gallery(args.embeddings, args.meta, args.pilot)
    gallery_ids = info["image_id"].tolist()
    store = ImageStore(args.images_root)
    model_name = resolve_model(args, emb.shape[1])
    _validate_query_pipeline(args, model_name, _gallery_config(args.embeddings))

    # 本地离线工具：权重从本地缓存加载（HF_HUB_OFFLINE=1），不访问外网。
    # 缓存缺失时 timm 会给出明确的权重找不到错误。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # 模型：默认与 gallery 特征一致（微调特征 → 加载对应权重）
    if embedder is not None:
        model = embedder
    elif model_name == "dinov2":
        model = DINOv2Adapter(weight_path=args.dinov2_weight)
    elif model_name == "metric-learning":
        model = MegaDescriptorMetricAdapter(ckpt_path=args.metric_ckpt)
    else:
        model = MegaDescriptorAdapter()
    if model.feat_dim != emb.shape[1]:
        raise SystemExit(
            f"查询模型维度 {model.feat_dim} ≠ gallery 特征维度 {emb.shape[1]}："
            f"跨模型特征不可比，请用与 gallery 相同的模型（当前 gallery 为 "
            f"{args.embeddings} 旁 config 记录的特征）。")

    # YOLO 背鳍检测裁剪：查询图先检测再裁剪（E2：散图场景显著更优，特写图打平）
    if args.detect:
        if detector is None:
            from ultralytics import YOLO
            detector = YOLO(str(args.det_weights))
    else:
        detector = None

    app = FastAPI(title="中华白海豚个体查询")
    app.state.n_gallery = len(info)
    app.state.model_name = model.name

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (Path(__file__).parent / "query_app.html").read_text(encoding="utf-8")
        return html

    @app.get("/api/gallery_size")
    def gallery_size():
        return {"n": len(info), "model": model.name}

    @app.post("/api/query")
    async def query_image(file: UploadFile = File(...)):
        """上传图片 → YOLO 检测裁剪 → 特征 → 全库 Top-K → 二态候选提示。"""
        raw = await file.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": f"图片无法读取: {e}"}, status_code=400)

        crop_info = {"detect": False, "fallback": False}
        if detector is not None:
            w, h = img.size
            res = detector.predict(img, conf=args.det_conf, imgsz=args.det_imgsz,
                                   device=resolve_yolo_device(args.det_device),
                                   verbose=False)
            if len(res) and len(res[0].boxes):
                b = res[0].boxes[0]  # 最高置信度框（ultralytics 按 conf 排序）
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
                box = expand_box(x0, y0, x1, y1, w, h,
                                 args.det_pad_x, args.det_pad_up, args.det_pad_down)
                img = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                crop_info = {"detect": True, "fallback": False}
            else:
                box = center_fallback_box(w, h)
                img = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                crop_info = {"detect": True, "fallback": True}

        q_emb = model.encode([img])          # (1, D) L2 归一化
        # 先排序全部图像，再按已确认个体去重；否则只取图像
        # Top-K 会让同一个体的多张照片挤掉其他个体。
        scores, idx = cosine_topk(q_emb, emb, k=len(info))
        s = scores[0]
        top_idx = idx[0]
        identity_best = []
        seen = set()
        for score, gi in zip(s, top_idx):
            row = info.iloc[gi]
            identity = str(row["confirmed_identity"]).strip()
            session = str(row["session_id"]).strip()
            identity_key = (session, identity)
            if identity_key in seen:
                continue
            seen.add(identity_key)
            identity_best.append((identity_key, float(score), int(gi)))
            if len(identity_best) >= args.k:
                break

        max_score = identity_best[0][1]
        threshold = args.threshold
        if max_score >= threshold:
            status = "known"          # 候选，仍需人工核验
        else:
            status = "unknown"        # 疑似未知个体（可能新个体）

        candidates = []
        for j, ((session, identity), score, gi) in enumerate(identity_best):
            row = info.iloc[gi]
            candidates.append({
                "rank": j + 1,
                "confirmed_identity": identity,
                "identity_display": f"{session}::{identity}",
                "image_id": row["image_id"],
                "score": score,
                "source_group": _safe(row, "source_group"),
                "session_id": session,
                "quality_band": _safe(row, "quality_band"),
                "cluster": _safe(row, "cluster"),
                "relative_path": str(row["relative_path"]),
            })

        return {
            "status": status,
            "threshold": threshold,
            "calibration_status": "provisional_unvalidated",
            "threshold_warning": (
                "当前阈值仅为历史实验参考值，尚未用独立 known/unknown 集按全库最大分数标定"
            ),
            "max_score": max_score,
            "n_gallery": len(info),
            "crop": crop_info,
            "candidates": candidates,
        }

    @app.get("/api/image/{image_id}")
    def image_file(image_id: str):
        """按 image_id 返回原图（供候选缩略图展示）。"""
        hit = info[info["image_id"] == image_id]
        if hit.empty:
            return Response(status_code=404)
        rel = str(hit.iloc[0]["relative_path"])
        if not store.exists(rel):
            return Response(content=f"图片不存在: {store.resolve(rel)}", status_code=404)
        return Response(store.read_bytes(rel), media_type="image/jpeg")

    return app


def _safe(row: pd.Series, col: str) -> str:
    """取字段，NaN/None → 空字符串。"""
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)
