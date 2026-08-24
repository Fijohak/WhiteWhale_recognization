"""
本地图库个体查询客户端（Web）。

功能（A+B 三态输出）：
- 上传单张照片 → YOLO 背鳍检测裁剪（未检出回退整图）→ r3 特征 → 全库 Top-K 检索；
- 三态判定（固定阈值）：
    known   → 最高相似度 ≥ 阈值，展示 Top-K 候选；
    unknown → 最高相似度 < 阈值，提示"疑似未知个体（可能新个体）"，
              仍返回 Top-K 供人工参考；
- 候选展示：缩略图 + 相似度 + 来源（source_group / session / quality / cluster）；
- 所有候选一律标注"待人工核验"，不自动判定身份。

链路（实验 E1/E2/E4 结论）：gallery 与查询统一为 "YOLO 检测裁剪 + r3 跨群
HN 微调特征"（散图场景检测裁剪显著更优、特写图打平，避免混合语义）。
阈值默认 0.55 = E4 标定 FA≤5% 区间 0.5-0.6 中值。

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

from whitewhale.detection.detector import expand_box
from whitewhale.reid.embedding import (DINOv2Adapter, MegaDescriptorAdapter,
                                       MegaDescriptorMetricAdapter)
from whitewhale.reid.retrieval import cosine_topk


def load_gallery(embeddings_path: Path, meta_path: Path, pilot_path: Path):
    """加载 gallery：embedding + meta + pilot 追溯字段，返回 (emb, info_df)。

    pilot 清单提供 source_group/cluster 等追溯字段；文件不存在时降级
    （只用 meta 自带字段），不阻塞查询。
    """
    emb = np.load(embeddings_path)
    meta = pd.read_csv(meta_path)
    assert len(emb) == len(meta), f"embedding {len(emb)} 与 meta {len(meta)} 数量不一致"
    df = meta
    if pilot_path is not None and Path(pilot_path).exists():
        pilot = pd.read_csv(pilot_path)
        df = meta.merge(pilot, on="image_id", how="left", suffixes=("", "_pilot"))
        assert len(df) == len(meta), "merge 后行数变化，请检查 image_id 是否唯一"
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
    candidates = [
        Path(embeddings_path).with_name(Path(embeddings_path).stem + "_config.json"),
        Path(embeddings_path).parent / "embedding_config.json",
    ]
    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8")).get("model", "unknown")
            except Exception:
                continue
    return "unknown"


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
        return args.model
    for candidate in ("metric-learning", "dinov2", "megadescriptor"):
        if _model_family_matches(candidate, gallery_model):
            return candidate
    raise SystemExit(
        f"无法确定查询模型：gallery 特征由 [{gallery_model}] 提取（--model 可显式指定）。")


def build_app(args, embedder=None, detector=None) -> FastAPI:
    """构建 FastAPI 应用（延迟构建，便于测试传参数）。

    embedder: 可选注入的特征提取器（测试用 mock，避免加载真实模型）。
    detector: 可选注入的检测器（测试用 mock；None 且 args.detect 时懒加载 YOLO）。
    """
    emb, info = load_gallery(args.embeddings, args.meta, args.pilot)
    gallery_ids = info["image_id"].tolist()
    images_root = Path(args.images_root)
    model_name = resolve_model(args, emb.shape[1])

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
        """上传图片 → YOLO 检测裁剪 → 特征 → 全库 Top-K → 三态结果。"""
        raw = await file.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": f"图片无法读取: {e}"}, status_code=400)

        crop_info = {"detect": False, "fallback": False}
        if detector is not None:
            w, h = img.size
            res = detector.predict(img, conf=args.det_conf, imgsz=args.det_imgsz,
                                   device=args.det_device, verbose=False)
            if len(res) and len(res[0].boxes):
                b = res[0].boxes[0]  # 最高置信度框（ultralytics 按 conf 排序）
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
                box = expand_box(x0, y0, x1, y1, w, h,
                                 args.det_pad_x, args.det_pad_up, args.det_pad_down)
                img = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                crop_info = {"detect": True, "fallback": False}
            else:
                crop_info = {"detect": True, "fallback": True}  # 未检出 → 整图直查

        q_emb = model.encode([img])          # (1, D) L2 归一化
        scores, idx = cosine_topk(q_emb, emb, k=args.k)
        s = scores[0]
        top_idx = idx[0]

        max_score = float(s[0])
        threshold = args.threshold
        if max_score >= threshold:
            status = "known"          # 候选，仍需人工核验
        else:
            status = "unknown"        # 疑似未知个体（可能新个体）

        candidates = []
        for j, gi in enumerate(top_idx):
            row = info.iloc[gi]
            candidates.append({
                "rank": j + 1,
                "image_id": row["image_id"],
                "score": float(s[j]),
                "source_group": _safe(row, "source_group"),
                "session_id": _safe(row, "session_id"),
                "quality_band": _safe(row, "quality_band"),
                "cluster": _safe(row, "cluster"),
                "relative_path": str(row["relative_path"]),
            })

        return {
            "status": status,
            "threshold": threshold,
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
        p = images_root / hit.iloc[0]["relative_path"]
        if not p.exists():
            return Response(content=f"图片不存在: {p}", status_code=404)
        return Response(p.read_bytes(), media_type="image/jpeg")

    return app


def _safe(row: pd.Series, col: str) -> str:
    """取字段，NaN/None → 空字符串。"""
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)
