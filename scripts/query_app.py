"""
本地图库个体查询客户端（Web）。

功能（A+B 三态输出）：
- 上传单张照片 → 提取特征 → 全库 Top-K 检索；
- 三态判定（固定阈值）：
    known   → 最高相似度 ≥ 阈值，展示 Top-K 候选；
    unknown → 最高相似度 < 阈值，提示"疑似未知个体（可能新个体）"，
              仍返回 Top-K 供人工参考；
- 候选展示：缩略图 + 相似度 + 来源（source_group / session / quality / cluster）；
- 所有候选一律标注"待人工核验"，不自动判定身份（CLAUDE.md 语义）。

语义约束：
- 目录名/源分组只是弱信息（source_group），不作为身份；
- 簇号是 Candidate Cluster，人工确认前不叫个体；
- 查询模型必须与 gallery 特征同模型（默认 MegaDescriptor），
  维度不一致时报错提示（跨模型特征不可比）。

用法：
    python scripts/query_app.py --port 8000
    浏览器打开 http://127.0.0.1:8000
    （DINOv2 查询：--model dinov2 --dinov2-weight D:/dolphin_data/dinov2_weights/dinov2_vitb14_pretrain.pth，
     但 gallery 特征需先用 DINOv2 重新提取，否则维度/分布不匹配）
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reid.embedding.base import DINOv2Adapter, MegaDescriptorAdapter  # noqa: E402
from src.reid.retrieval.cosine import cosine_topk  # noqa: E402


def load_gallery(embeddings_path: Path, meta_path: Path, pilot_path: Path):
    """加载 gallery：embedding + meta + pilot 追溯字段，返回 (emb, info_df)。"""
    emb = np.load(embeddings_path)
    meta = pd.read_csv(meta_path)
    pilot = pd.read_csv(pilot_path)
    assert len(emb) == len(meta), f"embedding {len(emb)} 与 meta {len(meta)} 数量不一致"
    df = meta.merge(pilot, on="image_id", how="left", suffixes=("", "_pilot"))
    assert len(df) == len(meta), "merge 后行数变化，请检查 image_id 是否唯一"
    # 归一化兜底（gallery 应已 L2 归一化）
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    return emb, df


def check_model_match(args, emb_dim: int) -> str:
    """校验查询模型与 gallery 特征同源（同模型），返回模型名。

    不同模型即使维度相同（如 MegaDescriptor/DINOv2 都是 768D），
    特征分布也不同，不可直接比较——必须与 embedding_config.json
    记录的提取模型一致。
    """
    cfg_path = Path(args.embeddings).parent / "embedding_config.json"
    gallery_model = "unknown"
    if cfg_path.exists():
        import json
        gallery_model = json.loads(cfg_path.read_text(encoding="utf-8")).get("model", "unknown")

    if args.model == "megadescriptor":
        compatible = ("megadescriptor", "hf-hub:BVRA/MegaDescriptor-T-224")
        if not any(m in str(gallery_model) for m in compatible):
            raise SystemExit(
                f"模型不匹配：gallery 特征由 [{gallery_model}] 提取，"
                f"查询用 [{args.model}] 特征分布不同，不可比较。"
                f"请先提取同模型 gallery 特征。")
        return "megadescriptor"
    if args.model == "dinov2":
        if "dinov2" not in str(gallery_model):
            raise SystemExit(
                f"模型不匹配：gallery 特征由 [{gallery_model}] 提取，"
                f"查询用 [dinov2] 特征分布不同，不可比较。"
                f"请先提取同模型 gallery 特征。")
        return "dinov2"
    raise SystemExit(f"未知模型: {args.model}")


def build_app(args, embedder=None) -> FastAPI:
    """构建 FastAPI 应用（延迟构建，便于测试传参数）。

    embedder: 可选注入的特征提取器（测试用 mock，避免加载真实模型）。
    """
    emb, info = load_gallery(args.embeddings, args.meta, args.pilot)
    gallery_ids = info["image_id"].tolist()
    images_root = Path(args.images_root)
    model_name = check_model_match(args, emb.shape[1])

    # 本地离线工具：权重从本地缓存加载（HF_HUB_OFFLINE=1），不访问外网。
    # 缓存缺失时 timm 会给出明确的权重找不到错误。
    os.environ.setdefault("HF_HUB_OFFLINE", "1") # 坏事做尽！！！

    # 模型：默认 MegaDescriptor（与 gallery 特征一致）；dinov2 需显式传权重
    if embedder is not None:
        model = embedder
    elif model_name == "dinov2":
        model = DINOv2Adapter(weight_path=args.dinov2_weight)
    else:
        model = MegaDescriptorAdapter()
    if model.feat_dim != emb.shape[1]:
        raise SystemExit(
            f"查询模型维度 {model.feat_dim} ≠ gallery 特征维度 {emb.shape[1]}："
            f"跨模型特征不可比，请用与 gallery 相同的模型（当前 gallery 为 "
            f"outputs/embeddings/embedding_config.json 记录的特征）。")

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
        """上传图片 → 特征 → 全库 Top-K → 三态结果。"""
        raw = await file.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": f"图片无法读取: {e}"}, status_code=400)

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


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="本地图库个体查询 Web 客户端")
    parser.add_argument("--embeddings", type=Path, default=base / "embeddings" / "embeddings.npy")
    parser.add_argument("--meta", type=Path, default=base / "embeddings" / "embeddings_meta.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"),
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--model", choices=["megadescriptor", "dinov2"],
                        default="megadescriptor")
    parser.add_argument("--dinov2-weight", type=str, default=None,
                        help="DINOv2 官方权重 .pth（--model dinov2 时必填）")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="三态判定阈值：最高分低于此值 → unknown（疑似未知个体）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    app = build_app(args)
    print(f"[query_app] gallery {app.state.n_gallery} 张 / 模型 {app.state.model_name} / "
          f"阈值 {args.threshold}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
