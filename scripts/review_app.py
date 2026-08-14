"""
中华白海豚个体识别：人工审核网页（面向领域专家，无需学习 FiftyOne）。

界面：按候选簇分组显示照片，每张照片下三个中文操作：
- 个体名 + [确认]：归属某个体（如 CI-001，同一只海豚的照片必须用同一个名字）；
- [不确定]：无法判断；
- [排除]：确认不是任何已审核个体。
操作即时写入 outputs/review/review_annotations.csv（可追溯、可恢复），
审核完成后运行 --export 导出 Confirmed Individual 雏形
（outputs/review/confirmed_individuals.csv）。

数据语义：簇号是 Candidate Cluster，审核确认后才叫个体；
"个体名"只是审核过程中的标签（CI-xxx），与正式个体 ID 的对应由审核者约定。

用法：
    python scripts/review_app.py --port 8001
    浏览器打开 http://127.0.0.1:8001
    审完后：
    python scripts/review_app.py --export
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 审核标注约定：confirmed = CI-xxx 个体名；uncertain / reject 为特殊状态
UNCERTAIN = "uncertain"
REJECT = "reject"


def load_photos(clusters_csv: Path) -> pd.DataFrame:
    """加载候选簇照片表（行 = 待审核照片）。"""
    df = pd.read_csv(clusters_csv)
    for col in ("cluster", "individual_id", "source_group", "session_id",
                "quality_band", "relative_path"):
        if col not in df.columns:
            df[col] = ""
    return df


def load_annotations(csv_path: Path) -> dict:
    """读取已审核标注：{image_id: label}。文件不存在或为空 → 空。"""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return {}
    return dict(zip(df["image_id"], df["label"]))


def load_embeddings(embeddings_path, meta_path):
    """加载特征库（按 image_id 对齐）用于相似度辅助；缺失时返回 None（功能降级）。"""
    import numpy as np

    if not embeddings_path or not meta_path:
        return None, {}
    if not Path(embeddings_path).exists() or not Path(meta_path).exists():
        return None, {}
    try:
        emb = np.load(embeddings_path)
        m = pd.read_csv(meta_path)
        if len(emb) != len(m) or "image_id" not in m.columns:
            return None, {}
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb, {iid: i for i, iid in enumerate(m["image_id"])}
    except Exception:
        return None, {}


def save_annotations(csv_path: Path, photos: pd.DataFrame, ann: dict) -> None:
    """全量写回标注（原子写：临时文件 + replace，防审核中崩溃丢数据）。"""
    rows = [{"image_id": p, "label": a} for p, a in ann.items() if a]
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(csv_path)


def build_app(args, photos: pd.DataFrame | None = None) -> FastAPI:
    """构建审核应用（photos 可注入，便于测试）。"""
    photos = photos if photos is not None else load_photos(args.clusters)
    ann: dict = load_annotations(args.annotations)
    images_root = Path(args.images_root)
    _img_cache: dict[str, bytes] = {}
    # 特征库（相似度辅助，可选：缺失时审核功能照常，只是没有相似提醒）
    emb, emb_idx = load_embeddings(
        getattr(args, "embeddings", None), getattr(args, "embeddings_meta", None))

    app = FastAPI(title="中华白海豚个体审核")

    def photo_list(cluster_filter: str | None = None) -> list[dict]:
        # 已命名个体（相似度辅助的比对目标；ann 动态变化，每次现算）
        named = [(iid, lab) for iid, lab in ann.items()
                 if lab and lab not in (UNCERTAIN, REJECT) and iid in emb_idx]
        sim_all = None
        if emb is not None and named:
            sim_all = emb @ emb[[emb_idx[iid] for iid, _ in named]].T  # (N, M)
        items = []
        for _, r in photos.iterrows():
            iid = str(r["image_id"])
            cl = int(r["cluster"]) if str(r["cluster"]).lstrip("-").isdigit() else -1
            label = ann.get(iid, "")
            if cluster_filter and cluster_filter != "all":
                if cluster_filter == "noise":
                    if cl != -1:
                        continue
                elif cluster_filter == "unreviewed":
                    if label:
                        continue
                elif cluster_filter == "reviewed":
                    if not label:
                        continue
                elif cluster_filter.isdigit() and cl != int(cluster_filter):
                    continue
            # 与已命名个体的最相似 Top-2（排除自身，低于 0.40 不提示）
            similar = []
            if sim_all is not None and iid in emb_idx:
                row = sim_all[emb_idx[iid]]
                for j in np.argsort(-row)[:2]:
                    if named[j][0] == iid:
                        continue
                    if float(row[j]) < 0.40:
                        continue
                    similar.append({"name": named[j][1],
                                    "score": round(float(row[j]), 2)})
            items.append({
                "image_id": iid,
                "cluster": cl,
                "source_group": str(r.get("individual_id", "")),
                "session_id": str(r.get("session_id", "")),
                "quality_band": str(r.get("quality_band", "")),
                "label": label,
                "similar": similar,
            })
        return items

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (Path(__file__).parent / "review_app.html").read_text(encoding="utf-8")
        return html

    @app.get("/api/state")
    def state(filter: str = "all"):
        """照片列表（?filter=all|noise|unreviewed|reviewed|簇号）。"""
        from collections import Counter
        cnt = Counter(int(r["cluster"]) if str(r["cluster"]).lstrip("-").isdigit() else -1
                      for _, r in photos.iterrows())
        labels = {k: v for k, v in ann.items() if v}
        return {
            "n_total": len(photos),
            "n_reviewed": len(labels),
            "clusters": {str(k): v for k, v in sorted(cnt.items())},
            "photos": photo_list(filter),
        }

    @app.post("/api/annotate")
    async def annotate(request: Request):
        """提交标注：{image_id, action: confirm|uncertain|reject|clear, identity}。"""
        body = await request.json()
        image_id = str(body.get("image_id", ""))
        if image_id not in set(photos["image_id"]):
            return JSONResponse({"error": f"未知 image_id: {image_id}"}, status_code=400)
        action = body.get("action", "")
        if action == "confirm":
            ident = str(body.get("identity", "")).strip()
            if not ident:
                return JSONResponse({"error": "确认需填写个体名"}, status_code=400)
            ann[image_id] = ident
        elif action == "uncertain":
            ann[image_id] = UNCERTAIN
        elif action == "reject":
            ann[image_id] = REJECT
        elif action == "clear":
            ann.pop(image_id, None)
        else:
            return JSONResponse({"error": f"未知操作: {action}"}, status_code=400)
        save_annotations(args.annotations, photos, ann)
        return {"image_id": image_id, "label": ann.get(image_id, ""),
                "n_reviewed": len([v for v in ann.values() if v])}

    @app.get("/api/image/{image_id}")
    def image_file(image_id: str, full: int = 0):
        """缩略图（宽 ≤ 480px）或原图（?full=1，审核放大看背鳍细节，不压缩）。"""
        if image_id in _img_cache:
            return Response(_img_cache[image_id], media_type="image/jpeg")
        hit = photos[photos["image_id"] == image_id]
        if hit.empty:
            return Response(status_code=404)
        p = images_root / str(hit.iloc[0]["relative_path"])
        if not p.exists():
            return Response(content=f"图片不存在: {p}", status_code=404)
        data = p.read_bytes()
        if not full:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img = ImageOps.exif_transpose(img)
            if img.width > 480:
                img = img.resize((480, int(img.height * 480 / img.width)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            data = buf.getvalue()
            _img_cache[image_id] = data
        # 原图按真实扩展名给媒体类型（多数为 JPG）
        ext = Path(str(hit.iloc[0]["relative_path"])).suffix.lower().lstrip(".")
        media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "bmp": "image/bmp"}.get(ext, "image/jpeg")
        return Response(data, media_type=media)

    return app


def export_confirmed(args) -> None:
    """导出审核结果：label 为个体名（CI-*）或非特殊状态的行。"""
    photos = load_photos(args.clusters)
    ann = load_annotations(args.annotations)
    rows = []
    for _, r in photos.iterrows():
        label = ann.get(r["image_id"], "")
        if not label or label in (UNCERTAIN, REJECT):
            continue
        rows.append({
            "image_id": r["image_id"],
            "confirmed_identity": label,
            "status": "confirmed",
            "cluster": r.get("cluster", ""),
            "source_group": r.get("individual_id", ""),
            "session_id": r.get("session_id", ""),
            "source_path": r.get("relative_path", ""),
        })
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    n_id = out["confirmed_identity"].nunique() if len(out) else 0
    print(f"[review] 导出 {len(out)} 条已确认 → {args.out}")
    print(f"[review] 个体数: {n_id}，个体: {sorted(out['confirmed_identity'].unique()) if n_id else '无'}")
    # 提醒未审/特殊状态
    total, done = len(photos), len(ann)
    print(f"[review] 进度: 已审 {done}/{total}"
          f"（未审 {total - done}，uncertain {sum(1 for v in ann.values() if v == UNCERTAIN)}，"
          f"reject {sum(1 for v in ann.values() if v == REJECT)}）")


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="中华白海豚个体识别：人工审核网页")
    parser.add_argument("--clusters", type=Path, default=base / "clusters" / "clusters.csv",
                        help="候选簇照片表（行 = 待审核照片）")
    parser.add_argument("--annotations", type=Path, default=base / "review" / "review_annotations.csv",
                        help="审核标注保存位置（每次操作即时写入）")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"),
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--embeddings", type=Path, default=base / "embeddings" / "embeddings.npy",
                        help="特征库（相似度提醒用，可选）")
    parser.add_argument("--embeddings-meta", type=Path, default=base / "embeddings" / "embeddings_meta.csv",
                        help="特征库 meta（含 image_id）")
    parser.add_argument("--cluster-filter", default="all",
                        help="启动时默认筛选：all / noise / 簇号（如 1）")
    parser.add_argument("--export", action="store_true", help="导出审核结果后退出")
    parser.add_argument("--out", type=Path, default=base / "review" / "confirmed_individuals.csv",
                        help="导出路径（--export 时生效）")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.export:
        export_confirmed(args)
        return

    import uvicorn
    app = build_app(args)
    print(f"[review] 审核网页就绪: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
