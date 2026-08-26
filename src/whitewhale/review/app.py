"""
人工审核网页（面向领域专家，无需学习 FiftyOne）。

界面按候选簇（子簇单元）分组显示照片，三种操作即时写入
outputs/review/review_annotations.csv（可追溯、可恢复）：
- 个体名 + [确认]：归属某个体（如 CI-001，同一只海豚必须同名）；
- [不确定] / [排除]：特殊状态（uncertain / reject，不产出确认记录）。

数据语义（项目红线）：
- 簇号 = Candidate Cluster，审核确认后才叫个体；
- uncertain / reject 必须保留（不能强制分配）；
- export_confirmed 只导出确认行 → Confirmed Individual 雏形。

CLI 入口见 scripts/launch_review.py。
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageOps

# 审核标注约定：confirmed = CI-xxx 个体名；uncertain / reject 为特殊状态
UNCERTAIN = "uncertain"
REJECT = "reject"


def _fmt(v):
    """数值格式化：空 / NaN → 空字符串，否则保留 4 位小数（供前端直接展示）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        s = str(v)
        return s if s not in ("", "nan") else ""
    return round(f, 4) if not np.isnan(f) else ""


def _int_col(v):
    """数字列解析：兼容 int/float/字符串（含 "1.0" 浮点写法）；解析失败 → -1（噪声/残噪语义）。"""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return -1


def load_photos(clusters_csv: Path) -> pd.DataFrame:
    """加载候选簇照片表（行 = 待审核照片）。"""
    df = pd.read_csv(clusters_csv)
    for col in ("cluster", "individual_id", "source_group", "session_id",
                "quality_band", "relative_path"):
        if col not in df.columns:
            df[col] = ""
    # 子簇列（子簇化管线产物）：旧格式无此列 → 整簇视为一个子簇（0），噪声为 -1
    if "subcluster" not in df.columns:
        def _old_sc(x):
            try:
                return -1 if int(float(x)) == -1 else 0
            except (TypeError, ValueError):
                return -1
        df["subcluster"] = df["cluster"].apply(_old_sc)
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


def save_annotations(csv_path: Path, photos: pd.DataFrame, ann: dict,
                     reviewer: str = "") -> None:
    """全量写回标注（原子写：临时文件 + replace，防审核中崩溃丢数据）。"""
    rows = [{"image_id": p, "label": a, "reviewer": reviewer}
            for p, a in ann.items() if a]
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
    # 历史库对照照片目录（跨时间审核用，按个体分文件夹）；未指定 → 不提供对照
    history_root = Path(args.history_lookup) if getattr(args, "history_lookup", None) else None
    # 历史对照图质量表（filename → clear/low；未提供 → 全部默认清晰，低质图前端可隐藏）
    hist_quality: dict[str, str] = {}
    if getattr(args, "history_quality", None) and Path(args.history_quality).exists():
        try:
            hq = pd.read_csv(args.history_quality, dtype={"filename": str})
            hist_quality = dict(zip(hq["filename"], hq["quality"]))
        except Exception:
            hist_quality = {}
    # 批次特征库（簇内相似度辅助：把混簇中可疑的"离群者"沉底标红；可选）
    batch_emb, batch_idx = None, {}
    if getattr(args, "batch_embeddings", None) and Path(args.batch_embeddings).exists():
        meta_p = Path(args.batch_embeddings).with_name(
            Path(args.batch_embeddings).stem + "_meta.csv")
        try:
            be = np.load(args.batch_embeddings)
            bm = pd.read_csv(meta_p)
            if len(be) == len(bm) and "image_id" in bm.columns:
                be = be / np.linalg.norm(be, axis=1, keepdims=True)
                batch_emb = be
                batch_idx = {str(iid): i for i, iid in enumerate(bm["image_id"])}
        except Exception:
            batch_emb = None
    # 每张照片与其子簇内均值的相似度（低 = 可能是混入的别的个体）
    # 按 (cluster, subcluster) 分组：同一簇的不同子簇可能是不同个体，不能共用一个混均值
    in_sim: dict[str, float] = {}
    if batch_emb is not None:
        for (cl, sc), grp in photos.groupby(["cluster", "subcluster"]):
            if _int_col(cl) == -1 or _int_col(sc) == -1 or len(grp) < 2:
                continue
            rows = [(str(r["image_id"]), i)
                    for i, r in grp.iterrows()
                    if str(r["image_id"]) in batch_idx]
            if len(rows) < 2:
                continue
            ids, idxs = zip(*rows)
            sub = batch_emb[list(idxs)]
            sims = sub @ sub.mean(axis=0)
            for iid, s in zip(ids, sims):
                in_sim[iid] = round(float(s), 3)
    # 特征库（相似度辅助，可选：缺失时审核功能照常，只是没有相似提醒）
    emb, emb_idx = load_embeddings(
        getattr(args, "embeddings", None), getattr(args, "embeddings_meta", None))

    app = FastAPI(title="中华白海豚个体审核")

    def gkey_of(cl: int, sc: int) -> str:
        """审核单元分组键：噪声 → "noise"；子簇 → "簇号.子簇号"（残噪 -1 也在内）。"""
        return "noise" if cl == -1 else f"{cl}.{sc}"

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
            cl = _int_col(r["cluster"])
            sc = _int_col(r["subcluster"])
            gk = gkey_of(cl, sc)
            label = ann.get(iid, "")
            if cluster_filter and cluster_filter != "all":
                if cluster_filter == "noise":
                    # 噪声池 + 残噪子簇（逐图处理的单元都归入"噪声"筛选）
                    if cl != -1 and sc != -1:
                        continue
                elif cluster_filter == "unreviewed":
                    if label:
                        continue
                elif cluster_filter == "reviewed":
                    if not label:
                        continue
                elif cluster_filter.isdigit():
                    # 旧式簇号筛选：匹配该簇全部单元（子簇 + 残噪）
                    if not gk.startswith(cluster_filter + "."):
                        continue
                elif gk != cluster_filter:
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
                "subcluster": sc,
                "gkey": gk,
                "source_group": str(r.get("individual_id", "")),
                "session_id": str(r.get("session_id", "")),
                "quality_band": str(r.get("quality_band", "")),
                "label": label,
                "similar": similar,
                # 跨时间管线产物（pipeline_archival）带簇级匹配建议；旧格式无这些列 → 空
                "top1": "" if str(r.get("top1", "")) in ("", "nan") else str(r.get("top1", "")),
                "top1_score": _fmt(r.get("top1_score", "")),
                "vote1_ratio": _fmt(r.get("vote1_ratio", "")),
                "status": str(r.get("status", "")),
                "in_sim": in_sim.get(iid, ""),
            })
        return items

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (Path(__file__).parent / "review_app.html").read_text(encoding="utf-8")
        return html

    @app.get("/api/state")
    def state(filter: str = "all"):
        """照片列表（?filter=all|noise|unreviewed|reviewed|单元键如 1.0 / 旧式簇号 1）。"""
        from collections import Counter
        cnt = Counter(gkey_of(_int_col(r["cluster"]), _int_col(r["subcluster"]))
                      for _, r in photos.iterrows())
        labels = {k: v for k, v in ann.items() if v}
        # 名字 → 使用位置（与筛选无关；供前端跨单元同名警告，filter 视图下依然有效）
        gk_by_id = {str(r["image_id"]): gkey_of(_int_col(r["cluster"]), _int_col(r["subcluster"]))
                    for _, r in photos.iterrows()}
        name_locations: dict[str, dict] = {}
        for iid, lab in labels.items():
            if lab in (UNCERTAIN, REJECT) or iid not in gk_by_id:
                continue
            if lab in name_locations:
                name_locations[lab]["count"] += 1
            else:
                name_locations[lab] = {"key": gk_by_id[iid], "count": 1}
        return {
            "n_total": len(photos),
            "n_reviewed": len(labels),
            "clusters": {str(k): v for k, v in sorted(cnt.items())},
            "photos": photo_list(filter),
            "name_locations": name_locations,
            "has_history": history_root is not None,
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
        save_annotations(args.annotations, photos, ann, reviewer=args.reviewer)
        return {"image_id": image_id, "label": ann.get(image_id, ""),
                "n_reviewed": len([v for v in ann.values() if v])}

    @app.post("/api/annotate_batch")
    async def annotate_batch(request: Request):
        """整簇批量标注：{image_ids, action, identity} —— 簇级审核一次操作全簇成员。"""
        body = await request.json()
        ids = [str(x) for x in body.get("image_ids", [])]
        known = set(photos["image_id"])
        unknown = [i for i in ids if i not in known]
        if unknown:
            return JSONResponse({"error": f"未知 image_id: {unknown[:3]}"}, status_code=400)
        action = body.get("action", "")
        if action == "confirm":
            ident = str(body.get("identity", "")).strip()
            if not ident:
                return JSONResponse({"error": "确认需填写个体名"}, status_code=400)
            for iid in ids:
                ann[iid] = ident
        elif action == "uncertain":
            for iid in ids:
                ann[iid] = UNCERTAIN
        elif action == "reject":
            for iid in ids:
                ann[iid] = REJECT
        elif action == "clear":
            for iid in ids:
                ann.pop(iid, None)
        else:
            return JSONResponse({"error": f"未知操作: {action}"}, status_code=400)
        save_annotations(args.annotations, photos, ann, reviewer=args.reviewer)
        return {"n_reviewed": len([v for v in ann.values() if v]),
                "labels": {i: ann.get(i, "") for i in ids}}

    @app.get("/api/history/{individual_id}")
    def history(individual_id: str):
        """历史个体对照照片清单（history_lookup/<个体>/ 目录内图片；未配置目录 → 空）。"""
        if not history_root:
            return {"photos": []}
        d = history_root / individual_id
        if not d.is_dir():
            return {"photos": []}
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        photos_list = []
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                from urllib.parse import quote
                photos_list.append({
                    "name": f.name,
                    "url": f"/api/history_photo/{quote(individual_id)}/{quote(f.name)}",
                    "quality": hist_quality.get(f.name, ""),
                })
        return {"photos": photos_list}

    @app.get("/api/history_photo/{individual_id}/{name}")
    def history_photo(individual_id: str, name: str):
        """历史个体对照照片原图。"""
        if not history_root:
            return Response(status_code=404)
        p = history_root / individual_id / name
        if not p.exists():
            return Response(status_code=404)
        ext = p.suffix.lower().lstrip(".")
        media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "bmp": "image/bmp"}.get(ext, "image/jpeg")
        return Response(p.read_bytes(), media_type=media)

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
            "subcluster": r.get("subcluster", ""),
            "source_group": r.get("individual_id", ""),
            "session_id": r.get("session_id", ""),
            "source_path": r.get("relative_path", ""),
            "reviewer": args.reviewer,
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
