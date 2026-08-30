"""
保守口径未命中 query 诊断（E5.2 补充）。

对保守口径（完整同串约束后打分）cluster_r1=0 的个体，逐 query 图输出：
- 同体跨串最高 cos、同体库照片数（剔除同串后）；
- top1-3 候选个体与 cos、实际产生该分数的库图路径；
- 每张 query 一张证据拼图：query 图（红框）+ 同体最高分库图（绿框）+
  top1-3 实际最高分库图（蓝框），
  供人工按图分类失败原因（同体外观变化 / 异体顶替 / 检测或序列问题）。

复用 eval51_common 的拆分与评分逻辑，保证与评估口径一致。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval51_common import (  # noqa: E402
    exclude_same_series, load_data, score_img_to_individual, split_probe_gallery)
from whitewhale.data.image_store import get_image_store  # noqa: E402

# 拼图配色：红=query，绿=同体库，蓝=top 候选
COLORS = {"Q": (200, 40, 40), "S": (40, 160, 70), "T": (40, 90, 200)}
DETAIL_COLUMNS = [
    "individual", "cluster_true_rank", "cluster_topk_hit", "cluster_n_candidates",
    "q_no", "q_image_id", "q_path", "q_det_conf", "q_fallback",
    "q_series_id", "same_cos_max", "same_gallery_n", "evidence_tile",
    "same_image_id", "same_path",
    "top1_ind", "top1_cos", "top1_image_id", "top1_path",
    "top2_ind", "top2_cos", "top2_image_id", "top2_path",
    "top3_ind", "top3_cos", "top3_image_id", "top3_path",
]


def img_path(meta: pd.DataFrame, idx: int) -> Path:
    """原图路径：relative_path（含 session 前缀）+ 数据根。"""
    return get_image_store().resolve(str(meta.loc[idx, "relative_path"]))


def _as_bool(value: object) -> bool:
    """稳定解析 CSV 布尔字段，避免字符串 ``"False"`` 被当作真。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def img_crop(meta: pd.DataFrame, idx: int, pad: float = 0.15,
             h: int = 180) -> Image.Image:
    """背鳍特写：YOLO 框区域外扩 pad 后裁剪缩放；回退图整图缩略。"""
    im = Image.open(img_path(meta, idx)).convert("RGB")
    x, y, w, hh = (float(meta.loc[idx, k]) for k in ("x", "y", "w", "h"))
    if _as_bool(meta.loc[idx, "fallback"]) or w <= 0:
        return thumb(im, h)
    x0 = max(0, int(x - w * pad))
    y0 = max(0, int(y - hh * pad))
    x1 = min(im.width, int(x + w * (1 + pad)))
    y1 = min(im.height, int(y + hh * (1 + pad)))
    return thumb(im.crop((x0, y0, x1, y1)), h)


def thumb(im: Image.Image, h: int = 150) -> Image.Image:
    w = max(1, int(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def labeled_thumb(img: Image.Image, label: str, h: int = 180) -> Image.Image:
    t = thumb(img, h)
    pad = 18
    out = Image.new("RGB", (t.width, h + pad), (245, 245, 245))
    out.paste(t, (0, 0))
    d = ImageDraw.Draw(out)
    color = COLORS.get(label[:1], (80, 80, 80))
    d.rectangle([0, h, t.width, h + pad], fill=color)
    d.text((4, h + 3), label, fill=(255, 255, 255))
    return out


def tile(labels_paths, out_path: Path, gap: int = 6) -> None:
    """三行网格拼图：QUERY 行 / SAME 行 / TOP 行，行首标签列。"""
    rows = {"Q": [], "S": [], "T": []}
    for lbl, p in labels_paths:
        rows[lbl[:1]].append((lbl, p))
    row_ims = []
    for key in ("Q", "S", "T"):
        if not rows[key]:
            continue
        ims = [labeled_thumb(img, lbl) for lbl, img in rows[key]]
        H = max(i.height for i in ims)
        W = sum(i.width for i in ims) + gap * (len(ims) - 1)
        canvas = Image.new("RGB", (W, H), (245, 245, 245))
        x = 0
        for i in ims:
            canvas.paste(i, (x, 0))
            x += i.width + gap
        row_ims.append(canvas)
    W = max(i.width for i in row_ims)
    H = sum(i.height for i in row_ims) + gap * (len(row_ims) - 1)
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    y = 0
    for r in row_ims:
        canvas.paste(r, (0, y))
        y += r.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"  [tile] {out_path} ({len(labels_paths)} 图)")


def score_with_representatives(img_emb: np.ndarray, emb: np.ndarray,
                               gal_idx: np.ndarray,
                               gal_ind: np.ndarray) -> tuple[dict, dict]:
    """按个体取最高 cosine，并返回实际产生最高分的 gallery 行号。"""
    sims = img_emb @ emb[gal_idx].T
    scores, representatives = {}, {}
    for individual in np.unique(gal_ind):
        if individual == "":
            continue
        positions = np.flatnonzero(gal_ind == individual)
        best_pos = int(positions[int(np.argmax(sims[positions]))])
        key = str(individual)
        scores[key] = float(sims[best_pos])
        representatives[key] = int(gal_idx[best_pos])
    return scores, representatives


def image_fields(meta: pd.DataFrame, idx: int | None,
                 prefix: str) -> dict:
    """把证据图片标识与路径写入 CSV，便于从数值追溯到具体照片。"""
    if idx is None:
        return {f"{prefix}_image_id": "", f"{prefix}_path": ""}
    return {
        f"{prefix}_image_id": str(meta.loc[idx, "image_id"]),
        f"{prefix}_path": str(meta.loc[idx, "relative_path"]),
    }


def main():
    base = REPO_ROOT
    ap = argparse.ArgumentParser(description="保守口径未命中 query 诊断")
    ap.add_argument("--feats-stem", type=str, default="embeddings_eval51_all_r4")
    ap.add_argument("--out", type=Path,
                    default=base / "outputs" / "reports" / "missed_diag")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"FATAL: 诊断输出目录已存在，拒绝覆盖旧证据: {args.out}")

    meta, emb = load_data(base, stem=args.feats_stem)
    q_rows, gal_idx, gal_ind = split_probe_gallery(meta, emb)
    args.out.mkdir(parents=True, exist_ok=False)

    # 全量保守口径打分 → 簇级 hit_rank
    skipped = 0
    per_q = []
    for target, q_idx in q_rows.items():
        per_img = []
        for qi in q_idx:
            g_idx, g_ind = exclude_same_series(meta, qi, gal_idx, gal_ind)
            scores = score_img_to_individual(emb[qi], emb, g_idx, g_ind)
            if target not in scores:
                skipped += 1
                continue
            per_img.append(scores)
        if not per_img:
            continue
        cands = sorted({g for s in per_img for g in s})
        cluster = {g: float(np.mean([s[g] for s in per_img if g in s]))
                   for g in cands}
        top = sorted(cluster, key=cluster.get, reverse=True)
        hit_rank = next((r for r, g in enumerate(top[: args.k], 1)
                         if g == target), 0)
        if hit_rank == 1:
            continue  # 只诊断未命中 top1 的个体（含 rank 2-5）
        true_rank = next((r for r, g in enumerate(top, 1) if g == target), 0)
        print(f"[miss] {target}  rank={true_rank}  n_query={len(q_idx)}")

        # 逐 query 证据：所有分数都绑定实际产生 max cosine 的 gallery 图
        q_detail = []
        for q_no, qi in enumerate(q_idx, 1):
            g_idx, g_ind = exclude_same_series(meta, qi, gal_idx, gal_ind)
            scores, representatives = score_with_representatives(
                emb[qi], emb, g_idx, g_ind)
            same_cos = scores.get(target, float("nan"))
            same_n = int(np.sum(g_ind == target))
            t3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            same_idx = representatives.get(target)
            identity_digest = hashlib.sha256(
                str(target).encode("utf-8")).hexdigest()[:12]
            evidence_tile = f"miss_{identity_digest}__q{q_no:02d}.png"
            record = {
                "individual": target,
                "cluster_true_rank": int(true_rank),
                "cluster_topk_hit": bool(0 < true_rank <= args.k),
                "cluster_n_candidates": int(len(top)),
                "q_no": q_no,
                "q_image_id": str(meta.loc[qi, "image_id"]),
                "q_path": str(meta.loc[qi, "relative_path"]),
                "q_det_conf": float(meta.loc[qi, "det_conf"]),
                "q_fallback": _as_bool(meta.loc[qi, "fallback"]),
                "q_series_id": str(meta.loc[qi, "series_id"]),
                "same_cos_max": same_cos,
                "same_gallery_n": same_n,
                "evidence_tile": evidence_tile,
                **image_fields(meta, same_idx, "same"),
            }
            fig_rows = [("Q", img_crop(meta, qi))]
            if same_idx is not None:
                fig_rows.append(("SAME", img_crop(meta, same_idx)))
            for rank in range(1, 4):
                if rank <= len(t3):
                    cand, score = t3[rank - 1]
                    rep_idx = representatives[cand]
                    record[f"top{rank}_ind"] = cand
                    record[f"top{rank}_cos"] = score
                    record.update(image_fields(meta, rep_idx, f"top{rank}"))
                    fig_rows.append((f"T{rank}", img_crop(meta, rep_idx)))
                else:
                    record[f"top{rank}_ind"] = ""
                    record[f"top{rank}_cos"] = float("nan")
                    record.update(image_fields(meta, None, f"top{rank}"))
            q_detail.append(record)
            tile(fig_rows, args.out / evidence_tile)

        per_q.extend(q_detail)

    df = pd.DataFrame(per_q, columns=DETAIL_COLUMNS)
    df.to_csv(args.out / "missed_query_detail.csv", index=False,
              encoding="utf-8-sig")
    identity_summary = df[[
        "individual", "cluster_true_rank", "cluster_topk_hit",
        "cluster_n_candidates",
    ]].drop_duplicates("individual")
    summary = {
        "n_missed_identities": int(identity_summary["individual"].nunique()),
        "n_missed_query_images": int(len(df)),
        "n_identities_outside_topk": int(
            (~identity_summary["cluster_topk_hit"].astype(bool)).sum()),
        "n_query_images_outside_topk": int(
            (~df["cluster_topk_hit"].astype(bool)).sum()),
        "n_skipped_queries_no_cross_series_positive": int(skipped),
        "topk": int(args.k),
        "protocol": "within_session_cross_series_conservative",
        "features_stem": args.feats_stem,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 未命中 query 详情 {len(df)} 行 → "
          f"{args.out / 'missed_query_detail.csv'}（skipped {skipped}）")


if __name__ == "__main__":
    main()
