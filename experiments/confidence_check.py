"""
中华白海豚个体识别：微调前后置信度体检。

对比预训练特征与确认个体标签微调特征：
1. 对级相似度分布（同批次同个体 / 同批次不同个体 / 跨批次未对齐代理）；
   相似度分数作为"置信度"越可信；
2. 阈值判定表只使用批次内已确认关系，且只表示图对诊断，不是开放集 FA 标定；
3. leave-one-out Precision@1/@5 限定同批次并剔除同串候选。

用法：
    python experiments/confidence_check.py
输出：
    outputs/reports/confidence_check/summary.txt + pair_stats.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
base = REPO_ROOT / "outputs"

from experiments.artifact_utils import load_aligned_embeddings  # noqa: E402
from whitewhale.data.eval_set import attach_stable_series_from_manifest  # noqa: E402
from whitewhale.reid.embedding import read_metadata_csv  # noqa: E402


def load():
    """加载两套特征 + meta + pilot（行序与 embeddings 对齐）。"""
    pilot = base / "pilot" / "pilot_set.csv"
    p, pre = load_aligned_embeddings(base / "embeddings" / "embeddings.npy", pilot)
    p_finetuned, fin = load_aligned_embeddings(
        base / "embeddings" / "embeddings_metric.npy", pilot)
    if p["image_id"].tolist() != p_finetuned["image_id"].tolist():
        raise ValueError("预训练与微调特征的 image_id 行序不一致，不能直接比较")
    for emb in (pre, fin):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("特征中存在零向量，无法计算余弦相似度")
        emb /= norms

    if "individual_id" not in p.columns:
        raise ValueError("pilot 清单缺少已确认的 individual_id")
    reviewed = (p["confirmed_identity"].fillna("").astype(str).str.strip()
                if "confirmed_identity" in p.columns
                else pd.Series("", index=p.index))
    # 来源 individual_id 已由数据提供方确认为批次内个体；人工复核字段非空时优先使用。
    p["confirmed_identity"] = reviewed.where(
        reviewed != "", p["individual_id"].astype(str))
    full_manifest = read_metadata_csv(base / "index" / "dataset_manifest.csv")
    p = attach_stable_series_from_manifest(p, full_manifest)
    return p, pre, fin


def session_key(session_id: str) -> str:
    """返回完整调查批次键；不同日期的同尾号批次不能混为同一组。"""
    return str(session_id).strip()


def pair_stats(emb: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """confirmed 照片两两配对，标注对类型并算相似度。"""
    idx = df.index.to_numpy()
    sim = emb[idx][:, :] @ emb[idx].T
    rows = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = df.iloc[i], df.iloc[j]
            same_id = a["confirmed_identity"] == b["confirmed_identity"]
            cross = (session_key(a["session_id"]) != session_key(b["session_id"])
                     and bool(str(a["session_id"]).strip())
                     and bool(str(b["session_id"]).strip()))
            same_series = (
                bool(str(a.get("series_id", "")).strip())
                and str(a.get("series_id", "")) == str(b.get("series_id", ""))
            )
            if cross:
                kind = "cross_batch_unverified"
            elif same_series:
                kind = "same_series_excluded"
            elif same_id:
                kind = "same_individual"
            else:
                kind = "same_session_different_identity"
            rows.append({"i": a["image_id"], "j": b["image_id"],
                         "kind": kind, "sim": float(sim[i, j])})
    return pd.DataFrame(rows, columns=["i", "j", "kind", "sim"])


def threshold_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """批次内跨串图对精确率诊断；不等价于查询对全库最大分数的 FA。"""
    valid = pairs[pairs["kind"].isin([
        "same_individual", "same_session_different_identity"])]
    out = []
    for T in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        pos = valid[valid["sim"] >= T]
        tp = (pos["kind"] == "same_individual").sum()
        out.append({"threshold": T, "n_above": len(pos), "n_same": int(tp),
                    "precision": tp / len(pos) if len(pos) else float("nan")})
    return pd.DataFrame(out)


def precision_at_k(emb: np.ndarray, df: pd.DataFrame, k: int) -> float:
    """批次内跨串 leave-one-out；无跨串同体正样本的 query 不计入分母。"""
    if len(df) < 2:
        return float("nan")
    idx = df.index.to_numpy()
    labs = df["confirmed_identity"].to_numpy()
    sim = emb[idx][:, :] @ emb[idx].T
    np.fill_diagonal(sim, -np.inf)
    hits = []
    sessions = df["session_id"].astype(str).to_numpy()
    series = (df["series_id"].fillna("").astype(str).to_numpy()
              if "series_id" in df.columns
              else np.full(len(df), "", dtype=object))
    for i in range(len(sim)):
        valid = sessions == sessions[i]
        valid[i] = False
        if series[i].strip():
            valid &= series != series[i]
        positive = valid & (labs == labs[i])
        if not positive.any():
            continue
        row = sim[i].copy()
        row[~valid] = -np.inf
        n_candidates = int(valid.sum())
        top = np.argsort(-row)[:min(k, n_candidates)]
        hits.append(bool((labs[top] == labs[i]).any()))
    return float(np.mean(hits)) if hits else float("nan")


def main():
    p, pre, fin = load()
    keep = (p["confirmed_identity"].notna()
            & (p["confirmed_identity"].astype(str).str.strip() != "")).to_numpy()
    conf = p.loc[keep].reset_index(drop=True)
    pre = pre[keep]
    fin = fin[keep]
    print(f"[conf] confirmed {len(conf)} 张 / {conf['confirmed_identity'].nunique()} 个体")

    out_dir = base / "reports" / "confidence_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, emb in (("pretrained", pre), ("finetuned", fin)):
        pairs = pair_stats(emb, conf)
        pairs.to_csv(out_dir / f"pairs_{name}.csv", index=False, encoding="utf-8-sig")
        tt = threshold_table(pairs)
        tt.to_csv(out_dir / f"threshold_{name}.csv", index=False, encoding="utf-8-sig")
        p1, p5 = precision_at_k(emb, conf, 1), precision_at_k(emb, conf, 5)

        lines.append(f"=== {name}（{'预训练' if name=='pretrained' else '确认标签微调'}） ===")
        for kind, cn in (
                ("same_individual", "批内跨串同个体"),
                ("same_session_different_identity", "批内不同个体"),
                ("cross_batch_unverified", "跨批次未对齐代理")):
            g = pairs[pairs["kind"] == kind]["sim"]
            lines.append(f"  {cn:>10s}: n={len(g):5d}  "
                         f"p50={g.median():.3f}  p90={g.quantile(.9):.3f}  p95={g.quantile(.95):.3f}  "
                         f">0.70 占 {100*(g>0.70).mean():.1f}%")
        lines.append(f"  leave-one-out Precision@1 = {p1:.3f}   Precision@5 = {p5:.3f}")
        sep = "\n".join(f"    ≥{r.threshold:.2f}: 判同体 {r.n_above:5d} 对，其中真同体 {int(r.n_same):4d} 对 → 精确率 {r.precision:.1%}"
                        for r in tt.itertuples())
        lines.append("  批内跨串图对诊断（非开放集 FA 标定）:" + "\n" + sep)
        lines.append("  跨批次身份未对齐，相关分布不作为负例或阈值推荐。")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
