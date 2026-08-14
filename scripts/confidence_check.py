"""
中华白海豚个体识别：微调前后置信度体检。

对比预训练特征与伪标签微调特征：
1. 对级相似度分布（同个体 / 同群不同个体 / 跨群 01↔03）——分布分得越开，
   相似度分数作为"置信度"越可信；
2. 阈值判定表：相似度 ≥ T 判"同一只"时，判对的概率（对级精确率）；
3. leave-one-out Precision@1/@5（用初审标签，训练个体口径有乐观偏差，跨群对口径无偏）。

用法：
    python scripts/confidence_check.py
输出：
    outputs/reports/confidence_check/summary.txt + pair_stats.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

base = Path(__file__).resolve().parents[1] / "outputs"


def load():
    """加载两套特征 + meta + pilot（行序与 embeddings 对齐）。"""
    p = pd.read_csv(base / "pilot" / "pilot_set.csv")
    m = pd.read_csv(base / "embeddings" / "embeddings_meta.csv")
    assert list(m.columns) == ["image_id"], f"meta 列意外: {list(m.columns)}"
    p = p.merge(m, on="image_id", validate="one_to_one")
    pre = np.load(base / "embeddings" / "embeddings.npy")
    fin = np.load(base / "embeddings" / "embeddings_metric.npy")
    pre /= np.linalg.norm(pre, axis=1, keepdims=True)
    fin /= np.linalg.norm(fin, axis=1, keepdims=True)
    return p, pre, fin


def group_prefix(sg: str) -> str:
    """source_group '01_2.0' → '01'；空串 → ''。"""
    return str(sg).split("_")[0] if str(sg) else ""


def pair_stats(emb: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """confirmed 照片两两配对，标注对类型并算相似度。"""
    idx = df.index.to_numpy()
    sim = emb[idx][:, :] @ emb[idx].T
    rows = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = df.iloc[i], df.iloc[j]
            same_id = a["confirmed_identity"] == b["confirmed_identity"]
            cross = (group_prefix(a["source_group"]) != group_prefix(b["source_group"])
                     and bool(a["source_group"]) and bool(b["source_group"]))
            if same_id:
                kind = "same_individual"
            elif cross:
                kind = "cross_group"
            else:
                kind = "same_group_diff"
            rows.append({"i": a["image_id"], "j": b["image_id"],
                         "kind": kind, "sim": float(sim[i, j])})
    return pd.DataFrame(rows)


def threshold_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """阈值 T 判"同一只"的对级精确率 = 真同体对 / 判同体对。"""
    out = []
    for T in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        pos = pairs[pairs["sim"] >= T]
        tp = (pos["kind"] == "same_individual").sum()
        out.append({"threshold": T, "n_above": len(pos), "n_same": int(tp),
                    "precision": tp / len(pos) if len(pos) else float("nan")})
    return pd.DataFrame(out)


def precision_at_k(emb: np.ndarray, df: pd.DataFrame, k: int) -> float:
    """leave-one-out：每张 confirmed 照片查同集合，Top-K 中含同体照片比例。"""
    idx = df.index.to_numpy()
    labs = df["confirmed_identity"].to_numpy()
    sim = emb[idx][:, :] @ emb[idx].T
    np.fill_diagonal(sim, -1)
    hits = 0
    for i in range(len(sim)):
        top = np.argsort(-sim[i])[:k]
        if (labs[top] == labs[i]).any():
            hits += 1
    return hits / len(sim)


def main():
    p, pre, fin = load()
    conf = p[p["confirmed_identity"].notna()
             & (p["confirmed_identity"].astype(str).str.strip() != "")].copy()
    conf = conf.reset_index(drop=True)
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

        lines.append(f"=== {name}（{'预训练' if name=='pretrained' else '伪标签微调'}） ===")
        for kind, cn in (("same_individual", "同个体"), ("same_group_diff", "同群不同个体"),
                         ("cross_group", "跨群 01↔03")):
            g = pairs[pairs["kind"] == kind]["sim"]
            lines.append(f"  {cn:>10s}: n={len(g):5d}  "
                         f"p50={g.median():.3f}  p90={g.quantile(.9):.3f}  p95={g.quantile(.95):.3f}  "
                         f">0.70 占 {100*(g>0.70).mean():.1f}%")
        lines.append(f"  leave-one-out Precision@1 = {p1:.3f}   Precision@5 = {p5:.3f}")
        sep = "\n".join(f"    ≥{r.threshold:.2f}: 判同体 {r.n_above:5d} 对，其中真同体 {int(r.n_same):4d} 对 → 精确率 {r.precision:.1%}"
                        for r in tt.itertuples())
        lines.append("  阈值判定（相似度≥T 即判同一只）:" + "\n" + sep)
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
