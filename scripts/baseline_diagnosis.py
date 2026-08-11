"""
基线诊断报告。
综合 Top-K 检索与聚类结果，判断 embedding 是否编码了"个体身份"，
还是退化为"背景 / 拍摄批次 / 摄影师"等非目标信号。

输出 markdown 报告（outputs/diagnosis/baseline_report.md），只依赖
topk_results.csv / clusters.csv / pilot_set.csv，不需要图片，mock 数据也可运行。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_frac(d: dict) -> str:
    """把 {k: v} 转成 'k: 12.3%' 排序文本。"""
    total = sum(d.values())
    if total == 0:
        return "无数据"
    items = sorted(d.items(), key=lambda x: -x[1])
    return "，".join(f"{k}: {v / total:.1%}" for k, v in items)


def _load_topk(path: Path) -> pd.DataFrame:
    """读 topk_results.csv，并把 JSON 列表列还原成 list。"""
    df = pd.read_csv(path)
    for c in ("retrieved", "retrieved_individuals", "retrieved_scores"):
        if c in df.columns:
            df[c] = df[c].apply(lambda s: json.loads(s) if pd.notna(s) else [])
    return df


def diagnose(topk_path: Path, cluster_path: Path, pilot_path: Path, out_dir: Path):
    topk = _load_topk(topk_path)
    clusters = pd.read_csv(cluster_path)
    pilot = pd.read_csv(pilot_path)
    topk["session_id"] = topk["session_id"].astype(str)
    clusters["session_id"] = clusters["session_id"].astype(str)

    lines = []
    add = lines.append
    add("# 基线诊断报告（Pilot Set）")
    add("")

    # 1. 数据规模
    add("## 1. 数据规模")
    add(f"- 检索：{len(topk)} 张（labeled {len(topk[topk['split'] == 'labeled'])} / "
        f"loose_known {len(topk[topk['split'] == 'loose_known'])}）")
    add(f"- 聚类：{len(clusters)} 张，簇 {clusters['cluster'].nunique() - (1 if -1 in clusters['cluster'].values else 0)} 个，"
        f"噪声 {int((clusters['cluster'] == -1).sum())} 张")
    add("")

    # 2. Top-K 命中（同调查内）
    add("## 2. Top-K 检索（同调查内，cosine）")
    labeled = topk[topk["split"] == "labeled"]
    rank1 = labeled["rank1_hit"].mean() if "rank1_hit" in labeled else None
    rank5 = labeled["rank5_hit"].mean() if "rank5_hit" in labeled else None
    if rank1 is not None:
        add(f"- rank-1 命中：**{rank1:.1%}**（{int(labeled['rank1_hit'].sum())}/{len(labeled)}）")
        add(f"- rank-5 命中：**{rank5:.1%}**（{int(labeled['rank5_hit'].sum())}/{len(labeled)}）")
    add("")

    # 3. 检索是否落在同一连拍序列（怀疑：背景相似导致的假命中）
    # sequence_guess 即"连拍组"：session::帧号_日期_...，同一 key 的图片属于同一连拍簇，
    # 背景/光线几乎相同，是假命中的最大来源。
    add("## 3. 假命中诊断：Top-1 是否来自同一连拍序列")
    if "sequence_guess" in topk.columns:
        seq_lookup = {r["image_id"]: r["sequence_guess"] for _, r in topk.iterrows()}
        labeled = topk[topk["split"] == "labeled"]
        n_false = 0
        n_same_seq = 0
        for _, r in labeled.iterrows():
            if not r["retrieved_individuals"]:
                continue
            top1_id = r["retrieved_individuals"][0]
            if top1_id == r["individual_id"]:
                continue  # 真命中不算假命中
            n_false += 1
            q_seq = r.get("sequence_guess")
            r_seq = seq_lookup.get(r["retrieved"][0])
            if isinstance(q_seq, str) and q_seq == r_seq:
                n_same_seq += 1
        if n_false:
            add(f"- 错命中的 Top-1 共 **{n_false}** 例，其中来自同一连拍序列（sequence_guess 相同）："
                f"**{n_same_seq}** 例（{n_same_seq / n_false:.0%}）")
            add("  - 若该比例偏高，说明相似性可能来自同一连拍（背景/光线），而非个体身份")
        else:
            add("- 无错命中样本")
    else:
        add("- 缺少 sequence_guess 字段，跳过")
    add("")

    # 4. 聚类一致性：簇内个体纯度
    add("## 4. 聚类与个体对照")
    cl = clusters.copy()
    cl["ind"] = cl["individual_id"]
    purity_rows = []
    for cid, sub in cl[cl["cluster"] >= 0].groupby("cluster"):
        inds = sub[sub["ind"] != "loose_unknown"]["ind"]
        if len(inds) >= 2:
            majority = inds.value_counts().iloc[0]
            purity_rows.append(majority / len(sub))
    if purity_rows:
        add(f"- 多图簇中，主个体占比中位数：**{np.median(purity_rows):.1%}**"
            f"（簇数 {len(purity_rows)}）")
        # 簇内 session 多样性：同一簇是否横跨多个 session（跨 session 不应合并）
        mixed = cl[cl["cluster"] >= 0].groupby("cluster")["session_id"].nunique()
        n_mixed = int((mixed > 1).sum())
        add(f"- 跨 session 的簇：**{n_mixed}** 个（若偏高说明聚类混入了拍摄批次信号）")
    add("")

    # 5. 检索命中 vs 同 session 连拍：rank-1 是否总落在同批
    add("## 5. 风险小结")
    risk = []
    if rank1 is not None and rank1 < 0.3:
        risk.append("rank-1 命中率过低，说明当前特征对个体区分不足（one-shot 场景常见），需迁移学习。")
    elif rank1 is not None and rank1 >= 0.6:
        risk.append("rank-1 命中率较高，需确认是否来自连拍相邻帧的『背景相似』而非个体身份。")
    if purity_rows and np.median(purity_rows) < 0.5:
        risk.append("簇内个体纯度低，聚类结果不能直接作为伪标签，需人工核验。")
    if n_mixed > len(set(cl[cl["cluster"] >= 0]["cluster"])) * 0.3:
        risk.append("大量簇跨 session，聚类可能被拍摄批次主导。")
    if not risk:
        risk.append("当前未发现明显风险信号，但样本量小，结论需谨慎。")
    for r in risk:
        add(f"- {r}")
    add("")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "baseline_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"诊断报告 → {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基线诊断报告")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--topk", type=Path, default=base / "retrieval" / "topk_results.csv")
    parser.add_argument("--clusters", type=Path, default=base / "clusters" / "clusters.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--out", type=Path, default=base / "diagnosis")
    args = parser.parse_args()
    diagnose(args.topk, args.clusters, args.pilot, args.out)
