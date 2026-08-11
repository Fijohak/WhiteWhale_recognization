"""
基线诊断报告（方向调整后 v2）。

数据语义（2026-08-11）：目录不直接等于个体 ID。本土数据无可靠 Ground Truth，
因此**不再计算 rank-1/5 命中率式"识别准确率"**（那是旧假设下的假指标）。

本脚本仅报告可供判断的弱信号：
1. 检索规模与审核表产出
2. 同组回召（Anchor 文件夹多帧是否互相在 Top-K 中）——仅反映同一次挑选/连拍的
   相似性，**不代表跨时间身份能力**，明确标注
3. 相似度分数分布（供人工审核设阈值参考）
4. 明确声明：本土无 ground truth，不报告识别准确率

输入：outputs/retrieval/topk_results.csv + topk_for_review.csv + pilot_set.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def diagnose(topk_path: Path, review_path: Path, pilot_path: Path,
             out_dir: Path) -> None:
    topk = pd.read_csv(topk_path)
    review = pd.read_csv(review_path)
    pilot = pd.read_csv(pilot_path, dtype={"session_id": str})

    lines = []
    add = lines.append
    add("# 基线诊断报告（Anchor 检索，方向调整后）")
    add("")
    add("> 数据语义：目录 ≠ individual_id。本报告不含识别准确率，只含弱信号与供审核的信息。")
    add("")

    # 1. 规模
    n_query = len(topk)
    n_review = len(review)
    n_groups = pilot["individual_id"].nunique()
    multi = int((pilot.groupby("individual_id").size() > 1).sum())
    add("## 1. 检索规模")
    add(f"- Anchor 查询：**{n_query}** 个（{n_groups} 个 Anchor 组，其中多帧组 {multi} 个）")
    add(f"- 审核表：**{n_review}** 行（{n_query} × Top-K）→ `topk_for_review.csv`")
    add("")

    # 2. 同组回召弱信号（来自 topk_stats.json 的诊断字段）
    stats_path = topk_path.parent / "topk_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    diag = stats.get("weak_diagnostic", {})
    n_g2 = diag.get("anchor_groups_with_2plus", 0)
    n_hit = diag.get("groups_sibling_in_topk", 0)
    ratio = diag.get("ratio")
    add("## 2. 同组回召（弱信号，不代表身份）")
    add(f"- 含 ≥2 帧的 Anchor 组：**{n_g2}** 个，组内其他照片出现在 Top-K 的组：**{n_hit}**"
        f"（{ratio:.1%}" if ratio is not None else f"- 含 ≥2 帧的 Anchor 组：**{n_g2}** 个")
    add("  - 说明：仅反映同一次挑选/连拍帧的相似性，**不能**证明跨时间识别能力；")
    add("    只能说明 embedding 是否编码了可区分的信息。")
    add("")

    # 3. 分数分布（供人工审核设阈值）
    if "score" in review and len(review):
        add("## 3. 相似度分数分布（审核阈值参考）")
        for rk, sub in review.groupby("rank"):
            add(f"- Top-{int(rk)} 平均 {sub['score'].mean():.3f}（min {sub['score'].min():.3f} / "
                f"max {sub['score'].max():.3f}）")
        add("  - 建议：分数高于同组平均水平的候选优先人工审核；低分候选降权。")
        add("")

    # 4. 明确声明
    add("## 4. 声明")
    add("- 本土数据无可靠 Ground Truth，**不报告任何识别准确率 / Recall@K**；")
    add("- 检索结果仅为候选，正式身份须专业人员人工审核；")
    add("- 定量指标（Recall@K / mAP）只在公开可靠鲸豚数据上计算（主路线 A）。")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "baseline_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"诊断报告 → {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基线诊断报告（Anchor 检索弱信号）")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--topk", type=Path, default=base / "retrieval" / "topk_results.csv")
    parser.add_argument("--review", type=Path, default=base / "retrieval" / "topk_for_review.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--out", type=Path, default=base / "diagnosis")
    args = parser.parse_args()
    diagnose(args.topk, args.review, args.pilot, args.out)
