"""
r4 训练集准备：历史库(20140806 01/03) individual_id → confirmed_identity。

动机（2026-08-25 用户决定）：直接用 individual_id（当年分组文件夹编号）
作为训练标签重训度量学习模型；训练集 = 历史库 43 个体 202 张（模型已见过
该批个体，训练量最大化），评估集 = 新批次 32 个体 150 张（模型未见，
无泄漏评估）。

关键兼容点：
- train_reid.py 的 load_confirmed 按 confirmed_identity 非空过滤 → 本文件
  把 individual_id 填进 confirmed_identity；
- r3 链路的 BalancedGroupSampler 硬编码群编号 1/3 → session_id 重映射
  （20140806 01 → 1，20140806 03 → 3）。
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

HIST_SESSIONS = {"20140806 01": 1, "20140806 03": 3}
OUT = REPO_ROOT / "outputs" / "pilot" / "pilot_set_train_hist.csv"


def main():
    p = pd.read_csv(REPO_ROOT / "outputs" / "pilot" / "pilot_set.csv",
                    dtype={"session_id": str})
    df = p[p["session_id"].isin(HIST_SESSIONS)].copy()
    assert df["individual_id"].notna().all(), "历史库有行缺 individual_id"
    df["confirmed_identity"] = df["individual_id"]
    df["session_id"] = df["session_id"].map(HIST_SESSIONS)  # 1/3 整数群号
    n_ind = df["confirmed_identity"].nunique()
    print(f"[prep] 训练集 {len(df)} 张 / {n_ind} 个体 "
          f"(01 群 {int((df['session_id'] == 1).sum())} 张，"
          f"03 群 {int((df['session_id'] == 3).sum())} 张)")
    assert df["session_id"].nunique() == 2, "两群缺一，无法跑 BalancedGroupSampler"
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[prep] -> {OUT}")


if __name__ == "__main__":
    main()
