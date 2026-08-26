"""
E5.2 图表与训练日志生成：r4 重训曲线 + r3/r4 评估对照 + 训练报告。

产物：
- outputs/metric_learning/r4/training_curves.png   训练曲线（r4 实线 vs r3 虚线）
- outputs/metric_learning/r4/train_report.md       训练报告（配置/过程/结果/复现）
- outputs/reports/cluster_retrieval_v2/eval51_r3_vs_r4.png  三口径评估对照
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
ML = REPO_ROOT / "outputs" / "metric_learning"
RPT = REPO_ROOT / "outputs" / "reports" / "cluster_retrieval_v2"

# 学术编辑风配色
C_DARK, C_GOLD, C_RED = "#1E3A5F", "#E8B339", "#BF0000"
C_R3, C_R4 = "#9FB4C7", C_DARK


def load_history(r_dir: Path) -> pd.DataFrame:
    """读 history.csv，拼出全局 epoch（stage1 1-20、stage2 21-45）。"""
    h = pd.read_csv(r_dir / "history.csv")
    h["epoch_global"] = h["epoch"] + (h["stage"] - 1) * 20
    return h


def load_metrics(r_dir: Path) -> dict:
    with open(r_dir / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def fig_training_curves():
    """训练曲线：CE loss + val R@1，r4 实线 / r3 虚线对照（验证集不同，虚线仅参考）。"""
    r3, r4 = load_history(ML / "r3"), load_history(ML / "r4")
    m3, m4 = load_metrics(ML / "r3"), load_metrics(ML / "r4")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    # 左：CE loss
    ax = axes[0]
    ax.plot(r4["epoch_global"], r4["loss"], color=C_DARK, lw=2,
            label=f"r4 CE loss（训练 {m4['n_train']} 张/{m4['n_train_individuals']} 个体）")
    ax.plot(r3["epoch_global"], r3["loss"], color=C_R3, lw=1.5, ls="--",
            label=f"r3 CE loss（训练 {m3['n_train']} 张/{m3['n_train_individuals']} 个体）")
    ax.axvline(20, color=C_RED, ls=":", lw=1, alpha=0.7)
    ax.text(20, ax.get_ylim()[1] * 0.9, "s1→s2", color=C_RED, fontsize=9, ha="right")
    ax.set_xlabel("epoch（全局）"); ax.set_ylabel("CE loss")
    ax.set_title("CE loss（两阶段：s1 训 head → s2 解冻微调）", fontsize=11, color=C_DARK)
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3)
    # 右：val R@1
    ax = axes[1]
    ax.plot(r4["epoch_global"], r4["val_r1"], color=C_DARK, lw=2,
            label=f"r4 val R@1（验证 {m4['n_val']} 张/{m4['n_val_individuals']} 个体）")
    ax.plot(r3["epoch_global"], r3["val_r1"], color=C_R3, lw=1.5, ls="--",
            label=f"r3 val R@1（验证 {m3['n_val']} 张/{m3['n_val_individuals']} 个体）")
    best = r4.loc[r4["val_r1"].idxmax()]
    ax.scatter([best["epoch_global"]], [best["val_r1"]], color=C_RED, zorder=5, s=40)
    ax.annotate(f"best {best['val_r1']:.3f} @ ep{int(best['epoch_global'])}",
                (best["epoch_global"], best["val_r1"]),
                textcoords="offset points", xytext=(6, -14), color=C_RED, fontsize=9)
    ax.axhline(m4["pretrained_baseline_r1"], color="#888", ls=":", lw=1)
    ax.text(0.02, m4["pretrained_baseline_r1"] + 0.008,
            f"r4 基线 {m4['pretrained_baseline_r1']:.3f}", fontsize=8, color="#666")
    ax.set_xlabel("epoch（全局）"); ax.set_ylabel("val R@1")
    ax.set_title("验证个体 leave-one-out R@1（虚线 r3 仅参考，验证集不同）",
                 fontsize=11, color=C_DARK)
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3)
    fig.suptitle("r4 度量学习重训曲线（individual_id 标签，历史库 37 个体）",
                 fontsize=13, fontweight="bold", color=C_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ML / "r4" / "training_curves.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {out}")


def _load_eval(prefix: str) -> dict:
    """读评估 json：r3 无后缀，r4 后缀 __r4。返回 {group: {r3, r4}} 簇级 R@1。"""
    rows = {}
    for tag, label in (("", "r3"), ("__r4", "r4")):
        path = RPT / f"{prefix}{tag}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for group, key in (("整体", "overall"), ("历史库", "history_20140806"),
                           ("新批次", "new_batches")):
            rows.setdefault(group, {})[label] = d[key].get("cluster_R@1", float("nan"))
    return rows


def fig_eval_compare():
    """r3 vs r4 三口径簇级 R@1 对照（常规/保守/串抽样 × 整体/历史库/新批次）。"""
    prefixes = {"常规": "metrics", "保守(剔同串)": "metrics_conservative",
                "串抽样": "metrics_series_split"}
    groups = ["整体", "历史库", "新批次"]
    n_g = len(groups)
    x = [i * 3.0 for i in range(len(prefixes))]
    width = 0.9
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for gi, g in enumerate(groups):
        off = (gi - (n_g - 1) / 2) * width
        v3, v4 = [], []
        for pre in prefixes.values():
            r = _load_eval(pre).get(g, {})
            v3.append(r.get("r3", float("nan")))
            v4.append(r.get("r4", float("nan")))
        b3 = ax.bar([xi + off for xi in x], v3, width, color=C_R3,
                    label="r3" if gi == 0 else None)
        b4 = ax.bar([xi + off + width for xi in x], v4, width, color=C_R4,
                    label="r4" if gi == 0 else None)
        for b in list(b3) + list(b4):
            h = b.get_height()
            if h == h:  # 非 NaN
                ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.3f}",
                        ha="center", fontsize=8, color="#444")
    ax.set_xticks([xi + 0.9 for xi in x])
    ax.set_xticklabels(list(prefixes.keys()), fontsize=11)
    ax.set_ylabel("簇级 R@1")
    ax.set_ylim(0, 0.78)
    ax.set_title("E5.1 三口径簇级 R@1：r3 vs r4（同拆分、同 seed；r4 训练/评估个体隔离）",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=10, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    # 组标签
    for gi, g in enumerate(groups):
        off = (gi - (n_g - 1) / 2) * width + width / 2
        ax.text(x[-1] + 1.0, 0.04, g, ha="center", fontsize=10, color="#444")
    out = RPT / "eval51_r3_vs_r4.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {out}")


def write_train_report():
    """训练报告 markdown（存 r4 目录）。"""
    m = load_metrics(ML / "r4")
    r3_m = load_metrics(ML / "r3")
    rows = []
    for pre, name in (("metrics", "常规（对半拆）"),
                      ("metrics_conservative", "保守（剔除同串）"),
                      ("metrics_series_split", "串抽样")):
        for group, key in (("整体", "overall"), ("历史库", "history_20140806"),
                           ("新批次", "new_batches")):
            d = _load_eval(pre).get(group, {})
            v3, v4 = d.get("r3"), d.get("r4")
            if v3 is None or v4 is None:
                continue
            rows.append((name, group, f"{v3:.3f}", f"{v4:.3f}",
                         f"{v4 - v3:+.3f}"))
    tbl = "\n".join(f"| {a} | {b} | {c} | {d} | {e} |"
                    for a, b, c, d, e in rows)
    report = f"""# r4 度量学习重训报告（2026-08-25）

**动机**：r3 训练见过历史库 01/03 群 31 个个体，E5.1 历史库侧评估为"背题成绩"。
重训 r4 实现训练/评估彻底隔离：训练集 = 历史库 43 个体，评估集 = 新批次 32 个体
（模型从未见过）。

## 训练配置

| 项 | r4 | r3（对照） |
|---|---|---|
| 训练标签 | individual_id（用户决定，候选级） | confirmed_identity（人工初审） |
| 训练 | {m['n_train']} 张 / {m['n_train_individuals']} 个体 | {r3_m['n_train']} 张 / {r3_m['n_train_individuals']} 个体 |
| 验证 | {m['n_val']} 张 / {m['n_val_individuals']} 个体 | {r3_m['n_val']} 张 / {r3_m['n_val_individuals']} 个体 |
| 阶段 | s1 冻结 backbone 训 head 20ep（lr 0.001）→ s2 解冻微调 25ep（lr 5e-6） | 同左 |
| 跨群 HN | λ={m['lambda_hn']}，batch 16，01/03 各半采样 | 同左 |
| 初始化 | r2 best.pt（head 形状不匹配自动跳过） | r2 best.pt |
| seed / val_n | {m['seed']} / {m['val_n']} | 同左 |

## 训练过程

- 预训练基线（r2 继承 backbone）val R@1 = **{m['pretrained_baseline_r1']:.3f}**；
- 两阶段完成后 best val R@1 = **{m['best_val_r1']:.3f}**（stage2 第 {m['best_epoch']} 个 epoch = 全局 {m['best_epoch'] + 20}）；
- r3 对照：基线 {r3_m['pretrained_baseline_r1']:.3f} → best {r3_m['best_val_r1']:.3f}
  （验证集不同：r4 77 张 vs r3 17 张，直接对比仅参考）；
- 曲线图：`training_curves.png`（同目录）。

## 评估对照（簇级 R@1，同一拆分与 seed=7）

| 口径 | 分组 | r3 | r4 | Δ |
|---|---|---|---|---|
{tbl}

**核心结论**：
- 保守口径（剔除同串）新批次 0.238 → 0.381（+14.3pp）：模型在未见个体上
  的真实跨目击识别率显著提升——训练/评估隔离后的最硬能力证据；
- 保守口径整体 0.364 → 0.477，且 r4 下保守（0.477）反超串抽样（0.419）：
  跨串检索从"抽样才能打中"变成"普遍可打中"；
- 串抽样口径平均数持平（0.419）是巧合配平：43 个体中 4 个命中互换，
  单图 0.341 → 0.311 为真实波动。

## 产物与复现

- 权重：`outputs/metric_learning/r4/best.pt`（+ best_stage1.pt）；
- 训练数据：`outputs/pilot/pilot_set_train_hist.csv`（由 `experiments/prep_train_hist.py` 生成）；
- 特征：`outputs/embeddings/embeddings_eval51_all_r4.npy(+meta)`（`experiments/eval51_extract_all.py --ckpt ... --out ...`）；
- 评估：`outputs/reports/cluster_retrieval_v2/metrics{{_conservative,_series_split}}__r4.json`；
- 复现命令：
```powershell
python experiments/prep_train_hist.py
python scripts/train_reid.py --pilot outputs/pilot/pilot_set_train_hist.csv --out outputs/metric_learning/r4
python experiments/eval51_extract_all.py --ckpt outputs/metric_learning/r4/best.pt --out outputs/embeddings/embeddings_eval51_all_r4.npy
python experiments/eval_cluster_retrieval_v2.py --feats outputs/embeddings/embeddings_eval51_all_r4.npy --meta outputs/embeddings/embeddings_eval51_all_r4_meta.csv
python experiments/eval51_conservative.py --feats-stem embeddings_eval51_all_r4
python experiments/eval51_series_split.py --feats-stem embeddings_eval51_all_r4
```
"""
    out = ML / "r4" / "train_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"[doc] -> {out}")


if __name__ == "__main__":
    fig_training_curves()
    fig_eval_compare()
    write_train_report()
