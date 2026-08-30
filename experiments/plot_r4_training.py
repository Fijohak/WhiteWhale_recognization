"""
r4 历史训练曲线与审计报告生成器。

r4 权重保留为当前严格协议的基线，但其旧训练/评测语义已发现缺陷。本脚本只生成
历史曲线和审计报告，不再绘制或发布旧的 r3/r4“提升”对照图。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
ML = REPO_ROOT / "outputs" / "metric_learning"
STRICT_RETRIEVAL = (
    REPO_ROOT / "outputs" / "verification" / "r4_yolocrop_v3" / "retrieval.json")
STRICT_E5 = (
    REPO_ROOT / "outputs" / "reports" / "cluster_retrieval_session_local_v1")

C_DARK = "#1E3A5F"
C_RED = "#BF0000"
C_R3 = "#9FB4C7"


def load_history(run_dir: Path) -> pd.DataFrame:
    """读取历史训练日志并生成连续 epoch 编号。"""
    history = pd.read_csv(run_dir / "history.csv")
    stage1_epochs = int(history.loc[history["stage"] == 1, "epoch"].max())
    history["epoch_global"] = history["epoch"] + (
        history["stage"] - 1) * stage1_epochs
    return history


def load_metrics(run_dir: Path) -> dict:
    """读取训练摘要。"""
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict:
    """读取已实际生成的评测 JSON；缺失时明确失败，禁止补造数字。"""
    if not path.is_file():
        raise FileNotFoundError(f"审计报告缺少实测产物: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fig_training_curves() -> Path:
    """绘制 r3/r4 历史训练曲线；图题明确其不可作为公平模型比较。"""
    r3, r4 = load_history(ML / "r3"), load_history(ML / "r4")
    metrics4 = load_metrics(ML / "r4")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].plot(r4["epoch_global"], r4["loss"], color=C_DARK, lw=2,
                 label="r4 历史 loss")
    axes[0].plot(r3["epoch_global"], r3["loss"], color=C_R3, lw=1.5,
                 ls="--", label="r3 历史 loss（验证集不同）")
    axes[0].set(xlabel="epoch", ylabel="记录的 loss",
                title="历史训练 loss（仅供审计）")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].plot(r4["epoch_global"], r4["val_r1"], color=C_DARK, lw=2,
                 label="r4 历史 val R@1")
    axes[1].plot(r3["epoch_global"], r3["val_r1"], color=C_R3, lw=1.5,
                 ls="--", label="r3 历史 val R@1（验证集不同）")
    best = r4.loc[r4["val_r1"].idxmax()]
    axes[1].scatter([best["epoch_global"]], [best["val_r1"]],
                    color=C_RED, zorder=5, s=40)
    axes[1].axhline(metrics4["pretrained_baseline_r1"],
                    color="#888", ls=":", lw=1)
    axes[1].set(xlabel="epoch", ylabel="历史 val R@1",
                title="历史验证曲线（不代表修正后协议）")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.suptitle("r4 历史训练审计：旧曲线不可用于证明相对提升",
                 fontsize=13, fontweight="bold", color=C_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ML / "r4" / "training_curves.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] -> {out}")
    return out


def write_train_report() -> Path:
    """依据严格重评产物写 r4 审计报告，不再复述旧提升结论。"""
    training = load_metrics(ML / "r4")
    retrieval = _read_json(STRICT_RETRIEVAL)
    conservative = _read_json(
        STRICT_E5 / "metrics_conservative__r4_v2.json")
    series_split = _read_json(
        STRICT_E5 / "metrics_series_split__r4_v2.json")
    missed = _read_json(
        REPO_ROOT / "outputs" / "reports"
        / "missed_diag_r4_v2_full_manifest_v3" / "summary.json")

    report = f"""# r4 历史训练审计报告（2026-08-29）

## 定位

r4 权重继续作为严格协议下的历史基线，但不再视为“修正后训练模型”，也不再据旧 r3/r4 图表宣称提升。现有照片时间间隔较短，本报告不能证明跨月或跨年能力。

## 已确认的旧训练缺陷

1. ArcFace 曾把不同 session 中未对齐的本地身份类当作负类。
2. CE 曾把同一连拍串中共现的其他身份原型当作负类；旧 HN 只在 triplet 中剔除同串。
3. 旧采样在固定 seed 的 40 批中有 6 批没有合法跨串 anchor，却静默退化为纯 CE。
4. 旧训练/验证拆分和旧评测产物未完整记录 full-manifest 稳定串及生成期行绑定。

以上问题已在新训练链路中修复，但不能反向证明旧 r4 的训练过程有效。

## r4 权重的严格重评（基线，不是提升证明）

- 正式历史库：202 张，43 个批次内身份；完整 manifest 全局分串，query/gallery 串重叠 0。
- 图级跨串检索：49 query / 153 gallery，R@1 = **{retrieval['r1']:.3f}**，mAP = **{retrieval['map']:.3f}**。
- E5 保守口径（140 张可评 query，11 张 skipped）：单图 R@1 = **{conservative['overall']['single_R@1']:.3f}**，簇级 R@1 = **{conservative['overall']['cluster_R@1']:.3f}**，MRR@10 = **{conservative['overall']['cluster_MRR@10']:.3f}**。
- E5 串抽样口径（97 张可评 query）：单图 R@1 = **{series_split['overall']['single_R@1']:.3f}**，簇级 R@1 = **{series_split['overall']['cluster_R@1']:.3f}**，MRR@10 = **{series_split['overall']['cluster_MRR@10']:.3f}**。
- 保守口径未命中：{missed['n_missed_identities']} 个身份 / {missed['n_missed_query_images']} 张 query；其中 {missed['n_identities_outside_topk']} 个身份不在 Top-{missed['topk']}。

E5 只在 query 所属 session 内构造候选，并逐 query 排除完整同串；跨 session 身份未对齐，
不作为候选或负类。单真值簇排名报告的是 MRR@10，不是 mAP。旧跨 session E5 数字无效，
不得继续引用。上述特征来自 schema 2、生成期 row-binding 的重提产物；旧 schema 1 回填
产物仅保留为历史诊断。

## 旧训练记录

- train/val：{training['n_train']} / {training['n_val']} 张；历史 best val R@1 = {training['best_val_r1']:.3f}。
- 权重：`outputs/metric_learning/r4/best.pt`。
- 该 val 数字沿用旧协议，不能与新 r5 的修正验证结果直接比较。

## 后续门禁

新候选模型必须从原始 MegaDescriptor 初始化，使用 session-aware + series-aware CE、非零合法-anchor HN、完整串隔离和严格产物重提。只有在相同严格评测协议上超过本报告的 r4 基线后，才可考虑切换生产配置。
"""
    out = ML / "r4" / "train_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"[doc] -> {out}")
    return out


if __name__ == "__main__":
    fig_training_curves()
    write_train_report()
