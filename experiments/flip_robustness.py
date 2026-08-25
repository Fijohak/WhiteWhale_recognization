"""
4.2 增强策略分析：外观特征对水平翻转的鲁棒性实测（历史库 199 张）。

问题：训练时水平翻转增强是否安全？若翻转后同体匹配相似度大幅下降，
说明特征与侧别强相关，翻转增强会把标签语义弄混；若几乎不变则翻转安全。

设计：
- 对每张裁剪图生成水平翻转（ImageOps.mirror），用 r3 微调权重提取特征（flip）
- 同体对：原图 vs 同体翻转图 的余弦（应≈原vs原，否则翻转破坏同体匹配）
- 跨体对：原图 vs 跨体翻转图（负样本分布）
- 对照：原vs原 同体/跨体 分布（现有 r3 特征直接算）
- 检索：翻转 query → 原图 gallery（模拟跨侧观测到的图，R@1/R@5）

输出：outputs/reports/flip_robustness/metrics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whitewhale.reid.embedding import MegaDescriptorMetricAdapter  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
CROPS = BASE / "outputs" / "crops_yolo"
CKPT = BASE / "outputs" / "metric_learning" / "r3" / "best.pt"
REPORT = BASE / "outputs" / "reports" / "flip_robustness"


def main():
    meta = pd.read_csv(BASE / "outputs" / "embeddings"
                       / "embeddings_metric_r3_yolocrop_v2_meta.csv")
    emb = np.load(BASE / "outputs" / "embeddings"
                  / "embeddings_metric_r3_yolocrop_v2.npy")
    # 只取裁剪图实际存在的 199 张（与轮廓实验同口径）
    ids = [str(iid) for iid in meta["image_id"]
           if (CROPS / f"{iid}.jpg").exists()]
    keep = [i for i, iid in enumerate(meta["image_id"]) if str(iid) in ids]
    meta = meta.iloc[keep].reset_index(drop=True)
    emb = emb[keep]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    n = len(ids)
    print(f"[data] {n} 张（r3 yolo 裁剪特征）")

    # 提取翻转特征
    model = MegaDescriptorMetricAdapter(CKPT)
    flip_feats = []
    batch = []
    for iid in ids:
        im = Image.open(CROPS / f"{iid}.jpg").convert("RGB")
        batch.append(ImageOps.mirror(im))
        if len(batch) == 16:
            flip_feats.append(model.encode(batch))
            batch = []
    if batch:
        flip_feats.append(model.encode(batch))
    flip = np.concatenate(flip_feats)
    flip = flip / np.linalg.norm(flip, axis=1, keepdims=True)
    print(f"[flip] 翻转特征提取完成 {flip.shape}")

    labels = np.asarray([str(x) for x in meta["confirmed_identity"]])
    same_pairs, diff_pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                diff_pairs.append((i, j))
            else:
                same_pairs.append((i, j))
    same_pairs = np.asarray(same_pairs)
    diff_pairs = np.asarray(diff_pairs)

    def stats(x):
        x = np.asarray(x)
        return {"mean": float(x.mean()), "p50": float(np.median(x)),
                "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90)), "n": int(len(x))}

    sim_orig_same = np.einsum("ij,ij->i", emb[same_pairs[:, 0]], emb[same_pairs[:, 1]])
    sim_orig_diff = np.einsum("ij,ij->i", emb[diff_pairs[:, 0]], emb[diff_pairs[:, 1]])
    sim_flip_same = np.einsum("ij,ij->i", emb[same_pairs[:, 0]], flip[same_pairs[:, 1]])
    sim_flip_diff = np.einsum("ij,ij->i", emb[diff_pairs[:, 0]], flip[diff_pairs[:, 1]])

    def best_acc(pos, neg):
        best, bt = 0.0, 0.0
        for t in np.linspace(-0.2, 0.99, 120):
            acc = ((pos >= t).mean() + (neg < t).mean()) / 2
            if acc > best:
                best, bt = float(acc), float(t)
        return best, bt

    acc_orig, t_orig = best_acc(sim_orig_same, sim_orig_diff)
    acc_flip, t_flip = best_acc(sim_flip_same, sim_flip_diff)
    results = {
        "_meta": {"data": "历史库 199 张 r3 yolo 裁剪特征",
                  "model": "megadescriptor-metric-learning-r3",
                  "flip": "ImageOps.mirror 水平翻转（模拟跨侧观测）"},
        "同体_原vs原": stats(sim_orig_same),
        "跨体_原vs原": stats(sim_orig_diff),
        "同体_原vs翻转": stats(sim_flip_same),
        "跨体_原vs翻转": stats(sim_flip_diff),
        "区分准确率_原vs原": {"acc": acc_orig, "threshold": t_orig},
        "区分准确率_原vs翻转": {"acc": acc_flip, "threshold": t_flip},
    }
    # 翻转 query 检索（leave-one-out，同体命中；与 E4/E5 口径一致）
    S = flip @ emb.T
    vc = pd.Series(labels).value_counts()
    q_mask = np.asarray([vc[lab] >= 2 for lab in labels])
    r1s, r5s = [], []
    for i in np.where(q_mask)[0]:
        s = S[i].copy()
        s[i] = -1.0
        order = np.argsort(-s)[:5]
        top = labels[order]
        r1s.append(int(labels[i] == top[0]))
        r5s.append(int(labels[i] in top))
    results["翻转query检索"] = {"R@1": float(np.mean(r1s)), "R@5": float(np.mean(r5s)),
                               "n_query": int(np.sum(q_mask))}
    print("\n[同体] 原vs原 p50={:.3f} | 原vs翻转 p50={:.3f} | 翻转后下降 {:.3f}".format(
        np.median(sim_orig_same), np.median(sim_flip_same),
        np.median(sim_orig_same) - np.median(sim_flip_same)))
    print("[跨体] 原vs原 p50={:.3f} | 原vs翻转 p50={:.3f}".format(
        np.median(sim_orig_diff), np.median(sim_flip_diff)))
    print("[区分] 原vs原 acc={:.3f} | 原vs翻转 acc={:.3f}".format(acc_orig, acc_flip))
    print("[检索] 翻转 query R@1={:.3f} R@5={:.3f}（对照 E4 原图 R@1≈0.495，"
          "当前同图口径历史库 ≈0.728）".format(np.mean(r1s), np.mean(r5s)))

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] -> {REPORT / 'metrics.json'}")


if __name__ == "__main__":
    main()
