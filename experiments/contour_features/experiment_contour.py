"""
轮廓特征主实验（4.3 + A14 + 4.2 + 4.5）：历史库 199 张背鳍轮廓特征。

一、A14 镜像对称性验证（特征级）：
    - 同体 原vs镜像 相似度（应高 → 背鳍剪影近似对称）
    - 跨体 原vs镜像 相似度（应低 → 特征有区分度）
    - 对照：对称化特征（|曲率|）版本
二、检索对比（leave-one-out，同体命中）：
    A. 轮廓特征 原图→原图     B. 轮廓特征 镜像query→原图gallery（跨侧模拟）
    C. 对称化特征 原图→原图   D. 外观特征(r3) 基线
    E. 镜像增强 gallery：gallery 每图+镜像 → 镜像 query（A14 应用）
三、错误分析（4.5）：R@1 失败案例清单与个体级分布

输出：outputs/reports/contour_features/metrics.json + fail_cases.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contour_features import contour_feature, feature_sim, free_edge, \
    mirror_edge, segment_fin  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
CROPS = BASE / "outputs" / "crops_yolo"
REPORT = BASE / "outputs" / "reports" / "contour_features"
K_LIST = (1, 5)


def load_all() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """加载 meta、轮廓特征（非对称/对称化）、外观特征（r3，按存在图对齐）。"""
    meta = pd.read_csv(REPORT / "features_meta.csv")
    z = np.load(REPORT / "features.npz", allow_pickle=False)
    feats, feats_sym = z["features"], z["features_sym"]
    row_ids = [str(x) for x in z["image_ids"]]
    # 外观特征（r3 yolo 裁剪，正式特征源）按 image_id 对齐
    a_meta = pd.read_csv(BASE / "outputs" / "embeddings"
                         / "embeddings_metric_r3_yolocrop_v2_meta.csv")
    a_emb = np.load(BASE / "outputs" / "embeddings"
                    / "embeddings_metric_r3_yolocrop_v2.npy")
    a_map = {str(iid): i for i, iid in enumerate(a_meta["image_id"])}
    a_feats = np.stack([a_emb[a_map[iid]] for iid in row_ids])
    meta = meta.reset_index(drop=True)
    return meta, feats, feats_sym, a_feats


def compute_mirror_feats(row_ids: list[str], sym: bool) -> np.ndarray:
    """对每张图算真实镜像特征（edge → mirror_edge → feature），模拟跨侧观测。"""
    out = np.empty((len(row_ids), 27), dtype=np.float32)
    for k, iid in enumerate(row_ids):
        img = _read(iid)
        mask = segment_fin(img)
        edge = free_edge(mask)
        f = contour_feature(mirror_edge(edge), sym_curv=sym)
        out[k] = f if f is not None else np.full(27, np.nan)
    return out


def _read(iid: str) -> np.ndarray:
    """读裁剪图（PIL 而非 cv2.imread：Windows 下 cv2 不支持中文绝对路径）。"""
    from PIL import Image
    return np.asarray(Image.open(CROPS / f"{iid}.jpg").convert("RGB"))


def cosine_matrix(F: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    """余弦相似度矩阵 (N, N)：cos(F_i, Q_j)。Q=None 时 Q=F。"""
    Q = F if Q is None else Q
    fn = np.linalg.norm(F, axis=1, keepdims=True)
    qn = np.linalg.norm(Q, axis=1, keepdims=True)
    fn[fn == 0] = 1.0
    qn[qn == 0] = 1.0
    return (F @ Q.T) / (fn @ qn.T)


def retrieval_eval(F: np.ndarray, meta: pd.DataFrame, ids: list[str],
                   mirror_query: bool = False, mirror_gallery: bool = False,
                   mirror_qf: np.ndarray | None = None) -> dict:
    """leave-one-out 检索：query=多图个体照片，gallery=排除自身。

    mirror_query=True 时 query 用镜像特征（模拟跨侧观测到的图），
    mirror_gallery=True 时 gallery 每张图拼接其镜像特征（镜像增强）。
    返回 R@1/R@5 与逐图 top1 表。
    """
    labels = np.asarray([str(x) for x in meta["confirmed_identity"]])
    n = len(F)
    # 多图个体照片才作 query（gallery 中须有同体另一张）
    vc = pd.Series(labels).value_counts()
    q_mask = np.asarray([vc[lab] >= 2 for lab in labels])
    Q = mirror_qf if (mirror_query and mirror_qf is not None) else F
    if mirror_gallery:
        Gm = mirror_qf
        S = np.hstack([cosine_matrix(Q, F), cosine_matrix(Q, Gm)])
        gal_ids = np.concatenate([labels, labels])
    else:
        S = cosine_matrix(Q, F)
        gal_ids = labels
    rows = []
    for i in np.where(q_mask)[0]:
        scores = S[i].copy()
        scores[i] = -1.0  # 排除自身（gallery 拼接时原图列排除）
        if mirror_gallery:
            scores[n + i] = -1.0  # 同时排除自己的镜像列（镜像 query 的平凡命中）
        order = np.argsort(-scores)[:10]
        top_ids = [gal_ids[j] for j in order]
        r1 = int(labels[i] == top_ids[0])
        r5 = int(labels[i] in top_ids[:5])
        rows.append({"image_id": ids[i], "identity": labels[i],
                     "top1": top_ids[0], "top1_sim": float(scores[order[0]]),
                     "r1": r1, "r5": r5})
    df = pd.DataFrame(rows)
    return {"R@1": float(df["r1"].mean()), "R@5": float(df["r5"].mean()),
            "n_query": len(df), "cases": df}


def main():
    meta, feats, feats_sym, a_feats = load_all()
    ids = [str(x) for x in meta["image_id"]]
    n = len(ids)
    print(f"[data] {n} 张 / {meta['confirmed_identity'].nunique()} 个体")
    # 镜像特征（模拟跨侧观测；对每张图真实翻转轮廓重算）
    mirror_f = compute_mirror_feats(ids, sym=False)
    mirror_f_sym = compute_mirror_feats(ids, sym=True)
    nan_cnt = int(np.isnan(mirror_f).sum() // 27)
    print(f"[mirror] 镜像特征计算完成（失败 {nan_cnt} 张）")

    results: dict = {"_meta": {
        "data": "历史库 20140806 01/03 labeled 199 张（YOLO 裁剪），43 组",
        "feature": "轮廓特征 27 维（曲率直方图16+缺口8+比例3）；对称化=|曲率|",
        "query": "多图个体照片 leave-one-out（排除自身）；同体=confirmed_identity",
        "sim": "cosine"}}

    # ---------- 一、A14 镜像对称性验证 ----------
    labels = np.asarray([str(x) for x in meta["confirmed_identity"]])
    same_pairs, diff_pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                diff_pairs.append((i, j))
            elif labels[i] != "nan":
                same_pairs.append((i, j))
    # 同体/跨体 原vs原（基线：轮廓特征本身的区分度，与镜像无关）
    same_orig = [feature_sim(feats[i], feats[j]) for i, j in same_pairs]
    diff_orig = [feature_sim(feats[i], feats[j]) for i, j in diff_pairs]
    # 同体 原vs镜像 相似度：i 的原特征 vs i 的镜像特征（自对称性）
    self_mirror = [feature_sim(feats[i], mirror_f[i]) for i in range(n)]
    # 同体 跨图原vs镜像：i 原特征 vs j 同体镜像特征
    same_cross = [feature_sim(feats[i], mirror_f[j]) for i, j in same_pairs]
    diff_cross = [feature_sim(feats[i], mirror_f[j]) for i, j in diff_pairs]
    # 对称化特征版本对照
    self_mirror_sym = [feature_sim(feats_sym[i], mirror_f_sym[i]) for i in range(n)]
    same_cross_sym = [feature_sim(feats_sym[i], mirror_f_sym[j]) for i, j in same_pairs]
    diff_cross_sym = [feature_sim(feats_sym[i], mirror_f_sym[j]) for i, j in diff_pairs]

    def dist_stats(x):
        x = np.asarray(x)
        return {"mean": float(x.mean()), "p50": float(np.median(x)),
                "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90)), "n": int(len(x))}

    a14 = {
        "基线_同体原vs原": dist_stats(same_orig),
        "基线_跨体原vs原": dist_stats(diff_orig),
        "非对称特征_自身原vs镜像": dist_stats(self_mirror),
        "非对称特征_同体跨图原vs镜像": dist_stats(same_cross),
        "非对称特征_跨体原vs镜像": dist_stats(diff_cross),
        "对称化特征_自身原vs镜像": dist_stats(self_mirror_sym),
        "对称化特征_同体跨图原vs镜像": dist_stats(same_cross_sym),
        "对称化特征_跨体原vs镜像": dist_stats(diff_cross_sym),
    }
    # 阈值扫描：用 原vs镜像 相似度区分 同体/跨体 的最佳准确率（验证镜像特征可用）
    def best_acc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        best, bt = 0.0, 0.0
        for t in np.linspace(0.30, 0.95, 131):
            acc = ((pos >= t).mean() + (neg < t).mean()) / 2
            if acc > best:
                best, bt = float(acc), float(t)
        return best, bt
    acc1, t1 = best_acc(same_cross, diff_cross)
    acc2, t2 = best_acc(same_cross_sym, diff_cross_sym)
    a14["区分准确率_非对称"] = {"acc": acc1, "threshold": t1}
    a14["区分准确率_对称化"] = {"acc": acc2, "threshold": t2}
    results["A14_镜像对称性"] = a14
    print("\n[A14-基线] 同体 原vs原 p50={:.3f} (n={}) | 跨体 原vs原 p50={:.3f} (n={})".format(
        np.median(same_orig), len(same_orig), np.median(diff_orig), len(diff_orig)))
    print("[A14] 非对称: 自身 原vs镜像 p50={:.3f} | 同体跨图 p50={:.3f} "
          "(n={}) | 跨体 p50={:.3f} (n={}) | 区分 acc={:.3f}".format(
        np.median(self_mirror), np.median(same_cross), len(same_cross),
        np.median(diff_cross), len(diff_cross), acc1))
    print("[A14] 对称化: 自身 p50={:.3f} | 同体跨图 p50={:.3f} | 跨体 p50={:.3f} "
          "| 区分 acc={:.3f}".format(np.median(self_mirror_sym),
                                     np.median(same_cross_sym),
                                     np.median(diff_cross_sym), acc2))

    # ---------- 二、检索对比 ----------
    ret = {}
    variants = {
        "A_轮廓_原图": (feats, False, False, None),
        "B_轮廓_镜像query": (feats, True, False, mirror_f),
        "C_对称化_原图": (feats_sym, False, False, None),
        "E_轮廓_镜像query+镜像gallery": (feats, True, True, mirror_f),
    }
    for name, (F, mq, mg, qf) in variants.items():
        r = retrieval_eval(F, meta, ids, mirror_query=mq, mirror_gallery=mg,
                           mirror_qf=qf)
        ret[name] = {"R@1": r["R@1"], "R@5": r["R@5"], "n_query": r["n_query"]}
        print(f"[检索] {name}: R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
              f"(n={r['n_query']})")
    # D. 外观特征基线（r3 yolo 裁剪）
    r = retrieval_eval(a_feats, meta, ids)
    ret["D_外观_r3基线"] = {"R@1": r["R@1"], "R@5": r["R@5"], "n_query": r["n_query"]}
    print(f"[检索] D_外观_r3基线: R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
          f"(n={r['n_query']})")
    results["检索对比"] = ret

    # ---------- 三、错误分析（4.5） ----------
    cases_a = retrieval_eval(feats, meta, ids)["cases"]
    cases_d = retrieval_eval(a_feats, meta, ids)["cases"]
    fail = pd.DataFrame({
        "image_id": cases_a["image_id"],
        "identity": cases_a["identity"],
        "轮廓_top1": cases_a["top1"],
        "轮廓_top1_sim": cases_a["top1_sim"],
        "轮廓_r1": cases_a["r1"],
        "外观_top1": cases_d["top1"],
        "外观_top1_sim": cases_d["top1_sim"],
        "外观_r1": cases_d["r1"],
    })
    fail["两者皆败"] = (fail["轮廓_r1"] == 0) & (fail["外观_r1"] == 0)
    fail.to_csv(REPORT / "fail_cases.csv", index=False, encoding="utf-8-sig")
    # 个体级成功率
    ind_fail = fail.groupby("identity").agg(
        轮廓成功率=("轮廓_r1", "mean"), 外观成功率=("外观_r1", "mean"),
        张数=("image_id", "count"))
    ind_fail = ind_fail.sort_values("轮廓成功率").round(3)
    ind_fail.to_csv(REPORT / "individual_rates.csv", encoding="utf-8-sig")
    results["错误分析"] = {
        "失败案例文件": "fail_cases.csv",
        "两者皆败张数": int(fail["两者皆败"].sum()),
        "两者皆败占比": float(fail["两者皆败"].mean()),
        "个体级成功率表": "individual_rates.csv",
        "两者皆败最多的个体": ind_fail[ind_fail["两者皆败"].map(
            lambda _: True)].head(5).to_dict() if False else
            ind_fail.sort_values("张数", ascending=False).head(5).to_dict(),
    }
    print(f"\n[错误分析] 两者皆败 {fail['两者皆败'].sum()}/{len(fail)} "
          f"({fail['两者皆败'].mean():.1%})")
    print(f"[错误分析] 个体级成功率(轮廓) 最低 5 个:")
    print(ind_fail.head(5).to_string())

    (REPORT / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] -> {REPORT / 'metrics.json'}")


if __name__ == "__main__":
    main()
