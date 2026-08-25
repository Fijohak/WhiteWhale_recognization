"""
批量提取背鳍轮廓特征（4.3 原型，历史库 199 张裁剪图）。

输入：outputs/crops_yolo/*.jpg（YOLO 背鳍裁剪图）
输出：outputs/reports/contour_features/features.npz
      {image_id: 特征向量} + features_meta.csv（对齐 individual_id / session_id）
统计提取成功率与失败原因。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contour_features import run_all  # noqa: E402

CROPS = Path("outputs/crops_yolo")
META = Path("outputs/embeddings/embeddings_metric_r3_yolocrop_v2_meta.csv")
OUT = Path("outputs/reports/contour_features")


def main():
    meta = pd.read_csv(META)
    # 只取实际存在的裁剪图（meta 202 行中可能缺图）
    image_ids = [iid for iid in meta["image_id"]
                 if (CROPS / f"{iid}.jpg").exists()]
    print(f"裁剪图存在: {len(image_ids)} / {len(meta)}")
    feats = run_all(CROPS, image_ids)
    print(f"特征提取成功: {len(feats)} / {len(image_ids)}")
    failed = [i for i in image_ids if i not in feats]
    if failed:
        print(f"失败 {len(failed)} 张: {failed[:10]}")

    # 保存特征（按 image_ids 稳定顺序对齐 meta）；feats_sym = 对称化曲率版本（A14 对照）
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [iid for iid in image_ids if iid in feats]
    F = np.stack([feats[iid] for iid in rows])
    feats_sym = run_all(CROPS, rows, sym_curv=True)
    F_sym = np.stack([feats_sym[iid] for iid in rows])
    np.savez_compressed(OUT / "features.npz",
                        image_ids=np.asarray(rows),
                        features=F,
                        features_sym=F_sym)
    m = meta[meta["image_id"].isin(rows)].copy()
    m.to_csv(OUT / "features_meta.csv", index=False, encoding="utf-8-sig")
    print(f"特征维度: {F.shape[1]} | 已保存 {len(rows)} 张 → {OUT}")
    # 个体分布
    vc = m["confirmed_identity"].value_counts()
    multi = (vc >= 2).sum()
    print(f"个体数 {m['confirmed_identity'].nunique()}（多图个体 {multi}，"
          f"单图 {m['confirmed_identity'].nunique() - multi}）")
    print(f"覆盖: {F.shape[0]} 张特征中多图个体照片 {vc[vc>=2].sum()}")


if __name__ == "__main__":
    main()
