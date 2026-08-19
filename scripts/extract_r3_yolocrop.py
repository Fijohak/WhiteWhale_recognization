"""
用 r3（跨群 hard negative 微调）模型提取 YOLO 检测裁剪图的特征（2026-08-17）。

背景（E1/E2/E4 结论）：散图归档场景 YOLO 检测裁剪显著优于中心裁剪（Top1 分数
Wilcoxon p=3.4e-16）；特写图三者打平。工具链（query_app / assign_pool）统一改为
"YOLO 检测裁剪 + r3 特征"，本脚本一次性提取两份特征：

- pilot 199 张裁剪图（outputs/crops_yolo）→ 查询/归档 gallery 特征
- 散图池 202 张裁剪图（outputs/crops_yolo_pool）→ 散图归档 query 特征

fallback 图（未检出 → 中心 0.45 窗回退）直接对裁剪图提特征，meta 保留 fallback
标记可追溯。预处理与训练一致（Resize 256 → CenterCrop 224），与 r3 整图特征同分布。

用法：
    python scripts/extract_r3_yolocrop.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_metric_learning import (  # noqa: E402
    DEVICE, ReIDModel, make_backbone, INPUT_SIZE,
)


def extract(crops_dir: Path, manifest_csv: Path, pilot_csv: Path, out_npy: Path,
            ckpt: Path):
    """裁剪图目录 → r3 特征 + meta（行序 = manifest 顺序，含追溯字段）。"""
    import torchvision.transforms as T
    from PIL import Image

    m = pd.read_csv(manifest_csv)
    p = pd.read_csv(pilot_csv)
    ckpt_state = torch.load(ckpt, map_location=DEVICE)
    model = ReIDModel(make_backbone(), n_classes=ckpt_state["state"]["head.W"].shape[1])
    model.load_state_dict(ckpt_state["state"])
    model.eval()
    tf = T.Compose([
        T.Resize(INPUT_SIZE + 32), T.CenterCrop(INPUT_SIZE),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    missing = []
    with torch.no_grad():
        for iid in m["image_id"]:
            img_path = crops_dir / f"{iid}.jpg"
            if not img_path.exists():
                missing.append(iid)
                feats.append(np.full(768, np.nan, dtype=np.float32))
                continue
            x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
            feats.append(model.encode(x).cpu().numpy()[0])
    emb = np.stack(feats).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, emb)
    # meta：pilot 追溯字段 merge；pool 散图不在 pilot 中，session_id 从路径解析
    meta = m.merge(p[["image_id", "source_group", "session_id", "quality_band",
                      "confirmed_identity"]], on="image_id", how="left")
    missing_sess = meta["session_id"].isna()
    if missing_sess.any():
        meta.loc[missing_sess, "session_id"] = (
            meta.loc[missing_sess, "relative_path"]
            .str.extract(r"^(0[13])/")[0].map({"01": 1, "03": 3})
        )
    meta.to_csv(out_npy.with_name(out_npy.stem + "_meta.csv"),
                index=False, encoding="utf-8-sig")
    print(f"[extract] {crops_dir.name}: {len(emb)} 张 → {out_npy} "
          f"(fallback {int(m['fallback'].sum())} 张, 缺失 {len(missing)} 张)")

    # 记录特征来源（模型 + 裁剪方式），供工具链自动匹配
    import json
    cfg = {"model": "metric-learning-r3", "crop": "yolo",
           "ckpt": str(ckpt),
           "preprocess": "Resize256+CenterCrop224", "n": int(len(emb))}
    (out_npy.parent / f"{out_npy.stem}_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="r3 + YOLO 裁剪特征提取（工具链 gallery/query）")
    parser.add_argument("--crops", type=Path, default=base / "crops_yolo")
    parser.add_argument("--crops-pool", type=Path, default=base / "crops_yolo_pool")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--ckpt", type=Path,
                        default=base / "metric_learning" / "r3" / "best.pt")
    parser.add_argument("--out", type=Path, default=base / "embeddings")
    args = parser.parse_args()

    extract(args.crops, args.crops / "crops_manifest.csv", args.pilot,
            args.out / "embeddings_metric_r3_yolocrop.npy", args.ckpt)
    extract(args.crops_pool, args.crops_pool / "crops_manifest.csv", args.pilot,
            args.out / "embeddings_pool_r3_yolocrop.npy", args.ckpt)


if __name__ == "__main__":
    main()
