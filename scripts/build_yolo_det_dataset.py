"""
构建 YOLOv8 背鳍检测训练数据集。

输入：
- outputs/det_labels/sam_candidates.csv    SAM 辅助预标注框（xywh 像素坐标）
- outputs/det_labels/exclude.txt           人工抽查标记的不可用 image_id（每行一个，可选）
- outputs/pilot/pilot_set.csv              pilot 清单（取 sequence_guess / session_id 用于划分）

输出：
- datasets/dorsal_fin/
    images/{image_id}.jpg       图片（软拷贝：按 image_id 命名，可追溯）
    labels/{image_id}.txt       YOLO 格式标签（class 0 = 背鳍，归一化 xywh）
    train.txt / val.txt         训练/验证图片清单（相对路径）
    data.yaml                   数据集配置（供 ultralytics 使用）

划分约束（全局约束 3）：同一 Sequence（连拍号前缀）不得跨 train/val；
同一 session 尽量不跨 train/val（01/03 是两个独立群，检测任务亦保守处理）。

用法：
    python scripts/build_yolo_det_dataset.py
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="构建 YOLO 检测训练集")
    parser.add_argument("--candidates", type=Path,
                        default=base / "outputs" / "det_labels" / "sam_candidates.csv")
    parser.add_argument("--exclude", type=Path,
                        default=base / "outputs" / "det_labels" / "exclude.txt")
    parser.add_argument("--pilot", type=Path, default=base / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("src_dataset"), help="原始图片根目录（只读）")
    parser.add_argument("--out", type=Path, default=base / "datasets" / "dorsal_fin")
    parser.add_argument("--val-sessions", default=None,
                        help="指定验证 session 列表（逗号分隔，如 '3'）；默认按 sequence 洗牌 20%")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates)
    pilot = pd.read_csv(args.pilot)

    # 人工抽查剔除清单
    exclude = set()
    if args.exclude.exists():
        exclude = {ln.strip() for ln in args.exclude.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if exclude:
        before = len(cand)
        cand = cand[~cand["image_id"].isin(exclude)]
        print(f"[build] 剔除人工标记不可用 {before - len(cand)} 张")

    # 合并 sequence 字段用于划分
    df = cand.merge(pilot[["image_id", "sequence_guess", "session_id", "label"]], on="image_id", how="left")
    missing = df["sequence_guess"].isna().sum()
    print(f"[build] 候选 {len(df)} 张（缺 sequence_guess {missing} 张，回退用 label 前缀）")

    imgs_dir = args.out / "images"
    labels_dir = args.out / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        src = args.images_root / r["relative_path"]
        if not src.exists():
            print(f"[build] 跳过（原图不存在）: {r['image_id']}")
            continue
        dst_img = imgs_dir / f"{r['image_id']}{src.suffix.lower()}"
        shutil.copy2(src, dst_img)
        # YOLO 标签：归一化 cx cy w h（像素 xywh 原图坐标系 → [0,1]）
        img_w, img_h = _read_size(src)
        cx = (r["x"] + r["w"] / 2) / img_w
        cy = (r["y"] + r["h"] / 2) / img_h
        bw = r["w"] / img_w
        bh = r["h"] / img_h
        cx, cy, bw, bh = [max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh)]
        (labels_dir / f"{r['image_id']}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        rows.append({
            "image_id": r["image_id"], "img_rel": f"images/{dst_img.name}",
            "sequence": r["sequence_guess"] or r["label"] or r["image_id"],
            "session": int(r["session_id"]) if pd.notna(r["session_id"]) else -1,
        })

    d = pd.DataFrame(rows)
    print(f"[build] 成功拷贝 {len(d)} 张（失败 {len(df) - len(d)}）")

    # 划分：先按 sequence 分组，sequence 整体归属 train/val
    import numpy as np
    rng = np.random.default_rng(args.seed)
    seqs = d["sequence"].unique()
    rng.shuffle(seqs)
    if args.val_sessions:
        val_sessions = {int(s) for s in args.val_sessions.split(",")}
        val_mask = d["session"].isin(val_sessions)
    else:
        n_val_seq = max(1, int(len(seqs) * 0.2))
        val_seqs = set(seqs[:n_val_seq])
        val_mask = d["sequence"].isin(val_seqs)
    train_ids, val_ids = d[~val_mask]["img_rel"], d[val_mask]["img_rel"]
    # 清单写绝对路径（Windows 反斜杠）：ultralytics img2label_paths 依赖
    # os.sep 子串（\images\）做 images→labels 替换，相对/正斜杠路径会匹配失败
    (args.out / "train.txt").write_text(
        "\n".join(str((args.out / x).resolve()) for x in train_ids) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text(
        "\n".join(str((args.out / x).resolve()) for x in val_ids) + "\n", encoding="utf-8")
    print(f"[build] train {len(train_ids)} / val {len(val_ids)}")

    # data.yaml：path 写运行时生成的绝对路径（正斜杠）。
    # 原因：ultralytics 8.4 对相对 path 基于 cwd 解析，中文路径 cwd 下不稳定；
    # 绝对路径由 args.out 推导，非硬编码。
    (args.out / "data.yaml").write_text(
        f"path: {str(args.out.resolve()).replace(chr(92), '/')}\n"
        "train: train.txt\nval: val.txt\n"
        "names:\n  0: dorsal_fin\n", encoding="utf-8")


def _read_size(src: Path):
    from PIL import Image
    with Image.open(src) as im:
        return im.size


if __name__ == "__main__":
    main()
