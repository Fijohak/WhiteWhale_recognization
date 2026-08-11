"""
基线 embedding 提取。
用预训练模型（默认 MegaDescriptor-T-224，timm hf-hub 加载）对 Pilot Set 图片提取特征，
不裁剪（高分样本背鳍已是主体）。输出 embeddings.npy + embeddings_meta.csv。

离线验证模式：--mock 用随机向量代替真实特征，用于验证 pipeline 代码逻辑。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def load_model(name: str):
    import timm

    if name.lower() in ("mock", "none"):
        return None, 128  # mock：128 维随机特征
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, model.num_features


def preprocess(image: Image.Image, device: torch.device, size: int = 224):
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tf(image).unsqueeze(0)
    return x.to(device)


def extract_embeddings(pilot_csv: Path, images_root: Path, model_name: str,
                       out_dir: Path, batch_size: int = 16, device: str = "auto"):
    df = pd.read_csv(pilot_csv)
    model, feat_dim = load_model(model_name)
    is_mock = model is None

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")

    embeddings = np.zeros((len(df), feat_dim), dtype=np.float32)
    errors = []
    rng = np.random.default_rng(42)
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        if is_mock:
            # mock：随机特征，用于离线验证 pipeline 逻辑
            embeddings[start:start + len(batch_df)] = rng.standard_normal(
                (len(batch_df), feat_dim)).astype(np.float32)
            continue
        imgs = []
        for _, row in batch_df.iterrows():
            p = images_root / row["relative_path"]
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB"), dev))
            except Exception as e:  # noqa: BLE001
                errors.append((row["image_id"], str(p), str(e)))
                imgs.append(torch.zeros(1, 3, 224, 224, device=dev))
        if imgs:
            with torch.no_grad():
                out = model(torch.cat(imgs, 0))
            embeddings[start:start + len(batch_df)] = out.float().cpu().numpy()

    # L2 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    df[["image_id"]].to_csv(out_dir / "embeddings_meta.csv", index=False)

    print(f"模型: {model_name} (mock={is_mock}) 设备: {dev}")
    print(f"提取 {len(df)} 张 → embeddings.npy {embeddings.shape} (L2 归一化)")
    if errors:
        print(f"警告: {len(errors)} 张图片读取失败: {errors[:3]}")

    # 写入实验记录，便于回溯
    with open(out_dir / "embedding_config.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "mock": is_mock, "device": str(dev),
                   "feat_dim": feat_dim, "batch_size": batch_size,
                   "images_root": str(images_root), "n": len(df)}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 Pilot Set 基线 embedding")
    parser.add_argument("--pilot", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"),
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--model", default="hf-hub:BVRA/MegaDescriptor-T-224")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "embeddings")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mock", action="store_true", help="离线验证模式（随机特征）")
    args = parser.parse_args()
    extract_embeddings(args.pilot, args.images_root, "mock" if args.mock else args.model,
                       args.out, args.batch_size, args.device)
