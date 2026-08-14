"""
中华白海豚个体识别：伪标签度量学习训练（ArcFace + 余弦 margin）。

数据：pilot_set.csv 中 confirmed_identity 非空的行（人工初审伪标签，
     31 个体 / ~135 张；未复核，结果只叫 Candidate 特征）。
方法：MegaDescriptor-T-224 backbone（timm，离线缓存）+ ArcFace 头
     （s=32, m=0.3）。两阶段：先冻结 backbone 只训 head，
     再解冻 backbone 低学习率微调（防过拟合小数据集）。
语义约束（CLAUDE.md）：
- 伪标签来自人工初审，不是专家复核 → 结果叫 Candidate，不叫个体；
- 不使用水平翻转（左右侧背鳍特征可能不同）；
- 划分按个体（同一体不跨 train/val）；
- 评估协议与 local_reid_benchmark 一致（leave-one-out R@1）。

用法：
    python scripts/train_metric_learning.py --out outputs/metric_learning/r1
    # 用训练好的模型重新提取 199 张特征，再跑对比：
    python scripts/train_metric_learning.py --extract --out outputs/metric_learning/r1
    python scripts/local_reid_benchmark.py \\
        --embeddings outputs/embeddings/embeddings_metric.npy \\
        --meta outputs/embeddings/embeddings_meta.csv \\
        --out outputs/reports/benchmark_metric
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# MegaDescriptor 输入规格（与 src/model/reid/embedding/base.py 一致）
INPUT_SIZE = 224
FEAT_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ArcFaceHead(nn.Module):
    """余弦 margin 分类头：feat L2 归一化 × W L2 归一化 → cos+m 角度 → ×s。"""

    def __init__(self, feat_dim: int, n_classes: int, s: float = 32.0, m: float = 0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(feat_dim, n_classes) * 0.1)

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(feat, dim=1)
        w = F.normalize(self.W, dim=0)
        cos = torch.clamp(feat @ w, -1.0 + 1e-7, 1.0 - 1e-7)     # (N, C)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)                       # 类内角 + margin
        one_hot = F.one_hot(labels, num_classes=cos.size(1)).float()
        cos_margin = cos * (1 - one_hot) + target * one_hot
        return cos_margin * self.s


class ReIDModel(nn.Module):
    """backbone + ArcFace 头。backbone 输出归一化特征（可检索）。"""

    def __init__(self, backbone, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = ArcFaceHead(FEAT_DIM, n_classes)

    def forward(self, x, labels):
        feat = F.normalize(self.backbone(x), dim=1)
        return self.head(feat, labels)

    def encode(self, x) -> torch.Tensor:
        return F.normalize(self.backbone(x), dim=1)


def make_backbone():
    """离线加载 MegaDescriptor（timm hf-hub 缓存已有，不访问外网）。"""
    import os
    import timm

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                              pretrained=True, num_classes=0)
    model.eval()
    return model.to(DEVICE)


def load_confirmed(pilot_csv: Path, images_root: Path) -> pd.DataFrame:
    """读取人工初审确认的照片（confirmed_identity 非空），映射个体 → 类别号。"""
    p = pd.read_csv(pilot_csv)
    df = p[p["confirmed_identity"].notna()
           & (p["confirmed_identity"].astype(str).str.strip() != "")].copy()
    df["path"] = df["relative_path"].map(lambda rp: str(images_root / rp))
    id2idx = {iid: i for i, iid in enumerate(sorted(df["confirmed_identity"].unique()))}
    df["label_idx"] = df["confirmed_identity"].map(id2idx)
    return df


def split_by_individual(df: pd.DataFrame, val_n: int, seed: int):
    """按个体留出验证：同一体不跨 train/val。验证个体照片数 ≥2（可 leave-one-out）。"""
    rng = np.random.default_rng(seed)
    sizes = df.groupby("confirmed_identity").size()
    candidates = sizes[sizes >= 2].index.tolist()
    rng.shuffle(candidates)
    val_ids = set(candidates[:val_n])
    train = df[~df["confirmed_identity"].isin(val_ids)]
    val = df[df["confirmed_identity"].isin(val_ids)]
    assert not train.empty and not val.empty, "留出后有空集，请减小 val_n"
    return train, val


class DolphinDataset(Dataset):
    """加载原图（80 分以上为背鳍特写，直接用原图 resize）。"""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def make_transforms(train: bool):
    import torchvision.transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        # 模拟拍摄变化；刻意不用水平翻转（左右侧特征不同，CLAUDE.md）
        return T.Compose([
            T.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
            T.RandomRotation(8),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(), normalize,
        ])
    return T.Compose([
        T.Resize(INPUT_SIZE + 32), T.CenterCrop(INPUT_SIZE),
        T.ToTensor(), normalize,
    ])


@torch.no_grad()
def eval_retrieval(model: ReIDModel, loader: DataLoader) -> float:
    """验证个体内 leave-one-out R@1（cosine 检索，排除自身）。"""
    model.eval()
    feats, labs = [], []
    for x, y in loader:
        f = model.encode(x.to(DEVICE))
        feats.append(f.cpu()); labs.append(y)
    feats = torch.cat(feats).numpy()          # (N, D) 已 L2 归一化
    labs = torch.cat(labs).numpy()
    sim = feats @ feats.T
    np.fill_diagonal(sim, -1.0)
    top1 = sim.argmax(axis=1)
    return float((labs[top1] == labs).mean())


def train_one_epoch(model, loader, opt, n_classes):
    model.train()
    total, correct = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x, y)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total += len(y)
        correct += (logits.argmax(1) == y).sum().item()
    return float(loss.item()), correct / total


def run_training(args, df: pd.DataFrame, out_dir: Path):
    """两阶段训练：冻结 backbone 训 head → 解冻低学习率微调。"""
    train_df, val_df = split_by_individual(df, args.val_n, args.seed)
    # 重新映射训练个体编号为 0..n-1（load_confirmed 的编号是全局的，
    # 留出验证个体后会出现空洞，导致 one_hot / CUDA 索引越界）
    train_df = train_df.copy()
    id2idx = {iid: i for i, iid in enumerate(sorted(train_df["confirmed_identity"].unique()))}
    train_df["label_idx"] = train_df["confirmed_identity"].map(id2idx)
    print(f"[train] 训练 {len(train_df)} 张 / {train_df['confirmed_identity'].nunique()} 个体；"
          f"验证 {len(val_df)} 张 / {val_df['confirmed_identity'].nunique()} 个体")
    tr_loader = DataLoader(DolphinDataset(train_df, make_transforms(True)),
                           batch_size=args.batch, shuffle=True,
                           num_workers=0, drop_last=False)
    val_loader = DataLoader(DolphinDataset(val_df, make_transforms(False)),
                            batch_size=32, shuffle=False)

    model = ReIDModel(make_backbone(), n_classes=train_df["label_idx"].nunique())
    model = model.to(DEVICE)
    history = []
    best_r1, best_epoch = 0.0, 0

    # 阶段一：冻结 backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.head.parameters() if p.requires_grad],
                            lr=args.lr_head, weight_decay=5e-4)
    print("[train] 阶段一：冻结 backbone，训练 ArcFace head "
          f"({args.epochs_stage1} epochs, lr={args.lr_head})")
    for ep in range(1, args.epochs_stage1 + 1):
        loss, acc = train_one_epoch(model, tr_loader, opt, train_df["label_idx"].nunique())
        r1 = eval_retrieval(model, val_loader)
        history.append({"stage": 1, "epoch": ep, "loss": loss, "acc": acc, "val_r1": r1})
        if r1 > best_r1:
            best_r1, best_epoch = r1, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "val_r1": r1},
                       out_dir / "best_stage1.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"  [s1] ep {ep:2d}  loss {loss:.4f}  acc {acc:.3f}  val R@1 {r1:.3f}")

    # 阶段二：解冻 backbone 低学习率微调
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=5e-4)
    print("[train] 阶段二：解冻 backbone 微调 "
          f"({args.epochs_stage2} epochs, lr={args.lr_backbone})")
    for ep in range(1, args.epochs_stage2 + 1):
        loss, acc = train_one_epoch(model, tr_loader, opt, train_df["label_idx"].nunique())
        r1 = eval_retrieval(model, val_loader)
        history.append({"stage": 2, "epoch": ep, "loss": loss, "acc": acc, "val_r1": r1})
        if r1 > best_r1:
            best_r1, best_epoch = r1, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "val_r1": r1,
                        "stage": 2},
                       out_dir / "best.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"  [s2] ep {ep:2d}  loss {loss:.4f}  acc {acc:.3f}  val R@1 {r1:.3f}")

    # 预训练基线：同一验证个体上，未微调 MegaDescriptor 的检索 R@1
    pretrained_r1 = eval_retrieval(ReIDModel(make_backbone(), 1).to(DEVICE), val_loader)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "伪标签来自人工初审（非专家复核），指标只代表验证个体上的检索一致性",
            "n_train": len(train_df), "n_val": len(val_df),
            "n_train_individuals": int(train_df["confirmed_identity"].nunique()),
            "n_val_individuals": int(val_df["confirmed_identity"].nunique()),
            "val_n": args.val_n, "seed": args.seed,
            "pretrained_baseline_r1": pretrained_r1,
            "best_val_r1": best_r1, "best_epoch": best_epoch,
        }, f, indent=2, ensure_ascii=False)
    print(f"[train] 完成。预训练基线 val R@1 = {pretrained_r1:.3f} → "
          f"微调后 {best_r1:.3f}（epoch {best_epoch}）→ {out_dir}")


def extract_features(args, out_dir: Path, pilot_csv: Path, images_root: Path):
    """用 best.pt 重新提取全部 pilot 特征（供 local_reid_benchmark 对比）。"""
    import torchvision.transforms as T

    p = pd.read_csv(pilot_csv)
    ckpt = torch.load(out_dir / "best.pt", map_location=DEVICE)
    model = ReIDModel(make_backbone(), n_classes=ckpt["state"]["head.W"].shape[1])
    model.load_state_dict(ckpt["state"])
    model.eval()
    tf = T.Compose([
        T.Resize(INPUT_SIZE + 32), T.CenterCrop(INPUT_SIZE),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for rp in p["relative_path"]:
            img = Image.open(images_root / rp).convert("RGB")
            x = tf(img).unsqueeze(0).to(DEVICE)
            feats.append(model.encode(x).cpu().numpy()[0])
    emb = np.stack(feats).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out_emb = args.embeddings_out
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, emb)
    # meta 与特征同行序配套保存（行序 = pilot_set.csv 顺序，含 image_id 可追溯）
    p[["image_id"]].to_csv(out_emb.with_name(out_emb.stem + "_meta.csv"),
                           index=False, encoding="utf-8-sig")
    with open(out_emb.with_name("embedding_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "megadescriptor-metric-learning-r1",
                   "note": "伪标签 ArcFace 微调后特征（人工初审标签，Candidate 级）",
                   "feat_dim": FEAT_DIM, "n": len(p), "source": str(out_dir / "best.pt")},
                  f, indent=2, ensure_ascii=False)
    print(f"[extract] {len(p)} 张特征 → {out_emb}（模型 {out_dir / 'best.pt'}）")


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="伪标签度量学习训练（ArcFace）")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"))
    parser.add_argument("--out", type=Path, default=base / "metric_learning" / "r1")
    parser.add_argument("--val-n", type=int, default=6, help="验证个体数（按个体留出）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-stage1", type=int, default=30)
    parser.add_argument("--epochs-stage2", type=int, default=30)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--extract", action="store_true",
                        help="用已训练的 best.pt 重新提取特征")
    parser.add_argument("--embeddings-out", type=Path,
                        default=base / "embeddings" / "embeddings_metric.npy")
    args = parser.parse_args()

    if args.extract:
        extract_features(args, args.out, args.pilot, args.images_root)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_confirmed(args.pilot, args.images_root)
    if df["confirmed_identity"].nunique() < args.val_n + 2:
        raise SystemExit(f"个体数 {df['confirmed_identity'].nunique()} 过少，无法留出 {args.val_n} 个验证个体")
    run_training(args, df, args.out)


if __name__ == "__main__":
    main()
