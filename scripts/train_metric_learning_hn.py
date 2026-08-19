"""
跨群 hard negative 微调（r3）：ArcFace + 跨群 triplet 辅助损失。

背景（E3 预演结论）：r2 微调特征已把跨群高分对压得极低（静态挖掘 ≥0.5 仅 13 对），
E3 的真正瓶颈是"同体对分数不够高"（known recall 36-47%，已知个体约一半被误标为
疑似新个体）。因此本实验主机制为 **batch 内动态 hard negative mining**：

- Batch 采样：每 batch 强制 01/03 两群各半（01 群仅 19 张，带替换重采样）——
  保证每个 anchor 的 batch 内必然存在跨群负样本，且 01 群样本不被淹没；
- Triplet 辅助损失（stage2）：anchor 的 positive = 同个体内最高 cos；
  negative = batch 内跨群最高 cos（动态 hardest cross-group negative）。
  跨群对默认不同个体（README §5.2 语义确认），负样本可靠；
- 总损失 = ArcFace CE + λ × 跨群 triplet（λ 默认 0.2）。

初始化：r2/best.pt 的 backbone 权重（head 因类别数变化重新初始化）。
评估：验证个体（模型未见，按个体留出）leave-one-out R@1 —— 防过拟合信号。

语义约束（CLAUDE.md）：不使用水平翻转；验证按个体留出；伪标签为人工初审（Candidate 级）。

用法：
    python scripts/train_metric_learning_hn.py --out outputs/metric_learning/r3
    python scripts/train_metric_learning_hn.py --extract --out outputs/metric_learning/r3
        --embeddings-out outputs/embeddings/embeddings_metric_r3.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_metric_learning import (  # noqa: E402
    DEVICE, FEAT_DIM, ReIDModel, make_backbone, load_confirmed,
    split_by_individual, DolphinDataset, make_transforms, eval_retrieval,
)


class DolphinDatasetHn(DolphinDataset):
    """训练集 Dataset：携带 session（跨群负样本判定用）+ 预缩放内存缓存。

    原图 3000-4000px，每次 __getitem__ 现读会卡 CPU（GPU 空转）。
    preload() 把全部训练图一次性缩放到 256px 放进内存（133 张约 30MB），
    之后每 epoch 只做增强变换，IO 归零。
    """

    def __init__(self, df, transform, cache_size: int = 256):
        super().__init__(df, transform)
        self.sessions = df["sess"].tolist()
        self._cache = {}
        self.cache_size = cache_size

    def preload(self):
        from PIL import Image

        for i, path in enumerate(self.paths):
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.cache_size, self.cache_size), Image.LANCZOS)
            self._cache[i] = img
        print(f"[ds] 预加载 {len(self._cache)} 张训练图到内存缓存（~{self.cache_size}px）")

    def __getitem__(self, i):
        img = self._cache.get(i)
        if img is None:
            from PIL import Image

            img = Image.open(self.paths[i]).convert("RGB")  # 兜底：未 preload 时现读
        return self.transform(img), self.labels[i], self.sessions[i]


class BalancedGroupSampler:
    """每 batch 强制 01/03 各半（带替换重采样），保证跨群负样本恒定在场。"""

    def __init__(self, sessions, batch_size: int, batches_per_epoch: int, seed: int):
        self.idx = {1: [i for i, s in enumerate(sessions) if s == 1],
                    3: [i for i, s in enumerate(sessions) if s == 3]}
        assert self.idx[1] and self.idx[3], "两群都必须有训练样本"
        self.half = max(1, batch_size // 2)
        self.n_batches = batches_per_epoch
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            parts = [self.rng.choice(self.idx[s], self.half, replace=True) for s in (1, 3)]
            batch = np.concatenate(parts)
            self.rng.shuffle(batch)
            yield torch.from_numpy(batch)

    def __len__(self):
        return self.n_batches


def triplet_hn_loss(feats: torch.Tensor, labels: torch.Tensor, sessions: torch.Tensor,
                    margin: float = 0.3) -> torch.Tensor:
    """跨群 hard negative triplet：pos = 同个体最高 cos，neg = 跨群最高 cos（动态挖掘）。"""
    cos = feats @ feats.T
    d = 1 - cos
    n = feats.size(0)
    ar = torch.arange(n, device=feats.device)
    losses = []
    for i in range(n):
        same = (labels == labels[i]) & (ar != i)
        if same.sum() == 0:
            continue  # 无同个体样本（单图个体），跳过
        pos_d = d[i][same].min()
        cross = sessions != sessions[i]  # 跨群对均为可靠负样本（即使编号相同）
        neg_d = d[i][cross].max()
        losses.append(F.relu(pos_d - neg_d + margin))
    return torch.stack(losses).mean() if losses else cos.new_zeros(())


def train_one_epoch_hn(model, loader, opt, lambda_hn: float):
    model.train()
    ce_total = torch.zeros((), device=DEVICE)
    hn_total = torch.zeros((), device=DEVICE)
    n_batch = 0
    for x, y, s in loader:
        x, y, s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
        opt.zero_grad()
        logits = model(x, y)
        ce = F.cross_entropy(logits, y)
        feats = model.encode(x)
        hn = triplet_hn_loss(feats, y, s, margin=0.3)
        loss = ce + lambda_hn * hn
        loss.backward()
        opt.step()
        ce_total += ce.detach()
        hn_total += hn.detach()
        n_batch += 1
    return (ce_total / n_batch).item(), (hn_total / n_batch).item()


def run(args, df: pd.DataFrame, out_dir: Path):
    train_df, val_df = split_by_individual(df, args.val_n, args.seed)
    train_df = train_df.copy()
    id2idx = {iid: i for i, iid in enumerate(sorted(train_df["confirmed_identity"].unique()))}
    train_df["label_idx"] = train_df["confirmed_identity"].map(id2idx)
    train_df["sess"] = train_df["session_id"].astype(int)
    print(f"[train] 训练 {len(train_df)} 张 / {train_df['confirmed_identity'].nunique()} 个体 "
          f"（01:{int((train_df['sess']==1).sum())} 03:{int((train_df['sess']==3).sum())}）；"
          f"验证 {len(val_df)} 张 / {val_df['confirmed_identity'].nunique()} 个体")

    tr_ds = DolphinDatasetHn(train_df, make_transforms(True))
    tr_ds.preload()  # 原图大，预缩放到内存缓存，避免 CPU 解码成为瓶颈
    sampler = BalancedGroupSampler(tr_ds.sessions, args.batch, args.batches_per_epoch, args.seed)
    tr_loader = DataLoader(tr_ds, batch_sampler=sampler)
    val_loader = DataLoader(DolphinDataset(val_df, make_transforms(False)), batch_size=32, shuffle=False)

    model = ReIDModel(make_backbone(), n_classes=train_df["label_idx"].nunique()).to(DEVICE)
    if args.init_ckpt and args.init_ckpt.exists():
        ckpt = torch.load(args.init_ckpt, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
        print(f"[train] 初始化自 {args.init_ckpt}（missing {len(missing)} 项 = head 层；"
              f"unexpected {len(unexpected)} 项）")
    pretrained_r1 = eval_retrieval(model, val_loader)
    print(f"[train] 预训练基线（验证个体）val R@1 = {pretrained_r1:.3f}")

    history, best_r1, best_epoch = [], 0.0, 0
    # 阶段一：冻结 backbone 训 head（与 r1/r2 流程一致）
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.head.parameters() if p.requires_grad],
                            lr=args.lr_head, weight_decay=5e-4)
    print(f"[s1] 冻结 backbone 训 head（{args.epochs_stage1} epochs, lr={args.lr_head}）")
    for ep in range(1, args.epochs_stage1 + 1):
        ce, hn = train_one_epoch_hn(model, tr_loader, opt, 0.0)  # stage1 无 triplet（特征不动）
        r1 = eval_retrieval(model, val_loader)
        history.append({"stage": 1, "epoch": ep, "loss": ce, "hn": hn, "val_r1": r1})
        if r1 > best_r1:
            best_r1, best_epoch = r1, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "val_r1": r1}, out_dir / "best_stage1.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"  [s1] ep {ep:2d} ce {ce:.4f}  val R@1 {r1:.3f}")

    # 阶段二：解冻 backbone，CE + λ×跨群 triplet
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=5e-4)
    print(f"[s2] 解冻微调（{args.epochs_stage2} epochs, lr={args.lr_backbone}），"
          f"loss = CE + {args.lambda_hn} × cross-group triplet")
    for ep in range(1, args.epochs_stage2 + 1):
        ce, hn = train_one_epoch_hn(model, tr_loader, opt, args.lambda_hn)
        r1 = eval_retrieval(model, val_loader)
        history.append({"stage": 2, "epoch": ep, "loss": ce, "hn": hn, "val_r1": r1})
        if r1 > best_r1:
            best_r1, best_epoch = r1, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "val_r1": r1, "stage": 2},
                       out_dir / "best.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"  [s2] ep {ep:2d} ce {ce:.4f} hn {hn:.4f}  val R@1 {r1:.3f}")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "r3 跨群 hard negative 微调：batch 内动态挖掘 + 两群对半采样；伪标签为人工初审（Candidate 级）",
            "n_train": len(train_df), "n_val": len(val_df),
            "n_train_individuals": int(train_df["confirmed_identity"].nunique()),
            "n_val_individuals": int(val_df["confirmed_identity"].nunique()),
            "lambda_hn": args.lambda_hn, "batch": args.batch,
            "init_ckpt": str(args.init_ckpt),
            "pretrained_baseline_r1": pretrained_r1,
            "best_val_r1": best_r1, "best_epoch": best_epoch,
        }, f, indent=2, ensure_ascii=False)
    print(f"[train] 完成。初始（r2 权重）val R@1 = {pretrained_r1:.3f} → {best_r1:.3f}（epoch {best_epoch}）")


def extract_features(args, out_dir: Path, pilot_csv: Path, images_root: Path):
    """用 best.pt 重新提取全部 pilot 特征（供 benchmark / E3 预演复跑）。"""
    import torchvision.transforms as T

    from scripts.train_metric_learning import extract_features as _extract

    _extract(args, out_dir, pilot_csv, images_root)


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="跨群 hard negative 微调（r3）")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"))
    parser.add_argument("--out", type=Path, default=base / "metric_learning" / "r3")
    parser.add_argument("--init-ckpt", type=Path, default=base / "metric_learning" / "r2" / "best.pt")
    parser.add_argument("--val-n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-stage1", type=int, default=20)
    parser.add_argument("--epochs-stage2", type=int, default=25)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--batch", type=int, default=16, help="每 batch 两群各半")
    parser.add_argument("--batches-per-epoch", type=int, default=40)
    parser.add_argument("--lambda-hn", type=float, default=0.2, help="跨群 triplet 权重")
    parser.add_argument("--extract", action="store_true", help="用 best.pt 重新提取特征")
    parser.add_argument("--embeddings-out", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r3.npy")
    args = parser.parse_args()

    if args.extract:
        extract_features(args, args.out, args.pilot, args.images_root)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_confirmed(args.pilot, args.images_root)
    df["sess"] = df["session_id"].astype(int)
    if df["confirmed_identity"].nunique() < args.val_n + 2:
        raise SystemExit("个体数过少，无法留出验证个体")
    run(args, df, args.out)


if __name__ == "__main__":
    main()
