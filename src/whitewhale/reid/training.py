"""
度量学习训练（伪标签 ArcFace + 可选跨群 hard negative）。

合并自 scripts/train_metric_learning.py（r1/r2，纯 ArcFace CE）与
scripts/train_metric_learning_hn.py（r3，CE + λ×跨群 triplet，batch 内
动态挖掘 + 两群对半采样）。二者共享模型/数据/两阶段训练骨架，差异全部
由参数控制：

    hard_negative=True  → 跨群 triplet 辅助损失（r3 链路，当前正式特征源）
    hard_negative=False → 纯 ArcFace CE（r1/r2 历史链路）

语义约束：伪标签来自人工初审（Candidate 级，非专家复核）；不使用水平翻转
（左右侧背鳍特征可能不同）；划分按个体（同一体不跨 train/val）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from whitewhale.data.image_store import ImageStore
from whitewhale.reid.embedding import (DEVICE, FEAT_DIM, INPUT_SIZE, ReIDModel,
                                       make_backbone, make_embedder)


def load_confirmed(pilot_csv: Path, images_root: Path) -> pd.DataFrame:
    """读取人工初审确认的照片（confirmed_identity 非空），映射个体 → 类别号。"""
    p = pd.read_csv(pilot_csv)
    df = p[p["confirmed_identity"].notna()
           & (p["confirmed_identity"].astype(str).str.strip() != "")].copy()
    store = ImageStore(images_root)
    df["path"] = df["relative_path"].map(lambda rp: str(store.resolve(rp)))
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
        # 模拟拍摄变化；刻意不用水平翻转（左右侧特征不同）
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
    """纯 ArcFace CE 单轮（r1/r2 链路）。"""
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


class DolphinDatasetHn(DolphinDataset):
    """训练集 Dataset（HN 链路）：携带 session（跨群负样本判定）+ 预缩放内存缓存。

    原图 3000-4000px，每次 __getitem__ 现读会卡 CPU（GPU 空转）。
    preload() 把全部训练图一次性缩放到 256px 放进内存（约 30MB），
    之后每 epoch 只做增强变换，IO 归零。
    """

    def __init__(self, df, transform, cache_size: int = 256):
        super().__init__(df, transform)
        self.sessions = df["sess"].tolist()
        self._cache = {}
        self.cache_size = cache_size

    def preload(self):
        for i, path in enumerate(self.paths):
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.cache_size, self.cache_size), Image.LANCZOS)
            self._cache[i] = img
        print(f"[ds] 预加载 {len(self._cache)} 张训练图到内存缓存（~{self.cache_size}px）")

    def __getitem__(self, i):
        img = self._cache.get(i)
        if img is None:
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
    """HN 链路单轮：CE + λ×跨群 triplet。"""
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


def run_training(args, df: pd.DataFrame, out_dir: Path):
    """两阶段训练主流程（冻结 backbone 训 head → 解冻低学习率微调）。

    args 字段：val_n / seed / epochs_stage1 / epochs_stage2 / lr_head /
    lr_backbone / batch；hard_negative=True 时另需 init_ckpt /
    batches_per_epoch / lambda_hn。
    """
    hard_negative = bool(getattr(args, "hard_negative", False))
    train_df, val_df = split_by_individual(df, args.val_n, args.seed)
    # 重新映射训练个体编号为 0..n-1（load_confirmed 的编号是全局的，
    # 留出验证个体后会出现空洞，导致 one_hot / CUDA 索引越界）
    train_df = train_df.copy()
    id2idx = {iid: i for i, iid in enumerate(sorted(train_df["confirmed_identity"].unique()))}
    train_df["label_idx"] = train_df["confirmed_identity"].map(id2idx)
    print(f"[train] 训练 {len(train_df)} 张 / {train_df['confirmed_identity'].nunique()} 个体；"
          f"验证 {len(val_df)} 张 / {val_df['confirmed_identity'].nunique()} 个体")

    model = ReIDModel(make_backbone(), n_classes=train_df["label_idx"].nunique()).to(DEVICE)
    if hard_negative:
        train_df["sess"] = train_df["session_id"].astype(int)
        tr_ds = DolphinDatasetHn(train_df, make_transforms(True))
        tr_ds.preload()  # 原图大，预缩放到内存缓存，避免 CPU 解码成为瓶颈
        sampler = BalancedGroupSampler(tr_ds.sessions, args.batch,
                                       args.batches_per_epoch, args.seed)
        tr_loader = DataLoader(tr_ds, batch_sampler=sampler)
        if getattr(args, "init_ckpt", None) and Path(args.init_ckpt).exists():
            ckpt = torch.load(args.init_ckpt, map_location=DEVICE)
            missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
            print(f"[train] 初始化自 {args.init_ckpt}（missing {len(missing)} 项 = head 层；"
                  f"unexpected {len(unexpected)} 项）")
    else:
        tr_loader = DataLoader(DolphinDataset(train_df, make_transforms(True)),
                               batch_size=args.batch, shuffle=True,
                               num_workers=0, drop_last=False)
    val_loader = DataLoader(DolphinDataset(val_df, make_transforms(False)),
                            batch_size=32, shuffle=False)

    # 预训练基线：训练前用同一个 backbone 评估（避免二次加载触发网络）
    pretrained_r1 = eval_retrieval(model, val_loader)
    print(f"[train] 预训练基线（验证个体）val R@1 = {pretrained_r1:.3f}")
    history = []
    best_r1, best_epoch = 0.0, 0

    # 阶段一：冻结 backbone 训 head
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.head.parameters() if p.requires_grad],
                            lr=args.lr_head, weight_decay=5e-4)
    print(f"[s1] 冻结 backbone 训 head（{args.epochs_stage1} epochs, lr={args.lr_head}）")
    for ep in range(1, args.epochs_stage1 + 1):
        if hard_negative:
            ce, hn = train_one_epoch_hn(model, tr_loader, opt, 0.0)  # stage1 无 triplet
        else:
            ce, _ = train_one_epoch(model, tr_loader, opt,
                                    train_df["label_idx"].nunique())
            hn = 0.0
        r1 = eval_retrieval(model, val_loader)
        history.append({"stage": 1, "epoch": ep, "loss": ce, "hn": hn, "val_r1": r1})
        if r1 > best_r1:
            best_r1, best_epoch = r1, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "val_r1": r1},
                       out_dir / "best_stage1.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"  [s1] ep {ep:2d}  ce {ce:.4f}  hn {hn:.4f}  val R@1 {r1:.3f}")

    # 阶段二：解冻 backbone 低学习率微调（HN 链路加 λ×跨群 triplet）
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=5e-4)
    lambda_hn = getattr(args, "lambda_hn", 0.2) if hard_negative else 0.0
    print(f"[s2] 解冻微调（{args.epochs_stage2} epochs, lr={args.lr_backbone}），"
          f"loss = CE + {lambda_hn} × cross-group triplet")
    for ep in range(1, args.epochs_stage2 + 1):
        if hard_negative:
            ce, hn = train_one_epoch_hn(model, tr_loader, opt, lambda_hn)
        else:
            ce, _ = train_one_epoch(model, tr_loader, opt,
                                    train_df["label_idx"].nunique())
            hn = 0.0
        r1 = eval_retrieval(model, val_loader)
        history.append({"stage": 2, "epoch": ep, "loss": ce, "hn": hn, "val_r1": r1})
        if r1 > best_r1:
            best_r1, best_epoch = r1, ep
            torch.save({"state": model.state_dict(), "epoch": ep, "val_r1": r1,
                        "stage": 2},
                       out_dir / "best.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"  [s2] ep {ep:2d}  ce {ce:.4f}  hn {hn:.4f}  val R@1 {r1:.3f}")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    note = ("r3 跨群 hard negative 微调：batch 内动态挖掘 + 两群对半采样" if hard_negative
            else "伪标签 ArcFace 微调（人工初审标签，Candidate 级）")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": f"{note}；伪标签来自人工初审（非专家复核），指标只代表验证个体上的检索一致性",
            "hard_negative": hard_negative,
            "n_train": len(train_df), "n_val": len(val_df),
            "n_train_individuals": int(train_df["confirmed_identity"].nunique()),
            "n_val_individuals": int(val_df["confirmed_identity"].nunique()),
            "val_n": args.val_n, "seed": args.seed,
            "lambda_hn": lambda_hn, "batch": args.batch,
            "init_ckpt": str(getattr(args, "init_ckpt", "")),
            "pretrained_baseline_r1": pretrained_r1,
            "best_val_r1": best_r1, "best_epoch": best_epoch,
        }, f, indent=2, ensure_ascii=False)
    print(f"[train] 完成。预训练基线 val R@1 = {pretrained_r1:.3f} → "
          f"微调后 {best_r1:.3f}（epoch {best_epoch}）→ {out_dir}")


def extract_features(args, out_dir: Path, pilot_csv: Path, images_root: Path):
    """用 best.pt 重新提取全部 pilot 特征（供 benchmark / 开放集预演复跑）。"""
    from whitewhale.reid.embedding import extract_embeddings

    p = pd.read_csv(pilot_csv)
    model = make_embedder("metric-learning", metric_ckpt=out_dir / "best.pt")
    out_emb = args.embeddings_out
    extract_embeddings(
        p, model, images_root=images_root, out_path=out_emb,
        merge_from=None,
        model_cfg={"model": model.name,
                   "note": "伪标签 ArcFace 微调后特征（人工初审标签，Candidate 级）",
                   "feat_dim": FEAT_DIM, "source": str(out_dir / "best.pt")},
    )
    print(f"[extract] {len(p)} 张特征 → {out_emb}（模型 {out_dir / 'best.pt'}）")
