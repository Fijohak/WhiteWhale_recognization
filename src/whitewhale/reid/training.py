"""
度量学习训练（确认个体标签 ArcFace + 可选批内 hard negative）。

合并自 scripts/train_metric_learning.py（r1/r2，纯 ArcFace CE）与
scripts/train_metric_learning_hn.py（历史 r3，CE + λ×triplet）。当前实现只在
同一 session 的已确认不同个体间挖掘负样本，并屏蔽同串共现身份，
避免把跨批次未对齐身份或同串关系误作负例。
二者共享模型/数据/两阶段训练骨架，差异全部
由参数控制：

    hard_negative=True  → 批内 batch-hard triplet 辅助损失
    hard_negative=False → 纯 ArcFace CE（r1/r2 历史链路）

语义约束：confirmed_identity 只接收已确认个体标签；批次内 individual_id
虽已确认，但现有多图样本时间间隔短，训练结果不等同于跨年识别能力。
不使用水平翻转（左右侧背鳍特征可能不同）；划分按个体（同一体不跨 train/val）。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from whitewhale.data.image_store import ImageStore
from whitewhale.data.sequence_groups import series_units
from whitewhale.reid.embedding import (DEVICE, FEAT_DIM, INPUT_SIZE, ReIDModel,
                                       make_backbone, make_embedder)


def _identity_units(df: pd.DataFrame) -> pd.Series:
    """生成批次命名空间身份，禁止把跨 session 同名编号自动合并。"""
    identities = df["confirmed_identity"].astype(str).str.strip()
    if "session_id" not in df.columns:
        return identities
    sessions = df["session_id"].astype(str).str.strip()
    if (sessions == "").any():
        raise ValueError("确认身份清单存在空 session_id，无法建立批次内身份命名空间")
    return sessions + "\x1f" + identities


def load_confirmed(pilot_csv: Path, images_root: Path) -> pd.DataFrame:
    """读取已确认照片并补完整连拍串，映射个体 → 类别号。"""
    p = pd.read_csv(pilot_csv, dtype=str, keep_default_na=False)
    if "confirmed_identity" not in p.columns:
        raise ValueError("pilot_set.csv 缺少 confirmed_identity")
    df = p[p["confirmed_identity"].astype(str).str.strip() != ""].copy()
    store = ImageStore(images_root)
    df["path"] = df["relative_path"].map(lambda rp: str(store.resolve(rp)))
    required_series = {"series_id", "sequence_key", "frame"}
    missing_series = sorted(required_series - set(df.columns))
    if missing_series:
        raise ValueError(
            "pilot 缺少由完整 dataset manifest 生成的稳定连拍串字段 "
            f"{missing_series}；请先重新运行 prepare_data.py build-pilot，"
            "禁止在训练子集上重新分串")
    df["series_unit"] = series_units(df)
    df["identity_unit"] = _identity_units(df)
    id2idx = {iid: i for i, iid in enumerate(sorted(df["identity_unit"].unique()))}
    df["label_idx"] = df["identity_unit"].map(id2idx)
    return df


def _seed_training(seed: int) -> None:
    """固定训练使用的 Python、NumPy 与 PyTorch 随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _isolate_test_sessions(
        frame: pd.DataFrame, test_sessions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """先完整隔离指定 session；测试数据不进入 train/val 候选。"""
    if not test_sessions:
        return frame.copy(), frame.iloc[0:0].copy()
    sessions = frame["session_id"].astype(str).str.strip()
    missing = sorted(set(test_sessions) - set(sessions))
    if missing:
        raise ValueError(f"指定 test session 在确认清单中不存在：{missing}")
    test_mask = sessions.isin(test_sessions)
    remaining = frame.loc[~test_mask].copy()
    test = frame.loc[test_mask].copy()
    if remaining.empty:
        raise ValueError("隔离 test session 后没有可用于 train/val 的样本")
    return remaining, test


def split_by_individual(df: pd.DataFrame, val_n: int, seed: int):
    """按个体留出验证；验证个体必须至少有两个完整 series。"""
    if "series_unit" not in df.columns:
        raise ValueError("训练清单缺少 series_unit，请通过 load_confirmed 读取")
    identity_units = (df["identity_unit"].astype(str)
                      if "identity_unit" in df.columns else _identity_units(df))
    series_counts = df.assign(_identity_unit=identity_units).groupby(
        "_identity_unit")["series_unit"].nunique()
    candidates = series_counts[series_counts >= 2].index.tolist()
    if len(candidates) < val_n:
        raise ValueError(
            f"只有 {len(candidates)} 个身份具备至少两个 series，无法留出 {val_n} 个验证身份")
    def stable_identity_key(identity: object) -> bytes:
        group = df[identity_units == identity]
        if "image_id" in group.columns:
            members = sorted(group["image_id"].astype(str).tolist())
        else:
            members = sorted(str(index) for index in group.index)
        return hashlib.sha256(
            (str(seed) + "\x1f" + "\x1f".join(members)).encode("utf-8")
        ).digest()

    candidates.sort(key=stable_identity_key)
    val_ids = set(candidates[:val_n])
    train = df[~identity_units.isin(val_ids)].copy()
    val = df[identity_units.isin(val_ids)].copy()
    # 一个真实连拍串可能被错误地分到多个身份目录。身份留出后，把与 val
    # 共享 series 的 train 行整体清除，避免背景/时刻泄漏进模型训练。
    val_series = set(val["series_unit"].astype(str).str.strip()) - {""}
    shared_mask = train["series_unit"].astype(str).str.strip().isin(val_series)
    purged_shared_series_rows = int(shared_mask.sum())
    train = train[~shared_mask].copy()
    train.attrs["purged_shared_series_rows"] = purged_shared_series_rows
    train_series = set(train["series_unit"].astype(str).str.strip()) - {""}
    if train_series & val_series:
        raise AssertionError("训练/验证仍共享完整 series")
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


class DolphinDatasetSession(DolphinDataset):
    """ArcFace 训练数据：携带 session/series 以屏蔽非法负类。"""

    def __init__(self, df: pd.DataFrame, transform):
        super().__init__(df, transform)
        self.sessions = df["sess"].tolist()
        self.series = df["series_idx"].tolist()

    def __getitem__(self, i):
        image, label = super().__getitem__(i)
        return image, label, self.sessions[i], self.series[i]


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
def eval_retrieval(model: ReIDModel, loader: DataLoader,
                   series_ids: list[str] | None = None,
                   session_ids: list[str] | None = None,
                   return_details: bool = False) -> float | tuple[float, int, int]:
    """批次内跨串 leave-one-out R@1；跨批次未对齐样本不作为负例。"""
    model.eval()
    feats, labs = [], []
    for x, y in loader:
        f = model.encode(x.to(DEVICE))
        feats.append(f.cpu()); labs.append(y)
    feats = torch.cat(feats).numpy()          # (N, D) 已 L2 归一化
    labs = torch.cat(labs).numpy()
    return retrieval_r1_from_features(
        feats, labs, series_ids=series_ids, session_ids=session_ids,
        return_details=return_details)


def retrieval_r1_from_features(
    feats: np.ndarray,
    labels: np.ndarray,
    *,
    series_ids: list[str] | None = None,
    session_ids: list[str] | None = None,
    return_details: bool = False,
) -> float | tuple[float, int, int]:
    """纯数组实现的批次内跨串 R@1，供训练验证和回归测试共用。"""
    if series_ids is not None and len(series_ids) != len(labels):
        raise ValueError("series_ids 数量与验证特征不一致")
    if session_ids is not None and len(session_ids) != len(labels):
        raise ValueError("session_ids 数量与验证特征不一致")
    sim = feats @ feats.T
    np.fill_diagonal(sim, -np.inf)
    hits = []
    skipped = 0
    for i in range(len(labels)):
        row = sim[i].copy()
        if session_ids is not None:
            same_session = np.asarray([
                str(value).strip() == str(session_ids[i]).strip()
                for value in session_ids
            ])
            row[~same_session] = -np.inf
        if series_ids is not None and str(series_ids[i]).strip():
            same_series = np.asarray([
                str(value).strip() == str(series_ids[i]).strip()
                for value in series_ids
            ])
            row[same_series] = -np.inf
        valid_positive = (labels == labels[i]) & np.isfinite(row)
        valid_negative = (labels != labels[i]) & np.isfinite(row)
        if not valid_positive.any() or not valid_negative.any():
            skipped += 1
            continue
        hits.append(labels[int(np.argmax(row))] == labels[i])
    if not hits:
        raise ValueError("验证集中没有同时具备跨串同体正样本和批次内异体负样本的 query")
    score = float(np.mean(hits))
    if return_details:
        return score, len(hits), skipped
    return score


def train_one_epoch(model, loader, opt, n_classes,
                    class_sessions: torch.Tensor | None = None,
                    class_series_membership: torch.Tensor | None = None):
    """ArcFace CE 单轮；屏蔽跨 session 类别和同串共现身份。"""
    model.train()
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        # 仅 requires_grad=False 不会冻结 BatchNorm 统计量；阶段一必须同时保持
        # backbone 为 eval，才能保证训练前后检索特征与验证指标不变。
        model.backbone.eval()
    total, correct = 0.0, 0.0
    for batch in loader:
        if len(batch) == 4:
            x, y, sessions, series = batch
            sessions, series = sessions.to(DEVICE), series.to(DEVICE)
        elif len(batch) == 3:
            x, y, sessions = batch
            sessions, series = sessions.to(DEVICE), None
        else:
            x, y = batch
            sessions, series = None, None
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x, y)
        if class_sessions is not None:
            if sessions is None or series is None or class_series_membership is None:
                raise ValueError("series-aware CE 缺少样本 series 或类别-series 映射")
            loss = session_aware_cross_entropy(
                logits, y, sessions, class_sessions, series,
                class_series_membership)
        else:
            loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total += len(y)
        correct += (logits.argmax(1) == y).sum().item()
    return float(loss.item()), correct / total


class DolphinDatasetHn(DolphinDataset):
    """训练集 Dataset（HN 链路）：携带 session/series + 预缩放缓存。

    原图 3000-4000px，每次 __getitem__ 现读会卡 CPU（GPU 空转）。
    preload() 保持宽高比，把全部训练图的短边缩放到 cache_size 后放进内存；
    默认短边 256px，可避免后续 224px 随机裁剪从低分辨率缓存上采样。缓存
    占用随照片宽高比变化，预加载时会输出 RGB 像素内存估算。
    """

    def __init__(self, df, transform, cache_size: int = 256):
        super().__init__(df, transform)
        self.sessions = df["sess"].tolist()
        self.series = df["series_idx"].tolist()
        self._cache = {}
        self.cache_size = cache_size

    def preload(self):
        rgb_bytes = 0
        for i, path in enumerate(self.paths):
            with Image.open(path) as source:
                img = source.convert("RGB")
            width, height = img.size
            if width <= height:
                resized = (self.cache_size,
                           max(self.cache_size,
                               round(height * self.cache_size / width)))
            else:
                resized = (max(self.cache_size,
                               round(width * self.cache_size / height)),
                           self.cache_size)
            img = img.resize(resized, Image.Resampling.LANCZOS)
            self._cache[i] = img
            rgb_bytes += resized[0] * resized[1] * 3
        print(f"[ds] 预加载 {len(self._cache)} 张训练图到内存缓存"
              f"（短边 {self.cache_size}px，RGB 约 {rgb_bytes / 1024 ** 2:.1f} MiB）")

    def __getitem__(self, i):
        img = self._cache.get(i)
        if img is None:
            img = Image.open(self.paths[i]).convert("RGB")  # 兜底：未 preload 时现读
        return self.transform(img), self.labels[i], self.sessions[i], self.series[i]


class WithinSessionIdentitySampler:
    """每个 batch 取同一 session 的至少两个身份。

    这样正样本与负样本都来自批次内已确认关系；跨 session 身份未对齐，不作为
    confirmed-different。每批优先让至少 50% 的入选身份具备跨 series
    正对；若当前 session 的多串身份不足，则将它们全部纳入。输出前校验
    至少一个合法 triplet anchor。
    单串身份仍可参加 ArcFace CE，但同帧增强或同串照片不会被当作 HN 正样本。
    """

    def __init__(self, labels, sessions, series, batch_size: int,
                 batches_per_epoch: int, seed: int):
        if batch_size < 4:
            raise ValueError("hard-negative batch 至少为 4，才能包含两个身份的正负样本")
        if not (len(labels) == len(sessions) == len(series)):
            raise ValueError("labels/sessions/series 数量不一致")
        grouped: dict[object, dict[object, dict[object, list[int]]]] = {}
        series_label_membership: dict[object, set[object]] = {}
        for index, (label, session, series_id) in enumerate(
                zip(labels, sessions, series)):
            grouped.setdefault(session, {}).setdefault(label, {}).setdefault(
                series_id, []).append(index)
            series_label_membership.setdefault(series_id, set()).add(label)
        self.grouped = {}
        self.cross_series_identities = {}
        self.valid_identity_pairs = {}
        for session, identities in grouped.items():
            eligible = [
                identity for identity, by_series in identities.items()
                if len(by_series) >= 2
            ]
            valid_pairs = []
            for anchor_identity in eligible:
                for anchor_series in identities[anchor_identity]:
                    if not any(other != anchor_series
                               for other in identities[anchor_identity]):
                        continue
                    for negative_identity in identities:
                        if (negative_identity != anchor_identity
                                and negative_identity not in
                                series_label_membership[anchor_series]):
                            valid_pairs.append(
                                (anchor_identity, negative_identity, anchor_series))
            if len(identities) >= 2 and valid_pairs:
                self.grouped[session] = identities
                self.cross_series_identities[session] = eligible
                self.valid_identity_pairs[session] = valid_pairs
        if not self.grouped:
            raise ValueError(
                "没有同时具备跨串正样本和剔除同串共现身份后确认异体的 session，"
                "无法做批内 hard negative")
        self.sessions = list(self.grouped)
        self.labels = np.asarray(labels)
        self.sample_sessions = np.asarray(sessions)
        self.sample_series = np.asarray(series)
        self.series_label_membership = series_label_membership
        self.batch_size = batch_size
        self.n_batches = batches_per_epoch
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            session = self.rng.choice(self.sessions)
            identities = list(self.grouped[session])
            n_identities = min(len(identities), max(2, self.batch_size // 2))
            anchor_identity, negative_identity, forced_anchor_series = (
                self.valid_identity_pairs[session][
                    int(self.rng.integers(len(self.valid_identity_pairs[session])))]
            )
            eligible = self.cross_series_identities[session]
            n_cross_series = min(
                len(eligible), max(1, (n_identities + 1) // 2))
            selected = [anchor_identity, negative_identity]
            extra_eligible = [identity for identity in eligible
                              if identity not in selected]
            n_extra_eligible = min(
                len(extra_eligible),
                max(0, n_cross_series
                    - sum(identity in eligible for identity in selected)))
            if n_extra_eligible:
                selected.extend(self.rng.choice(
                    extra_eligible, n_extra_eligible, replace=False).tolist())
            remaining = [identity for identity in identities if identity not in selected]
            if len(selected) < n_identities:
                selected.extend(self.rng.choice(
                    remaining, n_identities - len(selected),
                    replace=False).tolist())
            base, remainder = divmod(self.batch_size, n_identities)
            parts = []
            for position, identity in enumerate(selected):
                count = base + (1 if position < remainder else 0)
                by_series = self.grouped[session][identity]
                series_ids = list(by_series)
                chosen: list[int] = []
                if count >= 2 and len(series_ids) >= 2:
                    # 先从两个不同完整串各取一张，确保该身份可产生合法 HN 正对。
                    if identity == anchor_identity:
                        positive_series = self.rng.choice([
                            value for value in series_ids
                            if value != forced_anchor_series
                        ])
                        first_series = [forced_anchor_series, positive_series]
                    else:
                        first_series = self.rng.choice(series_ids, 2, replace=False)
                    chosen.extend(int(self.rng.choice(by_series[value]))
                                  for value in first_series)
                all_indices = [index for values in by_series.values() for index in values]
                if len(chosen) < count:
                    chosen.extend(self.rng.choice(
                        all_indices, count - len(chosen), replace=True).astype(int).tolist())
                parts.append(np.asarray(chosen, dtype=np.int64))
            batch = np.concatenate(parts).astype(np.int64, copy=False)
            self.rng.shuffle(batch)
            if _count_valid_triplet_anchors(
                    self.labels[batch], self.sample_sessions[batch],
                    self.sample_series[batch],
                    self.series_label_membership) == 0:
                raise RuntimeError("sampler 生成了零合法 anchor 的 HN batch")
            yield torch.from_numpy(batch)

    def __len__(self):
        return self.n_batches


def _count_valid_triplet_anchors(
        labels, sessions, series,
        series_label_membership: dict[object, set[object]]) -> int:
    """按全量串→身份关系统计同时具有合法正负样本的 anchor。"""
    labels = np.asarray(labels)
    sessions = np.asarray(sessions)
    series = np.asarray(series)
    if not (len(labels) == len(sessions) == len(series)):
        raise ValueError("labels/sessions/series 数量不一致")
    same_session = sessions[:, None] == sessions[None, :]
    cross_series = series[:, None] != series[None, :]
    same_label = labels[:, None] == labels[None, :]
    not_self = ~np.eye(len(labels), dtype=bool)
    has_positive = (same_session & cross_series & same_label & not_self).any(axis=1)
    forbidden = np.asarray([
        [candidate_label in series_label_membership[anchor_series]
         for candidate_label in labels]
        for anchor_series in series
    ], dtype=bool)
    has_negative = (same_session & ~same_label & ~forbidden).any(axis=1)
    return int((has_positive & has_negative).sum())


def triplet_hn_loss(feats: torch.Tensor, labels: torch.Tensor, sessions: torch.Tensor,
                    series: torch.Tensor,
                    class_series_membership: torch.Tensor,
                    margin: float = 0.3,
                    return_details: bool = False
                    ) -> torch.Tensor | tuple[torch.Tensor, int]:
    """批内 HN；anchor 所属串中出现过的其他身份整体不得作为负类。"""
    membership = class_series_membership.to(device=labels.device, dtype=torch.bool)
    cos = feats @ feats.T
    d = 1 - cos
    n = feats.size(0)
    ar = torch.arange(n, device=feats.device)
    losses = []
    for i in range(n):
        same = ((labels == labels[i]) & (sessions == sessions[i])
                & (series != series[i]) & (ar != i))
        if same.sum() == 0:
            continue  # 无同个体样本（单图个体），跳过
        # 若某身份曾在 anchor 所属完整串中出现，则该身份的所有照片都被排除；
        # 不能借用它在另一串中的照片绕过同串共现约束。
        forbidden_labels = membership[series[i], labels]
        negative = ((labels != labels[i]) & (sessions == sessions[i])
                    & ~forbidden_labels)
        if negative.sum() == 0:
            continue
        pos_d = d[i][same].max()
        neg_d = d[i][negative].min()
        losses.append(F.relu(pos_d - neg_d + margin))
    loss = torch.stack(losses).mean() if losses else cos.new_zeros(())
    if return_details:
        return loss, len(losses)
    return loss


def session_aware_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                                sessions: torch.Tensor,
                                class_sessions: torch.Tensor,
                                series: torch.Tensor,
                                class_series_membership: torch.Tensor,
                                ) -> torch.Tensor:
    """按 session 计算 CE；同串共现的其他身份原型不作负类。"""
    if logits.ndim != 2 or len(class_sessions) != logits.shape[1]:
        raise ValueError("class_sessions 与 ArcFace logits 类别维度不一致")
    allowed = sessions[:, None] == class_sessions[None, :]
    target_allowed = allowed.gather(1, labels[:, None]).squeeze(1)
    if not bool(target_allowed.all()):
        raise ValueError("样本标签与类别 session 映射不一致")
    if (class_series_membership.ndim != 2
            or class_series_membership.shape[1] != logits.shape[1]):
        raise ValueError("class_series_membership 与 ArcFace logits 类别维度不一致")
    if len(series) != len(labels):
        raise ValueError("series 与样本数量不一致")
    if bool((series < 0).any()) or bool(
            (series >= class_series_membership.shape[0]).any()):
        raise ValueError("样本 series 索引超出类别-series 映射")
    membership = class_series_membership.bool()
    target_in_series = membership[series, labels]
    if not bool(target_in_series.all()):
        raise ValueError("样本标签与类别-series 映射不一致")
    allowed = allowed & ~membership[series]
    # 同串内的其他身份被屏蔽，但样本自身目标类始终保留。
    allowed.scatter_(1, labels[:, None], True)
    masked_logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked_logits, labels)


def train_one_epoch_hn(model, loader, opt, lambda_hn: float,
                       class_sessions: torch.Tensor,
                       class_series_membership: torch.Tensor,
                       return_stats: bool = False):
    """HN 链路单轮：CE + λ×批内 batch-hard triplet。"""
    model.train()
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        model.backbone.eval()
    ce_total = torch.zeros((), device=DEVICE)
    hn_total = torch.zeros((), device=DEVICE)
    n_batch = 0
    valid_anchors_per_batch: list[int] = []
    anchors_per_batch: list[int] = []
    for x, y, s, series in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        s, series = s.to(DEVICE), series.to(DEVICE)
        opt.zero_grad()
        # CE 与 triplet 共用同一次随机前向：避免重复计算 backbone，也避免
        # DropPath 等训练态随机性让两项损失基于两套不一致的特征。
        feats = model.encode(x)
        logits = model.head(feats, y)
        ce = session_aware_cross_entropy(
            logits, y, s, class_sessions, series, class_series_membership)
        hn, valid_anchors = triplet_hn_loss(
            feats, y, s, series, class_series_membership,
            margin=0.3, return_details=True)
        if valid_anchors == 0:
            raise RuntimeError("HN batch 没有同时具备跨串正负样本的合法 anchor")
        loss = ce + lambda_hn * hn
        loss.backward()
        opt.step()
        ce_total += ce.detach()
        hn_total += hn.detach()
        n_batch += 1
        valid_anchors_per_batch.append(valid_anchors)
        anchors_per_batch.append(len(y))
    if n_batch == 0:
        raise ValueError("HN loader 未产生任何 batch")
    stats = {
        "valid_anchors": int(sum(valid_anchors_per_batch)),
        "total_anchors": int(sum(anchors_per_batch)),
        "valid_anchor_ratio": float(
            sum(valid_anchors_per_batch) / sum(anchors_per_batch)),
        "zero_valid_batches": int(sum(value == 0 for value in valid_anchors_per_batch)),
        "min_valid_anchors_per_batch": int(min(valid_anchors_per_batch)),
        "max_valid_anchors_per_batch": int(max(valid_anchors_per_batch)),
        "valid_anchors_per_batch": valid_anchors_per_batch,
        "anchors_per_batch": anchors_per_batch,
    }
    result = (ce_total / n_batch).item(), (hn_total / n_batch).item()
    if return_stats:
        return result[0], result[1], stats
    return result


def _save_checkpoint(payload: dict, path: Path) -> None:
    """先写临时文件再原子替换，避免中断留下半个权重。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_history(history: list[dict], path: Path) -> None:
    """原子更新逐轮历史，训练中断后仍可读取最后一个完整 epoch。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(history).to_csv(tmp, index=False)
    tmp.replace(path)


def _save_epoch_progress(model: ReIDModel, optimizer, history: list[dict],
                         out_dir: Path, *, stage: int, epoch: int,
                         val_r1: float) -> None:
    """保存可诊断的最近训练状态；当前仅用于中断取证，不提供自动续训。"""
    _save_history(history, out_dir / "history.csv")
    _save_checkpoint({
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stage": stage,
        "epoch": epoch,
        "val_r1": val_r1,
        "history_rows": len(history),
    }, out_dir / "last.pt")


def _validated_init_state(checkpoint: object,
                          current_state: dict[str, torch.Tensor]) -> tuple[dict, int]:
    """校验显式初始化 checkpoint 完整覆盖当前 backbone；head 可因类别数不同跳过。"""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state"), dict):
        raise ValueError("初始化权重缺少训练 checkpoint 的 state 字典")
    source_state = checkpoint["state"]
    backbone_keys = [key for key in current_state if key.startswith("backbone.")]
    matched_backbone = [
        key for key in backbone_keys
        if key in source_state and getattr(source_state[key], "shape", None)
        == current_state[key].shape
    ]
    missing_backbone = sorted(set(backbone_keys) - set(matched_backbone))
    if missing_backbone:
        raise ValueError(
            f"初始化权重未完整覆盖 backbone：匹配 {len(matched_backbone)}/"
            f"{len(backbone_keys)}，缺失示例 {missing_backbone[:5]}")
    compatible = {
        key: value for key, value in source_state.items()
        if key in current_state and getattr(value, "shape", None) == current_state[key].shape
    }
    return compatible, len(matched_backbone)


def run_training(args, df: pd.DataFrame, out_dir: Path):
    """两阶段训练主流程（冻结 backbone 训 head → 解冻低学习率微调）。

    args 字段：val_n / seed / epochs_stage1 / epochs_stage2 / lr_head /
    lr_backbone / batch；hard_negative=True 时另需 init_ckpt /
    batches_per_epoch / lambda_hn。
    """
    _seed_training(args.seed)
    hard_negative = bool(getattr(args, "hard_negative", False))
    raw_test_sessions = getattr(args, "test_session", None) or []
    if isinstance(raw_test_sessions, str):
        raw_test_sessions = [raw_test_sessions]
    test_sessions = list(dict.fromkeys(
        str(session).strip() for session in raw_test_sessions))
    split_candidates, test_df = _isolate_test_sessions(df, test_sessions)
    train_df, val_df = split_by_individual(
        split_candidates, args.val_n, args.seed)
    purged_shared_series_rows = int(
        train_df.attrs.get("purged_shared_series_rows", 0))
    # 重新映射训练个体编号为 0..n-1（load_confirmed 的编号是全局的，
    # 留出验证个体后会出现空洞，导致 one_hot / CUDA 索引越界）
    train_df = train_df.copy()
    train_df["identity_unit"] = _identity_units(train_df)
    val_df = val_df.copy()
    val_df["identity_unit"] = _identity_units(val_df)
    id2idx = {iid: i for i, iid in enumerate(sorted(train_df["identity_unit"].unique()))}
    train_df["label_idx"] = train_df["identity_unit"].map(id2idx)

    session_keys = train_df["session_id"].astype(str).str.strip()
    if (session_keys == "").any():
        raise ValueError("训练清单存在空 session_id，无法屏蔽跨批次未对齐类别")
    session_to_idx = {
        session: index for index, session in enumerate(sorted(session_keys.unique()))
    }
    train_df["sess"] = session_keys.map(session_to_idx).astype(int)
    n_classes = int(train_df["label_idx"].nunique())
    label_session_counts = train_df.groupby("label_idx")["sess"].nunique()
    if (label_session_counts > 1).any():
        raise ValueError("同一训练身份跨越多个 session，拒绝自动合并未对齐身份")
    class_sessions = torch.full((n_classes,), -1, dtype=torch.long, device=DEVICE)
    for label, session in train_df.groupby("label_idx")["sess"].first().items():
        class_sessions[int(label)] = int(session)
    if bool((class_sessions < 0).any()):
        raise ValueError("训练类别的 session 映射不完整")

    series_keys = train_df["series_unit"].astype(str).str.strip()
    if (series_keys == "").any():
        raise ValueError("训练清单存在空 series_unit，无法执行同串负类剔除")
    series_to_idx = {
        series: index for index, series in enumerate(sorted(series_keys.unique()))
    }
    train_df["series_idx"] = series_keys.map(series_to_idx).astype(int)
    class_series_membership = torch.zeros(
        (len(series_to_idx), n_classes), dtype=torch.bool, device=DEVICE)
    for series, label in train_df[["series_idx", "label_idx"]].drop_duplicates(
            ).itertuples(index=False, name=None):
        class_series_membership[int(series), int(label)] = True

    print(f"[train] 训练 {len(train_df)} 张 / {train_df['identity_unit'].nunique()} 个体；"
          f"验证 {len(val_df)} 张 / {val_df['identity_unit'].nunique()} 个体")

    model = ReIDModel(make_backbone(), n_classes=n_classes).to(DEVICE)
    init_checkpoint_loaded = False
    init_checkpoint_matched_backbone_keys = 0
    if hard_negative:
        tr_ds = DolphinDatasetHn(train_df, make_transforms(True))
        tr_ds.preload()  # 原图大，预缩放到内存缓存，避免 CPU 解码成为瓶颈
        sampler = WithinSessionIdentitySampler(
            tr_ds.labels, tr_ds.sessions, tr_ds.series, args.batch,
            args.batches_per_epoch, args.seed)
        tr_loader = DataLoader(tr_ds, batch_sampler=sampler)
        if getattr(args, "init_ckpt", None):
            if not Path(args.init_ckpt).is_file():
                raise FileNotFoundError(f"初始化权重不存在：{args.init_ckpt}")
            ckpt = torch.load(args.init_ckpt, map_location=DEVICE)
            # head 层形状随训练个体数变化（768×n_classes）：形状不匹配的键
            # 跳过（head 阶段一会重新训练），backbone 知识照常继承
            cur_state = model.state_dict()
            state, init_checkpoint_matched_backbone_keys = _validated_init_state(
                ckpt, cur_state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            init_checkpoint_loaded = True
            print(f"[train] 初始化自 {args.init_ckpt}（missing {len(missing)} 项 = head 层；"
                  f"unexpected {len(unexpected)} 项）")
    else:
        tr_loader = DataLoader(DolphinDatasetSession(train_df, make_transforms(True)),
                               batch_size=args.batch, shuffle=True,
                               num_workers=0, drop_last=False)
    val_loader = DataLoader(DolphinDataset(val_df, make_transforms(False)),
                            batch_size=32, shuffle=False)
    val_series = val_df["series_unit"].fillna("").astype(str).tolist()
    val_sessions = val_df["session_id"].fillna("").astype(str).tolist()
    test_loader = None
    test_series: list[str] = []
    test_session_ids: list[str] = []
    if not test_df.empty:
        test_loader = DataLoader(DolphinDataset(test_df, make_transforms(False)),
                                 batch_size=32, shuffle=False)
        test_series = test_df["series_unit"].fillna("").astype(str).tolist()
        test_session_ids = test_df["session_id"].fillna("").astype(str).tolist()

    # 预训练基线：训练前用同一个 backbone 评估（避免二次加载触发网络）
    pretrained_r1, val_evaluable_queries, val_skipped_queries = eval_retrieval(
        model, val_loader, val_series, val_sessions, return_details=True)
    print(f"[train] 预训练基线（批次内跨串正负候选）val R@1 = {pretrained_r1:.3f}"
          f"（有效 query {val_evaluable_queries}，跳过 {val_skipped_queries}）")
    baseline_test_result = None
    if test_loader is not None:
        baseline_test_result = eval_retrieval(
            model, test_loader, test_series, test_session_ids,
            return_details=True)
    history = []
    triplet_observability = {1: [], 2: []}
    best_r1, best_stage, best_epoch = pretrained_r1, 0, 0
    # best.pt 从训练前基线起步。阶段一只训练分类头，不参与检索 backbone
    # 的择优；阶段二只有严格优于基线/当前最佳时才可替换它。
    _save_checkpoint({
        "state": model.state_dict(),
        "epoch": 0,
        "val_r1": pretrained_r1,
        "stage": 0,
        "selection": "pretrained_baseline",
    }, out_dir / "best.pt")
    stage1_final_r1 = None

    # 阶段一：冻结 backbone 训 head
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.head.parameters() if p.requires_grad],
                            lr=args.lr_head, weight_decay=5e-4)
    print(f"[s1] 冻结 backbone 训 head（{args.epochs_stage1} epochs, lr={args.lr_head}）")
    for ep in range(1, args.epochs_stage1 + 1):
        if hard_negative:
            ce, hn, hn_stats = train_one_epoch_hn(
                model, tr_loader, opt, 0.0, class_sessions,
                class_series_membership, return_stats=True)  # stage1 无 triplet
        else:
            ce, _ = train_one_epoch(model, tr_loader, opt,
                                    n_classes, class_sessions,
                                    class_series_membership)
            hn = 0.0
            hn_stats = None
        # 检索只使用已冻结且保持 eval 的 backbone，因此阶段一 val 与训练前
        # 基线严格相同，无需重复跑完全相同的验证前向。
        r1 = pretrained_r1
        epoch_record = {
            "stage": 1, "epoch": ep, "loss": ce, "hn": hn, "val_r1": r1,
        }
        if hn_stats is not None:
            epoch_record.update({
                "triplet_valid_anchors": hn_stats["valid_anchors"],
                "triplet_total_anchors": hn_stats["total_anchors"],
                "triplet_valid_anchor_ratio": hn_stats["valid_anchor_ratio"],
                "triplet_zero_valid_batches": hn_stats["zero_valid_batches"],
                "triplet_min_valid_anchors_per_batch": hn_stats[
                    "min_valid_anchors_per_batch"],
                "triplet_max_valid_anchors_per_batch": hn_stats[
                    "max_valid_anchors_per_batch"],
                "triplet_valid_anchors_per_batch": json.dumps(
                    hn_stats["valid_anchors_per_batch"], separators=(",", ":")),
            })
            triplet_observability[1].append(
                {"stage": 1, "epoch": ep, "lambda_hn": 0.0, **hn_stats})
        history.append(epoch_record)
        _save_epoch_progress(
            model, opt, history, out_dir, stage=1, epoch=ep, val_r1=r1)
        if ep == args.epochs_stage1:
            stage1_final_r1 = r1
            payload = {"state": model.state_dict(), "epoch": ep,
                       "val_r1": r1, "stage": 1,
                       "selection": "final_epoch_unconditional"}
            _save_checkpoint(payload, out_dir / "best_stage1.pt")
        if ep % 5 == 0 or ep == 1:
            anchor_note = ("" if hn_stats is None else
                           f"  valid anchors {hn_stats['valid_anchors']}/"
                           f"{hn_stats['total_anchors']}")
            print(f"  [s1] ep {ep:2d}  ce {ce:.4f}  hn {hn:.4f}{anchor_note}  "
                  f"val R@1 {r1:.3f}")

    # 阶段二：解冻 backbone 低学习率微调（HN 链路加 λ×批次内 batch-hard triplet）
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=5e-4)
    lambda_hn = getattr(args, "lambda_hn", 0.2) if hard_negative else 0.0
    print(f"[s2] 解冻微调（{args.epochs_stage2} epochs, lr={args.lr_backbone}），"
          f"loss = CE + {lambda_hn} × within-session batch-hard triplet")
    for ep in range(1, args.epochs_stage2 + 1):
        if hard_negative:
            ce, hn, hn_stats = train_one_epoch_hn(
                model, tr_loader, opt, lambda_hn, class_sessions,
                class_series_membership, return_stats=True)
        else:
            ce, _ = train_one_epoch(model, tr_loader, opt,
                                    n_classes, class_sessions,
                                    class_series_membership)
            hn = 0.0
            hn_stats = None
        r1 = eval_retrieval(model, val_loader, val_series, val_sessions)
        epoch_record = {
            "stage": 2, "epoch": ep, "loss": ce, "hn": hn, "val_r1": r1,
        }
        if hn_stats is not None:
            epoch_record.update({
                "triplet_valid_anchors": hn_stats["valid_anchors"],
                "triplet_total_anchors": hn_stats["total_anchors"],
                "triplet_valid_anchor_ratio": hn_stats["valid_anchor_ratio"],
                "triplet_zero_valid_batches": hn_stats["zero_valid_batches"],
                "triplet_min_valid_anchors_per_batch": hn_stats[
                    "min_valid_anchors_per_batch"],
                "triplet_max_valid_anchors_per_batch": hn_stats[
                    "max_valid_anchors_per_batch"],
                "triplet_valid_anchors_per_batch": json.dumps(
                    hn_stats["valid_anchors_per_batch"], separators=(",", ":")),
            })
            triplet_observability[2].append(
                {"stage": 2, "epoch": ep, "lambda_hn": lambda_hn, **hn_stats})
        history.append(epoch_record)
        _save_epoch_progress(
            model, opt, history, out_dir, stage=2, epoch=ep, val_r1=r1)
        if r1 > best_r1:
            best_r1, best_stage, best_epoch = r1, 2, ep
            _save_checkpoint(
                {"state": model.state_dict(), "epoch": ep, "val_r1": r1,
                 "stage": 2, "selection": "strict_val_improvement"},
                out_dir / "best.pt")
        if ep % 5 == 0 or ep == 1:
            anchor_note = ("" if hn_stats is None else
                           f"  valid anchors {hn_stats['valid_anchors']}/"
                           f"{hn_stats['total_anchors']}")
            print(f"  [s2] ep {ep:2d}  ce {ce:.4f}  hn {hn:.4f}{anchor_note}  "
                  f"val R@1 {r1:.3f}")

    # 正常结束时再原子落盘一次；训练过程中已在每个 epoch 后持续更新。
    _save_history(history, out_dir / "history.csv")
    best_test_result = None
    if test_loader is not None:
        best_checkpoint = torch.load(out_dir / "best.pt", map_location=DEVICE)
        model.load_state_dict(best_checkpoint["state"])
        best_test_result = eval_retrieval(
            model, test_loader, test_series, test_session_ids,
            return_details=True)
    note = ("批内 batch-hard 微调：同 session 跨串确认负样本 + 身份均衡采样" if hard_negative
            else "确认个体标签 ArcFace 微调")
    def summarise_triplet(records: list[dict]) -> dict:
        valid = sum(item["valid_anchors"] for item in records)
        total = sum(item["total_anchors"] for item in records)
        return {
            "total_valid_anchors": valid,
            "total_anchors": total,
            "valid_anchor_ratio": valid / total if total else None,
            "zero_valid_batches": sum(
                item["zero_valid_batches"] for item in records),
            "per_epoch": records,
        }

    stage1_triplet = summarise_triplet(triplet_observability[1])
    stage2_triplet = summarise_triplet(triplet_observability[2])
    test_evaluation = None
    if baseline_test_result is not None and best_test_result is not None:
        baseline_test_r1, baseline_test_evaluable, baseline_test_skipped = (
            baseline_test_result)
        best_test_r1, best_test_evaluable, best_test_skipped = best_test_result
        test_evaluation = {
            "protocol": "within_session_cross_series",
            "test_not_used_for_selection": True,
            "sessions": test_sessions,
            "n_rows": int(len(test_df)),
            "n_identity_units": int(_identity_units(test_df).nunique()),
            "pretrained_baseline_r1": baseline_test_r1,
            "pretrained_evaluable_queries": baseline_test_evaluable,
            "pretrained_skipped_queries": baseline_test_skipped,
            "best_checkpoint_r1": best_test_r1,
            "best_evaluable_queries": best_test_evaluable,
            "best_skipped_queries": best_test_skipped,
        }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": f"{note}；指标只代表当前验证个体与时间跨度上的检索一致性，不外推为跨年能力",
            "hard_negative": hard_negative,
            "n_train": len(train_df), "n_val": len(val_df),
            "n_test": len(test_df),
            "purged_train_rows_sharing_val_series": purged_shared_series_rows,
            "n_train_individuals": int(train_df["identity_unit"].nunique()),
            "n_val_individuals": int(val_df["identity_unit"].nunique()),
            "n_test_individuals": (
                int(_identity_units(test_df).nunique()) if not test_df.empty else 0),
            "class_loss_scope": "within_session_excluding_same_series_classes",
            "triplet_scope": (
                "within_session_cross_series_excluding_"
                "anchor_series_cooccurring_identities"),
            "triplet_sampler_target_cross_series_identity_fraction": (
                0.5 if hard_negative else None),
            "triplet_valid_anchor_observability": {
                "primary_total_scope": "stage2_lambda_hn_only",
                **stage2_triplet,
                "stage1_lambda_zero_diagnostic": stage1_triplet,
            },
            "val_n": args.val_n, "seed": args.seed,
            "lambda_hn": lambda_hn, "batch": args.batch,
            "init_ckpt": (str(args.init_ckpt)
                          if getattr(args, "init_ckpt", None) else None),
            "init_checkpoint_loaded": init_checkpoint_loaded,
            "init_checkpoint_matched_backbone_keys": (
                init_checkpoint_matched_backbone_keys),
            "pretrained_baseline_r1": pretrained_r1,
            "val_evaluable_queries": val_evaluable_queries,
            "val_skipped_queries": val_skipped_queries,
            "best_checkpoint_selection": (
                "baseline_stage0_then_stage2_strict_improvement"),
            "best_val_r1": best_r1,
            "best_stage": best_stage,
            "best_epoch": best_epoch,
            "stage1_selection": {
                "checkpoint": ("best_stage1.pt"
                               if args.epochs_stage1 > 0 else None),
                "policy": ("final_epoch_unconditional"
                           if args.epochs_stage1 > 0 else "not_run"),
                "epoch": (args.epochs_stage1
                          if args.epochs_stage1 > 0 else None),
                "val_r1": stage1_final_r1,
                "eligible_for_best_pt": False,
            },
            "test_evaluation": test_evaluation,
        }, f, indent=2, ensure_ascii=False)
    print(f"[train] 完成。预训练基线 val R@1 = {pretrained_r1:.3f}；"
          f"best.pt = stage {best_stage} epoch {best_epoch}，"
          f"val R@1 {best_r1:.3f} → {out_dir}")


def extract_features(args, out_dir: Path, pilot_csv: Path, images_root: Path):
    """用 best.pt 重新提取全部 pilot 特征（供 benchmark / 开放集预演复跑）。"""
    from whitewhale.reid.embedding import extract_embeddings

    p = pd.read_csv(pilot_csv, dtype=str, keep_default_na=False)
    model = make_embedder("metric-learning", metric_ckpt=out_dir / "best.pt")
    out_emb = args.embeddings_out
    extract_embeddings(
        p, model, images_root=images_root, out_path=out_emb,
        merge_from=None,
        model_cfg={"model": model.name,
                   "note": "确认个体标签 ArcFace 微调后特征；当前数据时间间隔短",
                   "feat_dim": FEAT_DIM, "source": str(out_dir / "best.pt"),
                   "crop": "whole", "preprocess": model.preprocess_id},
    )
    print(f"[extract] {len(p)} 张特征 → {out_emb}（模型 {out_dir / 'best.pt'}）")
