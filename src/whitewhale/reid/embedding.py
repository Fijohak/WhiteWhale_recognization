"""
ReID 模型定义与统一特征提取接口。

合并了三处历史实现（scripts/train_metric_learning.py 的 ReIDModel 模型族、
src/model/reid/embedding/base.py 的预训练适配器、scripts/extract_embeddings.py
与 scripts/extract_r3_yolocrop.py 的特征提取流程），对外只保留一个入口：

    extract_embeddings(manifest, model, images_root=..., crops_dir=..., ...)

差异（预训练 / 微调 / YOLO 裁剪 / 中心裁剪 / 整图）全部由参数表达，
不再为每个组合维护独立脚本。输出 L2 归一化 embedding + 配套 meta，
并写 embedding_config.json 记录特征来源（供查询端自动匹配模型）。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# MegaDescriptor 输入规格
INPUT_SIZE = 224
FEAT_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 微调模型族（ArcFace 度量学习，train_reid 训练产物）
# ---------------------------------------------------------------------------

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

    # 必须在 import timm 之前设置：huggingface_hub 的 offline 常量在 import 时缓存，
    # 之后再设 env 无效（实测会发起 HEAD 请求失败）
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm

    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224",
                              pretrained=True, num_classes=0)
    model.eval()
    return model.to(DEVICE)


def load_reid_model(ckpt: Path | str, device: str = DEVICE) -> ReIDModel:
    """从 train_reid 的 best.pt 加载模型（backbone 初始化自 MegaDescriptor 缓存）。"""
    ckpt_state = torch.load(ckpt, map_location=device)
    model = ReIDModel(make_backbone(), n_classes=ckpt_state["state"]["head.W"].shape[1])
    model.load_state_dict(ckpt_state["state"])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# 预训练适配器（query / benchmark 共用）
# ---------------------------------------------------------------------------

class EmbeddingModel(ABC):
    """embedding 提取器接口。"""

    name: str = "base"
    feat_dim: int = 0

    @abstractmethod
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """输入 PIL 图片列表，返回 L2 归一化 embedding (N, D) float32。"""

    def encode_paths(self, paths: list[Path | str], batch_size: int = 32) -> np.ndarray:
        """从路径批量提取（分批读图避免整批驻留内存，失败即抛错）。"""
        outs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            outs.append(self.encode(imgs))
        return np.concatenate(outs, axis=0)


class _HFImageModel(EmbeddingModel):
    """hf 权重模型的通用实现（预处理 + forward + L2）。"""

    def _load(self):
        raise NotImplementedError

    def __init__(self, device: str = "auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self._load()

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        import torchvision.transforms as T

        tf = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return torch.stack([tf(im) for im in images]).to(self.device)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x = self._preprocess(images)
            out = self.model(x)
        emb = out.float().cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms


class DINOv2Adapter(_HFImageModel):
    """DINOv2 通用自监督视觉表征（对照组）。

    权重来源二选一：timm 在线下载（默认）或 weight_path 加载官方 .pth（离线）。
    模型固定 518x518（官方推荐分辨率，pos_embed 训练尺寸匹配）。
    """

    name = "dinov2"
    input_size = 518
    feat_dim = 768

    def __init__(self, device: str = "auto", weight_path: str | None = None):
        self.weight_path = weight_path
        super().__init__(device)

    def _load(self):
        import timm

        self.model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=self.weight_path is None, num_classes=0)
        if self.weight_path:
            self._load_official_weight()
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.feat_dim = self.model.num_features

    def _load_official_weight(self):
        """加载 facebookresearch/dinov2 官方 .pth（离线）。

        官方权重含预训练任务键 mask_token，timm 模型无此键 → 剔除。
        """
        sd = torch.load(self.weight_path, map_location="cpu")
        sd = sd.get("model", sd)
        sd = {k: v for k, v in sd.items() if k in self.model.state_dict()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise ValueError(f"DINOv2 权重加载不完整，缺 {len(missing)} 个键: "
                             f"{sorted(missing)[:5]}")


class MegaDescriptorAdapter(_HFImageModel):
    """MegaDescriptor-T-224（WildlifeTools，timm hf-hub 加载）。"""

    name = "megadescriptor"
    input_size = 224
    feat_dim = 768

    def _load(self):
        import os

        # 全程离线：权重已在本地缓存，禁止联网校验
        os.environ["HF_HUB_OFFLINE"] = "1"

        import timm

        self.model = timm.create_model(
            "hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.feat_dim = self.model.num_features


class MegaDescriptorMetricAdapter(MegaDescriptorAdapter):
    """MegaDescriptor + 伪标签 ArcFace 微调权重（Candidate 特征）。

    train_reid 保存的 best.pt 是 ReIDModel 完整 state_dict（含 head），
    此处只取 backbone 部分做特征提取。预处理与训练一致
    （Resize 256 + CenterCrop 224），保证查询特征与 gallery 同分布。
    """

    def __init__(self, ckpt_path: Path | str, device: str = "auto"):
        self.ckpt_path = str(ckpt_path)
        # 版本名取自权重目录（r1/r2/r3/...），避免硬编码版本
        self.name = f"megadescriptor-metric-learning-{Path(ckpt_path).parent.name}"
        super().__init__(device)

    def _load(self):
        super()._load()
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        state = ckpt.get("state", ckpt)
        # ReIDModel.state_dict 的 backbone.* → 本模型 state_dict 去前缀
        sd = {k.removeprefix("backbone."): v for k, v in state.items()
              if k.startswith("backbone.")}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise ValueError(f"微调权重加载不完整，缺 {len(missing)} 个键: "
                             f"{sorted(missing)[:5]}（{self.ckpt_path}）")
        self.model.eval()

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """与训练一致：Resize 256 → CenterCrop 224（训练时中心裁剪语义）。"""
        import torchvision.transforms as T

        tf = T.Compose([
            T.Resize(self.input_size + 32),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return torch.stack([tf(im) for im in images]).to(self.device)


def make_embedder(model: str, metric_ckpt: Path | str | None = None,
                  dinov2_weight: str | None = None,
                  device: str = "auto") -> EmbeddingModel:
    """按模型名构建 embedding 提取器（query 端 / 特征提取共用）。

    model ∈ {"megadescriptor", "dinov2", "metric-learning", "mock"}。
    metric-learning 需 metric_ckpt 指向 train_reid 的 best.pt。
    """
    if model == "metric-learning":
        if not metric_ckpt:
            raise ValueError("metric-learning 模型需要 --metric-ckpt 指定微调权重")
        return MegaDescriptorMetricAdapter(metric_ckpt, device=device)
    if model == "dinov2":
        return DINOv2Adapter(device=device, weight_path=dinov2_weight)
    if model == "megadescriptor":
        return MegaDescriptorAdapter(device=device)
    if model == "mock":
        return _MockEmbedder(128)
    raise ValueError(f"未知模型: {model}")


class _MockEmbedder(EmbeddingModel):
    """mock：随机特征，用于离线验证 pipeline 逻辑（不加载真实模型）。"""

    name = "mock"
    feat_dim = 128

    def __init__(self, feat_dim: int = 128, seed: int = 42):
        self.feat_dim = feat_dim
        self._rng = np.random.default_rng(seed)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        emb = self._rng.standard_normal((len(images), self.feat_dim)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms


# ---------------------------------------------------------------------------
# 统一特征提取入口
# ---------------------------------------------------------------------------

def extract_embeddings(
    manifest: pd.DataFrame | Path,
    model: EmbeddingModel,
    images_root: Path | None = None,
    crops_dir: Path | None = None,
    out_path: Path | None = None,
    merge_from: pd.DataFrame | None = None,
    missing: str = "error",
    model_cfg: dict | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, pd.DataFrame]:
    """统一 embedding 提取：整图 / 裁剪图目录两种输入，任意 embedder。

    场景映射（重构前的三个入口）：
    - 整图预训练/微调基线：crops_dir=None，读 images_root/relative_path；
    - YOLO/中心裁剪图：crops_dir 非 None，读 crops_dir/{image_id}.jpg；
    - merge_from（如 pilot_set.csv）提供 source_group/session 等追溯字段，
      只 merge 实际存在的列；session 仍缺失时从 relative_path 首段目录解析。

    缺失图片行为由 missing 控制：
    - "error"（整图模式默认）：任一图缺失直接 SystemExit（禁止静默 0 向量）；
    - "nan"（裁剪图模式）：该行置 NaN 特征，输出后由调用方处理。

    返回 (emb, meta)；out_path 非 None 时写 {out}.npy / {out}_meta.csv /
    {stem}_config.json（记录模型与裁剪来源，供查询端自动匹配）。
    """
    if isinstance(manifest, (str, Path)):
        m = pd.read_csv(manifest)
    else:
        m = manifest.copy()
    assert len(m) > 0, "清单为空"

    feat_dim = model.feat_dim or FEAT_DIM
    feats, missing_rows = [], []
    with torch.no_grad():
        for i in range(0, len(m), batch_size):
            chunk = m.iloc[i:i + batch_size]
            chunk_emb = np.full((len(chunk), feat_dim), np.nan, dtype=np.float32)
            imgs, ok_pos = [], []
            for j, (_, row) in enumerate(chunk.iterrows()):
                if crops_dir is not None:
                    p = crops_dir / f"{row['image_id']}.jpg"
                else:
                    if images_root is None:
                        raise ValueError("整图模式需要 images_root")
                    p = images_root / row["relative_path"]
                if not p.exists():
                    missing_rows.append((row["image_id"], str(p)))
                    continue
                imgs.append(Image.open(p).convert("RGB"))
                ok_pos.append(j)
            if imgs:
                embs = model.encode(imgs)
                for pos, j in enumerate(ok_pos):  # 按原清单行序填回
                    chunk_emb[j] = embs[pos]
            feats.append(chunk_emb)
    emb = np.concatenate(feats, axis=0).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # 零范数保护（模型异常输出全零时避免 NaN）
    emb = emb / norms
    assert len(emb) == len(m), f"特征行数 {len(emb)} ≠ 清单行数 {len(m)}"

    # meta：追溯字段（merge_from 存在列才 merge；session 兜底从路径首段解析）
    meta = m.copy()
    if merge_from is not None:
        merge_cols = [c for c in ["image_id", "source_group", "session_id",
                                  "quality_band", "confirmed_identity"]
                      if c in merge_from.columns]
        if "session_id" in meta.columns:  # 清单自带 session_id，避免 merge 同名冲突
            merge_cols = [c for c in merge_cols if c != "session_id"]
        if merge_cols:
            meta = meta.merge(merge_from[merge_cols], on="image_id", how="left")
    missing_sess = meta["session_id"].isna() | (meta["session_id"].astype(str).str.strip() == "")
    if missing_sess.any() and "relative_path" in meta.columns:
        # 列先转 object 再赋值（float 列直接写字符串会触发 pandas LossySetitemError）
        sess = meta["session_id"].astype("object")
        sess.loc[missing_sess] = (
            meta.loc[missing_sess, "relative_path"].str.split("/").str[0].to_numpy())
        meta["session_id"] = sess

    if missing_rows:
        if missing == "error":
            raise SystemExit(
                f"FATAL: {len(missing_rows)} 张图片读取失败（禁止静默产出 0 向量 embedding）。"
                f"请检查 images_root / crops_dir 与 relative_path。示例: {missing_rows[0][1]}")
        print(f"[extract] 警告: {len(missing_rows)} 张图片缺失，特征置 NaN: "
              f"{missing_rows[:3]}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb)
        meta.to_csv(out_path.with_name(out_path.stem + "_meta.csv"),
                    index=False, encoding="utf-8-sig")
        cfg = dict(model_cfg or {})
        cfg.setdefault("model", model.name)
        cfg.setdefault("feat_dim", model.feat_dim or FEAT_DIM)
        cfg.setdefault("n", int(len(m)))
        (out_path.parent / f"{out_path.stem}_config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[extract] {len(emb)} 张 → {out_path} "
              f"(model={model.name}, 缺失 {len(missing_rows)} 张)")
    return emb, meta
