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
import hashlib
import platform
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from whitewhale.data.image_store import ImageStore, validate_safe_image_ids
from whitewhale.data.manifest import compute_sha256

# MegaDescriptor 输入规格
INPUT_SIZE = 224
FEAT_DIM = 768

IDENTIFIER_COLUMNS = {
    "image_id", "confirmed_identity", "individual_id",
    "legacy_confirmed_identity", "legacy_individual_id",
    "session_id", "group_id", "source_group", "relative_path",
    "filename", "series_id", "series_unit", "sequence_guess", "reviewer",
}


def read_metadata_csv(path: Path | str) -> pd.DataFrame:
    """读取产物/清单 CSV，并对所有标识字段保留前导零与原始字符串。"""
    path = Path(path)
    columns = pd.read_csv(path, nrows=0).columns
    dtypes = {column: str for column in columns if column in IDENTIFIER_COLUMNS}
    return pd.read_csv(path, dtype=dtypes, keep_default_na=False)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _git_source_state(repo_root: Path) -> dict[str, str | bool]:
    """读取产物对应的源码提交；非 Git 环境时返回空信息。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    return {"source_commit": commit, "source_dirty": dirty}


def _artifact_provenance(out_path: Path, meta_path: Path,
                         model_cfg: dict, *, generated: bool = True) -> dict:
    """生成 embedding 产物的文件哈希、运行环境与权重来源信息。"""
    repo_root = Path(__file__).resolve().parents[3]
    provenance = {
        "artifact_schema_version": 2 if generated else 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_file": out_path.name,
        "embedding_sha256": compute_sha256(out_path),
        "meta_file": meta_path.name,
        "meta_sha256": compute_sha256(meta_path),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    if generated:
        meta = read_metadata_csv(meta_path)
        if "image_id" not in meta.columns:
            raise ValueError("新产物 meta 缺少 image_id，无法建立行绑定")
        ordered_ids = "\n".join(meta["image_id"].astype(str).tolist()).encode("utf-8")
        provenance.update({
            "provenance_level": "generated_with_row_binding",
            "row_binding": "embedding_row_i_to_meta_image_id_i",
            "ordered_image_ids_sha256": hashlib.sha256(ordered_ids).hexdigest(),
        })
    else:
        provenance["provenance_level"] = "legacy_backfilled_unverified_row_alignment"
    provenance.update(_git_source_state(repo_root))

    checkpoint = model_cfg.get("ckpt") or model_cfg.get("checkpoint")
    if checkpoint is None and model_cfg.get("source"):
        source = Path(str(model_cfg["source"]))
        checkpoint = source if source.suffix.lower() in {".pt", ".pth", ".ckpt"} else None
    if checkpoint is not None:
        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = repo_root / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"模型 config 声明的 checkpoint 不存在: {checkpoint_path}")
        provenance["checkpoint_file"] = str(checkpoint_path.resolve())
        provenance["checkpoint_sha256"] = compute_sha256(checkpoint_path)
    return provenance


def embedding_config_path(embeddings_path: Path) -> Path:
    """返回特征专属 config；仅在旧单一产物布局下回退 embedding_config.json。"""
    embeddings_path = Path(embeddings_path)
    specific = embeddings_path.with_name(f"{embeddings_path.stem}_config.json")
    if specific.exists():
        return specific
    legacy = embeddings_path.parent / "embedding_config.json"
    return legacy


def load_verified_embedding_artifact(
    embeddings_path: Path,
    meta_path: Path | None = None,
    *,
    require_hashes: bool = True,
    allow_nonfinite: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """加载并校验 embedding/meta/config 三件套，发现错配立即失败。

    校验文件名、SHA-256、行数、维度、唯一 image_id 与数值有效性。旧产物可用
    ``require_hashes=False`` 做结构校验，但正式查询/归档应保持默认 fail closed。
    """
    embeddings_path = Path(embeddings_path)
    meta_path = (Path(meta_path) if meta_path is not None else
                 embeddings_path.with_name(f"{embeddings_path.stem}_meta.csv"))
    config_path = embedding_config_path(embeddings_path)
    for name, path in (("embedding", embeddings_path), ("meta", meta_path),
                       ("config", config_path)):
        if not path.exists():
            raise FileNotFoundError(f"{name} 产物不存在: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    expected_embedding = config.get("embedding_file")
    expected_meta = config.get("meta_file")
    if expected_embedding and Path(str(expected_embedding)).name != embeddings_path.name:
        raise ValueError("config 记录的 embedding_file 与实际文件不匹配")
    if expected_meta and Path(str(expected_meta)).name != meta_path.name:
        raise ValueError("config 记录的 meta_file 与实际文件不匹配")

    for key, path in (("embedding_sha256", embeddings_path),
                      ("meta_sha256", meta_path)):
        expected_hash = config.get(key)
        if require_hashes and not expected_hash:
            raise ValueError(f"config 缺少 {key}，请重新提取或回填产物溯源")
        if expected_hash and compute_sha256(path) != expected_hash:
            raise ValueError(f"{path.name} 的 SHA-256 与 config 不一致，拒绝混用产物")

    emb = np.load(embeddings_path, allow_pickle=False)
    meta = read_metadata_csv(meta_path)
    if emb.ndim != 2:
        raise ValueError(f"embedding 必须是二维数组，实际 shape={emb.shape}")
    if len(emb) != len(meta):
        raise ValueError(f"embedding {len(emb)} 与 meta {len(meta)} 数量不一致")
    if config.get("n") is not None and int(config["n"]) != len(emb):
        raise ValueError("config.n 与 embedding 行数不一致")
    if config.get("feat_dim") is not None and int(config["feat_dim"]) != emb.shape[1]:
        raise ValueError("config.feat_dim 与 embedding 维度不一致")
    if "image_id" not in meta.columns:
        raise ValueError("meta 缺少 image_id")
    image_ids = meta["image_id"].astype(str)
    if ((image_ids.str.strip() == "").any()
            or image_ids.duplicated().any()):
        raise ValueError("meta.image_id 为空或重复，无法唯一追溯特征行")
    ordered_digest = config.get("ordered_image_ids_sha256")
    if ordered_digest:
        actual_digest = hashlib.sha256(
            "\n".join(meta["image_id"].astype(str).tolist()).encode("utf-8")
        ).hexdigest()
        if str(ordered_digest) != actual_digest:
            raise ValueError("meta.image_id 行序摘要与 config 不一致，拒绝错配 embedding 行")
    finite_rows = np.isfinite(emb).all(axis=1)
    if allow_nonfinite:
        invalid_rows = ~finite_rows
        if np.isinf(emb).any() or (
                invalid_rows.any()
                and not np.isnan(emb[invalid_rows]).all(axis=1).all()):
            raise ValueError("缺失特征只能用整行 NaN 表示，拒绝 Inf 或部分损坏的特征行")
    elif not finite_rows.all():
        raise ValueError("embedding 含 NaN/Inf，正式图库拒绝加载")
    if finite_rows.any() and np.any(
            np.linalg.norm(emb[finite_rows], axis=1) == 0):
        raise ValueError("embedding 含零向量，无法进行余弦检索")
    return emb, meta, config


def require_generated_artifact_provenance(config: dict) -> None:
    """正式查询/发布只接受提取时建立行绑定的新 schema 产物。"""
    if config.get("artifact_schema_version") != 2:
        raise ValueError(
            "正式流程只接受 artifact_schema_version=2 的生成期产物；请重新提取")
    if config.get("provenance_level") != "generated_with_row_binding":
        raise ValueError(
            "产物仅有事后回填哈希，无法证明 embedding/meta 原始行对齐；请重新提取")
    for key in ("created_at_utc", "ordered_image_ids_sha256", "row_binding"):
        if not config.get(key):
            raise ValueError(f"生成期 provenance 缺少 {key}")


def require_compatible_embedding_configs(
    left: dict,
    right: dict,
    *,
    left_name: str = "query",
    right_name: str = "gallery",
) -> None:
    """校验两套特征是否可直接比较，避免混用模型、裁剪或权重。"""
    for key in ("model", "crop", "preprocess"):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value in (None, "") or right_value in (None, ""):
            raise ValueError(
                f"无法验证特征兼容性：{left_name}/{right_name} config 缺少 {key}")
        if str(left_value) != str(right_value):
            raise ValueError(
                f"特征配置不兼容：{key} 分别为 "
                f"{left_value!r} / {right_value!r}")

    model = str(left.get("model", ""))
    left_hash = left.get("checkpoint_sha256")
    right_hash = right.get("checkpoint_sha256")
    if "metric-learning" in model or left_hash or right_hash:
        if not left_hash or not right_hash:
            raise ValueError(
                f"无法验证特征权重：{left_name}/{right_name} 缺少 checkpoint_sha256")
        if str(left_hash) != str(right_hash):
            raise ValueError("特征权重 SHA-256 不一致，拒绝跨 checkpoint 检索")

    if str(left.get("crop", "")).strip().lower() != "yolo":
        return
    # YOLO 特征空间同时受检测器与裁剪参数影响；只比较
    # crop="yolo" 会把不同检测链路误当为可比特征。pad_x 表示
    # 左/右对称 padding，pad_up/pad_down 分别表示上/下。
    exact_fields = (
        "crop_schema_version",
        "detector_checkpoint_sha256",
        "detector_fallback_policy",
    )
    numeric_fields = (
        "detector_conf",
        "detector_imgsz",
        "detector_pad_x",
        "detector_pad_up",
        "detector_pad_down",
    )
    for key in (*exact_fields, *numeric_fields):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value in (None, "") or right_value in (None, ""):
            raise ValueError(
                f"无法验证 YOLO 裁剪兼容性："
                f"{left_name}/{right_name} config 缺少 {key}")
        if key in numeric_fields:
            try:
                matches = np.isclose(
                    float(left_value), float(right_value), rtol=0.0, atol=1e-12)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"YOLO 裁剪参数 {key} 不是数值") from exc
        else:
            matches = str(left_value) == str(right_value)
        if not matches:
            raise ValueError(
                f"YOLO 裁剪配置不兼容：{key} 分别为 "
                f"{left_value!r} / {right_value!r}")


def backfill_artifact_provenance(embeddings_path: Path,
                                 meta_path: Path | None = None) -> Path:
    """为可信的历史三件套回填当前文件哈希；不伪造原始生成时间。"""
    embeddings_path = Path(embeddings_path)
    meta_path = (Path(meta_path) if meta_path is not None else
                 embeddings_path.with_name(f"{embeddings_path.stem}_meta.csv"))
    config_path = embedding_config_path(embeddings_path)
    if not config_path.exists():
        raise FileNotFoundError(f"缺少历史 config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    emb = np.load(embeddings_path, allow_pickle=False)
    meta = read_metadata_csv(meta_path)
    if emb.ndim != 2 or len(emb) != len(meta):
        raise ValueError("历史 embedding/meta 结构不一致，拒绝回填 provenance")
    if "image_id" not in meta.columns:
        raise ValueError("历史 meta.image_id 不唯一或为空，拒绝回填 provenance")
    image_ids = meta["image_id"].astype(str)
    if ((image_ids.str.strip() == "").any()
            or image_ids.duplicated().any()):
        raise ValueError("历史 meta.image_id 不唯一或为空，拒绝回填 provenance")
    config["n"] = int(len(emb))
    config["feat_dim"] = int(emb.shape[1])
    provenance = _artifact_provenance(
        embeddings_path, meta_path, config, generated=False)
    provenance["created_at_utc"] = config.get("created_at_utc")
    provenance["provenance_recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["provenance_backfilled"] = True
    config.update(provenance)
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(config_path)
    return config_path


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
    source_id = "timm:vit_base_patch14_dinov2.lvd142m"
    input_size = 518
    feat_dim = 768
    preprocess_id = "Resize518"

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
    source_id = "hf-hub:BVRA/MegaDescriptor-T-224"
    input_size = 224
    feat_dim = 768
    preprocess_id = "Resize224"

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
    """MegaDescriptor + 确认个体标签 ArcFace 微调权重。

    train_reid 保存的 best.pt 是 ReIDModel 完整 state_dict（含 head），
    此处只取 backbone 部分做特征提取。预处理与训练一致
    （Resize 256 + CenterCrop 224），保证查询特征与 gallery 同分布。
    """

    preprocess_id = "Resize256+CenterCrop224"

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
    if missing not in {"error", "nan"}:
        raise ValueError("missing 只能是 'error' 或 'nan'")
    if isinstance(manifest, (str, Path)):
        m = read_metadata_csv(manifest)
    else:
        m = manifest.copy()
    assert len(m) > 0, "清单为空"
    if "image_id" not in m.columns:
        raise ValueError("特征清单缺少 image_id")
    validate_safe_image_ids(m["image_id"])

    feat_dim = model.feat_dim or FEAT_DIM
    feats, missing_rows = [], []
    store = ImageStore(images_root) if images_root is not None else None
    with torch.no_grad():
        for i in range(0, len(m), batch_size):
            chunk = m.iloc[i:i + batch_size]
            chunk_emb = np.full((len(chunk), feat_dim), np.nan, dtype=np.float32)
            imgs, ok_pos = [], []
            for j, (_, row) in enumerate(chunk.iterrows()):
                if crops_dir is not None:
                    p = crops_dir / f"{row['image_id']}.jpg"
                else:
                    if store is None:
                        raise ValueError("整图模式需要 images_root")
                    p = store.resolve(row["relative_path"])
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
        if "image_id" not in merge_from.columns:
            raise ValueError("merge_from 缺少 image_id，无法对齐追溯字段")
        if merge_from["image_id"].duplicated().any():
            duplicate = merge_from.loc[
                merge_from["image_id"].duplicated(), "image_id"].iloc[0]
            raise ValueError(f"merge_from 的 image_id 重复，无法唯一对齐: {duplicate}")
        enrich = merge_from.set_index("image_id")
        for col in ("source_group", "session_id", "quality_band",
                    "confirmed_identity", "individual_id", "series_id",
                    "sequence_key", "frame"):
            if col not in enrich.columns:
                continue
            mapped = meta["image_id"].map(enrich[col])
            if col not in meta.columns:
                meta[col] = mapped
                continue
            # 清单已有字段时只填空值，既避免 _x/_y 列，也不覆盖调用方显式值。
            fill_mask = meta[col].isna() | (meta[col].astype(str).str.strip() == "")
            meta.loc[fill_mask, col] = mapped.loc[fill_mask]
    if "session_id" not in meta.columns:
        meta["session_id"] = ""
    missing_sess = meta["session_id"].isna() | (meta["session_id"].astype(str).str.strip() == "")
    if missing_sess.any() and "relative_path" in meta.columns:
        # 列先转 object 再赋值（float 列直接写字符串会触发 pandas LossySetitemError）
        sess = meta["session_id"].astype("object")
        relative = meta.loc[missing_sess, "relative_path"].astype(str).str.replace(
            "\\", "/", regex=False)
        has_parent = relative.str.contains("/", regex=False)
        derived = relative.str.split("/").str[0].where(has_parent, "")
        sess.loc[missing_sess] = derived.to_numpy()
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
        meta_path = out_path.with_name(out_path.stem + "_meta.csv")
        meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
        cfg = dict(model_cfg or {})
        cfg.setdefault("model", model.name)
        cfg.setdefault("model_source", getattr(model, "source_id", model.name))
        if getattr(model, "ckpt_path", None):
            cfg.setdefault("ckpt", str(model.ckpt_path))
        if getattr(model, "weight_path", None):
            cfg.setdefault("checkpoint", str(model.weight_path))
        cfg.setdefault("feat_dim", model.feat_dim or FEAT_DIM)
        cfg.setdefault("n", int(len(m)))
        cfg.update(_artifact_provenance(out_path, meta_path, cfg))
        (out_path.parent / f"{out_path.stem}_config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[extract] {len(emb)} 张 → {out_path} "
              f"(model={model.name}, 缺失 {len(missing_rows)} 张)")
    return emb, meta
