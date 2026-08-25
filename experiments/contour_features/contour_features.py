"""
背鳍轮廓特征模块（4.3 研究原型）。

流程：YOLO 裁剪图 → Otsu 分割背鳍剪影 → 提取顶部自由边 → 曲率/缺口/凹陷特征。

特征设计（CurvRank 风格，尺寸归一化）：
- 曲率直方图：自由边各点曲率分布（形状整体弯曲特征）
- 缺口描述子：前 K 个最深的缺口（深度 + 归一化位置）——背鳍后缘刻痕
- 比例特征：自由边高宽比、顶部弧长占比

说明：本模块为实验代码（experiments/），验证有效后收敛到 src/whitewhale/。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FREE_EDGE_RATIO = 0.45      # 自由边占掩码高度的比例（顶部 45%）
RESAMPLE_N = 128            # 自由边重采样点数
N_CURV_BINS = 16            # 曲率直方图 bin 数
N_NOTCHES = 4               # 保留的最深缺口数


# ---------------------------------------------------------------------------
# 分割与自由边
# ---------------------------------------------------------------------------

def segment_fin(img: np.ndarray) -> np.ndarray:
    """灰度 Otsu + 最大连通域，返回背鳍二值掩码 (H, W) uint8。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).sum() > bw.size * 0.5:  # 背景暗 → 取反使鳍为前景
        bw = cv2.bitwise_not(bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return np.zeros_like(bw)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def free_edge(mask: np.ndarray, cut_ratio: float = FREE_EDGE_RATIO) -> np.ndarray:
    """背鳍顶部自由边：从最高点沿左右两侧向下至 cut_ratio 高度的闭合多边形。

    返回 (N, 2) 点序列（x, y），顺序：左侧顶→底 + 右侧底→顶。向量化实现。
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros((0, 2))
    y_min, y_max = int(ys.min()), int(ys.max())
    y_cut = y_min + cut_ratio * (y_max - y_min)
    sel = ys <= y_cut
    ys, xs = ys[sel].astype(np.int32), xs[sel].astype(np.int32)
    if len(ys) == 0:
        return np.zeros((0, 2))
    order = np.argsort(ys, kind="stable")
    ys, xs = ys[order], xs[order]
    uniq, starts = np.unique(ys, return_index=True)
    starts = np.append(starts, len(ys))
    left_x = np.minimum.reduceat(xs, starts[:-1])
    right_x = np.maximum.reduceat(xs, starts[:-1])
    left = np.stack([left_x, uniq], axis=1).astype(np.float32)
    right = np.stack([right_x, uniq], axis=1).astype(np.float32)
    return np.concatenate([left, right[::-1]], axis=0)


def resample_edge(edge: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    """按弧长等距重采样自由边为 n 点（消除像素离散差异）。

    返回 (n, 2)，点间距均匀。
    """
    if len(edge) < 4:
        return edge.astype(np.float32)
    seg = np.linalg.norm(np.diff(edge, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return edge.astype(np.float32)
    targets = np.linspace(0.0, cum[-1], n)
    out = np.empty((n, 2), dtype=np.float32)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = max(0, min(j, len(edge) - 2))
        w = (t - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
        out[i] = edge[j] * (1 - w) + edge[j + 1] * w
    return out


# ---------------------------------------------------------------------------
# 曲率与缺口
# ---------------------------------------------------------------------------

def curvatures(pts: np.ndarray, win: int = 5) -> np.ndarray:
    """逐点曲率（三点法，邻域窗口 win 平滑）。

    曲率符号：凸(外凸)为正，凹(内凹/缺口)为负。返回 (N,)。
    """
    n = len(pts)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = pts[(i - win) % n]
        b = pts[i]
        c = pts[(i + win) % n]
        v1 = a - b
        v2 = c - b
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-9:
            continue
        sin_theta = cross / denom
        # 角度变化量：以 b 为顶点，v1→v2 的转角
        out[i] = float(np.clip(sin_theta, -1.0, 1.0))
    return out


def detect_notches(pts: np.ndarray, curv: np.ndarray,
                   max_k: int = N_NOTCHES) -> np.ndarray:
    """检测缺口：曲率负峰（内凹点），去重相邻峰后按深度取前 K 个。

    返回 (M, 2)：每行 = [深度(归一化), 位置(沿自由边 0~1)]。
    深度 = 内凹点到其两侧最近凸点的连线距离，按自由边高度归一化。
    轮廓方向修正：按多边形有向面积统一为逆时针，保证镜像下曲率符号一致
    （否则水平镜像翻转轮廓方向 → 凸/凹符号全翻转 → 缺口检测错乱）。
    """
    n = len(pts)
    if n < 8:
        return np.zeros((0, 2))
    # 有向面积（鞋带公式）：逆时针为正
    area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                              - pts[:, 1] * np.roll(pts[:, 0], -1)))
    if area < 0:  # 顺时针 → 翻转曲率符号，负峰仍为内凹
        curv = -curv
    h = float(pts[:, 1].max() - pts[:, 1].min()) or 1.0
    # 内凹候选：曲率小于负阈值
    thr = -0.3
    cand = np.where(curv < thr)[0]
    if len(cand) == 0:
        return np.zeros((0, 2))
    # 按曲率从小到大（最凹优先）去重：邻域 ±win 内只留最深一个
    order = cand[np.argsort(curv[cand])]
    kept = []
    for i in order:
        if all(abs(i - j) >= 6 for j in kept):
            kept.append(int(i))
    notches = []
    for i in kept:
        # 深度：内凹点相对局部弦（左右第 8 个点）的距离
        j1, j2 = (i - 8) % n, (i + 8) % n
        p = pts[i]
        v1, v2 = pts[j1], pts[j2]
        seg_len = np.linalg.norm(v2 - v1)
        if seg_len < 1e-9:
            continue
        dist = abs((v2[0] - v1[0]) * (v1[1] - p[1])
                   - (v1[0] - p[0]) * (v2[1] - v1[1])) / seg_len
        notches.append((dist / h, i / n))
    notches.sort(reverse=True)  # 深度降序
    return np.asarray(notches[:max_k], dtype=np.float32)


# ---------------------------------------------------------------------------
# 特征向量
# ---------------------------------------------------------------------------

def contour_feature(edge: np.ndarray, sym_curv: bool = False) -> np.ndarray | None:
    """自由边 → 特征向量（曲率直方图 + 缺口描述子 + 比例特征）。

    特征构成（N_CURV_BINS + N_NOTCHES*2 + 3 维）：
    - 曲率直方图（16）：自由边曲率分布；sym_curv=True 时用 |曲率|（镜像不变，
      供 A14 验证对照——有符号曲率在水平镜像下符号翻转，对称化后特征镜像稳定）
    - 缺口描述子（8）：前 4 个缺口的 [深度, 位置]（镜像不变）
    - 比例特征（3）：高宽比、曲率均值、曲率方差
    全部分量尺寸归一化，与拍摄距离无关。
    失败（自由边过短）返回 None。
    """
    if len(edge) < 32:
        return None
    pts = resample_edge(edge, RESAMPLE_N)
    curv = curvatures(pts)
    curv_e = np.abs(curv) if sym_curv else curv
    hist, _ = np.histogram(curv_e, bins=N_CURV_BINS,
                           range=(0.0, 1.0) if sym_curv else (-1.0, 1.0),
                           density=False)
    hist = hist / (hist.sum() + 1e-9)
    notches = detect_notches(pts, curv)
    notch_feat = np.zeros(N_NOTCHES * 2, dtype=np.float32)
    for k, (depth, pos) in enumerate(notches[:N_NOTCHES]):
        notch_feat[2 * k] = depth
        notch_feat[2 * k + 1] = pos
    h = float(pts[:, 1].max() - pts[:, 1].min()) or 1.0
    w = float(pts[:, 0].max() - pts[:, 0].min()) or 1.0
    ratio = h / w
    extra = np.array([ratio, float(curv_e.mean()), float(curv_e.std())],
                     dtype=np.float32)
    return np.concatenate([hist, notch_feat, extra])


# ---------------------------------------------------------------------------
# 检索距离（特征间相似度）
# ---------------------------------------------------------------------------

def feature_sim(a: np.ndarray, b: np.ndarray) -> float:
    """轮廓特征相似度：余弦相似度（混合量纲维度对余弦更稳健）。"""
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))


def mirror_edge(edge: np.ndarray) -> np.ndarray:
    """自由边水平镜像（x → -x，再平移回正区域），用于 A14 对称性验证。"""
    if len(edge) == 0:
        return edge
    out = edge.copy()
    out[:, 0] = -out[:, 0] + (out[:, 0].max() + out[:, 0].min())
    return out


def edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    """两条自由边的形状距离（对齐质心后逐点最近距离均值，按高度归一化）。

    与镜像验证配套：a = 原边, b = 镜像边 → 度量左右对称性。
    """
    if len(a) < 4 or len(b) < 4:
        return float("inf")
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    ha = float(a_c[:, 1].max() - a_c[:, 1].min()) or 1.0
    # 最近点距离（a 每个点 → b 最近点），用 KD 树
    from scipy.spatial import cKDTree

    tree = cKDTree(b_c)
    d, _ = tree.query(a_c)
    return float(np.median(d) / ha)


def run_all(crops_dir: Path, image_ids: list[str],
            sym_curv: bool = False) -> dict[str, np.ndarray]:
    """批量提取：{image_id: 轮廓特征}。返回特征字典（失败图不收录）。"""
    out: dict[str, np.ndarray] = {}
    for iid in image_ids:
        p = crops_dir / f"{iid}.jpg"
        if not p.exists():
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        mask = segment_fin(img)
        edge = free_edge(mask)
        feat = contour_feature(edge, sym_curv=sym_curv)
        if feat is not None:
            out[iid] = feat
    return out
