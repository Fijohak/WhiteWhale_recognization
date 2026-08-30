"""
跨时间批次管线驱动（正式入口 7/7，E7 首跑验证）。

配置指定的历史库批次 → YOLO 裁剪 → r4 特征 → 逐个新批次跑
批内归档管线（检测裁剪 → 特征 → HDBSCAN → 子簇化 → 簇级多帧投票匹配
历史库）→ 代表图 + 候选簇拼图（人工审核材料）。

用法：
    python scripts/run_cross_time_batch.py            # 全流程（7 个新批次）
    python scripts/run_cross_time_batch.py --sessions "20140419 02"   # 只跑指定批次
    python scripts/run_cross_time_batch.py --only-gallery   # 只读校验活动历史库

该入口禁止原地重建活动 gallery。如需重建，使用
``scripts/rebuild_r4_artifacts.py --out <新版本目录>`` 发布不可覆盖的新版本。
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.pipeline.cross_time import main  # noqa: E402

if __name__ == "__main__":
    main()
