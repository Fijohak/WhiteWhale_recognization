"""
同群散图划分（Pool Assignment，正式工具）。

场景（用户确认，2026-08-14）：优先把同群内的未归属散图划分到群内已确认
个体，而不是跨群全库检索。query 与 gallery 都限制在同一群（session）。

语义：输出是 Candidate（候选划归），不代表自动确认；低分散图可能是
新个体候选，需人工审核。阈值默认 0.50 沿用历史 E4 参考值，正式使用前需重标定。

用法：
    python scripts/assign_pool.py
    python scripts/assign_pool.py --topk 5 --threshold 0.50
    python scripts/assign_pool.py --eval            # 同 session 跨串 leave-one-out
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.pipeline.assign_pool import main  # noqa: E402

if __name__ == "__main__":
    main()
