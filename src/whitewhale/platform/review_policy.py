"""不可由客户端覆盖的审核人数和票型裁决规则。"""
from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass


_CHOICES = frozenset({
    "confirm_cluster", "reject", "uncertain", "split_required", "unusable",
    "existing", "new",
    "confirm_multi_target",
})


@dataclass(frozen=True)
class ReviewVote:
    choice: str
    individual_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        if self.choice not in _CHOICES:
            raise ValueError(f"审核选项不允许: {self.choice}")
        if self.choice == "existing" and self.individual_id is None:
            raise ValueError("existing 票必须包含 individual_id")
        if self.choice != "existing" and self.individual_id is not None:
            raise ValueError("只有 existing 票允许 individual_id")


@dataclass(frozen=True)
class ReviewDecision:
    status: str
    conclusion: str | None = None
    individual_id: uuid.UUID | None = None
    flags: frozenset[str] = frozenset()
    creates_formal_identity: bool = False


def decide_cluster_purity(votes: list[ReviewVote]) -> ReviewDecision:
    if not votes:
        return ReviewDecision("pending")
    if len(votes) != 1:
        raise ValueError("普通簇纯度审核固定为 1 名审核人")
    mapping = {
        "confirm_cluster": "batch_cluster_confirmed",
        "reject": "rejected",
        "uncertain": "uncertain",
        "split_required": "split_required",
        "unusable": "unusable",
    }
    conclusion = mapping.get(votes[0].choice)
    if conclusion is None:
        raise ValueError("普通簇纯度审核不接受身份匹配选项")
    return ReviewDecision("resolved", conclusion)


def decide_identity_match(votes: list[ReviewVote]) -> ReviewDecision:
    if len(votes) < 3:
        return ReviewDecision("pending")
    if len(votes) > 3:
        raise ValueError("跨时间身份审核固定为 3 名审核人")
    if any(vote.choice not in {"existing", "new", "uncertain"}
           for vote in votes):
        raise ValueError("跨时间身份审核包含非法选项")

    existing_ids = [
        vote.individual_id for vote in votes if vote.choice == "existing"]
    if len(existing_ids) == 3 and len(set(existing_ids)) == 1:
        return ReviewDecision(
            "resolved", "confirm_existing", individual_id=existing_ids[0])

    choices = Counter(vote.choice for vote in votes)
    if choices["new"] >= 2:
        flags: set[str] = set()
        if choices["existing"]:
            flags.add("possible_duplicate")
        if choices["uncertain"]:
            flags.add("low_consensus")
        return ReviewDecision(
            "resolved", "confirm_new", flags=frozenset(flags))

    return ReviewDecision(
        "conflict",
        "uncertain",
        flags=frozenset({"manual_resolution_required"}),
    )


def decide_multi_target(votes: list[ReviewVote]) -> ReviewDecision:
    if len(votes) < 3:
        return ReviewDecision("pending")
    if len(votes) > 3:
        raise ValueError("多目标真实性审核固定为 3 名审核人")
    allowed = {"confirm_multi_target", "reject", "uncertain"}
    if any(vote.choice not in allowed for vote in votes):
        raise ValueError("多目标真实性审核包含非法选项")
    counts = Counter(vote.choice for vote in votes)
    if counts["confirm_multi_target"] >= 2:
        return ReviewDecision("resolved", "multi_target_confirmed")
    if counts["reject"] >= 2:
        return ReviewDecision("resolved", "multi_target_rejected")
    return ReviewDecision(
        "conflict", "uncertain",
        flags=frozenset({"manual_resolution_required"}),
    )
