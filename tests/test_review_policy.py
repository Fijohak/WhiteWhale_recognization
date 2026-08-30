"""M2 审核票型由服务端固定，客户端不能降低门槛。"""
from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.review_policy import (  # noqa: E402
    ReviewVote,
    decide_cluster_purity,
    decide_identity_match,
)


class TestReviewPolicy(unittest.TestCase):
    def test_cluster_purity_is_only_a_batch_local_single_reviewer_result(self):
        decision = decide_cluster_purity([
            ReviewVote(choice="confirm_cluster"),
        ])
        self.assertEqual(decision.status, "resolved")
        self.assertEqual(decision.conclusion, "batch_cluster_confirmed")
        self.assertFalse(decision.creates_formal_identity)

    def test_existing_requires_three_votes_for_the_same_uuid(self):
        existing_a = uuid.uuid4()
        resolved = decide_identity_match([
            ReviewVote(choice="existing", individual_id=existing_a),
            ReviewVote(choice="existing", individual_id=existing_a),
            ReviewVote(choice="existing", individual_id=existing_a),
        ])
        self.assertEqual(resolved.conclusion, "confirm_existing")
        self.assertEqual(resolved.individual_id, existing_a)

        conflict = decide_identity_match([
            ReviewVote(choice="existing", individual_id=existing_a),
            ReviewVote(choice="existing", individual_id=existing_a),
            ReviewVote(choice="new"),
        ])
        self.assertEqual(conflict.status, "conflict")
        self.assertIsNone(conflict.individual_id)

    def test_two_new_votes_create_a_candidate_with_explicit_risk_flags(self):
        possible_duplicate = decide_identity_match([
            ReviewVote(choice="new"),
            ReviewVote(choice="new"),
            ReviewVote(choice="existing", individual_id=uuid.uuid4()),
        ])
        self.assertEqual(possible_duplicate.conclusion, "confirm_new")
        self.assertIn("possible_duplicate", possible_duplicate.flags)

        low_consensus = decide_identity_match([
            ReviewVote(choice="new"),
            ReviewVote(choice="new"),
            ReviewVote(choice="uncertain"),
        ])
        self.assertIn("low_consensus", low_consensus.flags)

    def test_incomplete_or_malformed_votes_never_resolve(self):
        self.assertEqual(decide_identity_match([
            ReviewVote(choice="new"), ReviewVote(choice="new"),
        ]).status, "pending")
        with self.assertRaisesRegex(ValueError, "individual_id"):
            ReviewVote(choice="existing")
        with self.assertRaisesRegex(ValueError, "不允许"):
            ReviewVote(choice="confirmed_kinship")


if __name__ == "__main__":
    unittest.main()
