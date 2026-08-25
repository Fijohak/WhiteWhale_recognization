"""
数据工具冒烟测试：3.6 回填 / 1.9 分串 / 3.2 评估集划分 / 3.3 关系表。

运行：python -m pytest tests/test_data_tools.py -v
（unittest 风格，兼容 pytest 收集）
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.data.eval_set import build_draft, build_eval_split  # noqa: E402
from whitewhale.data.history_verify import (  # noqa: E402
    check_consistency, load_summary, mark_verified)
from whitewhale.data.relations import build_relations  # noqa: E402
from whitewhale.data.sequence_groups import (  # noqa: E402
    build_sequence_groups, sample_checklist)


def _tmp_dir():
    return tempfile.TemporaryDirectory()


def _pilot_df():
    return pd.DataFrame({
        "image_id": [f"i{n}" for n in range(6)],
        "individual_id": ["20140806 01_1.0"] * 3 + ["20140806 03_2.0"] * 3,
        "review_status": ["unreviewed"] * 6,
    })


class TestHistoryVerify(unittest.TestCase):
    """3.6 回填：汇总表 → 可信基准 + pilot_set 更新。"""

    def _summary_df(self):
        return pd.DataFrame({
            "group": ["20140806 01_1.0", "20140806 03_2.0"],
            "n_images": [3, 3],
            "n_confirmed": [3, 2],
            "n_uncertain": [0, 1],
            "n_reject": [0, 0],
            "结论": ["通过", "需复核"],
        })

    def test_mark_verified_ok(self):
        """结论=通过 的组登记为可信基准，pilot_set 对应照片 verified，改前备份。"""
        with _tmp_dir() as tmp:
            tmp = Path(tmp)
            summary = tmp / "history_verify_summary.csv"
            pilot = tmp / "pilot_set.csv"
            self._summary_df().to_csv(summary, index=False, encoding="utf-8-sig")
            _pilot_df().to_csv(pilot, index=False, encoding="utf-8-sig")
            before = pilot.read_bytes()

            result = mark_verified(summary, pilot, tmp / "review",
                                   verified_date="2026-08-25")

            self.assertEqual(result["verified_groups"], 1)
            self.assertEqual(result["verified_images"], 3)
            bench = pd.read_csv(tmp / "review" / "history_verified_individuals.csv")
            self.assertEqual(bench["individual_id"].tolist(),
                             ["20140806 01_1.0"])
            self.assertEqual(bench["verified_date"].tolist(), ["2026-08-25"])
            updated = pd.read_csv(pilot)
            mask = updated["individual_id"] == "20140806 01_1.0"
            self.assertTrue((updated.loc[mask, "review_status"] == "verified").all())
            self.assertTrue((updated.loc[~mask, "review_status"] == "unreviewed").all())
            # 备份存在且与改前一致
            backups = list(tmp.glob("pilot_set.csv.bak_*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), before)

    def test_consistency_rejects_pass_with_uncertain(self):
        """结论=通过 但组内有不确定 → 自洽检查失败，拒绝回填。"""
        df = self._summary_df()
        df.loc[1, "结论"] = "通过"  # 03_2.0 有 1 张不确定却标通过
        problems = check_consistency(df)
        self.assertTrue(any("03_2.0" in p for p in problems))

    def test_consistency_rejects_count_mismatch(self):
        """确认+不确定+排除 ≠ 总数 → 自洽校验失败。"""
        df = self._summary_df()
        df.loc[0, "n_confirmed"] = 2  # 1.0 组实际 3 张却只确认 2
        problems = check_consistency(df)
        self.assertTrue(any("数量自洽" in p for p in problems))

    def test_mark_verified_rejects_count_mismatch_with_pilot(self):
        """汇总表 n_images 与 pilot_set 实际照片数不符 → 拒绝回填（防填错）。"""
        with _tmp_dir() as tmp:
            tmp = Path(tmp)
            summary = tmp / "summary.csv"
            pilot = tmp / "pilot.csv"
            self._summary_df().to_csv(summary, index=False, encoding="utf-8-sig")
            # pilot 里 1.0 组只有 2 张（汇总表登记 3）→ 数量对不上
            pilot_df = _pilot_df().drop(index=2).reset_index(drop=True)
            pilot_df.to_csv(pilot, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "数量不一致"):
                mark_verified(summary, pilot, tmp / "review")

    def test_mark_verified_rejects_unknown_group(self):
        """通过组不在 pilot_set → 组名笔误防护。"""
        with _tmp_dir() as tmp:
            tmp = Path(tmp)
            summary = tmp / "summary.csv"
            pilot = tmp / "pilot.csv"
            df = self._summary_df()
            df.loc[0, "group"] = "20140806 01_9.9"  # 不存在
            df.to_csv(summary, index=False, encoding="utf-8-sig")
            _pilot_df().to_csv(pilot, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "不存在"):
                mark_verified(summary, pilot, tmp / "review")

    def test_mark_verified_no_pass_group(self):
        """没有通过组 → 拒绝（核验未完成时不应误回填）。"""
        with _tmp_dir() as tmp:
            tmp = Path(tmp)
            summary = tmp / "summary.csv"
            pilot = tmp / "pilot.csv"
            df = self._summary_df()
            df["结论"] = "需复核"
            df.to_csv(summary, index=False, encoding="utf-8-sig")
            _pilot_df().to_csv(pilot, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "没有结论=通过"):
                mark_verified(summary, pilot, tmp / "review")

    def test_load_summary_rejects_bad_verdict(self):
        """非法结论取值 → 报错。"""
        with _tmp_dir() as tmp:
            summary = Path(tmp) / "summary.csv"
            df = self._summary_df()
            df.loc[0, "结论"] = "差不多通过"
            df.to_csv(summary, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "非法"):
                load_summary(summary)


def _loose_df():
    """8 张散图：2 串（s1: 2 帧间隔2；s2: 3 帧连续）+ 切分样本 + MO 单帧。"""
    rows = [
        # s1：同 key 20140418_SCi_01_RAY，连拍号 0024/0026（间隔2，容忍删帧）
        ("s1a", "0155_20140418_SCi_01_RAY_0024.JPG", "20140418_SCi_01_RAY", 24),
        ("s1b", "0156_20140418_SCi_01_RAY_0026.JPG", "20140418_SCi_01_RAY", 26),
        # s2：同 key，连拍号 0010/0011/0013 → 3 帧一串
        ("s2a", "0005_20151017_HSi_52_1DX_JL_0010.JPG", "20151017_HSi_52_1DX_JL", 10),
        ("s2b", "0006_20151017_HSi_52_1DX_JL_0011.JPG", "20151017_HSi_52_1DX_JL", 11),
        ("s2c", "0007_20151017_HSi_52_1DX_JL_0013.JPG", "20151017_HSi_52_1DX_JL", 13),
        # s3：间隔 9（0005→0014）>2 → 切分为单帧，不成串
        ("s3a", "0143_20140419HSi05_RZ_0005.JPG", "20140419HSi05_RZ", 5),
        ("s3b", "0146_20140419HSi05_RZ_0014.JPG", "20140419HSi05_RZ", 14),
        # MO：无连拍信息，不分组
        ("mo1", "RES20001.JPG", None, None),
    ]
    df = pd.DataFrame([{
        "image_id": iid,
        "session_id": "20140806 01" if key is None else "20151017 02"
        if key.startswith("20151017") else "20140418 01",
        "sequence_guess": "x",  # manifest 旧字段不再使用
        "filename": fn,
        "relative_path": "20140806 01/70-79/" + fn if key is None
        else (key.split("_")[0] + "/70-79/" + fn),
        "label_status": ["loose_known"] * len(rows),
        "candidate_groups": ["20140806 01_1.0;20140806 01_2.0"] * 3
                            + ["20151017 02_3.0"] * 3 + [""] * 2,
        "sequence_source": ["filename_trailing_number"] * len(rows),
        "sequence_confidence": ["medium"] * len(rows),
    } for iid, fn, key, frame in rows])
    return df


class TestSequenceGroups(unittest.TestCase):
    """1.9 分串：按文件名连拍号分组，间隔切串，MO 不成串。"""

    def test_build_sequence_groups(self):
        groups = build_sequence_groups(_loose_df(), min_frames=2)
        self.assertEqual(len(groups), 2)  # s1(2帧) + s2(3帧)；s3 切为单帧
        s1 = groups[groups["sequence_id"].str.contains("SCi_01_RAY")]
        self.assertEqual(s1["n_frames"].iloc[0], 2)
        self.assertIn("s1a", s1["image_ids"].iloc[0])
        self.assertIn("0024", s1["frame_numbers"].iloc[0])
        s2 = groups[groups["sequence_id"].str.contains("1DX_JL")]
        self.assertEqual(s2["n_frames"].iloc[0], 3)
        self.assertIn("s2a", s2["image_ids"].iloc[0])

    def test_gap_splits_and_single_frames_excluded(self):
        """连拍号间隔 >2 切分；切分后的单帧与 MO 帧不成串。"""
        groups = build_sequence_groups(_loose_df(), min_frames=2)
        ids = set(groups["image_ids"].str.split(";").explode())
        self.assertNotIn("s3a", ids)   # 间隔切分后只剩单帧
        self.assertNotIn("s3b", ids)
        self.assertNotIn("mo1", ids)   # MO 无连拍信息

    def test_sample_checklist_lists_all_frames(self):
        groups = build_sequence_groups(_loose_df(), min_frames=2)
        sample = sample_checklist(groups, _loose_df(), n=2)
        self.assertEqual(len(sample), 5)  # 抽满 2 串 = 2+3 帧
        self.assertIn("relative_path", sample.columns)
        self.assertIn("frame_number", sample.columns)
        self.assertIn("sequence_id", sample.columns)

    def test_all_mo_frames_outputs_headers(self):
        """全 MO（无可解析连拍号）→ 空结果也带完整表头（下游可读）。"""
        df = pd.DataFrame({
            "image_id": ["m1", "m2"],
            "session_id": ["20140806 01"] * 2,
            "filename": ["RES20001.JPG", "RES20002.JPG"],
            "relative_path": ["a.JPG", "b.JPG"],
            "label_status": ["loose_known"] * 2,
            "candidate_groups": [""] * 2,
            "sequence_source": ["parent_folder_fallback"] * 2,
            "sequence_confidence": ["low"] * 2,
        })
        groups = build_sequence_groups(df, min_frames=2)
        self.assertEqual(len(groups), 0)
        self.assertEqual(list(groups.columns),
                         ["sequence_id", "session_id", "n_frames",
                          "image_ids", "filenames", "frame_numbers",
                          "candidate_groups", "sequence_source"])

    def test_split_by_gap_empty_and_single(self):
        """split_by_gap 边界：空列表与单元素不崩。"""
        from whitewhale.data.sequence_groups import split_by_gap
        self.assertEqual(split_by_gap([]), [])
        self.assertEqual(split_by_gap([7]), [[7]])
        self.assertEqual(split_by_gap([1, 2, 5]), [[1, 2], [5]])


def _confirmed_manifest():
    confirmed = pd.DataFrame({
        "image_id": ["i0", "i1", "i2", "i3", "i4", "i5"],
        "confirmed_identity": ["A"] * 3 + ["B"] * 3,
        "status": ["confirmed"] * 6,
        "session_id": ["20140806 01"] * 6,
        "source_group": ["1.0"] * 3 + ["2.0"] * 3,
    })
    manifest = pd.DataFrame({
        "image_id": ["i0", "i1", "i2", "i3", "i4", "i5"],
        "session_id": ["20140806 01"] * 6,
        "sequence_guess": ["sq1"] * 2 + ["sq2"] + ["sq3"] * 3,
        "sequence_source": ["filename_trailing_number"] * 6,
        "quality_band": ["70_79"] * 6,
        "relative_path": ["p" + str(n) for n in range(6)],
    })
    return confirmed, manifest


class TestEvalSet(unittest.TestCase):
    """3.2 评估集划分：序列为最小单元，防泄漏。"""

    def test_split_never_splits_sequence(self):
        confirmed, manifest = _confirmed_manifest()
        confirmed = confirmed.drop(columns=["session_id"])  # 与正式入口同口径
        df = confirmed.merge(manifest, on="image_id")
        draft = build_eval_split(df)
        # A 有 2 序列（sq1×2, sq2×1）→ 出 query；B 只有 1 序列 → 全 gallery
        a = draft[draft["individual_id"] == "A"]
        self.assertEqual(a[a["split"] == "query"]["sequence_guess"].nunique(), 1)
        b = draft[draft["individual_id"] == "B"]
        self.assertTrue((b["split"] == "gallery").all())
        # 同序列照片必须同 split（最小单元不拆分）
        by_seq = a.groupby("sequence_guess")["split"].nunique()
        self.assertTrue((by_seq == 1).all())

    def test_no_sequence_images_go_gallery(self):
        confirmed, manifest = _confirmed_manifest()
        manifest.loc[5, "sequence_guess"] = ""  # i5 无序列
        confirmed = confirmed.drop(columns=["session_id"])
        df = confirmed.merge(manifest, on="image_id")
        draft = build_eval_split(df)
        self.assertEqual(draft.loc[draft["image_id"] == "i5", "split"].iloc[0],
                         "gallery")

    def test_build_draft_outputs(self):
        with _tmp_dir() as tmp:
            tmp = Path(tmp)
            confirmed, manifest = _confirmed_manifest()
            confirmed.to_csv(tmp / "confirmed.csv", index=False,
                             encoding="utf-8-sig")
            manifest.to_csv(tmp / "manifest.csv", index=False,
                            encoding="utf-8-sig")
            result = build_draft(tmp / "confirmed.csv", tmp / "manifest.csv",
                                 tmp / "out")
            self.assertTrue((tmp / "out" / "eval_set_draft.csv").exists())
            stats = json.loads((tmp / "out" / "eval_set_draft_stats.json")
                               .read_text(encoding="utf-8"))
            self.assertEqual(stats["n_individuals"], 2)
            self.assertEqual(stats["n_query"], 2)  # sq1×2 作 query


class TestRelations(unittest.TestCase):
    """3.3 关系表：同体对导出 + 空表结构就绪。"""

    def test_confirmed_same_pairs(self):
        with _tmp_dir() as tmp:
            tmp = Path(tmp)
            confirmed, _ = _confirmed_manifest()
            confirmed.to_csv(tmp / "confirmed.csv", index=False,
                             encoding="utf-8-sig")
            paths = build_relations(tmp / "confirmed.csv", tmp / "out")
            same = pd.read_csv(tmp / "out" / "relations_confirmed_same.csv")
            # A 组 3 张 → 3 对；B 组 3 张 → 3 对；共 6 对
            self.assertEqual(len(same), 6)
            self.assertTrue((same["relation"] == "confirmed_same").all())
            self.assertEqual(same["individual_id"].nunique(), 2)
            # 空表只有表头（无可靠数据源，不伪造）
            for key in ("confirmed_different", "possibly_same"):
                df = pd.read_csv(tmp / "out" / f"relations_{key}.csv")
                self.assertEqual(len(df), 0)
            note = json.loads((tmp / "out" / "relations_note.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(note["confirmed_same"]["n_pairs"], 6)
            self.assertEqual(note["confirmed_different"]["n_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
