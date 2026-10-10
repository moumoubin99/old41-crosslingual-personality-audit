#!/usr/bin/env python3
"""Tests for the B0 English rater shard summariser.

The point of the summariser is that the B0 qualification surface is read from real
score shards and never from a hand-written table, so these tests concentrate on the
refusals: a shard that is silently wrong must stop the summary rather than produce a
plausible number.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.xlpt import summarize_b0_rater as summarizer

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _row(seed, *, trait="extraversion", scores=None, source_sha=None, **overrides):
    scores = scores if scores is not None else {f"{trait}_{i}": 4 for i in range(1, 7)}
    row = {
        "schema_version": 2,
        "source_phase": "qualification",
        "source_scenario_id": "xlpt-billing-refund-v1",
        "checkpoint_id": "ckpt-0",
        "setting": "extraversion_high",
        "seed": seed,
        "trait": trait,
        "instrument_language": "English",
        "scores": scores,
        "trait_score": sum(scores.values()) / 6.0,
        "rater_raw_by_item": {key: str(value) for key, value in scores.items()},
        "prompt_config_sha256": DIGEST_A,
        "freeze_receipt_sha256": DIGEST_B,
        "rater_runtime_manifest_sha256": DIGEST_C,
        "source_shard_sha256": source_sha or f"{seed:064d}",
        "eligible_for_gate": True,
        "eligible_for_analysis": False,
        "scoring_elapsed_seconds": 319.0,
        "rater_calls": 600,
        "rater_calls_per_second": 1.88,
    }
    row.update(overrides)
    return row


def _write_shard(directory, seed, rows):
    path = Path(directory) / f"seed-{seed}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _two_good_shards(self):
        first = _write_shard(self.dir, 20260811, [
            _row(20260811, trait="extraversion"),
            _row(20260811, trait="agreeableness",
                 scores={f"agreeableness_{i}": 5 for i in range(1, 7)}),
        ])
        second = _write_shard(self.dir, 20260812, [
            _row(20260812, trait="extraversion",
                 scores={f"extraversion_{i}": 6 for i in range(1, 7)}),
            _row(20260812, trait="agreeableness",
                 scores={f"agreeableness_{i}": 3 for i in range(1, 7)}),
        ])
        return [first, second]

    def test_summary_reports_real_descriptive_statistics(self):
        payload = summarizer.summarize(self._two_good_shards())
        self.assertEqual(payload["seed_count"], 2)
        self.assertEqual(payload["rows_total"], 4)
        self.assertEqual(payload["seeds_summarised"], [20260811, 20260812])
        self.assertTrue(payload["rows_per_shard_consistent"])
        self.assertEqual(payload["by_trait"]["extraversion"]["mean"], 5.0)
        self.assertEqual(payload["by_trait"]["agreeableness"]["mean"], 4.0)
        self.assertEqual(payload["source_phase_counts"], {"qualification": 4})
        self.assertEqual(payload["frozen_digests"]["prompt_config_sha256"], DIGEST_A)

    def test_summary_never_decides_a_gate(self):
        payload = summarizer.summarize(self._two_good_shards())
        self.assertEqual(payload["gate_decision"], "NOT_DECIDED_BY_THIS_TOOL")
        self.assertFalse(payload["eligible_for_analysis"])
        self.assertFalse(payload["uses_gpu"])
        self.assertNotIn("verdict", payload)
        self.assertNotIn("qualification_pass", payload)

    def test_no_shards_is_refused_rather_than_summarised_as_empty(self):
        with self.assertRaisesRegex(ValueError, "no score shards"):
            summarizer.summarize([])

    def test_sealed_evaluation_rows_are_refused(self):
        path = _write_shard(self.dir, 20260911, [_row(20260911, source_phase="evaluation")])
        with self.assertRaisesRegex(ValueError, "sealed by rule E"):
            summarizer.summarize([path])

    def test_analysis_eligible_row_is_refused(self):
        path = _write_shard(self.dir, 20260811, [_row(20260811, eligible_for_analysis=True)])
        with self.assertRaisesRegex(ValueError, "eligible_for_analysis"):
            summarizer.summarize([path])

    def test_trait_score_is_recomputed_not_trusted(self):
        path = _write_shard(self.dir, 20260811, [_row(20260811, trait_score=6.5)])
        with self.assertRaisesRegex(ValueError, "equal-weight mean"):
            summarizer.summarize([path])

    def test_wrong_item_count_is_refused(self):
        path = _write_shard(
            self.dir, 20260811,
            [_row(20260811, scores={f"extraversion_{i}": 4 for i in range(1, 6)})],
        )
        with self.assertRaisesRegex(ValueError, "exactly 6 rubric items"):
            summarizer.summarize([path])

    def test_out_of_range_and_non_integer_items_are_refused(self):
        high = _write_shard(
            self.dir, 20260811,
            [_row(20260811, scores=dict({f"extraversion_{i}": 4 for i in range(2, 7)},
                                        extraversion_1=8))],
        )
        with self.assertRaisesRegex(ValueError, r"outside \[1, 7\]"):
            summarizer.summarize([high])
        floaty = _write_shard(
            self.dir, 20260812,
            [_row(20260812, scores=dict({f"extraversion_{i}": 4 for i in range(2, 7)},
                                        extraversion_1=4.0))],
        )
        with self.assertRaisesRegex(ValueError, "not an integer"):
            summarizer.summarize([floaty])

    def test_mixed_frozen_apparatus_is_refused(self):
        good, _ = self._two_good_shards()[0], None
        drifted = _write_shard(self.dir, 20260813,
                               [_row(20260813, freeze_receipt_sha256="d" * 64)])
        with self.assertRaisesRegex(ValueError, "differs from an earlier shard"):
            summarizer.summarize([good, drifted])

    def test_two_seeds_scored_from_one_generation_shard_are_refused(self):
        first = _write_shard(self.dir, 20260811, [_row(20260811, source_sha="e" * 64)])
        second = _write_shard(self.dir, 20260812, [_row(20260812, source_sha="e" * 64)])
        with self.assertRaisesRegex(ValueError, "double-count"):
            summarizer.summarize([first, second])

    def test_shard_mixing_two_seeds_is_refused(self):
        path = _write_shard(self.dir, 20260811, [_row(20260811), _row(20260812)])
        with self.assertRaisesRegex(ValueError, "mixes seeds"):
            summarizer.summarize([path])

    def test_empty_shard_is_refused(self):
        path = Path(self.dir) / "seed-20260811.jsonl"
        path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no score rows"):
            summarizer.summarize([path])

    def test_cli_writes_the_summary_inside_the_project(self):
        shard_dir = self.dir / "process/results/xlpt/b0-rater/qwen3-8b"
        shard_dir.mkdir(parents=True)
        for path in self._two_good_shards():
            path.rename(shard_dir / path.name)
        code = summarizer.main([
            "--project", str(self.dir),
            "--output-json", "process/results/xlpt/B0_RATER_SUMMARY.json",
        ])
        self.assertEqual(code, 0)
        written = json.loads(
            (self.dir / "process/results/xlpt/B0_RATER_SUMMARY.json").read_text(encoding="utf-8")
        )
        self.assertEqual(written["seed_count"], 2)
        self.assertEqual(written["gate_decision"], "NOT_DECIDED_BY_THIS_TOOL")
        self.assertEqual(len(written["shards"][0]["shard_sha256"]), 64)

    def test_missing_shard_directory_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            summarizer.main(["--project", str(self.dir), "--shard-dir", "process/nowhere"])


if __name__ == "__main__":
    unittest.main()
