#!/usr/bin/env python3
"""Tests for the B1-vs-B2 rater-free indicator contrast.

The contrast is the one honest inferential statement this surface supports, so the tests
guard the two ways it could quietly become dishonest: by pairing things that are not
actually paired, and by drifting into a gate or transfer claim it has no data for.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.xlpt import analyze_b3_indicators as contrast

SEEDS = [20260911, 20260912, 20260913, 20260914]
SCENARIOS = ["xlpt-a-v1", "xlpt-b-v1", "xlpt-c-v1"]


def _record(seed, scenario, offset=0.0):
    return {
        "seed": seed,
        "scenario_id": scenario,
        "eligible_for_analysis": True,
        "refusal_turn_rate": 0.10 + offset,
        "escalation_turn_rate": 0.20 + offset,
        "apology_turn_rate": 0.30 - offset,
        "mean_agent_words_per_turn": 40.0 + 10.0 * offset,
    }


def _write_phase(directory: Path, *, offset=0.0, seeds=SEEDS, scenarios=SCENARIOS):
    directory.mkdir(parents=True, exist_ok=True)
    for index, seed in enumerate(seeds):
        # a little per-seed jitter, so the bootstrap has real dispersion to work with
        jitter = 0.001 * index
        records = [_record(seed, name, offset + jitter) for name in scenarios]
        (directory / f"seed-{seed}.json").write_text(
            json.dumps({"source": "test", "records": records}), encoding="utf-8")


class IndicatorContrastTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.baseline = self.root / "process/results/xlpt/b3-rater-free/b1-neutral"
        self.contrast = self.root / "process/results/xlpt/b3-rater-free/b2-compliance"

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_contrast_is_paired_within_seed_clusters(self):
        _write_phase(self.baseline, offset=0.0)
        _write_phase(self.contrast, offset=0.05)
        report = contrast.build_report(self.baseline, self.contrast)
        self.assertEqual(report["clusters"], len(SEEDS))
        self.assertEqual(report["scenarios_per_cluster"], len(SCENARIOS))
        self.assertEqual(report["resampling_unit"], "seed cluster")
        block = report["summary"]["refusal_turn_rate"]
        self.assertAlmostEqual(block["mean_difference"], 0.05, places=6)
        self.assertEqual(block["clusters_positive"], len(SEEDS))
        # a sign that flips direction per indicator must survive, not be absolute-valued away
        self.assertAlmostEqual(
            report["summary"]["apology_turn_rate"]["mean_difference"], -0.05, places=6)

    def test_a_null_contrast_is_reported_as_null(self):
        _write_phase(self.baseline, offset=0.0)
        _write_phase(self.contrast, offset=0.0)
        report = contrast.build_report(self.baseline, self.contrast)
        for name, block in report["summary"].items():
            self.assertAlmostEqual(block["mean_difference"], 0.0, places=9, msg=name)
            self.assertFalse(block["interval_excludes_zero"], name)
            self.assertEqual(block["clusters_zero"], len(SEEDS), name)

    def test_the_bootstrap_interval_is_deterministic_and_brackets_the_mean(self):
        _write_phase(self.baseline, offset=0.0)
        _write_phase(self.contrast, offset=0.05)
        first = contrast.build_report(self.baseline, self.contrast)
        second = contrast.build_report(self.baseline, self.contrast)
        self.assertEqual(first["summary"], second["summary"])
        block = first["summary"]["escalation_turn_rate"]
        self.assertLessEqual(block["bootstrap"]["low"], block["mean_difference"])
        self.assertGreaterEqual(block["bootstrap"]["high"], block["mean_difference"])
        self.assertEqual(block["bootstrap"]["draws"], contrast.BOOTSTRAP_DRAWS)

    def test_unpaired_seed_blocks_are_refused_rather_than_intersected(self):
        _write_phase(self.baseline, seeds=SEEDS)
        _write_phase(self.contrast, seeds=SEEDS[:-1])
        with self.assertRaisesRegex(ValueError, "do not share one seed block"):
            contrast.build_report(self.baseline, self.contrast)

    def test_a_scenario_mismatch_inside_a_cluster_is_refused(self):
        _write_phase(self.baseline, scenarios=SCENARIOS)
        _write_phase(self.contrast, scenarios=SCENARIOS[:-1] + ["xlpt-other-v1"])
        with self.assertRaisesRegex(ValueError, "same scenarios"):
            contrast.build_report(self.baseline, self.contrast)

    def test_a_record_not_eligible_for_analysis_is_refused(self):
        _write_phase(self.baseline)
        _write_phase(self.contrast)
        path = self.contrast / f"seed-{SEEDS[0]}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["records"][0]["eligible_for_analysis"] = False
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not eligible for analysis"):
            contrast.build_report(self.baseline, self.contrast)

    def test_a_repeated_scenario_inside_one_cluster_is_refused(self):
        _write_phase(self.baseline)
        _write_phase(self.contrast)
        path = self.baseline / f"seed-{SEEDS[1]}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["records"].append(dict(payload["records"][0]))
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "repeats scenario"):
            contrast.build_report(self.baseline, self.contrast)

    def test_the_readout_never_claims_a_gate_or_a_transfer_reading(self):
        _write_phase(self.baseline)
        _write_phase(self.contrast, offset=0.05)
        report = contrast.build_report(self.baseline, self.contrast)
        self.assertEqual(report["gate_decision"], "NOT_DECIDED_BY_THIS_TOOL")
        self.assertEqual(report["status"], "exploratory")
        self.assertFalse(report["is_c1_transfer_reading"])
        self.assertFalse(report["uses_gpu"])
        self.assertFalse(report["trains_or_updates_weights"])
        self.assertIn("never ran", report["note"])

    def test_cli_writes_inside_the_project_and_leaves_inputs_untouched(self):
        _write_phase(self.baseline)
        _write_phase(self.contrast, offset=0.05)
        before = (self.baseline / f"seed-{SEEDS[0]}.json").read_bytes()
        self.assertEqual(contrast.main(["--project", str(self.root)]), 0)
        written = json.loads(
            (self.root / "process/results/xlpt/B3_INDICATOR_CONTRAST.json")
            .read_text(encoding="utf-8"))
        self.assertEqual(written["clusters"], len(SEEDS))
        self.assertEqual(written["project"], str(self.root))
        self.assertEqual((self.baseline / f"seed-{SEEDS[0]}.json").read_bytes(), before)

    def test_missing_inputs_are_refused_rather_than_defaulted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "does not exist"):
                contrast.main(["--project", tmp])


if __name__ == "__main__":
    unittest.main()
