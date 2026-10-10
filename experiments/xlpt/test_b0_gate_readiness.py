#!/usr/bin/env python3
"""Tests for the rule-F gate readiness readout.

The readout exists to say "the gate cannot run, and here is exactly which registered
requirement is unmet" without ever producing a substitute statistic. So the tests check
both directions: a short, single-cell, single-setting surface must come back blocked with
named reasons, and a surface that did satisfy the registered requirements must come back
runnable -- the verdict is derived from data, not hard-coded.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.xlpt import b0_gate_readiness as readiness

CKPTS = ["gemma-3-12b-it", "Qwen3-8B", "Llama-3.1-8B-Instruct", "aya-expanse-8b"]
L3 = ["English", "Chinese", "Swahili"]


def _gate_spec(**overrides):
    spec = {
        "artifact": "QUALIFICATION_GATE.json",
        "cells": {"checkpoints": list(CKPTS), "languages_L3": list(L3), "n_cells": 12},
        "gates": [{"id": gate_id} for gate_id in readiness.GATE_IDS],
        "results": None,
    }
    spec.update(overrides)
    return spec


def _summary(seeds, *, checkpoints=("Qwen3-8B",), settings=("NEUTRAL",), **overrides):
    payload = {
        "kind": "b0-english-rater-descriptive-summary",
        "gate_decision": "NOT_DECIDED_BY_THIS_TOOL",
        "eligible_for_analysis": False,
        "seeds_summarised": list(seeds),
        "distinct_checkpoints": list(checkpoints),
        "distinct_settings": list(settings),
        "rows_total": 100 * len(list(seeds)),
        "frozen_digests": {"prompt_config_sha256": "a" * 64},
    }
    payload.update(overrides)
    return payload


REGISTERED = list(readiness.QUALIFICATION_SEEDS)
B0_ACTUAL = [seed for seed in REGISTERED if seed != 20260829]   # seed 20260829 never completed


class GateReadinessTests(unittest.TestCase):
    def test_real_b0_surface_is_not_gate_runnable(self):
        report = readiness.build_report(_summary(B0_ACTUAL), _gate_spec())
        self.assertFalse(report["gate_runnable"])
        self.assertEqual(report["gate_decision"], "NOT_DECIDED_BY_THIS_TOOL")
        self.assertFalse(report["writes_qualification_gate"])
        self.assertFalse(report["uses_gpu"])

    def test_the_missing_seed_cluster_is_named_not_rounded_away(self):
        report = readiness.build_report(_summary(B0_ACTUAL), _gate_spec())
        block = report["seed_block"]
        self.assertEqual(block["seed_clusters_available"], 23)
        self.assertEqual(block["seed_clusters_required"], 24)
        self.assertEqual(block["missing_seeds"], [20260829])
        self.assertFalse(block["complete"])
        reasons = " ".join(entry["requirement"] for entry in report["blocking"])
        self.assertIn("G = 24", reasons)

    def test_single_setting_surface_blocks_the_diagonal_admission_effect(self):
        report = readiness.build_report(_summary(REGISTERED), _gate_spec())
        self.assertFalse(report["surface"]["has_pole_contrast"])
        self.assertTrue(any("f_diag" in entry["requirement"] for entry in report["blocking"]))
        self.assertFalse(report["per_gate"]["G5_diagonal_admission"]["computable_from_b0_rater_shards"])

    def test_one_of_twelve_cells_blocks_the_language_level_admission_map(self):
        report = readiness.build_report(_summary(REGISTERED), _gate_spec())
        surface = report["surface"]
        self.assertEqual(surface["checkpoint_coverage"], "1/4")
        self.assertEqual(surface["language_coverage"], "1/3")
        self.assertEqual(surface["cells_covered"], 1)
        self.assertEqual(surface["cells_required"], 12)
        self.assertTrue(any("L = 3 mandatory" in entry["requirement"] for entry in report["blocking"]))

    def test_no_gate_is_reported_as_computable_from_rater_shards(self):
        report = readiness.build_report(_summary(B0_ACTUAL), _gate_spec())
        self.assertEqual(sorted(report["per_gate"]), sorted(readiness.GATE_IDS))
        for gate_id, entry in report["per_gate"].items():
            self.assertFalse(entry["computable_from_b0_rater_shards"], gate_id)
            self.assertTrue(entry["why"].strip())

    def test_verdict_is_derived_from_data_not_hard_coded(self):
        report = readiness.build_report(
            _summary(REGISTERED, checkpoints=CKPTS,
                     settings=("NEUTRAL", "extraversion_high", "extraversion_low")),
            _gate_spec(),
        )
        # 4 checkpoints x 3 languages is still 4 cells here because the rater instrument is
        # English-only, so the readout must not claim runnable on checkpoint coverage alone.
        self.assertEqual(report["surface"]["cells_covered"], 4)
        self.assertFalse(report["gate_runnable"])
        self.assertTrue(report["surface"]["has_pole_contrast"])
        self.assertEqual(
            [entry["requirement"] for entry in report["blocking"]],
            ["12 checkpoint-language cells; L = 3 mandatory, the L = 2 branch is VOID (R16-F3)"],
        )

    def test_a_gate_that_already_has_results_is_never_touched(self):
        with self.assertRaisesRegex(ValueError, "already carries a results block"):
            readiness.build_report(_summary(B0_ACTUAL), _gate_spec(results={"cell_admission": {}}))

    def test_seeds_outside_the_registered_block_are_refused(self):
        with self.assertRaisesRegex(ValueError, "outside the registered qualification block"):
            readiness.build_report(_summary([20260811, 20260911]), _gate_spec())

    def test_a_summary_carrying_a_gate_decision_is_refused(self):
        bad = _summary(B0_ACTUAL, gate_decision="PASS")
        with self.assertRaisesRegex(ValueError, "already carries a gate decision"):
            readiness.build_report(bad, _gate_spec())

    def test_an_analysis_eligible_summary_is_refused(self):
        bad = _summary(B0_ACTUAL, eligible_for_analysis=True)
        with self.assertRaisesRegex(ValueError, "gate-only"):
            readiness.build_report(bad, _gate_spec())

    def test_a_truncated_gate_specification_is_refused(self):
        short = _gate_spec()
        short["gates"] = [{"id": "G1_competence"}]
        with self.assertRaisesRegex(ValueError, "missing registered gates"):
            readiness.build_report(_summary(B0_ACTUAL), short)
        cells = _gate_spec()
        cells["cells"]["languages_L3"] = ["English", "Chinese"]
        with self.assertRaisesRegex(ValueError, "3 registered languages"):
            readiness.build_report(_summary(B0_ACTUAL), cells)

    def test_cli_writes_the_readout_inside_the_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "process/results/xlpt").mkdir(parents=True)
            (root / "refine-logs").mkdir(parents=True)
            (root / "process/results/xlpt/B0_RATER_SUMMARY.json").write_text(
                json.dumps(_summary(B0_ACTUAL)), encoding="utf-8")
            (root / "refine-logs/QUALIFICATION_GATE.json").write_text(
                json.dumps(_gate_spec()), encoding="utf-8")
            self.assertEqual(readiness.main(["--project", str(root)]), 0)
            written = json.loads(
                (root / "refine-logs/B0_GATE_READINESS.json").read_text(encoding="utf-8"))
            self.assertFalse(written["gate_runnable"])
            self.assertEqual(written["seed_block"]["missing_seeds"], [20260829])
            self.assertEqual(written["gate_decision"], "NOT_DECIDED_BY_THIS_TOOL")
            # the registered gate artifact is left byte-identical
            self.assertIsNone(json.loads(
                (root / "refine-logs/QUALIFICATION_GATE.json").read_text(encoding="utf-8"))["results"])

    def test_per_checkpoint_summaries_union_into_one_surface(self):
        report = readiness.build_report(
            [_summary(B0_ACTUAL),
             _summary(B0_ACTUAL, checkpoints=("aya-expanse-8b",)),
             _summary(B0_ACTUAL, checkpoints=("Llama-3.1-8B-Instruct",))],
            _gate_spec())
        self.assertEqual(report["surface"]["checkpoint_coverage"], "3/4")
        self.assertEqual(report["surface"]["cells_covered"], 3)
        # the seed block is a union, not a sum: the same 23 clusters were scored per arm
        self.assertEqual(report["seed_block"]["seed_clusters_available"], len(B0_ACTUAL))
        self.assertEqual(report["seed_block"]["missing_seeds"], [20260829])
        self.assertFalse(report["gate_runnable"])

    def test_the_same_checkpoint_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "more than one summary"):
            readiness.build_report([_summary(B0_ACTUAL), _summary(B0_ACTUAL)], _gate_spec())

    def test_summaries_from_different_apparatus_are_refused(self):
        other = _summary(B0_ACTUAL, checkpoints=("aya-expanse-8b",),
                         frozen_digests={"prompt_config_sha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "one frozen apparatus"):
            readiness.build_report([_summary(B0_ACTUAL), other], _gate_spec())

    def test_missing_inputs_are_refused_rather_than_defaulted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "does not exist"):
                readiness.main(["--project", tmp])


if __name__ == "__main__":
    unittest.main()
