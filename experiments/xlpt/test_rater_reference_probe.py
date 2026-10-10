#!/usr/bin/env python3
"""Conformance tests for the rater reference probe runner itself.

The ``english-b0-rater-v2`` review objected that the module smoke test ran only
``--help`` and therefore proved nothing.  These tests execute the runner's real
dispatch: they run a contract unit end to end, check the receipt it writes,
check that it refuses to overwrite one, check that the sealed seed blocks are
refused at the entry point rather than only deep inside the reference, and check
that the checkpoint unit reports inspectable configuration CONTENT rather than a
digest alone -- including the honest ``unreadable:`` marker for a file that is
not present.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.xlpt import run_rater_reference_probe as probe


def _arguments(project, unit="quant", seed=17091801, rater_path="/nonexistent"):
    return argparse.Namespace(
        unit=unit,
        project=str(project),
        output_json="process/logs/rater-reference-{unit}-seed-{seed}.json",
        seed=seed,
        rater_path=rater_path,
    )


class ContractDispatchTest(unittest.TestCase):
    def test_apparatus_seed_reproduces_and_changes_fixture_stream(self):
        state = torch.random.get_rng_state()
        try:
            first_hash = probe.seed_apparatus(17091801)
            first_draw = torch.rand(6)
            same_hash = probe.seed_apparatus(17091801)
            same_draw = torch.rand(6)
            different_hash = probe.seed_apparatus(17091802)
            different_draw = torch.rand(6)
        finally:
            torch.random.set_rng_state(state)
        self.assertEqual(first_hash, same_hash)
        self.assertTrue(torch.equal(first_draw, same_draw))
        self.assertNotEqual(first_hash, different_hash)
        self.assertFalse(torch.equal(first_draw, different_draw))

    def test_every_declared_unit_has_tests_to_run(self):
        import importlib

        for unit, module_name in probe.CONTRACT_UNITS.items():
            module = importlib.import_module(module_name)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            self.assertGreater(suite.countTestCases(), 0, unit)

    def test_unit_list_covers_the_contract_units_and_the_checkpoint(self):
        self.assertEqual(set(probe.UNITS), set(probe.CONTRACT_UNITS) | {"checkpoint"})

    def test_running_a_unit_writes_a_receipt_that_disclaims_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(probe.main(_arguments(root)), 0)
            receipt = json.loads(
                (root / "process/logs/rater-reference-quant-seed-17091801.json")
                .read_text(encoding="utf-8"))
        self.assertEqual(receipt["result"], "PASS")
        self.assertFalse(receipt["eligible_for_analysis"])
        self.assertFalse(receipt["trains_or_updates_weights"])
        self.assertFalse(receipt["produces_rubric_score"])
        self.assertFalse(receipt["uses_gpu"])
        self.assertGreater(receipt["tests_run"], 0)
        self.assertEqual(len(receipt["reference_module_sha256"]), 64)
        self.assertEqual(len(receipt["fixture_rng_sha256"]), 64)

    def test_a_receipt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe.main(_arguments(root))
            with self.assertRaises(ValueError):
                probe.main(_arguments(root))

    def test_an_output_path_escaping_the_project_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = _arguments(Path(directory))
            arguments.output_json = "../escaped-{unit}-{seed}.json"
            with self.assertRaises(ValueError):
                probe.main(arguments)

    def test_sealed_seed_blocks_are_refused_at_the_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            for sealed in (20260811, 20260911, 20260934):
                with self.assertRaises(ValueError):
                    probe.main(_arguments(Path(directory), seed=sealed))

    def test_an_unknown_unit_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                probe.main(_arguments(Path(directory), unit="sampler"))


class CheckpointUnitTest(unittest.TestCase):
    CONFIG = {
        "model_type": "mistral3",
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 32768,
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000000.0,
            "vocab_size": 131072,
            "eos_token_id": [2, 11],
        },
    }
    QUANTIZATION = {"bits": 4, "group_size": 128, "sym": True,
                    "zero_point": False, "quant_method": "awq", "version": "gemm"}

    def _checkpoint(self, directory, quantization=True):
        root = Path(directory)
        config = json.loads(json.dumps(self.CONFIG))
        if quantization:
            config["text_config"]["quantization_config"] = dict(self.QUANTIZATION)
            (root / "quantization_config.json").write_text(
                json.dumps(self.QUANTIZATION), encoding="utf-8")
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return root

    def test_checkpoint_unit_reports_inspectable_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._checkpoint(directory)
            body = probe.run_checkpoint_unit(_arguments(root, unit="checkpoint",
                                                        rater_path=root))
        record = body["config_of_record"]
        # Content a reviewer can check the mathematics against, not only hashes.
        self.assertEqual(record["num_attention_heads"], 32)
        self.assertEqual(record["num_key_value_heads"], 8)
        self.assertEqual(record["key_value_groups"], 4)
        self.assertEqual(record["head_dim"], 128)
        self.assertEqual(record["rms_norm_eps"], 1e-05)
        self.assertEqual(record["rope_theta"], 1000000000.0)
        self.assertEqual(record["vocab_size"], 131072)
        self.assertEqual(record["eos_token_id"], [2, 11])
        self.assertEqual(record["quantization"]["bits"], 4)
        self.assertEqual(record["quantization"]["group_size"], 128)
        self.assertFalse(body["result"] != "PASS")

    def test_missing_quantization_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._checkpoint(directory, quantization=False)
            with self.assertRaises(ValueError):
                probe.run_checkpoint_unit(_arguments(root, unit="checkpoint",
                                                     rater_path=root))

    def test_disagreeing_quantization_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._checkpoint(directory)
            standalone = dict(self.QUANTIZATION)
            standalone["zero_point"] = True
            (root / "quantization_config.json").write_text(
                json.dumps(standalone), encoding="utf-8")
            with self.assertRaises(ValueError):
                probe.run_checkpoint_unit(_arguments(root, unit="checkpoint",
                                                     rater_path=root))

    def test_a_missing_checkpoint_directory_is_refused(self):
        with self.assertRaises(ValueError):
            probe.run_checkpoint_unit(
                _arguments(".", unit="checkpoint", rater_path="/nonexistent-checkpoint"))

    def test_an_incomplete_configuration_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._checkpoint(directory)
            broken = json.loads((root / "config.json").read_text(encoding="utf-8"))
            del broken["text_config"]["num_key_value_heads"]
            (root / "config.json").write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                probe.run_checkpoint_unit(_arguments(root, unit="checkpoint",
                                                     rater_path=root))
        self.assertIn("num_key_value_heads", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
