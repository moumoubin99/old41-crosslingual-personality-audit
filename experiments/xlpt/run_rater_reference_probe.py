#!/usr/bin/env python3
"""Single dispatch point for the reviewable Mistral/AWQ rater deployment units.

The deployed rater is vLLM 0.11.0 driving an int4-AWQ Mistral-Small-3.1-24B
checkpoint.  Its forward bottoms out in fused CUDA kernels -- the AWQ GEMM, the
unified attention op, the fused RMSNorm and activation -- which are neither
capturable Python source nor executable under the deployment probe's CPU
constraint.  The ``english-b0-rater-v2`` review rejected the packet for exactly
that reason, and the deployment-formula protocol's own instruction is to "split
an oversized run into a reviewable unit or report that the current
implementation cannot cover it; never omit a dependency to get a PASS".

This runner is that split.  Every unit executes the captured pure-torch
reference in ``mistral_awq_reference.py`` and nothing else.

Units
-----
``quant``       packed AWQ extraction, group indexing, dequantization, the fused
                GEMM equivalence, and the elementwise kernels.
``attention``   rotary frequencies and tables, rotate-half, the QKV split axes,
                grouped-query repetition, the cache-aware causal mask, scaled
                masked attention, and the key-value cache updates.
``decode``      the composed decoder block, the model forward with and without a
                cache, vocabulary-parallel logits, greedy selection, end-of-
                sequence set membership, and the inspectable config of record.
``checkpoint``  reads the deployed checkpoint's ``config.json``,
                ``quantization_config.json`` and generation configuration and
                records their INSPECTABLE CONTENT -- head counts, key-value
                groups, head dimension, RMS epsilon, rotary base, vocabulary
                size, quantization bits and group size, terminal token ids --
                next to their hashes, so a reviewer can check the mathematics
                against the checkpoint rather than against a digest.

Scope, stated rather than assumed
---------------------------------
No output of any unit is a scientific estimator and no output is a rubric score.
Every receipt carries ``eligible_for_analysis: false``.  Nothing here trains,
fine-tunes or updates a weight, and nothing here loads the rater weights or
touches the GPU: all four units are CPU-only by construction.  The sealed
evaluation and qualification seed blocks are refused by
``mistral_awq_reference.check_seed``.

What this runner does NOT decide
--------------------------------
Whether the launched rater is routed through these reviewed functions, or the
production vLLM symbols are instead bound to them, is the open first
requirement of the ``english-b0-rater-v2`` handoff.  This runner takes no
position on it and must not be read as having taken one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import socket
import sys
import time
import unittest
from pathlib import Path

import torch

# Run either as ``python -m experiments.xlpt.run_rater_reference_probe`` or as a
# plain script path; in the second case the package root is not on sys.path.
_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from experiments.xlpt import mistral_awq_reference as ref  # noqa: E402

CONTRACT_UNITS = {
    "quant": "experiments.xlpt.test_mistral_awq_quant",
    "attention": "experiments.xlpt.test_mistral_awq_attention",
    "decode": "experiments.xlpt.test_mistral_awq_decode",
}
UNITS = tuple(CONTRACT_UNITS) + ("checkpoint",)

# The configuration files a reviewer needs in order to check the mathematics.
CHECKPOINT_FILES = (
    "config.json",
    "quantization_config.json",
    "generation_config.json",
    "tokenizer_config.json",
)


def seed_apparatus(seed: int) -> str:
    """Seed fixture construction and return a reproducible RNG witness.

    Contract modules are imported only after this call, so any randomized
    fixtures they construct are controlled by the launch's ``--seed`` rather
    than by a module-level constant.  The witness is not a scientific result.
    """
    torch.manual_seed(int(seed))
    witness = torch.rand(8, dtype=torch.float64).numpy().tobytes()
    return hashlib.sha256(witness).hexdigest()


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_contract_unit(unit):
    """Execute the unit's contract suite in this interpreter and record it."""
    module_name = CONTRACT_UNITS[unit]
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    if suite.countTestCases() == 0:
        raise RuntimeError(f"{module_name} declares no tests")
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    elapsed = time.monotonic() - started
    if not result.wasSuccessful():
        raise RuntimeError(
            f"{module_name}: {len(result.failures)} failures, {len(result.errors)} errors")
    return {
        "unit": unit,
        "kind": "contract",
        "test_module": module_name,
        "test_file": source.name,
        "test_file_sha256": sha256_path(source),
        "tests_run": result.testsRun,
        "elapsed_seconds": elapsed,
        "result": "PASS",
    }


def run_checkpoint_unit(args):
    """Record the checkpoint's inspectable configuration content, not only hashes."""
    model_path = Path(args.rater_path)
    if not model_path.is_dir():
        raise ValueError(f"--rater-path {model_path} is not a directory")

    identity = ref.checkpoint_identity(model_path, CHECKPOINT_FILES)
    config = ref.load_json(model_path / "config.json")
    quantization_file = model_path / "quantization_config.json"
    if not quantization_file.is_file():
        raise ValueError("deployed checkpoint is missing quantization_config.json")
    quantization = ref.load_json(quantization_file)

    ref.validate_config(config)
    ref.validate_quantization_config(config, quantization)
    generation_file = model_path / "generation_config.json"
    generation = ref.load_json(generation_file) if generation_file.exists() else {}
    of_record = ref.config_of_record(config, quantization, generation)

    return {
        "unit": "checkpoint",
        "kind": "identity",
        "model_path": str(model_path),
        "checkpoint_files": identity,
        "config_of_record": of_record,
        "generation_config": generation,
        "note": ("Inspectable content, not only digests: head counts, key-value groups, "
                 "head dimension, RMS epsilon, rotary base, vocabulary size, quantization "
                 "bits and group size, and terminal token ids are all readable here and "
                 "are what the reference mathematics is checked against."),
        "result": "PASS",
    }


def main(args):
    if args.unit not in UNITS:
        raise ValueError(f"unknown unit {args.unit}")
    root = Path(args.project).resolve()
    if not root.is_dir():
        raise ValueError(f"--project {root} is not a directory")
    ref.check_seed(args.seed)
    fixture_rng_sha256 = seed_apparatus(args.seed)

    if args.unit == "checkpoint":
        body = run_checkpoint_unit(args)
    else:
        body = run_contract_unit(args.unit)

    module_dir = Path(__file__).resolve().parent
    record = {
        "schema_version": 1,
        "mode": "mistral-awq-reference-unit",
        "eligible_for_analysis": False,
        "trains_or_updates_weights": False,
        "produces_rubric_score": False,
        "uses_gpu": False,
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "seed": args.seed,
        "fixture_rng_sha256": fixture_rng_sha256,
        "reference_module_sha256": sha256_path(module_dir / "mistral_awq_reference.py"),
        "runner_sha256": sha256_path(Path(__file__).resolve()),
        "note": ("Apparatus verification on the captured pure-torch int4-AWQ Mistral "
                 "reference. No output enters a gate, an admission decision, a rubric "
                 "score or a reported effect."),
    }
    record.update(body)

    output = (root / args.output_json.replace("{unit}", args.unit)
              .replace("{seed}", str(args.seed))).resolve()
    if not output.is_relative_to(root):
        raise ValueError("output escapes the project root")
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"unit": args.unit, "result": record["result"],
                      "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run one reviewable Mistral/AWQ rater deployment unit.")
    ap.add_argument("--unit", required=True, choices=UNITS)
    ap.add_argument("--project", default=".")
    ap.add_argument("--output-json",
                    default="process/logs/rater-reference-{unit}-seed-{seed}.json")
    ap.add_argument("--seed", type=int, default=17091801)
    ap.add_argument(
        "--rater-path",
        default=("<LOCAL_PATH>/models"
                 "OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"))
    raise SystemExit(main(ap.parse_args()))
