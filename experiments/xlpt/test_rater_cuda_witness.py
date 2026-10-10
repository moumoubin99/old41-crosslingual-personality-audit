#!/usr/bin/env python3
"""CPU-side validation of the exact-worker CUDA conformance witness.

The deployment probe deliberately clears CUDA_VISIBLE_DEVICES for contract
tests.  Real kernels are exercised by ``rater_cuda_conformance.py`` immediately
before the probe; this test binds that worker result to the reviewed packet.
"""

from pathlib import Path

from .rater_runtime import validate_cuda_witness


ROOT = Path.cwd().resolve()
MANIFEST = ROOT / "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json"
witness_sha = validate_cuda_witness(MANIFEST, ROOT)
if len(witness_sha) != 64:
    raise AssertionError("exact-worker CUDA witness digest is malformed")

print("PASS: bound exact-worker AWQ, native-op, prefill, cache, and two-step CUDA witness")
