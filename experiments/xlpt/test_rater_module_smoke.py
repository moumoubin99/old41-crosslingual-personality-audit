#!/usr/bin/env python3
"""Smoke the exact module launch path from the deployment working directory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[2]
external_project = Path("<LOCAL_PATH>/41-2026-09-05-aris-3")
completed = subprocess.run(
    [
        sys.executable,
        "-m",
        "experiments.xlpt.run_rater_scoring",
        "--project",
        str(external_project),
        "--seed",
        "20260811",
        "--rater-path",
        "<LOCAL_PATH>/OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym",
        "--runtime-manifest",
        "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
        "--gpu-memory-utilization",
        "0.88",
        "--contract-smoke",
    ],
    cwd=root,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
    capture_output=True,
    text=True,
    timeout=180,
)
if completed.returncode != 0 or '"contract_smoke": "PASS"' not in completed.stdout:
    raise AssertionError(f"module launch failed: {completed.stderr}")
payload = json.loads(completed.stdout.splitlines()[-1])
if (payload.get("contract_smoke") != "PASS"
        or payload.get("entrypoint") != str(root / "experiments/xlpt/run_rater_scoring.py")
        or payload.get("cwd") != str(root)
        or payload.get("processor")
        != "experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"
        or len(payload.get("runtime_manifest_sha256", "")) != 64):
    raise AssertionError(f"module smoke did not behaviorally bind the scoring launch: {payload}")

print("PASS: behavioral python -m scoring smoke, runtime closure, and argmax ties")
