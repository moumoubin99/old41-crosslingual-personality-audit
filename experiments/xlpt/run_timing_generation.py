#!/usr/bin/env python3
"""Run the deployed eq1--eq7 contract immediately before a timing generation."""

from __future__ import annotations

import json
import os
import sys

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from run_generation import parser, run
from test_math_contract import main as verify_math_contract
from test_runtime_math import main as verify_runtime_contract


if __name__ == "__main__":
    try:
        verify_math_contract()
        verify_runtime_contract()
        sys.exit(run(parser().parse_args()))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise
