#!/usr/bin/env python3
"""Execute the CPU-safe production dependency closure on the actual worker."""

from __future__ import annotations

from pathlib import Path

from .rater_production_closure import run_production_cpu_contracts


ROOT = Path.cwd().resolve()
MANIFEST = ROOT / "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json"
MODEL = Path("<LOCAL_PATH>/OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym")

report = run_production_cpu_contracts(MANIFEST, MODEL)
if report != {
    "dependencies": 53,
    "awq_branches": ["fused_gemm", "dequantize_matmul"],
    "attention": "prefill_plus_cached_decode",
    "eos_token_ids": [2],
    "checkpoint_gqa_groups": 4,
    "tokenizer_io": "real_template_tokens_and_score_text",
    "safetensors_loader": "known_names_shapes_values",
    "entrypoint": "llm_processor_logits_plugin",
    "construction": "registry_llama_plus_awq",
}:
    raise AssertionError(f"production closure report changed: {report}")

print("PASS: production Mistral/AWQ dependency, dispatch, cache, argmax, and EOS closure")
