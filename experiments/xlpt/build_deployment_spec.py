#!/usr/bin/env python3
"""Build a deployment-only formula packet without modifying reviewed MATH_SPEC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEPLOYMENT_FORMULAS = [
    {
        "id": "timing_ar",
        "name": "Autoregressive sequence probability",
        "kind": "other",
        "latex": "p_<LOCAL_PATH> c)=<LOCAL_PATH> c,x_{<t})",
        "explanation": "The deployed Qwen/vLLM decoder produces each continuation token conditionally on the prompt and preceding generated tokens.",
        "expr": "np.prod(token_conditional_probability, axis=-1)",
        "reference_expr": "np.prod(token_conditional_probability, axis=-1)",
        "inputs": ["token_conditional_probability"],
        "output": {"name": "sequence_probability", "shape": ["batch"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "timing_temperature",
        "name": "Temperature-scaled categorical probability",
        "kind": "other",
        "latex": "p_i=<LOCAL_PATH>)}{<LOCAL_PATH>)}",
        "explanation": "vLLM divides logits by the positive temperature and applies softmax before random sampling.",
        "expr": "np.exp(logits / temperature) / np.sum(np.exp(logits / temperature), axis=-1, keepdims=True)",
        "reference_expr": "np.exp(logits / temperature) / np.sum(np.exp(logits / temperature), axis=-1, keepdims=True)",
        "inputs": ["logits", "temperature"],
        "output": {"name": "token_probability", "shape": ["batch", "vocabulary"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "timing_nucleus",
        "name": "Top-p nucleus renormalization",
        "kind": "other",
        "latex": "q_i=<LOCAL_PATH> A_p<LOCAL_PATH> A_p}p_j}",
        "explanation": "The smallest highest-probability nucleus reaching top_p is retained and its mass is renormalized.",
        "expr": "np.where(nucleus_mask, token_probability, 0.0) / np.sum(np.where(nucleus_mask, token_probability, 0.0), axis=-1, keepdims=True)",
        "reference_expr": "np.where(nucleus_mask, token_probability, 0.0) / np.sum(np.where(nucleus_mask, token_probability, 0.0), axis=-1, keepdims=True)",
        "inputs": ["token_probability", "nucleus_mask"],
        "output": {"name": "nucleus_probability", "shape": ["batch", "vocabulary"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "timing_sample",
        "name": "Seeded categorical exponential-race draw",
        "kind": "other",
        "latex": "X=<LOCAL_PATH> q_i/E_i,<LOCAL_PATH> E_i<LOCAL_PATH>)",
        "explanation": "The deployed vLLM sampler uses a seeded exponential race, which is categorically distributed with nucleus probabilities q.",
        "expr": "np.argmax(nucleus_probability / exponential_noise, axis=-1)",
        "reference_expr": "np.argmax(nucleus_probability / exponential_noise, axis=-1)",
        "inputs": ["nucleus_probability", "exponential_noise"],
        "output": {"name": "sampled_token", "shape": ["batch"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "timing_rate",
        "name": "Measured generation-call throughput",
        "kind": "other",
        "latex": "R=N_{<LOCAL_PATH>",
        "explanation": "Timing throughput divides the counted engine.generate calls by measured positive elapsed seconds.",
        "expr": "generated_call_count / elapsed_seconds",
        "reference_expr": "generated_call_count / elapsed_seconds",
        "inputs": ["generated_call_count", "elapsed_seconds"],
        "output": {"name": "requests_per_second", "shape": []},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "surface_features",
        "name": "Frozen-surface count feature vector",
        "kind": "other",
        "latex": "f_i=(c_i,w_i,<LOCAL_PATH>)",
        "explanation": "Each frozen rubric item is represented by exact character, whitespace-token and line counts for infrastructure timing only.",
        "expr": "np.stack((character_count, token_count, line_count), axis=-1)",
        "reference_expr": "np.stack((character_count, token_count, line_count), axis=-1)",
        "inputs": ["character_count", "token_count", "line_count"],
        "output": {"name": "surface_feature", "shape": ["item", "feature"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "surface_standardize",
        "name": "Population-standardized timing features",
        "kind": "other",
        "latex": "z_{ij}=(f_{ij}-<LOCAL_PATH> f_j)/<LOCAL_PATH> f_j)^2},10^{-6})",
        "explanation": "The tiny timing model centers each feature and divides by its population RMS deviation with a fixed zero-variance guard.",
        "expr": "(surface_feature - feature_mean) / feature_scale",
        "reference_expr": "(surface_feature - feature_mean) / feature_scale",
        "inputs": ["surface_feature", "feature_mean", "feature_scale"],
        "output": {"name": "standardized_feature", "shape": ["item", "feature"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "surface_parameter",
        "name": "Seeded timing-model parameter draw",
        "kind": "other",
        "latex": "<LOCAL_PATH>,<LOCAL_PATH> U_j<LOCAL_PATH>,1)",
        "explanation": "A non-scientific seed creates reproducible fixed linear parameters for the timing run.",
        "expr": "uniform_parameter_draw - 0.5",
        "reference_expr": "uniform_parameter_draw - 0.5",
        "inputs": ["uniform_parameter_draw"],
        "output": {"name": "linear_parameter", "shape": ["feature"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "surface_score",
        "name": "Timing-only bounded linear score",
        "kind": "other",
        "latex": "s_i=1+6\\,<LOCAL_PATH> w+b)",
        "explanation": "The actual GPU microbenchmark maps standardized count features through one frozen linear layer and a sigmoid; outputs are never rater scores.",
        "expr": "1.0 + 6.0 / (1.0 + np.exp(-(standardized_feature @ linear_weight + linear_bias)))",
        "reference_expr": "1.0 + 6.0 / (1.0 + np.exp(-(standardized_feature @ linear_weight + linear_bias)))",
        "inputs": ["standardized_feature", "linear_weight", "linear_bias"],
        "output": {"name": "timing_surface_score", "shape": ["item"]},
        "invariants": [],
        "status": "deployment-standard",
    },
    {
        "id": "surface_rate",
        "name": "Measured item-score throughput",
        "kind": "other",
        "latex": "R=N_{<LOCAL_PATH>",
        "explanation": "The microbenchmark divides the counted GPU item scores by measured positive elapsed seconds.",
        "expr": "scored_item_count / elapsed_seconds",
        "reference_expr": "scored_item_count / elapsed_seconds",
        "inputs": ["scored_item_count", "elapsed_seconds"],
        "output": {"name": "items_per_second", "shape": []},
        "invariants": [],
        "status": "deployment-standard",
    },
]


def symbol(name, shape, dtype, role, defined_at):
    return {"name": name, "latex": name, "shape": shape, "dtype": dtype, "role": role,
            "description": "Deployment timing formula symbol.", "defined_at": defined_at}


DEPLOYMENT_SYMBOLS = [
    symbol("token_conditional_probability", ["batch", "token"], "float", "input", "timing_ar"),
    symbol("sequence_probability", ["batch"], "float", "output", "timing_ar"),
    symbol("logits", ["batch", "vocabulary"], "float", "input", "timing_temperature"),
    symbol("temperature", [], "float", "input", "timing_temperature"),
    symbol("token_probability", ["batch", "vocabulary"], "float", "intermediate", "timing_temperature"),
    symbol("nucleus_mask", ["batch", "vocabulary"], "bool", "input", "timing_nucleus"),
    symbol("nucleus_probability", ["batch", "vocabulary"], "float", "intermediate", "timing_nucleus"),
    symbol("exponential_noise", ["batch", "vocabulary"], "float", "input", "timing_sample"),
    symbol("sampled_token", ["batch"], "int", "output", "timing_sample"),
    symbol("generated_call_count", [], "int", "input", "timing_rate"),
    symbol("elapsed_seconds", [], "float", "input", "timing_rate"),
    symbol("requests_per_second", [], "float", "output", "timing_rate"),
    symbol("character_count", ["item"], "int", "input", "surface_features"),
    symbol("token_count", ["item"], "int", "input", "surface_features"),
    symbol("line_count", ["item"], "int", "input", "surface_features"),
    symbol("surface_feature", ["item", "feature"], "float", "intermediate", "surface_features"),
    symbol("feature_mean", ["feature"], "float", "input", "surface_standardize"),
    symbol("feature_scale", ["feature"], "float", "input", "surface_standardize"),
    symbol("standardized_feature", ["item", "feature"], "float", "intermediate", "surface_standardize"),
    symbol("uniform_parameter_draw", ["feature"], "float", "input", "surface_parameter"),
    symbol("linear_parameter", ["feature"], "float", "output", "surface_parameter"),
    symbol("linear_weight", ["feature"], "float", "input", "surface_score"),
    symbol("linear_bias", [], "float", "input", "surface_score"),
    symbol("timing_surface_score", ["item"], "float", "output", "surface_score"),
    symbol("scored_item_count", [], "int", "input", "surface_rate"),
    symbol("items_per_second", [], "float", "output", "surface_rate"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.project).resolve()
    source = json.loads((root / "refine-logs/MATH_SPEC.json").read_text(encoding="utf-8"))
    packet = dict(source)
    packet["formulas"] = list(source["formulas"]) + DEPLOYMENT_FORMULAS
    packet["symbol_table"] = list(source.get("symbol_table", [])) + DEPLOYMENT_SYMBOLS
    packet["deployment_extension"] = {
        "scope": "non-scientific timing only",
        "authoritative_scientific_spec": "refine-logs/MATH_SPEC.json",
        "note": "Standard inference/timing formulas are appended for deployment review; reviewed scientific formulas are copied unchanged.",
    }
    target = (root / args.output).resolve()
    if not target.is_relative_to(root):
        raise ValueError("output escapes project")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(target)


if __name__ == "__main__":
    main()
