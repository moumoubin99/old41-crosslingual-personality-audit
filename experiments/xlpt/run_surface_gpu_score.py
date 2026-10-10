#!/usr/bin/env python3
"""Run a timing-only GPU scoring microbenchmark on the frozen rubric surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch

from test_math_contract import main as verify_math_contract


SEALED_SEED_RANGES = (range(20260811, 20260811 + 24), range(20260911, 20260911 + 24))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def surface_features(rows, device):
    values = []
    for row in rows:
        text = row["text"]
        values.append([len(text), len(text.split()), len(text.splitlines())])
    return torch.tensor(values, dtype=torch.float32, device=device)


def standardize_features(features):
    mean = features.mean(dim=0, keepdim=True)
    scale = torch.sqrt(torch.mean((features - mean) ** 2, dim=0, keepdim=True)).clamp_min(1e-6)
    return (features - mean) / scale


def seeded_linear_parameters(seed, width, device):
    if seed < 0 or any(seed in reserved for reserved in SEALED_SEED_RANGES):
        raise ValueError("surface timing seed must be nonnegative and outside sealed scientific blocks")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.rand(width, generator=generator, dtype=torch.float32) - 0.5
    bias = torch.rand(1, generator=generator, dtype=torch.float32) - 0.5
    return weights.to(device), bias.to(device)


def linear_surface_score(standardized_features, weights, bias):
    return 1.0 + 6.0 * torch.sigmoid(standardized_features @ weights + bias)


def measured_throughput(scored_items, elapsed_seconds):
    if scored_items < 0 or elapsed_seconds <= 0:
        raise ValueError("throughput requires nonnegative item count and positive elapsed time")
    return scored_items / elapsed_seconds


def main(args):
    verify_math_contract()
    root = Path(args.project).resolve()
    source = (root / args.input_jsonl).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("input must be an existing project-relative JSONL")
    rows = load_jsonl(source)
    if len(rows) != 12 or any(row.get("surface") != "rater_rubric_item" for row in rows):
        raise ValueError("timing scorer requires the frozen 12-item rubric surface")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU timing scorer")
    device = torch.device("cuda:0")
    features = surface_features(rows, device)
    standardized = standardize_features(features)
    weights, bias = seeded_linear_parameters(args.seed, standardized.shape[1], device)
    torch.cuda.synchronize()
    start = time.monotonic()
    scored_items = 0
    latest = None
    while time.monotonic() - start < args.duration_seconds:
        latest = linear_surface_score(standardized, weights, bias)
        scored_items += len(rows)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    scores = latest.detach().cpu().tolist()
    output_relative = args.output_json.replace("{seed}", str(args.seed))
    output = (root / output_relative).resolve()
    if not output.is_relative_to(root):
        raise ValueError("output escapes project")
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    record = {
        "schema_version": 1,
        "mode": "surface-gpu-timing",
        "eligible_for_analysis": False,
        "input": str(source.relative_to(root)),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": args.seed,
        "duration_seconds": elapsed,
        "scored_items": scored_items,
        "items_per_second": measured_throughput(scored_items, elapsed),
        "record_ids": [row["record_id"] for row in rows],
        "last_timing_only_scores": scores,
        "host": socket.gethostname(),
        "device": torch.cuda.get_device_name(0),
        "note": "Infrastructure timing only; these fixed-feature linear scores are not rater outputs and enter no gate or estimate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "scored_items": scored_items,
                      "items_per_second": record["items_per_second"]}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".")
    ap.add_argument("--input-jsonl", default="data/xlpt-support/rendered_items_and_scenarios/rubric_items_en.jsonl")
    ap.add_argument("--output-json", default="process/logs/surface-gpu-timing-seed-{seed}.json")
    ap.add_argument("--seed", type=int, default=17091701)
    ap.add_argument("--duration-seconds", type=float, default=180.0)
    main(ap.parse_args())
