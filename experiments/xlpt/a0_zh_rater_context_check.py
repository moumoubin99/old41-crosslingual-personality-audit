#!/usr/bin/env python3
"""A0 pre-flight: tokenize every zh-b4 rater item request against the rater context limit.

CPU only.  Rebuilds each request exactly as ``run_rater_scoring.run`` does (same transcript
rendering, rubric items, request text and date-frozen chat template) and applies the same
``len(prompt) + RATER_MAX_TOKENS >= RATER_MAX_MODEL_LEN`` refusal.  Nothing is truncated and
the limit is not changed; the output is a per checkpoint x pole overflow count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.xlpt.run_rater_scoring import (
    QUALIFICATION_SEEDS,
    RATER_MAX_MODEL_LEN,
    RATER_MAX_TOKENS,
    build_item_request,
    load_rater_chat_template,
    read_jsonl,
)
from experiments.xlpt.score_rater import rubric_items, transcript

PHASES = ("zh-b4-extraversion-high", "zh-b4-extraversion-low")
CHECKPOINTS = ("qwen3-8b", "aya-expanse-8b", "Llama-3.1-8B-Instruct", "gemma-3-12b-it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--rater-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    project = Path(args.project).resolve()
    rater_path = Path(args.rater_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(rater_path), trust_remote_code=False)
    chat_template = load_rater_chat_template(rater_path)
    rubric = project / "refine-logs/RUBRIC_BEHAVIOR.md"
    items = {trait: rubric_items(rubric, trait) for trait in ("extraversion", "agreeableness")}

    cells = []
    for phase in PHASES:
        for checkpoint in CHECKPOINTS:
            n_requests = n_overflow = max_prompt = shards = rows_seen = 0
            missing = []
            overflow_rows = set()
            for seed in sorted(QUALIFICATION_SEEDS):
                path = project / f"process/results/xlpt/{phase}/{checkpoint}/seed-{seed}.jsonl"
                if not path.exists():
                    missing.append(seed)
                    continue
                shards += 1
                for row in read_jsonl(path):
                    rows_seen += 1
                    rendered = transcript(row)
                    for trait, trait_items in items.items():
                        for item in trait_items:
                            request = build_item_request(item, rendered)
                            ids = tokenizer.apply_chat_template(
                                [{"role": "user", "content": request}],
                                chat_template=chat_template,
                                tokenize=True,
                                add_generation_prompt=True,
                            )
                            n = len(ids)
                            n_requests += 1
                            max_prompt = max(max_prompt, n)
                            if n + RATER_MAX_TOKENS >= RATER_MAX_MODEL_LEN:
                                n_overflow += 1
                                overflow_rows.add((seed, row.get("scenario_id")))
            cells.append({
                "phase": phase,
                "checkpoint": checkpoint,
                "shards": shards,
                "missing_seeds": missing,
                "transcripts": rows_seen,
                "item_requests": n_requests,
                "overflow_requests": n_overflow,
                "overflow_transcripts": len(overflow_rows),
                "max_prompt_tokens": max_prompt,
            })
            print(json.dumps(cells[-1]), flush=True)

    report = {
        "check": "A0 zh-b4 rater context pre-flight",
        "max_model_len": RATER_MAX_MODEL_LEN,
        "output_budget": RATER_MAX_TOKENS,
        "rule": "overflow iff prompt_tokens + output_budget >= max_model_len (run_rater_scoring.validate_context_window)",
        "tokenizer": "transformers.AutoTokenizer(rater_path) + frozen chat_template.json",
        "rater_path": str(rater_path),
        "rubric_sha256": hashlib.sha256(rubric.read_bytes()).hexdigest(),
        "cells": cells,
        "total_overflow_requests": sum(c["overflow_requests"] for c in cells),
        "all_shards_present": all(not c["missing_seeds"] for c in cells),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["total_overflow_requests"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
