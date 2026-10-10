#!/usr/bin/env python3
"""Compute deterministic English rater-free indicators from one generation shard."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


REFUSAL = re.compile(r"\b(?:cannot|can't|unable to|won't|will not)\b", re.IGNORECASE)
ESCALATION = re.compile(r"\b(?:supervisor|manager|specialist|escalat(?:e|ed|ion))\b", re.IGNORECASE)
APOLOGY = re.compile(r"\b(?:sorry|apolog(?:y|ize|ise|ized|ised))\b", re.IGNORECASE)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(pattern: re.Pattern[str], turns: list[str]) -> float:
    return sum(bool(pattern.search(text)) for text in turns) / len(turns) if turns else 0.0


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run(args) -> int:
    project = Path(args.project).resolve()
    source = project / f"process/results/xlpt/{args.phase}/qwen3-8b/seed-{args.seed}.jsonl"
    rows = read_jsonl(source)
    if not rows or any(row.get("seed") != args.seed or row.get("phase") != args.phase for row in rows):
        raise ValueError("source shard is empty or does not match the requested immutable cell")
    records = []
    for row in rows:
        turns = [str(text) for text in row.get("agent_turns", [])]
        words = [len(text.split()) for text in turns]
        records.append({
            "scenario_id": row["scenario_id"],
            "phase": args.phase,
            "seed": args.seed,
            "setting": row["setting"],
            "refusal_turn_rate": rate(REFUSAL, turns),
            "escalation_turn_rate": rate(ESCALATION, turns),
            "apology_turn_rate": rate(APOLOGY, turns),
            "mean_agent_words_per_turn": sum(words) / len(words) if words else 0.0,
            "eligible_for_analysis": bool(row.get("eligible_for_analysis")),
            "surface_scope": "English rater-free indicators; no BFI-2/TIPI item text",
        })
    output = project / f"process/results/xlpt/b3-rater-free/{args.phase}/seed-{args.seed}.json"
    if output.exists():
        raise ValueError(f"refusing to overwrite existing immutable indicator shard: {output}")
    atomic_json(output, {"source": str(source), "records": records})
    print(json.dumps({"output": str(output), "records": len(records)}, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--phase", choices=("b1-neutral", "b2-compliance"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
