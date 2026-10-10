#!/usr/bin/env python3
"""CPU inventory of completed English B1/B2/B3 files. Not a paper result."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shard_inventory(path: Path) -> dict:
    rows = read_jsonl(path)
    return {
        "path": str(path),
        "rows": len(rows),
        "seeds": sorted({row.get("seed") for row in rows}),
        "phases": dict(Counter(str(row.get("phase") or row.get("source_phase")) for row in rows)),
        "settings": dict(Counter(str(row.get("setting")) for row in rows)),
        "eligible_for_analysis": dict(
            Counter(str(row.get("eligible_for_analysis")) for row in rows)
        ),
    }


def mean_indicators(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    if not records:
        return {"path": str(path), "records": 0}
    keys = (
        "refusal_turn_rate",
        "escalation_turn_rate",
        "apology_turn_rate",
        "mean_agent_words_per_turn",
    )
    out = {"path": str(path), "records": len(records)}
    for key in keys:
        values = [float(row[key]) for row in records if key in row]
        out[key] = round(sum(values) / len(values), 6) if values else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output-json", default="process/results/xlpt/ENGLISH_SURFACE_INVENTORY.json")
    args = parser.parse_args()
    root = Path(args.project).resolve()
    b1 = sorted((root / "process/results/xlpt/b1-neutral/qwen3-8b").glob("seed-*.jsonl"))
    b2 = sorted((root / "process/results/xlpt/b2-compliance/qwen3-8b").glob("seed-*.jsonl"))
    b3 = sorted((root / "process/results/xlpt/b3-rater-free").rglob("seed-*.json"))
    payload = {
        "schema_version": 1,
        "tool": "experiments/xlpt/inventory_english_surface.py",
        "kind": "english-surface-file-inventory",
        "eligible_for_analysis": False,
        "confirmatory": False,
        "uses_gpu": False,
        "note": (
            "File inventory and B3 descriptive means only. This is not Rule-F admission, "
            "not the 6x6 transfer matrix, and not a paper result."
        ),
        "b1_neutral_shards": len(b1),
        "b2_compliance_shards": len(b2),
        "b3_indicator_shards": len(b3),
        "b1": [shard_inventory(path) for path in b1],
        "b2": [shard_inventory(path) for path in b2],
        "b3": [mean_indicators(path) for path in b3],
    }
    output = (root / args.output_json).resolve()
    if not str(output).startswith(str(root)):
        raise ValueError("output must stay inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)
    print(json.dumps({
        "output": str(output),
        "b1": payload["b1_neutral_shards"],
        "b2": payload["b2_compliance_shards"],
        "b3": payload["b3_indicator_shards"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
