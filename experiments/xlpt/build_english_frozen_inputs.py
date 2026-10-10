#!/usr/bin/env python3
"""Materialize bounded English B0/B1 scenario packets from the frozen corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, path)


project = Path(__file__).resolve().parents[2]
rendered = read_jsonl(project / "data/xlpt-support/rendered_items_and_scenarios/scenarios_en.jsonl")
canonical = {row["scenario_id"]: row for row in read_jsonl(project / "data/xlpt-support/scenarios_en_v1.0.jsonl")}
for name, partition, limit in (
    ("english_b0_competence_scenarios_v1.jsonl", "qualification", 50),
    ("english_b1_control_scenarios_v1.jsonl", "evaluation", 90),
):
    ids = [row["record_id"] for row in rendered if row["partition"].lower() == partition][:limit]
    rows = [dict(canonical[scenario_id], prereg_partition=partition.upper()) for scenario_id in ids]
    if len(rows) != limit:
        raise ValueError(f"{partition}: expected {limit} rows, found {len(rows)}")
    atomic_jsonl(project / "process/consumption/frozen" / name, rows)
    print(name, len(rows))
