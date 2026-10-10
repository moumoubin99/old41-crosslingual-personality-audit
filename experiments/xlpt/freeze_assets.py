#!/usr/bin/env python3
"""Hash execution assets and enforce qualification/evaluation freeze boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASE_ASSETS = [
    "data/xlpt-support/scenarios_en_v1.0.jsonl",
    "refine-logs/SCENARIO_PARTITION.json",
    "data/xlpt-support/rendered_items_and_scenarios/scenarios_en.jsonl",
    "data/xlpt-support/rendered_items_and_scenarios/rubric_items_en.jsonl",
    "refine-logs/RUBRIC_BEHAVIOR.md",
    "refine-logs/design_card.json",
    "refine-logs/MATH_SPEC.json",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root = Path(args.project).resolve()
    files = list(BASE_ASSETS)
    if args.prompt_config:
        files.append(args.prompt_config)
    missing = [rel for rel in files if not (root / rel).is_file()]
    if missing:
        raise SystemExit("missing freeze inputs: " + ", ".join(missing))
    pending = root / "data/xlpt-support/rendered_items_and_scenarios/PENDING_SELF_REPORT_ITEMS.md"
    self_report_ready = all((root / f"data/xlpt-support/rendered_items_and_scenarios/{name}").is_file()
                            for name in ("bfi2_items_en.jsonl", "tipi_items_en.jsonl"))
    if args.scope == "evaluation" and not self_report_ready:
        raise SystemExit("evaluation freeze refused: authoritative BFI-2/TIPI item surfaces remain absent")
    receipt = {
        "schema_version": 1,
        "status": "FROZEN",
        "scope": args.scope,
        "files": {rel: {"sha256": digest(root / rel), "bytes": (root / rel).stat().st_size} for rel in files},
        "residuals": [],
    }
    if not self_report_ready:
        receipt["residuals"].append({
            "id": "missing-authoritative-self-report-wording",
            "detail": "BFI-2/TIPI verbatim items are absent because official-source egress is blocked; no text was reconstructed.",
            "record": str(pending.relative_to(root)) if pending.exists() else None,
            "effect": "self-report administration and full contamination coverage remain unavailable; behavior/timing surfaces may proceed",
        })
    canonical = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    receipt["self_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "sha256": receipt["self_sha256"],
                      "self_report_ready": self_report_ready}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".")
    ap.add_argument("--scope", choices=("timing", "behavior-qualification", "evaluation"), required=True)
    ap.add_argument("--prompt-config")
    ap.add_argument("--output", required=True)
    main(ap.parse_args())
