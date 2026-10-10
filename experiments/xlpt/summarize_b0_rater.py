"""Reduce the completed B0 English rater score shards to one auditable summary.

This is a deterministic, CPU-only, read-only reducer. It loads no model, takes no
accelerator, writes into no shard, and decides no gate. It exists because the B0
qualification surface must be read from the real score shards produced by
``experiments/xlpt/run_rater_scoring.py`` -- never from a hand-written table.

What it enforces before it reports anything:

* every row is ``schema_version`` 2, ``eligible_for_gate`` true and
  ``eligible_for_analysis`` false -- a shard claiming analysis eligibility is refused;
* no row carries ``source_phase`` ``evaluation``: that split is sealed by rule E and must
  never reach a descriptive summary;
* every row has exactly six integer rubric items in [1, 7] and a ``trait_score`` equal to
  their equal-weight mean (rule G), recomputed here rather than trusted;
* the prompt-config, freeze-receipt and rater-runtime manifest digests are identical across
  every shard, so the summary describes one frozen apparatus and not a mixture;
* each shard holds exactly one seed, and no two seeds share a ``source_shard_sha256``, so one
  generation shard cannot be counted twice under two seed labels.

What it deliberately does NOT do: it decides no qualification gate, writes no
``QUALIFICATION_GATE.json``, compares against no threshold, and touches no confirmatory
split. ``gate_decision`` in the output is always ``NOT_DECIDED_BY_THIS_TOOL``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

SCHEMA_VERSION = 2
TRAITS = ("extraversion", "agreeableness")
ITEMS_PER_TRAIT = 6
SCORE_MIN = 1
SCORE_MAX = 7
SEALED_PHASE = "evaluation"
FROZEN_DIGEST_FIELDS = (
    "prompt_config_sha256",
    "freeze_receipt_sha256",
    "rater_runtime_manifest_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {number} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} line {number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} contains no score rows")
    return rows


def validate_row(row: dict, *, shard: str, index: int) -> None:
    where = f"{shard} row {index}"
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{where}: schema_version must be {SCHEMA_VERSION}")
    if row.get("eligible_for_analysis") is not False:
        raise ValueError(f"{where}: eligible_for_analysis must be false for a B0 rater shard")
    if row.get("eligible_for_gate") is not True:
        raise ValueError(f"{where}: eligible_for_gate must be true for a B0 rater shard")
    if row.get("source_phase") == SEALED_PHASE:
        raise ValueError(
            f"{where}: source_phase is '{SEALED_PHASE}'; that split is sealed by rule E and "
            "must not be summarised"
        )
    if row.get("instrument_language") != "English":
        raise ValueError(f"{where}: instrument_language must be English on the B0 surface")
    if row.get("trait") not in TRAITS:
        raise ValueError(f"{where}: trait must be one of {TRAITS}")
    scores = row.get("scores")
    if not isinstance(scores, dict) or len(scores) != ITEMS_PER_TRAIT:
        raise ValueError(f"{where}: scores must hold exactly {ITEMS_PER_TRAIT} rubric items")
    for item_id, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: rubric item {item_id} is not an integer")
        if not SCORE_MIN <= value <= SCORE_MAX:
            raise ValueError(f"{where}: rubric item {item_id} is outside [1, 7]")
    recomputed = sum(scores.values()) / float(ITEMS_PER_TRAIT)
    reported = row.get("trait_score")
    if isinstance(reported, bool) or not isinstance(reported, (int, float)):
        raise ValueError(f"{where}: trait_score is missing or not numeric")
    if abs(float(reported) - recomputed) > 1e-9:
        raise ValueError(
            f"{where}: trait_score {reported!r} is not the equal-weight mean {recomputed!r}"
        )
    for field in FROZEN_DIGEST_FIELDS + ("source_shard_sha256",):
        value = row.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{where}: {field} is not a sha256 digest")


def _describe(values: list) -> dict:
    floats = [float(value) for value in values]
    return {
        "n": len(floats),
        "mean": round(statistics.fmean(floats), 6),
        "sd": round(statistics.stdev(floats), 6) if len(floats) > 1 else None,
        "min": min(floats),
        "max": max(floats),
    }


def summarize(shard_paths: list) -> dict:
    if not shard_paths:
        raise ValueError("no score shards found; nothing to summarise")

    shards = []
    frozen = {}
    source_by_seed = {}
    by_cell = {}
    by_trait = {trait: [] for trait in TRAITS}
    by_item = {}
    phases = {}
    checkpoints = set()
    settings = set()
    scenarios = set()
    total_rows = 0

    for path in sorted(shard_paths):
        rows = read_jsonl(path)
        seeds = set()
        for index, row in enumerate(rows, start=1):
            validate_row(row, shard=path.name, index=index)
            for field in FROZEN_DIGEST_FIELDS:
                seen = frozen.setdefault(field, row[field])
                if seen != row[field]:
                    raise ValueError(
                        f"{path.name} row {index}: {field} differs from an earlier shard; the "
                        "summary would mix two frozen apparatus states"
                    )
            seeds.add(row["seed"])
            trait = row["trait"]
            score = float(row["trait_score"])
            by_cell.setdefault((row["setting"], row["checkpoint_id"], trait), []).append(score)
            by_trait[trait].append(score)
            for item_id, value in row["scores"].items():
                by_item.setdefault((trait, item_id), []).append(value)
            phases[row["source_phase"]] = phases.get(row["source_phase"], 0) + 1
            checkpoints.add(row["checkpoint_id"])
            settings.add(row["setting"])
            scenarios.add(row["source_scenario_id"])
            total_rows += 1

        if len(seeds) != 1:
            raise ValueError(f"{path.name} mixes seeds {sorted(seeds)}; one shard is one seed")
        seed = seeds.pop()
        if seed in source_by_seed:
            raise ValueError(f"seed {seed} appears in more than one shard")
        source_sha = rows[0]["source_shard_sha256"]
        for other_seed, other_sha in source_by_seed.items():
            if other_sha == source_sha:
                raise ValueError(
                    f"seed {seed} and seed {other_seed} were scored from the same generation "
                    f"shard {source_sha[:12]}...; that would double-count one sample"
                )
        source_by_seed[seed] = source_sha

        shards.append({
            "shard": path.name,
            "seed": seed,
            "rows": len(rows),
            "rater_calls": rows[0].get("rater_calls"),
            "scoring_elapsed_seconds": rows[0].get("scoring_elapsed_seconds"),
            "rater_calls_per_second": rows[0].get("rater_calls_per_second"),
            "source_shard_sha256": source_sha,
            "shard_sha256": sha256_file(path),
        })

    rows_per_shard = {entry["rows"] for entry in shards}

    return {
        "schema_version": 1,
        "tool": "experiments/xlpt/summarize_b0_rater.py",
        "kind": "b0-english-rater-descriptive-summary",
        "gate_decision": "NOT_DECIDED_BY_THIS_TOOL",
        "eligible_for_analysis": False,
        "uses_gpu": False,
        "trains_or_updates_weights": False,
        "note": (
            "Descriptive reduction of completed B0 English rater score shards. It reads the "
            "qualification surface only, decides no gate and writes no gate file. The sealed "
            "evaluation split is refused on sight."
        ),
        "frozen_digests": frozen,
        "shards": shards,
        "seeds_summarised": sorted(source_by_seed),
        "seed_count": len(source_by_seed),
        "rows_total": total_rows,
        "rows_per_shard_consistent": len(rows_per_shard) == 1,
        "rows_per_shard_observed": sorted(rows_per_shard),
        "source_phase_counts": dict(sorted(phases.items())),
        "distinct_checkpoints": sorted(checkpoints),
        "distinct_settings": sorted(settings),
        "distinct_scenarios": len(scenarios),
        "by_trait": {trait: _describe(values) for trait, values in by_trait.items() if values},
        "by_cell": {
            "|".join(str(part) for part in cell): _describe(values)
            for cell, values in sorted(by_cell.items(), key=lambda kv: tuple(str(p) for p in kv[0]))
        },
        "by_item": {
            f"{trait}|{item_id}": _describe(values)
            for (trait, item_id), values in sorted(by_item.items())
        },
        "parse_failure_note": (
            "run_rater_scoring.parse_item_score raises unless the rater returns exactly one "
            "integer 1..7, so a completed shard cannot contain an unparseable item and the rule "
            "K(b) midpoint-4 imputation did not fire in the shards summarised here."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise completed B0 English rater score shards.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--shard-dir", default="process/results/xlpt/b0-rater/qwen3-8b")
    ap.add_argument("--output-json", default="process/results/xlpt/B0_RATER_SUMMARY.json")
    ap.add_argument("--glob", default="seed-*.jsonl")
    args = ap.parse_args(argv)

    root = Path(args.project).resolve()
    shard_dir = (root / args.shard_dir).resolve()
    if not str(shard_dir).startswith(str(root)):
        raise ValueError("--shard-dir must stay inside the project")
    if not shard_dir.is_dir():
        raise ValueError(f"score-shard directory does not exist: {shard_dir}")

    payload = summarize(sorted(shard_dir.glob(args.glob)))
    payload["project"] = str(root)
    payload["shard_dir"] = str(shard_dir)

    output = (root / args.output_json).resolve()
    if not str(output).startswith(str(root)):
        raise ValueError("--output-json must stay inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)

    print(json.dumps({
        "output": str(output),
        "seeds": payload["seed_count"],
        "rows": payload["rows_total"],
        "gate_decision": payload["gate_decision"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
