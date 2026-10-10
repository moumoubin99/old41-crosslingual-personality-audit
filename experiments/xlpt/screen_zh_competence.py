#!/usr/bin/env python3
"""Reduce the ``zh-b0-competence`` generation shards to the G1 competence screen for D = zh.

G1 asks one question per checkpoint: on the 50-prompt qualification sample, does this model
actually hold a Chinese customer-support conversation when asked to? It is a property of the
transcripts, not of the rubric scores, so it is computed here from the generation shards and
not from anything the rater produced.

Two quantities, both registered in the design card:

* **response-language accuracy** -- the fraction of agent turns written in Chinese, where a
  turn counts as Chinese when Han characters make up at least ``--han-ratio`` of its
  script-bearing characters (Han + Latin letters). Punctuation, digits, whitespace and
  emoji are script-neutral and are excluded from the denominator, so "订单 #4471-8820" is
  not penalised for carrying an order number. Threshold: >= 0.95.
* **instruction compliance** -- the fraction of scenarios that produced exactly ``--turns``
  non-empty agent turns (L = 3 is mandatory, R16-F3). A truncated, empty or over-long
  dialogue is non-compliant regardless of what it says. Threshold: >= 0.90.

A checkpoint that misses either threshold is reported ``OUT_OF_COMPETENCE`` for Chinese:
its Chinese pole cells are then reported as a competence failure rather than as a persona
effect, because a model that cannot hold the conversation cannot show a persona effect in it.
That reporting rule is what this screen exists to support; the screen itself decides no
qualification gate, writes no ``QUALIFICATION_GATE.results`` and freezes no
``ELIGIBLE_PAIRS``.

Deterministic, CPU-only, read-only with respect to every shard.

    python3 -m experiments.xlpt.screen_zh_competence --project .
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

PHASE = "zh-b0-competence"
SETTING = "NEUTRAL_DZH"
SCHEMA_VERSION = 2
DEFAULT_OUTPUT = "process/results/xlpt/ZH_G1_COMPETENCE_SCREEN.json"
LANGUAGE_THRESHOLD = 0.95
COMPLIANCE_THRESHOLD = 0.90
MANDATORY_TURNS = 3
HAN_RATIO = 0.5

HAN_BLOCKS = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # Extension A
    (0xF900, 0xFAFF),    # Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Extension B
)


def is_han(ch: str) -> bool:
    point = ord(ch)
    return any(low <= point <= high for low, high in HAN_BLOCKS)


def script_ratio(text: str) -> tuple[float, int]:
    """Han share of the script-bearing characters, and how many there are."""
    han = sum(1 for ch in text if is_han(ch))
    latin = sum(1 for ch in text if ch.isalpha() and not is_han(ch))
    total = han + latin
    return (han / total if total else 0.0), total


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(phase_root: Path):
    """Load every shard under the phase directory, one checkpoint per subdirectory."""
    shards = []
    for shard in sorted(phase_root.glob("*/seed-*.jsonl")):
        rows = read_jsonl(shard)
        if not rows:
            raise ValueError(f"{shard}: empty shard")
        checkpoint = shard.parent.name
        seeds = {row.get("seed") for row in rows}
        if len(seeds) != 1:
            raise ValueError(f"{shard}: holds more than one seed")
        for row in rows:
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{shard}: unexpected schema_version {row.get('schema_version')!r}")
            if row.get("phase") != PHASE or row.get("setting") != SETTING:
                raise ValueError(f"{shard}: row is not a {PHASE}/{SETTING} row")
            # The directory is the queue's slug and the record carries --model-id.
            # For the four registered checkpoints those differ only in case
            # (qwen3-8b / Qwen3-8B), so the identity check is case-insensitive;
            # a different checkpoint still fails.
            if str(row.get("checkpoint_id")).lower() != checkpoint.lower():
                raise ValueError(f"{shard}: checkpoint_id disagrees with the shard directory")
            if not row.get("eligible_for_gate") or row.get("eligible_for_analysis"):
                raise ValueError(f"{shard}: the G1 screen surface is gate-eligible and analysis-ineligible")
            if row.get("dialogue_language") != "Chinese":
                raise ValueError(f"{shard}: row is not recorded as D = zh")
        shards.append({"path": shard, "checkpoint": checkpoint, "seed": seeds.pop(), "rows": rows})
    if not shards:
        raise ValueError(f"no {PHASE} shards under {phase_root}")
    return shards


def frozen_apparatus(shards) -> dict:
    keys = ("prompt_config_sha256", "freeze_receipt_sha256")
    seen = {key: {row.get(key) for s in shards for row in s["rows"]} for key in keys}
    for key, values in seen.items():
        if len(values) != 1:
            raise ValueError(f"shards disagree on {key}; this is not one frozen apparatus")
    return {key: values.pop() for key, values in seen.items()}


def score_shard(shard, han_ratio: float, turns_required: int) -> dict:
    turns_total = turns_chinese = 0
    scenarios_compliant = 0
    non_chinese = []
    for row in shard["rows"]:
        turns = [str(t) for t in row.get("agent_turns", [])]
        non_empty = [t for t in turns if t.strip()]
        if len(turns) == turns_required and len(non_empty) == turns_required:
            scenarios_compliant += 1
        for index, turn in enumerate(non_empty):
            turns_total += 1
            ratio, bearing = script_ratio(turn)
            if bearing and ratio >= han_ratio:
                turns_chinese += 1
            else:
                non_chinese.append({
                    "scenario_id": row["scenario_id"],
                    "turn_index": index,
                    "han_ratio": round(ratio, 4),
                    "script_chars": bearing,
                    "excerpt": turn[:160],
                })
    return {
        "checkpoint": shard["checkpoint"],
        "seed": shard["seed"],
        "scenarios": len(shard["rows"]),
        "scenarios_compliant": scenarios_compliant,
        "instruction_compliance": scenarios_compliant / len(shard["rows"]),
        "agent_turns": turns_total,
        "agent_turns_chinese": turns_chinese,
        "response_language_accuracy": (turns_chinese / turns_total) if turns_total else 0.0,
        "non_chinese_turns": non_chinese,
    }


def summarize_checkpoint(name: str, per_seed: list[dict]) -> dict:
    turns = sum(s["agent_turns"] for s in per_seed)
    chinese = sum(s["agent_turns_chinese"] for s in per_seed)
    scenarios = sum(s["scenarios"] for s in per_seed)
    compliant = sum(s["scenarios_compliant"] for s in per_seed)
    accuracy = (chinese / turns) if turns else 0.0
    compliance = (compliant / scenarios) if scenarios else 0.0
    failures = []
    if accuracy < LANGUAGE_THRESHOLD:
        failures.append(f"response_language_accuracy {accuracy:.4f} < {LANGUAGE_THRESHOLD}")
    if compliance < COMPLIANCE_THRESHOLD:
        failures.append(f"instruction_compliance {compliance:.4f} < {COMPLIANCE_THRESHOLD}")
    return {
        "checkpoint": name,
        "seeds": sorted(s["seed"] for s in per_seed),
        "seed_count": len(per_seed),
        "scenarios": scenarios,
        "agent_turns": turns,
        "response_language_accuracy": round(accuracy, 6),
        "response_language_accuracy_seed_mean": round(
            statistics.fmean(s["response_language_accuracy"] for s in per_seed), 6),
        "instruction_compliance": round(compliance, 6),
        "instruction_compliance_seed_mean": round(
            statistics.fmean(s["instruction_compliance"] for s in per_seed), 6),
        "screen": "IN_COMPETENCE" if not failures else "OUT_OF_COMPETENCE",
        "failures": failures,
        "per_seed": [{k: v for k, v in s.items() if k != "non_chinese_turns"} for s in per_seed],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".")
    ap.add_argument("--phase-root", default=f"process/results/xlpt/{PHASE}")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--expect-shards", type=int, default=0,
                    help="refuse to reduce unless exactly this many shards are present")
    ap.add_argument("--han-ratio", type=float, default=HAN_RATIO)
    ap.add_argument("--turns", type=int, default=MANDATORY_TURNS)
    ap.add_argument("--max-examples", type=int, default=20,
                    help="how many non-Chinese turns to carry into the report per checkpoint")
    args = ap.parse_args()

    root = Path(args.project).resolve()
    phase_root = (root / args.phase_root).resolve()
    if not str(phase_root).startswith(str(root)):
        raise ValueError("--phase-root must stay inside the project")

    shards = collect(phase_root)
    if args.expect_shards and len(shards) != args.expect_shards:
        raise ValueError(
            f"{len(shards)} {PHASE} shards present, expected {args.expect_shards}; "
            "refusing to screen an incomplete block"
        )
    frozen = frozen_apparatus(shards)

    scored = [score_shard(s, args.han_ratio, args.turns) for s in shards]
    by_checkpoint: dict[str, list[dict]] = {}
    for entry in scored:
        by_checkpoint.setdefault(entry["checkpoint"], []).append(entry)

    checkpoints = [summarize_checkpoint(name, sorted(v, key=lambda e: e["seed"]))
                   for name, v in sorted(by_checkpoint.items())]
    examples = {
        name: [ex for entry in sorted(v, key=lambda e: e["seed"])
               for ex in entry["non_chinese_turns"]][: args.max_examples]
        for name, v in sorted(by_checkpoint.items())
    }

    payload = {
        "screen": "G1_competence",
        "surface": "D = zh (Chinese dialogue) on the frozen Chinese qualification packet",
        "phase": PHASE,
        "setting": SETTING,
        "project": str(root),
        "phase_root": str(phase_root),
        "thresholds": {
            "response_language_accuracy": LANGUAGE_THRESHOLD,
            "instruction_compliance": COMPLIANCE_THRESHOLD,
            "turn_is_chinese_when_han_ratio_at_least": args.han_ratio,
            "mandatory_agent_turns": args.turns,
        },
        "language_identification": (
            "Han-character ratio over script-bearing characters (Han + Latin letters); "
            "digits, punctuation and whitespace are script-neutral and excluded"
        ),
        "frozen_apparatus": frozen,
        "shard_count": len(shards),
        "checkpoints": checkpoints,
        "non_chinese_examples": examples,
        "gate_decision": "NOT_DECIDED_BY_THIS_TOOL",
        "limitations": [
            "The Chinese scenarios are a single-stratum NLLB-200-3.3B rendering, not the "
            "registered translation ensemble; a competence failure here can be a rendering "
            "artefact as well as a model limitation, and both readings stay open.",
            "Han-ratio language identification cannot distinguish Chinese from Japanese kanji "
            "text; it is a script screen, not a full language identifier.",
            "Instruction compliance here is turn-structure compliance only. Task fidelity is "
            "scored by the English behavior rubric on the pole arms, not by this screen.",
            "This screen writes no QUALIFICATION_GATE.results and freezes no ELIGIBLE_PAIRS.",
        ],
    }

    output = (root / (args.output_json or DEFAULT_OUTPUT)).resolve()
    if not str(output).startswith(str(root)):
        raise ValueError("--output-json must stay inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)

    print(json.dumps({
        "output": str(output),
        "shards": len(shards),
        "checkpoints": {c["checkpoint"]: c["screen"] for c in checkpoints},
        "response_language_accuracy": {c["checkpoint"]: c["response_language_accuracy"]
                                       for c in checkpoints},
        "instruction_compliance": {c["checkpoint"]: c["instruction_compliance"]
                                   for c in checkpoints},
        "gate_decision": payload["gate_decision"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
