#!/usr/bin/env python3
"""Build the Mistral-rater scoring queue for the Extraversion persona-pole shards.

The B0 rater units score the NEUTRAL arm against the v1 freeze.  A pole shard was
generated under a different phase and a different freeze, so it needs its own launch:
``--source-phase`` selects the phase registry entry in ``run_rater_scoring``, and the
prompt config / freeze receipt must be the v2 pair that actually produced the shard --
``run_rater_scoring`` refuses any other combination rather than relabeling provenance.

Each (checkpoint x pole) launch therefore gets its own review unit, derived from the
already-deployed Qwen rater manifest exactly the way ``build_rater_aya_llama_queue``
derives its per-arm units: same spec, same design card, same bindings, restricted to the
launch surface (``rater_score_parse``, ``english_trait_mean``, ``english_rate``).  The
deltas are the argv, the v2 frozen inputs in ``source_roots``, the arm's HF runtime
manifest and the pole rater contract test.  The kernel units are unchanged and are not
rebuilt here.

Codex review is waived by the operator (``.aris/OPERATOR_SKIP_FORMULA``), so
``pilot_budget`` skips ``verify_deployment`` and never opens the evidence directory: the
evidence pointer on each job is provenance for the reused code path, not a PASS receipt.
No verdict file is written by hand.

Pole scores are exploratory within-checkpoint contrasts.  They are analysis-eligible and
never gate-eligible: nothing here writes ``QUALIFICATION_GATE.results`` or freezes
``ELIGIBLE_PAIRS``.  Seed 20260829 has no shard on any arm and is never scheduled.

Run from the project root, after the pole generation queue has written its shards:
    python3 -m experiments.xlpt.build_pole_rater_queue
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.xlpt.build_pole_queue import POLE_SEEDS
from experiments.xlpt.build_rater_aya_llama_queue import (
    IDEA,
    KERNEL_ONLY_ROOTS,
    MAX_SOURCE_BYTES,
    PROJECT,
    PROMPT_DATE_TEST,
    PYTHON,
    QUALIFICATION_SEEDS,
    RATER_PATH,
    RATER_RUNTIME_MANIFEST,
    SCORING_FORMULAS,
    SCORING_TESTS,
    SOURCE_MANIFEST,
    packet_bytes,
    pending_state,
    queue,
    write_json,
)

# The poles reuse the qualification seed block exactly; a divergence would silently
# score a different set of shards than the ones that were generated.
assert POLE_SEEDS == QUALIFICATION_SEEDS, "pole seeds diverged from the qualification block"

PROMPT_CONFIG_V1 = "process/consumption/frozen/english_execution_v1.json"
FREEZE_RECEIPT_V1 = "process/consumption/frozen/english_execution_receipt_v1.json"
PROMPT_CONFIG_V2 = "process/consumption/frozen/english_execution_v2.json"
FREEZE_RECEIPT_V2 = "process/consumption/frozen/english_execution_receipt_v2.json"

POLE_CONTRACT_TEST = {
    "id": "pole_rater_contract",
    "python_args": ["-m", "experiments.xlpt.test_pole_rater_contract"],
    "test_files": ["experiments/xlpt/test_pole_rater_contract.py"],
    "timeout_seconds": 180,
}

# Mirrors run_rater_scoring.SOURCE_PHASES; test_pole_rater_contract pins that registry
# against the generator's own phase table, so these names are not re-derived here.
POLES = (
    ("b4-extraversion-high", "b4-extraversion-high-rater", "high"),
    ("b4-extraversion-low", "b4-extraversion-low-rater", "low"),
)

# The four registered checkpoints, with the runtime manifest each generation arm used.
ARMS = {
    "qwen3-8b": {
        "slug": "qwen3-8b",
        "runtime": "process/consumption/deployment/qwen3-runtime-manifest.json",
    },
    "aya-expanse-8b": {
        "slug": "aya-expanse-8b",
        "runtime": "process/consumption/deployment/aya-expanse-8b-runtime-manifest.json",
    },
    "Llama-3.1-8B-Instruct": {
        "slug": "Llama-3.1-8B-Instruct",
        "runtime": "process/consumption/deployment/Llama-3.1-8B-Instruct-runtime-manifest.json",
    },
    "gemma-3-12b-it": {
        "slug": "gemma-3-12b-it",
        "runtime": "process/consumption/deployment/gemma-3-12b-it-runtime-manifest.json",
    },
}

NOTES = (
    "Mistral AWQ int4 rater scoring for the Extraversion persona-pole shards: four "
    "registered checkpoints x high/low x the 23 qualification seeds (20260811-20260834 "
    "minus the stuck 20260829, which has no shard and is never retried). Each launch "
    "carries --source-phase plus the v2 freeze that produced the shard; run_rater_scoring "
    "refuses any other pairing. Pole scores are analysis-eligible and never gate-eligible: "
    "this queue writes nothing to QUALIFICATION_GATE.results and does not freeze "
    "ELIGIBLE_PAIRS. Operator skip_formula is in effect (no Codex review was run for these "
    "units); the manifest/evidence pointers are provenance for the reused rater code path, "
    "not a PASS receipt. Rater prompt date stays frozen at 2026-09-18, identical to the "
    "NEUTRAL B0 arms, so pole and NEUTRAL scores are mergeable. Runs only after the "
    "corresponding generation shards exist. Do not touch 17/40 on westb."
)


def pole_unit(source: dict, arm: str, phase: str) -> dict:
    """Derive this (checkpoint, pole) launch-surface review unit from the Qwen manifest."""
    argv = list(source["launch_argv"])
    if "--source-checkpoint" in argv or "--source-phase" in argv:
        raise ValueError("source manifest already binds a source checkpoint or phase")
    argv += [
        "--source-checkpoint", arm,
        "--source-phase", phase,
        "--prompt-config", PROMPT_CONFIG_V2,
        "--freeze-receipt", FREEZE_RECEIPT_V2,
    ]

    tests_by_id = {test["id"]: test for test in source["tests"]}
    for extra in (PROMPT_DATE_TEST, POLE_CONTRACT_TEST):
        if extra["id"] in tests_by_id:
            raise ValueError(f"source manifest already declares {extra['id']}")
        tests_by_id[extra["id"]] = dict(extra)
    scoring_test_ids = tuple(SCORING_TESTS) + (POLE_CONTRACT_TEST["id"],)
    kernel_test_ids = [t for t in tests_by_id if t not in scoring_test_ids]

    bindings = []
    for binding in source["bindings"]:
        if binding["formula_id"] not in SCORING_FORMULAS:
            continue
        test_ids = [t for t in binding["test_ids"] if t in scoring_test_ids]
        if not test_ids:
            raise ValueError(f"{binding['formula_id']} has no test on the launch surface")
        bindings.append({**json.loads(json.dumps(binding)), "test_ids": test_ids})
    if len(bindings) != len(SCORING_FORMULAS):
        raise ValueError("the launch-surface formulas are not all bound in the source manifest")

    roots = [
        r for r in source["source_roots"]
        if r not in KERNEL_ONLY_ROOTS
        and r not in (PROMPT_CONFIG_V1, FREEZE_RECEIPT_V1)
        and not any(r.endswith(f"test_{t}.py") for t in kernel_test_ids)
    ]
    roots += [
        PROMPT_CONFIG_V2,
        FREEZE_RECEIPT_V2,
        PROMPT_DATE_TEST["test_files"][0],
        POLE_CONTRACT_TEST["test_files"][0],
        ARMS[arm]["runtime"],
    ]

    return {
        "version": 1,
        "deployment_root": source["deployment_root"],
        "spec": source["spec"],
        "design_card": source["design_card"],
        "formula_ids": [f for f in source["formula_ids"] if f in SCORING_FORMULAS],
        "launch_argv": argv,
        "bindings": bindings,
        "source_roots": sorted(set(roots)),
        "tests": [tests_by_id[t] for t in tests_by_id if t in scoring_test_ids],
    }


def manifest_path(arm: str, short: str) -> str:
    return (f"process/consumption/deployment/"
            f"english-b4-{short}-rater-{ARMS[arm]['slug']}-scoring-manifest.json")


def evidence_path(arm: str, short: str) -> str:
    return (f"process/consumption/deployment/"
            f"english-b4-{short}-rater-{ARMS[arm]['slug']}-scoring")


def scoring_job(arm: str, phase: str, score_dir: str, short: str, seed: int) -> dict:
    slug = ARMS[arm]["slug"]
    return {
        "id": f"b4-{short}-rater-{slug}-s{seed}",
        "idea": IDEA,
        "cmd": (
            f"{PYTHON} -m experiments.xlpt.run_rater_scoring "
            f"--project {PROJECT} --seed {seed} "
            f"--source-checkpoint {arm} "
            f"--source-phase {phase} "
            f"--prompt-config {PROMPT_CONFIG_V2} "
            f"--freeze-receipt {FREEZE_RECEIPT_V2} "
            f"--rater-path {RATER_PATH} "
            f"--runtime-manifest {RATER_RUNTIME_MANIFEST}"
        ),
        "formula_manifest": manifest_path(arm, short),
        "formula_evidence": evidence_path(arm, short),
        "est_hours": 0.35,
        "expected_output": (
            f"{PROJECT}/process/results/xlpt/{score_dir}/{slug}/seed-{seed}.jsonl"
        ),
    }


def run(args) -> None:
    project = Path(args.project).resolve()
    source = json.loads((project / SOURCE_MANIFEST).read_text(encoding="utf-8"))

    sizes = {}
    for arm in ARMS:
        for phase, _score_dir, short in POLES:
            unit = pole_unit(source, arm, phase)
            size = packet_bytes(project, unit)
            if size >= MAX_SOURCE_BYTES:
                raise ValueError(
                    f"{arm} {phase} packet is {size} B, at or over the "
                    f"{MAX_SOURCE_BYTES} B ceiling the probe enforces"
                )
            sizes[f"{ARMS[arm]['slug']}-{short}"] = size
            if not args.check:
                write_json(project / manifest_path(arm, short), unit)

    phases = [
        {
            "name": f"b4-{short}-rater-{ARMS[arm]['slug']}",
            "depends_on": [],
            "jobs": [scoring_job(arm, phase, score_dir, short, seed) for seed in POLE_SEEDS],
        }
        for phase, score_dir, short in POLES
        for arm in ARMS
    ]
    payload = queue(phases, NOTES)

    out = Path(args.out)
    if not out.is_absolute():
        out = project / out
    path = out / "english_b4_poles_rater_queue.json"

    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        status = "identical" if current == json.dumps(payload, indent=2) + "\n" else "DRIFT"
    else:
        write_json(path, payload)
        write_json(out / "english_b4_poles_rater_queue_state.queued.json", pending_state(
            payload, path,
            "Pole generation shards present and bound by validate_source_shard under their "
            "own phase and the v2 freeze; rater prompt date frozen; operator skip_formula "
            "stamp present; Codex review waived by the operator, not passed",
        ))
        status = "written"

    print(json.dumps({
        "queue": str(path),
        "status": status,
        "seeds": len(POLE_SEEDS),
        "phases": [(p["name"], len(p["jobs"])) for p in phases],
        "total_jobs": sum(len(p["jobs"]) for p in phases),
        "est_hours_total": round(sum(j["est_hours"] for p in phases for j in p["jobs"]), 2),
        "packet_bytes": sizes,
    }, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="process/experiment_queue")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed queue instead of writing anything")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
