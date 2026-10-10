#!/usr/bin/env python3
"""Build the Mistral-rater scoring queue for the Chinese-surface (D = zh) shards.

Same construction as ``build_pole_rater_queue``: each (checkpoint x zh phase)
launch gets its own review unit derived from the already-deployed Qwen rater
manifest, restricted to the launch surface (``rater_score_parse``,
``english_trait_mean``, ``english_rate``).  The deltas are the argv, the **v3**
frozen inputs in ``source_roots``, the arm's HF runtime manifest and the pole
rater contract test.  ``run_rater_scoring`` refuses any pairing other than the
freeze that produced the shard, so the v3 pair is not optional here.

The instrument stays **English** on every row (R16-F1).  Scoring a Chinese
transcript with the English rubric is the registered design, not an oversight:
G3 instrument-language invariance is a separate, later comparison and is not
what this queue produces.

Only the analysis-eligible phases are scored: the four persona-pole arms.
``zh-b0-competence`` is the G1 competence screen and is read by the screen
reducer off the raw transcripts (response-language accuracy, instruction
compliance), not by the rubric rater; ``zh-b5-neutral-pzh`` is a reference arm
that is neither gate- nor analysis-eligible.  Feeding either through this queue
would produce rubric scores nothing downstream is allowed to read, so they are
not scheduled.

Pole scores are exploratory within-checkpoint contrasts: analysis-eligible and
never gate-eligible.  Nothing here writes ``QUALIFICATION_GATE.results`` or
freezes ``ELIGIBLE_PAIRS``.  Seed 20260829 has no shard on any arm and is never
scheduled.

Run from the project root, after the zh generation shards exist:
    python3 -m experiments.xlpt.build_zh_rater_queue
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.xlpt.build_pole_rater_queue import (
    ARMS,
    POLE_CONTRACT_TEST,
    PROMPT_CONFIG_V1,
    FREEZE_RECEIPT_V1,
)
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
from experiments.xlpt.build_zh_queue import ZH_SEEDS

assert ZH_SEEDS == QUALIFICATION_SEEDS, "zh seeds diverged from the qualification block"

PROMPT_CONFIG_V3 = "process/consumption/frozen/english_execution_v3.json"
FREEZE_RECEIPT_V3 = "process/consumption/frozen/english_execution_receipt_v3.json"

# (source phase, score directory, short job tag).  Mirrors run_rater_scoring.SOURCE_PHASES;
# test_pole_rater_contract pins that registry against the generator's own phase table,
# so these names are not re-derived here.  Analysis-eligible phases only.
ZH_SCORED = (
    ("zh-b4-extraversion-high", "zh-b4-extraversion-high-rater", "zh-b4-high"),
    ("zh-b4-extraversion-low", "zh-b4-extraversion-low-rater", "zh-b4-low"),
    ("zh-b5-extraversion-high-pzh", "zh-b5-extraversion-high-pzh-rater", "zh-b5-high"),
    ("zh-b5-extraversion-low-pzh", "zh-b5-extraversion-low-pzh-rater", "zh-b5-low"),
)

NOTES = (
    "Mistral AWQ int4 rater scoring for the Chinese-surface persona-pole shards: four registered "
    "checkpoints x four analysis-eligible zh phases x the 23 qualification seeds (20260811-20260834 "
    "minus the stuck 20260829, which has no shard and is never retried). Each launch carries "
    "--source-phase plus the v3 freeze that produced the shard; run_rater_scoring refuses any other "
    "pairing. The instrument stays ENGLISH on every row per R16-F1: scoring a Chinese transcript "
    "with the English rubric is the registered design, and G3 instrument-language invariance is a "
    "separate later comparison this queue does not produce. zh-b0-competence is the G1 screen and is "
    "read off the raw transcripts by the screen reducer, not by the rubric rater; zh-b5-neutral-pzh "
    "is a reference arm that is neither gate- nor analysis-eligible; neither is scheduled here. Pole "
    "scores are analysis-eligible and never gate-eligible: this queue writes nothing to "
    "QUALIFICATION_GATE.results and does not freeze ELIGIBLE_PAIRS. Operator skip_formula is in "
    "effect (no Codex review was run for these units); the manifest/evidence pointers are provenance "
    "for the reused rater code path, not a PASS receipt. Rater prompt date stays frozen at "
    "2026-09-18, identical to the English NEUTRAL and pole arms, so English and Chinese scores are "
    "mergeable. Runs only after the corresponding generation shards exist. Do not touch 17/40 on westb."
)


def zh_unit(source: dict, arm: str, phase: str) -> dict:
    """Derive this (checkpoint, zh phase) launch-surface review unit from the Qwen manifest."""
    argv = list(source["launch_argv"])
    if "--source-checkpoint" in argv or "--source-phase" in argv:
        raise ValueError("source manifest already binds a source checkpoint or phase")
    argv += [
        "--source-checkpoint", arm,
        "--source-phase", phase,
        "--prompt-config", PROMPT_CONFIG_V3,
        "--freeze-receipt", FREEZE_RECEIPT_V3,
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
        PROMPT_CONFIG_V3,
        FREEZE_RECEIPT_V3,
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
            f"{short}-rater-{ARMS[arm]['slug']}-scoring-manifest.json")


def evidence_path(arm: str, short: str) -> str:
    return f"process/consumption/deployment/{short}-rater-{ARMS[arm]['slug']}-scoring"


def scoring_job(arm: str, phase: str, score_dir: str, short: str, seed: int) -> dict:
    slug = ARMS[arm]["slug"]
    return {
        "id": f"{short}-rater-{slug}-s{seed}",
        "idea": IDEA,
        "cmd": (
            f"{PYTHON} -m experiments.xlpt.run_rater_scoring "
            f"--project {PROJECT} --seed {seed} "
            f"--source-checkpoint {arm} "
            f"--source-phase {phase} "
            f"--prompt-config {PROMPT_CONFIG_V3} "
            f"--freeze-receipt {FREEZE_RECEIPT_V3} "
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
        for phase, _score_dir, short in ZH_SCORED:
            unit = zh_unit(source, arm, phase)
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
            "name": f"{short}-rater-{ARMS[arm]['slug']}",
            "depends_on": [],
            "jobs": [scoring_job(arm, phase, score_dir, short, seed) for seed in ZH_SEEDS],
        }
        for phase, score_dir, short in ZH_SCORED
        for arm in ARMS
    ]
    payload = queue(phases, NOTES)

    out = Path(args.out)
    if not out.is_absolute():
        out = project / out
    path = out / "zh_surface_rater_queue.json"

    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        status = "identical" if current == json.dumps(payload, indent=2) + "\n" else "DRIFT"
    else:
        write_json(path, payload)
        write_json(out / "zh_surface_rater_queue_state.queued.json", pending_state(
            payload, path,
            "Chinese-surface generation shards present and bound by validate_source_shard under "
            "their own phase and the v3 freeze; zh-b0-competence G1 screen read and reported; "
            "rater prompt date frozen; operator skip_formula stamp present; Codex review waived by "
            "the operator, not passed",
        ))
        status = "written"

    print(json.dumps({
        "queue": str(path),
        "status": status,
        "seeds": len(ZH_SEEDS),
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
