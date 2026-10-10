#!/usr/bin/env python3
"""Build the Chinese-surface (D = zh) generation queue for bjb1.

Six phases x the four registered checkpoints x the 23 qualification seeds the
English arms actually used (20260811-20260834 minus the stuck 20260829, which is
never scheduled or retried).

The phases split into three blocks, and the queue keeps them in that order
because the later blocks are only worth running if the first one clears:

* ``zh-b0-competence`` -- the G1 competence screen for D = zh. Gate-eligible,
  never analysis-eligible. If a checkpoint misses response-language accuracy
  >= 0.95 or instruction compliance >= 0.90 on this arm it is out of competence
  for Chinese and its pole cells are reported as such rather than as an effect.
* ``zh-b4-extraversion-{high,low}`` -- English persona text, Chinese dialogue
  (P = en, D = zh): the cell that carries Delta_mis(en -> zh).
* ``zh-b5-neutral-pzh`` and ``zh-b5-extraversion-{high,low}-pzh`` -- the persona
  text itself rendered into Chinese (P = zh, D = zh): the matched-language pole,
  plus its NEUTRAL reference arm.

Every job is the same generator argv as that checkpoint's own B0 job with only
--phase, --prompt-config and --freeze-receipt changed, so each job carries the
B0 formula manifest/evidence of the identical code path it reuses.  With
.aris/OPERATOR_SKIP_FORMULA present, pilot_budget skips verify_deployment and
never opens that directory, so the pointer is provenance for the reused code
path and not a review PASS for the zh argv; the manifest notes say so.

The Chinese scenarios are the NLLB-200-3.3B single-stratum rendering frozen by
english_execution_v3.json.  That is one engine, not the registered translation
ensemble, and it is carried as a documented deviation everywhere the zh cells
are reported.  Nothing here writes QUALIFICATION_GATE.results or freezes
ELIGIBLE_PAIRS.

Regenerates process/experiment_queue/zh_surface_queue.json byte-for-byte:
    python3 -m experiments.xlpt.build_zh_queue
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.xlpt.build_pole_queue import (
    DEPLOYMENT_ROOT,
    IDEA,
    MODELS,
    POLE_SEEDS,
    PROJECT,
    PYTHON,
)

PROMPT_CONFIG = "process/consumption/frozen/english_execution_v3.json"
FREEZE_RECEIPT = "process/consumption/frozen/english_execution_receipt_v3.json"

ZH_SEEDS = POLE_SEEDS

# (generator phase, queue phase name, short job tag).  Ordered: the competence
# screen first, then the P = en poles, then the P = zh block with its own
# NEUTRAL reference arm.
ZH_PHASES = (
    ("zh-b0-competence", "zh-b0-competence-generation", "zh-b0"),
    ("zh-b4-extraversion-high", "zh-b4-extraversion-high-generation", "zh-b4-high"),
    ("zh-b4-extraversion-low", "zh-b4-extraversion-low-generation", "zh-b4-low"),
    ("zh-b5-neutral-pzh", "zh-b5-neutral-pzh-generation", "zh-b5-neutral"),
    ("zh-b5-extraversion-high-pzh", "zh-b5-extraversion-high-pzh-generation", "zh-b5-high"),
    ("zh-b5-extraversion-low-pzh", "zh-b5-extraversion-low-pzh-generation", "zh-b5-low"),
)

# The screen has to be scored and read before the pole blocks are worth the GPU
# time, so the later phases declare it as a dependency rather than racing it.
DEPENDS_ON = {
    "zh-b0-competence-generation": [],
}

NOTES = (
    "Chinese dialogue surface (D = zh) on the frozen qualification scenarios: six phases x the "
    "four registered checkpoints x 23 seeds (20260811-20260834 minus the stuck 20260829, which is "
    "never retried). zh-b0-competence is the G1 competence screen for Chinese and is the only "
    "gate-eligible phase here; the b4 phases keep the English persona text and ask for Chinese "
    "replies (P = en, D = zh); the b5 phases use the Chinese rendering of the persona text itself "
    "(P = zh, D = zh) plus its NEUTRAL reference arm. The Chinese scenarios and persona texts are "
    "the NLLB-200-3.3B (rev 1a07f7d1) single-stratum rendering frozen in english_execution_v3.json: "
    "one engine, not the registered translation ensemble, carried as a documented deviation "
    "wherever these cells are reported. Persona texts are project-authored behavioral descriptions; "
    "no BFI-2/TIPI item text exists in this project and none is reconstructed. These are exploratory "
    "within-checkpoint contrasts: nothing here writes QUALIFICATION_GATE.results or freezes "
    "ELIGIBLE_PAIRS. Formula Codex review is skipped under .aris/OPERATOR_SKIP_FORMULA; each job "
    "carries the reviewed B0 manifest/evidence of the identical generator code path it reuses, which "
    "is a provenance pointer and not a PASS receipt for the zh argv. Qwen keeps the project-owned "
    "captured forward (run_transformers_generation) so it stays comparable with its own English arms. "
    "Do not touch 17/40 on westb. Do not rerun Qwen English B1/B2."
)


def generation_job(model_id: str, phase: str, short: str, seed: int) -> dict:
    spec = MODELS[model_id]
    cmd = [
        PYTHON, "-m", spec["module"],
        "--project", PROJECT,
        "--phase", phase,
        "--model-path", spec["path"],
        "--model-id", model_id,
    ]
    if spec["runtime"]:
        cmd += ["--runtime-manifest", spec["runtime"]]
    cmd += [
        "--prompt-config", PROMPT_CONFIG,
        "--freeze-receipt", FREEZE_RECEIPT,
        "--seed", str(seed),
    ]
    return {
        "id": f"{short}-{spec['slug']}-s{seed}",
        "idea": IDEA,
        "cmd": " ".join(cmd),
        "formula_manifest": spec["manifest"],
        "formula_evidence": spec["evidence"],
        "est_hours": spec["est_hours"],
        "expected_output": f"{PROJECT}/process/results/xlpt/{phase}/{spec['slug']}/seed-{seed}.jsonl",
    }


def manifest() -> dict:
    screen = ZH_PHASES[0][1]
    return {
        "project": IDEA,
        "cwd": DEPLOYMENT_ROOT,
        "conda": "base",
        "activate_conda": False,
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 1000,
        "oom_retry": {"delay": 120, "max_attempts": 2},
        "notes": NOTES,
        "phases": [
            {
                "name": queue_name,
                "depends_on": DEPENDS_ON.get(queue_name, [screen]),
                "jobs": [
                    generation_job(model_id, phase, short, seed)
                    for model_id in MODELS
                    for seed in ZH_SEEDS
                ],
            }
            for phase, queue_name, short in ZH_PHASES
        ],
    }


def slice_manifest(payload: dict, phase_names, satisfied=()) -> dict:
    """Keep one or more phases, with dependencies dropped only when discharged.

    The competence screen decides nothing inside the queue manager: a human reads
    ZH_G1_COMPETENCE_SCREEN.json and only then is the pole block worth GPU time.
    Running the screen phase from its own manifest makes that stop explicit, and
    leaves the six-phase manifest byte-identical as the record of the whole plan.

    A dependency outside the slice is an error unless it is named in
    ``satisfied``, which is the operator asserting that phase has already been
    generated and its screen read.  The assertion is recorded in the slice's own
    notes, so a manifest that skipped the screen cannot look like one that
    cleared it.
    """
    names = [p["name"] for p in payload["phases"]]
    if isinstance(phase_names, str):
        phase_names = [phase_names]
    phase_names = list(phase_names)
    satisfied = list(satisfied)
    for name in phase_names + satisfied:
        if name not in names:
            raise SystemExit(f"unknown phase {name!r}; the manifest has {names}")
    kept = [dict(p) for p in payload["phases"] if p["name"] in phase_names]
    inside = set(phase_names)
    for phase in kept:
        missing = [d for d in phase["depends_on"] if d not in inside and d not in satisfied]
        if missing:
            raise SystemExit(
                f"{phase['name']} depends on {missing}; run those phases first, or "
                "declare them with --satisfied once their screen has been read, "
                "rather than slicing the dependency away"
            )
        # A dependency the state file has never heard of is never "completed", so
        # the manager would hold the phase pending forever.  Drop the discharged
        # ones from the edge list; the notes above keep the assertion on record.
        phase["depends_on"] = [d for d in phase["depends_on"] if d not in satisfied]
    if len(phase_names) == 1:
        # Byte-identical to the note the single-phase slicer has always written,
        # so the screen manifest already on the worker keeps verifying.
        note = f"SLICE of zh_surface_queue.json: the {phase_names[0]} phase only. "
    else:
        note = f"SLICE of zh_surface_queue.json: the {', '.join(phase_names)} phases only. "
    if satisfied:
        note += (
            "Dependencies declared already generated and read by the operator: "
            f"{', '.join(satisfied)}. "
        )
    sliced = dict(payload)
    sliced["phases"] = kept
    sliced["notes"] = note + payload["notes"]
    return sliced


def run(args) -> int:
    payload = manifest()
    name = "zh_surface_queue.json"
    if args.only_phase:
        phases = [p.strip() for p in args.only_phase.split(",") if p.strip()]
        satisfied = [p.strip() for p in (args.satisfied or "").split(",") if p.strip()]
        payload = slice_manifest(payload, phases, satisfied)
        if len(phases) == 1:
            name = f"{phases[0].replace('-', '_')}_queue.json"
        elif not args.slice_name:
            raise SystemExit("a multi-phase slice needs --slice-name for its file name")
        else:
            name = f"{args.slice_name.replace('-', '_')}_queue.json"
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(args.project).resolve() / out
    path = out / name
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        status = "identical" if current == text else "DRIFT"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        status = "written"
    print(json.dumps({
        "manifest": str(path),
        "status": status,
        "seeds": len(ZH_SEEDS),
        "phases": [(p["name"], len(p["jobs"])) for p in payload["phases"]],
        "total_jobs": sum(len(p["jobs"]) for p in payload["phases"]),
        "est_hours_total": round(sum(j["est_hours"] for p in payload["phases"] for j in p["jobs"]), 2),
    }, ensure_ascii=False))
    return 1 if status == "DRIFT" else 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="process/experiment_queue")
    ap.add_argument("--only-phase", default=None,
                    help="emit a slice of one phase (e.g. zh-b0-competence-generation) under its "
                         "own filename, or a comma-separated set of phases with --slice-name, "
                         "leaving the six-phase manifest untouched")
    ap.add_argument("--satisfied", default=None,
                    help="comma-separated phases outside the slice whose generation is already "
                         "done and whose screen has been read; recorded in the slice notes")
    ap.add_argument("--slice-name", default=None,
                    help="file stem for a multi-phase slice")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed manifest instead of rewriting it")
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
