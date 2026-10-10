#!/usr/bin/env python3
"""Build the Mistral-rater scoring deployment units and queue for the aya/llama B0 arms.

The Qwen arm was scored under ``english-b0-rater-v11-manifest.json``.  Scoring a second
and third checkpoint changes exactly one thing in the launch: ``--source-checkpoint``.
``deployment_formula_check.snapshot`` only tolerates a differing ``--seed`` between the
manifest's ``launch_argv`` and the actual launch, so each arm needs its own manifest
rather than one shared unit with a "representative" argv.

Derived from v11 without editing v11: same spec, design card and bindings.  The deltas are
(a) the arm's ``launch_argv``, (b) ``source_roots`` gains the new prompt/registry contract
test and the arm's HF runtime manifest, and (c) ``tests`` gains ``rater_prompt_date_freeze``.

The v11 unit can no longer be probed as one packet: its declared sources are now 403 KB,
over ``deployment_formula_check.MAX_SOURCE_BYTES`` (400 KB), and the files this arm adds
push it further.  Following the tool's own instruction to split the review unit, each arm
gets two units whose tests are disjoint and whose formula coverage is derived from the v11
execution traces, not guessed:

* ``-scoring`` -- the launch surface: the three formulas that ``rater_launch_integration``
  actually exercises (``rater_score_parse``, ``english_trait_mean``, ``english_rate``),
  plus ``rater_module_smoke`` and the new ``rater_prompt_date_freeze``.  This is the unit
  the queue jobs cite, because its ``launch_argv`` is the job's exact command.
* ``-kernel`` -- the remaining 23 rater-kernel formulas, carried by
  ``rater_runtime_contract``, ``rater_production_closure`` and ``rater_cuda_witness``,
  which the v11 traces show call every one of their bound symbols.

Neither unit is Codex-reviewed: the operator waived that review.  The probe evidence is
real and deterministic; no verdict file may be written by hand.

Seed 20260829 is excluded on operator instruction: it never produced a generation shard
on any arm and must not be retried.

The fourth registered checkpoint ``gemma-3-12b-it`` was generated later, on bjb1, and is
built here the same way.  It gets its own queue file (``english_b0_rater_gemma_queue.json``)
so the aya/llama queue and the scheduler state that records their completed runs are left
untouched; the aya/llama files are only written if they are absent.

Run from the project root:
    python3 -m experiments.xlpt.build_rater_aya_llama_queue
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = "<LOCAL_PATH>/41-2026-09-05-aris-3"
DEPLOYMENT_ROOT = "<LOCAL_PATH>/project41"
PYTHON = "<LOCAL_PATH>/python3.12"
IDEA = "xling-induced-personality-transfer-matrix"
RATER_PATH = (
    "<LOCAL_PATH>/models"
    "OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"
)
RATER_RUNTIME_MANIFEST = "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json"
SOURCE_MANIFEST = "process/consumption/deployment/english-b0-rater-v11-manifest.json"
# deployment_formula_check.MAX_SOURCE_BYTES: the probe refuses any larger packet.
MAX_SOURCE_BYTES = 400_000

# The generation arm 20260829 is the one seed that never completed; it stays unscheduled.
QUALIFICATION_SEEDS = tuple(s for s in range(20260811, 20260835) if s != 20260829)

PROMPT_DATE_TEST = {
    "id": "rater_prompt_date_freeze",
    "python_args": ["-m", "experiments.xlpt.test_rater_prompt_date_freeze"],
    "test_files": ["experiments/xlpt/test_rater_prompt_date_freeze.py"],
    "timeout_seconds": 180,
}

# Tests of the launch surface; everything else stays with the kernel unit.
SCORING_TESTS = ("rater_launch_integration", "rater_module_smoke", PROMPT_DATE_TEST["id"])
# The three formulas the v11 traces show rater_launch_integration fully exercises.
SCORING_FORMULAS = ("rater_score_parse", "english_trait_mean", "english_rate")

# Source files the kernel closure needs and the launch surface does not.
KERNEL_ONLY_ROOTS = (
    "experiments/xlpt/mistral_awq_reference.py",
    "experiments/xlpt/rater_cuda_conformance.py",
    "experiments/xlpt/rater_production_closure.py",
    "process/consumption/deployment/mistral-rater-semantic-closure-v1.json",
)

ARMS = {
    "aya-expanse-8b": {
        "slug": "aya-expanse-8b",
        "runtime": "process/consumption/deployment/aya-expanse-8b-runtime-manifest.json",
        "scoring_manifest": "process/consumption/deployment/english-b0-rater-v13-aya-scoring-manifest.json",
        "scoring_evidence": "process/consumption/deployment/english-b0-rater-v13-aya-scoring",
        "kernel_manifest": "process/consumption/deployment/english-b0-rater-v13-aya-kernel-manifest.json",
        "kernel_evidence": "process/consumption/deployment/english-b0-rater-v13-aya-kernel",
    },
    "Llama-3.1-8B-Instruct": {
        "slug": "Llama-3.1-8B-Instruct",
        "runtime": "process/consumption/deployment/Llama-3.1-8B-Instruct-runtime-manifest.json",
        "scoring_manifest": "process/consumption/deployment/english-b0-rater-v13-llama-scoring-manifest.json",
        "scoring_evidence": "process/consumption/deployment/english-b0-rater-v13-llama-scoring",
        "kernel_manifest": "process/consumption/deployment/english-b0-rater-v13-llama-kernel-manifest.json",
        "kernel_evidence": "process/consumption/deployment/english-b0-rater-v13-llama-kernel",
    },
    # Fourth registered checkpoint, generated on bjb1 after the aya/llama arms were
    # scored.  Same rater, same prompt freeze; only --source-checkpoint differs.
    "gemma-3-12b-it": {
        "slug": "gemma-3-12b-it",
        "runtime": "process/consumption/deployment/gemma-3-12b-it-runtime-manifest.json",
        "scoring_manifest": "process/consumption/deployment/english-b0-rater-v13-gemma-scoring-manifest.json",
        "scoring_evidence": "process/consumption/deployment/english-b0-rater-v13-gemma-scoring",
        "kernel_manifest": "process/consumption/deployment/english-b0-rater-v13-gemma-kernel-manifest.json",
        "kernel_evidence": "process/consumption/deployment/english-b0-rater-v13-gemma-kernel",
    },
}

# The already-executed arms: their queue manifest and scheduler state record what really
# ran and must never be rewritten by a later build.
EXECUTED_ARMS = ("aya-expanse-8b", "Llama-3.1-8B-Instruct")
PENDING_ARMS = ("gemma-3-12b-it",)


def arm_manifests(source: dict, arm: str) -> dict:
    """Split the Qwen-arm manifest into this arm's scoring and kernel review units."""
    argv = list(source["launch_argv"])
    if "--source-checkpoint" in argv:
        raise ValueError("source manifest already binds a source checkpoint")
    argv = argv + ["--source-checkpoint", arm]
    if any(test["id"] == PROMPT_DATE_TEST["id"] for test in source["tests"]):
        raise ValueError("source manifest already declares the prompt-date test")

    scoring_bindings = [b for b in source["bindings"] if b["formula_id"] in SCORING_FORMULAS]
    kernel_bindings = [b for b in source["bindings"] if b["formula_id"] not in SCORING_FORMULAS]
    if len(scoring_bindings) != len(SCORING_FORMULAS):
        raise ValueError("the launch-surface formulas are not all bound in the source manifest")

    def restrict(bindings, allowed):
        out = []
        for binding in bindings:
            test_ids = [t for t in binding["test_ids"] if t in allowed]
            if not test_ids:
                raise ValueError(f"{binding['formula_id']} has no test in this review unit")
            out.append({**json.loads(json.dumps(binding)), "test_ids": test_ids})
        return out

    tests_by_id = {test["id"]: test for test in source["tests"]}
    tests_by_id[PROMPT_DATE_TEST["id"]] = dict(PROMPT_DATE_TEST)
    kernel_test_ids = [t for t in tests_by_id if t not in SCORING_TESTS]

    roots = list(source["source_roots"])
    scoring_roots = sorted(
        [r for r in roots
         if r not in KERNEL_ONLY_ROOTS
         and not any(r.endswith(f"test_{t}.py") for t in kernel_test_ids)]
        + [PROMPT_DATE_TEST["test_files"][0], ARMS[arm]["runtime"]]
    )
    kernel_roots = sorted(
        r for r in roots if not any(r.endswith(f"test_{t}.py") for t in SCORING_TESTS)
    )

    def unit(formula_ids, bindings, test_ids, source_roots):
        return {
            "version": 1,
            "deployment_root": source["deployment_root"],
            "launch_argv": argv,
            "source_roots": source_roots,
            "spec": source["spec"],
            "design_card": source["design_card"],
            "formula_ids": list(formula_ids),
            "bindings": bindings,
            "tests": [tests_by_id[t] for t in test_ids],
        }

    scoring = unit(
        [f for f in source["formula_ids"] if f in SCORING_FORMULAS],
        restrict(scoring_bindings, set(SCORING_TESTS)),
        [t for t in tests_by_id if t in SCORING_TESTS],
        scoring_roots,
    )
    kernel = unit(
        [f for f in source["formula_ids"] if f not in SCORING_FORMULAS],
        restrict(kernel_bindings, set(kernel_test_ids)),
        kernel_test_ids,
        kernel_roots,
    )
    return {"scoring": scoring, "kernel": kernel}


def scoring_job(arm: str, seed: int) -> dict:
    spec = ARMS[arm]
    return {
        "id": f"b0-rater-{spec['slug']}-s{seed}",
        "idea": IDEA,
        "cmd": (
            f"{PYTHON} -m experiments.xlpt.run_rater_scoring "
            f"--project {PROJECT} --seed {seed} "
            f"--source-checkpoint {arm} "
            f"--rater-path {RATER_PATH} "
            f"--runtime-manifest {RATER_RUNTIME_MANIFEST}"
        ),
        "formula_manifest": spec["scoring_manifest"],
        "formula_evidence": spec["scoring_evidence"],
        "est_hours": 0.35,
        "expected_output": (
            f"{PROJECT}/process/results/xlpt/b0-rater/{spec['slug']}/seed-{seed}.jsonl"
        ),
    }


def queue(phases, notes: str) -> dict:
    return {
        "project": IDEA,
        "cwd": DEPLOYMENT_ROOT,
        "conda": "base",
        "activate_conda": False,
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 1000,
        "oom_retry": {"delay": 120, "max_attempts": 2},
        "notes": notes,
        "phases": phases,
    }


def pending_state(manifest, manifest_path: Path, release_condition: str) -> dict:
    jobs = []
    phases = []
    for phase in manifest["phases"]:
        phases.append({
            "name": phase["name"],
            "depends_on": phase.get("depends_on", []),
            "status": "pending",
        })
        for job in phase["jobs"]:
            jobs.append({
                **job,
                "phase": phase["name"],
                "status": "pending",
                "gpu": None,
                "screen_name": None,
                "pid": None,
                "attempts": 0,
                "started": None,
                "completed": None,
                "error": None,
            })
    return {
        "meta": {
            "project": IDEA,
            "started": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "manifest_path": manifest_path.name,
            "release_condition": release_condition,
        },
        "phases": phases,
        "jobs": jobs,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def packet_bytes(project: Path, unit: dict) -> int:
    """Byte size of the review packet the probe will capture for this unit."""
    total = 0
    for rel in unit["source_roots"]:
        path = project / rel
        if not path.exists():
            raise FileNotFoundError(f"declared source root is missing: {rel}")
        total += len(path.read_text(encoding="utf-8").encode("utf-8"))
    return total


def run(args) -> None:
    project = Path(args.project).resolve()
    source = json.loads((project / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    sizes = {}
    for arm, spec in ARMS.items():
        units = arm_manifests(source, arm)
        for kind, unit in units.items():
            size = packet_bytes(project, unit)
            if size >= MAX_SOURCE_BYTES:
                raise ValueError(
                    f"{arm} {kind} packet is {size} B, at or over the "
                    f"{MAX_SOURCE_BYTES} B ceiling the probe enforces"
                )
            sizes[f"{spec['slug']}-{kind}"] = size
            write_json(project / spec[f"{kind}_manifest"], unit)

    def phases_for(arms):
        return [
            {
                "name": f"b0-english-rater-{ARMS[arm]['slug']}",
                "depends_on": [],
                "jobs": [scoring_job(arm, seed) for seed in QUALIFICATION_SEEDS],
            }
            for arm in arms
        ]

    out = Path(args.out)
    if not out.is_absolute():
        out = project / out

    executed_phases = phases_for(EXECUTED_ARMS)
    executed_queue = queue(executed_phases, (
        "Mistral AWQ int4 rater scoring for the aya and llama English B0 arms on bjb1. "
        "Operator skip_formula (no Codex review was run for these units); "
        "QUALIFICATION_GATE.results stay null and ELIGIBLE_PAIRS stays unfrozen. "
        "Seed 20260829 has no source shard and must not be retried. "
        "Rater prompt date frozen to 2026-09-18 to match the Qwen shards. "
        "Do not touch 17/40 on westb."
    ))
    executed_path = out / "english_b0_rater_aya_llama_queue.json"
    # These two arms have already been scored; the manifest and state on disk record
    # what actually ran and are never regenerated.
    executed_written = not executed_path.exists()
    if executed_written:
        write_json(executed_path, executed_queue)
        write_json(out / "english_b0_rater_aya_llama_queue_state.queued.json", pending_state(
            executed_queue, executed_path,
            "Both HF arms generated and bound by validate_source_shard; rater prompt date frozen; "
            "operator skip_formula stamp present; Codex review waived by the operator, not passed",
        ))

    pending_phases = phases_for(PENDING_ARMS)
    pending_queue = queue(pending_phases, (
        "Mistral AWQ int4 rater scoring for the fourth registered English B0 arm "
        "gemma-3-12b-it on bjb1. Runs only after the gemma generation queue has "
        "written its 23 source shards. Operator skip_formula (no Codex review was run "
        "for these units); QUALIFICATION_GATE.results stay null and ELIGIBLE_PAIRS "
        "stays unfrozen. Seed 20260829 has no source shard and must not be retried. "
        "Rater prompt date frozen to 2026-09-18 to match the other arms. "
        "Do not touch 17/40 on westb."
    ))
    gemma_path = out / "english_b0_rater_gemma_queue.json"
    write_json(gemma_path, pending_queue)
    write_json(out / "english_b0_rater_gemma_queue_state.queued.json", pending_state(
        pending_queue, gemma_path,
        "gemma-3-12b-it generation shards present and bound by validate_source_shard; "
        "rater prompt date frozen; operator skip_formula stamp present; "
        "Codex review waived by the operator, not passed",
    ))

    print(json.dumps({
        "seeds": len(QUALIFICATION_SEEDS),
        "executed_queue": str(executed_path),
        "executed_queue_rewritten": executed_written,
        "pending_queue": str(gemma_path),
        "pending_jobs": sum(len(phase["jobs"]) for phase in pending_phases),
        "manifests": [ARMS[arm][f"{kind}_manifest"]
                      for arm in ARMS for kind in ("scoring", "kernel")],
        "packet_bytes": sizes,
    }, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="process/experiment_queue")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
