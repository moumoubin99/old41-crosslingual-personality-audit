#!/usr/bin/env python3
"""Build English B0 queues for the remaining three checkpoints on bjb1."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = "<LOCAL_PATH>/41-2026-09-05-aris-3"
DEPLOYMENT_ROOT = "<LOCAL_PATH>/project41"
PYTHON = "<LOCAL_PATH>/python3.12"
IDEA = "xling-induced-personality-transfer-matrix"
QUALIFICATION_SEEDS = tuple(range(20260811, 20260835))

MODELS = {
    "aya-expanse-8b": {
        "path": "<LOCAL_PATH>/CohereLabs__aya-expanse-8b",
        "slug": "aya-expanse-8b",
        "manifest": "process/consumption/deployment/english-b0-hf-aya-manifest.json",
        "evidence": "process/consumption/deployment/english-b0-hf-aya-v1",
        "runtime": "process/consumption/deployment/aya-expanse-8b-runtime-manifest.json",
    },
    "Llama-3.1-8B-Instruct": {
        "path": "<LOCAL_PATH>/meta-llama__Llama-3.1-8B-Instruct",
        "slug": "Llama-3.1-8B-Instruct",
        "manifest": "process/consumption/deployment/english-b0-hf-llama-manifest.json",
        "evidence": "process/consumption/deployment/english-b0-hf-llama-v1",
        "runtime": "process/consumption/deployment/llama31-8b-runtime-manifest.json",
    },
    "gemma-3-12b-it": {
        "path": "<LOCAL_PATH>/google__gemma-3-12b-it",
        "slug": "gemma-3-12b-it",
        "manifest": "process/consumption/deployment/english-b0-hf-gemma-manifest.json",
        "evidence": "process/consumption/deployment/english-b0-hf-gemma-v1",
        "runtime": "process/consumption/deployment/gemma-3-12b-it-runtime-manifest.json",
    },
}


def generation_job(model_id: str, seed: int, est_hours: float = 0.50):
    spec = MODELS[model_id]
    return {
        "id": f"b0-competence-{spec['slug']}-s{seed}",
        "idea": IDEA,
        "cmd": (
            f"{PYTHON} -m experiments.xlpt.run_hf_generation "
            f"--project {PROJECT} --phase b0-competence "
            f"--model-path {spec['path']} --model-id {model_id} "
            f"--runtime-manifest {spec['runtime']} --seed {seed}"
        ),
        "formula_manifest": spec["manifest"],
        "formula_evidence": spec["evidence"],
        "est_hours": est_hours,
        "expected_output": f"{PROJECT}/process/results/xlpt/b0-competence/{spec['slug']}/seed-{seed}.jsonl",
    }


def base(phases):
    return {
        "project": IDEA,
        "cwd": DEPLOYMENT_ROOT,
        "conda": "base",
        "activate_conda": False,
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 1000,
        "oom_retry": {"delay": 120, "max_attempts": 2},
        "notes": (
            "English B0 competence for remaining checkpoints on bjb1. "
            "Operator skip_formula; QUALIFICATION_GATE.results stay null. "
            "Do not retry seed 20260829. Do not touch 17/40 on westb."
        ),
        "phases": phases,
    }


def pending_state(manifest, manifest_path: Path, release_condition: str):
    jobs = []
    phases = []
    for phase in manifest["phases"]:
        phases.append({"name": phase["name"], "depends_on": phase.get("depends_on", []), "status": "pending"})
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


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(args):
    out = Path(args.out).resolve()
    aya = base([{
        "name": "b0-english-competence-aya",
        "depends_on": [],
        "jobs": [generation_job("aya-expanse-8b", seed) for seed in QUALIFICATION_SEEDS],
    }])
    aya_path = out / "english_b0_aya_queue.json"
    # The aya arm has already executed; never overwrite its manifest or the
    # scheduler state that records what actually ran.
    if not aya_path.exists():
        write_json(aya_path, aya)
        write_json(out / "english_b0_aya_queue_state.queued.json", pending_state(
            aya, aya_path,
            "Aya checkpoint copied, HF runtime manifest recorded, operator skip_formula on English B0 remaining cells",
        ))
    # Fourth registered checkpoint.  Seed 20260829 has no usable source cluster
    # anywhere in this study and is deliberately not scheduled or retried.
    gemma_seeds = tuple(seed for seed in QUALIFICATION_SEEDS if seed != 20260829)
    gemma = base([{
        "name": "b0-english-competence-gemma",
        "depends_on": [],
        "jobs": [generation_job("gemma-3-12b-it", seed, 0.75) for seed in gemma_seeds],
    }])
    gemma["notes"] = (
        "English B0 competence for the fourth registered checkpoint gemma-3-12b-it on bjb1. "
        "Operator skip_formula; QUALIFICATION_GATE.results stay null and ELIGIBLE_PAIRS stays unfrozen. "
        "Seed 20260829 is not scheduled. Do not touch 17/40 on westb."
    )
    gemma_path = out / "english_b0_gemma_queue.json"
    write_json(gemma_path, gemma)
    write_json(out / "english_b0_gemma_queue_state.queued.json", pending_state(
        gemma, gemma_path,
        "gemma-3-12b-it checkpoint present on bjb1, HF runtime manifest recorded, operator skip_formula on English B0 remaining cells",
    ))
    print(json.dumps({
        "aya_jobs": 24,
        "aya_manifest": str(aya_path),
        "gemma_jobs": len(gemma_seeds),
        "gemma_manifest": str(gemma_path),
    }, ensure_ascii=False))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="process/experiment_queue")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
