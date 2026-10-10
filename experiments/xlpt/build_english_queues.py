#!/usr/bin/env python3
"""Build immutable experiment-queue manifests for executable English XLPT cells."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = "<LOCAL_PATH>/41-2026-09-05-aris-3"
DEPLOYMENT_ROOT = "<LOCAL_PATH>/project41"
PYTHON = "<LOCAL_PATH>/python3.12"
QWEN = "<LOCAL_PATH>/Qwen__Qwen3-8B"
RATER = "<LOCAL_PATH>/OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"
IDEA = "xling-induced-personality-transfer-matrix"
QUALIFICATION_SEEDS = tuple(range(20260811, 20260835))
EVALUATION_SEEDS = tuple(range(20260911, 20260935))


def generation_job(phase: str, seed: int, manifest: str, evidence: str, estimate: float):
    return {
        "id": f"{phase}-qwen-s{seed}",
        "idea": IDEA,
        "cmd": (f"{PYTHON} -m experiments.xlpt.run_transformers_generation "
                f"--project {PROJECT} --phase {phase} --model-path {QWEN} "
                f"--model-id Qwen3-8B --seed {seed}"),
        "formula_manifest": manifest,
        "formula_evidence": evidence,
        "est_hours": estimate,
        "expected_output": f"{PROJECT}/process/results/xlpt/{phase}/qwen3-8b/seed-{seed}.jsonl",
    }


def indicator_job(phase: str, seed: int):
    return {
        "id": f"b3-indicator-{phase}-s{seed}",
        "idea": IDEA,
        "cmd": (f"{PYTHON} {DEPLOYMENT_ROOT}/experiments/xlpt/compute_rater_free_indicators.py "
                f"--project {PROJECT} --phase {phase} --seed {seed}"),
        "est_hours": 0.01,
        "expected_output": f"{PROJECT}/process/results/xlpt/b3-rater-free/{phase}/seed-{seed}.json",
    }


def rater_job(seed: int):
    return {
        "id": f"b0-rater-s{seed}",
        "idea": IDEA,
        "cmd": (f"{PYTHON} -m experiments.xlpt.run_rater_scoring "
                f"--project {PROJECT} --seed {seed} --rater-path {RATER}"),
        "formula_manifest": "process/consumption/deployment/english-b0-rater-manifest.json",
        "formula_evidence": "process/consumption/deployment/english-b0-rater-v1",
        "est_hours": 0.35,
        "expected_output": f"{PROJECT}/process/results/xlpt/b0-rater/qwen3-8b/seed-{seed}.jsonl",
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
    b0_manifest = base([{
        "name": "b0-english-competence-generation",
        "depends_on": [],
        "jobs": [generation_job(
            "b0-competence", seed,
            "process/consumption/deployment/english-b0-qwen-manifest.json",
            "process/consumption/deployment/english-b0-qwen-v5", 0.50,
        ) for seed in QUALIFICATION_SEEDS],
    }])
    future_manifest = base([
        {
            "name": "b1-english-neutral-generation",
            "depends_on": [],
            "jobs": [generation_job(
                "b1-neutral", seed,
                "process/consumption/deployment/english-b1-neutral-qwen-manifest.json",
                "process/consumption/deployment/english-b1-neutral-qwen-v1", 0.85,
            ) for seed in EVALUATION_SEEDS],
        },
        {
            "name": "b2-english-compliance-generation",
            "depends_on": ["b1-english-neutral-generation"],
            "jobs": [generation_job(
                "b2-compliance", seed,
                "process/consumption/deployment/english-b2-compliance-qwen-manifest.json",
                "process/consumption/deployment/english-b2-compliance-qwen-v1", 0.85,
            ) for seed in EVALUATION_SEEDS],
        },
        {
            "name": "b3-english-rater-free-indicators",
            "depends_on": ["b1-english-neutral-generation", "b2-english-compliance-generation"],
            "jobs": ([indicator_job("b1-neutral", seed) for seed in EVALUATION_SEEDS]
                     + [indicator_job("b2-compliance", seed) for seed in EVALUATION_SEEDS]),
        },
    ])
    b0_path = out / "english_b0_queue.json"
    rater_manifest = base([{
        "name": "b0-english-rater-scoring",
        "depends_on": [],
        "jobs": [rater_job(seed) for seed in QUALIFICATION_SEEDS],
    }])
    rater_path = out / "english_b0_rater_queue.json"
    future_path = out / "english_b1_b2_b3_queue.json"
    write_json(b0_path, b0_manifest)
    write_json(rater_path, rater_manifest)
    write_json(out / "english_b0_rater_queue_state.queued.json", pending_state(
        rater_manifest, rater_path,
        "accepted rater deployment evidence and corresponding B0 generation shards",
    ))
    write_json(future_path, future_manifest)
    write_json(out / "english_b1_b2_b3_queue_state.queued.json", pending_state(
        future_manifest, future_path,
        "B0 qualification gate plus accepted per-phase B1/B2 deployment evidence; do not start before both",
    ))
    print(json.dumps({"b0_generation_jobs": 24, "queued_b0_rater_jobs": 24, "queued_future_jobs": 96,
                      "b0_manifest": str(b0_path), "b0_rater_manifest": str(rater_path),
                      "future_manifest": str(future_path)}))


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="process/experiment_queue")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
