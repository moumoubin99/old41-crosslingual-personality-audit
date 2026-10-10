#!/usr/bin/env python3
"""Build the block-BC generation and rater queues for bjb1.

Block BC crosses the reviewed Chinese persona rendering (P = zh, the ``_PZHN_``
strings frozen by execution config v4) with both dialogue languages, on the two
Extraversion poles:

* ``bc-pzhn-extraversion-{high,low}-dzh`` -- P = zh, D = zh, the zh scenario
  surface and facts label that block A scores;
* ``bc-pzhn-extraversion-{high,low}-den`` -- P = zh, D = en, the English
  scenario rendering exactly as in v2.

Checkpoints: Qwen3-8B, aya-expanse-8b, Llama-3.1-8B-Instruct.  gemma-3-12b-it is
out of competence for Chinese (ZH_G1_COMPETENCE_SCREEN.json) and is not
scheduled.  Seeds: the 23 qualification seeds (20260811-20260834 minus the
stuck 20260829, which is never scheduled or retried).  B and C are one
indivisible block: both D levels are generated, then every output is scored
(ITT; no response-language filter).

Every generation job is that checkpoint's own B0 argv with only --phase,
--prompt-config and --freeze-receipt changed, and every rater unit is derived
from the zh rater unit with the v3 freeze replaced by v4.  Under
.aris/OPERATOR_SKIP_FORMULA the manifest/evidence pointers are provenance for
the reused code path, not a review PASS.  Nothing here writes
QUALIFICATION_GATE.results or freezes ELIGIBLE_PAIRS.

Run from the project root:
    python3 -m experiments.xlpt.build_bc_queue [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.xlpt.build_pole_queue import (
    DEPLOYMENT_ROOT,
    IDEA,
    MODELS,
    PROJECT,
    PYTHON,
)
from experiments.xlpt.build_pole_rater_queue import ARMS
from experiments.xlpt.build_rater_aya_llama_queue import (
    MAX_SOURCE_BYTES,
    RATER_PATH,
    RATER_RUNTIME_MANIFEST,
    SOURCE_MANIFEST,
    packet_bytes,
    pending_state,
    queue,
    write_json,
)
from experiments.xlpt.build_zh_queue import ZH_SEEDS
from experiments.xlpt.build_zh_rater_queue import (
    FREEZE_RECEIPT_V3,
    PROMPT_CONFIG_V3,
    zh_unit,
)

PROMPT_CONFIG_V4 = "process/consumption/frozen/english_execution_v4.json"
FREEZE_RECEIPT_V4 = "process/consumption/frozen/english_execution_receipt_v4.json"

BC_SEEDS = ZH_SEEDS

# BC scoring runs the item-missing wrapper of the frozen scorer: an unparseable item is recorded
# as missing instead of aborting the shard (FOLLOWUP_PLAN amendment 2026-09-30).
FROZEN_SCORER_MODULE = "experiments.xlpt.run_rater_scoring"
ITEM_MISSING_MODULE = "experiments.xlpt.run_rater_item_missing"
ITEM_MISSING_SOURCES = (
    "experiments/xlpt/run_rater_item_missing.py",
    "experiments/xlpt/test_rater_item_missing.py",
)
ITEM_MISSING_TEST = {
    "id": "rater_item_missing",
    "python_args": ["-m", "experiments.xlpt.test_rater_item_missing"],
    "test_files": ["experiments/xlpt/test_rater_item_missing.py"],
    "timeout_seconds": 180,
}

# Generator MODELS key -> rater ARMS key.  gemma is excluded by the zh G1 screen.
BC_CHECKPOINTS = (
    ("Qwen3-8B", "qwen3-8b"),
    ("aya-expanse-8b", "aya-expanse-8b"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
)

# (generator phase, score directory, short job tag).  Mirrors the PHASES and
# SOURCE_PHASES entries; test_pole_rater_contract pins the three tables together.
BC_PHASES = (
    ("bc-pzhn-extraversion-high-dzh", "bc-pzhn-extraversion-high-dzh-rater", "bc-high-dzh"),
    ("bc-pzhn-extraversion-low-dzh", "bc-pzhn-extraversion-low-dzh-rater", "bc-low-dzh"),
    ("bc-pzhn-extraversion-high-den", "bc-pzhn-extraversion-high-den-rater", "bc-high-den"),
    ("bc-pzhn-extraversion-low-den", "bc-pzhn-extraversion-low-den-rater", "bc-low-den"),
)

COMMON_NOTES = (
    "Block BC: reviewed Chinese persona rendering (P = zh, _PZHN_ strings, execution config v4) "
    "crossed with D in {zh, en} on the Extraversion poles; Qwen3-8B, aya-expanse-8b and "
    "Llama-3.1-8B-Instruct (gemma excluded by ZH_G1_COMPETENCE_SCREEN.json) x 23 seeds "
    "(20260811-20260834 minus the stuck 20260829, never retried). The persona is one reviewed "
    "rendering reviewed by a model, not a human native reader; the D = zh scenarios stay the "
    "NLLB-200-3.3B single-stratum rendering. Exploratory within-checkpoint contrasts: nothing here "
    "writes QUALIFICATION_GATE.results or freezes ELIGIBLE_PAIRS. Formula Codex review is skipped "
    "under .aris/OPERATOR_SKIP_FORMULA; manifest/evidence pointers are provenance for the reused "
    "code path, not a PASS receipt. The v3 _PZH phases are not run. Do not touch 17/40 on westb. "
    "Do not rerun Qwen English B1/B2."
)
GEN_NOTES = "Generation half of block BC. " + COMMON_NOTES
RATER_NOTES = (
    "Rater half of block BC: Mistral AWQ int4, English instrument on every row (R16-F1), rater "
    "prompt date frozen at 2026-09-18 as for every earlier arm. ITT: every generated output is "
    "scored; response language is reported, never filtered. Runs only after all BC generation "
    "shards exist. " + COMMON_NOTES
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
        "--prompt-config", PROMPT_CONFIG_V4,
        "--freeze-receipt", FREEZE_RECEIPT_V4,
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


def generation_manifest() -> dict:
    phases = [
        {
            "name": f"{phase}-generation",
            "depends_on": [],
            "jobs": [
                generation_job(model_id, phase, short, seed)
                for model_id, _arm in BC_CHECKPOINTS
                for seed in BC_SEEDS
            ],
        }
        for phase, _score_dir, short in BC_PHASES
    ]
    return queue(phases, GEN_NOTES)


def bc_unit(source: dict, arm: str, phase: str) -> dict:
    """The zh rater review unit for this (checkpoint, phase), rebound from the v3 to the v4 freeze
    and from the frozen scorer to its item-missing wrapper (parse policy strict_item_missing_v1,
    FOLLOWUP_PLAN amendment 2026-09-30), which reuses every frozen scoring object unchanged."""
    unit = zh_unit(source, arm, phase)
    swap = {PROMPT_CONFIG_V3: PROMPT_CONFIG_V4, FREEZE_RECEIPT_V3: FREEZE_RECEIPT_V4,
            FROZEN_SCORER_MODULE: ITEM_MISSING_MODULE}
    if not all(old in unit["launch_argv"] for old in swap) or not all(
            old in unit["source_roots"] for old in (PROMPT_CONFIG_V3, FREEZE_RECEIPT_V3)):
        raise ValueError("zh rater unit no longer binds the v3 freeze; rebuild this derivation")
    unit["launch_argv"] = [swap.get(token, token) for token in unit["launch_argv"]] + [
        "--namespace", "canonical"]
    unit["source_roots"] = sorted(
        {swap.get(root, root) for root in unit["source_roots"]} | set(ITEM_MISSING_SOURCES))
    unit["tests"] = unit["tests"] + [ITEM_MISSING_TEST]
    for binding in unit["bindings"]:
        binding["test_ids"] = binding["test_ids"] + [ITEM_MISSING_TEST["id"]]
    return unit


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
            f"{PYTHON} -m {ITEM_MISSING_MODULE} "
            f"--project {PROJECT} --seed {seed} "
            f"--source-checkpoint {arm} "
            f"--source-phase {phase} "
            f"--prompt-config {PROMPT_CONFIG_V4} "
            f"--freeze-receipt {FREEZE_RECEIPT_V4} "
            f"--rater-path {RATER_PATH} "
            f"--runtime-manifest {RATER_RUNTIME_MANIFEST} "
            f"--namespace canonical"
        ),
        "formula_manifest": manifest_path(arm, short),
        "formula_evidence": evidence_path(arm, short),
        "est_hours": 0.35,
        "expected_output": f"{PROJECT}/process/results/xlpt/{score_dir}/{slug}/seed-{seed}.jsonl",
    }


def rater_manifest(project: Path, source: dict):
    units, sizes = {}, {}
    for _model_id, arm in BC_CHECKPOINTS:
        for phase, _score_dir, short in BC_PHASES:
            unit = bc_unit(source, arm, phase)
            size = packet_bytes(project, unit)
            if size >= MAX_SOURCE_BYTES:
                raise ValueError(f"{arm} {phase} packet is {size} B, at or over {MAX_SOURCE_BYTES} B")
            units[manifest_path(arm, short)] = unit
            sizes[f"{ARMS[arm]['slug']}-{short}"] = size
    phases = [
        {
            "name": f"{short}-rater-{ARMS[arm]['slug']}",
            "depends_on": [],
            "jobs": [scoring_job(arm, phase, score_dir, short, seed) for seed in BC_SEEDS],
        }
        for phase, score_dir, short in BC_PHASES
        for _model_id, arm in BC_CHECKPOINTS
    ]
    return queue(phases, RATER_NOTES), units, sizes


def _text(payload) -> str:
    return json.dumps(payload, indent=2) + "\n"


def run(args) -> None:
    project = Path(args.project).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = project / out
    source = json.loads((project / SOURCE_MANIFEST).read_text(encoding="utf-8"))

    gen = generation_manifest()
    rater, units, sizes = rater_manifest(project, source)
    gen_path, rater_path = out / "bc_generation_queue.json", out / "bc_rater_queue.json"

    if args.check:
        drift = [str(p) for p, payload in ((gen_path, gen), (rater_path, rater))
                 if not p.exists() or p.read_text(encoding="utf-8") != _text(payload)]
        drift += [rel for rel, unit in units.items()
                  if not (project / rel).exists()
                  or (project / rel).read_text(encoding="utf-8") != _text(unit)]
        status = "DRIFT: " + ", ".join(drift) if drift else "identical"
    else:
        for rel, unit in units.items():
            write_json(project / rel, unit)
        write_json(gen_path, gen)
        write_json(rater_path, rater)
        write_json(out / "bc_generation_queue_state.queued.json", pending_state(
            gen, gen_path,
            "v4 freeze (config b02068de..., receipt 3304381a...) validated by the generator; block A "
            "rater queue finished; user go-ahead for Tier 2 recorded 2026-09-29; operator "
            "skip_formula stamp present; Codex formula review waived by the operator, not passed",
        ))
        write_json(out / "bc_rater_queue_state.queued.json", pending_state(
            rater, rater_path,
            "all 276 BC generation shards present and bound by validate_source_shard under their "
            "own phase and the v4 freeze; rater prompt date frozen; operator skip_formula stamp "
            "present; Codex review waived by the operator, not passed",
        ))
        status = "written"

    print(json.dumps({
        "status": status,
        "generation": [(p["name"], len(p["jobs"])) for p in gen["phases"]],
        "rater": [(p["name"], len(p["jobs"])) for p in rater["phases"]],
        "generation_jobs": sum(len(p["jobs"]) for p in gen["phases"]),
        "rater_jobs": sum(len(p["jobs"]) for p in rater["phases"]),
        "est_hours": round(sum(j["est_hours"] for m in (gen, rater)
                               for p in m["phases"] for j in p["jobs"]), 2),
        "packet_bytes": sizes,
    }, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="process/experiment_queue")
    ap.add_argument("--check", action="store_true",
                    help="compare against the written queues and units instead of writing anything")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
