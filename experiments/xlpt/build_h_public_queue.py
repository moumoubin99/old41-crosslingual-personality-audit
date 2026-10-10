#!/usr/bin/env python3
"""Build the block-H (public human-annotated data) rater queue.

Block H scores public dialogue with gold human personality labels using the main study's frozen
rater, as specified in refine-logs/followup/H_public_spec.md (revision 2):

* ``h-public-friendspersona`` -- FriendsPersona, 711 units, full rendering plus an E-only pass
  without stage directions (12,798 rater calls at most);
* ``h-public-cped`` -- CPED, 569 units, 12 items each (6,828 rater calls).

No human raters and no new annotation.  One job per corpus, one GPU, one job at a time.  The units
are rebuilt and byte-checked against H_PUBLIC_PROVENANCE.json before any call, so the queue binds
the provenance file rather than the 20 MB public data.  Under .aris/OPERATOR_SKIP_FORMULA the
manifest/evidence pointers are provenance for the reused rater path, not a review PASS.  Public H
is descriptive label concordance of a frozen support-oriented score on two public scripted-dialogue
corpora, not validation of the rater: nothing here writes QUALIFICATION_GATE.results, and every
record carries eligible_for_gate = false.

Run from the project root:
    python3 -m experiments.xlpt.build_h_public_queue [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.xlpt.build_pole_queue import IDEA, PROJECT, PYTHON
from experiments.xlpt.build_rater_aya_llama_queue import (
    MAX_SOURCE_BYTES,
    RATER_PATH,
    RATER_RUNTIME_MANIFEST,
    packet_bytes,
    pending_state,
    queue,
    write_json,
)
from experiments.xlpt.score_public_personality import CORPORA, OUT_DIR, PROVENANCE

DEPLOYMENT = "process/consumption/deployment"
H_TEST = {
    "id": "h_public_contract",
    "python_args": ["-m", "experiments.xlpt.test_h_public_contract"],
    "test_files": ["experiments/xlpt/test_h_public_contract.py"],
    "timeout_seconds": 900,
}
SMOKE_TEST = {
    "id": "rater_module_smoke",
    "python_args": ["-m", "experiments.xlpt.test_rater_module_smoke"],
    "test_files": ["experiments/xlpt/test_rater_module_smoke.py"],
    "timeout_seconds": 180,
}
SOURCE_ROOTS = sorted({
    "experiments/xlpt/__init__.py",
    "experiments/xlpt/inference_contract.py",
    "experiments/xlpt/rater_frozen_inputs.py",
    "experiments/xlpt/rater_runtime.py",
    "experiments/xlpt/run_rater_scoring.py",
    "experiments/xlpt/score_rater.py",
    "experiments/xlpt/score_public_personality.py",
    "experiments/xlpt/reduce_h_public.py",
    "experiments/xlpt/test_h_public_contract.py",
    "experiments/xlpt/test_rater_module_smoke.py",
    f"{DEPLOYMENT}/mistral-rater-config/chat_template.json",
    f"{DEPLOYMENT}/mistral-rater-config/config.json",
    f"{DEPLOYMENT}/mistral-rater-config/generation_config.json",
    f"{DEPLOYMENT}/mistral-rater-runtime-v2-manifest.json",
    "process/consumption/frozen/english_execution_v1.json",
    "process/consumption/frozen/english_execution_receipt_v1.json",
    "refine-logs/RUBRIC_BEHAVIOR.md",
    "refine-logs/followup/H_public_spec.md",
    PROVENANCE,
})
# Rater throughput on the main study is ~2.9 calls/s (100 rows x 12 items in ~7 min) plus ~4 min
# of engine load; the estimates round up with a 30 % margin and stay under PILOT_MAX_HOURS = 2.
EST_HOURS = {"friendspersona": 1.6, "cped": 1.0}
NOTES = (
    "Block H, public data (refine-logs/followup/H_public_spec.md rev 2): the frozen Mistral rater "
    "scores FriendsPersona (711 units; plus an E-only pass without stage directions) and CPED "
    "(569 units, 287 identities, K = 2) against the corpora's own human Big-Five labels. No human "
    "raters, no new annotation. Units are rebuilt and byte-checked against H_PUBLIC_PROVENANCE.json "
    "before any call; context overflow excludes a unit and never truncates. Descriptive label "
    "concordance of a frozen support-oriented score on two public scripted-dialogue corpora; not "
    "validation of the rater. Score shards are validated by reduce_h_public --validate-only before "
    "completion: eligible_for_gate = false, nothing writes QUALIFICATION_GATE.results. Formula Codex review "
    "is skipped under .aris/OPERATOR_SKIP_FORMULA; manifest/evidence pointers are provenance, not a "
    "PASS. Runs after block BC; one GPU, one job at a time. Do not touch 17/40 on westb."
)


def launch_argv(corpus: str) -> list[str]:
    return [
        PYTHON, "-m", "experiments.xlpt.score_public_personality",
        "--project", PROJECT,
        "--corpus", corpus,
        "--rater-path", RATER_PATH,
        "--runtime-manifest", RATER_RUNTIME_MANIFEST,
    ]


def manifest_path(corpus: str) -> str:
    return f"{DEPLOYMENT}/h-public-{corpus}-scoring-manifest.json"


def evidence_path(corpus: str) -> str:
    return f"{DEPLOYMENT}/h-public-{corpus}-scoring"


def unit(corpus: str) -> dict:
    return {
        "version": 1,
        "deployment_root": "<LOCAL_PATH>/project41",
        "spec": f"{DEPLOYMENT}/ENGLISH_EXECUTION_MATH_SPEC.json",
        "design_card": "refine-logs/design_card.json",
        "formula_ids": ["rater_score_parse", "english_trait_mean", "english_rate"],
        "launch_argv": launch_argv(corpus),
        "bindings": [
            {"formula_id": "rater_score_parse",
             "implementation": [{"file": "experiments/xlpt/run_rater_scoring.py", "symbol": "parse_item_score"}],
             "test_ids": ["h_public_contract"]},
            {"formula_id": "english_trait_mean",
             "implementation": [{"file": "experiments/xlpt/score_public_personality.py", "symbol": "run"}],
             "test_ids": ["h_public_contract"]},
            {"formula_id": "english_rate",
             "implementation": [{"file": "experiments/xlpt/rater_runtime.py", "symbol": "measured_rate"}],
             "test_ids": ["h_public_contract"]},
        ],
        "source_roots": SOURCE_ROOTS,
        "tests": [H_TEST, SMOKE_TEST],
    }


def job(corpus: str) -> dict:
    return {
        "id": f"h-public-{corpus}",
        "idea": IDEA,
        "cmd": " ".join(launch_argv(corpus)),
        "formula_manifest": manifest_path(corpus),
        "formula_evidence": evidence_path(corpus),
        "est_hours": EST_HOURS[corpus],
        "expected_output": f"{PROJECT}/{OUT_DIR}/scores-{corpus}.jsonl",
    }


def manifest() -> dict:
    return queue([{"name": "h-public-rater", "depends_on": [], "jobs": [job(c) for c in CORPORA]}], NOTES)


def _text(payload) -> str:
    return json.dumps(payload, indent=2) + "\n"


def run(args) -> None:
    project = Path(args.project).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = project / out
    queue_manifest = manifest()
    units = {manifest_path(c): unit(c) for c in CORPORA}
    size = packet_bytes(project, next(iter(units.values())))
    if size >= MAX_SOURCE_BYTES:
        raise ValueError(f"H packet is {size} B, at or over {MAX_SOURCE_BYTES} B")
    queue_path = out / "h_public_queue.json"

    if args.check:
        drift = [str(queue_path)] if (not queue_path.exists()
                                      or queue_path.read_text(encoding="utf-8") != _text(queue_manifest)) else []
        drift += [rel for rel, payload in units.items()
                  if not (project / rel).exists() or (project / rel).read_text(encoding="utf-8") != _text(payload)]
        status = "DRIFT: " + ", ".join(drift) if drift else "identical"
    else:
        for rel, payload in units.items():
            write_json(project / rel, payload)
        write_json(queue_path, queue_manifest)
        write_json(out / "h_public_queue_state.queued.json", pending_state(
            queue_manifest, queue_path,
            "block BC chain finished (.bc_chain_done) and no queue manager running; units and "
            "H_PUBLIC_PROVENANCE.json synced and byte-identical on the deployment root; user go-ahead "
            "for public-data H recorded 2026-09-29; operator skip_formula stamp present; Codex formula "
            "review waived by the operator, not passed",
        ))
        status = "written"

    print(json.dumps({
        "status": status,
        "jobs": [(j["id"], j["est_hours"]) for j in queue_manifest["phases"][0]["jobs"]],
        "est_hours": round(sum(EST_HOURS.values()), 2),
        "packet_bytes": size,
    }))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="process/experiment_queue")
    ap.add_argument("--check", action="store_true",
                    help="compare against the written queue and units instead of writing anything")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
