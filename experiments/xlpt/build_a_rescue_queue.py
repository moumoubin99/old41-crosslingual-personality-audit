#!/usr/bin/env python3
"""Rescue queue for block-A (zh-b4) rater shards whose frozen-scorer run crashed on an item parse.

Parse policy ``strict_item_missing_v1`` (FOLLOWUP_PLAN amendment of 2026-09-30).  The frozen
``run_rater_scoring`` aborts a whole shard when one greedy rater output is not exactly one integer
1..7; in block A that happened on outputs such as ``'<LOCAL_PATH>'``.  Those cells get one rescue run
with ``run_rater_item_missing --namespace rescue``: the same v3 freeze, engine, template, date and
parser, with the failed item recorded as missing.  Rescue shards land in ``{score_dir}-rescue`` and
carry ``rescue_run`` true; the scorer refuses a cell that already has a canonical or rescue shard.

Two steps, both deterministic:

* ``--units`` (local, before sync) writes one review unit per (checkpoint, zh-b4 phase): the
  block-A unit with the scorer swapped for its item-missing wrapper, ``--namespace rescue`` and the
  wrapper's contract test.  ``--units --check`` compares instead of writing.
* ``--select`` (server, after the A queue manager has exited) reads the A queue, its state and its
  job logs, and schedules exactly the cells that (a) are in state ``stuck`` with a scheduled
  block-A seed (never 20260829), (b) have no canonical and no rescue shard, and (c) whose log shows
  the frozen parser's error.  Every other unfinished job is
  listed with its reason in ``a_rescue_selection.json`` and is not rescued.  Nothing in the A queue,
  its state or its logs is modified.

    python3 -m experiments.xlpt.build_a_rescue_queue --units
    python3 -m experiments.xlpt.build_a_rescue_queue --select --project <data root>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.xlpt.build_bc_queue import (
    FROZEN_SCORER_MODULE,
    ITEM_MISSING_MODULE,
    ITEM_MISSING_SOURCES,
    ITEM_MISSING_TEST,
)
from experiments.xlpt.build_pole_rater_queue import ARMS
from experiments.xlpt.build_rater_aya_llama_queue import (
    MAX_SOURCE_BYTES,
    PROJECT,
    SOURCE_MANIFEST,
    packet_bytes,
    pending_state,
    queue,
    write_json,
)
from experiments.xlpt.build_zh_queue import ZH_SEEDS
from experiments.xlpt.build_zh_rater_queue import ZH_SCORED, zh_unit

PARSE_ERROR = "rater must return exactly one integer"
NEVER_SCHEDULED = frozenset({20260829})
RESCUE_SUFFIX = "-rescue"
A_PHASES = tuple(entry for entry in ZH_SCORED if entry[0].startswith("zh-b4-"))
A_QUEUE = "zh_b4_rater_queue.json"
A_STATE = "zh_b4_rater_queue_state.queued.json"
A_LOGS = "logs-zh-b4-rater"

NOTES = (
    "Block-A rescue (parse policy strict_item_missing_v1, FOLLOWUP_PLAN amendment 2026-09-30): "
    "only zh-b4 rater cells whose frozen-scorer run crashed on the frozen item parser are scored "
    "again, once, by run_rater_item_missing --namespace rescue with the same v3 freeze, engine, "
    "template, prompt date and parser. Rescue shards go to {score_dir}-rescue with rescue_run true "
    "and are reported as rescue cells; a cell never has both a canonical and a rescue shard. Failed "
    "items are null and never retried; a trait score needs 6/6 items. Exploratory within-checkpoint "
    "contrasts only: nothing here writes QUALIFICATION_GATE.results or freezes ELIGIBLE_PAIRS. "
    "Formula Codex review is skipped under .aris/OPERATOR_SKIP_FORMULA; manifest/evidence pointers "
    "are provenance, not a PASS receipt. Seed 20260829 is never scheduled. Do not touch 17/40 on westb."
)


def rescue_unit(source: dict, arm: str, phase: str) -> dict:
    """The block-A unit for this (checkpoint, phase), rebound to the item-missing wrapper."""
    unit = zh_unit(source, arm, phase)
    if FROZEN_SCORER_MODULE not in unit["launch_argv"]:
        raise ValueError("block-A unit no longer launches the frozen scorer; rebuild this derivation")
    unit["launch_argv"] = [ITEM_MISSING_MODULE if token == FROZEN_SCORER_MODULE else token
                           for token in unit["launch_argv"]] + ["--namespace", "rescue"]
    unit["source_roots"] = sorted(set(unit["source_roots"]) | set(ITEM_MISSING_SOURCES))
    unit["tests"] = unit["tests"] + [ITEM_MISSING_TEST]
    for binding in unit["bindings"]:
        binding["test_ids"] = binding["test_ids"] + [ITEM_MISSING_TEST["id"]]
    return unit


def manifest_path(arm: str, short: str) -> str:
    return (f"process/consumption/deployment/"
            f"{short}-rescue-rater-{ARMS[arm]['slug']}-scoring-manifest.json")


def evidence_path(arm: str, short: str) -> str:
    return f"process/consumption/deployment/{short}-rescue-rater-{ARMS[arm]['slug']}-scoring"


def rescue_units(project: Path, source: dict):
    units, sizes = {}, {}
    for arm in ARMS:
        for phase, _score_dir, short in A_PHASES:
            unit = rescue_unit(source, arm, phase)
            size = packet_bytes(project, unit)
            if size >= MAX_SOURCE_BYTES:
                raise ValueError(f"{arm} {phase} packet is {size} B, at or over {MAX_SOURCE_BYTES} B")
            units[manifest_path(arm, short)] = unit
            sizes[f"{ARMS[arm]['slug']}-{short}"] = size
    return units, sizes


def _text(payload) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _cell(job: dict) -> dict:
    """Parse arm, phase and seed back out of one block-A job and check its fixed shape."""
    argv = job["cmd"].split()
    if argv.count(FROZEN_SCORER_MODULE) != 1 or "--namespace" in argv:
        raise ValueError(f"{job['id']}: not a frozen-scorer block-A job")
    value = {flag: argv[argv.index(flag) + 1]
             for flag in ("--seed", "--source-checkpoint", "--source-phase")}
    by_phase = {phase: (score_dir, short) for phase, score_dir, short in A_PHASES}
    phase = value["--source-phase"]
    if phase not in by_phase or value["--source-checkpoint"] not in ARMS:
        raise ValueError(f"{job['id']}: unregistered block-A phase or checkpoint")
    score_dir, short = by_phase[phase]
    return {"arm": value["--source-checkpoint"], "phase": phase, "score_dir": score_dir,
            "short": short, "seed": int(value["--seed"]), "argv": argv}


def select(project: Path, qdir: Path):
    manifest = json.loads((qdir / A_QUEUE).read_text(encoding="utf-8"))
    state = json.loads((qdir / A_STATE).read_text(encoding="utf-8"))
    status = {job["id"]: job for job in state["jobs"]}
    results = project / "process/results/xlpt"
    by_phase, selected, skipped = {}, [], []
    for phase_entry in manifest["phases"]:
        for job in phase_entry["jobs"]:
            record = status.get(job["id"], {})
            if record.get("status") == "completed":
                continue
            cell = _cell(job)
            slug = ARMS[cell["arm"]]["slug"]
            canonical = results / cell["score_dir"] / slug / f"seed-{cell['seed']}.jsonl"
            rescue = results / f"{cell['score_dir']}{RESCUE_SUFFIX}" / slug / f"seed-{cell['seed']}.jsonl"
            log = qdir / A_LOGS / f"{job['id']}.log"
            log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            reason = None
            if cell["seed"] in NEVER_SCHEDULED or cell["seed"] not in ZH_SEEDS:
                reason = "seed is not a scheduled block-A seed"
            elif record.get("status") != "stuck":
                reason = "queue state is not 'stuck'"
            elif canonical.exists():
                reason = "canonical shard exists"
            elif rescue.exists():
                reason = "rescue shard exists"
            elif PARSE_ERROR not in log_text:
                reason = "log does not show the frozen item-parse error"
            entry = {"id": job["id"], "state": record.get("status"), "attempts": record.get("attempts"),
                     "error": record.get("error")}
            if reason:
                skipped.append({**entry, "reason": reason})
                continue
            argv = [ITEM_MISSING_MODULE if token == FROZEN_SCORER_MODULE else token
                    for token in cell["argv"]] + ["--namespace", "rescue"]
            name = f"{cell['short']}-rescue-rater-{slug}"
            by_phase.setdefault(name, []).append({
                "id": f"{cell['short']}-rescue-rater-{slug}-s{cell['seed']}",
                "idea": job["idea"],
                "cmd": " ".join(argv),
                "formula_manifest": manifest_path(cell["arm"], cell["short"]),
                "formula_evidence": evidence_path(cell["arm"], cell["short"]),
                "est_hours": job["est_hours"],
                "expected_output": f"{PROJECT}/process/results/xlpt/{cell['score_dir']}{RESCUE_SUFFIX}/"
                                   f"{slug}/seed-{cell['seed']}.jsonl",
            })
            selected.append({**entry, "rescue_id": by_phase[name][-1]["id"],
                             "parse_error_lines": [line.strip() for line in log_text.splitlines()
                                                   if PARSE_ERROR in line][:3]})
    phases = [{"name": name, "depends_on": [], "jobs": jobs} for name, jobs in by_phase.items()]
    return phases, selected, skipped


def run(args) -> None:
    project = Path(args.project).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = project / out
    if args.units:
        source = json.loads((project / SOURCE_MANIFEST).read_text(encoding="utf-8"))
        units, sizes = rescue_units(project, source)
        if args.check:
            drift = [rel for rel, unit in units.items() if not (project / rel).exists()
                     or (project / rel).read_text(encoding="utf-8") != _text(unit)]
            status = "DRIFT: " + ", ".join(drift) if drift else "identical"
        else:
            for rel, unit in units.items():
                write_json(project / rel, unit)
            status = "written"
        print(json.dumps({"status": status, "units": sorted(units), "packet_bytes": sizes}))
        return
    phases, selected, skipped = select(project, out)
    payload = queue(phases, NOTES)
    write_json(out / "a_rescue_selection.json", {
        "parse_policy": "strict_item_missing_v1", "parse_error": PARSE_ERROR,
        "selected": selected, "not_rescued": skipped,
    })
    # A fresh selection always replaces the queue artifacts, so an empty one can never leave an
    # older queue behind for the chain to run.
    for stale in ("a_rescue_queue.json", "a_rescue_queue_state.queued.json"):
        (out / stale).unlink(missing_ok=True)
    if phases:
        write_json(out / "a_rescue_queue.json", payload)
        write_json(out / "a_rescue_queue_state.queued.json", pending_state(
            payload, out / "a_rescue_queue.json",
            "block-A queue manager exited; each selected cell is 'stuck', has no canonical or rescue "
            "shard and its frozen-scorer log shows the item-parse error; operator skip_formula stamp "
            "present; Codex formula review waived by the operator, not passed",
        ))
    print(json.dumps({"rescue_jobs": len(selected), "not_rescued": len(skipped),
                      "phases": [(p["name"], len(p["jobs"])) for p in phases]}))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default="process/experiment_queue")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--units", action="store_true", help="write (or --check) the rescue review units")
    mode.add_argument("--select", action="store_true", help="build the rescue queue from the finished A queue")
    ap.add_argument("--check", action="store_true", help="with --units: compare instead of writing")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
