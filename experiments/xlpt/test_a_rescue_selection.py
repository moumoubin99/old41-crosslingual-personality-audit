#!/usr/bin/env python3
"""Contract test for ``build_a_rescue_queue --select`` on a copy of the real block-A queue.

Cases: only stuck cells with the frozen parse error, a scheduled seed and no shard are selected;
pending, running, missing-state and log-less cells are listed as not rescued; seed 20260829 is never
selected even when stuck with the parse error; an existing canonical or rescue shard blocks the
cell; an empty re-selection removes the stale rescue queue and state.  Nothing is written into the
real queue directory.

    python3 -m experiments.xlpt.test_a_rescue_selection
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from experiments.xlpt import build_a_rescue_queue as rq

PROJECT = Path(__file__).resolve().parents[2]
QDIR = PROJECT / "process/experiment_queue"
ERROR_LINE = "ValueError: " + rq.PARSE_ERROR + " 1..7, got '<LOCAL_PATH>'"


def check(cond, message):
    if not cond:
        raise AssertionError(message)


def setup(root: Path, statuses: dict, logs: dict, extra_jobs=()):
    """A queue dir with the real A manifest plus ``extra_jobs``; states default to completed."""
    qdir = root / "queue"
    (qdir / rq.A_LOGS).mkdir(parents=True)
    manifest = json.loads((QDIR / rq.A_QUEUE).read_text(encoding="utf-8"))
    if extra_jobs:
        manifest["phases"][0]["jobs"].extend(extra_jobs)
    (qdir / rq.A_QUEUE).write_text(json.dumps(manifest), encoding="utf-8")
    jobs = []
    for phase in manifest["phases"]:
        for job in phase["jobs"]:
            if statuses.get(job["id"], "completed") == "missing":
                continue
            jobs.append({"id": job["id"], "status": statuses.get(job["id"], "completed"),
                         "attempts": 1, "error": None})
    (qdir / rq.A_STATE).write_text(json.dumps({"meta": {}, "phases": [], "jobs": jobs}), encoding="utf-8")
    for job_id, text in logs.items():
        (qdir / rq.A_LOGS / f"{job_id}.log").write_text(text, encoding="utf-8")
    return qdir


def select(root: Path, qdir: Path):
    rq.run(SimpleNamespace(project=str(root), out=str(qdir), units=False, select=True, check=False))
    return json.loads((qdir / "a_rescue_selection.json").read_text(encoding="utf-8"))


def main():
    manifest = json.loads((QDIR / rq.A_QUEUE).read_text(encoding="utf-8"))
    ids = [job["id"] for phase in manifest["phases"] for job in phase["jobs"]]
    stuck_ok, stuck_shard, stuck_rescued, stuck_nolog, pending, running, missing = ids[:7]
    never = copy.deepcopy(manifest["phases"][0]["jobs"][0])
    old_seed = never["cmd"].split("--seed ")[1].split()[0]
    never["id"] = never["id"].replace(old_seed, "20260829")
    never["cmd"] = never["cmd"].replace(f"--seed {old_seed}", "--seed 20260829")
    statuses = {stuck_ok: "stuck", stuck_shard: "stuck", stuck_rescued: "stuck", stuck_nolog: "stuck",
                pending: "pending", running: "running", missing: "missing", never["id"]: "stuck"}
    logs = {i: f"Traceback\n{ERROR_LINE}\n" for i in (stuck_ok, stuck_shard, stuck_rescued, pending,
                                                        running, missing, never["id"])}
    logs[stuck_nolog] = "Traceback\nRuntimeError: CUDA out of memory\n"

    tmp = Path(tempfile.mkdtemp(prefix="a-rescue-select-"))
    try:
        qdir = setup(tmp, statuses, logs, extra_jobs=[never])
        results = tmp / "process/results/xlpt"
        cells = {job["id"]: rq._cell(job) for phase in manifest["phases"] for job in phase["jobs"]}

        def shard(job_id, suffix=""):
            cell = cells[job_id]
            path = (results / f"{cell['score_dir']}{suffix}" / rq.ARMS[cell["arm"]]["slug"]
                    / f"seed-{cell['seed']}.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

        shard(stuck_shard)
        shard(stuck_rescued, rq.RESCUE_SUFFIX)

        selection = select(tmp, qdir)
        chosen = [entry["id"] for entry in selection["selected"]]
        reasons = {entry["id"]: entry["reason"] for entry in selection["not_rescued"]}
        check(chosen == [stuck_ok], f"only the clean stuck parse-error cell may be selected, got {chosen}")
        check(reasons.get(never["id"]) == "seed is not a scheduled block-A seed",
              "seed 20260829 must never be selected")
        check(reasons.get(stuck_shard) == "canonical shard exists", "a canonical shard must block the cell")
        check(reasons.get(stuck_rescued) == "rescue shard exists", "a rescue shard must block the cell")
        check(reasons.get(stuck_nolog) == "log does not show the frozen item-parse error",
              "a crash without the parse error must not be rescued")
        for job_id in (pending, running, missing):
            check(reasons.get(job_id) == "queue state is not 'stuck'", f"{job_id} must not be rescued")

        queue = json.loads((qdir / "a_rescue_queue.json").read_text(encoding="utf-8"))
        jobs = [job for phase in queue["phases"] for job in phase["jobs"]]
        check(len(jobs) == 1 and "run_rater_item_missing" in jobs[0]["cmd"]
              and "run_rater_scoring" not in jobs[0]["cmd"]
              and jobs[0]["cmd"].endswith("--namespace rescue")
              and f"{rq.RESCUE_SUFFIX}/" in jobs[0]["expected_output"],
              "the rescue job must run the item-missing wrapper in the rescue namespace")
        check((qdir / "a_rescue_queue_state.queued.json").is_file(), "the rescue state must be written")

        # The rescued cell now has a shard: a re-selection is empty and must drop the stale queue.
        shard(stuck_ok, rq.RESCUE_SUFFIX)
        selection = select(tmp, qdir)
        check(selection["selected"] == [], "a re-selection after the rescue must select nothing")
        check(not (qdir / "a_rescue_queue.json").exists()
              and not (qdir / "a_rescue_queue_state.queued.json").exists(),
              "an empty selection must remove the stale rescue queue and state")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: A rescue selection (stuck + parse error + scheduled seed + no shard only; "
          "20260829 never; stale queue removed on empty selection)")


if __name__ == "__main__":
    main()
