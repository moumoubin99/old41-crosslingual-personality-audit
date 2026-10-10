"""Decide, from the real B0 rater shards, whether the rule-F gate can be run at all.

This is the step immediately downstream of ``summarize_b0_rater.py``. It answers one
question and writes down the answer with its reasons:

    can ``QUALIFICATION_GATE.json``'s results block be computed from what B0 actually
    produced?

It is deterministic, CPU-only and read-only with respect to every registered artifact.
It **never** writes ``QUALIFICATION_GATE.json``, never writes ``ELIGIBLE_PAIRS.json``,
decides no admission and produces no gate statistic. Its output is a readiness report:
``gate_runnable`` plus the enumerated blockers.

The requirements it checks are read from ``refine-logs/QUALIFICATION_GATE.json`` and from
``run_rater_scoring.QUALIFICATION_SEEDS`` -- they are not restated by hand here:

* **G = 24 seed clusters.** Section I fixes the resampling unit at G = 24 with V_hat on
  23 df at the R17-F2(b) calibrated level q* = 0.965. A pass run with fewer completed
  seeds is not the registered procedure, and this tool refuses to call the gate runnable
  on a short block rather than quietly re-deriving a critical value at a different df.
* **The 12 checkpoint-language cells.** The admission map is language-level and L = 3 is
  mandatory (R16-F3): a checkpoint keeps all three languages or is not admitted.
* **The pole contrast.** G5's null is ``E*(p,p) <= f_diag = 0.18`` on the diagonal cell
  effect. A surface generated entirely under one setting carries no induced-trait
  contrast, so no diagonal effect exists to compare against f_diag.
* **The single-step max-T family.** G5 runs ONE one-sided signed critical value over its
  enumerated limbs. A subset of limbs cannot be given a critical value of its own without
  changing the family, so partial data yields no admission -- not even a provisional one.

``gate_runnable`` is derived from the data, not asserted: a surface that did cover every
registered requirement would come back runnable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from .run_rater_scoring import QUALIFICATION_SEEDS
except ImportError:  # direct-script execution
    try:
        from run_rater_scoring import QUALIFICATION_SEEDS
    except ImportError:
        QUALIFICATION_SEEDS = tuple(range(20260811, 20260811 + 24))

SEED_CLUSTERS_REQUIRED = 24          # section I / R17-F2: G = 24, V_hat on 23 df
CHECKPOINTS_REQUIRED = 4             # QUALIFICATION_GATE.json cells.checkpoints
LANGUAGES_REQUIRED = 3               # cells.languages_L3, L = 3 mandatory (R16-F3)
GATE_IDS = ("G1_competence", "G2_validity_F2", "G3_instrument_language_invariance",
            "G4_perturbation_sensitivity", "G5_diagonal_admission")
NEUTRAL_SETTING = "NEUTRAL"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required input does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def check_summary(summary: dict) -> None:
    """Refuse a summary that is not the descriptive reduction this tool consumes."""
    if summary.get("kind") != "b0-english-rater-descriptive-summary":
        raise ValueError("input summary is not a B0 English rater descriptive summary")
    if summary.get("gate_decision") != "NOT_DECIDED_BY_THIS_TOOL":
        raise ValueError("input summary already carries a gate decision; refusing to read it")
    if summary.get("eligible_for_analysis") is not False:
        raise ValueError("input summary claims analysis eligibility; the B0 surface is gate-only")
    for field in ("seeds_summarised", "distinct_checkpoints", "distinct_settings"):
        if not isinstance(summary.get(field), list):
            raise ValueError(f"input summary is missing {field}")


def merge_summaries(summaries: list) -> dict:
    """Union one descriptive summary per checkpoint into a single surface view.

    ``summarize_b0_rater.py`` refuses to place one seed in two shards, so a surface that
    covers several checkpoints under the same seed block is necessarily reduced one
    checkpoint at a time. This joins those reductions without relaxing anything: the
    frozen digests must be identical, so the union still describes one apparatus, and no
    checkpoint may arrive twice, so no arm is counted twice.
    """
    if not summaries:
        raise ValueError("no summary was supplied")
    for summary in summaries:
        check_summary(summary)
    if len(summaries) == 1:
        return summaries[0]

    digests = {json.dumps(s.get("frozen_digests"), sort_keys=True) for s in summaries}
    if len(digests) != 1:
        raise ValueError(
            "the supplied summaries do not share one frozen apparatus; their "
            "prompt-config / freeze-receipt / rater-runtime digests differ"
        )

    checkpoints, seeds, settings, rows = [], set(), set(), 0
    for summary in summaries:
        for entry in summary["distinct_checkpoints"]:
            name = str(entry)
            if name in checkpoints:
                raise ValueError(f"checkpoint {name} appears in more than one summary")
            checkpoints.append(name)
        seeds.update(int(seed) for seed in summary["seeds_summarised"])
        settings.update(str(entry) for entry in summary["distinct_settings"])
        rows += int(summary.get("rows_total") or 0)

    merged = dict(summaries[0])
    merged.update({
        "distinct_checkpoints": sorted(checkpoints),
        "seeds_summarised": sorted(seeds),
        "seed_count": len(seeds),
        "distinct_settings": sorted(settings),
        "rows_total": rows,
        "merged_from": [s.get("shard_dir") or s.get("distinct_checkpoints") for s in summaries],
    })
    return merged


def check_gate_spec(gate: dict) -> dict:
    """Read the registered requirements and refuse to touch an already-decided gate."""
    if gate.get("artifact") != "QUALIFICATION_GATE.json":
        raise ValueError("gate specification is not QUALIFICATION_GATE.json")
    if gate.get("results") is not None:
        raise ValueError(
            "QUALIFICATION_GATE.json already carries a results block; this tool never "
            "rewrites or re-reads a decided gate"
        )
    cells = gate.get("cells") or {}
    checkpoints = cells.get("checkpoints")
    languages = cells.get("languages_L3")
    if not isinstance(checkpoints, list) or len(checkpoints) != CHECKPOINTS_REQUIRED:
        raise ValueError("gate specification does not enumerate the 4 registered checkpoints")
    if not isinstance(languages, list) or len(languages) != LANGUAGES_REQUIRED:
        raise ValueError("gate specification does not enumerate the 3 registered languages")
    registered = [entry.get("id") for entry in gate.get("gates") or []]
    missing = [gate_id for gate_id in GATE_IDS if gate_id not in registered]
    if missing:
        raise ValueError(f"gate specification is missing registered gates: {missing}")
    return {"checkpoints": checkpoints, "languages_L3": languages, "gates": registered}


def seed_block_status(seeds_present: list) -> dict:
    registered = list(QUALIFICATION_SEEDS)
    present = sorted({int(seed) for seed in seeds_present})
    unregistered = [seed for seed in present if seed not in registered]
    if unregistered:
        raise ValueError(
            f"score shards carry seeds outside the registered qualification block: {unregistered}"
        )
    missing = [seed for seed in registered if seed not in present]
    return {
        "registered_block": [registered[0], registered[-1]],
        "seed_clusters_required": SEED_CLUSTERS_REQUIRED,
        "seed_clusters_available": len(present),
        "missing_seeds": missing,
        "complete": len(present) == SEED_CLUSTERS_REQUIRED and not missing,
    }


def surface_status(summary: dict, spec: dict) -> dict:
    checkpoints = sorted(str(entry) for entry in summary["distinct_checkpoints"])
    settings = sorted(str(entry) for entry in summary["distinct_settings"])
    covered = [entry for entry in spec["checkpoints"] if entry in checkpoints]
    pole_settings = [entry for entry in settings if entry != NEUTRAL_SETTING]
    return {
        "checkpoints_observed": checkpoints,
        "checkpoints_registered": spec["checkpoints"],
        "checkpoints_covered": covered,
        "checkpoint_coverage": f"{len(covered)}/{CHECKPOINTS_REQUIRED}",
        "languages_registered": spec["languages_L3"],
        "instrument_languages_observed": ["English"],
        "language_coverage": f"1/{LANGUAGES_REQUIRED}",
        "settings_observed": settings,
        "pole_settings_observed": pole_settings,
        "has_pole_contrast": bool(pole_settings),
        "cells_required": CHECKPOINTS_REQUIRED * LANGUAGES_REQUIRED,
        "cells_covered": len(covered) * 1,
    }


def blockers(seeds: dict, surface: dict) -> list:
    out = []
    if not seeds["complete"]:
        out.append({
            "requirement": "G = 24 seed clusters (section I, R17-F2(b): V_hat on 23 df at q* = 0.965)",
            "observed": f"{seeds['seed_clusters_available']} completed seed clusters",
            "missing": seeds["missing_seeds"],
            "consequence": (
                "the registered bootstrap has no calibrated critical value at this cluster "
                "count; recalibrating one for a short block would be a different procedure "
                "from the one filed before B0"
            ),
        })
    if surface["cells_covered"] < surface["cells_required"]:
        out.append({
            "requirement": "12 checkpoint-language cells; L = 3 mandatory, the L = 2 branch is VOID (R16-F3)",
            "observed": (
                f"checkpoints {surface['checkpoint_coverage']}, "
                f"instrument languages {surface['language_coverage']}, "
                f"cells {surface['cells_covered']}/{surface['cells_required']}"
            ),
            "consequence": (
                "the language-level admission map cannot be evaluated for any checkpoint, "
                "because no checkpoint has all three languages"
            ),
        })
    if not surface["has_pole_contrast"]:
        out.append({
            "requirement": "G5 null E*(p,p) <= f_diag = 0.18 on the diagonal cell effect",
            "observed": f"settings {surface['settings_observed']} only",
            "consequence": (
                "a single-setting surface carries no induced-trait contrast, so no diagonal "
                "effect exists to compare against f_diag; nothing here can be read as an "
                "admission statistic"
            ),
        })
    return out


def per_gate_readout(surface: dict, seeds: dict) -> dict:
    short_block = None if seeds["complete"] else (
        f"only {seeds['seed_clusters_available']} of {SEED_CLUSTERS_REQUIRED} seed clusters"
    )
    return {
        "G1_competence": {
            "computable_from_b0_rater_shards": False,
            "why": (
                "G1 measures response-language accuracy and instruction compliance on the "
                "50-prompt qualification sample. Those are properties of the generated "
                "transcripts, not of the rubric scores, so the rater shards cannot decide it."
            ),
            "also_blocked_by": short_block,
        },
        "G2_validity_F2": {
            "computable_from_b0_rater_shards": False,
            "why": (
                "G2 needs 720 blind double-annotated adjudicated transcripts with a third "
                "adjudicator. No human annotation exists on this surface."
            ),
        },
        "G3_instrument_language_invariance": {
            "computable_from_b0_rater_shards": False,
            "why": (
                "G3 compares each transcript scored under the ENGLISH instrument against the "
                "same transcript scored under its own language rendering. Every B0 row is "
                "instrument_language English, so the second half of every pair is absent."
            ),
        },
        "G4_perturbation_sensitivity": {
            "computable_from_b0_rater_shards": False,
            "why": (
                "G4 reads 360 minimally edited perturbation pairs differing by one "
                "trait-diagnostic behavior. No perturbation pairs were generated in B0."
            ),
        },
        "G5_diagonal_admission": {
            "computable_from_b0_rater_shards": False,
            "why": (
                "G5's 24-limb family runs under ONE one-sided signed single-step max-T "
                "critical value. Its limbs need a pole contrast in 12 cells; B0 supplies "
                f"{surface['settings_observed']} in {surface['cells_covered']} cell(s). A "
                "subset of limbs cannot be given a critical value of its own without "
                "changing the family."
            ),
            "also_blocked_by": short_block,
        },
    }


def build_report(summary, gate: dict) -> dict:
    summary = merge_summaries(summary if isinstance(summary, list) else [summary])
    spec = check_gate_spec(gate)
    seeds = seed_block_status(summary["seeds_summarised"])
    surface = surface_status(summary, spec)
    blocking = blockers(seeds, surface)
    return {
        "schema_version": 1,
        "tool": "experiments/xlpt/b0_gate_readiness.py",
        "kind": "b0-rule-f-gate-readiness",
        "gate_decision": "NOT_DECIDED_BY_THIS_TOOL",
        "writes_qualification_gate": False,
        "writes_eligible_pairs": False,
        "uses_gpu": False,
        "trains_or_updates_weights": False,
        "gate_runnable": not blocking,
        "blocking": blocking,
        "seed_block": seeds,
        "surface": surface,
        "per_gate": per_gate_readout(surface, seeds),
        "downstream_disposition": (
            "B1/B2/B3 stay at queued_pending_gate. Rule F requires the admitted set to be "
            "frozen into ELIGIBLE_PAIRS.json before B1 opens, and "
            "schedule/make_eligible_pairs.py --freeze refuses while QUALIFICATION_GATE.json "
            "carries no results block. Queueing them is not launching them."
        ),
        "rows_summarised": summary.get("rows_total"),
        "frozen_digests": summary.get("frozen_digests"),
        "note": (
            "Readiness only. This tool computes no gate statistic, writes into no registered "
            "gate artifact and substitutes no endpoint. A blocked gate is reported as blocked."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Report whether the rule-F qualification gate can be run on the real B0 shards."
    )
    ap.add_argument("--project", required=True)
    ap.add_argument(
        "--summary-json",
        action="append",
        help=(
            "descriptive summary to read; repeat once per checkpoint, since one summary "
            "holds one checkpoint's seed block"
        ),
    )
    ap.add_argument("--gate-json", default="refine-logs/QUALIFICATION_GATE.json")
    ap.add_argument("--output-json", default="refine-logs/B0_GATE_READINESS.json")
    args = ap.parse_args(argv)

    summary_paths = args.summary_json or ["process/results/xlpt/B0_RATER_SUMMARY.json"]

    root = Path(args.project).resolve()
    inside = lambda relative: (root / relative).resolve()  # noqa: E731
    for relative in (*summary_paths, args.gate_json, args.output_json):
        if not str(inside(relative)).startswith(str(root)):
            raise ValueError(f"path must stay inside the project: {relative}")

    summaries = [load_json(inside(relative)) for relative in summary_paths]
    report = build_report(summaries, load_json(inside(args.gate_json)))
    report["project"] = str(root)
    report["inputs"] = {"summary": summary_paths, "gate_specification": args.gate_json}

    output = inside(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)

    print(json.dumps({
        "output": str(output),
        "gate_runnable": report["gate_runnable"],
        "blocking": len(report["blocking"]),
        "seed_clusters": f"{report['seed_block']['seed_clusters_available']}/{SEED_CLUSTERS_REQUIRED}",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
