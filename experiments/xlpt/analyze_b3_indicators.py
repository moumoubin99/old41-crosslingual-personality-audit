"""Contrast the B2 apparatus control against B1 on the rater-free behavioral indicators.

This is a deterministic, CPU-only reducer over the indicator files that
``compute_rater_free_indicators.py`` already wrote. It loads no model, takes no
accelerator, and writes into no registered gate artifact.

What it computes
----------------
The two phases ``b1-neutral`` and ``b2-compliance`` were generated on the **same** 90
evaluation scenarios under the **same** 24-seed block, differing only in the system text
(``NEUTRAL`` vs ``COMPLIANCE_CONTROL``). That makes the comparison paired at the level of
the registered resampling unit: the seed cluster. For each indicator the tool therefore

1. averages the indicator over the 90 scenarios **within** each seed cluster and phase,
2. takes the within-cluster difference ``b2 - b1``, giving 24 independent cluster values,
3. reports the mean difference with a percentile bootstrap interval resampled **over
   clusters**, plus the sign count, so the reader sees dispersion rather than one number.

What it deliberately does NOT do
--------------------------------
* It decides no qualification gate and writes no ``QUALIFICATION_GATE.json``. The rule-F
  gate never ran, so nothing here is an admission statistic.
* It is **not** the C1 transfer-matrix reading. Both phases are English dialogue with an
  English persona: the language switch that C1 is about was never applied, so no
  cross-lingual quantity exists in this surface at all.
* It applies no multiplicity correction across the four indicators and claims none. The
  registered single-step max-T family belongs to the gate, not to this readout, and
  borrowing its name for a different set of limbs would be a different procedure.

The output is therefore labelled ``exploratory`` and carries the reason in the file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

INDICATORS = (
    "refusal_turn_rate",
    "escalation_turn_rate",
    "apology_turn_rate",
    "mean_agent_words_per_turn",
)
BASELINE_PHASE = "b1-neutral"
CONTRAST_PHASE = "b2-compliance"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260920          # fixed so the interval is reproducible byte-for-byte
INTERVAL_MASS = 0.95


def load_phase(directory: Path) -> dict:
    """Return {seed: {scenario_id: {indicator: value}}} for one phase directory."""
    if not directory.is_dir():
        raise ValueError(f"indicator directory does not exist: {directory}")
    out: dict = {}
    for path in sorted(directory.glob("seed-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path.name} carries no records")
        by_scenario: dict = {}
        seeds = set()
        for record in records:
            if record.get("eligible_for_analysis") is not True:
                raise ValueError(f"{path.name} carries a record not eligible for analysis")
            scenario = record["scenario_id"]
            if scenario in by_scenario:
                raise ValueError(f"{path.name} repeats scenario {scenario}")
            seeds.add(int(record["seed"]))
            by_scenario[scenario] = {name: float(record[name]) for name in INDICATORS}
        if len(seeds) != 1:
            raise ValueError(f"{path.name} mixes seeds {sorted(seeds)}; one file is one cluster")
        seed = seeds.pop()
        if seed in out:
            raise ValueError(f"seed {seed} appears in more than one indicator file")
        out[seed] = by_scenario
    if not out:
        raise ValueError(f"no indicator files under {directory}")
    return out


def cluster_differences(baseline: dict, contrast: dict) -> dict:
    """One paired difference per seed cluster per indicator, on the shared scenario set."""
    if set(baseline) != set(contrast):
        raise ValueError(
            "the two phases do not share one seed block; a paired cluster contrast needs "
            f"the same seeds (baseline-only {sorted(set(baseline) - set(contrast))}, "
            f"contrast-only {sorted(set(contrast) - set(baseline))})"
        )
    differences = {name: [] for name in INDICATORS}
    per_cluster = []
    for seed in sorted(baseline):
        left, right = baseline[seed], contrast[seed]
        if set(left) != set(right):
            raise ValueError(f"seed {seed} does not cover the same scenarios in both phases")
        row = {"seed": seed, "scenarios": len(left)}
        for name in INDICATORS:
            base_mean = statistics.fmean(left[s][name] for s in sorted(left))
            cont_mean = statistics.fmean(right[s][name] for s in sorted(right))
            row[name] = {
                BASELINE_PHASE: base_mean,
                CONTRAST_PHASE: cont_mean,
                "difference": cont_mean - base_mean,
            }
            differences[name].append(cont_mean - base_mean)
        per_cluster.append(row)
    return {"per_cluster": per_cluster, "differences": differences}


def bootstrap_interval(values: list, draws: int, seed: int, mass: float) -> dict:
    """Percentile interval from resampling the clusters themselves, with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(draws):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    low = means[int((1.0 - mass) / 2.0 * draws)]
    high = means[min(draws - 1, int((1.0 + mass) / 2.0 * draws))]
    return {"draws": draws, "mass": mass, "low": low, "high": high}


def summarize(differences: dict) -> dict:
    out = {}
    for name, values in differences.items():
        positive = sum(1 for value in values if value > 0)
        negative = sum(1 for value in values if value < 0)
        out[name] = {
            "clusters": len(values),
            "mean_difference": statistics.fmean(values),
            "sd_difference": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median_difference": statistics.median(values),
            "clusters_positive": positive,
            "clusters_negative": negative,
            "clusters_zero": len(values) - positive - negative,
            "bootstrap": bootstrap_interval(values, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, INTERVAL_MASS),
            "interval_excludes_zero": None,
        }
        interval = out[name]["bootstrap"]
        out[name]["interval_excludes_zero"] = interval["low"] > 0.0 or interval["high"] < 0.0
    return out


def build_report(baseline_dir: Path, contrast_dir: Path) -> dict:
    baseline = load_phase(baseline_dir)
    contrast = load_phase(contrast_dir)
    paired = cluster_differences(baseline, contrast)
    return {
        "schema_version": 1,
        "tool": "experiments/xlpt/analyze_b3_indicators.py",
        "kind": "b1-b2-rater-free-indicator-contrast",
        "status": "exploratory",
        "gate_decision": "NOT_DECIDED_BY_THIS_TOOL",
        "is_c1_transfer_reading": False,
        "why_not_c1": (
            "both phases are English dialogue under an English persona; the language switch "
            "that claim C1 is about was never applied, so this surface carries no "
            "cross-lingual quantity"
        ),
        "multiplicity_correction": "none applied; the registered max-T family belongs to the gate",
        "uses_gpu": False,
        "trains_or_updates_weights": False,
        "baseline_phase": BASELINE_PHASE,
        "contrast_phase": CONTRAST_PHASE,
        "resampling_unit": "seed cluster",
        "clusters": len(paired["per_cluster"]),
        "scenarios_per_cluster": paired["per_cluster"][0]["scenarios"],
        "indicators": list(INDICATORS),
        "summary": summarize(paired["differences"]),
        "per_cluster": paired["per_cluster"],
        "note": (
            "Descriptive contrast of an apparatus control against its neutral reference. The "
            "rule-F qualification gate never ran, so no result here is a gate-admitted "
            "confirmatory number."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Paired seed-cluster contrast of B2 against B1 on the rater-free indicators."
    )
    ap.add_argument("--project", required=True)
    ap.add_argument("--baseline-dir", default="process/results/xlpt/b3-rater-free/b1-neutral")
    ap.add_argument("--contrast-dir", default="process/results/xlpt/b3-rater-free/b2-compliance")
    ap.add_argument("--output-json", default="process/results/xlpt/B3_INDICATOR_CONTRAST.json")
    args = ap.parse_args(argv)

    root = Path(args.project).resolve()
    inside = lambda relative: (root / relative).resolve()  # noqa: E731
    for relative in (args.baseline_dir, args.contrast_dir, args.output_json):
        if not str(inside(relative)).startswith(str(root)):
            raise ValueError(f"path must stay inside the project: {relative}")

    report = build_report(inside(args.baseline_dir), inside(args.contrast_dir))
    report["project"] = str(root)
    report["inputs"] = {"baseline": args.baseline_dir, "contrast": args.contrast_dir}

    output = inside(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)

    print(json.dumps({
        "output": str(output),
        "clusters": report["clusters"],
        "indicators": {
            name: round(block["mean_difference"], 4) for name, block in report["summary"].items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
