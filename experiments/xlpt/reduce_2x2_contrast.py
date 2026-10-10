#!/usr/bin/env python3
"""Exploratory en/zh 2x2 contrast of rater-score pole separation (FOLLOWUP_PLAN_2026-09-28 A, BC).

Reads the four frozen pole-score cells through ``summarize_pole_rater.collect`` (one call per
cell, so apparatus digests frozen under different execution units are never mixed into one
check) and reports, per checkpoint and rated trait:

* sep(P, D) = mean(high) - mean(low), taken per seed cluster over the common-complete scenarios
  of that cell, exactly as the per-cell summaries do;
* the D simple effects, the P simple effects and the interaction
  [sep(en,en) + sep(zh,zh)] - [sep(en,zh) + sep(zh,en)];
* ratios of separations as a scale-free companion.

The primary interval is the joint paired seed-cluster bootstrap of the plan: the seed clusters
are resampled jointly across all cells of a checkpoint, fixed seed 20260928, 9 999 draws,
percentile interval. The normal-approximation interval of the per-cell summaries is reported
beside it.

Everything here is DESCRIPTIVE. L = 2 is VOID under the idea card, the scenarios are exposed
qualification material, and the rater is not validated against human labels (H_PUBLIC_SUMMARY).
No gate is read or written.
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_pole_rater as spr  # noqa: E402

BOOTSTRAP_SEED = 20260928
BOOTSTRAP_DRAWS = 9999
CELLS = {
    "en_en": "en",
    "en_zh": "zh-den",
    "zh_en": "bc-den",
    "zh_zh": "bc-dzh",
}
CELL_LABEL = {
    "en_en": "P=en,D=en",
    "en_zh": "P=en,D=zh",
    "zh_en": "P=zh(v4 reviewed),D=en",
    "zh_zh": "P=zh(v4 reviewed),D=zh",
}


def seed_separations(unit: dict, high_phase: str, low_phase: str) -> dict:
    """(checkpoint, trait) -> {seed: sep} over the common-complete scenarios of one cell."""
    common = {}
    for (phase, checkpoint, trait, seed, scenario), score in unit.items():
        if phase != high_phase:
            continue
        low = unit.get((low_phase, checkpoint, trait, seed, scenario))
        if score is None or low is None:
            continue
        common.setdefault((checkpoint, trait, seed), []).append((score, low))
    out = {}
    for (checkpoint, trait, seed), pairs in common.items():
        out.setdefault((checkpoint, trait), {})[seed] = (
            statistics.fmean(high for high, _ in pairs)
            - statistics.fmean(low for _, low in pairs)
        )
    return out


def _percentile(sorted_values: list, q: float) -> float:
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator else None


def statistics_of(means: dict) -> dict:
    """Contrasts from the per-cell mean separations available for one checkpoint and trait."""
    out = {f"sep_{cell}": value for cell, value in means.items()}
    if {"en_en", "en_zh"} <= means.keys():
        out["d_effect_at_p_en"] = means["en_en"] - means["en_zh"]
        out["ratio_en_zh_over_en_en"] = _ratio(means["en_zh"], means["en_en"])
    if {"zh_en", "zh_zh"} <= means.keys():
        out["d_effect_at_p_zh"] = means["zh_en"] - means["zh_zh"]
        out["ratio_zh_zh_over_zh_en"] = _ratio(means["zh_zh"], means["zh_en"])
    if {"en_en", "zh_en"} <= means.keys():
        out["p_effect_at_d_en"] = means["en_en"] - means["zh_en"]
        out["ratio_zh_en_over_en_en"] = _ratio(means["zh_en"], means["en_en"])
    if {"en_zh", "zh_zh"} <= means.keys():
        out["p_effect_at_d_zh"] = means["en_zh"] - means["zh_zh"]
        out["ratio_zh_zh_over_en_zh"] = _ratio(means["zh_zh"], means["en_zh"])
    if len(means) == 4:
        out["interaction_match_minus_mismatch"] = (
            (means["en_en"] + means["zh_zh"]) - (means["en_zh"] + means["zh_en"])
        )
        out["d_main_effect"] = 0.5 * (out["d_effect_at_p_en"] + out["d_effect_at_p_zh"])
        out["p_main_effect"] = 0.5 * (out["p_effect_at_d_en"] + out["p_effect_at_d_zh"])
        out["ratio_zh_zh_over_en_en"] = _ratio(means["zh_zh"], means["en_en"])
    return out


def reduce_cell_block(by_cell: dict, rng: random.Random) -> dict:
    """by_cell: cell -> {seed: sep} for one checkpoint and trait."""
    seeds = sorted(set.intersection(*(set(values) for values in by_cell.values())))
    dropped = {cell: sorted(set(values) - set(seeds)) for cell, values in by_cell.items()}
    point = statistics_of(
        {cell: statistics.fmean(values[seed] for seed in seeds) for cell, values in by_cell.items()}
    )
    draws = {name: [] for name in point}
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [rng.choice(seeds) for _ in seeds]
        stats = statistics_of(
            {cell: statistics.fmean(values[seed] for seed in sample)
             for cell, values in by_cell.items()}
        )
        for name, value in stats.items():
            if value is not None:
                draws[name].append(value)
    estimates = {}
    for name, value in point.items():
        ordered = sorted(draws[name])
        entry = {
            "estimate": None if value is None else round(value, 6),
            "ci95_joint_seed_bootstrap": (
                [round(_percentile(ordered, 0.025), 6), round(_percentile(ordered, 0.975), 6)]
                if ordered else None
            ),
        }
        if not name.startswith("ratio_"):
            # per-seed linear combination, so the normal approximation of the per-cell summaries
            # applies; a ratio of means has no per-seed analogue
            signs = _seed_combination(name)
            per_seed = [sum(sign * by_cell[cell][seed] for cell, sign in signs.items())
                        for seed in seeds]
            sd = statistics.stdev(per_seed) if len(per_seed) > 1 else None
            if sd:
                half = 1.96 * sd / math.sqrt(len(per_seed))
                mean = statistics.fmean(per_seed)
                entry["ci95_normal_approx"] = [round(mean - half, 6), round(mean + half, 6)]
                entry["share_seeds_positive"] = round(
                    sum(1 for v in per_seed if v > 0) / len(per_seed), 6)
        estimates[name] = entry
    return {
        "n_seed_clusters": len(seeds),
        "cells_present": sorted(by_cell),
        "seeds_dropped_not_in_all_cells": {c: s for c, s in dropped.items() if s},
        "estimates": estimates,
    }


def _seed_combination(name: str) -> dict:
    table = {
        "d_effect_at_p_en": {"en_en": 1, "en_zh": -1},
        "d_effect_at_p_zh": {"zh_en": 1, "zh_zh": -1},
        "p_effect_at_d_en": {"en_en": 1, "zh_en": -1},
        "p_effect_at_d_zh": {"en_zh": 1, "zh_zh": -1},
        "interaction_match_minus_mismatch": {"en_en": 1, "zh_zh": 1, "en_zh": -1, "zh_en": -1},
        "d_main_effect": {"en_en": 0.5, "en_zh": -0.5, "zh_en": 0.5, "zh_zh": -0.5},
        "p_main_effect": {"en_en": 0.5, "zh_en": -0.5, "en_zh": 0.5, "zh_zh": -0.5},
    }
    if name.startswith("sep_"):
        return {name[len("sep_"):]: 1}
    return table[name]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--score-root", default="process/results/xlpt")
    ap.add_argument("--out", default="process/results/xlpt/XLPT_2X2_CONTRAST.json")
    args = ap.parse_args(argv)

    project = Path(args.project).resolve()
    score_root = project / args.score_root
    separations = {}
    inputs = {}
    for cell, pair in CELLS.items():
        high_phase, low_phase, summary_path, _ = spr.POLE_PAIRS[pair]
        shards, _, unit = spr.collect(score_root, "seed-*.jsonl", phases=(high_phase, low_phase))
        inputs[cell] = {
            "label": CELL_LABEL[cell],
            "high_phase": high_phase,
            "low_phase": low_phase,
            "shards": len(shards),
            "rows": sum(entry["rows"] for entry in shards),
            "incomplete_traits": sum(entry["incomplete_traits"] for entry in shards),
            "per_cell_summary": summary_path,
            "per_cell_summary_sha256": spr.sha256_file(project / summary_path),
        }
        for key, by_seed in seed_separations(unit, high_phase, low_phase).items():
            separations.setdefault(key, {})[cell] = by_seed

    # one generator, consumed in sorted key order, so the output is reproducible
    rng = random.Random(BOOTSTRAP_SEED)
    contrasts = {
        f"{checkpoint}|{trait}": reduce_cell_block(separations[(checkpoint, trait)], rng)
        for checkpoint, trait in sorted(separations)
    }
    out = {
        "schema_version": 1,
        "verdict": "DESCRIPTIVE_ONLY",
        "eligible_for_gate": False,
        "scope_note": (
            "Exploratory en/zh 2x2 on exposed qualification-seed pole surfaces. L = 2 is VOID "
            "under the idea card, so nothing here is H1 or Delta_mis_bar evidence. Scores are an "
            "English-instrument LLM-rater response; H_PUBLIC_SUMMARY shows no concordance of "
            "that score with public human Extraversion labels. sep = mean(high) - mean(low) = "
            "2*E* and is never compared with card thresholds."
        ),
        "bootstrap": {
            "kind": "joint paired seed-cluster percentile",
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "resampling_unit": "seed cluster, shared by every cell of a checkpoint",
        },
        "inputs": inputs,
        "contrasts": contrasts,
        "reducer_sha256": spr.sha256_file(Path(__file__).resolve()),
    }
    out_path = project / args.out
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(project)}: {len(contrasts)} checkpoint|trait blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
