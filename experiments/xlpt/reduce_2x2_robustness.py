#!/usr/bin/env python3
"""Robustness companions to the exploratory en/zh 2x2 contrast (descriptive, CPU only).

``reduce_2x2_contrast.py`` pairs high and low within each cell, so the four cells of one
checkpoint can rest on slightly different complete-case scenario sets. This reducer reports,
per checkpoint and rated trait:

* ``cellwise``            -- the primary estimand again (within-cell common-complete pairs,
  equal seed weights), recomputed here so the two files can be compared;
* ``all_cell_mask``       -- the same contrasts restricted to (seed, scenario) pairs that are
  complete in BOTH poles of EVERY cell of that checkpoint, with the joint seed-cluster bootstrap;
* ``denominators`` / ``missing`` -- pairs per cell, pairs in the all-cell mask, and every missing
  trait score by cell, pole, seed and scenario;
* ``missing_score_bounds`` -- each cell separation and linear contrast when every missing trait
  score is set to the rubric extremes (1 or 7) in the direction that minimises / maximises it;
* ``leave_one_scenario_out`` -- range of every contrast when one scenario is removed;
* ``crossed_seed_scenario_bootstrap`` -- percentile interval when seed clusters and scenarios are
  resampled independently (scenarios are crossed with seeds), on the all-cell mask;
* ``average_matched_advantage`` -- interaction / 2, i.e. mean of the two language-matched cells
  minus mean of the two mismatched cells.

Every interval is a pointwise exploratory interval, conditional on the exposed scenario bank,
the prompt renderings and the frozen rater. No gate is read or written.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_pole_rater as spr  # noqa: E402
from reduce_2x2_contrast import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CELLS, CELL_LABEL  # noqa: E402

RUBRIC_MIN, RUBRIC_MAX = 1.0, 7.0

LINEAR = {
    "d_effect_at_p_en": {"en_en": 1, "en_zh": -1},
    "d_effect_at_p_zh": {"zh_en": 1, "zh_zh": -1},
    "p_effect_at_d_en": {"en_en": 1, "zh_en": -1},
    "p_effect_at_d_zh": {"en_zh": 1, "zh_zh": -1},
    "interaction_match_minus_mismatch": {"en_en": 1, "zh_zh": 1, "en_zh": -1, "zh_en": -1},
    "average_matched_advantage": {"en_en": 0.5, "zh_zh": 0.5, "en_zh": -0.5, "zh_en": -0.5},
}
RATIOS = {
    "ratio_zh_zh_over_en_en": ("zh_zh", "en_en"),
    "ratio_en_zh_over_en_en": ("en_zh", "en_en"),
    "ratio_zh_en_over_en_en": ("zh_en", "en_en"),
}


def contrasts(sep: dict) -> dict:
    """sep: cell -> scalar or array of mean separations. Works elementwise on arrays."""
    out = {f"sep_{cell}": value for cell, value in sep.items()}
    for name, signs in LINEAR.items():
        if signs.keys() <= sep.keys():
            out[name] = sum(sign * sep[cell] for cell, sign in signs.items())
    for name, (num, den) in RATIOS.items():
        if num in sep and den in sep:
            out[name] = sep[num] / sep[den]
    return out


def interval(values) -> list:
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 6), round(float(high), 6)]


def seed_mean(diff):
    """diff: (cell, seed, scenario) with NaN for excluded pairs -> (cell, seed)."""
    return np.nanmean(diff, axis=2)


def reduce_block(high, low, cells, seeds, scenarios, rng) -> dict:
    """high / low: (cell, seed, scenario) arrays of trait scores, NaN where missing."""
    diff = high - low
    mask = ~np.isnan(diff).any(axis=0)                      # (seed, scenario) complete everywhere
    masked = np.where(mask[None], diff, np.nan)

    def point(d):
        per_seed = seed_mean(d)
        return contrasts({cell: per_seed[i].mean() for i, cell in enumerate(cells)})

    cellwise = point(diff)
    all_mask = point(masked)

    # joint seed-cluster bootstrap on the all-cell mask
    per_seed = seed_mean(masked)
    idx = rng.integers(0, len(seeds), size=(BOOTSTRAP_DRAWS, len(seeds)))
    seed_draws = contrasts({cell: per_seed[i][idx].mean(axis=1) for i, cell in enumerate(cells)})

    # crossed bootstrap: seeds and scenarios resampled independently
    crossed = {name: np.empty(BOOTSTRAP_DRAWS) for name in all_mask}
    for b in range(BOOTSTRAP_DRAWS):
        s = rng.integers(0, len(seeds), size=len(seeds))
        k = rng.integers(0, len(scenarios), size=len(scenarios))
        sample = masked[:, s][:, :, k]
        stats = contrasts({cell: np.nanmean(sample[i], axis=1).mean() for i, cell in enumerate(cells)})
        for name, value in stats.items():
            crossed[name][b] = value

    # leave one scenario out, on the all-cell mask
    loso = {name: [] for name in all_mask}
    for k in range(len(scenarios)):
        stats = point(np.delete(masked, k, axis=2))
        for name, value in stats.items():
            loso[name].append(value)

    # bounds under worst-case values for every missing trait score, all scenarios kept
    def bounded(direction):
        hi_fill, lo_fill = (RUBRIC_MAX, RUBRIC_MIN) if direction > 0 else (RUBRIC_MIN, RUBRIC_MAX)
        d = np.where(np.isnan(high), hi_fill, high) - np.where(np.isnan(low), lo_fill, low)
        return d.mean(axis=2).mean(axis=1)
    upper, lower = bounded(+1), bounded(-1)
    bounds = {f"sep_{cell}": [round(float(lower[i]), 6), round(float(upper[i]), 6)]
              for i, cell in enumerate(cells)}
    position = {cell: i for i, cell in enumerate(cells)}
    for name, signs in LINEAR.items():
        if signs.keys() <= position.keys():
            lo = sum(sign * (lower if sign > 0 else upper)[position[c]] for c, sign in signs.items())
            hi = sum(sign * (upper if sign > 0 else lower)[position[c]] for c, sign in signs.items())
            bounds[name] = [round(float(lo), 6), round(float(hi), 6)]

    missing = []
    for i, cell in enumerate(cells):
        for pole, array in (("high", high), ("low", low)):
            for s, k in zip(*np.nonzero(np.isnan(array[i]))):
                missing.append({"cell": cell, "pole": pole, "seed": seeds[s],
                                "scenario": scenarios[k]})

    estimates = {}
    for name in all_mask:
        values = loso[name]
        estimates[name] = {
            "cellwise_estimate": round(float(cellwise[name]), 6),
            "all_cell_mask_estimate": round(float(all_mask[name]), 6),
            "all_cell_mask_minus_cellwise": round(float(all_mask[name] - cellwise[name]), 6),
            "ci95_seed_bootstrap_all_cell_mask": interval(seed_draws[name]),
            "ci95_crossed_seed_scenario_bootstrap": interval(crossed[name]),
            "leave_one_scenario_out_range": [round(float(min(values)), 6),
                                             round(float(max(values)), 6)],
            "leave_one_scenario_out_argmin": scenarios[int(np.argmin(values))],
            "leave_one_scenario_out_argmax": scenarios[int(np.argmax(values))],
        }
    return {
        "cells_present": list(cells),
        "n_seed_clusters": len(seeds),
        "n_scenarios": len(scenarios),
        "denominators": {
            "pairs_possible_per_cell": len(seeds) * len(scenarios),
            "pairs_complete_per_cell": {cell: int((~np.isnan(diff[i])).sum())
                                        for i, cell in enumerate(cells)},
            "pairs_in_all_cell_mask": int(mask.sum()),
        },
        "missing_trait_scores": missing,
        "missing_score_bounds": bounds,
        "estimates": estimates,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--score-root", default="process/results/xlpt")
    ap.add_argument("--out", default="process/results/xlpt/XLPT_2X2_ROBUSTNESS.json")
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    score_root = project / args.score_root

    scores = {}                     # (checkpoint, trait) -> cell -> pole -> {(seed, scenario): score}
    for cell, pair in CELLS.items():
        high_phase, low_phase, _, _ = spr.POLE_PAIRS[pair]
        _, _, unit = spr.collect(score_root, "seed-*.jsonl", phases=(high_phase, low_phase))
        for (phase, checkpoint, trait, seed, scenario), score in unit.items():
            pole = "high" if phase == high_phase else "low"
            scores.setdefault((checkpoint, trait), {}).setdefault(cell, {"high": {}, "low": {}})[
                pole][(seed, scenario)] = score

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = {}
    for (checkpoint, trait), by_cell in sorted(scores.items()):
        cells = [cell for cell in CELLS if cell in by_cell]
        keys = [set(by_cell[cell][pole]) for cell in cells for pole in ("high", "low")]
        seeds = sorted(set.intersection(*({seed for seed, _ in k} for k in keys)))
        scenarios = sorted(set.union(*({scenario for _, scenario in k} for k in keys)))

        def grid(pole):
            out = np.full((len(cells), len(seeds), len(scenarios)), np.nan)
            for i, cell in enumerate(cells):
                for s, seed in enumerate(seeds):
                    for k, scenario in enumerate(scenarios):
                        value = by_cell[cell][pole].get((seed, scenario))
                        if value is not None:
                            out[i, s, k] = value
            return out

        blocks[f"{checkpoint}|{trait}"] = reduce_block(
            grid("high"), grid("low"), cells, seeds, scenarios, rng)

    out = {
        "schema_version": 1,
        "verdict": "DESCRIPTIVE_ONLY",
        "eligible_for_gate": False,
        "scope_note": (
            "Robustness companions to XLPT_2X2_CONTRAST.json. Intervals are pointwise and "
            "exploratory, conditional on the exposed 50-scenario bank, the prompt renderings and "
            "the frozen rater; the crossed bootstrap adds resampling of those 50 scenarios only."
        ),
        "rubric_range": [RUBRIC_MIN, RUBRIC_MAX],
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                      "generator": "numpy default_rng, consumed in sorted block order"},
        "cell_labels": CELL_LABEL,
        "blocks": blocks,
        "reducer_sha256": spr.sha256_file(Path(__file__).resolve()),
    }
    (project / args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    for key, block in blocks.items():
        d = block["denominators"]
        print(f"{key:40s} mask {d['pairs_in_all_cell_mask']}/{d['pairs_possible_per_cell']} "
              f"missing {len(block['missing_trait_scores'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
