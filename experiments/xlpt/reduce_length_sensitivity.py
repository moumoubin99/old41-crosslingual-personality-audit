#!/usr/bin/env python3
"""Association of the rater score with response length, per cell (descriptive sensitivity only).

Pole and length are almost perfectly confounded (the high pole is several times longer in
every cell), so nothing here identifies a length-independent pole effect. The reducer uses the
PRIMARY estimand's units: within one cell, the (seed, scenario) pairs whose high and low
transcripts both carry a complete trait score, with equal seed-cluster weights.

Reported per cell, checkpoint and rated trait:

* ``sep`` and ``log_length_gap``  -- equal-seed-weight mean of high minus low over those pairs
  (``sep`` reproduces XLPT_2X2_CONTRAST.json);
* ``within_pole_slope_seed_fe``   -- OLS slope of score on log(agent characters) inside a pole
  with seed fixed effects, averaged over the two poles;
* ``within_pole_slope_seed_scenario_fe`` -- the same with seed AND scenario fixed effects, which
  removes scenario-level length/score association (point estimate only);
* ``extrapolated_length_association_sep`` / ``_share`` -- slope times ``log_length_gap`` and its
  ratio to ``sep``. This multiplies a within-pole association by a between-pole gap the poles do
  not share: an extrapolated, association-based sensitivity quantity, not a mediated or
  explained share, and ``sep`` minus it is not an identified length-independent effect;
* ``length_chars``                -- per-pole distribution of total agent characters;
* ``common_interval``             -- ONE interval applied to both poles: the intersection of the
  two poles' central 90% log-length ranges, [max(p05_high, p05_low), min(p95_high, p95_low)].
  If it is empty the poles' central 90% length ranges do not overlap and no within-band
  separation exists (the full ranges may still overlap: see ``full_range_overlap``). When it
  is non-empty, counts on each side, mean characters on each side and, with at least 30
  transcripts on both sides, the unpaired transcript-mean separation inside the band.

Intervals: joint seed-cluster percentile bootstrap (seed 20260928, 9 999 draws); pointwise and
exploratory, conditional on the scenario bank, renderings and frozen rater. The within-pole
slopes pool residual cross-products over observations (observation-weighted OLS), unlike ``sep``
and ``log_length_gap``, which weight seed clusters equally.

* ``full_range_overlap``          -- intersection of the two poles' full (min-max) length ranges
  and the number of transcripts of each pole inside it; tail overlap only, no separation is
  computed on it.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_pole_rater as spr  # noqa: E402
from reduce_2x2_contrast import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CELLS, CELL_LABEL  # noqa: E402

MIN_BAND = 30
QUANTILES = (0, 5, 25, 50, 75, 95, 100)


def load_lengths(root: Path, phase: str) -> dict:
    """(checkpoint_id, seed, scenario) -> log total agent characters of one generation phase."""
    out = {}
    for path in sorted((root / phase).glob("*/seed-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            chars = sum(len(turn) for turn in row["agent_turns"])
            out[(row["checkpoint_id"], row["seed"], row["scenario_id"])] = math.log(max(chars, 1))
    return out


def _demean(values, groups):
    sums = np.bincount(groups, weights=values)
    return values - (sums / np.bincount(groups))[groups]


def seed_components(seed_idx, x, y, n_seeds):
    """Per-seed (sxy, sxx) after removing seed means: the seed-FE slope is sum(sxy)/sum(sxx)."""
    xd, yd = _demean(x, seed_idx), _demean(y, seed_idx)
    return (np.bincount(seed_idx, weights=xd * yd, minlength=n_seeds),
            np.bincount(seed_idx, weights=xd * xd, minlength=n_seeds))


def two_way_slope(seed_idx, scen_idx, x, y, sweeps=50):
    """Slope with seed and scenario fixed effects (alternating projections)."""
    xd, yd = x.copy(), y.copy()
    for _ in range(sweeps):
        xd, yd = _demean(xd, seed_idx), _demean(yd, seed_idx)
        xd, yd = _demean(xd, scen_idx), _demean(yd, scen_idx)
    denominator = float(xd @ xd)
    return float(xd @ yd) / denominator if denominator > 0 else float("nan")


def reduce_block(pairs: list, rng) -> dict:
    """pairs: (seed, scenario, x_high, y_high, x_low, y_low) common-complete within one cell."""
    seeds = sorted({p[0] for p in pairs})
    scenarios = sorted({p[1] for p in pairs})
    seed_idx = np.array([seeds.index(p[0]) for p in pairs])
    scen_idx = np.array([scenarios.index(p[1]) for p in pairs])
    xh, yh, xl, yl = (np.array([p[i] for p in pairs], dtype=float) for i in (2, 3, 4, 5))
    n = len(seeds)
    counts = np.bincount(seed_idx, minlength=n)
    sep_s = np.bincount(seed_idx, weights=yh - yl, minlength=n) / counts
    gap_s = np.bincount(seed_idx, weights=xh - xl, minlength=n) / counts
    h_xy, h_xx = seed_components(seed_idx, xh, yh, n)
    l_xy, l_xx = seed_components(seed_idx, xl, yl, n)

    def stats(index):
        slope_high = h_xy[index].sum(-1) / h_xx[index].sum(-1)
        slope_low = l_xy[index].sum(-1) / l_xx[index].sum(-1)
        slope = 0.5 * (slope_high + slope_low)
        sep, gap = sep_s[index].mean(-1), gap_s[index].mean(-1)
        return {
            "sep": sep,
            "log_length_gap": gap,
            "slope_high_pole_seed_fe": slope_high,
            "slope_low_pole_seed_fe": slope_low,
            "within_pole_slope_seed_fe": slope,
            "extrapolated_length_association_sep": slope * gap,
            "extrapolated_length_association_share": slope * gap / sep,
        }

    out = {"n_pairs": len(pairs), "n_seed_clusters": n}
    point = stats(np.arange(n))
    draws = stats(rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n)))
    for name, value in point.items():
        out[name] = round(float(value), 6)
        low, high = np.percentile(draws[name], [2.5, 97.5])
        out[f"{name}_ci95"] = [round(float(low), 6), round(float(high), 6)]

    two_way = 0.5 * (two_way_slope(seed_idx, scen_idx, xh, yh)
                     + two_way_slope(seed_idx, scen_idx, xl, yl))
    out["within_pole_slope_seed_scenario_fe"] = round(two_way, 6)
    out["extrapolated_length_association_share_seed_scenario_fe"] = round(
        two_way * point["log_length_gap"] / point["sep"], 6)

    out["length_chars"] = {
        pole: {f"p{q:02d}" if q not in (0, 100) else ("min" if q == 0 else "max"):
               int(round(math.exp(np.percentile(x, q)))) for q in QUANTILES}
        for pole, x in (("high", xh), ("low", xl))
    }
    lower = max(np.percentile(xh, 5), np.percentile(xl, 5))
    upper = min(np.percentile(xh, 95), np.percentile(xl, 95))
    band = {
        "definition": "intersection of the two poles' central 90% log-length ranges",
        "chars": [int(round(math.exp(lower))), int(round(math.exp(upper)))],
        "empty": bool(lower > upper),
    }
    if lower > upper:
        band.update({"n_high": 0, "n_low": 0, "sep_in_band": None,
                     "note": "the poles' central 90% length ranges do not overlap"})
    else:
        in_h, in_l = (xh >= lower) & (xh <= upper), (xl >= lower) & (xl <= upper)
        band.update({"n_high": int(in_h.sum()), "n_low": int(in_l.sum())})
        if in_h.sum() and in_l.sum():
            band["mean_chars_high"] = int(round(math.exp(xh[in_h].mean())))
            band["mean_chars_low"] = int(round(math.exp(xl[in_l].mean())))
        band["sep_in_band"] = (
            round(float(yh[in_h].mean() - yl[in_l].mean()), 6)
            if in_h.sum() >= MIN_BAND and in_l.sum() >= MIN_BAND else None)
        band["note"] = ("unpaired transcript means inside the band; the two sides come from "
                        "opposite tails of their pole and are not exchangeable")
    out["common_interval"] = band
    full_lower, full_upper = max(xh.min(), xl.min()), min(xh.max(), xl.max())
    out["full_range_overlap"] = {
        "chars": [int(round(math.exp(full_lower))), int(round(math.exp(full_upper)))],
        "empty": bool(full_lower > full_upper),
        "n_high": int(((xh >= full_lower) & (xh <= full_upper)).sum()),
        "n_low": int(((xl >= full_lower) & (xl <= full_upper)).sum()),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default="process/results/xlpt/XLPT_LENGTH_SENSITIVITY.json")
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    root = project / "process/results/xlpt"

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cells = {}
    for cell, pair in CELLS.items():
        high_phase, low_phase, _, _ = spr.POLE_PAIRS[pair]
        _, _, unit = spr.collect(root, "seed-*.jsonl", phases=(high_phase, low_phase))
        high_len, low_len = load_lengths(root, high_phase), load_lengths(root, low_phase)
        blocks = {}
        for (phase, checkpoint, trait, seed, scenario), score in sorted(unit.items()):
            if phase != high_phase or score is None:
                continue
            low = unit.get((low_phase, checkpoint, trait, seed, scenario))
            if low is None:
                continue
            blocks.setdefault((checkpoint, trait), []).append(
                (seed, scenario, high_len[(checkpoint, seed, scenario)], score,
                 low_len[(checkpoint, seed, scenario)], low))
        cells[CELL_LABEL[cell]] = {
            f"{checkpoint}|{trait}": reduce_block(pairs, rng)
            for (checkpoint, trait), pairs in sorted(blocks.items())
        }
    out = {
        "schema_version": 3,
        "verdict": "DESCRIPTIVE_ONLY",
        "eligible_for_gate": False,
        "scope_note": (
            "Pole and length are confounded by design of the persona. The slope is a within-pole "
            "association extrapolated across a length gap the poles do not share; no quantity "
            "here is a length-independent effect or a mediated share."
        ),
        "units": ("within-cell common-complete (seed, scenario) pairs; sep and log_length_gap use "
                  "equal seed-cluster weights, slopes are observation-weighted OLS"),
        "bootstrap": {"kind": "joint seed-cluster percentile", "seed": BOOTSTRAP_SEED,
                      "draws": BOOTSTRAP_DRAWS, "generator": "numpy default_rng"},
        "length": "log of total agent characters per transcript (3 agent turns)",
        "cells": cells,
        "reducer_sha256": spr.sha256_file(Path(__file__).resolve()),
    }
    (project / args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    for label, blocks in cells.items():
        for key, b in blocks.items():
            if "extraversion" not in key:
                continue
            band = b["common_interval"]
            print(f"{label:24s} {key:34s} sep={b['sep']:.3f} gap={b['log_length_gap']:.2f} "
                  f"slope={b['within_pole_slope_seed_fe']:+.3f}/"
                  f"{b['within_pole_slope_seed_scenario_fe']:+.3f} "
                  f"share={b['extrapolated_length_association_share']:+.2f}/"
                  f"{b['extrapolated_length_association_share_seed_scenario_fe']:+.2f} "
                  f"band={band['chars']} empty={band['empty']} "
                  f"n={band['n_high']}/{band['n_low']} sep_in={band['sep_in_band']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
