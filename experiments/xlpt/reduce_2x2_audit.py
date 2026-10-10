#!/usr/bin/env python3
"""Audit companions to the exploratory en/zh 2x2 (descriptive, CPU only, existing scores).

Nothing here is a new estimand. The reducer exposes what the cell separations are made of:

* ``phase_mapping``       -- cell -> (pair key, high phase, low phase, score directories), i.e.
  the contract through which ``summarize_pole_rater.collect`` feeds every 2x2 reducer;
* ``scenarios``           -- the scenario ids actually scored, per checkpoint, and whether every
  cell and pole of that checkpoint uses the same set;
* ``seed_separation``     -- per cell, the per-seed mean separation (min, median, max, count of
  seeds with a negative separation), on within-cell common-complete pairs;
* ``items``               -- per cell, pole and rubric item: n, mean and the share of scores at
  the rubric floor (1), midpoint (4) and ceiling (7); plus the item separation (high mean minus
  low mean over transcripts) and each item's share of the summed item separations;
* ``items_paired``        -- the item decomposition of the HEADLINE estimand: per checkpoint and
  trait, on the all-cell mask of XLPT_2X2_ROBUSTNESS.json ((seed, scenario) pairs complete in
  every cell and both poles), the equal-seed-weight mean high-minus-low difference of each item
  per cell, its share of the summed item separations, and each item's contribution to every
  linear language-cell contrast. The item separations average to the all-cell-mask trait
  separation and the contributions sum to the contrast;
* ``histograms``          -- full 1-7 counts per cell, pole and item over every non-missing
  item score (transcript-weighted, like ``items``);
* ``cped_labels``         -- CPED identities, label classes and clusters behind the block-H AUC;
* ``friends_labels``      -- annotation unit of the FriendsPersona labels: per identity, units
  labelled high and low on each trait, the informative identities and their pair counts.

No gate is read or written.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_pole_rater as spr  # noqa: E402
from reduce_2x2_contrast import CELLS, CELL_LABEL  # noqa: E402
from reduce_2x2_robustness import LINEAR  # noqa: E402

FLOOR, MIDPOINT, CEILING = 1, 4, 7


def item_rows(score_root: Path, phase: str):
    """Every rater row of one phase, canonical and rescue namespaces."""
    base = spr.POLE_PHASES[phase]["score_dir"]
    for score_dir in (base, base + spr.RESCUE_SUFFIX):
        root = score_root / score_dir
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/seed-*.jsonl")):
            yield from spr.read_jsonl(path)


def describe_items(rows) -> dict:
    """(checkpoint, item) -> n, mean and floor / midpoint / ceiling shares."""
    values = defaultdict(list)
    for row in rows:
        for item, score in row["scores"].items():
            if score is not None:
                values[(row["checkpoint_id"], item)].append(score)
    out = {}
    for key, scores in values.items():
        counts = Counter(scores)
        n = len(scores)
        out[key] = {
            "n": n,
            "mean": round(float(np.mean(scores)), 6),
            "share_floor_1": round(counts[FLOOR] / n, 6),
            "share_midpoint_4": round(counts[MIDPOINT] / n, 6),
            "share_ceiling_7": round(counts[CEILING] / n, 6),
        }
    return out


def histograms(rows) -> dict:
    """(checkpoint, item) -> {score: count} over every non-missing item score."""
    counts = defaultdict(Counter)
    for row in rows:
        for item, score in row["scores"].items():
            if score is not None:
                counts[(row["checkpoint_id"], item)][score] += 1
    return {key: {str(v): c[v] for v in range(FLOOR, CEILING + 1)} for key, c in counts.items()}


def complete_items(score_root: Path, phase: str, unit: dict) -> dict:
    """(checkpoint, trait, seed, scenario) -> item scores of the row behind a complete trait."""
    out = {}
    for row in item_rows(score_root, phase):
        key = (row["checkpoint_id"], row["trait"], row["seed"], row["source_scenario_id"])
        target = unit.get((phase, key[0], key[1], key[2], key[3]))
        scores = row["scores"]
        if target is None or any(v is None for v in scores.values()):
            continue
        if abs(float(np.mean(list(scores.values()))) - target) < 1e-9:
            out[key] = scores
    return out


def paired_items(per_cell: dict) -> dict:
    """per_cell: cell -> pole -> {(checkpoint, trait, seed, scenario): item scores}."""
    blocks = sorted({key[:2] for poles in per_cell.values() for key in poles["high"]})
    out = {}
    for checkpoint, trait in blocks:
        cells = [c for c in CELLS if any(k[:2] == (checkpoint, trait) for k in per_cell[c]["high"])]
        mask = set.intersection(*(
            {k[2:] for k in per_cell[c][pole] if k[:2] == (checkpoint, trait)}
            for c in cells for pole in ("high", "low")))
        seeds = sorted({seed for seed, _ in mask})
        items = spr.ITEM_IDS[trait]
        sep = {}
        for cell in cells:
            by_seed = defaultdict(list)
            for seed, scenario in mask:
                high = per_cell[cell]["high"][(checkpoint, trait, seed, scenario)]
                low = per_cell[cell]["low"][(checkpoint, trait, seed, scenario)]
                by_seed[seed].append([high[i] - low[i] for i in items])
            sep[cell] = np.mean([np.mean(by_seed[seed], axis=0) for seed in seeds], axis=0)
        block = {"pairs_in_all_cell_mask": len(mask), "n_seed_clusters": len(seeds),
                 "cells": {}, "contrasts": {}}
        for cell in cells:
            total = float(sep[cell].sum())
            block["cells"][CELL_LABEL[cell]] = {
                "trait_separation": round(total / len(items), 6),
                "item_separation": {i: round(float(v), 6) for i, v in zip(items, sep[cell])},
                "share_of_summed_item_separation": {
                    i: round(float(v) / total, 6) for i, v in zip(items, sep[cell])},
            }
        for name, signs in LINEAR.items():
            if not signs.keys() <= set(cells):
                continue
            contribution = sum(sign * sep[cell] for cell, sign in signs.items()) / len(items)
            block["contrasts"][name] = {
                "trait_contrast": round(float(contribution.sum()), 6),
                "item_contribution": {i: round(float(v), 6) for i, v in zip(items, contribution)},
            }
        out[f"{checkpoint}|{trait}"] = block
    return out


def cped_labels(path: Path) -> dict:
    identities, units = defaultdict(lambda: {"clusters": set(), "labels": defaultdict(set)}), 0
    for row in spr.read_jsonl(path):
        if row.get("rendering") != "full":
            continue
        units += 1
        entry = identities[row["identity"]]
        entry["clusters"].add(row["cluster"])
        for trait, label in row["labels"].items():
            entry["labels"][trait].add(label)
    out = {"n_units_full_rendering": units, "n_identities": len(identities),
           "n_clusters": len({c for e in identities.values() for c in e["clusters"]}),
           "every_identity_in_one_cluster": all(len(e["clusters"]) == 1
                                                for e in identities.values()),
           "traits": {}}
    for trait in sorted({t for e in identities.values() for t in e["labels"]}):
        labels = [e["labels"][trait] for e in identities.values()]
        out["traits"][trait] = {
            "identities_label_high": sum(l == {"high"} for l in labels),
            "identities_label_low": sum(l == {"low"} for l in labels),
            "identities_with_both_labels": sum(len(l) > 1 for l in labels),
        }
    return out


def seed_separation(unit: dict, high_phase: str, low_phase: str) -> dict:
    per = defaultdict(lambda: defaultdict(list))
    for (phase, checkpoint, trait, seed, scenario), score in unit.items():
        if phase != high_phase or score is None:
            continue
        low = unit.get((low_phase, checkpoint, trait, seed, scenario))
        if low is not None:
            per[(checkpoint, trait)][seed].append(score - low)
    out = {}
    for (checkpoint, trait), by_seed in sorted(per.items()):
        means = {seed: float(np.mean(d)) for seed, d in by_seed.items()}
        ordered = sorted(means.values())
        out[f"{checkpoint}|{trait}"] = {
            "n_seed_clusters": len(means),
            "min": round(ordered[0], 6),
            "median": round(float(np.median(ordered)), 6),
            "max": round(ordered[-1], 6),
            "seeds_with_negative_separation": sum(v < 0 for v in ordered),
            "argmin_seed": min(means, key=means.get),
        }
    return out


def friends_labels(path: Path) -> dict:
    per = defaultdict(lambda: defaultdict(Counter))
    units = 0
    for row in spr.read_jsonl(path):
        if row.get("rendering") != "full":
            continue
        units += 1
        for trait, label in row["labels"].items():
            per[trait][row["identity"]][label] += 1
    out = {"annotation_unit": "(scene, character) row: one label per trait per row",
           "n_units_full_rendering": units, "traits": {}}
    for trait, identities in sorted(per.items()):
        informative = {i: c for i, c in identities.items() if c["high"] and c["low"]}
        pairs = {i: c["high"] * c["low"] for i, c in informative.items()}
        total = sum(pairs.values())
        top = sorted(pairs, key=pairs.get, reverse=True)[:6]
        out["traits"][trait] = {
            "n_identities": len(identities),
            "n_identities_constant_label": len(identities) - len(informative),
            "n_informative_identities": len(informative),
            "same_identity_high_low_pairs": total,
            "top_identities": {
                i: {"high": informative[i]["high"], "low": informative[i]["low"],
                    "pairs": pairs[i], "pair_share": round(pairs[i] / total, 6)}
                for i in top
            },
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default="process/results/xlpt/XLPT_2X2_AUDIT.json")
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    root = project / "process/results/xlpt"

    mapping, seeds_out, items_out, hist_out, per_cell = {}, {}, {}, {}, {}
    scenario_sets = defaultdict(dict)
    for cell, pair in CELLS.items():
        high_phase, low_phase, _, _ = spr.POLE_PAIRS[pair]
        label = CELL_LABEL[cell]
        mapping[label] = {
            "cell": cell, "pole_pair": pair,
            "high_phase": high_phase, "low_phase": low_phase,
            "high_score_dir": spr.POLE_PHASES[high_phase]["score_dir"],
            "low_score_dir": spr.POLE_PHASES[low_phase]["score_dir"],
            "rescue_suffix": spr.RESCUE_SUFFIX,
        }
        _, _, unit = spr.collect(root, "seed-*.jsonl", phases=(high_phase, low_phase))
        seeds_out[label] = seed_separation(unit, high_phase, low_phase)
        for phase, checkpoint, _, _, scenario in unit:
            pole = "high" if phase == high_phase else "low"
            scenario_sets[checkpoint].setdefault((label, pole), set()).add(scenario)

        per_cell[cell] = {"high": complete_items(root, high_phase, unit),
                          "low": complete_items(root, low_phase, unit)}
        hist = defaultdict(dict)
        for pole, phase in (("high", high_phase), ("low", low_phase)):
            for (checkpoint, item), counts in sorted(histograms(item_rows(root, phase)).items()):
                hist[checkpoint].setdefault(item, {})[pole] = counts
        hist_out[label] = hist

        high = describe_items(item_rows(root, high_phase))
        low = describe_items(item_rows(root, low_phase))
        by_checkpoint = defaultdict(dict)
        for (checkpoint, item) in sorted(high):
            if (checkpoint, item) not in low:
                continue
            by_checkpoint[checkpoint][item] = {
                "high": high[(checkpoint, item)], "low": low[(checkpoint, item)],
                "item_separation": round(
                    high[(checkpoint, item)]["mean"] - low[(checkpoint, item)]["mean"], 6),
            }
        for checkpoint, items in by_checkpoint.items():
            for prefix in sorted({item[0] for item in items}):
                total = sum(v["item_separation"] for k, v in items.items() if k[0] == prefix)
                for k, v in items.items():
                    if k[0] == prefix:
                        v["share_of_summed_item_separation"] = (
                            round(v["item_separation"] / total, 6) if total else None)
        items_out[label] = by_checkpoint

    scenarios = {}
    for checkpoint, sets in sorted(scenario_sets.items()):
        union = sorted(set.union(*sets.values()))
        scenarios[checkpoint] = {
            "n_scenarios": len(union),
            "same_set_in_every_cell_and_pole": all(s == set(union) for s in sets.values()),
            "scenario_ids": union,
        }

    out = {
        "schema_version": 2,
        "verdict": "DESCRIPTIVE_ONLY",
        "eligible_for_gate": False,
        "scope_note": (
            "Audit companions computed from existing rater rows. `items` and `histograms` use "
            "every non-missing item score (transcript-weighted, unpaired). `items_paired` uses "
            "the all-cell mask and equal seed weights of XLPT_2X2_ROBUSTNESS.json and decomposes "
            "its all_cell_mask estimates exactly."
        ),
        "collector": "summarize_pole_rater.collect(score_root, 'seed-*.jsonl', phases=(high, low))",
        "phase_mapping": mapping,
        "scenarios": scenarios,
        "seed_separation": seeds_out,
        "items": items_out,
        "items_paired": paired_items(per_cell),
        "histograms": hist_out,
        "cped_labels": cped_labels(root / "h-public/scores-cped.jsonl"),
        "friends_labels": friends_labels(root / "h-public/scores-friendspersona.jsonl"),
        "reducer_sha256": spr.sha256_file(Path(__file__).resolve()),
    }
    (project / args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
