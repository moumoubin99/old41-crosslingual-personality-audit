#!/usr/bin/env python3
"""Reduction of the 2026-10-06 extension of the supplementary round (descriptive; decides no gate).

The extension added two things to the frozen scores of ``supp-20261004`` and changed none of them:

  * the canonical rater on seed clusters 20260814 to 20260816 for the edited-text scoring and the
    Chinese-rubric scoring (shards ``s2-mistral-surface-b`` and ``s3-mistral-zh-b``), so each of
    the three raters has six seed clusters in both conditions;
  * the Chinese-rubric scoring with three further machine renderings of the items and anchors
    (shards ``s4-<rater>-zh-<r2|r3|r4>``, the canonical rater in two halves of three seed
    clusters, the second with the suffix ``-b``; the first rendering is the one of
    ``s3-<rater>-zh``).

This script scores nothing. It imports ``reduce_supp.py`` and ``reduce_supp_common.py`` unchanged
and uses their estimators and their bootstrap, and writes:

  six_seeds    the ``same_seeds`` block and the canonical-rater item attribution of
               ``reduce_supp_common.py`` again, now on six seed clusters for every rater;
  renderings   per rater and rendering, the Chinese-rubric separation against the English-rubric
               separation of the same rater on (a) their joint mask and (b) one mask that is
               complete under every rater, the English rubric and every rendering; the
               generator-mean cell separations, contrasts and ratios; the range over renderings
               and whether each contrast has the same sign under every rendering; and the
               unit-level agreement of each further rendering with the first one.

Usage:
    python experiments/xlpt/reduce_supp_ext.py --project .
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reduce_supp as rs  # noqa: E402
import reduce_supp_common as rc  # noqa: E402

RATERS = rc.RATERS
RENDERINGS = ("r1", "r2", "r3", "r4")
EDITS = ("strip_edges", "swap_edges", "trunc2")
CONTRASTS = tuple(rs.LINEAR)


def merged(root, tags, out):
    """Union of the variant tables of several shards of one condition; keys must not overlap."""
    tables = {}
    for tag in tags:
        part, report = rs.load_supp(root, tag)
        if part is None:
            out["missing_shards"].append(tag)
            continue
        out["shards"][tag] = report
        for variant, table in part.items():
            have = tables.setdefault(variant, {})
            assert not have.keys() & table.keys(), (tag, variant)
            have.update(table)
    return tables


def seeds_of(table):
    return {k[3] for k in table}


def generator_stats(design, base, other, gens, names):
    """Generator-mean statistics of ``reduce_supp.derive`` for both scorings on each generator's
    joint all-cell mask, and the retained share (other / base) per cell and over cells."""
    per, pairs = [], {}
    cells = [c for c in rs.CELLS if all(c in rc.cells_of(g) for g in gens)]
    for gen in gens:
        a = np.stack([design.grid(base, gen, c, names) for c in cells])
        b = np.stack([design.grid(other, gen, c, names) for c in cells])
        mask = ~(np.isnan(a).any(axis=0) | np.isnan(b).any(axis=0))
        if not mask.any():
            return None
        pairs[gen] = int(mask.sum())
        per.append([design.estimate(np.where(mask[None], x, np.nan)) for x in (a, b)])
    out = {"generators": list(gens), "pairs_in_joint_mask": pairs}
    with np.errstate(invalid="ignore", divide="ignore"):
        for which, name in ((0, "en_rubric"), (1, "zh_rubric")):
            mean = [np.mean([p[which][j] for p in per], axis=0) for j in range(3)]
            stats = [rs.derive({c: m[..., i] for i, c in enumerate(cells)}) for m in mean]
            block = {k: rs.entry(stats[0][k], stats[1][k], stats[2][k]) for k in stats[0]}
            block["sep_mean_over_cells"] = rs.entry(*(m.mean(-1) for m in mean))
            out[name] = block
        out["retained_mean_over_cells"] = rs.entry(*(
            np.mean([p[1][j].mean(-1) / p[0][j].mean(-1) for p in per], axis=0) for j in range(3)))
        for i, c in enumerate(cells):
            out[f"retained_{c}"] = rs.entry(*(
                np.mean([p[1][j][..., i] / p[0][j][..., i] for p in per], axis=0)
                for j in range(3)))
    return out


def sign(x):
    return None if x is None else int(np.sign(x))


def across_renderings(per_rendering):
    """Range of the generator-mean statistics over the renderings of one rater."""
    names = ["retained_mean_over_cells"] + [f"retained_{c}" for c in rs.CELLS]
    out = {}
    for name in names:
        vals = {r: b[name]["estimate"] for r, b in per_rendering.items() if b and name in b}
        if vals:
            out[name] = {"by_rendering": vals, "min": min(vals.values()), "max": max(vals.values())}
    keys = ["sep_mean_over_cells"] + [f"sep_{c}" for c in rs.CELLS] + list(CONTRASTS) + list(rs.RATIOS)
    for key in keys:
        vals = {r: b["zh_rubric"][key]["estimate"] for r, b in per_rendering.items()
                if b and key in b["zh_rubric"]}
        if not vals:
            continue
        row = {"by_rendering": vals, "min": min(vals.values()), "max": max(vals.values())}
        if key in CONTRASTS:
            english = {r: b["en_rubric"][key]["estimate"] for r, b in per_rendering.items() if b}
            row["en_rubric_by_rendering_mask"] = english
            row["same_sign_under_every_rendering"] = len({sign(v) for v in vals.values()}) == 1
            row["sign_equals_en_rubric_sign_under_every_rendering"] = all(
                sign(vals[r]) == sign(english[r]) for r in vals)
            row["crossed_interval_excludes_zero_by_rendering"] = {
                r: (lambda c: None if c is None else bool(c[0] > 0 or c[1] < 0))(
                    b["zh_rubric"][key]["ci95_crossed_bootstrap"])
                for r, b in per_rendering.items() if b}
        if key in rs.RATIOS:
            row["below_one_under_every_rendering"] = all(v < 1 for v in vals.values())
        out[key] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default=f"process/results/xlpt/{rs.NAMESPACE}/SUPP_EXT_SUMMARY.json")
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    root = project / "process/results/xlpt"

    canonical = rs.load_canonical(root)
    _, scenarios = rs.axes(canonical)
    here = Path(__file__).resolve()
    out = {"schema_version": 1, "namespace": rs.NAMESPACE, "extension": "2026-10-06",
           "verdict": "DESCRIPTIVE_ONLY", "eligible_for_gate": False,
           "bootstrap": {"seed": rs.BOOT_SEED, "draws": rs.BOOT_DRAWS,
                         "note": "crossed seed x scenario multiplicities from reduce_supp.Design"},
           "reducer_sha256": rs.sha256_file(here),
           "reduce_supp_sha256": rs.sha256_file(Path(rs.__file__).resolve()),
           "reduce_supp_common_sha256": rs.sha256_file(Path(rc.__file__).resolve()),
           "shards": {}, "missing_shards": []}

    base = {"mistral": canonical}
    for rater in RATERS[1:]:
        base[rater] = merged(root, [f"s1-{rater}-none"], out)["none"]
    surface = {r: merged(root, [f"s2-{r}-surface"] + ([f"s2-{r}-surface-b"] if r == "mistral" else []),
                         out) for r in RATERS}
    zh = {"r1": {r: merged(root, [f"s3-{r}-zh"] + ([f"s3-{r}-zh-b"] if r == "mistral" else []),
                           out).get("none", {}) for r in RATERS}}
    for rid in RENDERINGS[1:]:
        zh[rid] = {r: merged(root, [f"s4-{r}-zh-{rid}"] + (
            [f"s4-{r}-zh-{rid}-b"] if r == "mistral" else []), out).get("none", {}) for r in RATERS}
    out["rendering_files"] = {}
    for rid in RENDERINGS:
        name = "supp_rubric_zh.json" if rid == "r1" else f"supp_rubric_zh_{rid}.json"
        path = project / "experiments/xlpt" / name
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            out["rendering_files"][rid] = {
                "file": f"experiments/xlpt/{name}", "sha256": rs.sha256_file(path),
                "renderer": meta.get("renderer", "first rendering of the supplementary round"),
                "all_fields_have_han": meta.get("all_fields_have_han"),
                "fields_with_latin_letters": meta.get("fields_with_latin_letters")}

    # ---- A. six seed clusters for every rater
    conditions = {v: {r: surface[r].get(v, {}) for r in RATERS} for v in EDITS}
    conditions["zh_rubric"] = zh["r1"]
    seed_sets = [seeds_of(t[r]) for t in conditions.values() for r in RATERS]
    common_seeds = sorted(set.intersection(*seed_sets))
    design = rs.Design(common_seeds, scenarios)
    ref = {r: rc.restrict(base[r], set(common_seeds)) for r in RATERS}
    six = {"seeds": common_seeds,
           "seeds_scored_by_rater_and_condition": {
               name: {r: sorted(seeds_of(t[r])) for r in RATERS} for name, t in conditions.items()}}
    for name, other in conditions.items():
        keys = set.intersection(*(rc.complete(ref[r]) & rc.complete(rc.restrict(other[r], set(common_seeds)))
                                  for r in RATERS))
        res = {"units_complete_under_every_rater_and_both_scorings": len(keys)}
        for r in RATERS:
            a, b = rc.keep(ref[r], keys), rc.keep(other[r], keys)
            res[r] = {score: {"per_generator": rs.paired(design, a, b, rs.GENERATORS, names),
                              "mean_of_full_design_generators":
                                  rc.generator_mean(design, a, b, rc.FULL, names)}
                      for score, names in rc.ITEM_SCORES.items()}
        six[name] = res
    keys = set.intersection(*(rc.complete(ref[r]) for r in RATERS))
    six["baseline"] = {"units_complete_under_every_rater": len(keys), **{
        r: rs.block(design, rc.keep(ref[r], keys), rs.GENERATORS, rs.E_ITEMS) for r in RATERS}}
    six["item_attribution_canonical_rater"] = {
        variant: rc.attribution(design, ref["mistral"], rc.restrict(surface["mistral"][variant],
                                                                    set(common_seeds)))
        for variant in EDITS if variant in surface["mistral"]}
    out["six_seeds"] = six

    # ---- B. renderings of the Chinese rubric
    have = [rid for rid in RENDERINGS if all(zh[rid][r] for r in RATERS)]
    seeds_r = sorted(set.intersection(*(seeds_of(zh[rid][r]) for rid in have for r in RATERS)))
    design_r = rs.Design(seeds_r, scenarios)
    ref_r = {r: rc.restrict(base[r], set(seeds_r)) for r in RATERS}
    rend = {"renderings_with_scores_from_every_rater": have, "seeds": seeds_r}
    common = set.intersection(*(rc.complete(ref_r[r]) for r in RATERS), *(
        rc.complete(rc.restrict(zh[rid][r], set(seeds_r))) for rid in have for r in RATERS))
    rend["units_complete_under_every_rater_en_rubric_and_every_rendering"] = len(common)
    for label, mask in (("own_joint_mask", None), ("one_common_mask", common)):
        section = {}
        for r in RATERS:
            per = {}
            for rid in have:
                a, b = ref_r[r], rc.restrict(zh[rid][r], set(seeds_r))
                if mask is not None:
                    a, b = rc.keep(a, mask), rc.keep(b, mask)
                per[rid] = generator_stats(design_r, a, b, rc.FULL, rs.E_ITEMS)
                if label == "own_joint_mask":
                    per[rid]["per_generator"] = rs.paired(design_r, a, b, rs.GENERATORS, rs.E_ITEMS)
            section[r] = {"by_rendering": per,
                          "across_renderings": across_renderings(per)}
        rend[label] = section
    rend["agreement_with_first_rendering"] = {
        r: {rid: {"trait_mean": res.get("trait_mean"), "units_in_both": res["units_in_both"],
                  "per_item": res.get("per_item")}
            for rid in have[1:]
            for res in [rs.agreement(zh["r1"][r], zh[rid][r], rs.E_ITEMS)]}
        for r in RATERS}
    rend["agreement_with_en_rubric"] = {
        r: {rid: rs.agreement(ref_r[r], zh[rid][r], rs.E_ITEMS).get("trait_mean") for rid in have}
        for r in RATERS}
    rend["parse"] = {
        r: {rid: {"units": len(zh[rid][r]),
                  "units_with_six_items": len(rc.complete(zh[rid][r]))} for rid in have}
        for r in RATERS}
    out["renderings"] = rend

    target = project / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(project)}; six-seed block seeds {common_seeds}; "
          f"renderings {have} on seeds {seeds_r}; missing {out['missing_shards']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
