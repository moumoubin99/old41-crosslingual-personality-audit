#!/usr/bin/env python3
"""Common-mask re-analyses of the frozen supplementary scores (descriptive; decides no gate).

Reads the same score shards as ``reduce_supp.py`` and scores nothing. It adds four things that
the first reducer does not give:

  item_attribution  edited-text separations for all six Extraversion items, the five items
                    without E1, and E1 alone on ONE mask: transcripts with all six items parsed
                    under the unedited and the edited scoring. On that mask
                    sep(E) = sep(E1) / 6 + 5 * sep(E without E1) / 6 holds exactly.
  same_seeds        the three raters on the seed clusters that every rater scored, on the
                    transcripts that are complete under every rater, for the baseline, the three
                    edits and the Chinese rubric rendering, with generator-mean intervals.
  agreement         between-rater agreement by generator and cell, and without the transcripts
                    of a generator that is one of the two raters.
  public_counts     requested records, records scored, complete Extraversion scores and units
                    that enter the concordance statistic, per rater and public corpus.

Usage:
    python experiments/xlpt/reduce_supp_common.py --project .
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reduce_supp as rs  # noqa: E402

RATERS = ("mistral", "llama", "gemma")
FULL = tuple(g for g in rs.GENERATORS if g not in rs.P_EN_ONLY)
ITEM_SCORES = {"E": rs.E_ITEMS, "E_without_E1": rs.E_ITEMS[1:], "E1": rs.E_ITEMS[:1]}
# gemma-3-12b-it failed the Chinese response-language screen: its en_zh cell is not Chinese dialogue.
FAILED_LANGUAGE_BLOCK = ("gemma-3-12b-it", "en_zh")


def complete(table, names=rs.E_ITEMS):
    return {k for k, v in table.items() if np.isfinite(rs.value(v, names))}


def keep(table, keys):
    return {k: v for k, v in table.items() if k in keys}


def restrict(table, seed_set):
    return {k: v for k, v in table.items() if k[3] in seed_set}


def cells_of(gen):
    return [c for c in rs.CELLS if not (gen in rs.P_EN_ONLY and c.startswith("zh"))]


def generator_mean(design, base, other, gens, names):
    """Generator-mean statistics on each generator's joint all-cell mask.

    ``retained_mean_over_cells`` is computed per generator and then averaged, as in the
    article; ``ratio_zh_zh_over_en_en`` is the ratio of the generator-mean separations."""
    per, pairs = [], {}
    for gen in gens:
        cells = cells_of(gen)
        a = np.stack([design.grid(base, gen, c, names) for c in cells])
        b = np.stack([design.grid(other, gen, c, names) for c in cells])
        mask = ~(np.isnan(a).any(axis=0) | np.isnan(b).any(axis=0))
        if not mask.any():
            return None
        pairs[gen] = int(mask.sum())
        per.append([design.estimate(np.where(mask[None], x, np.nan)) for x in (a, b)])
    cells = list(rs.CELLS) if all(g not in rs.P_EN_ONLY for g in gens) else cells_of(rs.P_EN_ONLY[0])
    out = {"generators": list(gens), "pairs_in_joint_mask": pairs}
    with np.errstate(invalid="ignore", divide="ignore"):
        for which, name in ((0, "base"), (1, "other")):
            mean = [np.mean([p[which][j] for p in per], axis=0) for j in range(3)]
            block = {f"sep_{c}": rs.entry(*(m[..., i] for m in mean)) for i, c in enumerate(cells)}
            if "zh_zh" in cells:
                i, j = cells.index("zh_zh"), cells.index("en_en")
                block["ratio_zh_zh_over_en_en"] = rs.entry(*(m[..., i] / m[..., j] for m in mean))
            block["sep_mean_over_cells"] = rs.entry(*(m.mean(-1) for m in mean))
            out[name] = block
        out["retained_mean_over_cells"] = rs.entry(*(
            np.mean([p[1][j].mean(-1) / p[0][j].mean(-1) for p in per], axis=0) for j in range(3)))
        for i, c in enumerate(cells):
            out[f"retained_{c}"] = rs.entry(*(
                np.mean([p[1][j][..., i] / p[0][j][..., i] for p in per], axis=0) for j in range(3)))
    return out


def attribution(design, ref, other):
    """All three item sets on the transcripts with six parsed items under both scorings."""
    keys = complete(ref) & complete(other)
    ref, other = keep(ref, keys), keep(other, keys)
    out = {"units_with_six_items_under_both": len(keys)}
    for score, names in ITEM_SCORES.items():
        out[score] = rs.paired(design, ref, other, rs.GENERATORS, names)
        out[score]["mean_of_full_design_generators"] = generator_mean(design, ref, other, FULL, names)
    # identity check: sep(E) = sep(E1)/6 + 5 sep(E without E1)/6 on the common mask
    worst = 0.0
    for gen, res in out["E"].items():
        if gen not in rs.GENERATORS:
            continue
        for which in ("base", "other"):
            for cell in cells_of(gen):
                e = res[which][f"sep_{cell}"]["estimate"]
                e1 = out["E1"][gen][which][f"sep_{cell}"]["estimate"]
                e5 = out["E_without_E1"][gen][which][f"sep_{cell}"]["estimate"]
                worst = max(worst, abs(e - (e1 + 5 * e5) / 6))
        masks = {out[s][gen]["pairs_in_joint_mask"] for s in ITEM_SCORES}
        assert len(masks) == 1, (gen, masks)
    out["max_abs_decomposition_error_rounded_inputs"] = round(worst, 4)
    return out


def stratified_agreement(table_a, table_b, a, b):
    out = {}
    for gen in rs.GENERATORS:
        for cell in cells_of(gen):
            ta = {k: v for k, v in table_a.items() if k[1] == gen and k[0] == cell}
            tb = {k: v for k, v in table_b.items() if k[1] == gen and k[0] == cell}
            res = rs.agreement(ta, tb, rs.E_ITEMS, (gen,))
            trait = res.get("trait_mean") or {}
            out[f"{gen}|{cell}"] = {
                "units_in_both": res["units_in_both"],
                "self_rated": gen in (rs.SELF.get(a), rs.SELF.get(b)),
                "failed_response_language_block": (gen, cell) == FAILED_LANGUAGE_BLOCK,
                **{k: trait.get(k) for k in ("all", "within_high", "within_low")}}
    return out


def strata_summary(strata, key, which="pearson"):
    """Range of one correlation over the generator-by-cell strata of one selection."""
    def rng(rows):
        vals = [r[key][which] for r in rows if r.get(key)]
        return None if not vals else {"n_strata": len(vals), "min": min(vals), "max": max(vals),
                                      "median": round(float(np.median(vals)), 4)}
    rows = list(strata.values())
    return {"all_strata": rng(rows),
            "not_self_rated": rng([r for r in rows if not r["self_rated"]]),
            "not_self_rated_and_language_screen_passed": rng(
                [r for r in rows if not r["self_rated"] and not r["failed_response_language_block"]])}


def public_counts(root, corpus, records):
    full = [r for r in records if r.get("rendering", "full") == "full"]
    scored = [r for r in full if r["status"] == "scored"]
    e_ok = [r for r in scored if r["trait_scores"].get("extraversion") is not None]
    labelled = [r for r in e_ok if r["labels"]["EXT"] in ("high", "low")]
    both = [r for r in labelled if r["trait_scores"].get("agreeableness") is not None
            and r["labels"]["AGR"] in ("high", "low")]
    units = rs.read_jsonl(root / "h-public" / f"unit-metadata-{corpus}.jsonl")
    return {"units_in_corpus_file": len(units), "records_requested": len(full),
            "records_scored": len(scored),
            "records_excluded_for_context": sum(r["status"] != "scored" for r in full),
            "complete_extraversion_score": len(e_ok),
            "records_with_item_parse_error": sum(bool(r.get("parse_errors")) for r in full),
            "item_parse_errors": sum(len(r.get("parse_errors") or []) for r in full),
            "eligible_labelled_units_E_to_EXT": len(labelled),
            "EXT_high": sum(r["labels"]["EXT"] == "high" for r in labelled),
            "EXT_low": sum(r["labels"]["EXT"] == "low" for r in labelled),
            "identities": len({r["identity"] for r in labelled}),
            "clusters": len({r["cluster"] for r in labelled}),
            "complete_on_both_traits_and_labels": len(both)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default=f"process/results/xlpt/{rs.NAMESPACE}/SUPP_COMMON_SUMMARY.json")
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    root = project / "process/results/xlpt"

    canonical = rs.load_canonical(root)
    _, scenarios = rs.axes(canonical)
    out = {"schema_version": 1, "namespace": rs.NAMESPACE, "verdict": "DESCRIPTIVE_ONLY",
           "eligible_for_gate": False,
           "bootstrap": {"seed": rs.BOOT_SEED, "draws": rs.BOOT_DRAWS,
                         "note": "crossed seed x scenario multiplicities from reduce_supp.Design, "
                                 "shared by every block with the same seed set"},
           "reducer_sha256": rs.sha256_file(Path(__file__).resolve()),
           "reduce_supp_sha256": rs.sha256_file(Path(rs.__file__).resolve()), "shards": {}}

    supp = {}
    for tag in [f"s1-{r}-none" for r in RATERS[1:]] + [f"s2-{r}-surface" for r in RATERS] + [
            f"s3-{r}-zh" for r in RATERS]:
        tables, report = rs.load_supp(root, tag)
        if tables is None:
            raise SystemExit(f"missing shard {tag}")
        supp[tag] = tables
        out["shards"][tag] = report
    base = {"mistral": canonical, "llama": supp["s1-llama-none"]["none"],
            "gemma": supp["s1-gemma-none"]["none"]}

    # ---- A. item attribution on one item-complete mask, each rater on its own seed set
    out["item_attribution"] = {}
    for rater in RATERS:
        shard = supp[f"s2-{rater}-surface"]
        seed_set = sorted({k[3] for t in shard.values() for k in t})
        design = rs.Design(seed_set, scenarios)
        ref = restrict(base[rater], set(seed_set))
        out["item_attribution"][rater] = {"seeds": seed_set, **{
            variant: attribution(design, ref, other) for variant, other in sorted(shard.items())
            if variant != "none"}}

    # ---- B. the three raters on the same seeds and the same transcripts
    seed_sets = [{k[3] for t in supp[f"s{n}-{r}-{s}"].values() for k in t}
                 for r in RATERS for n, s in ((2, "surface"), (3, "zh"))]
    common_seeds = sorted(set.intersection(*seed_sets))
    design = rs.Design(common_seeds, scenarios)
    ref = {r: restrict(base[r], set(common_seeds)) for r in RATERS}
    same = {"seeds": common_seeds, "caution": "three seed clusters: intervals resample three seeds"}
    conditions = {v: {r: supp[f"s2-{r}-surface"][v] for r in RATERS}
                  for v in ("strip_edges", "swap_edges", "trunc2")}
    conditions["zh_rubric"] = {r: supp[f"s3-{r}-zh"]["none"] for r in RATERS}
    for name, other in conditions.items():
        keys = set.intersection(*(complete(ref[r]) & complete(restrict(other[r], set(common_seeds)))
                                  for r in RATERS))
        res = {"units_complete_under_every_rater_and_both_scorings": len(keys)}
        for r in RATERS:
            a, b = keep(ref[r], keys), keep(other[r], keys)
            res[r] = {score: {"per_generator": rs.paired(design, a, b, rs.GENERATORS, names),
                              "mean_of_full_design_generators": generator_mean(design, a, b, FULL, names)}
                      for score, names in ITEM_SCORES.items()}
        same[name] = res
    keys = set.intersection(*(complete(ref[r]) for r in RATERS))
    same["baseline"] = {"units_complete_under_every_rater": len(keys), **{
        r: rs.block(design, keep(ref[r], keys), rs.GENERATORS, rs.E_ITEMS) for r in RATERS}}
    out["same_seeds"] = same

    # ---- C. agreement by generator and cell
    out["agreement"] = {}
    for i, a in enumerate(RATERS):
        for b in RATERS[i + 1:]:
            ta = restrict(base[a], {k[3] for k in base[b]})
            strata = stratified_agreement(ta, base[b], a, b)
            not_self = tuple(g for g in rs.GENERATORS if g not in (rs.SELF.get(a), rs.SELF.get(b)))
            passed = {k: v for k, v in ta.items() if (k[1], k[0]) != FAILED_LANGUAGE_BLOCK}
            out["agreement"][f"{a}_vs_{b}"] = {
                "pooled_all": rs.agreement(ta, base[b], rs.E_ITEMS)["trait_mean"],
                "pooled_excluding_self_rated_generators":
                    rs.agreement(ta, base[b], rs.E_ITEMS, not_self)["trait_mean"],
                "pooled_excluding_self_rated_and_failed_language_block":
                    rs.agreement(passed, base[b], rs.E_ITEMS, not_self)["trait_mean"],
                "by_generator_and_cell": strata,
                "range_over_strata": {k: strata_summary(strata, k)
                                      for k in ("all", "within_high", "within_low")}}

    # ---- D. public-corpus denominators
    out["public_counts"] = {}
    for corpus in ("friendspersona", "cped"):
        out["public_counts"][f"mistral|{corpus}"] = public_counts(
            root, corpus, rs.read_jsonl(root / "h-public" / f"scores-{corpus}.jsonl"))
        for rater in RATERS[1:]:
            records = rs.public_records(root, f"pub-{rater}-{corpus}")
            if records is not None:
                out["public_counts"][f"{rater}|{corpus}"] = public_counts(root, corpus, records)

    target = project / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(project)}; common seeds {common_seeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
