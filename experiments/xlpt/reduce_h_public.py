#!/usr/bin/env python3
"""Reduce the block-H public-data score shards into H_PUBLIC_SUMMARY.json (spec §5).

CPU only.  AUC = Mann-Whitney P(high outscores low), ties = 1/2.  Every interval is a
percentile bootstrap (9,999 draws, seed 20260929) whose draws are shared by every metric of a
corpus, so paired contrasts use the same resamples.

FriendsPersona: primary = pair-weighted within-identity AUC; sensitivities = equal-identity
macro AUC and unit AUC; crossed identity x scene bootstrap (unit weight = product of the two
multiplicities; a resampled identity contributes its within-identity pairs once per copy).
CPED: primary = identity-level AUC on identity means; secondary = unit AUC; franchise-component
bootstrap.  Draws with no high-low comparison are invalid, discarded and counted; fewer than
95% valid draws labels the interval unstable.  Descriptive only: no threshold, no gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from . import score_public_personality as sp
except ImportError:  # direct-script contract tests
    import score_public_personality as sp

OUT_DIR = "process/results/xlpt/h-public"
N_BOOT = 9999
BOOT_SEED = 20260929
STABLE_FRACTION = 0.95
E_ITEMS = tuple(f"E{i}" for i in range(1, 7))
A_ITEMS = tuple(f"A{i}" for i in range(1, 7))
TRAIT_KEY = {"E": "extraversion", "A": "agreeableness"}


# ---------------------------------------------------------------- kernels

def kernel(high_scores, low_scores):
    """K[h, l] = 1 if high > low, 1/2 if tied, 0 otherwise."""
    h = np.asarray(high_scores, dtype=float)[:, None]
    l = np.asarray(low_scores, dtype=float)[None, :]
    return (h > l) + 0.5 * (h == l)


def weighted_auc(scores, labels, weights):
    """Weighted AUC for each row of `weights` (draws x units); NaN where a class has no weight."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    hi, lo = labels == "high", labels == "low"
    if not hi.any() or not lo.any():
        return np.full(weights.shape[0], np.nan)
    k = kernel(scores[hi], scores[lo])
    wh, wl = weights[:, hi], weights[:, lo]
    num = ((wh @ k) * wl).sum(axis=1)
    den = wh.sum(axis=1) * wl.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def within_identity(scores, labels, identities, unit_weights, identity_weights):
    """Pair-weighted and equal-identity macro within-identity AUC per draw.

    unit_weights: draws x units (the scene multiplicity of each unit);
    identity_weights: dict identity -> draws vector (identity multiplicity).
    Returns (pair_weighted, macro, informative_identities, informative_pairs) at the draw level.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    identities = np.asarray(identities)
    unit_weights = np.atleast_2d(np.asarray(unit_weights, dtype=float))
    draws = unit_weights.shape[0]
    num_total = np.zeros(draws)
    den_total = np.zeros(draws)
    macro_sum = np.zeros(draws)
    macro_n = np.zeros(draws)
    for identity in sorted(set(identities.tolist())):
        idx = identities == identity
        hi = idx & (labels == "high")
        lo = idx & (labels == "low")
        if not hi.any() or not lo.any():
            continue
        k = kernel(scores[hi], scores[lo])
        wh, wl = unit_weights[:, hi], unit_weights[:, lo]
        num = ((wh @ k) * wl).sum(axis=1)
        den = wh.sum(axis=1) * wl.sum(axis=1)
        m = np.asarray(identity_weights[identity], dtype=float)
        num_total += m * num
        den_total += m * den
        ok = (den > 0) & (m > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per = np.where(ok, num / np.where(den > 0, den, 1), 0.0)
        macro_sum += np.where(ok, m * per, 0.0)
        macro_n += np.where(ok, m, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pair = np.where(den_total > 0, num_total / np.where(den_total > 0, den_total, 1), np.nan)
        macro = np.where(macro_n > 0, macro_sum / np.where(macro_n > 0, macro_n, 1), np.nan)
    return pair, macro


def informative(labels, identities):
    labels, identities = np.asarray(labels), np.asarray(identities)
    n_id = n_pairs = 0
    for identity in set(identities.tolist()):
        idx = identities == identity
        h, l = int((idx & (labels == "high")).sum()), int((idx & (labels == "low")).sum())
        if h and l:
            n_id += 1
            n_pairs += h * l
    return n_id, n_pairs


def interval(point, draws):
    draws = np.asarray(draws, dtype=float)
    valid = draws[~np.isnan(draws)]
    out = {"estimate": None if point is None or np.isnan(point) else float(point),
           "valid_draws": int(valid.size), "total_draws": int(draws.size)}
    if valid.size:
        lo, hi = np.percentile(valid, [2.5, 97.5])
        out.update({"ci95": [float(lo), float(hi)]})
    else:
        out["ci95"] = None
    out["unstable"] = valid.size < STABLE_FRACTION * draws.size
    return out


# ---------------------------------------------------------------- bootstrap designs

def multiplicities(rng, n_clusters, n_boot):
    return np.stack([np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
                     for _ in range(n_boot)]).astype(float)


class FriendsDesign:
    """Crossed identity x scene bootstrap; row 0 of every weight matrix is the point estimate."""

    def __init__(self, identities, scenes, n_boot=N_BOOT, seed=BOOT_SEED):
        rng = np.random.default_rng(seed)
        self.id_names = sorted(set(identities))
        self.scene_names = sorted(set(scenes))
        mid = multiplicities(rng, len(self.id_names), n_boot)
        ms = multiplicities(rng, len(self.scene_names), n_boot)
        self.mid = np.vstack([np.ones(len(self.id_names)), mid])
        self.ms = np.vstack([np.ones(len(self.scene_names)), ms])
        self.id_index = {name: i for i, name in enumerate(self.id_names)}
        self.scene_index = {name: i for i, name in enumerate(self.scene_names)}

    def metrics(self, scores, labels, identities, scenes):
        ii = np.array([self.id_index[x] for x in identities])
        si = np.array([self.scene_index[x] for x in scenes])
        scene_w = self.ms[:, si]
        unit_w = self.mid[:, ii] * scene_w
        id_w = {name: self.mid[:, self.id_index[name]] for name in set(identities)}
        pair, macro = within_identity(scores, labels, identities, scene_w, id_w)
        unit = weighted_auc(scores, labels, unit_w)
        return {"primary": pair, "macro_identity": macro, "unit": unit}


class ClusterDesign:
    """Franchise-component bootstrap for CPED."""

    def __init__(self, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
        rng = np.random.default_rng(seed)
        self.names = sorted(set(clusters))
        self.index = {name: i for i, name in enumerate(self.names)}
        self.mc = np.vstack([np.ones(len(self.names)), multiplicities(rng, len(self.names), n_boot)])

    def metrics(self, scores, labels, identities, clusters):
        scores = np.asarray(scores, dtype=float)
        unit_w = self.mc[:, [self.index[c] for c in clusters]]
        unit = weighted_auc(scores, labels, unit_w)
        by_id = defaultdict(list)
        for s, lab, ident, c in zip(scores, labels, identities, clusters):
            by_id[ident].append((s, lab, c))
        ids = sorted(by_id)
        id_scores = [float(np.mean([s for s, _, _ in by_id[i]])) for i in ids]
        id_labels = []
        for i in ids:
            labs = {lab for _, lab, _ in by_id[i]}
            if len(labs) != 1:
                raise ValueError(f"identity {i} has non-constant labels")
            id_labels.append(labs.pop())
        id_w = self.mc[:, [self.index[by_id[i][0][2]] for i in ids]]
        primary = weighted_auc(id_scores, id_labels, id_w)
        return {"primary": primary, "unit": unit}


# ---------------------------------------------------------------- summary

def load_scores(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


RECORD_KEYS = frozenset({"schema_version", "unit_id", "corpus", "rendering", "identity", "cluster",
                         "labels", "max_prompt_tokens", "status", "scores", "rater_raw_by_item",
                         "parse_errors", "trait_scores", *sp.SHARED_KEYS})
RUNTIME_MANIFEST = "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json"
RECEIPT_KEYS = frozenset({"schema_version", "corpus", "shard", "shard_sha256", "records", "rater_calls",
                          "units_sha256", "provenance_sha256", "code_sha256"})


def validate_receipt(project: Path, corpus: str, path: Path, records, ledger=None):
    """The shard must match the receipt written at creation and, when given, the completion ledger."""
    _, units_sha, provenance_sha = sp.load_verified_units(project, corpus)
    receipt_path = sp.receipt_file(project, corpus)
    if not receipt_path.is_file():
        raise ValueError(f"{corpus}: missing score receipt {receipt_path.name}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    shard_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {"schema_version": 1, "corpus": corpus, "shard": f"{OUT_DIR}/scores-{corpus}.jsonl",
                "shard_sha256": shard_sha, "records": len(records),
                "rater_calls": sum(len(r.get("scores", {})) for r in records),
                "units_sha256": units_sha, "provenance_sha256": provenance_sha,
                "code_sha256": sp.code_hashes()}
    if set(receipt) != RECEIPT_KEYS or receipt != expected:
        raise ValueError(f"{corpus}: score receipt does not match the shard or current inputs/code")
    if ledger is not None:
        entry = ledger.get("shards", {}).get(corpus)
        if (not isinstance(entry, dict) or entry.get("shard_sha256") != shard_sha
                or entry.get("receipt_sha256") != hashlib.sha256(receipt_path.read_bytes()).hexdigest()):
            raise ValueError(f"{corpus}: completion ledger digest does not match the shard/receipt")
    return shard_sha


def validate_score_shard(project: Path, corpus: str, records, *, runtime_manifest=RUNTIME_MANIFEST):
    """Fail closed unless the shard is exactly the scorer's output over the verified units."""
    units, units_sha, provenance_sha = sp.load_verified_units(project, corpus)
    _, config_sha, receipt_sha = sp.validate_rater_frozen_inputs(project, sp.FROZEN_CONFIG, sp.FROZEN_RECEIPT)
    runtime_sha = hashlib.sha256((project / runtime_manifest).read_bytes()).hexdigest()
    by_id = {u["unit_id"]: u for u in units}
    passes = dict(sp.scoring_passes(corpus))
    item_ids = {trait: [item[0] for item in sp.rubric_items(project / "refine-logs/RUBRIC_BEHAVIOR.md", trait)]
                for trait in ("extraversion", "agreeableness")}
    expected = {(u["unit_id"], rendering) for u in units for rendering in passes}
    seen, shared_ref = set(), None
    for rec in records:
        key = (rec.get("unit_id"), rec.get("rendering"))
        if key not in expected:
            raise ValueError(f"{corpus}: unexpected record {key}")
        if key in seen:
            raise ValueError(f"{corpus}: duplicate record {key}")
        seen.add(key)
        if set(rec) != RECORD_KEYS or rec["schema_version"] != 1:
            raise ValueError(f"{corpus}: {key} has an unexpected schema or key set")
        unit = by_id[key[0]]
        if (rec.get("corpus") != corpus or rec.get("identity") != unit["identity"]
                or rec.get("cluster") != unit["cluster"] or rec.get("labels") != unit["labels"]):
            raise ValueError(f"{corpus}: {key} disagrees with its verified unit")
        shared = {k: rec.get(k) for k in sp.SHARED_KEYS}
        if shared_ref is None:
            shared_ref = shared
        elif shared != shared_ref:
            raise ValueError(f"{corpus}: {key} carries different run metadata")
        tokens = rec.get("max_prompt_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"{corpus}: {key} has no prompt-token count")
        # Exclusion is recomputable: it holds iff the true token maximum hits the context limit.
        if (rec.get("status") == "excluded_context") != sp.context_excluded(tokens):
            raise ValueError(f"{corpus}: {key} exclusion status disagrees with its token count")
        traits = passes[key[1]]
        if rec.get("status") == "excluded_context":
            if (rec.get("scores") != {} or rec.get("rater_raw_by_item") != {} or rec.get("parse_errors") != []
                    or rec.get("trait_scores") != {}):
                raise ValueError(f"{corpus}: excluded record {key} carries scores")
            continue
        if rec.get("status") != "scored":
            raise ValueError(f"{corpus}: {key} has unknown status")
        ids = [i for trait in traits for i in item_ids[trait]]
        if list(rec["scores"]) != ids or list(rec["rater_raw_by_item"]) != ids:
            raise ValueError(f"{corpus}: {key} item set differs")
        errors = []
        for i in ids:
            try:
                value = sp.parse_item_score(rec["rater_raw_by_item"][i])
            except ValueError:
                value = None
                errors.append(i)
            if rec["scores"][i] != value or (value is not None and type(rec["scores"][i]) is not int):
                raise ValueError(f"{corpus}: {key} {i} score does not decode from its raw output")
        if rec["parse_errors"] != errors:
            raise ValueError(f"{corpus}: {key} parse errors differ from the raw outputs")
        if list(rec["trait_scores"]) != list(traits):
            raise ValueError(f"{corpus}: {key} trait set differs")
        for trait in traits:
            ordered = [rec["scores"][i] for i in item_ids[trait]]
            mean = None if any(v is None for v in ordered) else float(sp.trait_score(ordered).item())
            if rec["trait_scores"][trait] != mean:
                raise ValueError(f"{corpus}: {key} {trait} mean does not recompute")
    if seen != expected:
        raise ValueError(f"{corpus}: {len(expected - seen)} expected records missing")
    calls = sum(len(r["scores"]) for r in records)
    checks = {"rater_decoding": sp.RATER_DECODING_RECORD, "prompt_config_sha256": config_sha,
              "freeze_receipt_sha256": receipt_sha, "rater_runtime_manifest_sha256": runtime_sha,
              "units_sha256": units_sha, "provenance_sha256": provenance_sha, "code_sha256": sp.code_hashes(),
              "max_model_len": sp.RATER_MAX_MODEL_LEN, "eligible_for_gate": False,
              "eligible_for_analysis": True, "rater_calls": calls}
    for name, value in checks.items():
        if shared_ref[name] != value or type(shared_ref[name]) is not type(value):
            raise ValueError(f"{corpus}: {name} does not match the current verified inputs/code")
    elapsed = shared_ref["scoring_elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed <= 0:
        raise ValueError(f"{corpus}: scoring_elapsed_seconds must be positive")
    if shared_ref["rater_calls_per_second"] != (sp.measured_rate(calls, elapsed) if calls else 0.0):
        raise ValueError(f"{corpus}: rater_calls_per_second does not recompute")
    if (not isinstance(shared_ref["doc_sha256"], dict)
            or set(shared_ref["doc_sha256"]) != set(sp.DOC_FILES)):
        raise ValueError(f"{corpus}: doc_sha256 has an unexpected key set")
    return shared_ref


def unit_table(records, rendering):
    rows = []
    for r in records:
        if r["rendering"] != rendering or r["status"] != "scored":
            continue
        row = {"unit_id": r["unit_id"], "identity": r["identity"], "cluster": r["cluster"],
               "EXT": r["labels"]["EXT"], "AGR": r["labels"]["AGR"],
               "E": r["trait_scores"].get("extraversion"), "A": r["trait_scores"].get("agreeableness")}
        row.update({k: v for k, v in r["scores"].items()})
        rows.append(row)
    return rows


def evaluate(design, corpus, rows, score_key, label_key):
    use = [r for r in rows if r.get(score_key) is not None and r[label_key] in ("high", "low")]
    scores = [r[score_key] for r in use]
    labels = [r[label_key] for r in use]
    identities = [r["identity"] for r in use]
    clusters = [r["cluster"] for r in use]
    return design.metrics(scores, labels, identities, clusters), len(use)


def summarise(metric_draws):
    return {name: interval(draws[0], draws[1:]) for name, draws in metric_draws.items()}


def contrast(a, b):
    return a - b  # NaN propagates: a draw is valid only when both cells are


def reduce_corpus(corpus, records, *, n_boot=N_BOOT, seed=BOOT_SEED):
    rows = unit_table(records, "full")
    # The bootstrap design spans every identity/scene scored in any rendering, so a unit that
    # overflows the context only in 'full' can still enter the 'no_directions' check; when both
    # renderings score the same units this is exactly the 'full' design.
    scored = [r for r in records if r["status"] == "scored"]
    if corpus == "friendspersona":
        design = FriendsDesign([r["identity"] for r in scored], [r["cluster"] for r in scored],
                               n_boot, seed)
    else:
        design = ClusterDesign([r["cluster"] for r in scored], n_boot, seed)
    out = {"n_scored_units": len(rows),
           "n_excluded_context": sum(r["status"] == "excluded_context" and r["rendering"] == "full"
                                     for r in records),
           "parse_errors": parse_error_report(records),
           "missingness": missingness_report(records),
           "max_prompt_tokens": max(r["max_prompt_tokens"] for r in records),
           "max_prompt_tokens_scored": max((r["max_prompt_tokens"] for r in records if r["status"] == "scored"),
                                           default=None)}

    main, n_main = evaluate(design, corpus, rows, "E", "EXT")
    out["E_to_EXT_all_parsed"] = {"n_units": n_main, **summarise(main)}
    if corpus == "friendspersona":
        use = [r for r in rows if r["E"] is not None]
        n_id, n_pairs = informative([r["EXT"] for r in use], [r["identity"] for r in use])
        out["E_to_EXT_all_parsed"]["informative_identities"] = n_id
        out["E_to_EXT_all_parsed"]["informative_pairs"] = n_pairs

    complete = [r for r in rows if r["E"] is not None and r["A"] is not None
                and r["EXT"] in ("high", "low") and r["AGR"] in ("high", "low")]
    cells = {}
    for score_key in ("E", "A"):
        for label_key in ("EXT", "AGR"):
            cells[(score_key, label_key)], _ = evaluate(design, corpus, complete, score_key, label_key)
    matrix = {f"{s}_to_{l}": summarise({"primary": cells[(s, l)]["primary"]})["primary"]
              for (s, l) in cells}
    e_spec = contrast(cells[("E", "EXT")]["primary"], cells[("A", "EXT")]["primary"])
    a_spec = contrast(cells[("A", "AGR")]["primary"], cells[("E", "AGR")]["primary"])
    did = e_spec - contrast(cells[("E", "AGR")]["primary"], cells[("A", "AGR")]["primary"])
    out["trait_specificity"] = {
        "n_common_complete_units": len(complete),
        "n_dropped_from_scored": len(rows) - len(complete),
        "matrix": matrix,
        "E_specificity": interval(e_spec[0], e_spec[1:]),
        "A_specificity": interval(a_spec[0], a_spec[1:]),
        "difference_in_differences": interval(did[0], did[1:]),
        "label": "descriptive trait specificity",
    }

    per_item = {}
    for item in E_ITEMS + A_ITEMS:
        for label_key in ("EXT", "AGR"):
            draws, n = evaluate(design, corpus, rows, item, label_key)
            per_item[f"{item}_to_{label_key}"] = {"n_units": n, **summarise({"primary": draws["primary"]})["primary"]}
    out["per_item"] = per_item

    if corpus == "friendspersona":
        stripped = unit_table(records, "no_directions")
        draws, n = evaluate(design, corpus, stripped, "E", "EXT")
        out["E_to_EXT_no_directions"] = {"n_units": n, **summarise({"primary": draws["primary"]})["primary"]}
    return out


def parse_error_report(records):
    report = defaultdict(int)
    for r in records:
        for item in r.get("parse_errors", []):
            report[f"{r['rendering']}|{item}|EXT={r['labels']['EXT']}|AGR={r['labels']['AGR']}"
                   f"|cluster={r['cluster']}"] += 1
    return {"total": sum(report.values()), "by_rendering_item_labels_cluster": dict(sorted(report.items()))}


def missingness_report(records):
    """Mutually exclusive unit outcomes by rendering, EXT/AGR label and cluster."""
    report = defaultdict(int)
    for r in records:
        key = f"{r['rendering']}|EXT={r['labels']['EXT']}|AGR={r['labels']['AGR']}|cluster={r['cluster']}"
        if r["status"] == "excluded_context":
            outcome = "context_exclusion"
        else:
            e_fail = r["trait_scores"].get("extraversion") is None
            a_fail = "agreeableness" in r["trait_scores"] and r["trait_scores"]["agreeableness"] is None
            outcome = {(False, False): "complete", (True, False): "E_only_failure",
                       (False, True): "A_only_failure", (True, True): "both_failure"}[(e_fail, a_fail)]
        report[f"{key}|{outcome}"] += 1
        for label_key in ("EXT", "AGR"):
            if r["labels"][label_key] not in ("high", "low"):
                report[f"{key}|unknown_{label_key}_label"] += 1
    return dict(sorted(report.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=".")
    ap.add_argument("--validate-only", action="store_true",
                    help="validate both score shards against their receipts, the verified units and "
                         "current code; write nothing")
    ap.add_argument("--ledger", help="completion ledger written by chain_h_public.sh (required to reduce)")
    ap.add_argument("--runtime-manifest", default=RUNTIME_MANIFEST)
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    out_dir = project / OUT_DIR
    if not args.validate_only and not args.ledger:
        raise ValueError("reduction requires --ledger (the completion ledger with the shard digests)")
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8")) if args.ledger else None
    loaded, digests = {}, {}
    for corpus in ("friendspersona", "cped"):
        path = out_dir / f"scores-{corpus}.jsonl"
        records = load_scores(path)
        digests[corpus] = {
            "shard_sha256": validate_receipt(project, corpus, path, records, ledger),
            "receipt_sha256": hashlib.sha256(sp.receipt_file(project, corpus).read_bytes()).hexdigest()}
        loaded[corpus] = (path, records, validate_score_shard(project, corpus, records,
                                                              runtime_manifest=args.runtime_manifest))
    if args.validate_only:
        print(json.dumps({"ok": True, "records": {c: len(v[1]) for c, v in loaded.items()},
                          "shards": digests}))
        return 0
    target = out_dir / "H_PUBLIC_SUMMARY.json"
    if target.exists():
        raise ValueError(f"refusing to overwrite existing summary: {target}")
    summary = {"schema_version": 1, "spec": "refine-logs/followup/H_public_spec.md",
               "name": "Descriptive label concordance of a frozen support-oriented score on two public "
                       "scripted-dialogue corpora",
               "descriptive_only": True, "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED,
                                                       "interval": "percentile 2.5/97.5"},
               "inputs": {}, "run_metadata": {}, "corpora": {}}
    for corpus, (path, records, shared) in loaded.items():
        summary["inputs"][f"{OUT_DIR}/scores-{corpus}.jsonl"] = digests[corpus]["shard_sha256"]
        summary["inputs"][f"{OUT_DIR}/scores-{corpus}.receipt.json"] = digests[corpus]["receipt_sha256"]
        summary["run_metadata"][corpus] = shared
        summary["corpora"][corpus] = reduce_corpus(corpus, records)
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps({c: v["E_to_EXT_all_parsed"]["primary"] for c, v in summary["corpora"].items()},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
