#!/usr/bin/env python3
"""Reduce the supplementary score shards of namespace supp-20261004 (descriptive, CPU only).

Inputs
* the frozen canonical Mistral pole scores (``<phase>-rater[-rescue]/<generator>/seed-*.jsonl``),
  read only, never rewritten;
* the supplementary shards written by ``supp_score.py`` under
  ``process/results/xlpt/supp-20261004/`` (second and third rater, surface interventions,
  Chinese rubric rendering, public-label units).

Sections of the output
* ``canonical``      -- frozen-rater separations, linear contrasts and ratios on the all-cell mask
  with seed-only and crossed seed x scenario intervals, for the trait mean, the mean without E1,
  the three facets and every item;
* ``raters``         -- the same estimands for every rater on the supplementary seed subset;
* ``agreement``      -- transcript- and item-level agreement between raters;
* ``surface``        -- separation retained after each surface intervention;
* ``zh_rubric``      -- Chinese rubric rendering against the English rubric, same rater;
* ``public``         -- public-label AUCs per rater, leave-one-identity/cluster-out, length baseline;
* ``surface_model``  -- how much of the frozen score a few surface features predict
  (scenario-grouped cross-validation) and the length overlap of the two poles;
* ``script``         -- script of the agent turns in every executed pole cell;
* ``missing``        -- missing transcripts / scores against the masked pair positions;
* ``sanity``         -- synthetic checks of this reducer (not construct-validity evidence);
* ``shards``         -- parse errors and exclusions per supplementary shard.

Everything is exploratory. Every interval is a pointwise percentile interval conditional on the
exposed scenario bank, the prompt renderings and the rater. No gate is read or written.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reduce_h_public as rhp  # noqa: E402

NAMESPACE = "supp-20261004"
BOOT_SEED = 20261004
BOOT_DRAWS = 9999
CELLS = {
    "en_en": ("Pen_Den", "b4-extraversion-high", "b4-extraversion-low"),
    "en_zh": ("Pen_Dzh", "zh-b4-extraversion-high", "zh-b4-extraversion-low"),
    "zh_en": ("Pzh_Den", "bc-pzhn-extraversion-high-den", "bc-pzhn-extraversion-low-den"),
    "zh_zh": ("Pzh_Dzh", "bc-pzhn-extraversion-high-dzh", "bc-pzhn-extraversion-low-dzh"),
}
SUPP_CELL = {supp: cell for cell, (supp, _, _) in CELLS.items()}
GENERATORS = ("qwen3-8b", "aya-expanse-8b", "Llama-3.1-8B-Instruct", "gemma-3-12b-it")
P_EN_ONLY = ("gemma-3-12b-it",)
SELF = {"llama": "Llama-3.1-8B-Instruct", "gemma": "gemma-3-12b-it"}
E_ITEMS = tuple(f"E{i}" for i in range(1, 7))
A_ITEMS = tuple(f"A{i}" for i in range(1, 7))
SCORES = {
    "E": E_ITEMS,
    "E_without_E1": E_ITEMS[1:],
    "E_sociability": ("E1", "E2"),
    "E_assertiveness": ("E3", "E4"),
    "E_energy": ("E5", "E6"),
    **{item: (item,) for item in E_ITEMS},
    "A": A_ITEMS,
}
LINEAR = {
    "d_effect_at_p_en": {"en_en": 1, "en_zh": -1},
    "d_effect_at_p_zh": {"zh_en": 1, "zh_zh": -1},
    "p_effect_at_d_en": {"en_en": 1, "zh_en": -1},
    "p_effect_at_d_zh": {"en_zh": 1, "zh_zh": -1},
    "interaction_match_minus_mismatch": {"en_en": 1, "zh_zh": 1, "en_zh": -1, "zh_en": -1},
}
RATIOS = {
    "ratio_en_zh_over_en_en": ("en_zh", "en_en"),
    "ratio_zh_en_over_en_en": ("zh_en", "en_en"),
    "ratio_zh_zh_over_en_en": ("zh_zh", "en_en"),
    "ratio_zh_zh_over_zh_en": ("zh_zh", "zh_en"),
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def value(items: dict, names) -> float:
    """Mean of the named items; NaN unless every one of them was parsed."""
    try:
        return float(np.mean([items[name] for name in names]))
    except (KeyError, TypeError):
        return float("nan")


# ------------------------------------------------------------------ loading

def load_canonical(root: Path) -> dict:
    """(cell, generator, pole, seed, scenario) -> {item: score} from the frozen rater shards."""
    table = defaultdict(dict)
    for cell, (_, high, low) in CELLS.items():
        for pole, phase in (("high", high), ("low", low)):
            for suffix in ("-rater", "-rater-rescue"):
                for gen in GENERATORS:
                    for path in sorted((root / f"{phase}{suffix}" / gen).glob("seed-*.jsonl")):
                        for row in read_jsonl(path):
                            key = (cell, gen, pole, int(row["seed"]), row["source_scenario_id"])
                            table[key].update({k: v for k, v in (row.get("scores") or {}).items()
                                               if isinstance(v, int)})
    return dict(table)


def load_supp(root: Path, tag: str):
    """variant -> table like ``load_canonical``; None when the shard does not exist yet."""
    path = root / NAMESPACE / f"{tag}.jsonl"
    if not path.exists():
        return None, None
    tables = defaultdict(dict)
    report = {"records": 0, "excluded_context": 0, "records_with_parse_error": 0,
              "item_parse_errors": 0, "items_requested": 0, "sha256": sha256_file(path)}
    for row in read_jsonl(path):
        report["records"] += 1
        report["excluded_context"] += row["status"] != "scored"
        report["records_with_parse_error"] += bool(row.get("parse_errors"))
        report["item_parse_errors"] += len(row.get("parse_errors") or [])
        report["items_requested"] += len(row.get("rater_raw_by_item") or {})
        if "cell" not in row:
            continue
        key = (SUPP_CELL[row["cell"]], row["checkpoint_id"], row["pole"], int(row["seed"]),
               row["source_scenario_id"])
        tables[row["variant"]][key] = {k: v for k, v in (row.get("scores") or {}).items()
                                       if isinstance(v, int)}
    receipt = root / NAMESPACE / f"{tag}.receipt.json"
    if receipt.exists():
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        report["receipt_sha256_matches"] = recorded.get("sha256") == report["sha256"]
        report["calls_per_second"] = recorded.get("calls_per_second")
    return dict(tables), report


# ------------------------------------------------------------------ estimands

class Design:
    """Shared seed and scenario multiplicities, so every block uses the same resamples."""

    def __init__(self, seeds, scenarios, draws=BOOT_DRAWS):
        self.seeds, self.scenarios = list(seeds), list(scenarios)
        rng = np.random.default_rng(BOOT_SEED)
        s, k = len(self.seeds), len(self.scenarios)
        self.ws = rng.multinomial(s, np.full(s, 1 / s), size=draws).astype(float)
        self.wk = rng.multinomial(k, np.full(k, 1 / k), size=draws).astype(float)

    def grid(self, table, gen, cell, names):
        """high - low, (seed, scenario), NaN where either pole has no complete score."""
        out = np.full((len(self.seeds), len(self.scenarios)), np.nan)
        for i, seed in enumerate(self.seeds):
            for j, scenario in enumerate(self.scenarios):
                high = table.get((cell, gen, "high", seed, scenario))
                low = table.get((cell, gen, "low", seed, scenario))
                if high is not None and low is not None:
                    out[i, j] = value(high, names) - value(low, names)
        return out

    def estimate(self, arr):
        """arr: (..., seed, scenario). Returns the point (...,) and the seed-only and crossed
        draws (draws, ...). Seeds weigh equally; scenarios are averaged within seed."""
        seen = ~np.isnan(arr)
        filled = np.where(seen, arr, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_seed = filled.sum(-1) / seen.sum(-1)                       # (..., S)
            point = per_seed.mean(-1)
            seed_only = np.moveaxis(per_seed @ self.ws.T, -1, 0) / len(self.seeds)
            boot = (filled @ self.wk.T) / (seen.astype(float) @ self.wk.T)  # (..., S, B)
            crossed = np.einsum("...sb,bs->b...", boot, self.ws) / len(self.seeds)
        return point, seed_only, crossed


def ci(draws):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if not draws.size:
        return None
    low, high = np.percentile(draws, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def entry(point, seed_only, crossed):
    return {"estimate": None if not np.isfinite(point) else round(float(point), 4),
            "ci95_seed_bootstrap": ci(seed_only), "ci95_crossed_bootstrap": ci(crossed)}


def derive(sep: dict) -> dict:
    """sep: cell -> scalar or array. Separations, linear contrasts and ratios."""
    out = {f"sep_{cell}": v for cell, v in sep.items()}
    for name, signs in LINEAR.items():
        if signs.keys() <= sep.keys():
            out[name] = sum(sign * sep[cell] for cell, sign in signs.items())
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, (num, den) in RATIOS.items():
            if num in sep and den in sep:
                out[name] = sep[num] / sep[den]
    return out


def masked_stack(design, table, gen, names, cells):
    """(cell, seed, scenario) differences on the all-cell mask of one generator."""
    arr = np.stack([design.grid(table, gen, cell, names) for cell in cells])
    mask = ~np.isnan(arr).any(axis=0)
    return np.where(mask[None], arr, np.nan), int(mask.sum())


def block(design, table, gens, names):
    """Per generator and for generator means: every statistic of ``derive`` with intervals."""
    out, kept = {}, {}
    for gen in gens:
        cells = [c for c in CELLS if not (gen in P_EN_ONLY and c.startswith("zh"))]
        arr, n = masked_stack(design, table, gen, names, cells)
        if n == 0:
            continue
        point, seed_only, crossed = design.estimate(arr)
        kept[gen] = (cells, point, seed_only, crossed)
        stats = [derive({c: x[..., i] for i, c in enumerate(cells)})
                 for x in (point, seed_only, crossed)]
        out[gen] = {"pairs_in_mask": n,
                    **{k: entry(stats[0][k], stats[1][k], stats[2][k]) for k in stats[0]}}
    for label, members in (("mean_of_full_design_generators",
                            [g for g in kept if g not in P_EN_ONLY]),
                           ("mean_of_all_generators_p_en_cells", list(kept))):
        if len(members) < 2:
            continue
        cells = [c for c in CELLS if all(c in kept[g][0] for g in members)]
        stats = []
        for which in (1, 2, 3):
            stats.append(derive({c: np.mean([kept[g][which][..., kept[g][0].index(c)]
                                             for g in members], axis=0) for c in cells}))
        out[label] = {"generators": members,
                      **{k: entry(stats[0][k], stats[1][k], stats[2][k]) for k in stats[0]}}
    return out


def axes(*tables):
    keys = set().union(*(t.keys() for t in tables))
    return sorted({k[3] for k in keys}), sorted({k[4] for k in keys})


# ------------------------------------------------------------------ agreement

def quadratic_kappa(a, b, low=1, high=7):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = high - low + 1
    obs = np.zeros((n, n))
    for x, y in zip(a - low, b - low):
        obs[x, y] += 1
    w = (np.arange(n)[:, None] - np.arange(n)[None]) ** 2 / (n - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    denominator = (w * exp).sum()
    return float(1 - (w * obs).sum() / denominator) if denominator else float("nan")


def agreement(table_a, table_b, items, gens=GENERATORS):
    keys = sorted(k for k in table_a.keys() & table_b.keys() if k[1] in gens)
    out = {"units_in_both": len(keys)}
    if not keys:
        return out
    a = np.array([value(table_a[k], items) for k in keys])
    b = np.array([value(table_b[k], items) for k in keys])
    ok = np.isfinite(a) & np.isfinite(b)
    pole = np.array([k[2] for k in keys])
    cell = np.array([k[0] for k in keys])

    def corr(sel):
        sel = sel & ok
        if sel.sum() < 3 or np.std(a[sel]) == 0 or np.std(b[sel]) == 0:
            return None
        return {"n": int(sel.sum()), "pearson": round(float(sps.pearsonr(a[sel], b[sel])[0]), 4),
                "spearman": round(float(sps.spearmanr(a[sel], b[sel])[0]), 4),
                "mean_difference_a_minus_b": round(float((a[sel] - b[sel]).mean()), 4)}

    out["trait_mean"] = {"all": corr(np.ones(len(keys), bool)),
                         "within_high": corr(pole == "high"), "within_low": corr(pole == "low"),
                         **{f"cell_{c}": corr(cell == c) for c in CELLS}}
    per_item = {}
    for item in items:
        pairs = [(table_a[k][item], table_b[k][item]) for k in keys
                 if item in table_a[k] and item in table_b[k]]
        if not pairs:
            continue
        x, y = np.array(pairs).T
        per_item[item] = {"n": len(pairs), "exact": round(float((x == y).mean()), 4),
                          "within_one": round(float((abs(x - y) <= 1).mean()), 4),
                          "mean_difference_a_minus_b": round(float((x - y).mean()), 4),
                          "quadratic_weighted_kappa": round(quadratic_kappa(x, y), 4)}
    out["per_item"] = per_item
    return out


# ------------------------------------------------------------------ paired interventions

def paired(design, base, other, gens, names):
    """Separation under ``other`` against ``base`` on their joint all-cell mask."""
    out = {}
    for gen in gens:
        cells = [c for c in CELLS if not (gen in P_EN_ONLY and c.startswith("zh"))]
        a = np.stack([design.grid(base, gen, c, names) for c in cells])
        b = np.stack([design.grid(other, gen, c, names) for c in cells])
        mask = ~(np.isnan(a).any(axis=0) | np.isnan(b).any(axis=0))
        if not mask.any():
            continue
        est = [design.estimate(np.where(mask[None], x, np.nan)) for x in (a, b)]
        res = {"pairs_in_joint_mask": int(mask.sum())}
        for which, name in ((0, "base"), (1, "other")):
            stats = [derive({c: est[which][j][..., i] for i, c in enumerate(cells)})
                     for j in range(3)]
            res[name] = {k: entry(stats[0][k], stats[1][k], stats[2][k]) for k in stats[0]}
        with np.errstate(invalid="ignore", divide="ignore"):
            for i, c in enumerate(cells):
                res[f"retained_{c}"] = entry(*(est[1][j][..., i] / est[0][j][..., i]
                                               for j in range(3)))
                res[f"change_{c}"] = entry(*(est[1][j][..., i] - est[0][j][..., i]
                                             for j in range(3)))
            res["retained_mean_over_cells"] = entry(*(est[1][j].mean(-1) / est[0][j].mean(-1)
                                                      for j in range(3)))
        out[gen] = res
    return out


def pole_shift(base, other, names):
    """Mean score change per pole and cell (unit level, pooled over generators)."""
    out = {}
    for cell in CELLS:
        for pole in ("high", "low"):
            d = [value(other[k], names) - value(base[k], names)
                 for k in base.keys() & other.keys() if k[0] == cell and k[2] == pole]
            d = [x for x in d if np.isfinite(x)]
            if d:
                out[f"{cell}|{pole}"] = {"n": len(d), "mean_change": round(float(np.mean(d)), 4)}
    return out


# ------------------------------------------------------------------ public labels

def public_records(root: Path, tag: str):
    path = root / NAMESPACE / f"{tag}.jsonl"
    if not path.exists():
        return None
    records = read_jsonl(path)
    for r in records:
        r["rendering"] = "full"
    return records


def auc_points(corpus, rows, design_units, score_key="E", label_key="EXT"):
    design = (rhp.FriendsDesign([r["identity"] for r in design_units],
                                [r["cluster"] for r in design_units], 1, 0)
              if corpus == "friendspersona"
              else rhp.ClusterDesign([r["cluster"] for r in design_units], 1, 0))
    draws, n = rhp.evaluate(design, corpus, rows, score_key, label_key)
    return {name: (None if np.isnan(v[0]) else round(float(v[0]), 4)) for name, v in draws.items()
            } | {"n_units": n}


def public_block(root: Path, corpus: str, records, min_units=20):
    # The supplementary shards score the 'full' rendering only; the canonical reducer also reads a
    # 'no_directions' rendering for FriendsPersona, so that key is fed the full rows and dropped.
    table = rhp.unit_table
    rhp.unit_table = lambda recs, rendering: table(recs, "full")
    try:
        out = {"summary": rhp.reduce_corpus(corpus, records)}
    finally:
        rhp.unit_table = table
    out["summary"].pop("E_to_EXT_no_directions", None)
    rows = rhp.unit_table(records, "full")
    lengths = {u["unit_id"]: u["text_chars"] for u in read_jsonl(
        root / "h-public" / f"unit-metadata-{corpus}.jsonl")}
    for r in rows:
        r["LEN"] = lengths[r["unit_id"]]
    out["length_baseline_to_EXT"] = auc_points(corpus, rows, rows, "LEN")
    group = "identity" if corpus == "friendspersona" else "cluster"
    counts = defaultdict(int)
    for r in rows:
        counts[r[group]] += 1
    loo = {}
    for name, n in sorted(counts.items()):
        if n < min_units:
            continue
        kept = [r for r in rows if r[group] != name]
        loo[name] = {"units_removed": n, **auc_points(corpus, kept, kept)}
    primary = [v["primary"] for v in loo.values() if v["primary"] is not None]
    out[f"leave_one_{group}_out"] = {
        "groups_with_at_least": min_units, "by_group": loo,
        "primary_range": [min(primary), max(primary)] if primary else None}
    return out


# ------------------------------------------------------------------ surface features

FEATURES = ("log_agent_chars", "exclamations", "questions", "sentences", "log_first_turn_chars",
            "log_last_turn_chars")
SENTENCE_END = re.compile(r"[。！？!?…]+|\.(?=\s|$)|\n")


def features(turns) -> list:
    text = "\n".join(turns)
    return [np.log(len(text)), len(re.findall(r"[!！]", text)), len(re.findall(r"[?？]", text)),
            len(SENTENCE_END.findall(text)), np.log(len(turns[0])), np.log(len(turns[-1]))]


def load_transcripts(root: Path) -> dict:
    out = {}
    for cell, (_, high, low) in CELLS.items():
        for pole, phase in (("high", high), ("low", low)):
            for gen in GENERATORS:
                for path in sorted((root / phase / gen).glob("seed-*.jsonl")):
                    for row in read_jsonl(path):
                        turns = row.get("agent_turns") or []
                        if turns and all(isinstance(t, str) and t.strip() for t in turns):
                            out[(cell, gen, pole, int(row["seed"]), row["scenario_id"])] = turns
    return out


def grouped_cv_r2(x, y, groups, folds=5, ridge=1.0):
    """Out-of-fold R^2 of a ridge regression with whole scenarios held out."""
    names = sorted(set(groups))
    fold_of = {name: i % folds for i, name in enumerate(names)}
    fold = np.array([fold_of[g] for g in groups])
    pred = np.empty(len(y))
    for f in range(folds):
        train, test = fold != f, fold == f
        mean, std = x[train].mean(0), x[train].std(0)
        std[std == 0] = 1.0
        a = np.c_[np.ones(train.sum()), (x[train] - mean) / std]
        penalty = ridge * np.eye(a.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(a.T @ a + penalty, a.T @ y[train])
        pred[test] = np.c_[np.ones(test.sum()), (x[test] - mean) / std] @ beta
    total = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / total) if total else float("nan")


def auc(high, low):
    return float(rhp.kernel(high, low).mean())


def surface_model(transcripts, canonical):
    out = {}
    for gen in GENERATORS:
        for cell in CELLS:
            keys = sorted(k for k in transcripts.keys() & canonical.keys()
                          if k[0] == cell and k[1] == gen and np.isfinite(value(canonical[k], E_ITEMS)))
            if not keys:
                continue
            x = np.array([features(transcripts[k]) for k in keys], dtype=float)
            y = np.array([value(canonical[k], E_ITEMS) for k in keys])
            pole = np.array([k[2] for k in keys])
            scenario = [k[4] for k in keys]
            chars = np.exp(x[:, 0])
            hi, lo = chars[pole == "high"], chars[pole == "low"]
            shared = (max(hi.min(), lo.min()), min(hi.max(), lo.max()))
            res = {
                "n": len(keys),
                "r2_all_features_both_poles": round(grouped_cv_r2(x, y, scenario), 4),
                "r2_length_only_both_poles": round(grouped_cv_r2(x[:, :1], y, scenario), 4),
                "r2_pole_only": round(grouped_cv_r2((pole == "high")[:, None].astype(float), y,
                                                    scenario), 4),
                "r2_pole_plus_features": round(grouped_cv_r2(
                    np.c_[(pole == "high").astype(float), x], y, scenario), 4),
                "pole_auc_by_feature": {name: round(auc(x[pole == "high", i], x[pole == "low", i]), 4)
                                        for i, name in enumerate(FEATURES)},
                "pole_auc_frozen_E_score": round(auc(y[pole == "high"], y[pole == "low"]), 4),
                "median_agent_chars": {"high": float(np.median(hi)), "low": float(np.median(lo))},
                "shared_length_support_chars": [float(shared[0]), float(shared[1])],
            }
            for name, side in (("high", hi), ("low", lo)):
                inside = (side >= shared[0]) & (side <= shared[1])
                res[f"{name}_in_shared_support"] = {"n": int(inside.sum()), "of": int(side.size),
                                                    "share": round(float(inside.mean()), 4)}
            for name in ("high", "low"):
                sel = pole == name
                res[f"r2_all_features_within_{name}"] = round(grouped_cv_r2(
                    x[sel], y[sel], [s for s, keep in zip(scenario, sel) if keep]), 4)
                res[f"spearman_length_score_within_{name}"] = round(float(
                    sps.spearmanr(x[sel, 0], y[sel])[0]), 4)
            out[f"{gen}|{cell}"] = res
    return out


# ------------------------------------------------------------------ script of the replies

TONE_MARKS = set("āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ")


def is_han(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF


def turn_script(text: str):
    """(share of Han among Han and other letters, tone-marked Latin letters)."""
    han = sum(1 for ch in text if is_han(ch))
    other = sum(1 for ch in text if ch.isalpha() and not is_han(ch))
    return (han / (han + other) if han + other else float("nan")), sum(ch in TONE_MARKS for ch in text)


def script_report(transcripts) -> dict:
    """Per generator, cell and pole: turns in the dialogue language under the 0.5 Han-share
    rule of the response-language screen, mixed-script turns and tone-marked letters.
    Descriptive; it does not pass or fail any screen after the fact."""
    groups = defaultdict(list)
    for (cell, gen, pole, _seed, _scenario), turns in transcripts.items():
        groups[(gen, cell, pole)].append(turns)
    out = {}
    for (gen, cell, pole), items in sorted(groups.items()):
        chinese_dialogue = cell.endswith("_zh")
        turns = in_language = mixed = tone_turns = tone_letters = clean = 0
        for transcript in items:
            ok = True
            for turn in transcript:
                share, tones = turn_script(turn)
                turns += 1
                right = (share >= 0.5) if chinese_dialogue else (share < 0.5)
                in_language += bool(right)
                is_mixed = 0.05 < share < 0.95
                mixed += is_mixed
                tone_turns += tones > 0
                tone_letters += tones
                ok = ok and right
            clean += ok
        out[f"{gen}|{cell}|{pole}"] = {
            "transcripts": len(items), "turns": turns,
            "turn_rate_in_dialogue_language": round(in_language / turns, 4),
            "transcripts_with_every_turn_in_dialogue_language": clean,
            "transcript_rate_every_turn_in_dialogue_language": round(clean / len(items), 4),
            "turns_mixed_script_5_to_95_percent_han": mixed,
            "turn_rate_mixed_script": round(mixed / turns, 4),
            "turns_with_tone_marked_letters": tone_turns, "tone_marked_letters": tone_letters}
    return out


# ------------------------------------------------------------------ missingness

def missing_report(transcripts, canonical, seeds, scenarios):
    out = {}
    for gen in GENERATORS:
        cells = [c for c in CELLS if not (gen in P_EN_ONLY and c.startswith("zh"))]
        no_transcript, no_score, positions = [], [], set()
        for cell in cells:
            for pole in ("high", "low"):
                for seed in seeds:
                    for scenario in scenarios:
                        key = (cell, gen, pole, seed, scenario)
                        where = {"cell": cell, "pole": pole, "seed": seed, "scenario": scenario}
                        if key not in transcripts:
                            no_transcript.append(where)
                            positions.add((seed, scenario))
                        elif not np.isfinite(value(canonical.get(key, {}), E_ITEMS)):
                            no_score.append(where)
                            positions.add((seed, scenario))
        out[gen] = {"cells": cells, "transcripts_missing_or_invalid": no_transcript,
                    "scored_transcripts_without_complete_E": no_score,
                    "distinct_seed_scenario_positions_masked": len(positions),
                    "pairs_possible": len(seeds) * len(scenarios)}
    out["totals"] = {
        "transcripts_missing_or_invalid": sum(len(v["transcripts_missing_or_invalid"])
                                              for v in out.values()),
        "scored_transcripts_without_complete_E": sum(
            len(v["scored_transcripts_without_complete_E"]) for v in out.values()),
        "distinct_positions_masked_summed_over_generators": sum(
            v["distinct_seed_scenario_positions_masked"] for v in out.values())}
    return out


# ------------------------------------------------------------------ reducer sanity checks

def sanity() -> dict:
    rng = np.random.default_rng(7)
    seeds, scenarios = list(range(5)), [f"s{i}" for i in range(8)]
    design = Design(seeds, scenarios, draws=499)
    shift = {"en_en": 2.0, "en_zh": 1.0, "zh_en": 1.5, "zh_zh": 0.5}
    table = {}
    for cell in CELLS:
        for seed in seeds:
            for scenario in scenarios:
                low = {item: int(rng.integers(1, 4)) for item in E_ITEMS}
                high = dict(low)
                high["E1"] = low["E1"] + int(6 * shift[cell] / 2)   # all separation sits in E1
                table[(cell, "g", "high", seed, scenario)] = high
                table[(cell, "g", "low", seed, scenario)] = low
    swapped = {(c, g, "low" if p == "high" else "high", s, k): v
               for (c, g, p, s, k), v in table.items()}
    a = block(design, table, ("g",), E_ITEMS)["g"]
    b = block(design, swapped, ("g",), E_ITEMS)["g"]
    without = block(design, table, ("g",), E_ITEMS[1:])["g"]
    holed = dict(table)
    del holed[("en_zh", "g", "low", 0, "s0")]
    c = block(design, holed, ("g",), E_ITEMS)["g"]
    expected = {cell: int(6 * shift[cell] / 2) / 6 for cell in CELLS}
    return {
        "pole_swap_reverses_sign": all(
            abs(a[k]["estimate"] + b[k]["estimate"]) < 1e-9 for k in a
            if k.startswith(("sep_", "d_", "p_", "interaction"))),
        "pole_swap_preserves_ratios": all(
            abs(a[k]["estimate"] - b[k]["estimate"]) < 1e-9 for k in a if k.startswith("ratio_")),
        "planted_separations_recovered": all(
            abs(a[f"sep_{cell}"]["estimate"] - expected[cell]) < 1e-3 for cell in CELLS),
        "separation_vanishes_without_planted_item": all(
            abs(without[f"sep_{cell}"]["estimate"]) < 1e-9 for cell in CELLS),
        "one_missing_score_masks_one_pair_in_every_cell": c["pairs_in_mask"] == a["pairs_in_mask"] - 1,
        "note": "Synthetic checks of reducer arithmetic only; they say nothing about construct validity.",
    }


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default=f"process/results/xlpt/{NAMESPACE}/SUPP_SUMMARY.json")
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    root = project / "process/results/xlpt"

    canonical = load_canonical(root)
    transcripts = load_transcripts(root)
    seeds, scenarios = axes(canonical)
    full = Design(seeds, scenarios)
    out = {"schema_version": 1, "namespace": NAMESPACE, "verdict": "DESCRIPTIVE_ONLY",
           "eligible_for_gate": False,
           "bootstrap": {"seed": BOOT_SEED, "draws": BOOT_DRAWS,
                         "note": "seed and scenario multiplicities are shared by every block with "
                                 "the same seed set, so paired comparisons use the same resamples"},
           "reducer_sha256": sha256_file(Path(__file__).resolve()), "shards": {}}

    out["canonical"] = {"n_seeds": len(seeds), "n_scenarios": len(scenarios),
                        "scores": {name: block(full, canonical, GENERATORS, items)
                                   for name, items in SCORES.items()}}
    out["missing"] = missing_report(transcripts, canonical, seeds, scenarios)
    out["surface_model"] = surface_model(transcripts, canonical)
    out["script"] = script_report(transcripts)
    out["sanity"] = sanity()

    supp = {}
    for tag in ("s0-mistral-none", "s1-llama-none", "s1-gemma-none", "s2-mistral-surface",
                "s2-llama-surface", "s2-gemma-surface", "s3-mistral-zh", "s3-llama-zh",
                "s3-gemma-zh"):
        tables, report = load_supp(root, tag)
        if tables is not None:
            supp[tag] = tables
            out["shards"][tag] = report

    def restrict(table, seed_set):
        return {k: v for k, v in table.items() if k[3] in seed_set}

    # baseline (rubric as frozen, transcript unmodified) per rater
    base = {"mistral": canonical}
    for rater in ("llama", "gemma"):
        if f"s1-{rater}-none" in supp:
            base[rater] = supp[f"s1-{rater}-none"]["none"]

    out["raters"], out["agreement"] = {}, {}
    if len(base) > 1:
        sub_seeds = sorted({k[3] for r in base if r != "mistral" for k in base[r]})
        sub = Design(sub_seeds, scenarios)
        out["raters"]["seeds"] = sub_seeds
        for rater, table in base.items():
            table = restrict(table, set(sub_seeds))
            others = tuple(g for g in GENERATORS if g != SELF.get(rater))
            out["raters"][rater] = {
                name: {"all_generators": block(sub, table, GENERATORS, SCORES[name]),
                       **({"excluding_self_rated_generator": block(sub, table, others, SCORES[name])}
                          if rater in SELF else {})}
                for name in ("E", "E_without_E1", "E_sociability", "E_assertiveness", "E_energy", "A")}
        names = list(base)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair = {}
                for trait, items in (("E", E_ITEMS), ("A", A_ITEMS)):
                    pair[trait] = agreement(base[a], base[b], items)
                    not_self = tuple(g for g in GENERATORS if g not in (SELF.get(a), SELF.get(b)))
                    pair[f"{trait}_excluding_self_rated_generators"] = agreement(
                        base[a], base[b], items, not_self)
                out["agreement"][f"{a}_vs_{b}"] = pair
    if "s0-mistral-none" in supp:
        out["agreement"]["mistral_batched_vs_canonical"] = {
            trait: agreement(supp["s0-mistral-none"]["none"], canonical, items)
            for trait, items in (("E", E_ITEMS), ("A", A_ITEMS))}

    out["surface"], out["zh_rubric"] = {}, {}
    for rater, table in base.items():
        tag = f"s2-{rater}-surface"
        if tag in supp:
            seed_set = sorted({k[3] for t in supp[tag].values() for k in t})
            design = Design(seed_set, scenarios)
            ref = restrict(table, set(seed_set))
            out["surface"][rater] = {"seeds": seed_set}
            for variant, other in sorted(supp[tag].items()):
                out["surface"][rater][variant] = {
                    score: paired(design, ref, other, GENERATORS, SCORES[score])
                    for score in ("E", "E_without_E1", "E1")}
                out["surface"][rater][variant]["pole_shift_E"] = pole_shift(ref, other, E_ITEMS)
        tag = f"s3-{rater}-zh"
        if tag in supp:
            other = supp[tag]["none"]
            seed_set = sorted({k[3] for k in other})
            design = Design(seed_set, scenarios)
            ref = restrict(table, set(seed_set))
            out["zh_rubric"][rater] = {
                "seeds": seed_set,
                "E": paired(design, ref, other, GENERATORS, E_ITEMS),
                "E_without_E1": paired(design, ref, other, GENERATORS, E_ITEMS[1:]),
                "agreement_en_rubric_vs_zh_rubric": agreement(ref, other, E_ITEMS),
                "pole_shift_E": pole_shift(ref, other, E_ITEMS)}

    out["public"] = {}
    for corpus in ("friendspersona", "cped"):
        frozen = [r for r in read_jsonl(root / "h-public" / f"scores-{corpus}.jsonl")]
        out["public"][f"mistral|{corpus}"] = public_block(root, corpus, frozen)
        for rater in ("llama", "gemma"):
            records = public_records(root, f"pub-{rater}-{corpus}")
            if records is not None:
                out["public"][f"{rater}|{corpus}"] = public_block(root, corpus, records)
                _, out["shards"][f"pub-{rater}-{corpus}"] = load_supp(root, f"pub-{rater}-{corpus}")

    target = project / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(project)}; supplementary shards read: {sorted(out['shards'])}")
    print("sanity:", {k: v for k, v in out["sanity"].items() if k != "note"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
