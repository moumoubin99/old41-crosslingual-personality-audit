"""Describe the zh-b4 Extraversion pole block (P = en, D = zh) on script-neutral, rater-free indicators.

This is a deterministic, CPU-only reducer over the generation shards themselves. It loads
no model, takes no accelerator, and writes into no registered gate artifact.

What it computes
----------------
``zh-b4-extraversion-high`` and ``zh-b4-extraversion-low`` were generated on the **same**
50 competence-screen scenarios, the same 4 checkpoints and the same 23 qualification seeds,
differing only in the pole sentence of the system text. The comparison is therefore paired
at the registered resampling unit, the seed cluster. Per checkpoint and indicator the tool

1. averages the indicator over the 50 scenarios within each seed cluster and pole,
2. takes the within-cluster difference ``high - low`` (23 cluster values),
3. reports the mean difference with a percentile bootstrap interval resampled over clusters,
   plus the sign count.

The indicators are chosen to be meaningful in Chinese without a lexicon:

* ``chars_per_turn``      -- non-whitespace characters per agent turn (length);
* ``exclaim_turn_rate``   -- share of agent turns containing "!" or "！";
* ``question_turn_rate``  -- share of agent turns containing "?" or "？";
* ``chinese_turn_rate``   -- share of agent turns passing the registered G1 rule
  (Han >= 0.5 of Han + Latin letters), i.e. the G1 language rule applied to the pole block;
* ``no_han_turn_rate``    -- share of agent turns with zero Han characters.

What it deliberately does NOT do
--------------------------------
* It decides no gate and writes no ``QUALIFICATION_GATE.json`` / ``ELIGIBLE_PAIRS``.
* It is not a transfer estimate: no D = en pole generation is compared here, and the
  indicators are not the rubric. It is a manipulation check on the generator under D = zh.
* It applies no multiplicity correction across checkpoints or indicators.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

PHASES = {"high": "zh-b4-extraversion-high", "low": "zh-b4-extraversion-low"}
SETTINGS = {"high": "EXTRAVERSION_HIGH_DZH", "low": "EXTRAVERSION_LOW_DZH"}
INDICATORS = (
    "chars_per_turn",
    "exclaim_turn_rate",
    "question_turn_rate",
    "chinese_turn_rate",
    "no_han_turn_rate",
)
HAN_RATIO_RULE = 0.5
AGENT_TURNS = 3
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260928
INTERVAL_MASS = 0.95


def is_han(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0xF900 <= code <= 0xFAFF)


def is_latin_letter(ch: str) -> bool:
    return ch.isalpha() and not is_han(ch) and ord(ch) < 0x0250


def turn_indicators(text: str) -> dict:
    han = sum(1 for ch in text if is_han(ch))
    latin = sum(1 for ch in text if is_latin_letter(ch))
    script = han + latin
    return {
        "chars": sum(1 for ch in text if not ch.isspace()),
        "exclaim": ("!" in text) or ("！" in text),
        "question": ("?" in text) or ("？" in text),
        "chinese": script > 0 and han / script >= HAN_RATIO_RULE,
        "no_han": han == 0,
    }


def row_indicators(row: dict) -> dict:
    turns = row["agent_turns"]
    if len(turns) != AGENT_TURNS:
        raise ValueError(f"{row['scenario_id']} seed {row['seed']} has {len(turns)} agent turns")
    per = [turn_indicators(t) for t in turns]
    n = len(per)
    return {
        "chars_per_turn": sum(p["chars"] for p in per) / n,
        "exclaim_turn_rate": sum(p["exclaim"] for p in per) / n,
        "question_turn_rate": sum(p["question"] for p in per) / n,
        "chinese_turn_rate": sum(p["chinese"] for p in per) / n,
        "no_han_turn_rate": sum(p["no_han"] for p in per) / n,
    }


def load_pole(root: Path, pole: str) -> dict:
    """Return {checkpoint: {seed: {scenario: indicators}}} for one pole."""
    directory = root / PHASES[pole]
    if not directory.is_dir():
        raise ValueError(f"phase directory does not exist: {directory}")
    out: dict = {}
    for ckpt_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        per_seed: dict = {}
        for shard in sorted(ckpt_dir.glob("seed-*.jsonl")):
            by_scenario: dict = {}
            seeds = set()
            for line in shard.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("setting") != SETTINGS[pole]:
                    raise ValueError(f"{shard} carries setting {row.get('setting')}")
                if row.get("partition") != "QUALIFICATION" or row.get("eligible_for_gate") is not False:
                    raise ValueError(f"{shard} carries a row outside the descriptive qualification scope")
                if row.get("dialogue_language") != "Chinese" or row.get("persona_language") != "English":
                    raise ValueError(f"{shard} is not a P = en, D = zh row")
                scenario = row["scenario_id"]
                if scenario in by_scenario:
                    raise ValueError(f"{shard} repeats scenario {scenario}")
                seeds.add(int(row["seed"]))
                by_scenario[scenario] = row_indicators(row)
            if len(seeds) != 1:
                raise ValueError(f"{shard} mixes seeds {sorted(seeds)}")
            seed = seeds.pop()
            if seed in per_seed:
                raise ValueError(f"seed {seed} appears twice under {ckpt_dir.name}")
            per_seed[seed] = by_scenario
        out[ckpt_dir.name] = per_seed
    return out


def bootstrap_interval(values: list, rng: random.Random) -> list:
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(BOOTSTRAP_DRAWS))
    lo = means[int((1 - INTERVAL_MASS) / 2 * BOOTSTRAP_DRAWS)]
    hi = means[int((1 + INTERVAL_MASS) / 2 * BOOTSTRAP_DRAWS) - 1]
    return [lo, hi]


def contrast(high: dict, low: dict) -> dict:
    report = {}
    for ckpt in sorted(high):
        if ckpt not in low:
            raise ValueError(f"checkpoint {ckpt} has no low pole")
        seeds = sorted(high[ckpt])
        if seeds != sorted(low[ckpt]):
            raise ValueError(f"{ckpt}: pole seed blocks differ")
        for seed in seeds:
            if sorted(high[ckpt][seed]) != sorted(low[ckpt][seed]):
                raise ValueError(f"{ckpt} seed {seed}: pole scenario sets differ")
        rng = random.Random(f"{BOOTSTRAP_SEED}:{ckpt}")
        rows = {}
        for name in INDICATORS:
            def cluster_mean(block, seed):
                vals = [v[name] for v in block[seed].values()]
                return sum(vals) / len(vals)
            hi_means = [cluster_mean(high[ckpt], s) for s in seeds]
            lo_means = [cluster_mean(low[ckpt], s) for s in seeds]
            diffs = [h - l for h, l in zip(hi_means, lo_means)]
            rows[name] = {
                "high_mean": statistics.fmean(hi_means),
                "low_mean": statistics.fmean(lo_means),
                "mean_difference": statistics.fmean(diffs),
                "interval_95_cluster_bootstrap": bootstrap_interval(diffs, rng),
                "clusters_positive": sum(d > 0 for d in diffs),
                "clusters_negative": sum(d < 0 for d in diffs),
                "clusters_zero": sum(d == 0 for d in diffs),
            }
        report[ckpt] = {"seed_clusters": len(seeds), "scenarios_per_cluster": len(high[ckpt][seeds[0]]),
                        "indicators": rows}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    high = load_pole(args.results_root, "high")
    low = load_pole(args.results_root, "low")
    payload = {
        "readout": "zh-b4 pole contrast on script-neutral rater-free indicators",
        "surface": "P = en, D = zh; NLLB-200-3.3B single-stratum rendering of the qualification scenarios",
        "status": "exploratory",
        "verdict": "DESCRIPTIVE_ONLY",
        "gate_decision": "NOT_DECIDED_BY_THIS_TOOL",
        "eligible_for_gate": False,
        "not_a_transfer_estimate": "no D = en pole generation is compared; the language factor is constant within this readout",
        "multiplicity": "none applied, none claimed",
        "han_ratio_rule": HAN_RATIO_RULE,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "seed cluster",
                      "interval": "percentile", "mass": INTERVAL_MASS},
        "checkpoints": contrast(high, low),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
