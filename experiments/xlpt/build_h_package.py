#!/usr/bin/env python3
"""Build the block H human-annotation package exactly as ``refine-logs/followup/H_sampling_spec.md`` (v2).

CPU only and score-blind: the draw reads scenario_id, seed, checkpoint, phase and the eligibility fields of the
generation shards, never a rater score.  The only non-deterministic input is the 32-byte blinding salt, which is
written to the key file alone.  Outputs are immutable: the build refuses to run if any output already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiments.xlpt.score_rater import rubric_items, transcript

SPEC = "refine-logs/followup/H_sampling_spec.md"
PROTOCOL = "refine-logs/followup/H_annotation_protocol.md"
FORM = "refine-logs/followup/H_scoring_form.md"
RUBRIC = "refine-logs/RUBRIC_BEHAVIOR.md"
RUBRIC_JSONL = "data/xlpt-support/rendered_items_and_scenarios/rubric_items_en.jsonl"

CHECKPOINTS = ("Llama-3.1-8B-Instruct", "aya-expanse-8b", "qwen3-8b")
CELLS = (("en", "high"), ("en", "low"), ("zh", "high"), ("zh", "low"))
SEEDS = tuple(s for s in range(20260811, 20260835) if s != 20260829)
N_SEEDS = 8
RATERS = ("R1", "R2")
OUTPUTS = ("H_PACKAGE_R1.md", "H_PACKAGE_R2.md", "H_PRACTICE.md", "H_BLINDING_KEY.json", "H_SAMPLE_MANIFEST.json")
CSV_HEADER = "rater_id,order_index,presentation_id,E1,E2,E3,E4,E5,E6,language_mismatch,cannot_rate,note"
FENCE = "~~~~"


class BuildStop(RuntimeError):
    pass


def h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(D: str, k: str) -> str:
    return f"{'zh-' if D == 'zh' else ''}b4-extraversion-{k}"


def shard_rel(D: str, k: str, c: str, s: int) -> str:
    return f"process/results/xlpt/{phase(D, k)}/{c}/seed-{s}.jsonl"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wording_check(project: Path) -> dict:
    items = rubric_items(project / RUBRIC, "extraversion")
    jsonl = {r["record_id"]: r for r in read_rows(project / RUBRIC_JSONL)}
    per_item = {}
    for iid, facet, prompt, a1, a4, a7 in items:
        row = jsonl.get(iid)
        text_ok = row is not None and row["text"] == f"{prompt}\n1 = {a1}\n4 = {a4}\n7 = {a7}"
        facet_ok = row is not None and row["facet"].casefold() == facet.casefold()
        per_item[iid] = {"text_identical": text_ok, "facet_identical_casefold": facet_ok}
    ok = [i[0] for i in items] == ["E1", "E2", "E3", "E4", "E5", "E6"] and all(
        v["text_identical"] and v["facet_identical_casefold"] for v in per_item.values())
    if not ok:
        raise BuildStop(f"E1-E6 wording differs between {RUBRIC} and {RUBRIC_JSONL}: {per_item}")
    return {"result": "identical", "compared": "item text and 1/4/7 anchors byte-identical; facet label case-insensitive",
            "items": per_item}


def load_candidates(project: Path) -> tuple[dict, dict]:
    """Return {(c, D, k, s): [record, ...]} and {shard_rel: sha256}; en shards are checked against the rater shards."""
    candidates, shard_hashes = {}, {}
    for c in CHECKPOINTS:
        for D, k in CELLS:
            for s in SEEDS:
                rel = shard_rel(D, k, c, s)
                path = project / rel
                if not path.exists():
                    raise BuildStop(f"missing source shard {rel}")
                digest = file_sha(path)
                if D == "en":
                    rater = project / f"process/results/xlpt/{phase(D, k)}-rater/{c}/seed-{s}.jsonl"
                    carried = {r.get("source_shard_sha256") for r in read_rows(rater)}
                    if carried != {digest}:
                        raise BuildStop(f"{rel} sha256 {digest} does not match its rater shard {sorted(carried)}")
                shard_hashes[rel] = digest
                rows = []
                for r in read_rows(path):
                    if r.get("eligible_for_analysis") is True and r.get("partition") == "QUALIFICATION":
                        if r.get("seed") != s:
                            raise BuildStop(f"{rel} carries a record of seed {r.get('seed')}")
                        rows.append({"c": c, "D": D, "k": k, "s": s, "scenario_id": r["scenario_id"],
                                     "shard": rel, "row": r})
                candidates[(c, D, k, s)] = rows
    return candidates, shard_hashes


def ranked(items, key):
    keyed = [(key(x), x) for x in items]
    digests = [d for d, _ in keyed]
    if len(set(digests)) != len(digests):
        raise BuildStop("hash tie in a ranking")
    return [x for _, x in sorted(keyed, key=lambda t: t[0])]


def draw(candidates: dict) -> tuple[list, dict, list]:
    selections, seed_log, skip_log = [], {}, []
    for c in CHECKPOINTS:
        order = ranked(SEEDS, lambda s: h(f"XLPT-H-v2-seed|{c}|{s}"))
        chosen, skipped = [], []
        for s in order:
            if len(chosen) == N_SEEDS:
                break
            empty = [f"{D},{k}" for D, k in CELLS if not candidates[(c, D, k, s)]]
            if empty:
                skipped.append({"seed": s, "empty_cells": empty})
            else:
                chosen.append(s)
        if len(chosen) < N_SEEDS:
            raise BuildStop(f"{c}: only {len(chosen)} qualifying seeds")
        seed_log[c] = {"seed_rank_order": order, "selected_seeds": chosen, "skipped_seeds": skipped}
        used: dict[str, str] = {}
        for s in chosen:
            for D, k in CELLS:
                pool = ranked(candidates[(c, D, k, s)],
                              lambda x: h(f"XLPT-H-v2|{c}|{D}|{k}|{s}|{x['scenario_id']}"))
                pick = None
                for x in pool:
                    if x["scenario_id"] in used:
                        skip_log.append({"checkpoint": c, "D": D, "k": k, "seed": s,
                                         "skipped_scenario_id": x["scenario_id"],
                                         "collided_with": used[x["scenario_id"]]})
                        continue
                    pick = x
                    break
                if pick is None:
                    raise BuildStop(f"{c} {D} {k} {s}: ranked candidates exhausted")
                used[pick["scenario_id"]] = f"{c}|{D}|{k}|{s}"
                selections.append(pick)
    return selections, seed_log, skip_log


def practice(candidates: dict, selections: list) -> tuple[list, dict]:
    used = {x["scenario_id"] for x in selections}
    picks, pools = [], {}
    for D in ("en", "zh"):
        # No TUNING partition exists in these phases (checked at load: every eligible record is QUALIFICATION),
        # so the spec's fallback pool applies.
        pools[D] = "QUALIFICATION candidates whose scenario_id is not used among the 96 (no TUNING records)"
        for k in ("high", "low"):
            pool = [x for (c, d, kk, s), rows in candidates.items() if d == D and kk == k
                    for x in rows if x["scenario_id"] not in used]
            if not pool:
                raise BuildStop(f"practice pool empty for {D},{k}")
            picks.append(ranked(pool, lambda x: h(
                f"XLPT-H-v2-practice|{D}|{x['c']}|{k}|{x['s']}|{x['scenario_id']}"))[0])
    return picks, pools


def no_tuning(project: Path) -> bool:
    for c in CHECKPOINTS:
        for D, k in CELLS:
            for s in SEEDS:
                for r in read_rows(project / shard_rel(D, k, c, s)):
                    if r.get("partition") == "TUNING":
                        return False
    return True


def pid(salt_hex: str, x: dict) -> str:
    return "H" + h(f"{salt_hex}|{x['c']}|{x['D']}|{x['k']}|{x['scenario_id']}|{x['s']}")[:10]


def render(presentation_id: str, x: dict) -> str:
    body = transcript(x["row"])
    if FENCE in body:
        raise BuildStop(f"transcript of {presentation_id} contains the fence {FENCE}")
    return f"{FENCE}text\n{body}\n{FENCE}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out_dir)
    out = out if out.is_absolute() else project / out
    existing = [n for n in OUTPUTS if (out / n).exists()]
    if existing:
        raise BuildStop(f"outputs are immutable and already exist: {existing}")

    wording = wording_check(project)
    candidates, shard_hashes = load_candidates(project)
    if not no_tuning(project):
        raise BuildStop("TUNING records present; the practice fallback would not apply")
    selections, seed_log, skip_log = draw(candidates)
    practice_picks, practice_pools = practice(candidates, selections)
    if len(selections) != 96:
        raise BuildStop(f"expected 96 selections, got {len(selections)}")

    salt_hex = os.urandom(32).hex()
    for x in selections + practice_picks:
        x["pid"] = pid(salt_hex, x)
    ids = [x["pid"] for x in selections + practice_picks]
    if len(set(ids)) != len(ids):
        raise BuildStop("presentation IDs are not unique")

    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.mkdir(parents=True, exist_ok=True)
    texts = {}
    for rater in RATERS:
        order = sorted(selections, key=lambda x: h(f"{salt_hex}|{rater}|{x['pid']}"))
        lines = [f"# Block H package, rater {rater}", "",
                 f"{len(order)} transcripts. Rate each with the E1-E6 items and anchors in the item and anchor sheet.",
                 "Each transcript is shown exactly as the dialogue ran: CUSTOMER and AGENT turns in order.", ""]
        for i, x in enumerate(order, 1):
            lines += [f"## {i}. {x['pid']}", "", render(x["pid"], x), ""]
        lines += ["## Scoring form (CSV, one row per transcript in this order)", "", "```", CSV_HEADER]
        lines += [f"{rater},{i},{x['pid']},,,,,,,,," for i, x in enumerate(order, 1)]
        lines += ["```", ""]
        texts[f"H_PACKAGE_{rater}.md"] = "\n".join(lines)
    plines = ["# Block H practice transcripts", "",
              "Practice only: these 4 transcripts are not part of the rated package and are not analysed.", ""]
    for i, x in enumerate(sorted(practice_picks, key=lambda x: x["pid"]), 1):
        plines += [f"## Practice {i}. {x['pid']}", "", render(x["pid"], x), ""]
    texts["H_PRACTICE.md"] = "\n".join(plines)

    def entry(x, role):
        return {"role": role, "checkpoint": x["c"], "D": x["D"], "k": x["k"], "seed": x["s"],
                "scenario_id": x["scenario_id"], "source_shard": x["shard"],
                "source_shard_sha256": shard_hashes[x["shard"]]}

    key = {"spec_version": "v2", "built_utc": built, "salt_hex": salt_hex,
           "presentation_ids": {x["pid"]: entry(x, "rated") for x in selections}
           | {x["pid"]: entry(x, "practice") for x in practice_picks}}
    manifest = {
        "spec_version": "v2", "built_utc": built,
        "builder": "experiments/xlpt/build_h_package.py",
        "builder_sha256": file_sha(Path(__file__).resolve()),
        "spec_sha256": file_sha(project / SPEC), "protocol_sha256": file_sha(project / PROTOCOL),
        "form_sha256": file_sha(project / FORM),
        "agreeableness_items_included": False,
        "rubric_behavior_sha256": file_sha(project / RUBRIC), "rubric_items_en_sha256": file_sha(project / RUBRIC_JSONL),
        "e1_e6_wording_check": wording,
        "source_shards": shard_hashes,
        "en_shards_checked_against_rater_source_shard_sha256": True,
        "seed_blocks": seed_log,
        "selections": [{k: v for k, v in entry(x, "rated").items() if k != "role"} for x in selections],
        "candidate_skips": skip_log,
        "practice_selections": [{k: v for k, v in entry(x, "practice").items() if k != "role"} for x in practice_picks],
        "practice_pool_by_D": practice_pools,
        "presentation_id_uniqueness": {"n_rated": len(selections), "n_practice": len(practice_picks),
                                       "unique": True},
        "outputs_sha256": {n: h(t) for n, t in texts.items()},
        "salt_included": False,
        "standing": "descriptive block; admits nothing; writes nothing to QUALIFICATION_GATE.results",
    }
    texts["H_BLINDING_KEY.json"] = json.dumps(key, indent=2, ensure_ascii=False) + "\n"
    manifest["outputs_sha256"]["H_BLINDING_KEY.json"] = h(texts["H_BLINDING_KEY.json"])
    texts["H_SAMPLE_MANIFEST.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    for name, text in texts.items():
        with open(out / name, "x", encoding="utf-8") as fh:
            fh.write(text)
    print(json.dumps({"out_dir": str(out), "selections": len(selections), "practice": len(practice_picks),
                      "candidate_skips": len(skip_log),
                      "seed_skips": {c: len(v["skipped_seeds"]) for c, v in seed_log.items()}}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildStop as exc:
        print(f"BUILD STOP: {exc}")
        raise SystemExit(2)
