#!/usr/bin/env python3
"""Contract test for block H (public data): unit builder, rendering, scorer and reducer.

Pins spec revision 2 (refine-logs/followup/H_public_spec.md): deterministic units with the
expected counts (711 / 287 identities / 569), AGENT lines that are exactly the target's own
turns, unroled stage directions, the main study's request bytes, per-item parse-error capture,
context exclusion without truncation, an immutable score shard, and the reducer's AUC,
within-identity AUC and degenerate-draw handling on hand-made fixtures.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from . import reduce_h_public as reducer
    from . import run_rater_scoring
    from . import score_public_personality as sp
    from .score_rater import transcript
except ImportError:  # direct-script contract tests
    import reduce_h_public as reducer
    import run_rater_scoring
    import score_public_personality as sp
    from score_rater import transcript

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / sp.DATA_DIR


def check(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------- builder determinism and counts

friends, friends_counts = sp.build_friends(DATA)
cped, cped_counts = sp.build_cped(DATA)
check(sp.units_bytes(friends) == sp.units_bytes(sp.build_friends(DATA)[0]), "Friends build is not deterministic")
check(sp.units_bytes(cped) == sp.units_bytes(sp.build_cped(DATA)[0]), "CPED build is not deterministic")
sp.check_counts("friendspersona", friends_counts)
sp.check_counts("cped", cped_counts)
check(cped_counts["components"] == 32, "CPED franchise components changed")
check(len({u["identity"] for u in cped}) == 287, "CPED identity count changed")
check(max(Counter(u["identity"] for u in cped).values()) <= sp.CPED_K, "CPED identity exceeds K dialogues")
scenes = Counter(u["scene"] for u in friends)
check(len(scenes) == 484 and sum(n > 1 for n in scenes.values()) == 152, "Friends scene structure changed")
for corpus, units in (("friendspersona", friends), ("cped", cped)):
    on_disk = PROJECT / sp.OUT_DIR / f"units-{corpus}.jsonl"
    if on_disk.exists():
        check(on_disk.read_bytes() == sp.units_bytes(units), f"{corpus} unit file is not the rebuild")

# ---------------------------------------------------------------- AGENT lines are exactly the target's

with (DATA / sp.INPUT_FILES["friendspersona"][0]).open(encoding="utf-8", newline="") as handle:
    raw_rows = {f"fp-{int(r['']):04d}": r for r in csv.DictReader(handle)}
alias_merged = 0
for unit in friends:
    raw = raw_rows[unit["unit_id"]]
    own = [" ".join(html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).split())
           for m in re.finditer(r"<b>([^<]*?)</b>:(.*?)(?=<br><br>|$)", raw["text"], re.S)
           if " ".join(html.unescape(m.group(1)).split()) == raw["character"]]
    agent = [text for role, text in unit["lines"] if role == "AGENT"]
    check(agent == own, f"{unit['unit_id']}: AGENT lines are not exactly the target's turns")
    check(unit["identity"] == sp.FRIENDS_ALIASES.get(raw["character"], raw["character"]),
          f"{unit['unit_id']}: identity is not the whitelisted alias")
    alias_merged += unit["identity"] != unit["character"]
check(sp.FRIENDS_ALIASES.get("Phoebe Sr.", "Phoebe Sr.") == "Phoebe Sr.", "non-main characters must not merge")

cped_rows = defaultdict(list)
for relative in sp.INPUT_FILES["cped"]:
    with (DATA / relative).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cped_rows[row["Dialogue_ID"]].append(row)
for unit in cped:
    rows = sorted(cped_rows[unit["dialogue_id"]], key=lambda r: r["Utterance_ID"])
    rows = [r for r in rows if r["Speaker"] != sp.CPED_EXCLUDED_SPEAKER]
    check([r["Speaker"] == unit["character"] for r in rows] == [role == "AGENT" for role, _ in unit["lines"]],
          f"{unit['unit_id']}: AGENT lines are not exactly the target's utterances")
    check(sum(role == "AGENT" for role, _ in unit["lines"]) >= sp.CPED_MIN_UTTERANCES,
          f"{unit['unit_id']}: target has fewer than three utterances")
    check(unit["identity"].split("|", 1)[1] == unit["character"], "CPED identity/speaker mismatch")
    check(all(role != "direction" for role, _ in unit["lines"]), "CPED has no stage directions")

# ---------------------------------------------------------------- stage directions carry no role

directions = 0
for unit in friends:
    full = sp.render(unit["lines"]).split("\n")
    stripped = sp.render(unit["lines"], strip_directions=True).split("\n")
    for line in full:
        check(line.startswith(("AGENT: ", "CUSTOMER: ", sp.STAGE_PREFIX)), "unexpected rendered line")
    directions += sum(line.startswith(sp.STAGE_PREFIX) for line in full)
    check(not any(line.startswith(sp.STAGE_PREFIX) for line in stripped), "stripped rendering kept a direction")
    check(len(full) - len(stripped) == sum(role == "direction" for role, _ in unit["lines"]),
          "stripped rendering dropped a turn")
check(directions == 409, "Friends stage-direction count changed")

# ---------------------------------------------------------------- request bytes equal the main rater's

check(sp.build_item_request is run_rater_scoring.build_item_request, "request builder is not the main rater's")
check(sp.validate_context_window is run_rater_scoring.validate_context_window, "context guard is not the main rater's")
main_row = {"customer_turns": ["Hi, my order is late.", "Thanks."], "agent_turns": ["Let me check.", "Done."]}
lines = [["CUSTOMER", "Hi, my order is late."], ["AGENT", "Let me check."], ["CUSTOMER", "Thanks."], ["AGENT", "Done."]]
check(sp.render(lines) == transcript(main_row), "rendering differs from the main study's transcript form")

# ---------------------------------------------------------------- scorer end to end with a fake engine

OVERFLOW_MARKER = "Strip joint!"


class FakeTokenizer:
    def apply_chat_template(self, messages, chat_template, tokenize, add_generation_prompt):
        check(tokenize and add_generation_prompt and "strftime_now" not in chat_template, "bad template call")
        content = messages[0]["content"]
        seed = int(hashlib.sha256(content.encode("utf-8")).hexdigest()[:8], 16)
        length = 5000 if OVERFLOW_MARKER in content else 64
        return [seed] + [1] * (length - 1)


class FakeLLM:
    def __init__(self, **kwargs):
        check(kwargs["max_model_len"] == 4096 and kwargs["enable_prefix_caching"] is False, "engine changed")

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, params, use_tqdm):
        seed = prompts[0]["prompt_token_ids"][0]
        text = "N/A" if seed % 97 == 0 else f" {seed % 7 + 1} "
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]


def fake_params(**kwargs):
    check(kwargs == run_rater_scoring.rater_decoding_contract(), "decoding contract changed")
    return kwargs


ticks = iter(range(10**7))
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    shutil.copytree(DATA, root / sp.DATA_DIR)
    (root / "refine-logs").mkdir()
    shutil.copy(PROJECT / "refine-logs/RUBRIC_BEHAVIOR.md", root / "refine-logs/RUBRIC_BEHAVIOR.md")
    (root / "refine-logs/followup").mkdir()
    shutil.copy(PROJECT / "refine-logs/followup/H_public_spec.md", root / "refine-logs/followup/H_public_spec.md")
    rater = root / "rater"
    rater.mkdir()
    shutil.copy(PROJECT / "process/consumption/deployment/mistral-rater-config/chat_template.json", rater)
    # The real frozen surface and runtime manifest, so the reducer can check them against current values.
    for rel in (sp.FROZEN_CONFIG, sp.FROZEN_RECEIPT, reducer.RUNTIME_MANIFEST,
                "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PROJECT / rel, root / rel)
    manifest_sha = hashlib.sha256((root / reducer.RUNTIME_MANIFEST).read_bytes()).hexdigest()
    sp.build(root)
    provenance = json.loads((root / sp.PROVENANCE).read_text())
    check(provenance["builder_sha256"] == hashlib.sha256(Path(sp.__file__).read_bytes()).hexdigest(),
          "provenance must bind the current scorer")
    args = SimpleNamespace(project=str(root), corpus="friendspersona", rater_path=str(rater),
                           runtime_manifest=reducer.RUNTIME_MANIFEST, gpu_memory_utilization=0.88)
    kwargs = dict(llm_cls=FakeLLM, sampling_params_cls=fake_params,
                  frozen_validator=sp.validate_rater_frozen_inputs,
                  runtime_validator=lambda *a, **k: manifest_sha,
                  project_binding_validator=lambda *a, **k: None,
                  clock=lambda: float(next(ticks)))
    check(sp.run(args, **kwargs) == 0, "fake Friends scoring failed")
    records = [json.loads(x) for x in (root / sp.OUT_DIR / "scores-friendspersona.jsonl").read_text().splitlines()]
    check(len(records) == 711 * 2, "Friends needs a full and a no-directions record per unit")
    excluded = [r for r in records if r["status"] == "excluded_context"]
    check({r["unit_id"] for r in excluded} == {"fp-0000"} and len(excluded) == 2,
          "context overflow must exclude exactly the overflowing unit, never truncate")
    scored = [r for r in records if r["status"] == "scored"]
    calls = sum(len(r["scores"]) for r in scored)
    check(calls == 710 * 18 and records[0]["rater_calls"] == calls, "call count differs from 18 per unit")
    errors = sum(len(r["parse_errors"]) for r in scored)
    check(errors > 0, "fixture should exercise parse errors")
    for r in scored:
        for trait, value in r["trait_scores"].items():
            prefix = "E" if trait == "extraversion" else "A"
            items = [v for k, v in r["scores"].items() if k.startswith(prefix)]
            check((value is None) == any(v is None for v in items), "trait mean must need all six items")
        check(all(r["rater_raw_by_item"][i] == "N/A" for i in r["parse_errors"]), "raw output not kept")
        check(r["eligible_for_gate"] is False, "public H must never be gate-eligible")
    try:
        sp.run(args, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("an existing score shard was overwritten")
    summary = reducer.reduce_corpus("friendspersona", records, n_boot=40)
    check(summary["n_excluded_context"] == 1, "reducer lost the exclusion count")
    check(summary["parse_errors"]["total"] == errors, "reducer lost parse errors")
    check(len(summary["per_item"]) == 24 and "E_to_EXT_no_directions" in summary, "reducer output incomplete")
    # Asymmetric overflow: a unit (with its own identity and scene) excluded in 'full' but scored
    # without directions must still reach the no-directions check instead of a KeyError.
    asym = copy.deepcopy(records)
    target = next(r["unit_id"] for r in asym if r["status"] == "scored" and r["rendering"] == "full"
                  and r["trait_scores"].get("extraversion") is not None and r["labels"]["EXT"] in ("high", "low"))
    for r in asym:
        if r["unit_id"] == target:
            r.update({"identity": "only-no-directions", "cluster": "only-no-directions-scene"})
            if r["rendering"] == "full":
                r["status"] = "excluded_context"
    asym_summary = reducer.reduce_corpus("friendspersona", asym, n_boot=40)
    check(asym_summary["n_excluded_context"] == 2
          and asym_summary["E_to_EXT_no_directions"]["n_units"] == summary["E_to_EXT_no_directions"]["n_units"]
          and asym_summary["E_to_EXT_all_parsed"]["n_units"] == summary["E_to_EXT_all_parsed"]["n_units"] - 1,
          "an asymmetric context exclusion must drop the unit from 'full' only")

    args.corpus = "cped"
    check(sp.run(args, **kwargs) == 0, "fake CPED scoring failed")
    cped_records = [json.loads(x) for x in (root / sp.OUT_DIR / "scores-cped.jsonl").read_text().splitlines()]
    check(len(cped_records) == 569 and all(len(r["scores"]) == 12 for r in cped_records), "CPED needs 12 items per unit")
    cped_summary = reducer.reduce_corpus("cped", cped_records, n_boot=40)
    check(cped_summary["trait_specificity"]["n_common_complete_units"] <= 566, "CPED unknown AGR must drop out")
    check(sum(v for k, v in cped_summary["missingness"].items() if k.endswith("unknown_AGR_label")) == 3,
          "unknown AGR labels must be reported")
    check(summary["max_prompt_tokens"] == 5000 and summary["max_prompt_tokens_scored"] == 64,
          "the token maximum must span every request, overflowing ones included")
    check(all(r["max_prompt_tokens"] == 5000 for r in excluded), "overflow unit lost its true token maximum")

    # Shard validation: the fake shards pass; every kind of post-write tampering fails closed.
    check(reducer.main(["--project", str(root), "--validate-only"]) == 0, "valid shards rejected")
    shards = {}
    for corpus in sp.CORPORA:
        receipt = sp.receipt_file(root, corpus)
        check(receipt.is_file(), f"{corpus}: no receipt written with the shard")
        shards[corpus] = {"shard_sha256": sp._sha256(root / sp.OUT_DIR / f"scores-{corpus}.jsonl"),
                          "receipt_sha256": sp._sha256(receipt)}
    ledger = root / "ledger.json"
    ledger.write_text(json.dumps({"shards": shards}))
    try:
        reducer.main(["--project", str(root)])
    except ValueError:
        pass
    else:
        raise AssertionError("reduction ran without the completion ledger")

    # Coordinated tamper: raw output, score and mean rewritten consistently. Field checks pass, but
    # the shard no longer matches its creation receipt, and a rewritten receipt misses the ledger.
    shard_path = root / sp.OUT_DIR / "scores-friendspersona.jsonl"
    original_shard = shard_path.read_bytes()
    original_receipt = sp.receipt_file(root, "friendspersona").read_bytes()
    coordinated = json.loads(json.dumps(records))
    victim = next(r for r in coordinated if r["status"] == "scored" and not r["parse_errors"]
                  and r["rendering"] == "full")
    victim["rater_raw_by_item"]["E1"] = str(victim["scores"]["E1"] % 7 + 1)
    victim["scores"]["E1"] = victim["scores"]["E1"] % 7 + 1
    victim["trait_scores"]["extraversion"] = float(sp.trait_score(
        [victim["scores"][f"E{i}"] for i in range(1, 7)]).item())
    reducer.validate_score_shard(root, "friendspersona", coordinated)  # field checks alone cannot see it
    shard_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in coordinated))
    for rewrite_receipt in (False, True):
        if rewrite_receipt:
            receipt = json.loads(original_receipt)
            receipt["shard_sha256"] = sp._sha256(shard_path)
            sp.receipt_file(root, "friendspersona").write_text(json.dumps(receipt))
        try:
            reducer.main(["--project", str(root), "--ledger", str(ledger)])
        except ValueError:
            continue
        raise AssertionError(f"coordinated tamper accepted (receipt rewritten={rewrite_receipt})")
    shard_path.write_bytes(original_shard)
    sp.receipt_file(root, "friendspersona").write_bytes(original_receipt)
    check(reducer.main(["--project", str(root), "--validate-only"]) == 0, "restored shards rejected")
    shared = reducer.validate_score_shard(root, "friendspersona", records)
    check(shared["code_sha256"] == sp.code_hashes() and len(shared["code_sha256"]) == len(sp.CODE_FILES),
          "code hashes not bound")
    first = next(i for i, r in enumerate(records) if r["status"] == "scored" and not r["parse_errors"])

    def tampered(edit):
        copy = json.loads(json.dumps(records))
        edit(copy)
        return copy

    def swap_score(rs):
        rs[first]["scores"]["E1"] = rs[first]["scores"]["E1"] % 7 + 1

    def flip_label(rs):
        rs[first]["labels"]["EXT"] = "low" if rs[first]["labels"]["EXT"] == "high" else "high"

    def mean_edit(rs):
        rs[first]["trait_scores"]["extraversion"] = 7.0

    def code_edit(rs):
        for r in rs:
            r["code_sha256"] = {**r["code_sha256"], "experiments/xlpt/reduce_h_public.py": "0" * 64}

    def all_rows(field, value):
        def edit(rs):
            for r in rs:
                r[field] = value
        return edit

    def fabricated_exclusion(rs):
        # Turn a scored row into a context exclusion and adjust the call count consistently.
        dropped = len(rs[first]["scores"])
        rs[first].update({"status": "excluded_context", "scores": {}, "rater_raw_by_item": {},
                          "parse_errors": [], "trait_scores": {}})
        for r in rs:
            r["rater_calls"] -= dropped
            r["rater_calls_per_second"] = sp.measured_rate(r["rater_calls"], r["scoring_elapsed_seconds"])

    def extra_key(rs):
        rs[first]["note"] = "x"

    edits = {"missing": lambda rs: rs.pop(), "duplicate": lambda rs: rs.append(rs[first]),
             "score": swap_score, "label": flip_label, "mean": mean_edit, "code": code_edit,
             "decoding": all_rows("rater_decoding", {**sp.RATER_DECODING_RECORD, "temperature": 0.7}),
             "prompt_config": all_rows("prompt_config_sha256", "2" * 64),
             "freeze_receipt": all_rows("freeze_receipt_sha256", "3" * 64),
             "runtime": all_rows("rater_runtime_manifest_sha256", "4" * 64),
             "not_analysis_eligible": all_rows("eligible_for_analysis", False),
             "rate": all_rows("rater_calls_per_second", 123.0),
             "fabricated_exclusion": fabricated_exclusion, "extra_key": extra_key}
    for name, edit in edits.items():
        try:
            reducer.validate_score_shard(root, "friendspersona", tampered(edit))
        except ValueError:
            continue
        raise AssertionError(f"a tampered shard ({name}) was accepted")
    (root / sp.OUT_DIR / "H_PUBLIC_SUMMARY.json").write_text("{}")
    try:
        reducer.main(["--project", str(root), "--ledger", str(ledger)])
    except ValueError as exc:
        check("existing summary" in str(exc), f"summary refusal came from elsewhere: {exc}")
    else:
        raise AssertionError("an existing summary was overwritten")

    stale = dict(provenance, builder_sha256="5" * 64)
    (root / sp.PROVENANCE).write_text(json.dumps(stale))
    try:
        sp.load_verified_units(root, "friendspersona")
    except ValueError:
        pass
    else:
        raise AssertionError("provenance built by a different scorer was accepted")
    sp.atomic_json(root / sp.PROVENANCE, provenance)

    (root / sp.OUT_DIR / "units-cped.jsonl").write_bytes(b"{}\n")
    try:
        sp.load_verified_units(root, "cped")
    except ValueError:
        pass
    else:
        raise AssertionError("a tampered unit file was accepted")

# ---------------------------------------------------------------- reducer fixtures

auc = reducer.weighted_auc([3, 2, 1, 2], ["high", "high", "low", "low"], np.ones((1, 4)))
check(abs(auc[0] - 0.875) < 1e-12, "AUC must count ties as 1/2")
auc = reducer.weighted_auc([3, 2, 1, 2], ["high", "high", "low", "low"], np.array([[2.0, 0, 1, 1]]))
check(abs(auc[0] - 1.0) < 1e-12, "weighted AUC wrong")
check(np.isnan(reducer.weighted_auc([3, 1], ["high", "low"], np.array([[0.0, 1.0]]))[0]),
      "a draw with no high unit must be invalid")

# identity X: high 5 vs low 1 -> 1.0 (1 pair); identity Y: highs 1,1 vs low 2 -> 0.0 (2 pairs);
# identity Z has only high units and is uninformative even though it would favour high overall.
scores = [5, 1, 1, 1, 2, 9]
labels = ["high", "low", "high", "high", "low", "high"]
ids = ["X", "X", "Y", "Y", "Y", "Z"]
ones = np.ones((1, 6))
pair, macro = reducer.within_identity(scores, labels, ids, ones, {i: np.ones(1) for i in "XYZ"})
check(abs(pair[0] - 1 / 3) < 1e-12 and abs(macro[0] - 0.5) < 1e-12, "within-identity AUC wrong")
check(reducer.informative(labels, ids) == (2, 3), "informative identity/pair count wrong")
pair, macro = reducer.within_identity(scores, labels, ids, ones, {"X": np.array([2.0]), "Y": np.array([0.0]),
                                                                 "Z": np.ones(1)})
check(pair[0] == 1.0 and macro[0] == 1.0, "identity multiplicity must weight its pairs once per copy")
pair, macro = reducer.within_identity(scores, labels, ids, ones, {"X": np.zeros(1), "Y": np.zeros(1),
                                                                 "Z": np.ones(1)})
check(np.isnan(pair[0]) and np.isnan(macro[0]), "zero informative pairs must be an invalid draw")
out = reducer.interval(0.6, np.array([0.5, np.nan, 0.7, 0.6]))
check(out["valid_draws"] == 3 and out["total_draws"] == 4 and out["unstable"], "degenerate draws not counted")
out = reducer.interval(0.6, np.linspace(0.4, 0.8, 100))
check(not out["unstable"] and abs(out["ci95"][0] - 0.41) < 1e-9, "percentile interval wrong")

d1 = reducer.FriendsDesign(["a", "b"], ["s1", "s2"], n_boot=5, seed=1)
d2 = reducer.FriendsDesign(["a", "b"], ["s1", "s2"], n_boot=5, seed=1)
check(np.array_equal(d1.mid, d2.mid) and np.all(d1.mid[0] == 1), "bootstrap draws must be seeded; row 0 = point")

print(f"PASS: public H units (711/287/569), exact AGENT turns ({alias_merged} alias-merged Friends units), "
      "unroled directions, main-rater requests, fake-engine scoring and reducer fixtures")
