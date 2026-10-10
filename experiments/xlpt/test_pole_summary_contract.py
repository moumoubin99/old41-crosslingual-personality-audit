#!/usr/bin/env python3
"""Contract test for the phase-aware pole rater summarizer.

``summarize_b0_rater.py`` refuses any analysis-eligible row, so the pole surface needs its
own reducer; the risk is that the new reducer silently disagrees with the tool that wrote
the shards, or that it accepts a B0 gate shard, a mislabeled pole, or a double-counted
generation shard. This test pins all of that on synthetic shards, so it runs on CPU
anywhere -- it reads the writer's phase registry with ``ast`` instead of importing
``run_rater_scoring``, which needs the model stack.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from summarize_pole_rater import (  # noqa: E402
    FROZEN_DIGEST_FIELDS,
    HIGH_PHASE,
    ITEM_IDS,
    ITEM_MISSING_DECODING,
    SCORER_DIGEST_FILES,
    LOW_PHASE,
    POLE_PHASES,
    RESCUE_SUFFIX,
    collect,
    main,
    summarize,
)

MIRRORED_FIELDS = (
    "setting",
    "score_dir",
    "prompt_config",
    "freeze_receipt",
    "eligible_for_gate",
    "eligible_for_analysis",
)


def writer_registry() -> dict:
    """Read SOURCE_PHASES out of run_rater_scoring.py without importing it.

    Entries may carry values that are not literals -- the pole phases name their generation
    worker through the module constant ``POLE_GENERATION_WORKER`` -- so each field is
    evaluated on its own and the non-literal ones are dropped. Every field this test
    actually compares (MIRRORED_FIELDS) is a literal, and a field that stops being one is
    caught below as a missing key rather than silently passing.
    """
    tree = ast.parse((HERE / "run_rater_scoring.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SOURCE_PHASES"
            for target in node.targets
        ):
            registry = {}
            for phase_key, phase_value in zip(node.value.keys, node.value.values):
                fields = {}
                for field_key, field_value in zip(phase_value.keys, phase_value.values):
                    try:
                        fields[ast.literal_eval(field_key)] = ast.literal_eval(field_value)
                    except ValueError:
                        continue
                registry[ast.literal_eval(phase_key)] = fields
            return registry
    raise AssertionError("SOURCE_PHASES not found in run_rater_scoring.py")


# 1. The mirrored registry must agree with the writer's, field by field.
source_phases = writer_registry()
for name, mirrored in POLE_PHASES.items():
    if name not in source_phases:
        raise AssertionError(f"{name} is not a phase run_rater_scoring.py can score")
    for field in MIRRORED_FIELDS:
        if mirrored[field] != source_phases[name][field]:
            raise AssertionError(
                f"{name}.{field}: summarizer mirror {mirrored[field]!r} disagrees with the "
                f"writer {source_phases[name][field]!r}"
            )
    if mirrored["eligible_for_gate"] is not False:
        raise AssertionError(f"{name}: a pole phase must never be gate-eligible")
    if mirrored["eligible_for_analysis"] is not True:
        raise AssertionError(f"{name}: a pole phase must be analysis-eligible")
for name, phase in source_phases.items():
    if phase["eligible_for_analysis"] and name not in POLE_PHASES:
        raise AssertionError(f"{name} is analysis-eligible but the summarizer ignores it")

DIGESTS = {field: f"{index + 1}" * 64 for index, field in enumerate(FROZEN_DIGEST_FIELDS)}
# Schema-3 rows carry the digests of the scorer files shipped next to the summarizer.
SCORER_DIGESTS = {
    field: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    for field, name in SCORER_DIGEST_FILES.items()
}
SCENARIOS = ("sc-1", "sc-2")
TRAITS = ("extraversion", "agreeableness")


def row(phase: str, checkpoint: str, seed: int, scenario: str, trait: str, items: list,
        source_sha: str) -> dict:
    return {
        "schema_version": 2,
        "source_phase": phase,
        "source_scenario_id": scenario,
        "checkpoint_id": checkpoint,
        "setting": POLE_PHASES[phase]["setting"],
        "seed": seed,
        "trait": trait,
        "instrument_language": "English",
        "scores": dict(zip(ITEM_IDS[trait], items)),
        "trait_score": sum(items) / 6.0,
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "source_shard_sha256": source_sha,
        "rater_calls": 6 * len(SCENARIOS) * len(TRAITS),
        "scoring_elapsed_seconds": 1.0,
        "rater_calls_per_second": float(6 * len(SCENARIOS) * len(TRAITS)),
        **DIGESTS,
    }


def build(root: Path, *, high_items=(6, 6, 6, 6, 6, 6), low_items=(2, 2, 2, 2, 2, 2),
          phases=(HIGH_PHASE, LOW_PHASE)) -> None:
    """Two poles x one checkpoint x two seeds x two scenarios x two traits."""
    for phase, items in ((phases[0], list(high_items)), (phases[1], list(low_items))):
        for seed in (20260811, 20260812):
            directory = root / POLE_PHASES[phase]["score_dir"] / "qwen3-8b"
            directory.mkdir(parents=True, exist_ok=True)
            # one generation shard per (pole, seed): distinct, and stable across runs
            source_sha = hashlib.sha256(f"{phase}|{seed}".encode()).hexdigest()
            rows = [
                row(phase, "qwen3-8b", seed, scenario, trait, items, source_sha)
                for scenario in SCENARIOS
                for trait in TRAITS
            ]
            (directory / f"seed-{seed}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )


def rewrite(path: Path, mutate) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mutate(rows)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def item_missing(namespace: str, drop=None):
    """Mutator: turn rows into strict_item_missing_v1 (schema 3) rows of one namespace.

    `drop` = (scenario, trait, item_id) nulls that item in the matching row, as the item-missing
    scorer does when the frozen parser rejects one rater output."""
    def mutate(rows):
        for r in rows:
            failures = []
            raw = {item_id: str(value) for item_id, value in r["scores"].items()}
            if drop and (r["source_scenario_id"], r["trait"]) == drop[:2]:
                r["scores"][drop[2]] = None
                raw[drop[2]] = "7."
                failures = [{"item_id": drop[2], "raw_decoded": " 7.\n", "parser_input": "7.",
                             "error": "rater must return exactly one integer 1..7, got '7.'",
                             "token_ids": [22, 13]}]
            r.update({
                "schema_version": 3,
                "parse_policy": "strict_item_missing_v1",
                "score_namespace": namespace,
                "rescue_run": namespace == "rescue",
                "rater_raw_by_item": raw,
                "parse_failures": failures,
                "items_parsed": 6 - len(failures),
                "rater_decoding": dict(ITEM_MISSING_DECODING),
                **SCORER_DIGESTS,
            })
            if failures:
                r["trait_score"] = None
    return mutate


def to_rescue(root: Path, phase: str, seed: int, *, keep_canonical=False) -> Path:
    """Move (or copy) one canonical shard into its {score_dir}-rescue directory."""
    source = root / POLE_PHASES[phase]["score_dir"] / "qwen3-8b" / f"seed-{seed}.jsonl"
    target = root / (POLE_PHASES[phase]["score_dir"] + RESCUE_SUFFIX) / "qwen3-8b" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if not keep_canonical:
        source.unlink()
    return target


def refuses(root: Path, needle: str, phases=(HIGH_PHASE, LOW_PHASE)) -> None:
    try:
        shards, frozen, unit = collect(root, "seed-*.jsonl", phases)
        summarize(shards, frozen, unit)
    except ValueError as exc:
        if needle not in str(exc):
            raise AssertionError(f"refused for the wrong reason: {exc}") from exc
        return
    raise AssertionError(f"accepted a shard it should have refused ({needle})")


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    build(root)

    # 2. A clean block summarises, and the paired contrast is the real difference.
    shards, frozen, unit = collect(root, "seed-*.jsonl")
    payload = summarize(shards, frozen, unit)
    if payload["shard_count"] != 4 or payload["rows_total"] != 16:
        raise AssertionError(f"unexpected shard/row count: {payload['shard_count']}, {payload['rows_total']}")
    if payload["gate_decision"] != "NOT_DECIDED_BY_THIS_TOOL" or payload["verdict"] != "DESCRIPTIVE_ONLY":
        raise AssertionError("the summary must decide nothing")
    if payload["eligible_for_gate"] is not False:
        raise AssertionError("the summary must not claim gate eligibility")
    seed_pair = payload["paired_by_seed"]["qwen3-8b|extraversion"]
    if seed_pair["n"] != 2 or abs(seed_pair["mean_delta"] - 4.0) > 1e-9:
        raise AssertionError(f"paired seed contrast is wrong: {seed_pair}")
    scenario_pair = payload["paired_by_scenario"]["qwen3-8b|extraversion"]
    if scenario_pair["n"] != 4 or abs(scenario_pair["mean_delta"] - 4.0) > 1e-9:
        raise AssertionError(f"paired scenario contrast is wrong: {scenario_pair}")
    if abs(payload["by_pole_cell"][f"{HIGH_PHASE}|qwen3-8b|extraversion"]["mean"] - 6.0) > 1e-9:
        raise AssertionError("pole cell descriptives are wrong")

with tempfile.TemporaryDirectory() as tmp:
    # 3. A gate-eligible row (a B0 shard dropped into a pole directory) is refused.
    root = Path(tmp)
    build(root)
    rewrite(
        root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl",
        lambda rows: rows[0].update({"eligible_for_gate": True, "eligible_for_analysis": False}),
    )
    refuses(root, "eligible_for_gate must be false")

with tempfile.TemporaryDirectory() as tmp:
    # 4. A low-pole row sitting in the high-pole directory is refused.
    root = Path(tmp)
    build(root)
    rewrite(
        root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl",
        lambda rows: rows[0].update({"source_phase": LOW_PHASE}),
    )
    refuses(root, "but the shard was read from")

with tempfile.TemporaryDirectory() as tmp:
    # 5. A setting that does not match the phase is refused.
    root = Path(tmp)
    build(root)
    rewrite(
        root / POLE_PHASES[LOW_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl",
        lambda rows: rows[0].update({"setting": "NEUTRAL"}),
    )
    refuses(root, "setting is 'NEUTRAL'")

with tempfile.TemporaryDirectory() as tmp:
    # 6. A trait_score that is not the equal-weight mean of its items is refused (rule G).
    root = Path(tmp)
    build(root)
    rewrite(
        root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl",
        lambda rows: rows[0].update({"trait_score": 7.0}),
    )
    refuses(root, "is not the equal-weight mean")

with tempfile.TemporaryDirectory() as tmp:
    # 7. Two score shards scored from one generation shard are refused.
    root = Path(tmp)
    build(root)
    shared = "a" * 64
    for seed in (20260811, 20260812):
        rewrite(
            root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / f"seed-{seed}.jsonl",
            lambda rows: [r.update({"source_shard_sha256": shared}) for r in rows],
        )
    refuses(root, "double-count one sample")

with tempfile.TemporaryDirectory() as tmp:
    # 8. Mixed freeze digests across shards are refused.
    root = Path(tmp)
    build(root)
    rewrite(
        root / POLE_PHASES[LOW_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260812.jsonl",
        lambda rows: [r.update({"prompt_config_sha256": "f" * 64}) for r in rows],
    )
    refuses(root, "differs from an earlier shard")

with tempfile.TemporaryDirectory() as tmp:
    # 9. An out-of-range rubric item is refused.
    root = Path(tmp)
    build(root)
    rewrite(
        root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl",
        lambda rows: rows[0]["scores"].update({"E1": 9}),
    )
    refuses(root, "outside [1, 7]")

# 10. The real B0 score shards, if present, must be refused by this reducer -- the two
#     surfaces stay separate in both directions.
b0 = PROJECT / "process/results/xlpt/b0-rater/qwen3-8b"
if b0.is_dir():
    real = sorted(b0.glob("seed-*.jsonl"))
    if real:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b"
            target.mkdir(parents=True)
            (target / real[0].name).write_text(real[0].read_text(encoding="utf-8"), encoding="utf-8")
            refuses(root, "eligible_for_gate must be false")

with tempfile.TemporaryDirectory() as tmp:
    # 12. strict_item_missing_v1: a null item makes its trait incomplete; descriptives skip it,
    #     and both paired contrasts use common-complete scenarios only. The low pole's sc-1
    #     extraversion is set to 1 so an all-scenario low mean (1.5) would give a seed delta of
    #     4.5; the common-complete seed delta is 6 - 2 = 4.
    root = Path(tmp)
    build(root)
    high = root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl"
    low = root / POLE_PHASES[LOW_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl"
    rewrite(high, item_missing("canonical", drop=("sc-1", "extraversion", "E3")))
    rewrite(low, lambda rows: [
        r.update({"scores": dict.fromkeys(ITEM_IDS["extraversion"], 1), "trait_score": 1.0})
        for r in rows if (r["source_scenario_id"], r["trait"]) == ("sc-1", "extraversion")
    ])
    shards, frozen, unit = collect(root, "seed-*.jsonl")
    payload = summarize(shards, frozen, unit)
    cell = payload["missingness"][f"{HIGH_PHASE}|qwen3-8b|extraversion"]
    if cell != {"units": 4, "complete": 3, "incomplete": 1}:
        raise AssertionError(f"missingness is wrong: {cell}")
    if payload["by_pole_cell"][f"{HIGH_PHASE}|qwen3-8b|extraversion"]["n"] != 3:
        raise AssertionError("an incomplete trait entered the descriptives")
    denominators = payload["pair_denominators"]["qwen3-8b|extraversion"]
    if denominators != {"matched_scenarios": 4, "common_complete": 3, "dropped_incomplete": 1}:
        raise AssertionError(f"pair denominators are wrong: {denominators}")
    if payload["paired_by_scenario"]["qwen3-8b|extraversion"]["n"] != 3:
        raise AssertionError("an incomplete scenario entered the matched-scenario contrast")
    seed_pair = payload["paired_by_seed"]["qwen3-8b|extraversion"]
    if seed_pair["n"] != 2 or abs(seed_pair["mean_delta"] - 4.0) > 1e-9:
        raise AssertionError(f"the seed contrast is not over common-complete scenarios: {seed_pair}")
    if (payload["item_parse_failures_total"], payload["incomplete_traits_total"]) != (1, 1):
        raise AssertionError("the failure totals are wrong")
    if payload["parse_policy"] != "strict_item_missing_v1" or payload["row_schema_versions"] != [2, 3]:
        raise AssertionError("the summary does not declare the item-missing parse policy")

with tempfile.TemporaryDirectory() as tmp:
    # 13. A schema-3 rescue shard in {score_dir}-rescue is accepted and listed as a rescue cell.
    root = Path(tmp)
    build(root)
    rescued = to_rescue(root, HIGH_PHASE, 20260812)
    rewrite(rescued, item_missing("rescue", drop=("sc-2", "agreeableness", "A6")))
    shards, frozen, unit = collect(root, "seed-*.jsonl")
    payload = summarize(shards, frozen, unit)
    if payload["shard_count"] != 4 or payload["rescue_cells"] != [f"{HIGH_PHASE}|qwen3-8b|seed-20260812"]:
        raise AssertionError(f"the rescue shard was not summarised as a rescue cell: {payload['rescue_cells']}")
    if payload["pair_denominators"]["qwen3-8b|agreeableness"]["dropped_incomplete"] != 1:
        raise AssertionError("the rescue shard's incomplete trait was not dropped from the pair")

with tempfile.TemporaryDirectory() as tmp:
    # 14. A cell with both a canonical and a rescue shard is refused (its first repeated unit
    #     trips the per-unit check before the per-cell one).
    root = Path(tmp)
    build(root)
    rewrite(to_rescue(root, LOW_PHASE, 20260811, keep_canonical=True), item_missing("rescue"))
    refuses(root, "a second score for qwen3-8b/")

with tempfile.TemporaryDirectory() as tmp:
    # 15. A rescue-directory shard whose rows claim the canonical namespace is refused, and so
    #     is a schema-2 shard in the rescue directory.
    root = Path(tmp)
    build(root)
    rewrite(to_rescue(root, HIGH_PHASE, 20260811), item_missing("canonical"))
    refuses(root, "do not match the rescue directory")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    build(root)
    to_rescue(root, HIGH_PHASE, 20260811)
    refuses(root, "a rescue shard must be schema_version 3")

with tempfile.TemporaryDirectory() as tmp:
    # 16. A schema-3 row whose parse_failures do not name the null item is refused.
    root = Path(tmp)
    build(root)
    high = root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl"
    rewrite(high, item_missing("canonical", drop=("sc-1", "extraversion", "E3")))
    rewrite(high, lambda rows: [r.update({"parse_failures": []}) for r in rows])
    refuses(root, "parse_failures do not name exactly the null items")

with tempfile.TemporaryDirectory() as tmp:
    # 17. The collision key is the physical cell: a rescue shard whose rows spell the
    #     checkpoint differently from the canonical shards of that directory is refused ...
    root = Path(tmp)
    build(root)
    rescued = to_rescue(root, LOW_PHASE, 20260812, keep_canonical=True)
    rewrite(rescued, item_missing("rescue"))
    rewrite(rescued, lambda rows: [r.update({"checkpoint_id": "Qwen3-8B"}) for r in rows])
    refuses(root, "but other shards under 'qwen3-8b'")
with tempfile.TemporaryDirectory() as tmp:
    #     ... and so is one filed under a case-variant directory with the same spelling.
    root = Path(tmp)
    build(root)
    rescued = to_rescue(root, LOW_PHASE, 20260812, keep_canonical=True)
    rewrite(rescued, item_missing("rescue"))
    variant = rescued.parent.with_name("Qwen3-8B")
    rescued.parent.rename(variant)
    rewrite(variant / rescued.name, lambda rows: [r.update({"checkpoint_id": "Qwen3-8B"}) for r in rows])
    refuses(root, "but other shards under 'Qwen3-8B'")

with tempfile.TemporaryDirectory() as tmp:
    # 18. Rows whose seed disagrees with the file name are refused (no seed-metadata variant can
    #     open a second cell beside the canonical one).
    root = Path(tmp)
    build(root)
    rescued = to_rescue(root, HIGH_PHASE, 20260811, keep_canonical=True)
    rewrite(rescued, item_missing("rescue"))
    rewrite(rescued, lambda rows: [r.update({"seed": 20260899}) for r in rows])
    refuses(root, "the file name must be seed-20260899.jsonl")

def schema3_refuses(mutate, needle: str) -> None:
    """One canonical high shard turned schema 3 (one null item), then mutated, is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build(root)
        high = root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl"
        rewrite(high, item_missing("canonical", drop=("sc-1", "extraversion", "E3")))
        rewrite(high, mutate)
        refuses(root, needle)


def first(update):
    def mutate(rows):
        update(rows[0])
    return mutate


def null_row(update):
    def mutate(rows):
        update(next(r for r in rows if r["trait_score"] is None))
    return mutate


# 20. The stored rater outputs are re-parsed with the frozen parser: a score that disagrees with
#     its output, an unparseable output kept as a score, a parseable output recorded as a
#     failure, and an incomplete failure record are all refused.
schema3_refuses(first(lambda r: r["rater_raw_by_item"].update({"E1": "3"})),
                "stores 6 but its output parses to 3")
schema3_refuses(first(lambda r: r["rater_raw_by_item"].update({"E1": "6 (agree)"})),
                "has score 6 but its stored output does not parse")
schema3_refuses(null_row(lambda r: r["rater_raw_by_item"].update({"E3": "5"})),
                "stores None but its output parses to 5")
schema3_refuses(null_row(lambda r: r["parse_failures"][0].update({"error": "edited"})),
                "is not the complete record of its stored output")
schema3_refuses(null_row(lambda r: r["parse_failures"][0].pop("token_ids")),
                "is not the complete record of its stored output")
schema3_refuses(first(lambda r: r.update({"extra_field": 1})), "schema-3 key set differs")
schema3_refuses(first(lambda r: r.pop("rater_raw_by_item")), "schema-3 key set differs")
schema3_refuses(first(lambda r: r.update({"scorer_sha256": "z" * 64})),
                "scorer_sha256 is not a sha256 digest")
schema3_refuses(first(lambda r: r.update({"frozen_scorer_sha256": "A" * 64})),
                "frozen_scorer_sha256 is not a sha256 digest")

# 21. Decoding drift is refused, including a value of the right number but the wrong type.
schema3_refuses(first(lambda r: r["rater_decoding"].update({"temperature": 0.7})),
                "not the registered decoding block")
schema3_refuses(first(lambda r: r["rater_decoding"].update({"top_k": 1.0})),
                "not the registered decoding block")

# 22. Scorer-version mixing is refused across shards, and a uniform scorer digest that is not
#     the scorer shipped here is refused by the CLI's freeze check.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    build(root)
    for seed in (20260811, 20260812):
        rewrite(root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / f"seed-{seed}.jsonl",
                item_missing("canonical"))
    rewrite(root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260812.jsonl",
            lambda rows: [r.update({"scorer_sha256": "a" * 64}) for r in rows])
    refuses(root, "scorer_sha256 differs from an earlier shard")
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    build(root)
    for seed in (20260811, 20260812):
        rewrite(root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / f"seed-{seed}.jsonl",
                item_missing("canonical"))
        rewrite(root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / f"seed-{seed}.jsonl",
                lambda rows: [r.update({"frozen_scorer_sha256": "b" * 64}) for r in rows])
    try:
        main(["--project", str(root), "--score-root", ".",
              "--output-json", "B4_POLE_RATER_SUMMARY.json", "--expect-shards", "4"])
    except ValueError as exc:
        if "is not the scorer the schema-3 shards were written by" not in str(exc):
            raise AssertionError(f"refused for the wrong reason: {exc}") from exc
    else:
        raise AssertionError("the CLI accepted shards written by another frozen scorer")


def shard_refuses(mutate, needle: str) -> None:
    """The canonical schema-2 high shard, mutated, is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build(root)
        rewrite(root / POLE_PHASES[HIGH_PHASE]["score_dir"] / "qwen3-8b" / "seed-20260811.jsonl",
                mutate)
        refuses(root, needle)


def every(update):
    def mutate(rows):
        for r in rows:
            update(r)
    return mutate


# 23. Rows carry the rubric's own item ids in order and a finite trait score, and every shard
#     carries valid, uniform call/timing provenance of six calls per row.
shard_refuses(first(lambda r: r.update({"scores": dict(zip(
    ("X1", "X2", "X3", "X4", "X5", "X6"), r["scores"].values()))})), "must be keyed")
shard_refuses(first(lambda r: r.update({"scores": dict(reversed(r["scores"].items()))})),
              "must be keyed")
shard_refuses(first(lambda r: r.update({"trait_score": float("nan")})), "not a finite number")
shard_refuses(every(lambda r: r.update({"rater_calls_per_second": 99.0})),
              "is not rater_calls / scoring_elapsed_seconds")
shard_refuses(every(lambda r: r.update({"rater_calls_per_second": 24})),
              "is not rater_calls / scoring_elapsed_seconds")
shard_refuses(every(lambda r: r.update({"rater_calls": True})), "invalid call/timing provenance")
shard_refuses(every(lambda r: r.update({"scoring_elapsed_seconds": float("nan")})),
              "invalid call/timing provenance")
shard_refuses(every(lambda r: r.update({"scoring_elapsed_seconds": 0.0})),
              "invalid call/timing provenance")
shard_refuses(first(lambda r: r.update({"scoring_elapsed_seconds": 2.0,
                                        "rater_calls_per_second": 12.0})),
              "nonuniform call/timing provenance")
shard_refuses(every(lambda r: r.update({"rater_calls": 12, "rater_calls_per_second": 12.0})),
              "6 calls for each of its 4 rows")

BC_PHASES = ("bc-pzhn-extraversion-high-dzh", "bc-pzhn-extraversion-low-dzh")
with tempfile.TemporaryDirectory() as tmp:
    # 19. BC is under strict_item_missing_v1 from job 1: schema-3 canonical shards summarise,
    #     a legacy schema-2 shard is refused, and so is any BC rescue directory.
    root = Path(tmp)
    build(root, phases=BC_PHASES)
    for phase in BC_PHASES:
        for seed in (20260811, 20260812):
            rewrite(root / POLE_PHASES[phase]["score_dir"] / "qwen3-8b" / f"seed-{seed}.jsonl",
                    item_missing("canonical"))
    shards, frozen, unit = collect(root, "seed-*.jsonl", BC_PHASES)
    if summarize(shards, frozen, unit)["shard_count"] != 4:
        raise AssertionError("schema-3 BC shards were not summarised")
    rewrite(to_rescue(root, BC_PHASES[0], 20260811), item_missing("rescue"))
    refuses(root, "has no rescue namespace", BC_PHASES)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    build(root, phases=BC_PHASES)
    refuses(root, "must be schema_version 3", BC_PHASES)

# 11. The CLI writes the summary atomically inside the project, and refuses an incomplete block.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    build(root)
    code = main([
        "--project", str(root),
        "--score-root", ".",
        "--output-json", "B4_POLE_RATER_SUMMARY.json",
        "--expect-shards", "4",
    ])
    if code != 0:
        raise AssertionError(f"the CLI returned {code} on a complete block")
    written = json.loads((root / "B4_POLE_RATER_SUMMARY.json").read_text(encoding="utf-8"))
    if written["shard_count"] != 4 or written["gate_decision"] != "NOT_DECIDED_BY_THIS_TOOL":
        raise AssertionError(f"the written summary is wrong: {written.get('shard_count')}")
    if written["freeze_files_checked"] is None:
        raise AssertionError("the summary must record whether the freeze files were checkable")
    try:
        main([
            "--project", str(root),
            "--score-root", ".",
            "--output-json", "B4_POLE_RATER_SUMMARY.json",
            "--expect-shards", "184",
        ])
    except ValueError as exc:
        if "refusing to summarise an incomplete block" not in str(exc):
            raise AssertionError(f"refused for the wrong reason: {exc}") from exc
    else:
        raise AssertionError("the CLI summarised an incomplete block")
    if not list(root.glob("*.tmp-*")) == []:
        raise AssertionError("the atomic write left a temp file behind")

print("PASS: the pole summarizer mirrors the writer's registry, reduces the pole surface only, "
      "refuses gate shards, mislabeled poles, rule-G violations and double-counted shards, and "
      "reduces strict_item_missing_v1 and rescue shards over common-complete units")
