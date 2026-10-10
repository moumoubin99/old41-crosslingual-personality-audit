"""Reduce the completed Extraversion persona-pole rater score shards to one summary.

This is the phase-aware companion to ``summarize_b0_rater.py``. That reducer refuses any
row that claims analysis eligibility, because the B0 surface is a gate surface; the pole
surface is the exact opposite -- ``eligible_for_gate: false`` and
``eligible_for_analysis: true`` -- so it needs its own reducer rather than a loosened one.
Both are deterministic, CPU-only and read-only: they load no model, take no accelerator,
write into no shard and decide nothing.

What it enforces before it reports anything:

* every row is ``schema_version`` 2, ``eligible_for_gate`` false and
  ``eligible_for_analysis`` true -- a gate-eligible row (a B0 shard) is refused;
* the ``source_phase`` of every row matches the pole directory the shard was read from,
  and its ``setting`` matches that phase's persona setting, so a high shard cannot be
  counted as a low one;
* every row has exactly six integer rubric items in [1, 7] and a ``trait_score`` equal to
  their equal-weight mean (rule G), recomputed here rather than trusted;
* the prompt-config, freeze-receipt and rater-runtime manifest digests are identical across
  every shard, and -- when the frozen files are present -- equal to the v2 freeze on disk,
  so the summary describes one frozen apparatus and not a mixture;
* each shard holds exactly one seed for one checkpoint, and no generation shard
  (``source_shard_sha256``) is scored into two score shards.

What it reports: descriptives per (pole, checkpoint, trait), and the paired high-minus-low
contrast per (checkpoint, trait) at two units -- matched scenario and, as the primary unit,
the seed mean, since scenarios inside one seed are not independent draws.

What it deliberately does NOT do: it decides no qualification gate, writes no
``QUALIFICATION_GATE.json``, freezes no ``ELIGIBLE_PAIRS``, applies no threshold and
declares no manipulation check passed. ``gate_decision`` is always
``NOT_DECIDED_BY_THIS_TOOL`` and ``verdict`` is always ``DESCRIPTIVE_ONLY``: reading the
contrast as evidence for or against the pole hypothesis belongs to the analysis stage,
with its own review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

try:
    from .rater_runtime import measured_rate
    from .run_rater_scoring import parse_item_score
except ImportError:  # direct-script use
    from rater_runtime import measured_rate
    from run_rater_scoring import parse_item_score

SCHEMA_VERSION = 2
# Schema 3 rows come from run_rater_item_missing.py (parse policy strict_item_missing_v1,
# FOLLOWUP_PLAN amendment 2026-09-30): an item the frozen parser rejects is null with a
# parse_failures record, and a trait score needs all six items. Canonical schema-3 shards sit in
# the phase's own score directory; rescue shards (a crashed frozen-scorer cell scored once more)
# sit in "{score_dir}-rescue". A cell may have one shard across both.
ITEM_MISSING_SCHEMA_VERSION = 3
SHARD_NAME = re.compile(r"seed-(\d+)\.jsonl")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
SCHEMA_VERSIONS = (SCHEMA_VERSION, ITEM_MISSING_SCHEMA_VERSION)
PARSE_POLICY = "strict_item_missing_v1"
RESCUE_SUFFIX = "-rescue"
# The exact row written by run_rater_item_missing.py, its registered decoding block and the
# keys of one parse-failure record. Each schema-3 row is re-parsed with the frozen parser, and
# the two scorer digests must equal the wrapper and frozen scorer on disk (see collect).
ITEM_MISSING_KEYS = frozenset({
    "schema_version", "parse_policy", "score_namespace", "rescue_run", "source_phase",
    "source_scenario_id", "checkpoint_id", "setting", "seed", "trait", "instrument_language",
    "scores", "items_parsed", "trait_score", "rater_raw_by_item", "parse_failures",
    "rater_decoding", "prompt_config_sha256", "freeze_receipt_sha256",
    "rater_runtime_manifest_sha256", "source_shard_sha256", "scorer_sha256",
    "frozen_scorer_sha256", "eligible_for_gate", "eligible_for_analysis",
    "scoring_elapsed_seconds", "rater_calls", "rater_calls_per_second",
})
ITEM_MISSING_DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                         "max_new_tokens_per_item": 8, "seed": 20260911,
                         "batch_size": 1, "cross_request_batching": False}
FAILURE_KEYS = frozenset({"item_id", "raw_decoded", "parser_input", "error", "token_ids"})
SCORER_DIGEST_FILES = {
    "scorer_sha256": "run_rater_item_missing.py",
    "frozen_scorer_sha256": "run_rater_scoring.py",
}
TRAITS = ("extraversion", "agreeableness")
ITEMS_PER_TRAIT = 6
# The rubric's item ids, in the order both scorers write them.
ITEM_IDS = {"extraversion": tuple(f"E{i}" for i in range(1, 7)),
            "agreeableness": tuple(f"A{i}" for i in range(1, 7))}
SCORE_MIN = 1
SCORE_MAX = 7

# Mirrors experiments/xlpt/run_rater_scoring.py SOURCE_PHASES for the two pole phases.
# test_pole_summary_contract.py parses that registry out of the writer's source and fails
# on any drift, so this copy cannot quietly disagree with the tool that wrote the shards.
POLE_PHASES = {
    "b4-extraversion-high": {
        "setting": "EXTRAVERSION_HIGH",
        "score_dir": "b4-extraversion-high-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v2.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v2.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    "b4-extraversion-low": {
        "setting": "EXTRAVERSION_LOW",
        "score_dir": "b4-extraversion-low-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v2.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v2.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    # Chinese dialogue surface (execution config v3). D = zh with the English persona
    # (zh-b4-*) and D = zh with the NLLB-rendered persona (zh-b5-*-pzh). One summary run
    # reduces exactly one high/low pair (see POLE_PAIRS), so shards frozen under v2 and v3 are
    # never mixed into one apparatus digest.
    "zh-b4-extraversion-high": {
        "setting": "EXTRAVERSION_HIGH_DZH",
        "score_dir": "zh-b4-extraversion-high-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    "zh-b4-extraversion-low": {
        "setting": "EXTRAVERSION_LOW_DZH",
        "score_dir": "zh-b4-extraversion-low-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    "zh-b5-extraversion-high-pzh": {
        "setting": "EXTRAVERSION_HIGH_PZH",
        "score_dir": "zh-b5-extraversion-high-pzh-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    "zh-b5-extraversion-low-pzh": {
        "setting": "EXTRAVERSION_LOW_PZH",
        "score_dir": "zh-b5-extraversion-low-pzh-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    # Block BC (execution config v4): the reviewed Chinese persona rendering crossed with both
    # dialogue languages; scored by run_rater_item_missing --namespace canonical.
    "bc-pzhn-extraversion-high-dzh": {
        "setting": "EXTRAVERSION_HIGH_PZHN_DZH",
        "score_dir": "bc-pzhn-extraversion-high-dzh-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    "bc-pzhn-extraversion-low-dzh": {
        "setting": "EXTRAVERSION_LOW_PZHN_DZH",
        "score_dir": "bc-pzhn-extraversion-low-dzh-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    "bc-pzhn-extraversion-high-den": {
        "setting": "EXTRAVERSION_HIGH_PZHN_DEN",
        "score_dir": "bc-pzhn-extraversion-high-den-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
    "bc-pzhn-extraversion-low-den": {
        "setting": "EXTRAVERSION_LOW_PZHN_DEN",
        "score_dir": "bc-pzhn-extraversion-low-den-rater",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
    },
}
# Phases scored under parse policy strict_item_missing_v1 from their first job: only canonical
# schema-3 shards are accepted there (no legacy schema, no rescue directory).
ITEM_MISSING_ONLY_PHASES = frozenset(phase for phase in POLE_PHASES if phase.startswith("bc-"))
HIGH_PHASE = "b4-extraversion-high"
LOW_PHASE = "b4-extraversion-low"
# (high phase, low phase, default output file, human label) per cell of the P x D design
# that the pole surface covers. "en" is the original P=en,D=en pair and keeps its output name.
POLE_PAIRS = {
    "en": (HIGH_PHASE, LOW_PHASE, "process/results/xlpt/B4_POLE_RATER_SUMMARY.json", "P=en,D=en"),
    "zh-den": (
        "zh-b4-extraversion-high", "zh-b4-extraversion-low",
        "process/results/xlpt/ZH_DEN_POLE_RATER_SUMMARY.json", "P=en,D=zh",
    ),
    "zh-pzh": (
        "zh-b5-extraversion-high-pzh", "zh-b5-extraversion-low-pzh",
        "process/results/xlpt/ZH_PZH_POLE_RATER_SUMMARY.json", "P=zh,D=zh",
    ),
    "bc-dzh": (
        "bc-pzhn-extraversion-high-dzh", "bc-pzhn-extraversion-low-dzh",
        "process/results/xlpt/BC_DZH_POLE_RATER_SUMMARY.json", "P=zh(v4 reviewed),D=zh",
    ),
    "bc-den": (
        "bc-pzhn-extraversion-high-den", "bc-pzhn-extraversion-low-den",
        "process/results/xlpt/BC_DEN_POLE_RATER_SUMMARY.json", "P=zh(v4 reviewed),D=en",
    ),
}
FROZEN_DIGEST_FIELDS = (
    "prompt_config_sha256",
    "freeze_receipt_sha256",
    "rater_runtime_manifest_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {number} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} line {number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} contains no score rows")
    return rows


def check_item_missing_evidence(row: dict, where: str) -> None:
    """Check a schema-3 row against the raw evidence it stores.

    The key set and decoding block are exactly the registered ones, and the frozen parser is
    re-applied: a stored score must be what its stored output parses to, and a null item must be
    an output the parser rejects, recorded in full. The caller has already checked that
    parse_failures name exactly the null items, in order.
    """
    if set(row) != ITEM_MISSING_KEYS:
        extra = sorted(set(row) - ITEM_MISSING_KEYS)
        absent = sorted(ITEM_MISSING_KEYS - set(row))
        raise ValueError(f"{where}: schema-3 key set differs (extra {extra}, missing {absent})")
    decoding = row["rater_decoding"]
    if not isinstance(decoding, dict) or decoding != ITEM_MISSING_DECODING or any(
        type(decoding[k]) is not type(v) for k, v in ITEM_MISSING_DECODING.items()
    ):
        raise ValueError(f"{where}: rater_decoding is not the registered decoding block")
    scores = row["scores"]
    raw = row["rater_raw_by_item"]
    if not isinstance(raw, dict) or list(raw) != list(scores) or not all(
        isinstance(text, str) for text in raw.values()
    ):
        raise ValueError(f"{where}: rater_raw_by_item must hold one string per scored item")
    by_item = {f["item_id"]: f for f in row["parse_failures"]}
    for item_id, value in scores.items():
        text = raw[item_id]
        try:
            parsed = parse_item_score(text)
        except ValueError as exc:
            if value is not None:
                raise ValueError(f"{where}: item {item_id} has score {value} but its stored "
                                 "output does not parse") from exc
            failure = by_item[item_id]
            token_ids = failure.get("token_ids")
            if (set(failure) != FAILURE_KEYS or failure["parser_input"] != text
                    or not isinstance(failure["raw_decoded"], str)
                    or failure["raw_decoded"].strip() != text
                    or failure["error"] != str(exc)
                    or not isinstance(token_ids, list)
                    or any(isinstance(t, bool) or not isinstance(t, int) for t in token_ids)):
                raise ValueError(f"{where}: parse failure of item {item_id} is not the "
                                 "complete record of its stored output")
            continue
        if value is None or parsed != value:
            raise ValueError(f"{where}: item {item_id} stores {value!r} but its output "
                             f"parses to {parsed!r}")
    for field in SCORER_DIGEST_FILES:
        value = row[field]
        if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
            raise ValueError(f"{where}: {field} is not a sha256 digest")


def validate_row(row: dict, *, phase: str, shard: str, index: int,
                 namespace: str = "canonical") -> None:
    where = f"{shard} row {index}"
    contract = POLE_PHASES[phase]
    schema = row.get("schema_version")
    if schema not in SCHEMA_VERSIONS:
        raise ValueError(f"{where}: schema_version must be one of {SCHEMA_VERSIONS}")
    if namespace == "rescue" and schema != ITEM_MISSING_SCHEMA_VERSION:
        raise ValueError(f"{where}: a rescue shard must be schema_version {ITEM_MISSING_SCHEMA_VERSION}")
    if phase in ITEM_MISSING_ONLY_PHASES:
        if schema != ITEM_MISSING_SCHEMA_VERSION:
            raise ValueError(f"{where}: phase {phase} is scored under {PARSE_POLICY} from job 1 and "
                             f"must be schema_version {ITEM_MISSING_SCHEMA_VERSION}")
        if namespace != "canonical":
            raise ValueError(f"{where}: phase {phase} has no rescue namespace")
    if row.get("eligible_for_gate") is not False:
        raise ValueError(
            f"{where}: eligible_for_gate must be false on the pole surface; a gate-eligible "
            "row belongs to the B0 qualification surface and is summarised by "
            "summarize_b0_rater.py"
        )
    if row.get("eligible_for_analysis") is not True:
        raise ValueError(f"{where}: eligible_for_analysis must be true on the pole surface")
    if row.get("source_phase") != phase:
        raise ValueError(
            f"{where}: source_phase is {row.get('source_phase')!r} but the shard was read "
            f"from the {contract['score_dir']} directory of phase {phase}"
        )
    if row.get("setting") != contract["setting"]:
        raise ValueError(
            f"{where}: setting is {row.get('setting')!r}, not {contract['setting']!r}"
        )
    if row.get("instrument_language") != "English":
        raise ValueError(f"{where}: instrument_language must be English on this surface")
    if row.get("trait") not in TRAITS:
        raise ValueError(f"{where}: trait must be one of {TRAITS}")
    if not isinstance(row.get("checkpoint_id"), str) or not row["checkpoint_id"]:
        raise ValueError(f"{where}: checkpoint_id is missing")
    if not isinstance(row.get("source_scenario_id"), (str, int)):
        raise ValueError(f"{where}: source_scenario_id is missing")
    scores = row.get("scores")
    if not isinstance(scores, dict) or len(scores) != ITEMS_PER_TRAIT:
        raise ValueError(f"{where}: scores must hold exactly {ITEMS_PER_TRAIT} rubric items")
    if tuple(scores) != ITEM_IDS[row["trait"]]:
        raise ValueError(f"{where}: scores must be keyed {ITEM_IDS[row['trait']]} in order")
    # Every row of a shard carries the shard's call count and timing; collect() checks that
    # they are uniform and that the count is six calls per row.
    calls = row.get("rater_calls")
    elapsed = row.get("scoring_elapsed_seconds")
    try:
        rate_ok = row.get("rater_calls_per_second") == measured_rate(calls, elapsed)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid call/timing provenance ({exc})") from None
    if not rate_ok or type(row.get("rater_calls_per_second")) is not float:
        raise ValueError(f"{where}: rater_calls_per_second is not rater_calls / "
                         "scoring_elapsed_seconds")
    missing = []
    for item_id, value in scores.items():
        if value is None and schema == ITEM_MISSING_SCHEMA_VERSION:
            missing.append(item_id)
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: rubric item {item_id} is not an integer")
        if not SCORE_MIN <= value <= SCORE_MAX:
            raise ValueError(f"{where}: rubric item {item_id} is outside [1, 7]")
    reported = row.get("trait_score")
    if schema == ITEM_MISSING_SCHEMA_VERSION:
        if row.get("parse_policy") != PARSE_POLICY:
            raise ValueError(f"{where}: parse_policy must be {PARSE_POLICY}")
        if row.get("score_namespace") != namespace or row.get("rescue_run") is not (namespace == "rescue"):
            raise ValueError(
                f"{where}: score_namespace/rescue_run do not match the {namespace} directory "
                "the shard was read from"
            )
        failures = row.get("parse_failures")
        if not isinstance(failures, list) or [
            f.get("item_id") if isinstance(f, dict) else None for f in failures
        ] != missing:
            raise ValueError(f"{where}: parse_failures do not name exactly the null items")
        if row.get("items_parsed") != ITEMS_PER_TRAIT - len(missing):
            raise ValueError(f"{where}: items_parsed does not count the parsed items")
        if missing and reported is not None:
            raise ValueError(f"{where}: an incomplete trait must have a null trait_score")
        check_item_missing_evidence(row, where)
    if not missing:
        # Rule G: only a trait with all six items has a score, the equal-weight mean.
        recomputed = sum(scores.values()) / float(ITEMS_PER_TRAIT)
        if (isinstance(reported, bool) or not isinstance(reported, (int, float))
                or not math.isfinite(float(reported))):
            raise ValueError(f"{where}: trait_score is missing or not a finite number")
        if abs(float(reported) - recomputed) > 1e-9:
            raise ValueError(
                f"{where}: trait_score {reported!r} is not the equal-weight mean {recomputed!r}"
            )
    for field in FROZEN_DIGEST_FIELDS + ("source_shard_sha256",):
        value = row.get(field)
        if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
            raise ValueError(f"{where}: {field} is not a sha256 digest")


def _describe(values: list) -> dict:
    floats = [float(value) for value in values]
    return {
        "n": len(floats),
        "mean": round(statistics.fmean(floats), 6),
        "sd": round(statistics.stdev(floats), 6) if len(floats) > 1 else None,
        "min": min(floats),
        "max": max(floats),
    }


def _paired(deltas: list) -> dict:
    """Describe a list of paired high-minus-low differences.

    The interval is a normal approximation (mean +/- 1.96 * sd / sqrt(n)); with 23 seeds it
    is indicative, not a t-interval, and it decides nothing on its own.
    """
    if not deltas:
        return {"n": 0}
    floats = [float(value) for value in deltas]
    n = len(floats)
    mean = statistics.fmean(floats)
    sd = statistics.stdev(floats) if n > 1 else None
    out = {
        "n": n,
        "mean_delta": round(mean, 6),
        "sd_delta": round(sd, 6) if sd is not None else None,
        "min_delta": round(min(floats), 6),
        "max_delta": round(max(floats), 6),
        "share_positive": round(sum(1 for value in floats if value > 0) / n, 6),
        "share_zero": round(sum(1 for value in floats if value == 0) / n, 6),
    }
    if sd is not None and sd > 0:
        half = 1.96 * sd / math.sqrt(n)
        out["ci95_normal_approx"] = [round(mean - half, 6), round(mean + half, 6)]
        out["cohens_dz"] = round(mean / sd, 6)
    else:
        out["ci95_normal_approx"] = None
        out["cohens_dz"] = None
    return out


def collect(score_root: Path, pattern: str, phases=(HIGH_PHASE, LOW_PHASE)) -> tuple:
    """Load and validate every pole score shard of the selected phases under score_root."""
    shards = []
    frozen = {}
    seen_source = {}
    seen_cells = set()
    dir_checkpoint = {}
    # unit keys: (phase, checkpoint, trait, seed, scenario) -> trait_score, or None when the
    # trait is incomplete under parse policy strict_item_missing_v1
    unit = {}

    for phase, namespace in sorted(
        (phase, namespace) for phase in phases for namespace in ("canonical", "rescue")
    ):
        if phase not in POLE_PHASES:
            raise ValueError(f"{phase} is not a pole phase this tool can summarise")
        score_dir = POLE_PHASES[phase]["score_dir"]
        if namespace == "rescue":
            score_dir += RESCUE_SUFFIX
        phase_root = score_root / score_dir
        if not phase_root.is_dir():
            continue
        for checkpoint_dir in sorted(p for p in phase_root.iterdir() if p.is_dir()):
            for path in sorted(checkpoint_dir.glob(pattern)):
                rows = read_jsonl(path)
                seeds = set()
                checkpoints = set()
                for index, row in enumerate(rows, start=1):
                    validate_row(row, phase=phase, shard=path.name, index=index,
                                 namespace=namespace)
                    digest_fields = FROZEN_DIGEST_FIELDS
                    if row["schema_version"] == ITEM_MISSING_SCHEMA_VERSION:
                        digest_fields += tuple(SCORER_DIGEST_FILES)
                    for field in digest_fields:
                        seen = frozen.setdefault(field, row[field])
                        if seen != row[field]:
                            raise ValueError(
                                f"{path.name} row {index}: {field} differs from an earlier "
                                "shard; the summary would mix two frozen apparatus states"
                            )
                    seeds.add(row["seed"])
                    checkpoints.add(row["checkpoint_id"])
                    key = (
                        phase,
                        row["checkpoint_id"],
                        row["trait"],
                        row["seed"],
                        row["source_scenario_id"],
                    )
                    if key in unit:
                        raise ValueError(
                            f"{path.name} row {index}: a second score for "
                            f"{key[1]}/{key[2]}/seed-{key[3]}/{key[4]} in phase {phase}"
                        )
                    score = row["trait_score"]
                    unit[key] = None if score is None else float(score)

                if len(seeds) != 1:
                    raise ValueError(
                        f"{path.name} mixes seeds {sorted(seeds)}; one shard is one seed"
                    )
                if len(checkpoints) != 1:
                    raise ValueError(
                        f"{path.name} mixes checkpoints {sorted(checkpoints)}"
                    )
                seed = seeds.pop()
                checkpoint = checkpoints.pop()
                # The directory name follows the model repository's own spelling, which is
                # not always the spelling the runtime reports as checkpoint_id: the qwen arm
                # has always written 'Qwen3-8B' rows into 'qwen3-8b/', in the pole shards and
                # in the b0 shards alike.  Case is therefore not evidence of a mis-filed
                # shard; a different name still is, which is what this guard is for.
                if checkpoint.casefold() != checkpoint_dir.name.casefold():
                    raise ValueError(
                        f"{path.name}: rows say checkpoint {checkpoint!r} but the shard sits "
                        f"under {checkpoint_dir.name!r}"
                    )
                # The physical cell is (phase, checkpoint directory, file-name seed); row
                # metadata must agree with it, and one directory carries one checkpoint_id in
                # every shard and namespace, so no spelling variant can open a second cell.
                named = SHARD_NAME.fullmatch(path.name)
                if named is None or int(named.group(1)) != seed:
                    raise ValueError(f"{path.name}: rows say seed {seed}; the file name must be "
                                     f"seed-{seed}.jsonl")
                directory = checkpoint_dir.name.casefold()
                if dir_checkpoint.setdefault(directory, checkpoint) != checkpoint:
                    raise ValueError(
                        f"{path.name}: rows say checkpoint {checkpoint!r} but other shards under "
                        f"{checkpoint_dir.name!r} say {dir_checkpoint[directory]!r}"
                    )
                cell = (phase, directory, seed)
                if cell in seen_cells:
                    raise ValueError(
                        f"duplicate score shard for {cell} (a cell has one shard across the "
                        "canonical and rescue directories)"
                    )
                seen_cells.add(cell)

                source_sha = rows[0]["source_shard_sha256"]
                if any(row["source_shard_sha256"] != source_sha for row in rows):
                    raise ValueError(f"{path.name} mixes source generation shards")
                if source_sha in seen_source:
                    raise ValueError(
                        f"{cell} and {seen_source[source_sha]} were scored from the same "
                        f"generation shard {source_sha[:12]}...; that would double-count one "
                        "sample"
                    )
                seen_source[source_sha] = cell

                schemas = {row["schema_version"] for row in rows}
                if len(schemas) != 1:
                    raise ValueError(f"{path.name} mixes schema versions {sorted(schemas)}")
                timing = {(row["rater_calls"], row["scoring_elapsed_seconds"],
                           row["rater_calls_per_second"]) for row in rows}
                if len(timing) != 1:
                    raise ValueError(f"{path.name} has nonuniform call/timing provenance")
                if rows[0]["rater_calls"] != ITEMS_PER_TRAIT * len(rows):
                    raise ValueError(
                        f"{path.name}: rater_calls is {rows[0]['rater_calls']}, not "
                        f"{ITEMS_PER_TRAIT} calls for each of its {len(rows)} rows"
                    )
                shards.append({
                    "phase": phase,
                    "checkpoint_id": checkpoint,
                    "seed": seed,
                    "namespace": namespace,
                    "schema_version": schemas.pop(),
                    "item_parse_failures": sum(len(row.get("parse_failures") or []) for row in rows),
                    "incomplete_traits": sum(row["trait_score"] is None for row in rows),
                    "shard": str(path.relative_to(score_root)),
                    "rows": len(rows),
                    "rater_calls": rows[0].get("rater_calls"),
                    "scoring_elapsed_seconds": rows[0].get("scoring_elapsed_seconds"),
                    "rater_calls_per_second": rows[0].get("rater_calls_per_second"),
                    "source_shard_sha256": source_sha,
                    "shard_sha256": sha256_file(path),
                })

    return shards, frozen, unit


def self_digest(path: Path) -> str:
    """Recompute a frozen asset's self_sha256 the way the freeze convention defines it.

    Every frozen JSON in this project carries a digest of its own canonical body with the
    self_sha256 field removed (freeze_assets.py), and the scorer writes *that* value into the
    score rows (run_rater_scoring.py: config["self_sha256"], receipt["self_sha256"]) -- not the
    digest of the file's raw bytes.  The value is recomputed here rather than read, exactly as
    run_generation.py does, so a file whose stored self_sha256 has been edited to match a
    tampered body is still caught.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name} is not a JSON object; it cannot carry a freeze digest")
    claimed = obj.pop("self_sha256", None)
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if claimed is not None and claimed != recomputed:
        raise ValueError(
            f"{path.name} claims self_sha256 {claimed[:12]}... but its body hashes to "
            f"{recomputed[:12]}...; the frozen file has been edited since it was frozen"
        )
    return recomputed


def verify_freeze(project: Path, frozen: dict, high_phase: str = HIGH_PHASE) -> dict:
    """Compare the digests carried by the rows against the frozen files on disk."""
    checks = {}
    pairs = (
        ("prompt_config_sha256", POLE_PHASES[high_phase]["prompt_config"]),
        ("freeze_receipt_sha256", POLE_PHASES[high_phase]["freeze_receipt"]),
    )
    for field, relative in pairs:
        path = project / relative
        if not path.is_file():
            checks[relative] = "file_absent_here"
            continue
        on_disk = self_digest(path)
        if field not in frozen:
            checks[relative] = "no_rows_to_compare"
        elif on_disk == frozen[field]:
            checks[relative] = "matches_shards"
        else:
            raise ValueError(
                f"{relative} on disk ({on_disk[:12]}...) is not the freeze the shards were "
                f"scored against ({frozen[field][:12]}...)"
            )
    # Schema-3 rows name the scorer that wrote them; it must be the scorer shipped here.
    for field, name in SCORER_DIGEST_FILES.items():
        if field not in frozen:
            continue
        path = Path(__file__).resolve().with_name(name)
        on_disk = sha256_file(path)
        if on_disk != frozen[field]:
            raise ValueError(
                f"{name} here ({on_disk[:12]}...) is not the scorer the schema-3 shards were "
                f"written by ({frozen[field][:12]}...)"
            )
        checks[name] = "matches_shards"
    return checks


def summarize(shards: list, frozen: dict, unit: dict, *, high_phase: str = HIGH_PHASE,
              low_phase: str = LOW_PHASE, pair_label: str = "P=en,D=en") -> dict:
    if not shards:
        raise ValueError("no pole score shards found; nothing to summarise")

    # Descriptives use the complete units of each pole; an incomplete trait (null score under
    # strict_item_missing_v1) is counted in `missingness` and never imputed.
    by_pole_cell = {}
    by_pole_trait = {}
    missingness = {}
    for (phase, checkpoint, trait, seed, scenario), score in unit.items():
        cell = missingness.setdefault(f"{phase}|{checkpoint}|{trait}",
                                      {"units": 0, "complete": 0, "incomplete": 0})
        cell["units"] += 1
        if score is None:
            cell["incomplete"] += 1
            continue
        cell["complete"] += 1
        by_pole_cell.setdefault((phase, checkpoint, trait), []).append(score)
        by_pole_trait.setdefault((phase, trait), []).append(score)

    # Paired contrasts use common-complete units: a scenario enters only when both its high and
    # its low trait scores exist, and a seed mean is taken over those same scenarios on both
    # sides. With no incomplete trait this is exactly the all-scenario seed-mean contrast.
    scenario_pairs = {}
    common = {}
    pair_denominators = {}
    for (phase, checkpoint, trait, seed, scenario), score in unit.items():
        if phase != high_phase:
            continue
        key = (low_phase, checkpoint, trait, seed, scenario)
        if key not in unit:
            continue
        counts = pair_denominators.setdefault(f"{checkpoint}|{trait}", {
            "matched_scenarios": 0, "common_complete": 0, "dropped_incomplete": 0,
        })
        counts["matched_scenarios"] += 1
        low = unit[key]
        if score is None or low is None:
            counts["dropped_incomplete"] += 1
            continue
        counts["common_complete"] += 1
        scenario_pairs.setdefault((checkpoint, trait), []).append(score - low)
        common.setdefault((checkpoint, trait, seed), []).append((score, low))

    seed_pairs = {}
    for (checkpoint, trait, seed), pairs in sorted(common.items()):
        seed_pairs.setdefault((checkpoint, trait), []).append(
            statistics.fmean(high for high, _ in pairs) - statistics.fmean(low for _, low in pairs)
        )
    failures_total = sum(entry.get("item_parse_failures", 0) for entry in shards)
    incomplete_total = sum(entry.get("incomplete_traits", 0) for entry in shards)
    rescue_cells = sorted(
        f"{entry['phase']}|{entry['checkpoint_id']}|seed-{entry['seed']}"
        for entry in shards if entry.get("namespace") == "rescue"
    )
    schema_versions = sorted({entry.get("schema_version", SCHEMA_VERSION) for entry in shards})
    if ITEM_MISSING_SCHEMA_VERSION in schema_versions:
        parse_failure_note = (
            f"Shards of schema_version {ITEM_MISSING_SCHEMA_VERSION} were scored under parse "
            f"policy {PARSE_POLICY}: an item the frozen parser rejects is null with a "
            "parse_failures record, is never retried or imputed, and a trait score needs all "
            f"six items. Here: {failures_total} item parse failure(s), {incomplete_total} "
            f"incomplete trait row(s), {len(rescue_cells)} rescue cell(s). Descriptives use "
            "complete units; paired contrasts use common-complete units (see missingness and "
            "pair_denominators)."
        )
    else:
        parse_failure_note = (
            "run_rater_scoring.parse_item_score raises unless the rater returns exactly one "
            "integer 1..7, so a completed shard cannot contain an unparseable item and no "
            "midpoint imputation fired in the shards summarised here."
        )

    shards_by_phase = {}
    for entry in shards:
        shards_by_phase[entry["phase"]] = shards_by_phase.get(entry["phase"], 0) + 1
    seeds_by_phase = {}
    for entry in shards:
        seeds_by_phase.setdefault(entry["phase"], set()).add(entry["seed"])
    rows_per_shard = {entry["rows"] for entry in shards}

    return {
        "schema_version": 2,
        "tool": "experiments/xlpt/summarize_pole_rater.py",
        "kind": "b4-extraversion-pole-rater-descriptive-summary",
        "pair": {"high_phase": high_phase, "low_phase": low_phase, "cell": pair_label},
        "gate_decision": "NOT_DECIDED_BY_THIS_TOOL",
        "verdict": "DESCRIPTIVE_ONLY",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "uses_gpu": False,
        "trains_or_updates_weights": False,
        "note": (
            "Descriptive reduction of the Extraversion persona-pole rater score shards. The "
            "pole surface is analysis-eligible and never gate-eligible: this tool writes no "
            "QUALIFICATION_GATE.results, freezes no ELIGIBLE_PAIRS, applies no threshold and "
            "declares no manipulation check passed. The persona text is authored by this "
            "study; no BFI-2/TIPI item wording exists in this project."
        ),
        "frozen_digests": frozen,
        "shard_count": len(shards),
        "shards_by_phase": dict(sorted(shards_by_phase.items())),
        "seeds_by_phase": {
            phase: sorted(seeds) for phase, seeds in sorted(seeds_by_phase.items())
        },
        "rows_total": sum(entry["rows"] for entry in shards),
        "rows_per_shard_consistent": len(rows_per_shard) == 1,
        "rows_per_shard_observed": sorted(rows_per_shard),
        "shards": shards,
        "by_pole_trait": {
            f"{phase}|{trait}": _describe(values)
            for (phase, trait), values in sorted(by_pole_trait.items())
        },
        "by_pole_cell": {
            f"{phase}|{checkpoint}|{trait}": _describe(values)
            for (phase, checkpoint, trait), values in sorted(by_pole_cell.items())
        },
        "paired_contrast_unit": (
            "seed mean over common-complete scenarios (primary); matched scenario reported alongside"
        ),
        "paired_by_seed": {
            f"{checkpoint}|{trait}": _paired(deltas)
            for (checkpoint, trait), deltas in sorted(seed_pairs.items())
        },
        "paired_by_scenario": {
            f"{checkpoint}|{trait}": _paired(deltas)
            for (checkpoint, trait), deltas in sorted(scenario_pairs.items())
        },
        "contrast_direction": "high minus low (positive means the high pole scored higher)",
        "interval_note": (
            "ci95_normal_approx is a normal approximation, not a t-interval, and no "
            "multiplicity correction is applied; it is descriptive context only."
        ),
        "parse_policy": PARSE_POLICY if ITEM_MISSING_SCHEMA_VERSION in schema_versions else None,
        "row_schema_versions": schema_versions,
        "item_parse_failures_total": failures_total,
        "incomplete_traits_total": incomplete_total,
        "rescue_cells": rescue_cells,
        "missingness": dict(sorted(missingness.items())),
        "pair_denominators": dict(sorted(pair_denominators.items())),
        "parse_failure_note": parse_failure_note,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarise completed Extraversion persona-pole rater score shards.",
    )
    ap.add_argument("--project", required=True)
    ap.add_argument("--score-root", default="process/results/xlpt")
    ap.add_argument(
        "--pair",
        choices=sorted(POLE_PAIRS),
        default="en",
        help="which high/low pole pair to reduce (one P x D cell per summary file)",
    )
    ap.add_argument(
        "--output-json",
        default=None,
        help="summary path inside the project (default: the pair's own summary file)",
    )
    ap.add_argument("--glob", default="seed-*.jsonl")
    ap.add_argument(
        "--expect-shards",
        type=int,
        default=0,
        help="refuse to write unless this many pole score shards are present (0 disables)",
    )
    args = ap.parse_args(argv)

    root = Path(args.project).resolve()
    score_root = (root / args.score_root).resolve()
    if not str(score_root).startswith(str(root)):
        raise ValueError("--score-root must stay inside the project")
    if not score_root.is_dir():
        raise ValueError(f"score root does not exist: {score_root}")

    high_phase, low_phase, default_output, pair_label = POLE_PAIRS[args.pair]
    shards, frozen, unit = collect(score_root, args.glob, phases=(high_phase, low_phase))
    if args.expect_shards and len(shards) != args.expect_shards:
        raise ValueError(
            f"{len(shards)} pole score shards present, expected {args.expect_shards}; "
            "refusing to summarise an incomplete block"
        )

    payload = summarize(shards, frozen, unit, high_phase=high_phase, low_phase=low_phase,
                        pair_label=pair_label)
    payload["freeze_files_checked"] = verify_freeze(root, frozen, high_phase)
    payload["project"] = str(root)
    payload["score_root"] = str(score_root)

    output = (root / (args.output_json or default_output)).resolve()
    if not str(output).startswith(str(root)):
        raise ValueError("--output-json must stay inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)

    print(json.dumps({
        "output": str(output),
        "shards": payload["shard_count"],
        "rows": payload["rows_total"],
        "gate_decision": payload["gate_decision"],
        "verdict": payload["verdict"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
