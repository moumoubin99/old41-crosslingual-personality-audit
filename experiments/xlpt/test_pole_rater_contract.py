#!/usr/bin/env python3
"""Contract test for scoring the Extraversion persona-pole shards.

The rater was written for the single NEUTRAL B0 arm and hard-coded that phase,
its setting and its v1 freeze. This test pins the generalized registry: each
scorable phase must agree with the generator's own phase table on setting and
gate eligibility, pole scores must never be gate-eligible, each phase must write
to its own immutable score directory, and a shard may not be scored against a
freeze other than the one that produced it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from run_rater_scoring import SOURCE_PHASES, phase_source_surface, run, validate_source_shard
from run_transformers_generation import PHASES, phase_surface

PROJECT = Path(__file__).resolve().parents[2]
POLE_PHASES = ("b4-extraversion-high", "b4-extraversion-low")
ZH_POLE_PHASES = (
    "zh-b4-extraversion-high", "zh-b4-extraversion-low",
    "zh-b5-extraversion-high-pzh", "zh-b5-extraversion-low-pzh",
)

for name, phase in SOURCE_PHASES.items():
    generator = PHASES[name]
    surface = phase_surface(name)
    if phase["setting"] != generator["setting"]:
        raise AssertionError(f"{name}: rater setting disagrees with the generator phase table")
    if phase["eligible_for_gate"] is not bool(generator["gate"]):
        raise AssertionError(f"{name}: rater gate eligibility disagrees with the generator phase table")
    if phase["eligible_for_analysis"] is not surface["eligible_for_analysis"]:
        raise AssertionError(f"{name}: rater analysis eligibility disagrees with the generator")
    # The rater re-validates the language literals the generator stamped on every row, so the
    # two tables must agree on them or every zh shard would be refused as mislabeled.
    source_surface = phase_source_surface(name)
    for key in ("surface_scope", "persona_language", "dialogue_language"):
        if source_surface[key] != surface[key]:
            raise AssertionError(f"{name}: rater {key} disagrees with the generator")
    for relative in (phase["prompt_config"], phase["freeze_receipt"]):
        if not (PROJECT / relative).is_file():
            raise AssertionError(f"{name}: frozen input missing on disk: {relative}")

for name in ZH_POLE_PHASES:
    phase = SOURCE_PHASES[name]
    if phase["eligible_for_gate"] or not phase["eligible_for_analysis"]:
        raise AssertionError(f"{name}: zh persona-pole scores must be analysis-only, never gate-eligible")
    if not phase["prompt_config"].endswith("english_execution_v3.json"):
        raise AssertionError(f"{name}: the Chinese surface must be scored against the v3 freeze")
    if phase_source_surface(name)["dialogue_language"] != "Chinese":
        raise AssertionError(f"{name}: the Chinese surface must be scored as D = zh")

# Block BC: the reviewed P = zh persona (v4 `_PZHN_`) crossed with both dialogue languages. The
# rater must bind the same scenario packet the generator read, or D = en shards would be checked
# against the zh rendering (and vice versa).
from rater_frozen_inputs import RATER_FROZEN_SURFACES

BC_PHASES = {
    "bc-pzhn-extraversion-high-dzh": "Chinese", "bc-pzhn-extraversion-low-dzh": "Chinese",
    "bc-pzhn-extraversion-high-den": "English", "bc-pzhn-extraversion-low-den": "English",
}
for name, dialogue in BC_PHASES.items():
    phase = SOURCE_PHASES[name]
    source_surface = phase_source_surface(name)
    if phase["eligible_for_gate"] or not phase["eligible_for_analysis"]:
        raise AssertionError(f"{name}: BC pole scores must be analysis-only, never gate-eligible")
    if not phase["prompt_config"].endswith("english_execution_v4.json"):
        raise AssertionError(f"{name}: BC must be scored against the v4 freeze")
    if "_PZHN_" not in phase["setting"]:
        raise AssertionError(f"{name}: BC must use the reviewed _PZHN_ persona, not the v3 NLLB _PZH text")
    if source_surface["persona_language"] != "Chinese" or source_surface["dialogue_language"] != dialogue:
        raise AssertionError(f"{name}: BC language surface is not P = zh, D = {dialogue}")
    if not source_surface["scenarios"].endswith(PHASES[name]["scenarios"]):
        raise AssertionError(f"{name}: rater scenario packet differs from the generator's")
    if source_surface["scenarios"] not in RATER_FROZEN_SURFACES[phase["prompt_config"]]:
        raise AssertionError(f"{name}: scenario packet is outside the v4 rater frozen surface")
    if (PHASES[name].get("facts_label") == "zh") != (dialogue == "Chinese"):
        raise AssertionError(f"{name}: facts label language must follow D")

if SOURCE_PHASES["zh-b0-competence"]["eligible_for_analysis"] or not SOURCE_PHASES["zh-b0-competence"]["eligible_for_gate"]:
    raise AssertionError("zh-b0-competence is the G1 screen: gate-eligible, never analysis-eligible")
if SOURCE_PHASES["zh-b5-neutral-pzh"]["eligible_for_analysis"] or SOURCE_PHASES["zh-b5-neutral-pzh"]["eligible_for_gate"]:
    raise AssertionError("zh-b5-neutral-pzh is a reference arm: neither gate- nor analysis-eligible")

if SOURCE_PHASES["b0-competence"]["score_dir"] != "b0-rater":
    raise AssertionError("the existing NEUTRAL score shards must keep their directory")

dirs = {name: phase["score_dir"] for name, phase in SOURCE_PHASES.items()}
if len(set(dirs.values())) != len(dirs):
    raise AssertionError("two phases would write into the same immutable score directory")

for name in POLE_PHASES:
    phase = SOURCE_PHASES[name]
    if phase["eligible_for_gate"]:
        raise AssertionError(f"{name}: persona-pole scores must never be gate-eligible")
    if not phase["prompt_config"].endswith("english_execution_v2.json"):
        raise AssertionError(f"{name}: persona poles must be scored against the v2 freeze")

try:
    validate_source_shard(PROJECT, [], 20260811, "qwen3-8b", "b4-extraversion-mid")
except ValueError:
    pass
else:
    raise AssertionError("an unregistered source phase was accepted")


class Rejected(AssertionError):
    pass


def refuse(*_args, **_kwargs):
    raise Rejected("the launch reached the frozen-input validator despite a freeze mismatch")


args = SimpleNamespace(
    project=str(PROJECT),
    seed=20260811,
    rater_path="<LOCAL_PATH>/OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym",
    # v1 freeze against a pole phase: the shard's own provenance would be relabeled.
    prompt_config="process/consumption/frozen/english_execution_v1.json",
    freeze_receipt="process/consumption/frozen/english_execution_receipt_v1.json",
    runtime_manifest="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
    source_checkpoint="qwen3-8b",
    source_phase="b4-extraversion-high",
    gpu_memory_utilization=0.88,
)
try:
    run(args,
        frozen_validator=lambda *a, **k: (None, "0" * 64, "0" * 64),
        runtime_validator=refuse,
        project_binding_validator=lambda *a, **k: None)
except Rejected:
    raise AssertionError("a pole shard was accepted against the v1 freeze")
except ValueError:
    pass
else:
    raise AssertionError("a pole shard was accepted against the v1 freeze")

print("PASS: pole phases are scorable, non-gating, separately stored and bound to their own freeze")
