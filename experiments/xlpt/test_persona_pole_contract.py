#!/usr/bin/env python3
"""Contract test for the Extraversion persona-pole phases and their frozen v2 config.

Checks that the pole phases are bound to a real frozen config, that they cannot be run
against the v1 config that never froze them, that v1's own settings are untouched so the
existing NEUTRAL shards stay comparable, and that the pole prompts carry no self-report
instrument scaffolding.
"""

from __future__ import annotations

import json
from pathlib import Path

from run_transformers_generation import PHASES, setting_text, validate_frozen_inputs

PROJECT = Path(__file__).resolve().parents[2]
V1 = "process/consumption/frozen/english_execution_v1.json"
V1_RECEIPT = "process/consumption/frozen/english_execution_receipt_v1.json"
V2 = "process/consumption/frozen/english_execution_v2.json"
V2_RECEIPT = "process/consumption/frozen/english_execution_receipt_v2.json"

POLE_PHASES = ("b4-extraversion-high", "b4-extraversion-low")
# Generic self-report scaffolding. Any of these in a behavior prompt means an inventory
# item leaked into the persona surface.
INSTRUMENT_SCAFFOLDING = (
    "i see myself as",
    "disagree strongly",
    "agree strongly",
    "neither agree nor disagree",
    "rate yourself",
    "on a scale of",
    "1 = ",
    "questionnaire",
    "bfi",
    "tipi",
)

v1_config, v1_sha, _ = validate_frozen_inputs(PROJECT, V1, V1_RECEIPT)
v2_config, v2_sha, v2_receipt_sha = validate_frozen_inputs(PROJECT, V2, V2_RECEIPT)

if v1_sha == v2_sha:
    raise AssertionError("v2 must be a distinct frozen config, not a copy of v1")

for setting in ("NEUTRAL", "COMPLIANCE_CONTROL"):
    if v2_config["settings"][setting] != v1_config["settings"][setting]:
        raise AssertionError(f"v2 changed the frozen {setting} text; existing shards stop being comparable")

for phase in POLE_PHASES:
    spec = PHASES[phase]
    if spec["gate"] or spec["partition"] != "qualification" or spec["limit"] != 50:
        raise AssertionError(f"{phase} must be a non-gating 50-scenario qualification-surface phase")
    text = setting_text(v2_config, spec["setting"])
    if not text.strip():
        raise AssertionError(f"{phase} has an empty persona prompt")
    lowered = text.lower()
    for marker in INSTRUMENT_SCAFFOLDING:
        if marker in lowered:
            raise AssertionError(f"{phase} persona prompt contains instrument scaffolding: {marker!r}")
    if "do not claim actions that the facts do not permit" not in lowered:
        raise AssertionError(f"{phase} persona prompt dropped the task-fidelity constraint")
    try:
        setting_text(v1_config, spec["setting"])
    except ValueError:
        pass
    else:
        raise AssertionError(f"{phase} must not resolve against the v1 config")

high = v2_config["settings"]["EXTRAVERSION_HIGH"]
low = v2_config["settings"]["EXTRAVERSION_LOW"]
if high == low or high == v2_config["settings"]["NEUTRAL"] or low == v2_config["settings"]["NEUTRAL"]:
    raise AssertionError("the two poles must differ from each other and from NEUTRAL")

prov = v2_config["persona_pole_provenance"]
if prov["instrument_text_used"] != "none" or prov["authored_by"] != "project (this study)":
    raise AssertionError("pole provenance must record project authorship and no instrument text")

receipt = json.loads((PROJECT / V2_RECEIPT).read_text(encoding="utf-8"))
if receipt["prompt_config_self_sha256"] != v2_sha:
    raise AssertionError("receipt v2 does not bind config v2")

print("PASS: persona-pole phases are frozen, non-gating, instrument-free and bound to config v2")
