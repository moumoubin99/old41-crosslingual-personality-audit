#!/usr/bin/env python3
"""Contract test for the Chinese-surface (D = zh) phases and their frozen v3 config.

The Chinese arms reuse the English generator with three things swapped: the scenario
packet, the setting text, and the label that introduces the scenario facts. Each of those
is a place where the two surfaces could silently diverge, so this test pins them:

* v3 is a real, distinct freeze that leaves every v2 string byte-identical, so the English
  shards already generated under v2 stay comparable;
* the ``_DZH`` settings are the v2 persona texts with only the response-language sentence
  replaced -- the (P = en, D = zh) cell must differ from the English cell in the dialogue
  language and in nothing else;
* the ``_PZH`` settings are actually in Chinese, so (P = zh, D = zh) is a matched-language
  pole rather than a mislabeled copy of the English one;
* the facts label resolves from the freeze for every zh phase and refuses to resolve
  against v2, so a zh phase cannot be run under a config that never froze its label;
* no persona prompt on either surface carries self-report instrument scaffolding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run_transformers_generation import (
    PHASES,
    facts_label_text,
    phase_surface,
    setting_text,
    validate_frozen_inputs,
)

PROJECT = Path(__file__).resolve().parents[2]
V2 = "process/consumption/frozen/english_execution_v2.json"
V2_RECEIPT = "process/consumption/frozen/english_execution_receipt_v2.json"
V3 = "process/consumption/frozen/english_execution_v3.json"
V3_RECEIPT = "process/consumption/frozen/english_execution_receipt_v3.json"

ZH_PACKETS = ("zh_b0_competence_scenarios_v1.jsonl", "zh_b1_control_scenarios_v1.jsonl")
ZH_INSTRUCTION_EN = "Respond to the customer's request in Chinese."
EN_INSTRUCTION = "Respond to the customer's request in English."

# (zh phase, the v2 setting it is derived from, gate-eligible, analysis-eligible)
ZH_PHASES = (
    ("zh-b0-competence", "NEUTRAL", True, False),
    ("zh-b4-extraversion-high", "EXTRAVERSION_HIGH", False, True),
    ("zh-b4-extraversion-low", "EXTRAVERSION_LOW", False, True),
    ("zh-b5-neutral-pzh", "NEUTRAL", False, False),
    ("zh-b5-extraversion-high-pzh", "EXTRAVERSION_HIGH", False, True),
    ("zh-b5-extraversion-low-pzh", "EXTRAVERSION_LOW", False, True),
)
DZH_SETTINGS = ("NEUTRAL_DZH", "EXTRAVERSION_HIGH_DZH", "EXTRAVERSION_LOW_DZH")
PZH_SETTINGS = ("NEUTRAL_PZH", "EXTRAVERSION_HIGH_PZH", "EXTRAVERSION_LOW_PZH")

INSTRUMENT_SCAFFOLDING = (
    "i see myself as", "disagree strongly", "agree strongly", "neither agree nor disagree",
    "rate yourself", "on a scale of", "1 = ", "questionnaire", "bfi", "tipi",
)


def has_han(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


v2_config, v2_sha, _ = validate_frozen_inputs(PROJECT, V2, V2_RECEIPT)
v3_config, v3_sha, _ = validate_frozen_inputs(PROJECT, V3, V3_RECEIPT)

if v3_sha == v2_sha:
    raise AssertionError("v3 must be a distinct frozen config, not a copy of v2")
for name, text in v2_config["settings"].items():
    if v3_config["settings"].get(name) != text:
        raise AssertionError(f"v3 changed the frozen {name} text; the English shards stop being comparable")

# --- the settings the Chinese surface adds -------------------------------------------

for name in DZH_SETTINGS + PZH_SETTINGS:
    text = v3_config["settings"].get(name)
    if not isinstance(text, str) or not text.strip():
        raise AssertionError(f"v3 has no {name} persona prompt")
    lowered = text.lower()
    for marker in INSTRUMENT_SCAFFOLDING:
        if marker in lowered:
            raise AssertionError(f"{name} contains instrument scaffolding: {marker!r}")

for name in DZH_SETTINGS:
    text = v3_config["settings"][name]
    base = v2_config["settings"][name[: -len("_DZH")]]
    if not text.endswith(ZH_INSTRUCTION_EN):
        raise AssertionError(f"{name} must ask, in English, for a Chinese reply")
    if EN_INSTRUCTION in text:
        raise AssertionError(f"{name} still carries the English response instruction")
    if text != base[: -len(EN_INSTRUCTION)] + ZH_INSTRUCTION_EN:
        raise AssertionError(
            f"{name} differs from its English persona in more than the response-language "
            "sentence; P = en would no longer be held fixed across the dialogue-language switch"
        )
    if has_han(text):
        raise AssertionError(f"{name} is the P = en surface: its persona text must stay English")

for name in PZH_SETTINGS:
    text = v3_config["settings"][name]
    if not has_han(text):
        raise AssertionError(f"{name} is the P = zh surface: its persona text must be Chinese")
    if EN_INSTRUCTION in text or ZH_INSTRUCTION_EN in text:
        raise AssertionError(f"{name} kept an English instruction sentence on the P = zh surface")

for pair in (DZH_SETTINGS, PZH_SETTINGS):
    if len(set(v3_config["settings"][n] for n in pair)) != len(pair):
        raise AssertionError(f"the {pair[0].rsplit('_', 1)[-1]} settings are not pairwise distinct")

# --- the facts label -------------------------------------------------------------------

label = v3_config.get("facts_label", {}).get("zh")
if not isinstance(label, str) or not label.strip() or not has_han(label):
    raise AssertionError("v3 must freeze a Chinese facts_label['zh']")

# --- the phase table --------------------------------------------------------------------

for phase, _base, gate, analysis in ZH_PHASES:
    spec = PHASES[phase]
    surface = phase_surface(phase)
    if spec["partition"] != "qualification" or spec["limit"] != 50:
        raise AssertionError(f"{phase} must run the 50-scenario qualification surface")
    if spec["scenarios"] != ZH_PACKETS[0]:
        raise AssertionError(f"{phase} must read the frozen Chinese qualification packet")
    if bool(spec["gate"]) is not gate:
        raise AssertionError(f"{phase}: gate eligibility does not match the registered design")
    if surface["eligible_for_analysis"] is not analysis:
        raise AssertionError(f"{phase}: analysis eligibility does not match the registered design")
    if surface["dialogue_language"] != "Chinese":
        raise AssertionError(f"{phase} must be recorded as D = zh")
    if facts_label_text(v3_config, phase) != label:
        raise AssertionError(f"{phase} does not resolve the frozen Chinese facts label")
    try:
        facts_label_text(v2_config, phase)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{phase} resolved a facts label against v2, which never froze one")
    try:
        setting_text(v2_config, spec["setting"])
    except ValueError:
        pass
    else:
        raise AssertionError(f"{phase} must not resolve its setting against the v2 config")

if len({PHASES[p]["setting"] for p, _b, _g, _a in ZH_PHASES}) != len(ZH_PHASES):
    raise AssertionError("two Chinese phases share a setting")
if sum(1 for _p, _b, gate, _a in ZH_PHASES if gate) != 1:
    raise AssertionError("exactly one Chinese phase -- the G1 competence screen -- may be gate-eligible")

# --- provenance and the single-stratum deviation ------------------------------------------

prov = v3_config["zh_rendering_provenance"]
if prov["engine_stratum"] != "C" or prov["strata_used"] != 1:
    raise AssertionError("v3 must record the Chinese rendering as a single-stratum engine-C run")
if not prov["engine_id"] or not prov["engine_revision"]:
    raise AssertionError("v3 must pin the rendering engine and its revision")
if not any("single" in item.lower() and "stratum" in item.lower() for item in v3_config["limitations"]):
    raise AssertionError("the single-stratum deviation must be carried in v3's limitations")

receipt = json.loads((PROJECT / V3_RECEIPT).read_text(encoding="utf-8"))
if receipt["prompt_config_self_sha256"] != v3_sha:
    raise AssertionError("receipt v3 does not bind config v3")
for packet in ZH_PACKETS:
    # the receipt keys files by their project-relative path, as receipts v1 and v2 do
    paths = [p for p in receipt["files"] if Path(p).name == packet]
    if not paths:
        raise AssertionError(f"receipt v3 does not hash the frozen {packet}")
    digest = hashlib.sha256((PROJECT / paths[0]).read_bytes()).hexdigest()
    if digest != receipt["files"][paths[0]]:
        raise AssertionError(f"receipt v3's hash for {packet} does not match the file on disk")

print("PASS: the Chinese surface is frozen in v3, leaves v2 untouched, and separates P = en from P = zh")
