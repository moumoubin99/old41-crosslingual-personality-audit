#!/usr/bin/env python3
"""Dependency-neutral validation of the frozen English execution inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_hash(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_self_hash(payload, label: str) -> str:
    claimed = str(payload.get("self_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "self_sha256"}
    actual = canonical_hash(unsigned)
    if claimed != actual:
        raise ValueError(f"{label} self hash mismatch")
    return claimed


def validate_frozen_inputs(
    project: Path, config_rel: str, receipt_rel: str, *, required_files=None
):
    project = Path(project).resolve()
    config = read_json(project / config_rel)
    receipt = read_json(project / receipt_rel)
    if config.get("status") != "FROZEN" or receipt.get("status") != "FROZEN":
        raise ValueError("English execution config and receipt must both be FROZEN")
    config_sha = validate_self_hash(config, "prompt config")
    receipt_sha = validate_self_hash(receipt, "freeze receipt")
    receipt_files = receipt.get("files", {})
    selected = tuple(receipt_files) if required_files is None else tuple(required_files)
    if any(rel not in receipt_files for rel in selected):
        raise ValueError("rater requested an input not bound by the freeze receipt")
    for rel in selected:
        expected = receipt_files[rel]
        path = (project / rel).resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError(f"frozen input missing: {rel}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen input changed: {rel}")
    if receipt.get("prompt_config_self_sha256") != config_sha:
        raise ValueError("freeze receipt does not bind the prompt config")
    return config, config_sha, receipt_sha


# The rater validates a narrow surface on purpose: only the files it actually reads, so a
# change to an unrelated frozen input does not invalidate a scoring run. The surface is the
# same for every phase -- the prompt config in use, the registered 50-scenario qualification
# file the shard is bound to, and the behaviour rubric -- but the config itself differs by
# freeze version, so it is selected by the config the caller asks for rather than hard-coded.
# The B0 qualification surface is v1; the Extraversion persona poles are v2; the Chinese
# dialogue surface (D = zh, NLLB-200-3.3B single-stratum rendering of the same 50 scenarios)
# is v3, which binds the rendered packet instead of the English one.
RATER_FROZEN_SURFACES = {
    "process/consumption/frozen/english_execution_v1.json": (
        "process/consumption/frozen/english_execution_v1.json",
        "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "refine-logs/RUBRIC_BEHAVIOR.md",
    ),
    "process/consumption/frozen/english_execution_v2.json": (
        "process/consumption/frozen/english_execution_v2.json",
        "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "refine-logs/RUBRIC_BEHAVIOR.md",
    ),
    "process/consumption/frozen/english_execution_v3.json": (
        "process/consumption/frozen/english_execution_v3.json",
        "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "refine-logs/RUBRIC_BEHAVIOR.md",
    ),
    # v4 (block BC) crosses the reviewed Chinese persona with both dialogue languages, so its
    # shards bind either scenario packet; both are part of the surface.
    "process/consumption/frozen/english_execution_v4.json": (
        "process/consumption/frozen/english_execution_v4.json",
        "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "refine-logs/RUBRIC_BEHAVIOR.md",
    ),
}


def validate_rater_frozen_inputs(project: Path, config_rel: str, receipt_rel: str):
    """Validate only the frozen surfaces consumed by English rater scoring."""
    required = RATER_FROZEN_SURFACES.get(config_rel)
    if required is None:
        raise ValueError(f"no rater frozen-input surface is declared for {config_rel}")
    return validate_frozen_inputs(
        project,
        config_rel,
        receipt_rel,
        required_files=required,
    )
