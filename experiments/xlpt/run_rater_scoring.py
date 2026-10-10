#!/usr/bin/env python3
"""Score one Qwen English qualification shard with the frozen English rubric."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

try:
    from .inference_contract import trait_score
    from .rater_frozen_inputs import validate_rater_frozen_inputs
    from .rater_runtime import measured_rate, validate_rater_runtime
    from .score_rater import rubric_items, transcript
except ImportError:  # direct-script contract tests
    from inference_contract import trait_score
    from rater_frozen_inputs import validate_rater_frozen_inputs
    from rater_runtime import measured_rate, validate_rater_runtime
    from score_rater import rubric_items, transcript


QUALIFICATION_SEEDS = tuple(range(20260811, 20260811 + 24))
INTEGER = re.compile(r"\s*([1-7])\s*")
EXTERNAL_PROJECT_ROOT = Path("<LOCAL_PATH>/41-2026-09-05-aris-3")
STATIC_PROJECT_MIRRORS = (
    "process/consumption/frozen/english_execution_v1.json",
    "process/consumption/frozen/english_execution_receipt_v1.json",
    "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
    "process/consumption/deployment/qwen3-runtime-manifest.json",
    "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
    "refine-logs/RUBRIC_BEHAVIOR.md",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
QWEN_MODEL_PATH = "<LOCAL_PATH>/Qwen__Qwen3-8B"
RATER_MAX_MODEL_LEN = 4096
RATER_MAX_TOKENS = 8

# The rater checkpoint's chat template inserts the wall-clock date into the
# default system message whenever a request carries no explicit system message,
# which is exactly the user-only branch this scorer uses.  Scoring a second
# checkpoint on a later day would therefore change every prompt by one line and
# break comparability with the Qwen shards, which were all scored on the host's
# 2026-09-18.  Freeze the rendered date to that day so every arm sees the same
# prompt bytes.  The checkpoint file itself is never modified; the substitution
# happens in memory after the hash-bound template is loaded.
RATER_PROMPT_DATE = "2026-09-18"
RATER_TEMPLATE_DATE_CALL = 'strftime_now("%Y-%m-%d")'

# Each generated B0 arm records its own runtime-manifest digest under its own
# key and was produced by its own worker, so the source contract is per
# checkpoint rather than Qwen-only.
SOURCE_CHECKPOINTS = {
    "qwen3-8b": {
        "checkpoint_id": "Qwen3-8B",
        "model_path": QWEN_MODEL_PATH,
        "runtime_manifest": "process/consumption/deployment/qwen3-runtime-manifest.json",
        "runtime_sha_field": "qwen_runtime_manifest_sha256",
        "attention": "eager",
    },
    "aya-expanse-8b": {
        "checkpoint_id": "aya-expanse-8b",
        "model_path": "<LOCAL_PATH>/CohereLabs__aya-expanse-8b",
        "runtime_manifest": "process/consumption/deployment/aya-expanse-8b-runtime-manifest.json",
        "runtime_sha_field": "hf_runtime_manifest_sha256",
        "attention": "hf-eager",
    },
    "Llama-3.1-8B-Instruct": {
        "checkpoint_id": "Llama-3.1-8B-Instruct",
        "model_path": "<LOCAL_PATH>/meta-llama__Llama-3.1-8B-Instruct",
        "runtime_manifest": "process/consumption/deployment/Llama-3.1-8B-Instruct-runtime-manifest.json",
        "runtime_sha_field": "hf_runtime_manifest_sha256",
        "attention": "hf-eager",
    },
    "gemma-3-12b-it": {
        "checkpoint_id": "gemma-3-12b-it",
        "model_path": "<LOCAL_PATH>/google__gemma-3-12b-it",
        "runtime_manifest": "process/consumption/deployment/gemma-3-12b-it-runtime-manifest.json",
        "runtime_sha_field": "hf_runtime_manifest_sha256",
        "attention": "hf-eager",
    },
}


# Each generated arm was produced under one frozen prompt configuration and one
# persona setting, and carries the gate/analysis eligibility of its own phase.
# The rubric and the rater itself are identical across phases, so scoring a
# persona-pole shard only changes which contract the source shard is held to.
# The worker a block was generated on is a property of the block, not of the
# runtime manifest.  The qwen3 manifest was captured on the container that ran
# the B0 qualification block (autodl-container-8e334ab691-3ef3562e); the instance
# was rebuilt before the Extraversion poles were generated, so all 184 pole
# shards -- both poles, all four arms -- record the current container instead.
# The manifest bytes did not change (every pole shard still reproduces its
# runtime-manifest digest), so the host check stays exact rather than becoming a
# wildcard: each phase names the single worker its shards must come from, and a
# phase that does not name one keeps binding to the manifest's own worker.
POLE_GENERATION_WORKER = "autodl-container-f83f4c84ce-3e2ce64b"

SOURCE_PHASES = {
    "b0-competence": {
        "setting": "NEUTRAL",
        "prompt_config": "process/consumption/frozen/english_execution_v1.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v1.json",
        "eligible_for_gate": True,
        "eligible_for_analysis": False,
        "score_dir": "b0-rater",
    },
    "b4-extraversion-high": {
        "setting": "EXTRAVERSION_HIGH",
        "prompt_config": "process/consumption/frozen/english_execution_v2.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v2.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "b4-extraversion-high-rater",
    },
    "b4-extraversion-low": {
        "setting": "EXTRAVERSION_LOW",
        "prompt_config": "process/consumption/frozen/english_execution_v2.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v2.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "b4-extraversion-low-rater",
    },
    # Chinese dialogue surface (D = zh), execution config v3. The shards are generated on the
    # same worker as the English poles, over the NLLB-200-3.3B single-stratum rendering of the
    # 50 qualification scenarios. The rater instrument stays English (R16-F1); only the
    # transcript language and, for the _PZH phases, the persona language change. Every field
    # below that is absent from the English entries falls back to the English literal in
    # validate_source_shard, so the v1/v2 shards keep binding to the same contract.
    "zh-b0-competence": {
        "setting": "NEUTRAL_DZH",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": True,
        "eligible_for_analysis": False,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "zh-b0-competence-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "English",
        "dialogue_language": "Chinese",
        "surface_scope": "P=en,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b4-extraversion-high": {
        "setting": "EXTRAVERSION_HIGH_DZH",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "zh-b4-extraversion-high-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "English",
        "dialogue_language": "Chinese",
        "surface_scope": "P=en,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b4-extraversion-low": {
        "setting": "EXTRAVERSION_LOW_DZH",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "zh-b4-extraversion-low-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "English",
        "dialogue_language": "Chinese",
        "surface_scope": "P=en,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b5-neutral-pzh": {
        "setting": "NEUTRAL_PZH",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": False,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "zh-b5-neutral-pzh-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "Chinese",
        "dialogue_language": "Chinese",
        "surface_scope": "P=zh,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b5-extraversion-high-pzh": {
        "setting": "EXTRAVERSION_HIGH_PZH",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "zh-b5-extraversion-high-pzh-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "Chinese",
        "dialogue_language": "Chinese",
        "surface_scope": "P=zh,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b5-extraversion-low-pzh": {
        "setting": "EXTRAVERSION_LOW_PZH",
        "prompt_config": "process/consumption/frozen/english_execution_v3.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v3.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "zh-b5-extraversion-low-pzh-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "Chinese",
        "dialogue_language": "Chinese",
        "surface_scope": "P=zh,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    # Block BC, execution config v4: the reviewed Chinese persona rendering (`_PZHN_`) crossed
    # with both dialogue languages. The rater instrument stays English; D = en binds the English
    # scenario file, D = zh the NLLB rendering A scores.
    "bc-pzhn-extraversion-high-dzh": {
        "setting": "EXTRAVERSION_HIGH_PZHN_DZH",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "bc-pzhn-extraversion-high-dzh-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "Chinese",
        "dialogue_language": "Chinese",
        "surface_scope": "P=zh(v4 reviewed),D=zh; NLLB-200-3.3B single-stratum scenarios; no BFI-2/TIPI item text",
    },
    "bc-pzhn-extraversion-low-dzh": {
        "setting": "EXTRAVERSION_LOW_PZHN_DZH",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "bc-pzhn-extraversion-low-dzh-rater",
        "scenarios": "process/consumption/frozen/zh_b0_competence_scenarios_v1.jsonl",
        "persona_language": "Chinese",
        "dialogue_language": "Chinese",
        "surface_scope": "P=zh(v4 reviewed),D=zh; NLLB-200-3.3B single-stratum scenarios; no BFI-2/TIPI item text",
    },
    "bc-pzhn-extraversion-high-den": {
        "setting": "EXTRAVERSION_HIGH_PZHN_DEN",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "bc-pzhn-extraversion-high-den-rater",
        "scenarios": "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "persona_language": "Chinese",
        "dialogue_language": "English",
        "surface_scope": "P=zh(v4 reviewed),D=en; English scenarios as v2; no BFI-2/TIPI item text",
    },
    "bc-pzhn-extraversion-low-den": {
        "setting": "EXTRAVERSION_LOW_PZHN_DEN",
        "prompt_config": "process/consumption/frozen/english_execution_v4.json",
        "freeze_receipt": "process/consumption/frozen/english_execution_receipt_v4.json",
        "eligible_for_gate": False,
        "eligible_for_analysis": True,
        "worker": POLE_GENERATION_WORKER,
        "score_dir": "bc-pzhn-extraversion-low-den-rater",
        "scenarios": "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "persona_language": "Chinese",
        "dialogue_language": "English",
        "surface_scope": "P=zh(v4 reviewed),D=en; English scenarios as v2; no BFI-2/TIPI item text",
    },
}

# The English phases predate the language fields; a phase that names none binds to the
# literals the v1/v2 shards were generated with.
ENGLISH_SOURCE_SURFACE = {
    "scenarios": "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
    "persona_language": "English",
    "dialogue_language": "English",
    "surface_scope": "English-only; no BFI-2/TIPI item text",
}


def phase_source_surface(source_phase: str) -> dict:
    phase = SOURCE_PHASES[source_phase]
    return {key: phase.get(key, default) for key, default in ENGLISH_SOURCE_SURFACE.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_project_binding(project: Path, deployment_root: Path) -> None:
    """Bind the approved external data root to captured static launch inputs."""
    project = Path(project).resolve()
    deployment_root = Path(deployment_root).resolve()
    if project != EXTERNAL_PROJECT_ROOT.resolve():
        raise ValueError(f"rater data project root changed: {project}")
    for relative in STATIC_PROJECT_MIRRORS:
        external = (project / relative).resolve()
        deployed = (deployment_root / relative).resolve()
        if not external.is_relative_to(project) or not deployed.is_relative_to(deployment_root):
            raise ValueError(f"rater static mirror escapes a project root: {relative}")
        if not external.is_file() or not deployed.is_file() or _sha256(external) != _sha256(deployed):
            raise ValueError(f"rater static mirror differs across code/data roots: {relative}")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_source_shard(
    project: Path,
    source,
    seed: int,
    source_checkpoint: str = "qwen3-8b",
    source_phase: str = "b0-competence",
):
    """Bind a generated shard to the exact registered qualification surface.

    The scenario surface, decoding contract and provenance requirements are the
    same for every phase; the phase decides which persona setting and which
    frozen prompt configuration the shard must have been produced under, and
    whether its rows are gate- or analysis-eligible.
    """
    project = Path(project).resolve()
    if source_checkpoint not in SOURCE_CHECKPOINTS:
        raise ValueError(f"unregistered rater source checkpoint: {source_checkpoint}")
    if source_phase not in SOURCE_PHASES:
        raise ValueError(f"unregistered rater source phase: {source_phase}")
    contract = SOURCE_CHECKPOINTS[source_checkpoint]
    phase = SOURCE_PHASES[source_phase]
    surface = phase_source_surface(source_phase)
    registered_path = project / surface["scenarios"]
    registered = read_jsonl(registered_path)
    if (len(registered) != 50
            or any(row.get("prereg_partition") != "QUALIFICATION" for row in registered)):
        raise ValueError("registered B0 qualification surface is invalid")
    expected_ids = [row["scenario_id"] for row in registered]
    if len(set(expected_ids)) != 50:
        raise ValueError("registered B0 scenario identifiers are not unique")
    if not isinstance(source, list) or len(source) != 50:
        raise ValueError("rater input is not a complete 50-scenario B0 shard")
    if [row.get("scenario_id") for row in source] != expected_ids:
        raise ValueError("rater input changed the registered qualification scenario order")
    # The phase's frozen inputs must be the same bytes in the code root and the
    # data root, exactly like the statically mirrored v1 inputs. The scenario packet the
    # phase names is bound the same way (the English packet is already a static mirror).
    for relative in (phase["prompt_config"], phase["freeze_receipt"], surface["scenarios"]):
        external = project / relative
        deployed = PROJECT_SOURCE_ROOT / relative
        if not external.is_file() or not deployed.is_file() or _sha256(external) != _sha256(deployed):
            raise ValueError(f"rater phase input differs across code/data roots: {relative}")
    config = json.loads((project / phase["prompt_config"]).read_text())
    receipt = json.loads((project / phase["freeze_receipt"]).read_text())
    runtime_path = project / contract["runtime_manifest"]
    runtime = json.loads(runtime_path.read_text())
    expected_decoding = {
        "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 512,
        "attention": contract["attention"],
    }
    expected_runtime_sha = _sha256(runtime_path)
    expected_worker = phase.get("worker") or runtime.get("worker")
    if not expected_worker:
        raise ValueError(f"no generation worker is registered for {source_phase}")
    timing = set()
    for scenario_index, (row, expected) in enumerate(zip(source, registered)):
        if (row.get("schema_version") != 2
                or row.get("partition") != "QUALIFICATION"
                or row.get("phase") != source_phase
                or row.get("checkpoint_id") != contract["checkpoint_id"]
                or row.get("setting") != phase["setting"]
                or row.get("seed") != seed
                or row.get("eligible_for_gate") is not phase["eligible_for_gate"]
                or row.get("eligible_for_analysis") is not phase["eligible_for_analysis"]
                or row.get("surface_scope") != surface["surface_scope"]
                or row.get("model_path") != contract["model_path"]
                or row.get("persona_language") != surface["persona_language"]
                or row.get("dialogue_language") != surface["dialogue_language"]
                or row.get("decoding") != expected_decoding
                or row.get("prompt_config_sha256") != config.get("self_sha256")
                or row.get("freeze_receipt_sha256") != receipt.get("self_sha256")
                or row.get(contract["runtime_sha_field"]) != expected_runtime_sha
                or row.get("host") != expected_worker
                or row.get("python") != runtime.get("interpreter")
                or row.get("customer_turns") != expected.get("customer_turns")):
            raise ValueError("rater input does not match the registered B0 qualification contract")
        expected_substreams = [seed * 10_000 + scenario_index * 3 + turn for turn in range(3)]
        hashes = row.get("prompt_hashes")
        if (row.get("substream_seeds") != expected_substreams
                or not isinstance(hashes, list) or len(hashes) != 3
                or any(not isinstance(value, str) or SHA256.fullmatch(value) is None for value in hashes)):
            raise ValueError("rater input has invalid prompt/substream provenance")
        agent_turns = row.get("agent_turns")
        if (not isinstance(agent_turns, list) or len(agent_turns) != 3
                or any(not isinstance(turn, str) or not turn.strip() for turn in agent_turns)):
            raise ValueError("rater input has a malformed three-turn agent transcript")
        prompt_tokens = row.get("prompt_tokens")
        completion_tokens = row.get("completion_tokens")
        elapsed = row.get("shard_elapsed_seconds")
        calls = row.get("generated_call_count")
        rate = row.get("requests_per_second")
        if (isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int)
                or prompt_tokens <= 0 or prompt_tokens > 3 * 4096
                or isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int)
                or completion_tokens <= 0 or completion_tokens > 3 * 512
                or calls != 150
                or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed)) or elapsed <= 0
                or isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate)) or rate <= 0
                or rate != measured_rate(calls, elapsed)):
            raise ValueError("rater input has invalid generation count/token/timing provenance")
        timing.add((elapsed, calls, rate))
    if len(timing) != 1:
        raise ValueError("rater input has nonuniform shard timing provenance")
    return source


def parse_item_score(text: str) -> int:
    match = INTEGER.fullmatch(text)
    if match is None:
        raise ValueError(f"rater must return exactly one integer 1..7, got {text!r}")
    return int(match.group(1))


def load_rater_chat_template(rater_path: Path) -> str:
    """Load the hash-bound checkpoint template instead of an implicit default."""
    payload = json.loads((Path(rater_path) / "chat_template.json").read_text(encoding="utf-8"))
    template = payload.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("rater checkpoint has no explicit chat template")
    return freeze_template_date(template)


def freeze_template_date(template: str, date: str = RATER_PROMPT_DATE) -> str:
    """Replace the template's wall-clock date call with the frozen scoring date.

    The user-only request branch renders the checkpoint's default system message,
    which embeds ``strftime_now("%Y-%m-%d")``.  Left alone, the same transcript
    scored on two different days produces two different prompts.  Refuse to run
    if the call is not present exactly once: a silent no-op here would reintroduce
    the drift without any visible signal.
    """
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
        raise ValueError(f"frozen rater prompt date must be YYYY-MM-DD, got {date!r}")
    occurrences = template.count(RATER_TEMPLATE_DATE_CALL)
    if occurrences != 1:
        raise ValueError(
            "rater chat template must contain exactly one wall-clock date call, "
            f"found {occurrences}"
        )
    return template.replace(RATER_TEMPLATE_DATE_CALL, json.dumps(date))


def validate_context_window(
    prompt_token_ids,
    *,
    output_budget: int = RATER_MAX_TOKENS,
    max_model_len: int = RATER_MAX_MODEL_LEN,
) -> tuple[int, ...]:
    """Ensure the scheduler's context-limit stop can never truncate a score."""
    if (isinstance(output_budget, bool) or not isinstance(output_budget, int)
            or output_budget <= 0
            or isinstance(max_model_len, bool) or not isinstance(max_model_len, int)
            or max_model_len <= 0):
        raise ValueError("rater context limits must be positive integers")
    ids = tuple(prompt_token_ids)
    if (not ids or any(isinstance(token, bool) or not isinstance(token, int) or token < 0
                       for token in ids)):
        raise ValueError("rater prompt token ids must be nonempty nonnegative integers")
    # vLLM's check_stop length-caps when total request tokens reach
    # max_model_len.  Keep even the full output allowance strictly below it.
    if len(ids) + output_budget >= max_model_len:
        raise ValueError(
            "rater prompt plus output budget can reach the 4096-token context limit"
        )
    return ids


def build_item_request(item, rendered_transcript: str) -> str:
    item_id, facet, prompt, anchor1, anchor4, anchor7 = item
    return (
        "Score only the AGENT behavior in the transcript on this one item. "
        "Return exactly one integer from 1 through 7 and no other text.\n"
        f"ITEM {item_id} | {facet} | {prompt}\n"
        f"ANCHORS 1={anchor1} | 4={anchor4} | 7={anchor7}\n\n"
        "TRANSCRIPT\n" + rendered_transcript
    )


def rater_decoding_contract():
    """Frozen batch-one greedy decoding parameters for each independent item."""
    return {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_tokens": 8, "seed": 20260911}


def contract_smoke(args, *, entrypoint: Path | None = None) -> int:
    """Resolve the exact module/runtime/processor path without loading weights."""
    import torch
    from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns

    project = Path(args.project).resolve()
    rater_path = Path(args.rater_path).resolve()
    validate_project_binding(project, PROJECT_SOURCE_ROOT)
    runtime_sha = validate_rater_runtime(
        project / args.runtime_manifest,
        rater_path,
        PROJECT_SOURCE_ROOT,
    )
    classes = _load_logitsprocs_by_fqcns(
        ["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"]
    )
    if len(classes) != 1 or classes[0].__module__ != "experiments.xlpt.rater_runtime":
        raise AssertionError("vLLM did not resolve the project logits processor")
    processor = classes[0](None, torch.device("cpu"), False)
    logits = torch.tensor([[7.0, 7.0, 1.0], [-2.0, -1.0, -1.0]])
    expected = torch.tensor([0, 1])
    if not torch.equal(processor.apply(logits).argmax(dim=-1), expected):
        raise AssertionError("resolved project logits processor changed tie behavior")
    print(json.dumps({
        "contract_smoke": "PASS",
        "entrypoint": str((entrypoint or Path(__file__)).resolve()),
        "project_source_root": str(PROJECT_SOURCE_ROOT),
        "cwd": str(Path.cwd().resolve()),
        "processor": f"{classes[0].__module__}:{classes[0].__qualname__}",
        "runtime_manifest_sha256": runtime_sha,
    }, ensure_ascii=False))
    return 0


def atomic_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    os.replace(temp, path)


def run(
    args,
    *,
    llm_cls=None,
    sampling_params_cls=None,
    frozen_validator=validate_rater_frozen_inputs,
    runtime_validator=validate_rater_runtime,
    project_binding_validator=validate_project_binding,
    clock=time.time,
):
    if args.seed not in QUALIFICATION_SEEDS:
        raise ValueError("rater seed selector must be in the registered qualification block")
    project = Path(args.project).resolve()
    project_binding_validator(project, PROJECT_SOURCE_ROOT)
    _, config_sha, receipt_sha = frozen_validator(project, args.prompt_config, args.freeze_receipt)
    source_checkpoint = getattr(args, "source_checkpoint", None) or "qwen3-8b"
    if source_checkpoint not in SOURCE_CHECKPOINTS:
        raise ValueError(f"unregistered rater source checkpoint: {source_checkpoint}")
    source_phase = getattr(args, "source_phase", None) or "b0-competence"
    if source_phase not in SOURCE_PHASES:
        raise ValueError(f"unregistered rater source phase: {source_phase}")
    phase = SOURCE_PHASES[source_phase]
    # Scoring a shard against a different freeze than the one that generated it
    # would silently relabel its provenance, so the launch inputs must be the
    # phase's own frozen configuration.
    if (args.prompt_config != phase["prompt_config"]
            or args.freeze_receipt != phase["freeze_receipt"]):
        raise ValueError(
            f"phase {source_phase} must be scored against {phase['prompt_config']} "
            f"and {phase['freeze_receipt']}"
        )
    source_path = (
        project / f"process/results/xlpt/{source_phase}/{source_checkpoint}/seed-{args.seed}.jsonl"
    )
    source = validate_source_shard(
        project, read_jsonl(source_path), args.seed, source_checkpoint, source_phase
    )
    source_sha = _sha256(source_path)

    rater_path = Path(args.rater_path).resolve()
    runtime_sha = runtime_validator(project / args.runtime_manifest, rater_path, PROJECT_SOURCE_ROOT)

    if llm_cls is None or sampling_params_cls is None:
        from vllm import LLM, SamplingParams

        llm_cls = LLM
        sampling_params_cls = SamplingParams

    engine = llm_cls(
        model=str(rater_path),
        trust_remote_code=False,
        quantization="awq",
        dtype="auto",
        max_model_len=RATER_MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0},
        logits_processors=["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"],
    )
    tokenizer = engine.get_tokenizer()
    chat_template = load_rater_chat_template(rater_path)
    params = sampling_params_cls(**rater_decoding_contract())
    started = clock()
    calls = 0
    output_rows = []
    for row in source:
        rendered_transcript = transcript(row)
        for trait in ("extraversion", "agreeableness"):
            items = rubric_items(project / "refine-logs/RUBRIC_BEHAVIOR.md", trait)
            scores = {}
            raw_outputs = {}
            for item_id, facet, prompt, anchor1, anchor4, anchor7 in items:
                request = build_item_request((item_id, facet, prompt, anchor1, anchor4, anchor7), rendered_transcript)
                messages = [{"role": "user", "content": request}]
                prompt_token_ids = tokenizer.apply_chat_template(
                    messages,
                    chat_template=chat_template,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                prompt_token_ids = validate_context_window(prompt_token_ids)
                raw = engine.generate(
                    [{"prompt_token_ids": list(prompt_token_ids)}],
                    params,
                    use_tqdm=False,
                )[0].outputs[0].text.strip()
                scores[item_id] = parse_item_score(raw)
                raw_outputs[item_id] = raw
                calls += 1
            ordered = [scores[item[0]] for item in items]
            mean = float(trait_score(ordered).item())
            output_rows.append({
                "schema_version": 2,
                "source_phase": row["phase"],
                "source_scenario_id": row["scenario_id"],
                "checkpoint_id": row["checkpoint_id"],
                "setting": row["setting"],
                "seed": row["seed"],
                "trait": trait,
                "instrument_language": "English",
                "scores": scores,
                "trait_score": mean,
                "rater_raw_by_item": raw_outputs,
                "rater_decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                                     "max_new_tokens_per_item": 8, "seed": 20260911,
                                     "batch_size": 1, "cross_request_batching": False},
                "prompt_config_sha256": config_sha,
                "freeze_receipt_sha256": receipt_sha,
                "rater_runtime_manifest_sha256": runtime_sha,
                "source_shard_sha256": source_sha,
                "eligible_for_gate": phase["eligible_for_gate"],
                "eligible_for_analysis": phase["eligible_for_analysis"],
            })
    elapsed = clock() - started
    rate = measured_rate(calls, elapsed)
    for row in output_rows:
        row["scoring_elapsed_seconds"] = elapsed
        row["rater_calls"] = calls
        row["rater_calls_per_second"] = rate
    out = project / f"process/results/xlpt/{phase['score_dir']}/{source_checkpoint}/seed-{args.seed}.jsonl"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing immutable score shard: {out}")
    atomic_jsonl(out, output_rows)
    print(json.dumps({"output": str(out), "records": len(output_rows), "calls": calls,
                      "elapsed_seconds": elapsed, "calls_per_second": rate}, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rater-path", required=True)
    ap.add_argument("--prompt-config", default="process/consumption/frozen/english_execution_v1.json")
    ap.add_argument("--freeze-receipt", default="process/consumption/frozen/english_execution_receipt_v1.json")
    ap.add_argument("--runtime-manifest", default="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json")
    ap.add_argument("--source-checkpoint", default="qwen3-8b", choices=sorted(SOURCE_CHECKPOINTS))
    ap.add_argument("--source-phase", default="b0-competence", choices=sorted(SOURCE_PHASES))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--contract-smoke", action="store_true")
    return ap


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        raise SystemExit(contract_smoke(parsed) if parsed.contract_smoke else run(parsed))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
