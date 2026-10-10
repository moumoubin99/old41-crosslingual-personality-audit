#!/usr/bin/env python3
"""Generate the executable English-only XLPT neutral/control shards with Qwen3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

try:
    from .inference_contract import (
        autoregressive_generate_ids,
        measured_rate,
        substream_seed,
        validate_qwen_runtime,
    )
    from .qwen3_captured import CapturedQwen3ForCausalLM
except ImportError:  # direct-script contract tests
    from inference_contract import (
        autoregressive_generate_ids,
        measured_rate,
        substream_seed,
        validate_qwen_runtime,
    )
    from qwen3_captured import CapturedQwen3ForCausalLM


PHASES = {
    "b0-competence": {"partition": "qualification", "limit": 50, "setting": "NEUTRAL", "gate": True},
    "b1-neutral": {"partition": "evaluation", "limit": 90, "setting": "NEUTRAL", "gate": False},
    "b2-compliance": {"partition": "evaluation", "limit": 90, "setting": "COMPLIANCE_CONTROL", "gate": False},
    # Extraversion persona poles on the same qualification surface the NEUTRAL B0 arm used, so the
    # three settings are directly comparable per checkpoint. Exploratory: never gate-eligible.
    "b4-extraversion-high": {"partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_HIGH", "gate": False},
    "b4-extraversion-low": {"partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_LOW", "gate": False},
    # Chinese dialogue surface (D = zh). The scenarios are the NLLB-200-3.3B single-stratum
    # rendering of the same 50 qualification scenarios, frozen by execution config v3. The
    # `_DZH` settings keep the English persona text and only ask for Chinese replies (P = en,
    # D = zh); the `_PZH` settings are the NLLB rendering of the persona text itself
    # (P = zh, D = zh). zh-b0-competence is the G1 competence screen for D = zh and, like the
    # English B0 arm, is gate-eligible and never analysis-eligible. Every other zh phase is an
    # exploratory within-checkpoint contrast: never gate-eligible, never a Rule-F admission cell.
    "zh-b0-competence": {
        "partition": "qualification", "limit": 50, "setting": "NEUTRAL_DZH", "gate": True,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": False,
        "persona_language": "English", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=en,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b4-extraversion-high": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_HIGH_DZH", "gate": False,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "English", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=en,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b4-extraversion-low": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_LOW_DZH", "gate": False,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "English", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=en,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b5-neutral-pzh": {
        "partition": "qualification", "limit": 50, "setting": "NEUTRAL_PZH", "gate": False,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": False,
        "persona_language": "Chinese", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=zh,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b5-extraversion-high-pzh": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_HIGH_PZH", "gate": False,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "Chinese", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=zh,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    "zh-b5-extraversion-low-pzh": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_LOW_PZH", "gate": False,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "Chinese", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=zh,D=zh; NLLB-200-3.3B single-stratum rendering; no BFI-2/TIPI item text",
    },
    # Block BC: the reviewed Chinese persona rendering frozen by execution config v4 (`_PZHN_`),
    # crossed with both dialogue languages. D = zh reuses the zh scenario surface and label that A
    # scores; D = en keeps the English scenario rendering and label exactly as in v2. Exploratory
    # within-checkpoint contrasts: never gate-eligible, never a Rule-F admission cell.
    "bc-pzhn-extraversion-high-dzh": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_HIGH_PZHN_DZH", "gate": False,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "Chinese", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=zh(v4 reviewed),D=zh; NLLB-200-3.3B single-stratum scenarios; no BFI-2/TIPI item text",
    },
    "bc-pzhn-extraversion-low-dzh": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_LOW_PZHN_DZH", "gate": False,
        "scenarios": "zh_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "Chinese", "dialogue_language": "Chinese", "facts_label": "zh",
        "surface_scope": "P=zh(v4 reviewed),D=zh; NLLB-200-3.3B single-stratum scenarios; no BFI-2/TIPI item text",
    },
    "bc-pzhn-extraversion-high-den": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_HIGH_PZHN_DEN", "gate": False,
        "scenarios": "english_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "Chinese", "dialogue_language": "English",
        "surface_scope": "P=zh(v4 reviewed),D=en; English scenarios as v2; no BFI-2/TIPI item text",
    },
    "bc-pzhn-extraversion-low-den": {
        "partition": "qualification", "limit": 50, "setting": "EXTRAVERSION_LOW_PZHN_DEN", "gate": False,
        "scenarios": "english_b0_competence_scenarios_v1.jsonl", "analysis": True,
        "persona_language": "Chinese", "dialogue_language": "English",
        "surface_scope": "P=zh(v4 reviewed),D=en; English scenarios as v2; no BFI-2/TIPI item text",
    },
}
QUALIFICATION_SEEDS = tuple(range(20260811, 20260811 + 24))
EVALUATION_SEEDS = tuple(range(20260911, 20260911 + 24))

# The English phases predate the language fields and keep their original record literals
# byte for byte; a phase that names no language surface is the English one.
ENGLISH_SURFACE = {
    "persona_language": "English",
    "dialogue_language": "English",
    "surface_scope": "English-only; no BFI-2/TIPI item text",
}
FACTS_LABEL_EN = "Account and policy facts:"


def phase_surface(phase: str) -> dict:
    """Record-level language/eligibility fields for a phase (English defaults preserved)."""
    spec = PHASES[phase]
    return {
        "eligible_for_analysis": bool(spec["analysis"]) if "analysis" in spec else phase != "b0-competence",
        "surface_scope": spec.get("surface_scope", ENGLISH_SURFACE["surface_scope"]),
        "persona_language": spec.get("persona_language", ENGLISH_SURFACE["persona_language"]),
        "dialogue_language": spec.get("dialogue_language", ENGLISH_SURFACE["dialogue_language"]),
    }


def facts_label_text(config, phase: str) -> str:
    """The label that introduces the scenario facts, in the phase's dialogue language.

    English phases keep the literal the frozen v1/v2 shards were generated with. A phase that
    names a facts_label key reads the rendered label from the frozen config, so the text is
    bound by the freeze receipt rather than by this module.
    """
    key = PHASES[phase].get("facts_label")
    if key is None:
        return FACTS_LABEL_EN
    try:
        text = config["facts_label"][key]
    except (KeyError, TypeError):
        raise ValueError(
            f"frozen prompt config carries no facts_label[{key!r}]; "
            "run this phase against the config version that freezes it"
        ) from None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"frozen facts_label[{key!r}] is empty")
    return text


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_self_hash(payload, label: str) -> str:
    claimed = str(payload.get("self_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "self_sha256"}
    actual = canonical_hash(unsigned)
    if claimed != actual:
        raise ValueError(f"{label} self hash mismatch")
    return claimed


def validate_frozen_inputs(project: Path, config_rel: str, receipt_rel: str):
    config = read_json(project / config_rel)
    receipt = read_json(project / receipt_rel)
    if config.get("status") != "FROZEN" or receipt.get("status") != "FROZEN":
        raise ValueError("English execution config and receipt must both be FROZEN")
    config_sha = validate_self_hash(config, "prompt config")
    receipt_sha = validate_self_hash(receipt, "freeze receipt")
    for rel, expected in receipt.get("files", {}).items():
        path = (project / rel).resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError(f"frozen input missing: {rel}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen input changed: {rel}")
    if receipt.get("prompt_config_self_sha256") != config_sha:
        raise ValueError("freeze receipt does not bind the prompt config")
    return config, config_sha, receipt_sha


def setting_text(config, setting: str) -> str:
    """Fail loudly when a phase is run against a config version that lacks its setting."""
    try:
        return config["settings"][setting]
    except KeyError:
        raise ValueError(
            f"frozen prompt config defines no setting {setting!r}; "
            "run this phase against the config version that freezes it"
        ) from None


def phase_rows(project: Path, phase: str):
    spec = PHASES[phase]
    name = spec.get("scenarios") or (
        "english_b0_competence_scenarios_v1.jsonl" if spec["partition"] == "qualification"
        else "english_b1_control_scenarios_v1.jsonl")
    rows = read_jsonl(project / "process/consumption/frozen" / name)
    if len(rows) != spec["limit"]:
        raise ValueError(f"phase {phase} expected {spec['limit']} scenarios, found {len(rows)}")
    if any(row.get("prereg_partition", "").lower() != spec["partition"] for row in rows):
        raise ValueError("bounded scenario packet has the wrong preregistered partition")
    return rows


def _coalesce_same_role(messages):
    """Aya/Llama/Gemma chat templates reject consecutive same-role turns."""
    out = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if out and out[-1].get("role") == role:
            prev = out[-1].get("content") or ""
            out[-1] = {"role": role, "content": prev.rstrip() + "\n\n" + content}
        else:
            out.append({"role": role, "content": content})
    return out


def _apply_chat_template(tokenizer, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt(tokenizer, system_text: str, context: str, history, customer_turn: str,
                 facts_label: str = FACTS_LABEL_EN) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": facts_label + "\n" + context},
        *history,
        {"role": "user", "content": customer_turn},
    ]
    try:
        return _apply_chat_template(tokenizer, messages)
    except Exception as exc:
        err = str(exc).lower()
        if "alternate" not in err and "roles must" not in err:
            raise
        return _apply_chat_template(tokenizer, _coalesce_same_role(messages))


def atomic_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    os.replace(temp, path)


def generate_scenario(model, tokenizer, system_text: str, scenario, *, root_seed: int,
                      scenario_index: int, temperature: float, top_p: float,
                      max_new_tokens: int, max_model_len: int, eos_ids: set[int],
                      facts_label: str = FACTS_LABEL_EN):
    """Entrypoint-owned per-scenario loop, also exercised by fixture integration tests."""
    import torch

    history = []
    replies = []
    prompt_hashes = []
    prompt_tokens = 0
    completion_tokens = 0
    substreams = []
    for turn_index, customer_turn in enumerate(scenario["customer_turns"]):
        prompt = build_prompt(tokenizer, system_text, scenario["context"], history, customer_turn,
                              facts_label=facts_label)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if encoded.input_ids.shape[1] > max_model_len - max_new_tokens:
            raise ValueError("prompt exceeds the frozen model-length budget")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        draw_seed = substream_seed(root_seed, scenario_index, turn_index)
        token_ids = autoregressive_generate_ids(
            model,
            encoded["input_ids"],
            encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=draw_seed,
            eos_token_ids=eos_ids,
        )
        reply = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        replies.append(reply)
        history.extend([{"role": "user", "content": customer_turn}, {"role": "assistant", "content": reply}])
        prompt_hashes.append(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_tokens += int(encoded["input_ids"].numel())
        completion_tokens += len(token_ids)
        substreams.append(draw_seed)
    return {
        "agent_turns": replies,
        "prompt_hashes": prompt_hashes,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "substream_seeds": substreams,
        "generated_call_count": len(scenario["customer_turns"]),
    }


def load_generation_runtime(model_path: Path):
    """Load the artifact-bound tokenizer and project-owned Qwen forward."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False, local_files_only=True)
    model = CapturedQwen3ForCausalLM.from_pretrained(model_path, device="cuda").eval()
    return tokenizer, model


def run(args, *, runtime_factory=load_generation_runtime, runtime_validator=validate_qwen_runtime) -> int:
    project = Path(args.project).resolve()
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise ValueError("model path does not exist")
    phase = PHASES[args.phase]
    allowed = QUALIFICATION_SEEDS if phase["partition"] == "qualification" else EVALUATION_SEEDS
    if args.seed not in allowed:
        raise ValueError(f"seed {args.seed} is not in the registered {phase['partition']} block")
    config, config_sha, receipt_sha = validate_frozen_inputs(project, args.prompt_config, args.freeze_receipt)
    setting = phase["setting"]
    system_text = setting_text(config, setting)
    facts_label = facts_label_text(config, args.phase)
    surface = phase_surface(args.phase)
    scenarios = phase_rows(project, args.phase)

    runtime_manifest_sha = runtime_validator(project / args.runtime_manifest, model_path)
    tokenizer, model = runtime_factory(model_path)
    eos_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    started = time.time()
    calls = 0
    records = []
    for scenario_index, scenario in enumerate(scenarios):
        generated = generate_scenario(
            model, tokenizer, system_text, scenario,
            root_seed=args.seed,
            scenario_index=scenario_index,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            eos_ids=eos_ids,
            facts_label=facts_label,
        )
        calls += generated["generated_call_count"]
        records.append({
            "schema_version": 2,
            "phase": args.phase,
            "partition": phase["partition"].upper(),
            "eligible_for_gate": bool(phase["gate"]),
            "eligible_for_analysis": surface["eligible_for_analysis"],
            "surface_scope": surface["surface_scope"],
            "scenario_id": scenario["scenario_id"],
            "checkpoint_id": args.model_id,
            "model_path": str(model_path),
            "persona_language": surface["persona_language"],
            "dialogue_language": surface["dialogue_language"],
            "setting": setting,
            "seed": args.seed,
            "substream_seeds": generated["substream_seeds"],
            "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                         "max_new_tokens": args.max_new_tokens, "attention": "eager"},
            "prompt_config_sha256": config_sha,
            "freeze_receipt_sha256": receipt_sha,
            "qwen_runtime_manifest_sha256": runtime_manifest_sha,
            "prompt_hashes": generated["prompt_hashes"],
            "customer_turns": scenario["customer_turns"],
            "agent_turns": generated["agent_turns"],
            "prompt_tokens": generated["prompt_tokens"],
            "completion_tokens": generated["completion_tokens"],
            "host": socket.gethostname(),
            "python": sys.executable,
        })
    elapsed = time.time() - started
    rate = measured_rate(calls, elapsed)
    for row in records:
        row["shard_elapsed_seconds"] = elapsed
        row["generated_call_count"] = calls
        row["requests_per_second"] = rate
    out = project / f"process/results/xlpt/{args.phase}/qwen3-8b/seed-{args.seed}.jsonl"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing immutable shard: {out}")
    atomic_jsonl(out, records)
    print(json.dumps({"output": str(out), "records": len(records), "calls": calls,
                      "elapsed_seconds": elapsed, "requests_per_second": rate}, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--phase", choices=tuple(PHASES), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", default="Qwen3-8B")
    ap.add_argument("--prompt-config", default="process/consumption/frozen/english_execution_v1.json")
    ap.add_argument("--freeze-receipt", default="process/consumption/frozen/english_execution_receipt_v1.json")
    ap.add_argument("--runtime-manifest", default="process/consumption/deployment/qwen3-runtime-manifest.json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    return ap


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
