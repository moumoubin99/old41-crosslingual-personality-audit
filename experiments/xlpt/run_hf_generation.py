#!/usr/bin/env python3
"""English XLPT shards for non-Qwen checkpoints via Hugging Face eager forward.

Decode, seeds, frozen prompts, and I/O follow the reviewed Qwen generator.
The model forward is the pinned checkpoint's Transformers eager path, not the
project-owned Qwen capture. English-only. Not a Rule-F admission cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

try:
    from .inference_contract import measured_rate
    from .run_transformers_generation import (
        EVALUATION_SEEDS,
        PHASES,
        QUALIFICATION_SEEDS,
        atomic_jsonl,
        facts_label_text,
        generate_scenario,
        phase_rows,
        phase_surface,
        setting_text,
        validate_frozen_inputs,
    )
except ImportError:
    from inference_contract import measured_rate
    from run_transformers_generation import (
        EVALUATION_SEEDS,
        PHASES,
        QUALIFICATION_SEEDS,
        atomic_jsonl,
        facts_label_text,
        generate_scenario,
        phase_rows,
        phase_surface,
        setting_text,
        validate_frozen_inputs,
    )


SLUGS = {
    "aya-expanse-8b": "aya-expanse-8b",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "gemma-3-12b-it": "gemma-3-12b-it",
    "Qwen3-8B": "qwen3-8b",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_hf_runtime(manifest_path: Path, model_path: Path, model_id: str) -> str:
    manifest = read_json(manifest_path)
    if manifest.get("checkpoint_id") != model_id:
        raise ValueError("runtime manifest checkpoint_id does not match --model-id")
    if Path(manifest.get("model_path") or "").resolve() != model_path.resolve():
        raise ValueError("runtime manifest model_path does not match --model-path")
    import importlib.metadata

    for package, expected in manifest["package_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"runtime package version mismatch: {package}")
    for name, expected in manifest["model_files"].items():
        path = model_path / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"checkpoint/tokenizer artifact changed: {name}")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _add_eos(eos_ids: set[int], raw) -> None:
    if raw is None:
        return
    if isinstance(raw, (list, tuple)):
        eos_ids.update(int(x) for x in raw if x is not None)
    else:
        eos_ids.add(int(raw))


def collect_eos_ids(tokenizer, model) -> set[int]:
    eos_ids: set[int] = set()
    _add_eos(eos_ids, tokenizer.eos_token_id)
    _add_eos(eos_ids, getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    for token in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<|END_OF_TURN_TOKEN|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
            eos_ids.add(tid)
    if not eos_ids:
        raise ValueError("tokenizer has no usable EOS token id")
    return eos_ids


def load_hf_runtime(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
        local_files_only=True,
    )
    model.to("cuda")
    model.eval()
    return tokenizer, model


def run(args) -> int:
    project = Path(args.project).resolve()
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise ValueError("model path does not exist")
    if args.model_id not in SLUGS:
        raise ValueError(f"unsupported model-id {args.model_id}")
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
    runtime_sha = validate_hf_runtime(project / args.runtime_manifest, model_path, args.model_id)
    tokenizer, model = load_hf_runtime(model_path)
    eos_ids = collect_eos_ids(tokenizer, model)
    started = time.time()
    calls = 0
    records = []
    slug = SLUGS[args.model_id]
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
            "decoding": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "attention": "hf-eager",
            },
            "prompt_config_sha256": config_sha,
            "freeze_receipt_sha256": receipt_sha,
            "hf_runtime_manifest_sha256": runtime_sha,
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
    out = project / f"process/results/xlpt/{args.phase}/{slug}/seed-{args.seed}.jsonl"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing immutable shard: {out}")
    atomic_jsonl(out, records)
    print(json.dumps({
        "output": str(out),
        "records": len(records),
        "calls": calls,
        "elapsed_seconds": elapsed,
        "requests_per_second": rate,
        "checkpoint_id": args.model_id,
    }, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--phase", choices=tuple(PHASES), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompt-config", default="process/consumption/frozen/english_execution_v1.json")
    ap.add_argument("--freeze-receipt", default="process/consumption/frozen/english_execution_receipt_v1.json")
    ap.add_argument("--runtime-manifest", required=True)
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
