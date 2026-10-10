#!/usr/bin/env python3
"""Run an XLPT generation shard with the frozen decoding controls.

``timing`` mode is deliberately non-scientific: it uses the English source
surface, writes ``eligible_for_analysis=false``, and may never enter a gate or
estimate.  ``qualification`` and ``evaluation`` modes require a separately
frozen prompt configuration and reject incomplete surface inventories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

from runtime_math import timing_throughput


TIMING_SYSTEM = (
    "You are a customer-support agent. Use only the supplied account facts, do not invent "
    "actions you cannot take, and answer the customer clearly in English. This prompt is for "
    "infrastructure timing only and is excluded from every scientific gate and estimate."
)
TIMING_SEED = 17091701
SEALED_SEED_RANGES = (range(20260811, 20260811 + 24), range(20260911, 20260911 + 24))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_partition_ids(project: Path, partition: str) -> set[str]:
    rendered = read_jsonl(project / "data/xlpt-support/rendered_items_and_scenarios/scenarios_en.jsonl")
    wanted = partition.upper()
    return {row["record_id"] for row in rendered if str(row.get("partition", "")).upper() == wanted}


def load_prompt(mode: str, project: Path, prompt_config: str | None, setting: str) -> tuple[str, str]:
    if mode == "timing":
        return TIMING_SYSTEM, "timing-neutral-not-a-frozen-experiment-template"
    if not prompt_config:
        raise ValueError(f"{mode} mode requires --prompt-config")
    path = (project / prompt_config).resolve()
    if not path.is_relative_to(project) or not path.is_file():
        raise ValueError("prompt config must be a project-relative existing file")
    config = read_json(path)
    if config.get("status") != "FROZEN" or not config.get("self_sha256"):
        raise ValueError("scientific runs require a FROZEN prompt config with self_sha256")
    prompts = config.get("settings") or {}
    if setting not in prompts or not isinstance(prompts[setting], str) or not prompts[setting].strip():
        raise ValueError(f"setting {setting!r} missing from frozen prompt config")
    return prompts[setting], config["self_sha256"]


def select_scenarios(project: Path, partition: str, offset: int, limit: int, scenario_id: str | None):
    allowed = load_partition_ids(project, partition)
    corpus = read_jsonl(project / "data/xlpt-support/scenarios_en_v1.0.jsonl")
    rows = [row for row in corpus if row["scenario_id"] in allowed]
    if scenario_id:
        rows = [row for row in rows if row["scenario_id"] == scenario_id]
        if not rows:
            raise ValueError(f"scenario {scenario_id} is not in {partition}")
    rows = rows[offset:offset + limit]
    if not rows:
        raise ValueError("selected shard is empty")
    return rows


def atomic_write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    temp.write_text(body, encoding="utf-8")
    os.replace(temp, path)


def verify_freeze_receipt(project: Path, relative: str, mode: str):
    path = (project / relative).resolve()
    if not path.is_relative_to(project) or not path.is_file():
        raise ValueError("freeze receipt must be a project-relative existing file")
    receipt = read_json(path)
    if receipt.get("status") != "FROZEN":
        raise ValueError("freeze receipt is not FROZEN")
    claimed_self_hash = receipt.get("self_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "self_sha256"}
    actual_self_hash = sha256_bytes(json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    if not claimed_self_hash or claimed_self_hash != actual_self_hash:
        raise ValueError("freeze receipt self-hash is missing or invalid")
    if mode != "timing" and receipt.get("scope") == "timing":
        raise ValueError("timing-scope receipt cannot authorize scientific output")
    for rel, expected in (receipt.get("files") or {}).items():
        asset = (project / rel).resolve()
        if not asset.is_relative_to(project) or not asset.is_file():
            raise ValueError(f"frozen asset missing: {rel}")
        if sha256_bytes(asset.read_bytes()) != expected.get("sha256"):
            raise ValueError(f"frozen asset changed: {rel}")
    return claimed_self_hash


def build_prompt(tokenizer, system_text: str, context: str, messages, customer_turn: str) -> str:
    chat = [{"role": "system", "content": system_text},
            {"role": "user", "content": "Account and policy fact<LOCAL_PATH>" + context}]
    chat.extend(messages)
    chat.append({"role": "user", "content": customer_turn})
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(chat, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(chat, **kwargs)


def validate_execution_boundary(mode: str, partition: str, seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if mode == "timing":
        if partition != "tuning":
            raise ValueError("timing mode is development-only and may read only the TUNING partition")
        if any(seed in reserved for reserved in SEALED_SEED_RANGES):
            raise ValueError("timing mode cannot reuse a sealed qualification or evaluation seed")
        return
    raise ValueError(
        "scientific generation remains sealed until the finalized tensor, translated surfaces, "
        "gate receipts, immutable unit keys, and retry ledger are deployed"
    )


def run(args) -> int:
    project = Path(args.project).resolve()
    validate_execution_boundary(args.mode, args.partition, args.seed)
    freeze_sha = verify_freeze_receipt(project, args.freeze_receipt, args.mode)
    if args.mode == "evaluation" and not (project / "process/consumption/frozen/ELIGIBLE_PAIRS.json").exists():
        raise ValueError("evaluation split is sealed until the pre-B1 freeze receipt exists")
    system_text, prompt_config_sha = load_prompt(args.mode, project, args.prompt_config, args.setting)
    scenarios = select_scenarios(project, args.partition, args.offset, args.limit, args.scenario_id)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    model_path = str(Path(args.model_path).resolve())
    engine = LLM(
        model=model_path,
        trust_remote_code=False,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = engine.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        seed=args.seed,
    )
    started = time.time()
    records = []
    generated_call_count = 0
    for scenario_index, scenario in enumerate(scenarios):
        history = []
        replies = []
        prompt_hashes = []
        prompt_tokens = 0
        completion_tokens = 0
        for turn_index, customer_turn in enumerate(scenario["customer_turns"]):
            prompt = build_prompt(tokenizer, system_text, scenario["context"], history, customer_turn)
            prompt_hashes.append(sha256_bytes(prompt.encode("utf-8")))
            output = engine.generate([prompt], sampling, use_tqdm=False)[0]
            generated_call_count += 1
            choice = output.outputs[0]
            reply = choice.text.strip()
            replies.append(reply)
            history.extend([
                {"role": "user", "content": customer_turn},
                {"role": "assistant", "content": reply},
            ])
            prompt_tokens += len(output.prompt_token_ids or [])
            completion_tokens += len(choice.token_ids or [])
        records.append({
            "schema_version": 1,
            "mode": args.mode,
            "eligible_for_analysis": args.mode in {"qualification", "evaluation"},
            "partition": args.partition.upper(),
            "scenario_id": scenario["scenario_id"],
            "scenario_index_in_shard": scenario_index,
            "checkpoint_id": args.model_id,
            "model_path": model_path,
            "language": args.language,
            "setting": args.setting,
            "seed": args.seed,
            "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                         "max_new_tokens": args.max_new_tokens, "repetition_penalty": 1.0},
            "prompt_config_sha256": prompt_config_sha,
            "freeze_receipt_sha256": freeze_sha,
            "prompt_hashes": prompt_hashes,
            "customer_turns": scenario["customer_turns"],
            "agent_turns": replies,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "host": socket.gethostname(),
            "python": sys.executable,
            "platform": platform.platform(),
        })
    elapsed = time.time() - started
    for row in records:
        row["shard_elapsed_seconds"] = elapsed
        row["generated_call_count"] = generated_call_count
        row["requests_per_second"] = timing_throughput(generated_call_count, elapsed)
    output_relative = args.output_jsonl.replace("{seed}", str(args.seed))
    out = (project / output_relative).resolve()
    if not out.is_relative_to(project):
        raise ValueError("output must remain inside project")
    if out.exists() and not args.replace:
        raise ValueError(f"refusing to overwrite {out}; pass --replace for a prespecified retry")
    atomic_write_jsonl(out, records)
    summary = {
        "output": str(out), "records": len(records), "turns": len(records) * 3,
        "elapsed_seconds": elapsed, "mode": args.mode,
        "eligible_for_analysis": args.mode != "timing",
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--freeze-receipt", required=True)
    ap.add_argument("--mode", choices=("timing", "qualification", "evaluation"), default="timing")
    ap.add_argument("--partition", choices=("tuning", "qualification", "evaluation"), default="tuning")
    ap.add_argument("--prompt-config")
    ap.add_argument("--setting", default="NEUTRAL")
    ap.add_argument("--language", default="English")
    ap.add_argument("--scenario-id")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=TIMING_SEED)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--replace", action="store_true")
    return ap


if __name__ == "__main__":
    try:
        sys.exit(run(parser().parse_args()))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
