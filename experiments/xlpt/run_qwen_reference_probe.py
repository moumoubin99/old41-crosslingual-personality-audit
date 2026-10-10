#!/usr/bin/env python3
"""Single dispatch point for the five reviewable Qwen3 deployment units.

The vLLM generation path cannot be reviewed as deployed: it bottoms out in
``torch.ops.vllm.unified_attention``, a compiled FlashAttention CUDA kernel that
is neither capturable Python source nor executable under the deployment probe's
CPU constraint.  Two independent deployment reviews rejected the packet for that
reason.  The deployment-formula protocol's own instruction is to "split an
oversized run into a reviewable unit or report that the current implementation
cannot cover it; never omit a dependency to get a PASS".  This runner is that
split: every unit executes the captured pure-torch reference in
``qwen3_reference.py`` and nothing else.

Units
-----
``kernels``          RMS normalization, rotary tables, projection, SwiGLU.
``attention``        grouped-query causal attention and QK normalization.
``autoregression``   the decoder stack, cached decoding and greedy determinism.
``checkpoint``       checkpoint identity, weight loading and transformers parity.
``sampler``          the seeded nucleus sampler at temperature 0.7 / top_p 0.95.
``generation``       the apparatus run: the captured forward and sampler driven
                     over the **TUNING** partition of the English support
                     corpus on the released checkpoint.

Scope, stated rather than assumed
---------------------------------
No output of any unit is a scientific estimator.  Every receipt carries
``eligible_for_analysis: false``.  Nothing here trains, fine-tunes or updates a
weight.  The sealed evaluation and qualification seed blocks are refused by
``qwen3_reference.check_seed``, and ``generation`` additionally refuses any row
that is not English TUNING.  These runs feed apparatus verification and timing;
they feed no gate, no admission decision and no reported effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch

from qwen3_reference import (
    check_seed,
    checkpoint_identity,
    generate,
    load_checkpoint_params,
)

# Every contract unit is one module whose ``main`` raises on any failure.
CONTRACT_UNITS = {
    "kernels": "test_qwen3_kernels",
    "attention": "test_qwen3_attention",
    "autoregression": "test_qwen3_autoregression",
    "checkpoint": "test_qwen3_checkpoint_binding",
    "sampler": "test_qwen3_sampler",
}
UNITS = tuple(CONTRACT_UNITS) + ("generation",)

ALLOWED_PARTITION = "tuning"
ALLOWED_LANGUAGE = "en"
TEMPERATURE = 0.7
TOP_P = 0.95


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_inside(root, relative, must_exist=True):
    """A project-relative path, refused if it escapes the project root."""
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"{relative} escapes the project root {root}")
    if must_exist and not target.exists():
        raise ValueError(f"{relative} does not exist under {root}")
    return target


def load_jsonl(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but no CUDA device is visible")
        return torch.device("cuda:0")
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def device_record(device):
    if device.type != "cuda":
        return {"device": "cpu", "cuda_available": torch.cuda.is_available()}
    return {"device": torch.cuda.get_device_name(device.index or 0), "cuda_available": True}


# --------------------------------------------------------------------------
# Contract units
# --------------------------------------------------------------------------


def run_contract_unit(unit):
    """Execute the unit's contract test in this interpreter and record it."""
    import importlib

    module_name = CONTRACT_UNITS[unit]
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve()
    started = time.monotonic()
    status = module.main()
    elapsed = time.monotonic() - started
    if status != 0:
        raise RuntimeError(f"{module_name}.main returned {status}")
    return {
        "unit": unit,
        "kind": "contract",
        "test_module": module_name,
        "test_file": source.name,
        "test_file_sha256": sha256_path(source),
        "elapsed_seconds": elapsed,
        "result": "PASS",
    }


# --------------------------------------------------------------------------
# Apparatus generation unit
# --------------------------------------------------------------------------


def tuning_rows(path, limit):
    """English TUNING rows only; anything else is a hard refusal."""
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"{path} holds no rows")
    selected = [row for row in rows
                if row.get("partition") == ALLOWED_PARTITION
                and row.get("language") == ALLOWED_LANGUAGE]
    if not selected:
        raise ValueError(
            f"{path} holds no {ALLOWED_LANGUAGE} {ALLOWED_PARTITION} rows; this apparatus run "
            "is fail-closed to the tuning partition and must not read qualification or evaluation")
    foreign = {row.get("partition") for row in rows} - {ALLOWED_PARTITION}
    selected.sort(key=lambda row: row["record_id"])
    if limit > 0:
        selected = selected[:limit]
    return selected, sorted(part for part in foreign if part)


def run_generation_unit(args, root, device):
    """Drive the captured forward and sampler over English TUNING scenarios.

    This is apparatus verification and throughput measurement on the released
    checkpoint, not a scientific generation: no persona instruction is applied,
    because the confirmatory prompt templates are not finalized and inventing one
    here would manufacture an unregistered condition.  The continuations are
    recorded by digest and length, and are eligible for no analysis.
    """
    from transformers import AutoTokenizer

    check_seed(args.seed)
    scenarios = resolve_inside(root, args.scenarios)
    rows, excluded = tuning_rows(scenarios, args.max_scenarios)

    model_path = Path(args.model_path)
    identity = checkpoint_identity(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    params, config = load_checkpoint_params(model_path, device=device, dtype=torch.bfloat16)

    produced = []
    prompt_tokens = 0
    new_tokens = 0
    started = time.monotonic()
    for index, row in enumerate(rows):
        # One disjoint sub-stream per scenario, each checked against the sealed blocks.
        seed = args.seed + index
        check_seed(seed)
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"{row['record_id']} tokenized to nothing")
        if len(ids) > args.max_prompt_tokens:
            ids = ids[: args.max_prompt_tokens]
        tokens, _ = generate(
            params, config, ids, max_new_tokens=args.max_new_tokens,
            temperature=TEMPERATURE, top_p=TOP_P,
            generator=torch.Generator().manual_seed(seed),
            stop_ids=tuple(tokenizer.all_special_ids or ()),
        )
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        prompt_tokens += len(ids)
        new_tokens += len(tokens)
        produced.append({
            "record_id": row["record_id"],
            "partition": row["partition"],
            "language": row["language"],
            "seed": seed,
            "prompt_tokens": len(ids),
            "new_tokens": len(tokens),
            "continuation_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "continuation_chars": len(text),
        })
    elapsed = time.monotonic() - started

    return {
        "unit": "generation",
        "kind": "apparatus",
        "scenarios": str(scenarios.relative_to(root)),
        "scenarios_sha256": sha256_path(scenarios),
        "partition": ALLOWED_PARTITION,
        "language": ALLOWED_LANGUAGE,
        "excluded_partitions": excluded,
        "scenario_count": len(rows),
        "checkpoint": identity,
        "decoding": {"temperature": TEMPERATURE, "top_p": TOP_P,
                     "max_new_tokens": args.max_new_tokens,
                     "max_prompt_tokens": args.max_prompt_tokens,
                     "dtype": "bfloat16", "sampler": "qwen3_reference.nucleus_next_token"},
        "base_seed": args.seed,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "new_tokens_per_second": (new_tokens / elapsed) if elapsed > 0 else None,
        "items": produced,
        "result": "PASS",
    }


# --------------------------------------------------------------------------


def main(args):
    if args.unit not in UNITS:
        raise ValueError(f"unknown unit {args.unit}")
    root = Path(args.project).resolve()
    if not root.is_dir():
        raise ValueError(f"--project {root} is not a directory")
    device = select_device(args.device)

    if args.unit == "generation":
        body = run_generation_unit(args, root, device)
    else:
        body = run_contract_unit(args.unit)

    record = {
        "schema_version": 1,
        "mode": "qwen3-reference-unit",
        "eligible_for_analysis": False,
        "trains_or_updates_weights": False,
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "reference_module_sha256": sha256_path(
            Path(__file__).resolve().parent / "qwen3_reference.py"),
        "runner_sha256": sha256_path(Path(__file__).resolve()),
        "note": ("Apparatus verification on the captured pure-torch Qwen3 reference. "
                 "No output enters a gate, an admission decision or a reported effect."),
    }
    record.update(device_record(device))
    record.update(body)

    output = (root / args.output_json.replace("{unit}", args.unit)
              .replace("{seed}", str(args.seed))).resolve()
    if not output.is_relative_to(root):
        raise ValueError("output escapes the project root")
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"unit": args.unit, "result": record["result"],
                      "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run one reviewable Qwen3 deployment unit.")
    ap.add_argument("--unit", required=True, choices=UNITS)
    ap.add_argument("--project", default=".")
    ap.add_argument("--output-json",
                    default="process/logs/qwen-reference-{unit}-seed-{seed}.json")
    ap.add_argument("--seed", type=int, default=17091701)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--model-path", default="<LOCAL_PATH>/Qwen__Qwen3-8B")
    ap.add_argument("--scenarios",
                    default="data/xlpt-support/rendered_items_and_scenarios/scenarios_en.jsonl")
    ap.add_argument("--max-scenarios", type=int, default=6)
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    raise SystemExit(main(ap.parse_args()))
