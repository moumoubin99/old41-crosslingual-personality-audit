#!/usr/bin/env python3
"""Score all 24 English B0 qualification shards with one pinned rater engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from .rater_frozen_inputs import validate_rater_frozen_inputs
from .rater_runtime import measured_rate, validate_rater_runtime
from .run_rater_scoring import (
    QUALIFICATION_SEEDS,
    contract_smoke,
    parse_item_score,
    rater_decoding_contract,
    run,
    validate_source_shard,
)


class SharedEngineFactory:
    """Construct the production LLM once and reject argument drift across shards."""

    def __init__(self, constructor):
        self.constructor = constructor
        self.instance = None
        self.instances = 0
        self.kwargs = None

    def __call__(self, **kwargs):
        if self.instance is None:
            self.kwargs = dict(kwargs)
            self.instance = self.constructor(**kwargs)
            self.instances += 1
        elif kwargs != self.kwargs:
            raise AssertionError("rater engine arguments changed between qualification shards")
        return self.instance


class SharedParamsFactory:
    """Construct one immutable SamplingParams object for the full qualification block."""

    def __init__(self, constructor):
        self.constructor = constructor
        self.instance = None
        self.instances = 0
        self.kwargs = None

    def __call__(self, **kwargs):
        if self.instance is None:
            self.kwargs = dict(kwargs)
            self.instance = self.constructor(**kwargs)
            self.instances += 1
        elif kwargs != self.kwargs:
            raise AssertionError("rater sampling parameters changed between qualification shards")
        return self.instance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_completion_receipt(project: Path, relative_dir: str, report: dict) -> Path:
    """Atomically acknowledge one serial queue invocation after shard validation."""
    directory = (Path(project).resolve() / relative_dir).resolve()
    if not directory.is_relative_to(Path(project).resolve()):
        raise ValueError("qualification completion-receipt directory escapes the project")
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("chunk-[0-9][0-9].json"))
    indices = [int(path.stem.split("-")[1]) for path in existing]
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError("qualification completion receipts are not a contiguous serial ledger")
    index = len(indices) + 1
    path = directory / f"chunk-{index:02d}.json"
    payload = {
        "version": 1,
        "chunk_index": index,
        "status": report["qualification"],
        "completed_seeds": report["completed_seeds"],
        "validated_existing_seeds": report["validated_existing_seeds"],
        "validated_shard_count": len(report["completed_seeds"])
                                 + len(report["validated_existing_seeds"]),
        "total_shards": len(QUALIFICATION_SEEDS),
    }
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def validate_completed_shard(path: Path, seed: int, project: Path, args) -> None:
    """Permit crash-safe resume only for a complete immutable score shard."""
    project = Path(project).resolve()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 100:
        raise ValueError(f"existing rater shard is incomplete: {path}")
    source_path = project / f"process/results/xlpt/b0-competence/qwen3-8b/seed-{seed}.jsonl"
    source_rows = [
        json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    source_rows = validate_source_shard(project, source_rows, seed)
    source_sha = _sha256(source_path)
    source_by_scenario = {row["scenario_id"]: row for row in source_rows}
    if len(source_by_scenario) != 50:
        raise ValueError(f"source qualification scenarios are not unique: {source_path}")
    expected_pairs = [
        (row["scenario_id"], trait)
        for row in source_rows
        for trait in ("extraversion", "agreeableness")
    ]
    if [(row.get("source_scenario_id"), row.get("trait")) for row in records] != expected_pairs:
        raise ValueError(f"existing rater shard changed scenario/trait order or coverage: {path}")

    _, prompt_sha, receipt_sha = validate_rater_frozen_inputs(
        project, args.prompt_config, args.freeze_receipt
    )
    runtime_sha = _sha256(project / args.runtime_manifest)
    decoding = rater_decoding_contract()
    expected_decoding = {
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "top_k": decoding["top_k"],
        "max_new_tokens_per_item": decoding["max_tokens"],
        "seed": decoding["seed"],
        "batch_size": 1,
        "cross_request_batching": False,
    }
    expected_items = {
        "extraversion": tuple(f"E{item}" for item in range(1, 7)),
        "agreeableness": tuple(f"A{item}" for item in range(1, 7)),
    }
    timing = set()
    for row in records:
        source = source_by_scenario[row["source_scenario_id"]]
        expected_source = {
            "source_phase": source["phase"],
            "source_scenario_id": source["scenario_id"],
            "checkpoint_id": source["checkpoint_id"],
            "setting": source["setting"],
            "seed": source["seed"],
        }
        if any(row.get(field) != value for field, value in expected_source.items()):
            raise ValueError(f"existing rater shard is not bound to its source record: {path}")
        if (row.get("schema_version") != 2
                or row.get("instrument_language") != "English"
                or row.get("eligible_for_gate") is not True
                or row.get("eligible_for_analysis") is not False
                or row.get("prompt_config_sha256") != prompt_sha
                or row.get("freeze_receipt_sha256") != receipt_sha
                or row.get("rater_runtime_manifest_sha256") != runtime_sha
                or row.get("source_shard_sha256") != source_sha
                or row.get("rater_decoding") != expected_decoding):
            raise ValueError(f"existing rater shard has stale or invalid provenance: {path}")
        item_ids = expected_items[row["trait"]]
        scores = row.get("scores")
        raw = row.get("rater_raw_by_item")
        if (not isinstance(scores, dict) or tuple(scores) != item_ids
                or not isinstance(raw, dict) or tuple(raw) != item_ids):
            raise ValueError(f"existing rater shard changed rubric item coverage: {path}")
        parsed = []
        for item_id in item_ids:
            if isinstance(scores[item_id], bool) or not isinstance(scores[item_id], int):
                raise ValueError(f"existing rater shard has a noninteger score: {path}")
            if parse_item_score(raw[item_id]) != scores[item_id]:
                raise ValueError(f"existing rater shard score/raw-output mismatch: {path}")
            parsed.append(scores[item_id])
        trait_value = row.get("trait_score")
        if (isinstance(trait_value, bool) or not isinstance(trait_value, (int, float))
                or not math.isfinite(float(trait_value))
                or trait_value != sum(parsed) / 6.0):
            raise ValueError(f"existing rater shard has an invalid trait mean: {path}")
        elapsed = row.get("scoring_elapsed_seconds")
        calls = row.get("rater_calls")
        rate = row.get("rater_calls_per_second")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed)) or elapsed <= 0
                or calls != 600 or isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate)) or rate <= 0
                or rate != measured_rate(calls, elapsed)):
            raise ValueError(f"existing rater shard has invalid call metadata: {path}")
        timing.add((elapsed, calls, rate))
    if len(timing) != 1:
        raise ValueError(f"existing rater shard has nonuniform call metadata: {path}")


def run_all(
    args,
    *,
    shard_runner=run,
    engine_factory=None,
    params_factory=None,
    runtime_validator=validate_rater_runtime,
) -> int:
    project = Path(args.project).resolve()
    if engine_factory is None or params_factory is None:
        from vllm import LLM, SamplingParams

        engine_factory = SharedEngineFactory(LLM)
        params_factory = SharedParamsFactory(SamplingParams)
    initial_engine_instances = getattr(engine_factory, "instances", 0)
    runtime_cache = {}

    def cached_runtime_validator(manifest_path, model_path, deployment_root):
        key = (str(Path(manifest_path).resolve()), str(Path(model_path).resolve()),
               str(Path(deployment_root).resolve()))
        if key not in runtime_cache:
            runtime_cache[key] = runtime_validator(manifest_path, model_path, deployment_root)
        return runtime_cache[key]

    completed, skipped = [], []
    maximum_new = getattr(args, "max_new_shards", None)
    if (maximum_new is not None
            and (isinstance(maximum_new, bool) or not isinstance(maximum_new, int)
                 or maximum_new <= 0)):
        raise ValueError("max_new_shards must be a positive integer when supplied")
    for seed in QUALIFICATION_SEEDS:
        output = project / f"process/results/xlpt/b0-rater/qwen3-8b/seed-{seed}.jsonl"
        if output.exists():
            validate_completed_shard(output, seed, project, args)
            skipped.append(seed)
            continue
        shard_args = SimpleNamespace(**vars(args), seed=seed)
        shard_runner(
            shard_args,
            llm_cls=engine_factory,
            sampling_params_cls=params_factory,
            runtime_validator=cached_runtime_validator,
        )
        validate_completed_shard(output, seed, project, args)
        completed.append(seed)
        if maximum_new is not None and len(completed) >= maximum_new:
            break
    report = {
        "qualification": "complete" if len(completed) + len(skipped) == len(QUALIFICATION_SEEDS)
                         else "partial",
        "completed_seeds": completed,
        "validated_existing_seeds": skipped,
        "total_shards": len(QUALIFICATION_SEEDS),
        "engine_instances_created": getattr(engine_factory, "instances", 0) - initial_engine_instances,
    }
    receipt_dir = getattr(args, "completion_receipt_dir", None)
    if receipt_dir:
        report["completion_receipt"] = str(
            write_completion_receipt(project, receipt_dir, report)
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--project", required=True)
    argument_parser.add_argument("--rater-path", required=True)
    argument_parser.add_argument(
        "--prompt-config", default="process/consumption/frozen/english_execution_v1.json"
    )
    argument_parser.add_argument(
        "--freeze-receipt", default="process/consumption/frozen/english_execution_receipt_v1.json"
    )
    argument_parser.add_argument(
        "--runtime-manifest",
        default="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
    )
    argument_parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    argument_parser.add_argument("--max-new-shards", type=int)
    argument_parser.add_argument("--completion-receipt-dir")
    argument_parser.add_argument("--contract-smoke", action="store_true")
    return argument_parser


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.contract_smoke:
        smoke_args = SimpleNamespace(**vars(parsed), seed=QUALIFICATION_SEEDS[0])
        raise SystemExit(contract_smoke(smoke_args, entrypoint=Path(__file__)))
    raise SystemExit(run_all(parsed))
