#!/usr/bin/env python3
"""Launch-owned orchestration contract for the 24-shard rater command."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .run_rater_qualification import SharedEngineFactory, SharedParamsFactory, run_all
from .run_rater_scoring import QUALIFICATION_SEEDS


ROOT = Path.cwd().resolve()


class Constructed:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


runtime_calls = []
runner_calls = []


def runtime_validator(manifest_path, model_path, deployment_root):
    runtime_calls.append((str(manifest_path), str(model_path), str(deployment_root)))
    return "a" * 64


def shard_runner(args, *, llm_cls, sampling_params_cls, runtime_validator):
    engine = llm_cls(model="pinned", quantization="awq")
    params = sampling_params_cls(temperature=0.0)
    if runtime_validator("manifest", "model", "root") != "a" * 64:
        raise AssertionError("cached runtime validator changed")
    runner_calls.append((args.seed, id(engine), id(params)))
    project = Path(args.project)
    source = (project
              / f"process/results/xlpt/b0-competence/qwen3-8b/seed-{args.seed}.jsonl")
    source.parent.mkdir(parents=True, exist_ok=True)
    registered = [json.loads(line) for line in (
        project / "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    config = json.loads((project / args.prompt_config).read_text(encoding="utf-8"))
    receipt = json.loads((project / args.freeze_receipt).read_text(encoding="utf-8"))
    qwen_path = project / "process/consumption/deployment/qwen3-runtime-manifest.json"
    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    qwen_sha = hashlib.sha256(qwen_path.read_bytes()).hexdigest()
    source_rows = [{
        "schema_version": 2,
        "partition": "QUALIFICATION",
        "phase": "b0-competence",
        "eligible_for_gate": True,
        "eligible_for_analysis": False,
        "surface_scope": "English-only; no BFI-2/TIPI item text",
        "scenario_id": registered[scenario]["scenario_id"],
        "checkpoint_id": "Qwen3-8B",
        "model_path": "<LOCAL_PATH>/Qwen__Qwen3-8B",
        "persona_language": "English",
        "dialogue_language": "English",
        "setting": "NEUTRAL",
        "seed": args.seed,
        "substream_seeds": [args.seed * 10_000 + scenario * 3 + turn for turn in range(3)],
        "decoding": {"temperature": 0.7, "top_p": 0.95,
                     "max_new_tokens": 512, "attention": "eager"},
        "prompt_config_sha256": config["self_sha256"],
        "freeze_receipt_sha256": receipt["self_sha256"],
        "qwen_runtime_manifest_sha256": qwen_sha,
        "prompt_hashes": [hashlib.sha256(f"{args.seed}:{scenario}:{turn}".encode()).hexdigest()
                          for turn in range(3)],
        "customer_turns": registered[scenario]["customer_turns"],
        "agent_turns": [f"agent-{scenario}-{turn}" for turn in range(3)],
        "prompt_tokens": 300 + scenario,
        "completion_tokens": 30 + scenario,
        "host": qwen["worker"],
        "python": qwen["interpreter"],
        "shard_elapsed_seconds": 600.0,
        "generated_call_count": 150,
        "requests_per_second": 0.25,
    } for scenario in range(50)]
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = (Path(args.project)
              / f"process/results/xlpt/b0-rater/qwen3-8b/seed-{args.seed}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in range(50):
        for trait in ("extraversion", "agreeableness"):
            prefix = "E" if trait == "extraversion" else "A"
            items = tuple(f"{prefix}{item}" for item in range(1, 7))
            scores = {item: index + 1 for index, item in enumerate(items)}
            rows.append({
                "schema_version": 2,
                "source_phase": "b0-competence",
                "seed": args.seed,
                "source_scenario_id": registered[scenario]["scenario_id"],
                "checkpoint_id": "Qwen3-8B",
                "setting": "NEUTRAL",
                "trait": trait,
                "instrument_language": "English",
                "scores": scores,
                "trait_score": sum(scores.values()) / 6.0,
                "rater_raw_by_item": {item: str(value) for item, value in scores.items()},
                "rater_decoding": {
                    "temperature": 0.0, "top_p": 1.0, "top_k": 1,
                    "max_new_tokens_per_item": 8, "seed": 20260911,
                    "batch_size": 1, "cross_request_batching": False,
                },
                "prompt_config_sha256": json.loads(
                    (project / args.prompt_config).read_text(encoding="utf-8")
                )["self_sha256"],
                "freeze_receipt_sha256": json.loads(
                    (project / args.freeze_receipt).read_text(encoding="utf-8")
                )["self_sha256"],
                "rater_runtime_manifest_sha256": hashlib.sha256(
                    (project / args.runtime_manifest).read_bytes()
                ).hexdigest(),
                "source_shard_sha256": source_sha,
                "scoring_elapsed_seconds": 4.0,
                "rater_calls": 600,
                "rater_calls_per_second": 150.0,
                "eligible_for_gate": True,
                "eligible_for_analysis": False,
            })
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return 0


with tempfile.TemporaryDirectory(prefix="xlpt-rater-qualification-") as temporary:
    args = SimpleNamespace(
        project=temporary,
        rater_path="model",
        prompt_config="process/consumption/frozen/english_execution_v1.json",
        freeze_receipt="process/consumption/frozen/english_execution_receipt_v1.json",
        runtime_manifest="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
        gpu_memory_utilization=0.88,
        max_new_shards=2,
        completion_receipt_dir="process/results/xlpt/b0-rater/queue-receipts-v8",
        contract_smoke=False,
    )
    project = Path(temporary)
    frozen_files = [
        "process/consumption/frozen/english_execution_v1.json",
        "process/consumption/frozen/english_execution_receipt_v1.json",
        "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "process/consumption/deployment/qwen3-runtime-manifest.json",
        "refine-logs/RUBRIC_BEHAVIOR.md",
    ]
    for relative in frozen_files:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    runtime_target = project / args.runtime_manifest
    runtime_target.parent.mkdir(parents=True, exist_ok=True)
    runtime_target.write_bytes(b"runtime")
    invocation_factories = []
    for chunk_index in range(12):
        engine_factory = SharedEngineFactory(Constructed)
        params_factory = SharedParamsFactory(Constructed)
        before = len(runner_calls)
        if run_all(
            args,
            shard_runner=shard_runner,
            engine_factory=engine_factory,
            params_factory=params_factory,
            runtime_validator=runtime_validator,
        ) != 0:
            raise AssertionError("bounded qualification orchestration did not exit successfully")
        expected = list(QUALIFICATION_SEEDS[2 * chunk_index:2 * chunk_index + 2])
        if [seed for seed, _, _ in runner_calls[before:]] != expected:
            raise AssertionError("serial qualification invocation did not write its next exact pair")
        if engine_factory.instances != 1 or params_factory.instances != 1:
            raise AssertionError("bounded qualification did not use exactly one engine and params object")
        receipt = (project / args.completion_receipt_dir
                   / f"chunk-{chunk_index + 1:02d}.json")
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if (payload.get("chunk_index") != chunk_index + 1
                or payload.get("completed_seeds") != expected
                or payload.get("validated_shard_count") != 2 * (chunk_index + 1)):
            raise AssertionError("serial qualification completion receipt is not launch-owned")
        invocation_factories.append((engine_factory, params_factory))
    if [seed for seed, _, _ in runner_calls] != list(QUALIFICATION_SEEDS):
        raise AssertionError("qualification shards changed or ran out of order")
    if any(factory.instance is None for pair in invocation_factories for factory in pair):
        raise AssertionError("serial qualification invocation did not construct its owned objects")
    if len(runtime_calls) != 12:
        raise AssertionError("each bounded qualification invocation must validate the runtime once")
    if len(list((project / args.completion_receipt_dir).glob("chunk-*.json"))) != 12:
        raise AssertionError("serial qualification did not write exactly twelve completion receipts")
    resume_engine = SharedEngineFactory(Constructed)
    resume_params = SharedParamsFactory(Constructed)
    args.completion_receipt_dir = None
    if run_all(args, shard_runner=shard_runner, engine_factory=resume_engine,
               params_factory=resume_params, runtime_validator=runtime_validator) != 0:
        raise AssertionError("qualification resume did not exit successfully")
    if len(runner_calls) != len(QUALIFICATION_SEEDS):
        raise AssertionError("qualification resume re-ran an immutable completed shard")
    if resume_engine.instances != 0 or resume_params.instances != 0:
        raise AssertionError("all-skipped resume constructed a model or parameter object")
    if len(runtime_calls) != 12:
        raise AssertionError("all-skipped resume repeated the runtime closure")

    corrupt = (Path(temporary)
               / f"process/results/xlpt/b0-rater/qwen3-8b/seed-{QUALIFICATION_SEEDS[0]}.jsonl")
    original = corrupt.read_text(encoding="utf-8")
    changed = original.replace('"prompt_config_sha256": "', '"prompt_config_sha256": "0', 1)
    corrupt.write_text(changed, encoding="utf-8")
    try:
        run_all(args, shard_runner=shard_runner,
                engine_factory=SharedEngineFactory(Constructed),
                params_factory=SharedParamsFactory(Constructed),
                runtime_validator=runtime_validator)
    except ValueError:
        pass
    else:
        raise AssertionError("resume accepted a stale/corrupt completed shard")
    corrupt.write_text(original, encoding="utf-8")

    invalid_rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    first = invalid_rows[0]
    first["scores"] = {key: 1 for key in first["scores"]}
    first["rater_raw_by_item"] = {key: "1" for key in first["rater_raw_by_item"]}
    first["trait_score"] = True
    corrupt.write_text("".join(json.dumps(row) + "\n" for row in invalid_rows), encoding="utf-8")
    try:
        run_all(args, shard_runner=shard_runner,
                engine_factory=SharedEngineFactory(Constructed),
                params_factory=SharedParamsFactory(Constructed),
                runtime_validator=runtime_validator)
    except ValueError:
        pass
    else:
        raise AssertionError("resume accepted a boolean trait mean")
    invalid_rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    invalid_rows[0]["scoring_elapsed_seconds"] = True
    invalid_rows[0]["rater_calls_per_second"] = 600.0
    corrupt.write_text("".join(json.dumps(row) + "\n" for row in invalid_rows), encoding="utf-8")
    try:
        run_all(args, shard_runner=shard_runner,
                engine_factory=SharedEngineFactory(Constructed),
                params_factory=SharedParamsFactory(Constructed),
                runtime_validator=runtime_validator)
    except ValueError:
        pass
    else:
        raise AssertionError("resume accepted a boolean scoring elapsed time")
    corrupt.write_text(original, encoding="utf-8")

    source = (Path(temporary)
              / f"process/results/xlpt/b0-competence/qwen3-8b/seed-{QUALIFICATION_SEEDS[0]}.jsonl")
    original_source = source.read_text(encoding="utf-8")
    source.write_text(original_source.replace('"setting": "NEUTRAL"',
                                              '"setting": "PERSONA"', 1),
                      encoding="utf-8")
    try:
        run_all(args, shard_runner=shard_runner,
                engine_factory=SharedEngineFactory(Constructed),
                params_factory=SharedParamsFactory(Constructed),
                runtime_validator=runtime_validator)
    except ValueError:
        pass
    else:
        raise AssertionError("resume accepted a source shard from the wrong registered cell")
    source.write_text(original_source, encoding="utf-8")

    source.write_text(original_source.replace('"qwen_runtime_manifest_sha256": "',
                                              '"qwen_runtime_manifest_sha256": "0', 1),
                      encoding="utf-8")
    try:
        run_all(args, shard_runner=shard_runner,
                engine_factory=SharedEngineFactory(Constructed),
                params_factory=SharedParamsFactory(Constructed),
                runtime_validator=runtime_validator)
    except ValueError:
        pass
    else:
        raise AssertionError("resume accepted a source shard with fabricated generation provenance")
    source.write_text(original_source, encoding="utf-8")

print("PASS: twelve serial two-shard rater launches, exact coverage, and crash-safe resume")
