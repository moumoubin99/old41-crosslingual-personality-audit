#!/usr/bin/env python3
"""Launch-owned integration contract for run_rater_scoring.run."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .rater_frozen_inputs import validate_rater_frozen_inputs
from .rater_runtime import validate_rater_runtime
from .run_rater_scoring import run, validate_source_shard
from .inference_contract import substream_seed


ROOT = Path.cwd().resolve()


class FakeSamplingParams:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.__class__.instances.append(self)


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, chat_template, tokenize, add_generation_prompt
    ):
        if (tokenize is not True or add_generation_prompt is not True
                or "[INST]" not in chat_template):
            raise AssertionError("chat template flags changed")
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise AssertionError("rater request is not one user message")
        return [1, 3, 4]


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.calls = []
        self.tokenizer = FakeTokenizer()
        self.__class__.instances.append(self)

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, params, use_tqdm):
        if len(prompts) != 1 or use_tqdm is not False:
            raise AssertionError("rater calls must remain independent batch-one calls")
        if prompts != [{"prompt_token_ids": [1, 3, 4]}]:
            raise AssertionError("rater did not submit the checked real-token prompt shape")
        # Adjacent six-call blocks are the two traits for one scenario.  Their
        # deliberately asymmetric, unequal-mean patterns make the integration
        # test sensitive to item ordering, trait grouping, and aggregation.
        patterns = ([1, 2, 3, 4, 6, 7], [7, 7, 5, 4, 2, 1])
        block = (len(self.calls) // 6) % 2
        score = patterns[block][len(self.calls) % 6]
        self.calls.append((prompts[0], params, use_tqdm))
        return [SimpleNamespace(outputs=[SimpleNamespace(text=str(score))])]


validation = {}


def frozen_validator(project, prompt_config, freeze_receipt):
    del project
    result = validate_rater_frozen_inputs(ROOT, prompt_config, freeze_receipt)
    validation["frozen"] = result
    return result


def runtime_validator(manifest_path, model_path, deployment_root):
    if deployment_root != ROOT or manifest_path.name != "mistral-rater-runtime-v2-manifest.json":
        raise AssertionError("runtime validator did not receive the deployment working directory")
    result = validate_rater_runtime(
        ROOT / "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
        model_path,
        deployment_root,
    )
    validation["runtime"] = result
    return result


def project_binding_validator(project, deployment_root):
    if project.name != "xlpt-rater-integration-" and not project.name.startswith("xlpt-rater-integration-"):
        raise AssertionError("integration did not supply its isolated data root")
    from .run_rater_scoring import validate_project_binding

    validate_project_binding(
        Path("<LOCAL_PATH>/41-2026-09-05-aris-3"), deployment_root
    )
    validation["project_binding"] = True


with tempfile.TemporaryDirectory(prefix="xlpt-rater-integration-") as temporary:
    project = Path(temporary)
    static_files = (
        "refine-logs/RUBRIC_BEHAVIOR.md",
        "process/consumption/frozen/english_execution_v1.json",
        "process/consumption/frozen/english_execution_receipt_v1.json",
        "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "process/consumption/deployment/qwen3-runtime-manifest.json",
    )
    for relative in static_files:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    registered = [json.loads(line) for line in (
        project / "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    config = json.loads((project / "process/consumption/frozen/english_execution_v1.json").read_text())
    receipt = json.loads((project / "process/consumption/frozen/english_execution_receipt_v1.json").read_text())
    qwen_runtime_path = project / "process/consumption/deployment/qwen3-runtime-manifest.json"
    qwen_runtime = json.loads(qwen_runtime_path.read_text())
    qwen_runtime_sha = hashlib.sha256(qwen_runtime_path.read_bytes()).hexdigest()
    source_path = project / "process/results/xlpt/b0-competence/qwen3-8b/seed-20260811.jsonl"
    source_path.parent.mkdir(parents=True)
    rows = []
    for index, scenario in enumerate(registered):
        rows.append({
            "schema_version": 2,
            "phase": "b0-competence",
            "partition": "QUALIFICATION",
            "eligible_for_gate": True,
            "eligible_for_analysis": False,
            "surface_scope": "English-only; no BFI-2/TIPI item text",
            "scenario_id": scenario["scenario_id"],
            "checkpoint_id": "Qwen3-8B",
            "model_path": "<LOCAL_PATH>/Qwen__Qwen3-8B",
            "persona_language": "English",
            "dialogue_language": "English",
            "setting": "NEUTRAL",
            "seed": 20260811,
            "substream_seeds": [20260811 * 10_000 + index * 3 + turn for turn in range(3)],
            "decoding": {"temperature": 0.7, "top_p": 0.95,
                         "max_new_tokens": 512, "attention": "eager"},
            "prompt_config_sha256": config["self_sha256"],
            "freeze_receipt_sha256": receipt["self_sha256"],
            "qwen_runtime_manifest_sha256": qwen_runtime_sha,
            "prompt_hashes": [hashlib.sha256(f"{index}:{turn}".encode()).hexdigest()
                              for turn in range(3)],
            "customer_turns": scenario["customer_turns"],
            "agent_turns": [f"agent-{index}-{turn}" for turn in range(3)],
            "prompt_tokens": 300 + index,
            "completion_tokens": 30 + index,
            "host": qwen_runtime["worker"],
            "python": qwen_runtime["interpreter"],
            "shard_elapsed_seconds": 600.0,
            "generated_call_count": 150,
            "requests_per_second": 0.25,
        })
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    # The rater's varying --seed is an immutable upstream-generation shard
    # selector, not a switch in the scoring algorithm.  Exercise a second
    # registered seed, prove that every generated substream changes through
    # the frozen injective mapping, and prove that cross-seed relabeling is
    # rejected by the launch-owned source validator.  B0 generation is already
    # terminal; this rater launch binds its immutable shards, not generator code.
    second_seed = 20260812
    second_rows = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["seed"] = second_seed
        updated["substream_seeds"] = [
            substream_seed(second_seed, index, turn) for turn in range(3)
        ]
        second_rows.append(updated)
        first_streams = tuple(row["substream_seeds"])
        second_streams = tuple(updated["substream_seeds"])
        if (first_streams == second_streams
                or first_streams != tuple(substream_seed(20260811, index, turn) for turn in range(3))):
            raise AssertionError("root-seed perturbation did not change only registered substreams")
    validate_source_shard(project, rows, 20260811)
    validate_source_shard(project, second_rows, second_seed)
    try:
        validate_source_shard(project, second_rows, 20260811)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-seed shard relabeling was accepted")
    args = SimpleNamespace(
        project=str(project), seed=20260811,
        rater_path="<LOCAL_PATH>/OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym",
        prompt_config="process/consumption/frozen/english_execution_v1.json",
        freeze_receipt="process/consumption/frozen/english_execution_receipt_v1.json",
        runtime_manifest="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
        gpu_memory_utilization=0.88,
    )
    ticks = iter((10.0, 14.0))
    result = run(
        args,
        llm_cls=FakeLLM,
        sampling_params_cls=FakeSamplingParams,
        frozen_validator=frozen_validator,
        runtime_validator=runtime_validator,
        project_binding_validator=project_binding_validator,
        clock=lambda: next(ticks),
    )
    if (result != 0 or len(FakeLLM.instances) != 1 or len(FakeSamplingParams.instances) != 1
            or validation.get("project_binding") is not True):
        raise AssertionError("launch-owned run did not construct one engine and one parameter object")
    engine = FakeLLM.instances[0]
    if engine.kwargs != {
        "model": "<LOCAL_PATH>/OPEA__Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym",
        "trust_remote_code": False,
        "quantization": "awq",
        "dtype": "auto",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.88,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "limit_mm_per_prompt": {"image": 0},
        "logits_processors": ["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"],
    }:
        raise AssertionError(f"rater engine contract changed: {engine.kwargs}")
    expected_params = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_tokens": 8, "seed": 20260911}
    params = FakeSamplingParams.instances[0]
    if params.kwargs != expected_params or len(engine.calls) != 600:
        raise AssertionError("sampling parameters or total independent calls changed")
    if any(call_params is not params for _, call_params, _ in engine.calls):
        raise AssertionError("rater calls did not reuse the exact frozen SamplingParams")
    output = project / "process/results/xlpt/b0-rater/qwen3-8b/seed-20260811.jsonl"
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if len(records) != 100:
        raise AssertionError(f"expected 100 trait rows, found {len(records)}")
    for index, record in enumerate(records):
        prefix = "E" if index % 2 == 0 else "A"
        if list(record["scores"]) != [f"{prefix}{item}" for item in range(1, 7)]:
            raise AssertionError("rubric item ordering changed")
        expected_scores = ([1, 2, 3, 4, 6, 7] if prefix == "E"
                           else [7, 7, 5, 4, 2, 1])
        expected_mean = (23.0 / 6.0 if prefix == "E" else 26.0 / 6.0)
        if (list(record["scores"].values()) != expected_scores
                or record["trait_score"] != expected_mean):
            raise AssertionError("six-item aggregation changed")
        if record["rater_calls"] != 600 or record["rater_calls_per_second"] != 150.0:
            raise AssertionError("call totals or rate fields changed")
        if record["rater_runtime_manifest_sha256"] != validation.get("runtime"):
            raise AssertionError("runtime closure hash is missing")
        if record["source_shard_sha256"] != hashlib.sha256(source_path.read_bytes()).hexdigest():
            raise AssertionError("source shard digest is missing")
        frozen = validation.get("frozen")
        if frozen is None or record["prompt_config_sha256"] != frozen[1] or record["freeze_receipt_sha256"] != frozen[2]:
            raise AssertionError("frozen input hashes are missing")
        if record["rater_decoding"] != {
            "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "max_new_tokens_per_item": 8, "seed": 20260911,
            "batch_size": 1, "cross_request_batching": False,
        }:
            raise AssertionError("output decoding record changed")

print("PASS: launch-owned rater integration, ordering, aggregation, calls, and fields")
