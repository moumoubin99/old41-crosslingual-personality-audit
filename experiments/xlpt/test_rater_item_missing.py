#!/usr/bin/env python3
"""Contract for run_rater_item_missing.run (parse policy strict_item_missing_v1).

Run from the project (or deployment) root: python3 -m experiments.xlpt.test_rater_item_missing

* the frozen parser and every frozen launch input are reused unchanged;
* an unparseable item is null with its raw text, parser input, error and token ids, never retried;
* a trait score needs 6/6 items and otherwise is null;
* canonical and rescue shards land at fixed paths, and a cell that already has either refuses
  a second scoring.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import run_rater_item_missing as missing
from . import run_rater_scoring as frozen
from .rater_frozen_inputs import validate_rater_frozen_inputs
from .summarize_pole_rater import check_item_missing_evidence


ROOT = Path.cwd().resolve()
SEED = 20260811
FAILURES = {0: "�7", 13: "seven", 50: " �4 "}
RUNTIME_SHA = "0" * 64


# 1. The parser, the decoding contract and the shared helpers are the frozen objects.
if (missing.parse_item_score is not frozen.parse_item_score
        or frozen.INTEGER.pattern != r"\s*([1-7])\s*"
        or missing.rater_decoding_contract is not frozen.rater_decoding_contract
        or missing.build_item_request is not frozen.build_item_request
        or missing.validate_context_window is not frozen.validate_context_window
        or missing.load_rater_chat_template is not frozen.load_rater_chat_template
        or missing.SOURCE_PHASES is not frozen.SOURCE_PHASES):
    raise AssertionError("the item-missing scorer does not reuse the frozen scoring objects")
if missing.score_item(" 5\n", [22], "E1") != (5, None):
    raise AssertionError("a valid item changed its parse")
score, failure = missing.score_item("�7", [1, 29555], "E1")
if score is not None or failure != {
    "item_id": "E1", "raw_decoded": "�7", "parser_input": "�7",
    "error": f"rater must return exactly one integer 1..7, got {chr(0xFFFD) + '7'!r}",
    "token_ids": [1, 29555],
}:
    raise AssertionError(f"parse failure record changed: {failure}")
if missing.complete_trait_score([1, 2, 3, 4, 6, 7]) != 23.0 / 6.0:
    raise AssertionError("complete trait mean changed")
if missing.complete_trait_score([None, 2, 3, 4, 6, 7]) is not None:
    raise AssertionError("an incomplete trait produced a score")


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, chat_template, tokenize, add_generation_prompt):
        if tokenize is not True or add_generation_prompt is not True or "[INST]" not in chat_template:
            raise AssertionError("chat template flags changed")
        return [1, 3, 4]


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.calls = 0
        self.__class__.instances.append(self)

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, params, use_tqdm):
        if len(prompts) != 1 or use_tqdm is not False or prompts != [{"prompt_token_ids": [1, 3, 4]}]:
            raise AssertionError("rater calls must remain independent batch-one calls")
        index = self.calls
        self.calls += 1
        text = FAILURES.get(index, str(index % 7 + 1))
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[9, index])])]


def binding(project, deployment_root):
    frozen.validate_project_binding(frozen.EXTERNAL_PROJECT_ROOT, deployment_root)


def runtime(manifest_path, model_path, deployment_root):
    return RUNTIME_SHA


def frozen_inputs(project, prompt_config, freeze_receipt):
    return validate_rater_frozen_inputs(ROOT, prompt_config, freeze_receipt)


def score(project, namespace):
    args = SimpleNamespace(
        project=str(project), seed=SEED, namespace=namespace,
        rater_path=str(project / "rater"),
        prompt_config="process/consumption/frozen/english_execution_v1.json",
        freeze_receipt="process/consumption/frozen/english_execution_receipt_v1.json",
        runtime_manifest="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json",
        source_checkpoint="qwen3-8b", source_phase="b0-competence",
        gpu_memory_utilization=0.88,
    )
    ticks = iter((10.0, 14.0))
    return missing.run(args, llm_cls=FakeLLM, sampling_params_cls=FakeSamplingParams,
                       frozen_validator=frozen_inputs, runtime_validator=runtime,
                       project_binding_validator=binding, clock=lambda: next(ticks))


def refuses(project, namespace, reason):
    try:
        score(project, namespace)
    except ValueError:
        return
    raise AssertionError(reason)


def make_project(temporary: str) -> Path:
    project = Path(temporary)
    for relative in (
        "refine-logs/RUBRIC_BEHAVIOR.md",
        "process/consumption/frozen/english_execution_v1.json",
        "process/consumption/frozen/english_execution_receipt_v1.json",
        "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl",
        "process/consumption/deployment/qwen3-runtime-manifest.json",
    ):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    registered = [json.loads(line) for line in (
        project / "process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    config = json.loads((project / "process/consumption/frozen/english_execution_v1.json").read_text())
    receipt = json.loads((project / "process/consumption/frozen/english_execution_receipt_v1.json").read_text())
    runtime_path = project / "process/consumption/deployment/qwen3-runtime-manifest.json"
    runtime_manifest = json.loads(runtime_path.read_text())
    rows = []
    for index, scenario in enumerate(registered):
        rows.append({
            "schema_version": 2, "phase": "b0-competence", "partition": "QUALIFICATION",
            "eligible_for_gate": True, "eligible_for_analysis": False,
            "surface_scope": "English-only; no BFI-2/TIPI item text",
            "scenario_id": scenario["scenario_id"], "checkpoint_id": "Qwen3-8B",
            "model_path": "<LOCAL_PATH>/Qwen__Qwen3-8B",
            "persona_language": "English", "dialogue_language": "English",
            "setting": "NEUTRAL", "seed": SEED,
            "substream_seeds": [SEED * 10_000 + index * 3 + turn for turn in range(3)],
            "decoding": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 512, "attention": "eager"},
            "prompt_config_sha256": config["self_sha256"],
            "freeze_receipt_sha256": receipt["self_sha256"],
            "qwen_runtime_manifest_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            "prompt_hashes": [hashlib.sha256(f"{index}:{turn}".encode()).hexdigest() for turn in range(3)],
            "customer_turns": scenario["customer_turns"],
            "agent_turns": [f"agent-{index}-{turn}" for turn in range(3)],
            "prompt_tokens": 300 + index, "completion_tokens": 30 + index,
            "host": runtime_manifest["worker"], "python": runtime_manifest["interpreter"],
            "shard_elapsed_seconds": 600.0, "generated_call_count": 150, "requests_per_second": 0.25,
        })
    # A stand-in checkpoint template with the one wall-clock date call the frozen loader replaces.
    (project / "rater").mkdir()
    (project / "rater/chat_template.json").write_text(json.dumps(
        {"chat_template": '[INST] {{ strftime_now("%Y-%m-%d") }} {{ messages }} [/INST]'}
    ), encoding="utf-8")
    source = project / f"process/results/xlpt/b0-competence/qwen3-8b/seed-{SEED}.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return project


# 2. Canonical run: failures are recorded, never retried, and only 6/6 traits get a mean.
with tempfile.TemporaryDirectory(prefix="xlpt-item-missing-") as temporary:
    project = make_project(temporary)
    refuses(project, "primary", "an unregistered namespace was accepted")
    if score(project, "canonical") != 0:
        raise AssertionError("canonical scoring did not complete")
    engine = FakeLLM.instances[-1]
    if engine.calls != 600:
        raise AssertionError("a failed item was retried or skipped")
    if engine.kwargs["logits_processors"] != ["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"]:
        raise AssertionError("engine contract changed")
    paths = missing.score_paths(project, "b0-rater", "qwen3-8b", SEED)
    records = [json.loads(line) for line in paths["canonical"].read_text(encoding="utf-8").splitlines()]
    if len(records) != 100 or paths["rescue"].exists():
        raise AssertionError("canonical shard shape or location changed")
    scorer_sha = hashlib.sha256(Path(missing.__file__).read_bytes()).hexdigest()
    frozen_sha = hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest()
    failed_rows = {0: ["E1"], 2: ["E2"], 8: ["E3"]}
    for index, record in enumerate(records):
        if (record["schema_version"] != 3 or record["parse_policy"] != "strict_item_missing_v1"
                or record["score_namespace"] != "canonical" or record["rescue_run"] is not False
                or record["scorer_sha256"] != scorer_sha
                or record["rater_runtime_manifest_sha256"] != RUNTIME_SHA
                or record["rater_calls"] != 600):
            raise AssertionError("row provenance changed")
        # The rows the wrapper writes are exactly what the pole collector re-checks.
        check_item_missing_evidence(record, f"row {index}")
        if record["frozen_scorer_sha256"] != frozen_sha:
            raise AssertionError("frozen scorer digest is not the imported run_rater_scoring.py")
        expected_missing = failed_rows.get(index, [])
        nulls = [item for item, value in record["scores"].items() if value is None]
        if nulls != expected_missing or [f["item_id"] for f in record["parse_failures"]] != expected_missing:
            raise AssertionError(f"row {index}: missing items {nulls} != {expected_missing}")
        if record["items_parsed"] != 6 - len(expected_missing):
            raise AssertionError("items_parsed is wrong")
        if expected_missing:
            if record["trait_score"] is not None:
                raise AssertionError("an incomplete trait was averaged")
        else:
            ordered = list(record["scores"].values())
            if record["trait_score"] != float(frozen.trait_score(ordered).item()):
                raise AssertionError("a complete trait mean changed")
    last = records[8]["parse_failures"][0]
    if (last["raw_decoded"] != " �4 " or last["parser_input"] != "�4"
            or last["token_ids"] != [9, 50]):
        raise AssertionError("failure record lost the unstripped text or token ids")
    # 3. A scored cell refuses both namespaces afterwards.
    refuses(project, "canonical", "canonical shard was overwritten")
    refuses(project, "rescue", "a rescue shard was written beside a canonical one")

# 4. Rescue run: separate directory, flagged, and then the canonical path is closed too.
with tempfile.TemporaryDirectory(prefix="xlpt-item-missing-") as temporary:
    project = make_project(temporary)
    score(project, "rescue")
    paths = missing.score_paths(project, "b0-rater", "qwen3-8b", SEED)
    if paths["canonical"].exists() or not paths["rescue"].is_file():
        raise AssertionError("rescue shard location changed")
    records = [json.loads(line) for line in paths["rescue"].read_text(encoding="utf-8").splitlines()]
    if any(r["score_namespace"] != "rescue" or r["rescue_run"] is not True for r in records):
        raise AssertionError("rescue rows are not flagged")
    refuses(project, "canonical", "a canonical shard was written beside a rescue one")

# 5. Concurrent scorers: one per-cell lock covers both namespaces, publication never replaces.
with tempfile.TemporaryDirectory(prefix="xlpt-item-missing-") as temporary:
    project = make_project(temporary)
    paths = missing.score_paths(project, "b0-rater", "qwen3-8b", SEED)
    lock_path = missing.cell_lock_path(project, "b0-rater", "qwen3-8b", SEED)
    with missing.cell_lock(lock_path):
        calls_before = sum(engine.calls for engine in FakeLLM.instances)
        refuses(project, "canonical", "a canonical run started while another scorer held the cell")
        refuses(project, "rescue", "a rescue run started while another scorer held the cell")
        if paths["canonical"].exists() or paths["rescue"].exists():
            raise AssertionError("a locked-out run wrote a shard")
        if sum(engine.calls for engine in FakeLLM.instances) != calls_before:
            raise AssertionError("a locked-out run reached the rater")
    # The racing writer published first: the loser must fail and leave the winner's bytes intact.
    paths["canonical"].parent.mkdir(parents=True, exist_ok=True)
    paths["canonical"].write_text("winner\n", encoding="utf-8")
    try:
        missing.publish_no_replace(paths["canonical"], [{"loser": True}])
    except ValueError:
        pass
    else:
        raise AssertionError("publication replaced an existing shard")
    if paths["canonical"].read_text(encoding="utf-8") != "winner\n":
        raise AssertionError("the published shard was modified")
    if list(paths["canonical"].parent.glob("*.tmp-*")):
        raise AssertionError("a failed publication left a temp file")
    # Once released, the lock admits the next scorer; the existing shard still refuses it.
    refuses(project, "rescue", "a rescue shard was written beside a canonical one")

print("PASS: item-missing scorer reuses the frozen parser, records failures, needs 6/6, refuses collisions")
