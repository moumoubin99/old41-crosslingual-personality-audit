#!/usr/bin/env python3
"""B2-specific closure for request throughput, masking, and tokenization."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from .inference_contract import measured_rate, validate_qwen_runtime
from .qwen3_captured import causal_attention_mask
from .run_transformers_generation import generate_scenario


# Exercise visibility and causality together. Key 1 is invisible even after it
# becomes causally reachable.
visible = torch.tensor([[1, 0, 1]], dtype=torch.long)
actual_mask = causal_attention_mask(
    visible,
    query_length=3,
    past_length=0,
    dtype=torch.float32,
    device=torch.device("cpu"),
)
minimum = torch.finfo(torch.float32).min
expected_mask = torch.tensor(
    [[[[0.0, minimum, minimum],
       [0.0, minimum, minimum],
       [0.0, minimum, 0.0]]]],
    dtype=torch.float32,
)
if not torch.equal(actual_mask, expected_mask):
    raise AssertionError(f"causal/visibility mask changed: {actual_mask}")


class Encoded(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class TwoStepTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{row['role']}:{row['content']}" for row in messages)

    def __call__(self, prompt, return_tensors, add_special_tokens):
        ids = torch.tensor([[11, 12, 13]], dtype=torch.long)
        return Encoded(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, token_ids, skip_special_tokens):
        return " ".join(str(value) for value in token_ids)


class TwoStepModel:
    device = torch.device("cpu")

    def __init__(self):
        self.forward_steps = 0

    def __call__(self, input_ids, attention_mask, past_key_values, use_cache, return_dict):
        self.forward_steps += 1
        selected = 7 if past_key_values is None else 2
        logits = torch.full((1, input_ids.shape[1], 16), -1000.0)
        logits[:, -1, selected] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=("two-step-cache",))


scenario = {
    "scenario_id": "b2-throughput-contract",
    "context": "A frozen account fact.",
    "customer_turns": ["First.", "Second.", "Third."],
}
fixture_model = TwoStepModel()
generated = generate_scenario(
    fixture_model,
    TwoStepTokenizer(),
    "Frozen compliance instruction.",
    scenario,
    root_seed=20260911,
    scenario_index=0,
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=4,
    max_model_len=128,
    eos_ids={2},
)
if generated["generated_call_count"] != 3 or fixture_model.forward_steps != 6:
    raise AssertionError("request count must stay distinct from autoregressive forward-step count")
if measured_rate(generated["generated_call_count"], 2.0) != 1.5:
    raise AssertionError("generation-request throughput numerator changed")
if measured_rate(fixture_model.forward_steps, 2.0) == 1.5:
    raise AssertionError("request-throughput test did not separate the two numerators")


# Load the real local tokenizer and bind a fixed chat rendering to exact IDs.
if importlib.metadata.version("tokenizers") != "0.22.2":
    raise AssertionError("tokenizers package version changed")
from transformers import AutoTokenizer

model_path = Path(os.environ.get(
    "XLPT_QWEN_MODEL_PATH",
    "<LOCAL_PATH>/Qwen__Qwen3-8B",
)).resolve()
runtime_manifest = Path(
    "process/consumption/deployment/qwen3-b2-runtime-manifest.json"
).resolve()
runtime_sha = validate_qwen_runtime(runtime_manifest, model_path)
if len(runtime_sha) != 64:
    raise AssertionError("exact B2 runtime manifest did not produce a SHA-256 binding")

# Fail closed on environment drift and checkpoint/tokenizer-artifact drift. The
# artifact case corrupts the first small-file expectation, avoiding a second
# weight-shard scan before demonstrating refusal.
with tempfile.TemporaryDirectory(prefix="xlpt-b2-runtime-negative-") as temporary:
    temporary_root = Path(temporary)
    payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    bad_package = json.loads(json.dumps(payload))
    bad_package["package_versions"]["tokenizers"] = "0.0.0-invalid"
    bad_package_path = temporary_root / "bad-package.json"
    bad_package_path.write_text(json.dumps(bad_package), encoding="utf-8")
    try:
        validate_qwen_runtime(bad_package_path, model_path)
    except ValueError as exc:
        if "runtime package version mismatch: tokenizers" not in str(exc):
            raise
    else:
        raise AssertionError("altered tokenizers package version was accepted")

    bad_artifact = json.loads(json.dumps(payload))
    first_artifact = next(iter(bad_artifact["model_files"]))
    bad_artifact["model_files"][first_artifact] = "0" * 64
    bad_artifact_path = temporary_root / "bad-artifact.json"
    bad_artifact_path.write_text(json.dumps(bad_artifact), encoding="utf-8")
    try:
        validate_qwen_runtime(bad_artifact_path, model_path)
    except ValueError as exc:
        if "checkpoint/tokenizer artifact changed" not in str(exc):
            raise
    else:
        raise AssertionError("altered artifact hash was accepted")

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path), trust_remote_code=False, local_files_only=True,
)
messages = [
    {"role": "system", "content": "Tokenizer closure test."},
    {"role": "user", "content": "Return exactly OK."},
]
try:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
except TypeError:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
token_ids = tokenizer(rendered, add_special_tokens=False).input_ids
expected_ids = [
    151644, 8948, 198, 37434, 21955, 1273, 13, 151645, 198,
    151644, 872, 198, 5598, 6896, 10402, 13, 151645, 198,
    151644, 77091, 198, 151667, 271, 151668, 271,
]
if token_ids != expected_ids:
    raise AssertionError(f"real Qwen tokenizer rendering changed: {token_ids}")

print("PASS: B2 request throughput, invisible-key mask, and real tokenizer closure")
