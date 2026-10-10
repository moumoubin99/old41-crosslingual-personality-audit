#!/usr/bin/env python3
"""Fixture integration test through the actual generation entrypoint helpers."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from run_transformers_generation import canonical_hash, generate_scenario, parser, run


class Encoded(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class FixtureTokenizer:
    eos_token_id = 5

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{row['role']}:{row['content']}" for row in messages)

    def __call__(self, prompt, return_tensors, add_special_tokens):
        length = max(2, min(12, len(prompt.split())))
        ids = torch.arange(1, length + 1, dtype=torch.long)[None, :] % 15
        return Encoded(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, token_ids, skip_special_tokens):
        return " ".join(f"tok{x}" for x in token_ids)


class FixtureModel:
    device = torch.device("cpu")

    def __init__(self):
        self.mask_lengths = []
        self.past_seen = []

    def __call__(self, input_ids, attention_mask, past_key_values, use_cache, return_dict):
        self.mask_lengths.append(int(attention_mask.shape[1]))
        self.past_seen.append(past_key_values is not None)
        logits = torch.full((1, input_ids.shape[1], 16), -4.0)
        logits[:, -1, 5] = 4.0
        return SimpleNamespace(logits=logits, past_key_values=("fixture-cache", len(self.mask_lengths)))


scenario = {
    "scenario_id": "fixture-scenario",
    "context": "A frozen account fact.",
    "customer_turns": ["First request.", "Second request.", "Third request."],
}
model = FixtureModel()
result = generate_scenario(
    model,
    FixtureTokenizer(),
    "Frozen neutral instruction.",
    scenario,
    root_seed=20260811,
    scenario_index=4,
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=2,
    max_model_len=128,
    eos_ids=set(),
)
if result["generated_call_count"] != 3 or len(result["agent_turns"]) != 3:
    raise AssertionError("entrypoint did not execute all three frozen turns")
if result["substream_seeds"] != [202608110012, 202608110013, 202608110014]:
    raise AssertionError(f"unexpected substream map: {result['substream_seeds']}")
if len(model.mask_lengths) != 6 or not any(model.past_seen):
    raise AssertionError("autoregressive recurrence/cache path was not exercised")
for first, second in zip(model.mask_lengths[::2], model.mask_lengths[1::2]):
    if second != first + 1:
        raise AssertionError("attention mask did not grow by one after a sampled token")
if len(set(result["prompt_hashes"])) != 3:
    raise AssertionError("per-turn prompts were not independently rendered and hashed")


with tempfile.TemporaryDirectory(prefix="xlpt-entrypoint-") as directory:
    project = Path(directory)
    frozen = project / "process/consumption/frozen"
    frozen.mkdir(parents=True)
    model_path = project / "model"
    model_path.mkdir()
    config = {
        "status": "FROZEN",
        "settings": {"NEUTRAL": "Frozen neutral instruction.",
                     "COMPLIANCE_CONTROL": "Frozen compliance instruction."},
    }
    config["self_sha256"] = canonical_hash(config)
    config_path = frozen / "english_execution_v1.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rows = [{
        "scenario_id": f"fixture-{index:02d}",
        "prereg_partition": "qualification",
        "context": "A frozen account fact.",
        "customer_turns": ["First request.", "Second request.", "Third request."],
    } for index in range(50)]
    scenarios = frozen / "english_b0_competence_scenarios_v1.jsonl"
    scenarios.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    receipt = {
        "status": "FROZEN",
        "files": {"process/consumption/frozen/english_b0_competence_scenarios_v1.jsonl":
                  hashlib.sha256(scenarios.read_bytes()).hexdigest()},
        "prompt_config_self_sha256": config["self_sha256"],
    }
    receipt["self_sha256"] = canonical_hash(receipt)
    (frozen / "english_execution_receipt_v1.json").write_text(json.dumps(receipt), encoding="utf-8")
    args = parser().parse_args([
        "--project", str(project),
        "--phase", "b0-competence",
        "--model-path", str(model_path),
        "--seed", "20260811",
        "--max-new-tokens", "2",
    ])
    launcher_model = FixtureModel()
    exit_code = run(
        args,
        runtime_factory=lambda unused: (FixtureTokenizer(), launcher_model),
        runtime_validator=lambda manifest, model: "0" * 64,
    )
    output = project / "process/results/xlpt/b0-competence/qwen3-8b/seed-20260811.jsonl"
    output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if exit_code != 0 or len(output_rows) != 50 or len(launcher_model.mask_lengths) != 150:
        raise AssertionError("parser/run launcher fixture did not execute the exact 50x3 B0 unit")
    if any(row["agent_turns"] != ["tok5", "tok5", "tok5"] for row in output_rows):
        raise AssertionError("launcher fixture did not preserve forced-EOS outputs")

print("PASS: generation entrypoint fixture integration")
