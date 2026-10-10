#!/usr/bin/env python3
"""CPU contracts for the Python-owned generation and scoring mathematics."""

from __future__ import annotations

import math

import torch

from inference_contract import (
    autoregressive_generate_ids,
    gumbel_categorical,
    greedy_token,
    measured_rate,
    nucleus_probabilities,
    qwen_eager_attention_reference,
    seeded_categorical,
    substream_seed,
    temperature_probabilities,
    trait_score,
    validate_qwen_runtime,
)
from qwen3_captured import CapturedQwen3ForCausalLM, causal_attention_mask


def close(left, right, tol=1e-7):
    if not torch.allclose(left, right, atol=tol, rtol=tol):
        raise AssertionError(f"{left} != {right}")


logits = torch.tensor([[0.0, math.log(2.0), math.log(4.0)]], dtype=torch.float64)
prob = temperature_probabilities(logits, 1.0)
close(prob, torch.tensor([[1 / 7, 2 / 7, 4 / 7]], dtype=torch.float32))
nonunit = temperature_probabilities(torch.tensor([[0.0, math.log(4.0)]]), 2.0)
close(nonunit, torch.tensor([[1 / 3, 2 / 3]], dtype=torch.float32))

nucleus = nucleus_probabilities(prob, 0.70)
close(nucleus, torch.tensor([[0.0, 1 / 3, 2 / 3]], dtype=torch.float32))
if greedy_token(logits).tolist() != [2]:
    raise AssertionError("greedy argmax contract failed")

g1 = torch.Generator(device="cpu").manual_seed(17)
g2 = torch.Generator(device="cpu").manual_seed(17)
draw1 = seeded_categorical(nucleus, g1)
draw2 = seeded_categorical(nucleus, g2)
if not torch.equal(draw1, draw2) or int(draw1.item()) not in (1, 2):
    raise AssertionError("seeded categorical contract failed")

controlled_prob = torch.tensor([[0.25, 0.75, 0.0], [0.5, 0.5, 0.0]])
controlled_uniform = torch.tensor([[0.5, 0.5, 0.9], [0.9, 0.1, 0.9]])
if gumbel_categorical(controlled_prob, controlled_uniform).tolist() != [1, 0]:
    raise AssertionError("controlled-uniform Gumbel oracle failed")
for row in range(2):
    hand_scores = []
    for probability, uniform in zip(controlled_prob[row].tolist(), controlled_uniform[row].tolist()):
        hand_scores.append(-math.inf if probability == 0 else math.log(probability) - math.log(-math.log(uniform)))
    if hand_scores.index(max(hand_scores)) != [1, 0][row]:
        raise AssertionError("hand-computed Gumbel oracle is inconsistent")

if substream_seed(20260811, 0, 0) == substream_seed(20260812, 0, 0):
    raise AssertionError("distinct root seeds must produce distinct substreams")
if len({substream_seed(20260811, 7, turn) for turn in range(3)}) != 3:
    raise AssertionError("turn substreams must be distinct")

if trait_score(torch.tensor([1, 2, 3, 4, 5, 6])).item() != 3.5:
    raise AssertionError("six-item equal-weight mean failed")
if measured_rate(12, 3.0) != 4.0:
    raise AssertionError("throughput formula failed")
for bad in ([1, 2, 3], [1, 2, 3, 4, 5, 8]):
    try:
        trait_score(torch.tensor(bad))
    except ValueError:
        pass
    else:
        raise AssertionError("trait-score guard failed")

from transformers import Qwen3Config, Qwen3ForCausalLM

torch.manual_seed(123)
config = Qwen3Config(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=4,
    max_position_embeddings=64,
    eos_token_id=31,
    pad_token_id=0,
)
model = Qwen3ForCausalLM(config).eval()
ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
mask = torch.ones_like(ids)
captured_config = {
    "hidden_size": config.hidden_size,
    "intermediate_size": config.intermediate_size,
    "num_hidden_layers": config.num_hidden_layers,
    "num_attention_heads": config.num_attention_heads,
    "num_key_value_heads": config.num_key_value_heads,
    "head_dim": config.head_dim,
    "rms_norm_eps": config.rms_norm_eps,
    "rope_theta": config.rope_theta,
}
captured = CapturedQwen3ForCausalLM.from_state_dict(captured_config, model.state_dict())
with torch.inference_mode():
    expected_logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    actual = captured(input_ids=ids, attention_mask=mask, past_key_values=None,
                      use_cache=True, return_dict=True, last_token_only=False)
close(actual.logits, expected_logits, tol=2e-5)
prefill = captured(input_ids=ids[:, :2], attention_mask=mask[:, :2], past_key_values=None,
                   use_cache=True, return_dict=True)
decode = captured(input_ids=ids[:, 2:], attention_mask=mask, past_key_values=prefill.past_key_values,
                  use_cache=True, return_dict=True)
close(decode.logits[:, -1, :], expected_logits[:, -1, :], tol=2e-5)
if any(key.shape[2] != 3 or value.shape[2] != 3 for key, value in decode.past_key_values):
    raise AssertionError("captured Qwen cache did not grow to the exact full sequence length")
causal = causal_attention_mask(mask, query_length=3, past_length=0, dtype=torch.float32, device="cpu")
if causal[0, 0, 0, 1] >= -1e20 or causal[0, 0, 2, 1] != 0:
    raise AssertionError("nonzero causal mask does not hide future keys and preserve prior visibility")
first = autoregressive_generate_ids(
    model, ids, mask, temperature=0.7, top_p=0.95, max_new_tokens=4, seed=99, eos_token_ids={31}
)
second = autoregressive_generate_ids(
    model, ids, mask, temperature=0.7, top_p=0.95, max_new_tokens=4, seed=99, eos_token_ids={31}
)
if first != second or not 1 <= len(first) <= 4 or any(not 0 <= token < 32 for token in first):
    raise AssertionError("tiny-Qwen autoregressive contract failed")


class ForcedTokenModel:
    device = torch.device("cpu")

    def __init__(self, token):
        self.token = token
        self.calls = []

    def __call__(self, input_ids, attention_mask, past_key_values, use_cache, return_dict):
        self.calls.append({"ids": input_ids.clone(), "mask": attention_mask.clone(), "past": past_key_values})
        forced = torch.full((1, 1, 8), -torch.inf)
        forced[0, 0, self.token] = 0.0
        return __import__("types").SimpleNamespace(logits=forced, past_key_values=(len(self.calls),))


eos_model = ForcedTokenModel(7)
eos_result = autoregressive_generate_ids(
    eos_model, ids[:, :2], mask[:, :2], temperature=0.7, top_p=0.95,
    max_new_tokens=4, seed=11, eos_token_ids={7},
)
if eos_result != [7] or len(eos_model.calls) != 1:
    raise AssertionError("forced EOS must stop after exactly one forward call")
continuing = ForcedTokenModel(4)
continued_result = autoregressive_generate_ids(
    continuing, ids[:, :2], mask[:, :2], temperature=0.7, top_p=0.95,
    max_new_tokens=3, seed=11, eos_token_ids={7},
)
if continued_result != [4, 4, 4] or len(continuing.calls) != 3:
    raise AssertionError("non-EOS recurrence must execute exactly max_new_tokens calls")
if continuing.calls[1]["ids"].tolist() != [[4]] or continuing.calls[2]["ids"].tolist() != [[4]]:
    raise AssertionError("sampled token was not used as the exact next input")
if continuing.calls[1]["past"] != (1,) or continuing.calls[2]["past"] != (2,):
    raise AssertionError("autoregressive recurrence did not reuse the exact prior cache")
if continuing.calls[0]["mask"].tolist() != [[1, 1]] or continuing.calls[2]["mask"].tolist() != [[1, 1, 1, 1]]:
    raise AssertionError("autoregressive mask did not preserve prior values and append one visibility")
try:
    autoregressive_generate_ids(
        ForcedTokenModel(4), ids[:, :2].expand(2, -1), mask[:, :2].expand(2, -1),
        temperature=0.7, top_p=0.95, max_new_tokens=1, seed=11, eos_token_ids={7},
    )
except ValueError:
    pass
else:
    raise AssertionError("reviewed generator must reject a batch larger than one")

from types import SimpleNamespace
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

torch.manual_seed(7)
query = torch.randn(1, 2, 3, 4)
key = torch.randn(1, 2, 3, 4)
value = torch.randn(1, 2, 3, 4)
mask = torch.zeros(1, 1, 3, 3)
module = SimpleNamespace(num_key_value_groups=1, training=False)
actual_output, actual_weights = eager_attention_forward(module, query, key, value, mask, scaling=0.5)
reference_output, reference_weights = qwen_eager_attention_reference(query, key, value, mask, scaling=0.5)
close(actual_output, reference_output)
close(actual_weights, reference_weights)

runtime_sha = validate_qwen_runtime(
    __import__("pathlib").Path("process/consumption/deployment/qwen3-runtime-manifest.json"),
    __import__("pathlib").Path("<LOCAL_PATH>/Qwen__Qwen3-8B"),
)
if len(runtime_sha) != 64:
    raise AssertionError("runtime dependency manifest was not bound")

print("PASS: English generation/scoring inference contracts")
