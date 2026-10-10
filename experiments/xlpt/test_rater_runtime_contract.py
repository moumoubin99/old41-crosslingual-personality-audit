#!/usr/bin/env python3
"""Independent CPU contracts for the frozen Mistral/AWQ rater runtime."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .inference_contract import trait_score
from .rater_runtime import (
    ProjectArgmaxLogitsProcessor,
    absolute_cache_positions,
    append_key_value_cache,
    apply_rotary,
    attention_value,
    awq_linear,
    awq_dequantize,
    build_rope_tables,
    cache_aware_causal_mask,
    cache_next_length,
    embedding_lookup,
    eos_stop,
    final_logits,
    gated_silu,
    greedy_token,
    linear_projection,
    measured_rate,
    repeat_key_value,
    residual_update,
    rms_norm,
    scaled_attention_probabilities,
    single_rank_logits,
    split_fused_qkv,
    token_budget_stop,
    validate_runtime_sources,
    write_cache_slots,
)
from .run_rater_scoring import parse_item_score


_AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


ROOT = Path.cwd().resolve()
MANIFEST = ROOT / "process/consumption/deployment/mistral-rater-runtime-v2-manifest.json"


def close(actual, expected, label, tolerance=1e-6):
    actual_values = actual.reshape(-1).tolist()
    expected_values = expected.reshape(-1).tolist() if torch.is_tensor(expected) else list(expected)
    if len(actual_values) != len(expected_values):
        raise AssertionError(f"{label}: length {len(actual_values)} != {len(expected_values)}")
    for index, (got, want) in enumerate(zip(actual_values, expected_values)):
        if abs(got - want) > tolerance:
            raise AssertionError(f"{label}[{index}]: {got} != {want}")


def pack_awq_row(values):
    packed = 0
    for output_index, shift_index in enumerate(_AWQ_UNPACK_ORDER):
        packed |= int(values[output_index]) << (4 * shift_index)
    return packed


checked = validate_runtime_sources(MANIFEST, ROOT)
if checked != (
    "vllm.model_executor.models.mistral3",
    "vllm.model_executor.models.llama",
    "vllm.model_executor.layers.quantization.awq",
    "vllm.v1.sample.sampler",
):
    raise AssertionError(f"runtime dependency closure changed: {checked}")

# AWQ unpack/dequantization: values are packed independently of the deployed helper.
q_values = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]]
z_values = [[1] * 8, [2] * 8]
qweight = torch.tensor([[pack_awq_row(row)] for row in q_values], dtype=torch.int32)
qzeros = torch.tensor([[pack_awq_row(row)] for row in z_values], dtype=torch.int32)
scales = torch.tensor([[1.0] * 8, [0.5] * 8], dtype=torch.float32)
expected_awq = torch.tensor(
    [[float(value - 1) for value in q_values[0]],
     [0.5 * float(value - 2) for value in q_values[1]]],
    dtype=torch.float32,
)
close(awq_dequantize(qweight, scales, qzeros, 1), expected_awq, "AWQ dequantization")
awq_input = torch.tensor([[2.0, -1.0]])
close(
    awq_linear(awq_input, qweight, scales, qzeros, 1),
    torch.matmul(awq_input, expected_awq),
    "AWQ projection",
)

# Embedding lookup is checked against explicit selected rows.
embedding = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
tokens = torch.tensor([[2, 0], [1, 2]])
expected_embedding = torch.tensor([[[7.0, 11.0], [1.0, 2.0]], [[3.0, 5.0], [7.0, 11.0]]])
close(embedding_lookup(tokens, embedding), expected_embedding, "embedding")
for invalid_tokens in (torch.tensor([-1]), torch.tensor([3])):
    try:
        embedding_lookup(invalid_tokens, embedding)
    except (IndexError, RuntimeError):
        pass
    else:
        raise AssertionError(f"invalid embedding identifier accepted: {invalid_tokens.tolist()}")

# Fused Q/K/V boundaries, reshape, and batch/head/sequence permutation use
# disjoint literal values for all three branches.
fused = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
split_query, split_key, split_value = split_fused_qkv(fused, 2, 1, 2)
close(
    split_query,
    torch.tensor([[[[0.0, 1.0], [8.0, 9.0]], [[2.0, 3.0], [10.0, 11.0]]]]),
    "fused query axes",
)
close(split_key, torch.tensor([[[[4.0, 5.0], [12.0, 13.0]]]]), "fused key axes")
close(split_value, torch.tensor([[[[6.0, 7.0], [14.0, 15.0]]]]), "fused value axes")

# RMSNorm expected values are scalar arithmetic, not another normalization call.
hidden = torch.tensor([[[3.0, 4.0]]])
weight = torch.tensor([2.0, 0.5])
denominator = math.sqrt((9.0 + 16.0) / 2.0 + 1e-5)
expected_norm = torch.tensor([[[6.0 / denominator, 2.0 / denominator]]])
close(rms_norm(hidden, weight, 1e-5), expected_norm, "RMSNorm")

# RoPE uses an independently written half rotation for two batches.
rope_value = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]], [[[5.0, 6.0, 7.0, 8.0]]]])
cosine = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0, 1.0]]]])
sine = torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]], [[[1.0, 0.0, 1.0, 0.0]]]])
expected_rope = torch.tensor([[[[1.0, -4.0, 3.0, 2.0]]], [[[-7.0, 6.0, 5.0, 8.0]]]])
close(apply_rotary(rope_value, cosine, sine), expected_rope, "RoPE")
absolute_positions = torch.tensor([0, 3], dtype=torch.int64)
rope_cosine, rope_sine = build_rope_tables(absolute_positions, 4, 10_000.0)
expected_phase = torch.tensor([[0.0, 0.0], [3.0, 0.03]], dtype=torch.float32)
close(rope_cosine[:, :2], torch.cos(expected_phase), "RoPE cosine construction")
close(rope_sine[:, :2], torch.sin(expected_phase), "RoPE sine construction")
close(rope_cosine[:, 2:], rope_cosine[:, :2], "RoPE cosine duplication")
close(rope_sine[:, 2:], rope_sine[:, :2], "RoPE sine duplication")

# GQA repetition and masked scaled attention use a pure-Python softmax oracle.
key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
value = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
repeated = repeat_key_value(key, 2)
close(repeated, torch.cat((key, key), dim=1), "GQA repetition")
query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
mask = torch.tensor([[[[0.0, -torch.inf], [0.0, 0.0]]]])
probability = scaled_attention_probabilities(query, key, mask, 1.0 / math.sqrt(2.0), 2)
oracle = []
for head in query[0].tolist():
    rows = []
    for row_index, qrow in enumerate(head):
        raw = []
        for key_index, krow in enumerate(key[0, 0].tolist()):
            raw.append(sum(a * b for a, b in zip(qrow, krow)) / math.sqrt(2.0)
                       if key_index <= row_index else float("-inf"))
        peak = max(value for value in raw if value != float("-inf"))
        weights = [0.0 if value == float("-inf") else math.exp(value - peak) for value in raw]
        total = sum(weights)
        rows.append([weight / total for weight in weights])
    oracle.append(rows)
expected_probability = torch.tensor([oracle])
close(probability, expected_probability, "attention probabilities")
expected_context = torch.matmul(expected_probability, torch.cat((value, value), dim=1))
close(attention_value(probability, value, 2), expected_context, "attention values")

# Prefill plus two decode positions exercise absolute cache positions, content
# append, dense paged writes, and cache-aware visibility independently.
positions = absolute_cache_positions(2, 2)
if not torch.equal(positions, torch.tensor([2, 3])):
    raise AssertionError("absolute cache positions changed")
past_key = torch.tensor([[[[1.0], [2.0]]]])
past_value = torch.tensor([[[[10.0], [20.0]]]])
current_key = torch.tensor([[[[3.0], [4.0]]]])
current_value = torch.tensor([[[[30.0], [40.0]]]])
appended_key, appended_value = append_key_value_cache(
    past_key, past_value, current_key, current_value
)
close(appended_key, torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]]), "key cache append")
close(appended_value, torch.tensor([[[[10.0], [20.0], [30.0], [40.0]]]]), "value cache append")
prior_storage = torch.tensor([[[[1.0], [2.0], [99.0], [99.0]]]])
keep_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
append_selector = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
close(
    write_cache_slots(prior_storage, current_key, keep_mask, append_selector),
    appended_key,
    "absolute cache writes",
)
causal = cache_aware_causal_mask(
    positions, torch.tensor([0, 1, 2, 3], dtype=torch.int64)
)
minimum = torch.finfo(torch.float32).min
close(
    causal,
    torch.tensor([[0.0, 0.0, 0.0, minimum], [0.0, 0.0, 0.0, 0.0]]),
    "cache-aware causal mask",
)

# Linear, residual, gated-SiLU, and final-head contracts use hand-built values.
x = torch.tensor([[1.0, 2.0]])
projection_weight = torch.tensor([[3.0, 4.0], [-1.0, 2.0]])
close(linear_projection(x, projection_weight), torch.tensor([[11.0, 3.0]]), "linear")
close(residual_update(x, torch.tensor([[0.5, -3.0]])), torch.tensor([[1.5, -1.0]]), "residual")
gate_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
up_weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
down_weight = torch.eye(2)
silu = lambda number: number / (1.0 + math.exp(-number))
expected_mlp = torch.tensor([[silu(1.0) * 2.0, silu(2.0) * 6.0]])
close(gated_silu(x, gate_weight, up_weight, down_weight), expected_mlp, "gated SiLU")
final_hidden = torch.tensor([[[3.0, 4.0]]])
final_norm_weight = torch.tensor([1.0, 1.0])
lm_head = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
base = math.sqrt(12.5 + 1e-5)
expected_logits = torch.tensor([[[3.0 / base, 4.0 / base, 7.0 / base]]])
close(final_logits(final_hidden, final_norm_weight, lm_head, 1e-5), expected_logits, "LM head")
close(
    single_rank_logits(torch.tensor([[2.0, 3.0]]),
                       torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [9.0, 9.0]]), 3),
    torch.tensor([[2.0, 3.0, 5.0]]),
    "single-rank vocabulary truncation",
)

if cache_next_length(17) != 18 or not eos_stop(2, [1, 2]) or eos_stop(3, [1, 2]):
    raise AssertionError("autoregressive cache/EOS contract changed")
if not token_budget_stop(8, 8) or token_budget_stop(7, 8):
    raise AssertionError("max_tokens stopping contract changed")
for invalid_counts in ((True, 8), (1.5, 8), (-1, 8), (0, 0)):
    try:
        token_budget_stop(*invalid_counts)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid token-budget counts accepted: {invalid_counts!r}")

for score in range(1, 8):
    if parse_item_score(f"  {score}\n") != score:
        raise AssertionError("valid strict item score changed")
for malformed_score in ("", "0", "8", "+1", "1.0", "1 2", '{"score": 1}', "score=1"):
    try:
        parse_item_score(malformed_score)
    except ValueError:
        pass
    else:
        raise AssertionError(f"malformed item score was accepted: {malformed_score!r}")

# Distinct maxima and ties across multiple batches establish first-index ties.
logits = torch.tensor([[1.0, 5.0, 2.0], [7.0, 7.0, 1.0], [-3.0, -2.0, -2.0]])
expected_tokens = torch.tensor([1, 0, 1])
if not torch.equal(greedy_token(logits), expected_tokens):
    raise AssertionError("project argmax or tie behavior changed")
processor = ProjectArgmaxLogitsProcessor(None, torch.device("cpu"), False)
forced = processor.apply(logits)
if processor.is_argmax_invariant() or not torch.equal(forced.argmax(dim=-1), expected_tokens):
    raise AssertionError("vLLM project-argmax adapter changed")

# The six-item mean is checked with asymmetric batched fixtures, an independent
# sum/6 oracle, a final-axis permutation, and a one-item perturbation.  This
# catches accidental flattening or reduction over a leading batch dimension.
trait_fixture = torch.tensor([
    [[1, 2, 3, 4, 6, 7], [7, 7, 5, 4, 2, 1]],
    [[2, 2, 2, 5, 6, 7], [6, 1, 4, 7, 3, 2]],
], dtype=torch.int64)
expected_trait = torch.tensor([
    [(1 + 2 + 3 + 4 + 6 + 7) / 6, (7 + 7 + 5 + 4 + 2 + 1) / 6],
    [(2 + 2 + 2 + 5 + 6 + 7) / 6, (6 + 1 + 4 + 7 + 3 + 2) / 6],
], dtype=torch.float64)
actual_trait = trait_score(trait_fixture)
if actual_trait.shape != (2, 2) or not torch.equal(actual_trait, expected_trait):
    raise AssertionError("batched final-axis six-item trait mean changed")
permuted = trait_fixture[..., torch.tensor([5, 2, 0, 4, 1, 3])]
if not torch.equal(trait_score(permuted), expected_trait):
    raise AssertionError("trait mean is not invariant to item permutation")
perturbed = trait_fixture.clone()
perturbed[1, 0, 3] += 1
delta = trait_score(perturbed) - actual_trait
if (not math.isclose(delta[1, 0].item(), 1.0 / 6.0, rel_tol=0.0, abs_tol=1e-12)
        or torch.count_nonzero(delta).item() != 1):
    raise AssertionError("one-item trait perturbation no longer contributes exactly one sixth")

if measured_rate(12, 3.0) != 4.0:
    raise AssertionError("throughput formula changed")
for invalid_calls in (True, -1, 1.5, "12", None):
    try:
        measured_rate(invalid_calls, 3.0)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid completed_calls was accepted: {invalid_calls!r}")
for invalid_elapsed in (True, "3", None, 0.0, -1.0, float("inf"), float("-inf"), float("nan")):
    try:
        measured_rate(12, invalid_elapsed)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid elapsed_seconds was accepted: {invalid_elapsed!r}")

print("PASS: Mistral/AWQ runtime, project argmax, aggregation, and rate contracts")
