#!/usr/bin/env python3
"""Executable contract for deployment unit U3 - the decoder stack and decode loop.

This file answers the deployment review that rejected the earlier identity
assertion: the expected logits are produced by a *second, independent*
implementation of the whole forward pass, written here in plain Python lists
and `math` with no torch operation and no call into `qwen3_reference`. Only the
parameters are shared, because parameters are inputs, not the thing under test.

Covered: the residual decoder layer, the layer stack, the final normalization
and vocabulary projection, causal-prefix invariance of the logits, agreement
between cached incremental decoding and full-prefix evaluation, and greedy
determinism.
"""

from __future__ import annotations

import math

import torch

from qwen3_reference import (
    FIXTURE_CONFIG,
    fixture_params,
    generate,
    greedy_next_token,
    head_dim_of,
    model_logits,
    validate_config,
)

SEED = 17091701
LOGIT_TOL = 1e-8


# --------------------------------------------------------------------------
# Independent pure-Python oracle
# --------------------------------------------------------------------------


def matvec(weight, vector):
    return [sum(w * v for w, v in zip(row, vector)) for row in weight]


def oracle_rms(vector, gain, eps):
    mean_square = sum(value * value for value in vector) / len(vector)
    scale = 1.0 / math.sqrt(mean_square + eps)
    return [value * scale * g for value, g in zip(vector, gain)]


def oracle_rope(vector, position, theta):
    width = len(vector)
    half = width // 2
    out = [0.0] * width
    for j in range(half):
        angle = position * theta ** (-(2.0 * j) / width)
        cos, sin = math.cos(angle), math.sin(angle)
        out[j] = vector[j] * cos - vector[j + half] * sin
        out[j + half] = vector[j + half] * cos + vector[j] * sin
    return out


def oracle_softmax(scores):
    peak = max(scores)
    weights = [math.exp(s - peak) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def oracle_silu(value):
    return value / (1.0 + math.exp(-value))


def oracle_forward(params, config, token_ids):
    """Vocabulary logits for every position, computed without torch."""
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = head_dim_of(config)
    eps = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    n_rep = heads // kv_heads

    embed = params["embed"].tolist()
    hidden = [list(embed[token]) for token in token_ids]
    length = len(hidden)

    for layer in params["layers"]:
        weights = {key: (None if value is None else value.tolist()) for key, value in layer.items()}
        normed = [oracle_rms(row, weights["input_norm"], eps) for row in hidden]

        queries, keys, values = [], [], []
        for position, row in enumerate(normed):
            flat_q = matvec(weights["q_proj"], row)
            flat_k = matvec(weights["k_proj"], row)
            flat_v = matvec(weights["v_proj"], row)
            per_q, per_k, per_v = [], [], []
            for head in range(heads):
                vector = flat_q[head * head_dim : (head + 1) * head_dim]
                if weights["q_norm"] is not None:
                    vector = oracle_rms(vector, weights["q_norm"], eps)
                per_q.append(oracle_rope(vector, position, theta))
            for head in range(kv_heads):
                vector = flat_k[head * head_dim : (head + 1) * head_dim]
                if weights["k_norm"] is not None:
                    vector = oracle_rms(vector, weights["k_norm"], eps)
                per_k.append(oracle_rope(vector, position, theta))
                per_v.append(flat_v[head * head_dim : (head + 1) * head_dim])
            queries.append(per_q)
            keys.append(per_k)
            values.append(per_v)

        scale = 1.0 / math.sqrt(head_dim)
        contexts = []
        for position in range(length):
            joined = []
            for head in range(heads):
                source = head // n_rep
                query = queries[position][head]
                scores = [
                    sum(a * b for a, b in zip(query, keys[past][source])) * scale
                    for past in range(position + 1)
                ]
                attention = oracle_softmax(scores)
                joined.extend([
                    sum(w * values[past][source][d] for past, w in enumerate(attention))
                    for d in range(head_dim)
                ])
            contexts.append(joined)

        for position in range(length):
            projected = matvec(weights["o_proj"], contexts[position])
            hidden[position] = [h + p for h, p in zip(hidden[position], projected)]

        for position in range(length):
            row = oracle_rms(hidden[position], weights["post_attn_norm"], eps)
            gate = matvec(weights["gate_proj"], row)
            up = matvec(weights["up_proj"], row)
            activated = [oracle_silu(g) * u for g, u in zip(gate, up)]
            projected = matvec(weights["down_proj"], activated)
            hidden[position] = [h + p for h, p in zip(hidden[position], projected)]

    final_gain = params["final_norm"].tolist()
    head_weight = params["lm_head"].tolist()
    return [matvec(head_weight, oracle_rms(row, final_gain, eps)) for row in hidden]


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


def max_difference(tensor, nested):
    flat = tensor.reshape(-1).tolist()
    expected = [value for row in nested for value in row]
    if len(flat) != len(expected):
        raise AssertionError(f"{len(flat)} logits, expected {len(expected)}")
    return max(abs(a - b) for a, b in zip(flat, expected))


def build():
    config = dict(FIXTURE_CONFIG)
    validate_config(config)
    params = fixture_params(config, SEED, dtype=torch.float64)
    return config, params


def check_fixture_is_reproducible(config):
    first = fixture_params(config, SEED, dtype=torch.float64)
    second = fixture_params(config, SEED, dtype=torch.float64)
    if not torch.equal(first["embed"], second["embed"]):
        raise AssertionError("fixture_params is not reproducible for a fixed seed")
    other = fixture_params(config, SEED + 1, dtype=torch.float64)
    if torch.equal(first["embed"], other["embed"]):
        raise AssertionError("fixture_params ignores the seed")
    for name, tensor in (("embed", first["embed"]), ("lm_head", first["lm_head"])):
        if float(tensor.abs().max()) > 0.2:
            raise AssertionError(f"fixture {name} exceeds its declared draw bound")


def check_logits_against_oracle(config, params):
    tokens = [3, 7, 0, 5, 2]
    logits, cache = model_logits(params, config, tokens)
    if tuple(logits.shape) != (len(tokens), int(config["vocab_size"])):
        raise AssertionError(f"logits shape {tuple(logits.shape)}")
    if len(cache) != int(config["num_hidden_layers"]):
        raise AssertionError(f"cache holds {len(cache)} layers")
    for key, value in cache:
        expected = (int(config["num_key_value_heads"]), len(tokens), head_dim_of(config))
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise AssertionError(f"cache entry shape {tuple(key.shape)}, expected {expected}")

    difference = max_difference(logits, oracle_forward(params, config, tokens))
    if difference > LOGIT_TOL:
        raise AssertionError(
            f"deployed logits differ from the independent oracle by {difference}")

    # The oracle is not trivially reproducing the implementation: a perturbed
    # token sequence must move both, and by the same amount.
    moved = [3, 7, 1, 5, 2]
    moved_logits, _ = model_logits(params, config, moved)
    if float((moved_logits - logits).abs().max()) < 1e-6:
        raise AssertionError("changing an input token did not change the logits")
    if max_difference(moved_logits, oracle_forward(params, config, moved)) > LOGIT_TOL:
        raise AssertionError("the oracle disagrees on the perturbed sequence")


def check_causal_prefix_invariance(config, params):
    tokens = [3, 7, 0, 5, 2]
    full, _ = model_logits(params, config, tokens)
    for length in range(1, len(tokens) + 1):
        prefix, _ = model_logits(params, config, tokens[:length])
        difference = float((prefix[-1] - full[length - 1]).abs().max())
        if difference > 1e-9:
            raise AssertionError(
                f"prefix of length {length} disagrees with the full pass by {difference}")

    # A token appended after position i must not change the logits at position i.
    extended, _ = model_logits(params, config, tokens + [9])
    if float((extended[: len(tokens)] - full).abs().max()) > 1e-9:
        raise AssertionError("appending a token changed an earlier position's logits")


def check_cached_matches_full_prefix(config, params):
    prompt = [3, 7, 0]
    continuation = [5, 2, 8]
    logits, cache = model_logits(params, config, prompt)
    stepwise = [logits[-1]]
    for token in continuation:
        logits, cache = model_logits(params, config, [token], cache=cache)
        stepwise.append(logits[-1])

    whole = prompt + continuation
    full, _ = model_logits(params, config, whole)
    for offset, row in enumerate(stepwise):
        position = len(prompt) - 1 + offset
        difference = float((row - full[position]).abs().max())
        if difference > 1e-9:
            raise AssertionError(
                f"cached decoding disagrees with the full prefix at position {position} "
                f"by {difference}")

    # The cache must actually have grown by one key/value per step.
    for key, value in cache:
        if key.shape[-2] != len(whole) or value.shape[-2] != len(whole):
            raise AssertionError(f"cache length {key.shape[-2]}, expected {len(whole)}")

    # Feeding a stale cache with a wrong layer count is refused, not silently used.
    try:
        model_logits(params, config, [1], cache=cache[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("model_logits accepted a cache with the wrong layer count")


def check_greedy_decode(config, params):
    prompt = [3, 7, 0]
    produced, prompt_logits = generate(params, config, prompt, max_new_tokens=4)
    if len(produced) != 4:
        raise AssertionError(f"greedy decode produced {len(produced)} tokens")
    again, _ = generate(params, config, prompt, max_new_tokens=4)
    if produced != again:
        raise AssertionError(f"greedy decode is not deterministic: {produced} vs {again}")

    # The first emitted token is the argmax of the independently computed logits.
    oracle = oracle_forward(params, config, prompt)[-1]
    expected_first = max(range(len(oracle)), key=lambda index: oracle[index])
    if produced[0] != expected_first:
        raise AssertionError(f"first greedy token {produced[0]}, oracle says {expected_first}")
    if greedy_next_token(prompt_logits) != expected_first:
        raise AssertionError("greedy_next_token disagrees with the oracle argmax")

    # Each subsequent token is the argmax after appending the previous one.
    context = list(prompt)
    for step, token in enumerate(produced):
        row = oracle_forward(params, config, context)[-1]
        expected = max(range(len(row)), key=lambda index: row[index])
        if token != expected:
            raise AssertionError(f"greedy token {step} is {token}, oracle says {expected}")
        context.append(token)

    # A stop token ends the loop immediately.
    stopped, _ = generate(params, config, prompt, max_new_tokens=4, stop_ids=(produced[0],))
    if stopped != [produced[0]]:
        raise AssertionError(f"stop token was not honoured: {stopped}")
    if generate(params, config, prompt, max_new_tokens=0)[0] != []:
        raise AssertionError("max_new_tokens=0 produced tokens")


def check_guards(config, params):
    for bad in ([], [int(config["vocab_size"])], [-1]):
        try:
            model_logits(params, config, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"model_logits accepted token ids {bad}")
    try:
        model_logits(params, {k: v for k, v in config.items() if k != "rope_theta"}, [1])
    except ValueError:
        pass
    else:
        raise AssertionError("model_logits accepted a configuration missing rope_theta")
    try:
        generate(params, config, [1], max_new_tokens=2, temperature=0.7)
    except ValueError:
        pass
    else:
        raise AssertionError("stochastic decoding ran without a seeded generator")


def main():
    config, params = build()
    check_fixture_is_reproducible(config)
    check_logits_against_oracle(config, params)
    check_causal_prefix_invariance(config, params)
    check_cached_matches_full_prefix(config, params)
    check_greedy_decode(config, params)
    check_guards(config, params)
    print("PASS: decoder stack logits against an independent pure-Python forward, "
          "causal-prefix invariance, cached-incremental vs full-prefix agreement, "
          "greedy determinism and the input guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
