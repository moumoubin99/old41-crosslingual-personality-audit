#!/usr/bin/env python3
"""Executable contract for deployment unit U2 - grouped-query causal attention.

The expected values are computed here with a pure-Python softmax oracle and by
hand; the deployed implementation's output is never used as its own reference.

Covered: the grouped-query head repetition, the causal keep-mask in both the
full-prefix and the cached-increment geometry, the scaled softmax attention
itself, the causal-prefix invariance that the incremental decode path depends
on, and the Qwen3 per-head query/key normalization applied before the rotation.
"""

from __future__ import annotations

import math

import torch

from qwen3_reference import (
    apply_rope,
    causal_mask,
    repeat_kv,
    rms_norm,
    rope_tables,
    scaled_causal_attention,
)

TOL = 1e-5


def softmax_oracle(scores):
    """Pure-Python softmax over one row, written independently of torch."""
    finite = [s for s in scores if s != float("-inf")]
    peak = max(finite)
    weights = [0.0 if s == float("-inf") else math.exp(s - peak) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def attention_oracle(q, k, v, keep):
    """Pure-Python single-head attention over nested lists."""
    head_dim = len(q[0])
    scale = 1.0 / math.sqrt(head_dim)
    out = []
    for i, qi in enumerate(q):
        scores = []
        for j, kj in enumerate(k):
            dot = sum(a * b for a, b in zip(qi, kj))
            scores.append(dot * scale if keep[i][j] else float("-inf"))
        weights = softmax_oracle(scores)
        out.append([sum(w * vj[d] for w, vj in zip(weights, v)) for d in range(len(v[0]))])
    return out


def close(actual, expected, tol=TOL, label=""):
    actual_list = actual.reshape(-1).tolist() if torch.is_tensor(actual) else list(actual)
    expected_list = list(expected)
    if len(actual_list) != len(expected_list):
        raise AssertionError(f"{label}: {len(actual_list)} values, expected {len(expected_list)}")
    for index, (got, want) in enumerate(zip(actual_list, expected_list)):
        if abs(got - want) > tol:
            raise AssertionError(f"{label}[{index}]: got {got!r}, expected {want!r}")


def check_repeat_kv():
    x = torch.tensor([[[10.0, 11.0]], [[20.0, 21.0]]])  # (kv_heads=2, tokens=1, dim=2)
    close(repeat_kv(x, 1), [10.0, 11.0, 20.0, 21.0], label="repeat_kv n_rep=1")
    expanded = repeat_kv(x, 3)
    if tuple(expanded.shape) != (6, 1, 2):
        raise AssertionError(f"repeat_kv shape {tuple(expanded.shape)}, expected (6, 1, 2)")
    # Contiguous per source head: heads 0-2 read kv head 0, heads 3-5 read kv head 1.
    close(expanded, [10.0, 11.0] * 3 + [20.0, 21.0] * 3, label="repeat_kv n_rep=3")
    for bad in (0, -2):
        try:
            repeat_kv(x, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"repeat_kv accepted n_rep {bad}")


def check_causal_mask():
    square = causal_mask(3, 3).tolist()
    if square != [[True, False, False], [True, True, False], [True, True, True]]:
        raise AssertionError(f"causal_mask(3,3) = {square}")

    # One new token against a cache of three: every past key is visible.
    single = causal_mask(1, 4).tolist()
    if single != [[True, True, True, True]]:
        raise AssertionError(f"causal_mask(1,4) = {single}")

    # Two new tokens against a cache of three: offset 3.
    pair = causal_mask(2, 5).tolist()
    if pair != [[True, True, True, True, False], [True, True, True, True, True]]:
        raise AssertionError(f"causal_mask(2,5) = {pair}")

    try:
        causal_mask(4, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("causal_mask accepted a kv_len shorter than q_len")


def check_attention_values():
    q = [[1.0, 0.0], [0.0, 1.0]]
    k = [[1.0, 0.0], [0.0, 1.0]]
    v = [[1.0, 2.0], [3.0, 4.0]]
    keep = [[True, False], [True, True]]
    expected = attention_oracle(q, k, v, keep)

    # Row 0 sees only key 0, so it is exactly v[0]; check that by hand too.
    if [round(value, 12) for value in expected[0]] != [1.0, 2.0]:
        raise AssertionError(f"oracle row 0 = {expected[0]}, expected v[0]")
    # Row 1 mixes both by softmax([0, 1/sqrt(2)]).
    hand = softmax_oracle([0.0, 1.0 / math.sqrt(2.0)])
    hand_row = [hand[0] * 1.0 + hand[1] * 3.0, hand[0] * 2.0 + hand[1] * 4.0]
    close(expected[1], hand_row, label="oracle row 1 against hand softmax")

    context, weights = scaled_causal_attention(
        torch.tensor([q]), torch.tensor([k]), torch.tensor([v]))
    close(context, expected[0] + expected[1], label="scaled_causal_attention values")

    # Attention weights are a probability distribution on the visible prefix.
    rows = weights[0].tolist()
    if abs(rows[0][1]) > 0.0:
        raise AssertionError(f"masked position carries weight {rows[0][1]}")
    for index, row in enumerate(rows):
        if abs(sum(row) - 1.0) > 1e-6:
            raise AssertionError(f"attention row {index} sums to {sum(row)}")

    # Masking is load-bearing: perturbing a future value must not move row 0.
    v_moved = [[1.0, 2.0], [300.0, 400.0]]
    moved, _ = scaled_causal_attention(
        torch.tensor([q]), torch.tensor([k]), torch.tensor([v_moved]))
    close(moved[0][0], [1.0, 2.0], label="row 0 is independent of the future value")

    # A larger explicit scale sharpens row 1 toward the higher-scoring key.
    sharp, _ = scaled_causal_attention(
        torch.tensor([q]), torch.tensor([k]), torch.tensor([v]), scale=20.0)
    if not sharp[0][1][0].item() > context[0][1][0].item():
        raise AssertionError("increasing the scale did not sharpen the attention row")


def check_attention_shapes_and_guards():
    heads, q_len, kv_len, dim = 4, 3, 5, 2
    q = torch.randn(heads, q_len, dim, generator=torch.Generator().manual_seed(11))
    k = torch.randn(heads, kv_len, dim, generator=torch.Generator().manual_seed(12))
    v = torch.randn(heads, kv_len, dim, generator=torch.Generator().manual_seed(13))
    context, weights = scaled_causal_attention(q, k, v)
    if tuple(context.shape) != (heads, q_len, dim):
        raise AssertionError(f"context shape {tuple(context.shape)}")
    if tuple(weights.shape) != (heads, q_len, kv_len):
        raise AssertionError(f"weights shape {tuple(weights.shape)}")

    # Every head is checked against the independent pure-Python oracle.
    keep = causal_mask(q_len, kv_len).tolist()
    for head in range(heads):
        expected = attention_oracle(q[head].tolist(), k[head].tolist(), v[head].tolist(), keep)
        close(context[head], [value for row in expected for value in row],
              label=f"attention head {head} against the oracle")

    try:
        scaled_causal_attention(q, k, v[:, :-1, :])
    except ValueError:
        pass
    else:
        raise AssertionError("attention accepted mismatched key and value lengths")
    try:
        scaled_causal_attention(q, k[:-1], v[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("attention accepted mismatched head counts")
    try:
        scaled_causal_attention(q[0], k[0], v[0])
    except ValueError:
        pass
    else:
        raise AssertionError("attention accepted two-dimensional tensors")


def check_causal_prefix_invariance():
    """Row i over the full sequence equals the last row over the prefix 0..i.

    This is the property the cached incremental decode path relies on; if it
    fails, cached decoding and full-prefix evaluation cannot agree.
    """
    generator = torch.Generator().manual_seed(17091701)
    heads, length, dim = 2, 6, 4
    q = torch.randn(heads, length, dim, generator=generator)
    k = torch.randn(heads, length, dim, generator=generator)
    v = torch.randn(heads, length, dim, generator=generator)
    full, _ = scaled_causal_attention(q, k, v)
    for i in range(length):
        prefix, _ = scaled_causal_attention(q[:, : i + 1], k[:, : i + 1], v[:, : i + 1])
        close(prefix[:, -1], full[:, i].reshape(-1).tolist(), tol=1e-5,
              label=f"causal-prefix invariance at position {i}")

    # The cached geometry: one new query against the accumulated keys/values.
    for i in range(1, length):
        step, _ = scaled_causal_attention(q[:, i : i + 1], k[:, : i + 1], v[:, : i + 1])
        close(step[:, -1], full[:, i].reshape(-1).tolist(), tol=1e-5,
              label=f"cached-increment geometry at position {i}")


def check_qk_norm_before_rotation():
    """Qwen3 normalizes each query/key head before the rotary rotation.

    The check is that the per-head normalization is over head_dim (so each head
    is scaled independently) and that it commutes with neither head order nor
    the rotation - i.e. the order in the deployed layer is observable.
    """
    head_dim = 4
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    q = torch.tensor([[[3.0, 4.0, 0.0, 0.0]], [[0.0, 0.0, 6.0, 8.0]]])
    normed = rms_norm(q, weight, 1e-12)
    # head 0: mean(9,16,0,0)=6.25 -> 1/2.5 = 0.4 ; head 1: mean(0,0,36,64)=25 -> 0.2
    close(normed, [1.2, 1.6, 0.0, 0.0, 0.0, 0.0, 1.2, 1.6], label="per-head qk normalization")

    cos, sin = rope_tables(head_dim, [2.0], 10000.0)

    # With a unit gain the two operations commute: the rotation preserves the
    # per-pair norms, hence the RMS, hence the scale factor. This is a real
    # invariant of the layer and is asserted, not assumed.
    difference = (apply_rope(normed, cos, sin)
                  - rms_norm(apply_rope(q, cos, sin), weight, 1e-12)).abs().max().item()
    if difference > 1e-5:
        raise AssertionError(f"unit-gain normalization and rotation disagree by {difference}")

    # With the real per-channel gain they do not commute, so the deployed order
    # (normalize each head, then rotate) is observable rather than cosmetic.
    gain = torch.tensor([2.0, 0.5, 1.5, 1.0])
    norm_then_rope = apply_rope(rms_norm(q, gain, 1e-12), cos, sin)
    rope_then_norm = rms_norm(apply_rope(q, cos, sin), gain, 1e-12)
    spread = (norm_then_rope - rope_then_norm).abs().max().item()
    if spread < 1e-3:
        raise AssertionError(
            "the per-channel gain makes normalization and rotation commute, so the "
            f"deployed order would be unobservable (max difference {spread})")


def main():
    check_repeat_kv()
    check_causal_mask()
    check_attention_values()
    check_attention_shapes_and_guards()
    check_causal_prefix_invariance()
    check_qk_norm_before_rotation()
    print("PASS: grouped-query repetition, causal masking in both geometries, scaled softmax "
          "attention against a pure-Python oracle, causal-prefix invariance and QK-norm ordering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
