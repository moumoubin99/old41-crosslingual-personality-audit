#!/usr/bin/env python3
"""Executable contract for deployment unit U1 - qwen3 elementwise kernels.

Every selected formula is exercised against an independently computed expected
value (hand arithmetic or a pure-Python oracle written here), never against the
deployed implementation's own output.

Covered: RMS normalization, rotary tables, the half-split rotation, rotary
application, the bias-free projection, SiLU and the SwiGLU feed-forward block.
"""

from __future__ import annotations

import math

import torch

from qwen3_reference import (
    SEALED_SEED_RANGES,
    apply_rope,
    check_seed,
    linear,
    rms_norm,
    rope_tables,
    rotate_half,
    silu,
    swiglu_mlp,
)

TOL = 1e-6


def close(actual, expected, tol=TOL, label=""):
    actual_list = actual.reshape(-1).tolist() if torch.is_tensor(actual) else list(actual)
    expected_list = list(expected)
    if len(actual_list) != len(expected_list):
        raise AssertionError(f"{label}: {len(actual_list)} values, expected {len(expected_list)}")
    for index, (got, want) in enumerate(zip(actual_list, expected_list)):
        if abs(got - want) > tol:
            raise AssertionError(f"{label}[{index}]: got {got!r}, expected {want!r}")


def check_rms_norm():
    # mean(3^2, 4^2) = 12.5 ; 1/sqrt(12.5) = 0.2828427124746190
    scale = 1.0 / math.sqrt(12.5)
    x = torch.tensor([[3.0, 4.0]])
    unit = torch.tensor([1.0, 1.0])
    close(rms_norm(x, unit, 1e-12), [3.0 * scale, 4.0 * scale], label="rms_norm unit weight")

    weight = torch.tensor([2.0, 0.5])
    close(rms_norm(x, weight, 1e-12), [6.0 * scale, 2.0 * scale], label="rms_norm weighted")

    # A constant row normalizes to its own sign-preserving unit magnitude.
    close(rms_norm(torch.tensor([[5.0, 5.0, 5.0]]), torch.ones(3), 1e-12),
          [1.0, 1.0, 1.0], label="rms_norm constant row")
    close(rms_norm(torch.tensor([[-2.0, -2.0]]), torch.ones(2), 1e-12),
          [-1.0, -1.0], label="rms_norm negative row")

    # Scale invariance up to the weight: rms_norm(c*x) == rms_norm(x) for c > 0.
    base = torch.tensor([[0.5, -1.5, 2.0, 0.25]])
    weight4 = torch.tensor([1.1, 0.9, 1.0, 1.3])
    close(rms_norm(base * 7.0, weight4, 1e-12),
          rms_norm(base, weight4, 1e-12).reshape(-1).tolist(), tol=1e-5,
          label="rms_norm positive-scale invariance")

    # eps actually enters the denominator: an all-zero row stays finite and zero.
    close(rms_norm(torch.zeros(1, 3), torch.ones(3), 1e-6), [0.0, 0.0, 0.0], label="rms_norm zero row")

    # The normalization is over the trailing axis only, for any leading rank.
    stacked = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
    close(rms_norm(stacked, unit, 1e-12),
          [3.0 * scale, 4.0 * scale, 3.0 * scale, 4.0 * scale], tol=1e-5,
          label="rms_norm per-row over trailing axis")

    for bad in (0.0, -1e-6):
        try:
            rms_norm(x, unit, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rms_norm accepted non-positive eps {bad}")
    try:
        rms_norm(x, torch.ones(3), 1e-6)
    except ValueError:
        pass
    else:
        raise AssertionError("rms_norm accepted a weight of the wrong width")


def check_rotate_half():
    close(rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])), [-3.0, -4.0, 1.0, 2.0], label="rotate_half")
    close(rotate_half(torch.tensor([1.0, 2.0])), [-2.0, 1.0], label="rotate_half width 2")
    # Applying it four times is the identity (it is a quarter turn of each pair).
    value = torch.tensor([0.3, -1.2, 5.0, 0.0])
    close(rotate_half(rotate_half(rotate_half(rotate_half(value)))),
          value.tolist(), label="rotate_half fourth power")


def check_rope_tables():
    cos, sin = rope_tables(4, [0, 1], 100.0, dtype=torch.float64)
    if tuple(cos.shape) != (2, 4) or tuple(sin.shape) != (2, 4):
        raise AssertionError(f"rope table shape {tuple(cos.shape)}, expected (2, 4)")
    # inv_freq = 100 ** -[0/4*... ] -> exponents [0.0, 0.5] -> [1.0, 0.1]
    close(cos[0], [1.0, 1.0, 1.0, 1.0], label="rope cos at position 0")
    close(sin[0], [0.0, 0.0, 0.0, 0.0], label="rope sin at position 0")
    close(cos[1], [math.cos(1.0), math.cos(0.1), math.cos(1.0), math.cos(0.1)],
          label="rope cos at position 1")
    close(sin[1], [math.sin(1.0), math.sin(0.1), math.sin(1.0), math.sin(0.1)],
          label="rope sin at position 1")

    # theta scales the second half of the frequency ladder.
    cos2, _ = rope_tables(4, [1], 10000.0, dtype=torch.float64)
    close(cos2[0], [math.cos(1.0), math.cos(0.01), math.cos(1.0), math.cos(0.01)],
          label="rope cos with theta 10000")

    for bad_dim in (3, 5):
        try:
            rope_tables(bad_dim, [0], 100.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rope_tables accepted odd head_dim {bad_dim}")
    try:
        rope_tables(4, [0], 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("rope_tables accepted a non-positive theta")
    try:
        rope_tables(4, [[0, 1]], 100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("rope_tables accepted two-dimensional positions")


def check_apply_rope():
    # head_dim 2 -> a single frequency of 1.0 -> a plain 2-D rotation by the position.
    cos, sin = rope_tables(2, [1.0], 3.0, dtype=torch.float64)
    x = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    close(apply_rope(x, cos, sin), [math.cos(1.0), math.sin(1.0)], label="rope rotates the unit vector")
    y = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    close(apply_rope(y, cos, sin), [-math.sin(1.0), math.cos(1.0)], label="rope rotates the second axis")

    # Position 0 is the identity for any head_dim.
    cos0, sin0 = rope_tables(4, [0], 10000.0, dtype=torch.float64)
    arbitrary = torch.tensor([[[0.7, -2.0, 1.5, 0.25]]], dtype=torch.float64)
    close(apply_rope(arbitrary, cos0, sin0), arbitrary.reshape(-1).tolist(),
          label="rope at position 0 is the identity")

    # The rotation is norm-preserving on each half-split pair.
    cosn, sinn = rope_tables(4, [5.0], 10000.0, dtype=torch.float64)
    rotated = apply_rope(arbitrary, cosn, sinn).reshape(-1).tolist()
    flat = arbitrary.reshape(-1).tolist()
    for lo, hi in ((0, 2), (1, 3)):
        before = math.hypot(flat[lo], flat[hi])
        after = math.hypot(rotated[lo], rotated[hi])
        if abs(before - after) > 1e-9:
            raise AssertionError(f"rope changed the norm of pair ({lo},{hi}): {before} -> {after}")

    try:
        apply_rope(torch.zeros(1, 2, 4), *rope_tables(4, [0], 100.0))
    except ValueError:
        pass
    else:
        raise AssertionError("apply_rope accepted a table covering the wrong token count")


def check_linear():
    x = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # (out=3, in=2)
    # Row 0: [1*1+2*0, 1*0+2*2, 1*1+2*1] = [1, 4, 3]
    # Row 1: [0, -2, -1]
    close(linear(x, weight), [1.0, 4.0, 3.0, 0.0, -2.0, -1.0], label="linear")
    if tuple(linear(x, weight).shape) != (2, 3):
        raise AssertionError("linear produced the wrong shape")
    try:
        linear(x, torch.zeros(3, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("linear accepted a width mismatch")


def check_silu_and_mlp():
    close(silu(torch.tensor([0.0, 1.0, -1.0])),
          [0.0, 1.0 / (1.0 + math.exp(-1.0)), -1.0 / (1.0 + math.exp(1.0))], label="silu")

    x = torch.tensor([[1.0, 0.0]])
    gate_w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])      # gate(x) = [1, 0]
    up_w = torch.tensor([[2.0, 0.0], [0.0, 3.0]])        # up(x)   = [2, 0]
    down_w = torch.tensor([[1.0, 1.0], [1.0, -1.0]])     # (hidden=2, inter=2)
    activated = 1.0 / (1.0 + math.exp(-1.0))             # silu(1)
    product = activated * 2.0                            # elementwise with up
    close(swiglu_mlp(x, gate_w, up_w, down_w), [product, product], label="swiglu_mlp")

    # A zero gate zeroes the block regardless of the up projection.
    close(swiglu_mlp(torch.zeros(1, 2), gate_w, up_w, down_w), [0.0, 0.0], label="swiglu_mlp zero input")

    try:
        swiglu_mlp(x, gate_w, torch.zeros(3, 2), down_w)
    except ValueError:
        pass
    else:
        raise AssertionError("swiglu_mlp accepted mismatched gate and up widths")


def check_seed_guard():
    for allowed in (0, 17091701, 17091702, 20260810, 20260835, 20260910, 20260935):
        check_seed(allowed)
    for reserved in SEALED_SEED_RANGES:
        for sealed in (reserved.start, reserved.start + 7, reserved.stop - 1):
            try:
                check_seed(sealed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"check_seed accepted sealed seed {sealed}")
    for bad in (-1, True, 1.5):
        try:
            check_seed(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check_seed accepted {bad!r}")


def main():
    check_rms_norm()
    check_rotate_half()
    check_rope_tables()
    check_apply_rope()
    check_linear()
    check_silu_and_mlp()
    check_seed_guard()
    print("PASS: rms normalization, rotary tables and rotation, bias-free projection, "
          "SiLU/SwiGLU and the sealed-seed guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
