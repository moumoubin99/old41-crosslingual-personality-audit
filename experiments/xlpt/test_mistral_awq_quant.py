#!/usr/bin/env python3
"""Conformance tests for the captured AWQ and kernel mathematics.

Every check here compares the tensor implementation against an INDEPENDENT
scalar oracle: a plain Python loop over Python numbers that re-derives the
answer from the definition, written without reusing the module under test.
That is what the deployment review asked for -- a test that can fail when the
implementation is wrong, rather than a test that restates it.
"""

from __future__ import annotations

import math
import unittest

import torch

from experiments.xlpt import mistral_awq_reference as ref


_AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _pack_nibbles(rows: list[list[int]]) -> torch.Tensor:
    """Pack 4-bit values into int32 columns in the AWQ interleave.

    Independent of ``awq_unpack``: it builds the packed word from the format
    definition, so a wrong shift table in the module makes the round trip fail.
    """
    packed_rows = []
    for row in rows:
        if len(row) % 8:
            raise ValueError("test rows must be a multiple of eight wide")
        words = []
        for start in range(0, len(row), 8):
            word = 0
            for slot, nibble_index in enumerate(_AWQ_UNPACK_ORDER):
                word |= (row[start + slot] & 0xF) << (4 * nibble_index)
            words.append(word - (1 << 32) if word >= (1 << 31) else word)
        packed_rows.append(words)
    return torch.tensor(packed_rows, dtype=torch.int32)


class AwqExtractionTest(unittest.TestCase):
    def test_unpack_round_trips_the_registered_interleave(self):
        rows = [[(3 * i + 5 * j) % 16 for j in range(16)] for i in range(4)]
        packed = _pack_nibbles(rows)
        extracted = ref.awq_extract(packed)
        self.assertEqual(tuple(extracted.shape), (4, 2, 8))
        self.assertEqual(extracted.reshape(4, 16).tolist(), rows)
        unpacked = ref.awq_unpack(packed)
        # The downstream dequantizer consumes this separately explicit flatten.
        self.assertEqual(tuple(unpacked.shape), (4, 16))
        self.assertEqual(unpacked.tolist(), rows)

    def test_unpack_rejects_non_int32_input(self):
        with self.assertRaises(ValueError):
            ref.awq_unpack(torch.zeros(2, 2, dtype=torch.int64))

    def test_group_index_maps_rows_to_their_group(self):
        index = ref.awq_group_index(8, 4)
        self.assertEqual(index.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            ref.awq_group_index(9, 4)

    def test_dequantize_matches_a_scalar_oracle(self):
        input_dim, output_dim, group_size = 8, 8, 4
        groups = input_dim // group_size
        weight_rows = [[(i * 7 + j * 3) % 16 for j in range(output_dim)] for i in range(input_dim)]
        zero_rows = [[(g * 5 + j) % 16 for j in range(output_dim)] for g in range(groups)]
        scales = torch.tensor(
            [[0.125 * (1 + ((g + j) % 3)) for j in range(output_dim)] for g in range(groups)],
            dtype=torch.float32,
        )
        produced = ref.awq_dequantize(
            _pack_nibbles(weight_rows), scales, _pack_nibbles(zero_rows), group_size
        )
        for i in range(input_dim):
            group = i // group_size
            for j in range(output_dim):
                expected = (weight_rows[i][j] - zero_rows[group][j]) * scales[group][j].item()
                self.assertAlmostEqual(produced[i][j].item(), expected, places=5)

    def test_dequantize_rejects_scales_that_miss_a_group(self):
        weight = _pack_nibbles([[1] * 8 for _ in range(8)])
        zeros = _pack_nibbles([[0] * 8 for _ in range(2)])
        with self.assertRaises(ValueError):
            ref.awq_dequantize(weight, torch.ones(1, 8), zeros, 4)


class FusedGemmEquivalenceTest(unittest.TestCase):
    """The fused AWQ GEMM's contract is dequantize-then-matmul."""

    def test_awq_linear_equals_a_scalar_accumulation(self):
        input_dim, output_dim, group_size = 8, 8, 4
        groups = input_dim // group_size
        weight_rows = [[(i + 2 * j) % 16 for j in range(output_dim)] for i in range(input_dim)]
        zero_rows = [[8 for _ in range(output_dim)] for _ in range(groups)]
        scales = torch.full((groups, output_dim), 0.25, dtype=torch.float32)
        hidden_rows = [[0.5 * ((i * 3 + k) % 7) - 1.0 for k in range(input_dim)] for i in range(3)]
        hidden = torch.tensor([hidden_rows], dtype=torch.float32)

        produced = ref.awq_linear(
            hidden, _pack_nibbles(weight_rows), scales, _pack_nibbles(zero_rows), group_size
        )
        self.assertEqual(tuple(produced.shape), (1, 3, output_dim))
        for row_index, row in enumerate(hidden_rows):
            for j in range(output_dim):
                total = 0.0
                for i in range(input_dim):
                    group = i // group_size
                    weight = (weight_rows[i][j] - zero_rows[group][j]) * scales[group][j].item()
                    total += row[i] * weight
                self.assertAlmostEqual(produced[0][row_index][j].item(), total, places=4)

    def test_awq_linear_rejects_a_width_mismatch(self):
        weight = _pack_nibbles([[1] * 8 for _ in range(8)])
        zeros = _pack_nibbles([[8] * 8 for _ in range(2)])
        scales = torch.ones(2, 8)
        with self.assertRaises(ValueError):
            ref.awq_linear(torch.zeros(1, 2, 7), weight, scales, zeros, 4)


class KernelTest(unittest.TestCase):
    def test_embedding_lookup_gathers_the_named_rows(self):
        table = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        ids = torch.tensor([[2, 0, 1]])
        produced = ref.embedding_lookup(ids, table)
        self.assertEqual(produced.tolist(), [[[4.0, 5.0], [0.0, 1.0], [2.0, 3.0]]])

    def test_embedding_lookup_refuses_an_out_of_range_id(self):
        with self.assertRaises(ValueError):
            ref.embedding_lookup(torch.tensor([[3]]), torch.zeros(3, 2))

    def test_rms_norm_matches_a_scalar_oracle(self):
        values = [1.0, -2.0, 0.5, 3.0]
        gains = [0.5, 1.0, 2.0, -1.0]
        epsilon = 1e-5
        produced = ref.rms_norm(
            torch.tensor([[values]]), torch.tensor(gains), epsilon
        )[0][0]
        mean_square = sum(v * v for v in values) / len(values)
        denominator = math.sqrt(mean_square + epsilon)
        for index, value in enumerate(values):
            self.assertAlmostEqual(
                produced[index].item(), gains[index] * value / denominator, places=5
            )

    def test_rms_norm_is_scale_equivariant_only_through_the_gain(self):
        hidden = torch.randn(1, 3, 8)
        gain = torch.ones(8)
        base = ref.rms_norm(hidden, gain, 1e-12)
        scaled = ref.rms_norm(hidden * 7.0, gain, 1e-12)
        self.assertTrue(torch.allclose(base, scaled, atol=1e-4))

    def test_rms_norm_rejects_a_nonpositive_epsilon(self):
        with self.assertRaises(ValueError):
            ref.rms_norm(torch.zeros(1, 1, 2), torch.ones(2), 0.0)

    def test_silu_matches_its_definition(self):
        for value in (-3.0, -0.5, 0.0, 0.5, 4.0):
            expected = value / (1.0 + math.exp(-value))
            self.assertAlmostEqual(
                ref.silu(torch.tensor(value)).item(), expected, places=6
            )

    def test_gated_silu_mlp_matches_a_scalar_composition(self):
        hidden_values = [0.25 * (i - 3) for i in range(8)]
        hidden = torch.tensor([[hidden_values]], dtype=torch.float32)

        def projection(salt):
            weights = [[(3 * i + 5 * j + salt) % 16 for j in range(8)]
                       for i in range(8)]
            zeros = [[8] * 8]
            return ({"qweight": _pack_nibbles(weights),
                     "scales": torch.full((1, 8), 0.125),
                     "qzeros": _pack_nibbles(zeros), "group_size": 8}, weights)

        gate, gate_weights = projection(1)
        up, up_weights = projection(2)
        down, down_weights = projection(3)
        produced = ref.gated_silu_mlp(hidden, gate, up, down)[0, 0]

        def apply(values, weights):
            return [sum(values[i] * (weights[i][j] - 8) * 0.125
                        for i in range(8)) for j in range(8)]

        gate_values = apply(hidden_values, gate_weights)
        up_values = apply(hidden_values, up_weights)
        activated = [g / (1.0 + math.exp(-g)) * u
                     for g, u in zip(gate_values, up_values)]
        expected = apply(activated, down_weights)
        for got, want in zip(produced.tolist(), expected):
            self.assertAlmostEqual(got, want, places=4)

    def test_linear_contracts_the_last_axis(self):
        hidden = torch.tensor([[[1.0, 2.0, 3.0]]])
        weight = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
        produced = ref.linear(hidden, weight)
        self.assertEqual(produced.tolist(), [[[-2.0, 3.0]]])
        with self.assertRaises(ValueError):
            ref.linear(torch.zeros(1, 1, 4), weight)


class SealedSeedTest(unittest.TestCase):
    def test_apparatus_refuses_both_sealed_blocks(self):
        for sealed in (20260811, 20260834, 20260911, 20260934):
            with self.assertRaises(ValueError):
                ref.check_seed(sealed)
        self.assertEqual(ref.check_seed(20251201), 20251201)


if __name__ == "__main__":
    unittest.main()
