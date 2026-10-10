#!/usr/bin/env python3
"""Conformance tests for rotary position embedding, QKV handling, grouped-query
attention, masking and the KV cache.

As in the quantization suite, each numeric claim is checked against a Python
scalar oracle derived from the definition rather than from the module.
"""

from __future__ import annotations

import math
import unittest

import torch

from experiments.xlpt import mistral_awq_reference as ref


torch.manual_seed(17091812)


class RopeTableTest(unittest.TestCase):
    def test_inverse_frequencies_match_the_scalar_definition(self):
        head_dim, theta = 8, 1_000_000.0
        produced = ref.rope_inverse_frequencies(head_dim, theta)
        self.assertEqual(tuple(produced.shape), (head_dim // 2,))
        for k in range(head_dim // 2):
            expected = theta ** (-(2.0 * k) / head_dim)
            self.assertAlmostEqual(produced[k].item(), expected, places=12)

    def test_inverse_frequencies_reject_odd_or_nonpositive_shapes(self):
        with self.assertRaises(ValueError):
            ref.rope_inverse_frequencies(7, 10000.0)
        with self.assertRaises(ValueError):
            ref.rope_inverse_frequencies(8, 0.0)

    def test_tables_are_position_dependent_and_duplicated(self):
        head_dim, theta = 8, 1_000_000.0
        positions = torch.arange(5)
        cosine, sine = ref.rope_tables(positions, head_dim, theta)
        self.assertEqual(tuple(cosine.shape), (5, head_dim))
        half = head_dim // 2
        # The half-width table is duplicated to line up with rotate_half.
        self.assertTrue(torch.allclose(cosine[:, :half], cosine[:, half:], atol=1e-6))
        # Different positions must give different rows -- the defect a
        # position-independent table would hide.
        self.assertFalse(torch.allclose(cosine[0], cosine[1], atol=1e-6))
        for p in range(5):
            for k in range(half):
                angle = p * theta ** (-(2.0 * k) / head_dim)
                self.assertAlmostEqual(cosine[p][k].item(), math.cos(angle), places=5)
                self.assertAlmostEqual(sine[p][k].item(), math.sin(angle), places=5)

    def test_position_zero_is_the_identity_rotation(self):
        head_dim = 8
        cosine, sine = ref.rope_tables(torch.zeros(1), head_dim, 1e6)
        value = torch.randn(1, 2, 1, head_dim)
        rotated = ref.apply_rope(value, cosine.unsqueeze(0).unsqueeze(0), sine.unsqueeze(0).unsqueeze(0))
        self.assertTrue(torch.allclose(rotated, value, atol=1e-6))

    def test_rotate_half_swaps_the_halves_with_a_sign(self):
        value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        self.assertEqual(ref.rotate_half(value).tolist(), [[-3.0, -4.0, 1.0, 2.0]])
        with self.assertRaises(ValueError):
            ref.rotate_half(torch.zeros(1, 3))

    def test_rope_preserves_the_norm_of_each_rotated_pair(self):
        head_dim = 8
        positions = torch.arange(4)
        cosine, sine = ref.rope_tables(positions, head_dim, 1e6, dtype=torch.float64)
        value = torch.randn(1, 2, 4, head_dim, dtype=torch.float64)
        rotated = ref.apply_rope(
            value, cosine.unsqueeze(0).unsqueeze(0), sine.unsqueeze(0).unsqueeze(0)
        )
        half = head_dim // 2
        for k in range(half):
            before = value[..., k] ** 2 + value[..., k + half] ** 2
            after = rotated[..., k] ** 2 + rotated[..., k + half] ** 2
            self.assertTrue(torch.allclose(before, after, atol=1e-9))

    def test_rope_relative_phase_depends_only_on_the_position_difference(self):
        """<RoPE(q, p), RoPE(k, p')> is a function of p - p'."""
        head_dim = 8
        query = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)
        key = torch.randn(1, 1, 1, head_dim, dtype=torch.float64)

        def product(position_q: int, position_k: int) -> float:
            cq, sq = ref.rope_tables(torch.tensor([position_q]), head_dim, 1e6, torch.float64)
            ck, sk = ref.rope_tables(torch.tensor([position_k]), head_dim, 1e6, torch.float64)
            rq = ref.apply_rope(query, cq.unsqueeze(0).unsqueeze(0), sq.unsqueeze(0).unsqueeze(0))
            rk = ref.apply_rope(key, ck.unsqueeze(0).unsqueeze(0), sk.unsqueeze(0).unsqueeze(0))
            return float((rq * rk).sum())

        self.assertAlmostEqual(product(3, 1), product(9, 7), places=8)
        self.assertNotAlmostEqual(product(3, 1), product(3, 2), places=6)

    def test_apply_rope_rejects_mismatched_tables(self):
        value = torch.zeros(1, 1, 1, 8)
        cosine = torch.zeros(1, 1, 1, 8)
        with self.assertRaises(ValueError):
            ref.apply_rope(value, cosine, torch.zeros(1, 1, 1, 4))
        with self.assertRaises(ValueError):
            ref.apply_rope(torch.zeros(1, 1, 1, 6), cosine, cosine)


class QkvSplitTest(unittest.TestCase):
    def test_split_puts_heads_before_positions(self):
        batch, sequence, heads, kv_heads, head_dim = 1, 3, 4, 2, 2
        width = (heads + 2 * kv_heads) * head_dim
        fused = torch.arange(batch * sequence * width, dtype=torch.float32)
        fused = fused.reshape(batch, sequence, width)
        query, key, value = ref.split_qkv(fused, heads, kv_heads, head_dim)
        self.assertEqual(tuple(query.shape), (batch, heads, sequence, head_dim))
        self.assertEqual(tuple(key.shape), (batch, kv_heads, sequence, head_dim))
        self.assertEqual(tuple(value.shape), (batch, kv_heads, sequence, head_dim))
        # Head h at position t must be the contiguous slice of the fused row --
        # the property a transpose-before-reshape would break.
        for t in range(sequence):
            for h in range(heads):
                start = h * head_dim
                self.assertEqual(
                    query[0, h, t].tolist(),
                    fused[0, t, start : start + head_dim].tolist(),
                )
        key_start = heads * head_dim
        value_start = key_start + kv_heads * head_dim
        for t in range(sequence):
            for h in range(kv_heads):
                offset = key_start + h * head_dim
                self.assertEqual(
                    key[0, h, t].tolist(), fused[0, t, offset : offset + head_dim].tolist()
                )
                offset = value_start + h * head_dim
                self.assertEqual(
                    value[0, h, t].tolist(), fused[0, t, offset : offset + head_dim].tolist()
                )

    def test_split_rejects_a_width_that_contradicts_the_head_counts(self):
        with self.assertRaises(ValueError):
            ref.split_qkv(torch.zeros(1, 2, 15), 4, 2, 2)


class GroupedQueryTest(unittest.TestCase):
    def test_output_head_count_is_key_head_times_groups(self):
        hidden = torch.arange(2 * 2 * 3 * 2, dtype=torch.float32).reshape(2, 2, 3, 2)
        repeated = ref.repeat_kv(hidden, 3)
        self.assertEqual(tuple(repeated.shape), (2, 2 * 3, 3, 2))
        for key_head in range(2):
            for copy in range(3):
                self.assertTrue(
                    torch.equal(repeated[:, key_head * 3 + copy], hidden[:, key_head])
                )

    def test_single_group_is_the_identity(self):
        hidden = torch.randn(1, 2, 3, 4)
        self.assertTrue(torch.equal(ref.repeat_kv(hidden, 1), hidden))

    def test_repeat_rejects_a_nonpositive_factor(self):
        with self.assertRaises(ValueError):
            ref.repeat_kv(torch.zeros(1, 1, 1, 1), 0)


class MaskAndAttentionTest(unittest.TestCase):
    def test_mask_is_causal_and_cache_aware(self):
        mask = ref.causal_mask(2, 5)
        self.assertEqual(tuple(mask.shape), (2, 5))
        # query 0 sits at absolute position 3, query 1 at position 4.
        self.assertEqual(mask[0].tolist()[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(mask[0][4].item(), float("-inf"))
        self.assertEqual(mask[1].tolist(), [0.0] * 5)
        with self.assertRaises(ValueError):
            ref.causal_mask(5, 2)

    def test_attention_rows_are_probabilities(self):
        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn(1, 2, 3, 8)
        out = ref.scaled_causal_attention(query, key, value, groups=2)
        self.assertEqual(tuple(out.shape), (1, 4, 3, 8))
        # Convex-combination invariant: each output row lies inside the range of
        # the visible value rows, which holds only if the probabilities sum to
        # one and the mask really blocks the future.
        for head in range(4):
            for position in range(3):
                visible = ref.repeat_kv(value, 2)[0, head, : position + 1]
                produced = out[0, head, position]
                self.assertTrue(bool((produced <= visible.max(dim=0).values + 1e-5).all()))
                self.assertTrue(bool((produced >= visible.min(dim=0).values - 1e-5).all()))

    def test_attention_matches_a_scalar_oracle_with_scaling_and_mask(self):
        head_dim = 4
        query = torch.randn(1, 1, 2, head_dim, dtype=torch.float64)
        key = torch.randn(1, 1, 2, head_dim, dtype=torch.float64)
        value = torch.randn(1, 1, 2, head_dim, dtype=torch.float64)
        produced = ref.scaled_causal_attention(query, key, value, groups=1)
        scale = head_dim ** -0.5
        for i in range(2):
            scores = []
            for j in range(i + 1):
                dot = sum(
                    query[0, 0, i, d].item() * key[0, 0, j, d].item() for d in range(head_dim)
                )
                scores.append(dot * scale)
            peak = max(scores)
            weights = [math.exp(s - peak) for s in scores]
            total = sum(weights)
            for d in range(head_dim):
                expected = sum(
                    (w / total) * value[0, 0, j, d].item() for j, w in enumerate(weights)
                )
                self.assertAlmostEqual(produced[0, 0, i, d].item(), expected, places=9)

    def test_attention_rejects_a_group_count_that_misses_the_query_heads(self):
        with self.assertRaises(ValueError):
            ref.scaled_causal_attention(
                torch.randn(1, 4, 2, 4), torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4), groups=1
            )


class CacheTest(unittest.TestCase):
    def test_append_grows_the_sequence_axis(self):
        cached = torch.randn(1, 2, 3, 4)
        fresh = torch.randn(1, 2, 1, 4)
        appended = ref.cache_append(cached, fresh)
        self.assertEqual(tuple(appended.shape), (1, 2, 4, 4))
        self.assertTrue(torch.equal(appended[:, :, :3], cached))
        self.assertTrue(torch.equal(appended[:, :, 3:], fresh))
        self.assertTrue(torch.equal(ref.cache_append(None, fresh), fresh))
        with self.assertRaises(ValueError):
            ref.cache_append(cached, torch.randn(1, 3, 1, 4))

    def test_length_and_position_updates_agree(self):
        self.assertEqual(ref.cache_next_length(7), 8)
        self.assertEqual(ref.cache_next_length(7, 3), 10)
        self.assertEqual(ref.next_positions(7, 3).tolist(), [7, 8, 9])
        with self.assertRaises(ValueError):
            ref.next_positions(-1, 1)
        with self.assertRaises(ValueError):
            ref.next_positions(0, 0)

    def test_cached_attention_equals_the_uncached_prefix(self):
        """The cache is correct only if splitting the sequence changes nothing."""
        query = torch.randn(1, 2, 3, 8, dtype=torch.float64)
        key = torch.randn(1, 1, 3, 8, dtype=torch.float64)
        value = torch.randn(1, 1, 3, 8, dtype=torch.float64)
        whole = ref.scaled_causal_attention(query, key, value, groups=2)
        cached_key = ref.cache_append(key[:, :, :2], key[:, :, 2:])
        cached_value = ref.cache_append(value[:, :, :2], value[:, :, 2:])
        stepped = ref.scaled_causal_attention(
            query[:, :, 2:], cached_key, cached_value, groups=2
        )
        self.assertTrue(torch.allclose(whole[:, :, 2:], stepped, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
