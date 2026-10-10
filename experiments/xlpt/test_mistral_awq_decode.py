#!/usr/bin/env python3
"""Conformance tests for the composed decode path: layer, model, vocabulary-
parallel head, greedy selection, EOS-set stopping, and the config of record.

The quantization and attention suites verify the pieces against scalar oracles.
This suite verifies the COMPOSITION -- the properties that only hold if the
pieces are wired together correctly: residual structure, cache/uncached
agreement over a real forward, rank-ordered logit gathering, EOS membership,
and a configuration whose content is inspectable rather than merely hashed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.xlpt import mistral_awq_reference as ref


ROOT = Path(__file__).resolve().parents[2]
DEPLOYED_CONFIG = ROOT / "process/consumption/deployment/mistral-rater-config"
_AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
torch.manual_seed(17091813)


def _pack(rows: list[list[int]]) -> torch.Tensor:
    packed = []
    for row in rows:
        words = []
        for start in range(0, len(row), 8):
            word = 0
            for slot, nibble in enumerate(_AWQ_UNPACK_ORDER):
                word |= (row[start + slot] & 0xF) << (4 * nibble)
            words.append(word - (1 << 32) if word >= (1 << 31) else word)
        packed.append(words)
    return torch.tensor(packed, dtype=torch.int32)


def _awq_projection(input_dim: int, output_dim: int, group_size: int, salt: int) -> dict:
    """A small deterministic AWQ projection in the stored packed layout."""
    groups = input_dim // group_size
    weight = [[(i * 3 + j * 5 + salt) % 16 for j in range(output_dim)] for i in range(input_dim)]
    zeros = [[8 for _ in range(output_dim)] for _ in range(groups)]
    scales = torch.full((groups, output_dim), 0.05, dtype=torch.float32)
    return {
        "qweight": _pack(weight),
        "scales": scales,
        "qzeros": _pack(zeros),
        "group_size": group_size,
    }


def _tiny_model(layers: int = 2) -> tuple[dict, dict]:
    torch.manual_seed(0)
    hidden, heads, kv_heads, head_dim, vocab = 16, 4, 2, 4, 12
    group_size = 8
    config = {
        "hidden_size": hidden,
        "intermediate_size": 32,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "vocab_size": vocab,
        "eos_token_id": [2, 11],
    }
    qkv_width = (heads + 2 * kv_heads) * head_dim
    params = {
        "embed_tokens": torch.randn(vocab, hidden) * 0.2,
        "norm": torch.ones(hidden),
        "lm_head": torch.randn(vocab, hidden) * 0.2,
        "layers": [
            {
                "input_layernorm": torch.ones(hidden),
                "post_attention_layernorm": torch.ones(hidden),
                "qkv_proj": _awq_projection(hidden, qkv_width, group_size, salt=layer),
                "o_proj": _awq_projection(heads * head_dim, hidden, group_size, salt=layer + 1),
                "gate_proj": _awq_projection(hidden, 32, group_size, salt=layer + 2),
                "up_proj": _awq_projection(hidden, 32, group_size, salt=layer + 3),
                "down_proj": _awq_projection(32, hidden, group_size, salt=layer + 4),
            }
            for layer in range(layers)
        ],
    }
    return params, config


class DecoderLayerTest(unittest.TestCase):
    def test_layer_preserves_shape_and_grows_the_cache(self):
        params, config = _tiny_model(layers=1)
        hidden = torch.randn(1, 3, config["hidden_size"])
        positions = ref.next_positions(0, 3)
        out, key, value = ref.decoder_layer(hidden, params["layers"][0], config, positions)
        self.assertEqual(out.shape, hidden.shape)
        self.assertEqual(
            tuple(key.shape),
            (1, config["num_key_value_heads"], 3, config["head_dim"]),
        )
        self.assertEqual(key.shape, value.shape)

    def test_layer_is_residual(self):
        """Zeroing both output projections must leave the input untouched."""
        params, config = _tiny_model(layers=1)
        layer = dict(params["layers"][0])
        for name in ("o_proj", "down_proj"):
            zeroed = dict(layer[name])
            zeroed["scales"] = torch.zeros_like(zeroed["scales"])
            layer[name] = zeroed
        hidden = torch.randn(1, 3, config["hidden_size"])
        out, _, _ = ref.decoder_layer(hidden, layer, config, ref.next_positions(0, 3))
        self.assertTrue(torch.allclose(out, hidden, atol=1e-6))

    def test_layer_rejects_head_counts_that_do_not_divide(self):
        params, config = _tiny_model(layers=1)
        broken = dict(config)
        broken["num_key_value_heads"] = 3
        with self.assertRaises(ValueError):
            ref.decoder_layer(
                torch.randn(1, 2, config["hidden_size"]),
                params["layers"][0],
                broken,
                ref.next_positions(0, 2),
            )


class ModelForwardTest(unittest.TestCase):
    def test_logits_have_the_vocabulary_shape(self):
        params, config = _tiny_model()
        ids = torch.tensor([[1, 4, 7]])
        logits, cache = ref.model_logits(ids, params, config)
        self.assertEqual(tuple(logits.shape), (1, 3, config["vocab_size"]))
        self.assertEqual(len(cache), config["num_hidden_layers"])

    def test_causality_a_later_token_cannot_change_an_earlier_logit(self):
        params, config = _tiny_model()
        first = ref.model_logits(torch.tensor([[1, 4, 7]]), params, config)[0]
        second = ref.model_logits(torch.tensor([[1, 4, 9]]), params, config)[0]
        self.assertTrue(torch.allclose(first[:, :2], second[:, :2], atol=1e-5))
        self.assertFalse(torch.allclose(first[:, 2], second[:, 2], atol=1e-5))

    def test_cached_step_equals_the_full_recompute(self):
        """The decisive cache test: prefill two tokens, step the third, and
        compare with a single forward over all three."""
        params, config = _tiny_model()
        ids = torch.tensor([[1, 4, 7]])
        whole = ref.model_logits(ids, params, config)[0]
        prefill, cache = ref.model_logits(ids[:, :2], params, config)
        stepped, _ = ref.model_logits(
            ids[:, 2:], params, config, past_length=2, cache=cache
        )
        self.assertTrue(torch.allclose(whole[:, :2], prefill, atol=1e-4))
        self.assertTrue(torch.allclose(whole[:, 2:], stepped, atol=1e-4))

    def test_position_offset_is_load_bearing(self):
        """Stepping at the wrong position must change the answer -- otherwise
        the position update would be untested."""
        params, config = _tiny_model()
        ids = torch.tensor([[1, 4, 7]])
        _, cache = ref.model_logits(ids[:, :2], params, config)
        right, _ = ref.model_logits(ids[:, 2:], params, config, past_length=2, cache=cache)
        _, cache = ref.model_logits(ids[:, :2], params, config)
        wrong, _ = ref.model_logits(ids[:, 2:], params, config, past_length=0, cache=cache)
        self.assertFalse(torch.allclose(right, wrong, atol=1e-4))

    def test_cache_depth_must_match_the_layer_count(self):
        params, config = _tiny_model()
        with self.assertRaises(ValueError):
            ref.model_logits(torch.tensor([[1]]), params, config, past_length=0, cache=[])


class VocabularyParallelTest(unittest.TestCase):
    def test_sharded_head_equals_the_whole_head(self):
        params, config = _tiny_model()
        ids = torch.tensor([[1, 4, 7]])
        whole = ref.model_logits(ids, params, config)[0]
        head = params["lm_head"]
        sharded = dict(params)
        sharded["lm_head"] = [head[: head.shape[0] // 2], head[head.shape[0] // 2 :]]
        gathered = ref.model_logits(ids, sharded, config)[0]
        self.assertTrue(torch.allclose(whole, gathered, atol=1e-6))

    def test_rank_order_is_load_bearing(self):
        shards = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
        self.assertEqual(
            ref.vocabulary_parallel_logits(shards).tolist(), [[1.0, 2.0, 3.0, 4.0]]
        )
        self.assertNotEqual(
            ref.vocabulary_parallel_logits(list(reversed(shards))).tolist(),
            [[1.0, 2.0, 3.0, 4.0]],
        )
        with self.assertRaises(ValueError):
            ref.vocabulary_parallel_logits([])
        with self.assertRaises(ValueError):
            ref.vocabulary_parallel_logits([torch.zeros(1, 2), torch.zeros(2, 2)])


class SelectionAndStoppingTest(unittest.TestCase):
    def test_greedy_takes_the_first_maximum(self):
        logits = torch.tensor([[0.1, 0.9, 0.9, 0.2]])
        self.assertEqual(ref.greedy_next_token(logits).tolist(), [1])
        with self.assertRaises(ValueError):
            ref.greedy_next_token(torch.zeros(1, 0))

    def test_eos_is_set_membership_not_scalar_equality(self):
        eos = [2, 11, 32000]
        self.assertTrue(ref.eos_stop(11, eos))
        self.assertTrue(ref.eos_stop(2, eos))
        self.assertFalse(ref.eos_stop(3, eos))
        with self.assertRaises(ValueError):
            ref.eos_stop(1, [])

    def test_generation_stops_on_any_eos_member(self):
        params, config = _tiny_model()
        ids = torch.tensor([[1, 4]])
        # With every vocabulary entry an EOS member, decoding must stop after
        # exactly one token regardless of which id argmax selects.
        produced = ref.greedy_generate(
            ids, params, config, max_new_tokens=8, eos_tokens=range(config["vocab_size"])
        )
        self.assertEqual(produced.shape[-1], 1)

    def test_generation_respects_the_token_budget(self):
        params, config = _tiny_model()
        produced = ref.greedy_generate(
            torch.tensor([[1, 4]]), params, config, max_new_tokens=3, eos_tokens=[999]
        )
        self.assertEqual(produced.shape[-1], 3)

    def test_generation_matches_a_manual_cached_loop(self):
        params, config = _tiny_model()
        ids = torch.tensor([[1, 4]])
        produced = ref.greedy_generate(
            ids, params, config, max_new_tokens=3, eos_tokens=[999]
        ).tolist()[0]

        manual = []
        context = ids
        for _ in range(3):
            logits, _ = ref.model_logits(context, params, config)
            token = int(ref.greedy_next_token(logits[:, -1, :]).item())
            manual.append(token)
            context = torch.cat((context, torch.tensor([[token]])), dim=-1)
        self.assertEqual(produced, manual)


class ConfigOfRecordTest(unittest.TestCase):
    def test_record_exposes_inspectable_content(self):
        _, config = _tiny_model()
        quantization = {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "quant_method": "awq",
            "version": "gemm",
            "ignored": "extra",
        }
        record = ref.config_of_record(config, quantization)
        self.assertEqual(record["num_attention_heads"], 4)
        self.assertEqual(record["num_key_value_heads"], 2)
        self.assertEqual(record["key_value_groups"], 2)
        self.assertEqual(record["head_dim"], 4)
        self.assertEqual(record["rms_norm_eps"], 1e-5)
        self.assertEqual(record["rope_theta"], 1_000_000.0)
        self.assertEqual(record["eos_token_id"], [2, 11])
        self.assertEqual(record["quantization"]["bits"], 4)
        self.assertEqual(record["quantization"]["group_size"], 128)
        self.assertNotIn("ignored", record["quantization"])

    def test_multimodal_checkpoints_read_the_text_tower(self):
        _, inner = _tiny_model()
        record = ref.config_of_record({"text_config": inner, "eos_token_id": [2]})
        self.assertEqual(record["hidden_size"], inner["hidden_size"])

    def test_rope_base_accepted_from_either_layout(self):
        nested = {"rope_parameters": {"rope_theta": 5.0}}
        self.assertEqual(ref.rope_theta_of(nested), 5.0)
        self.assertEqual(ref.rope_theta_of({"rope_theta": 5.0}), 5.0)
        # Agreeing duplicates are fine; disagreeing ones are a real ambiguity.
        self.assertEqual(
            ref.rope_theta_of({"rope_theta": 5.0, "rope_parameters": {"rope_theta": 5.0}}), 5.0
        )
        with self.assertRaises(ValueError):
            ref.rope_theta_of({"rope_theta": 5.0, "rope_parameters": {"rope_theta": 6.0}})
        with self.assertRaises(ValueError):
            ref.rope_theta_of({})

    def test_head_dim_falls_back_to_hidden_over_heads(self):
        self.assertEqual(
            ref.head_dim_of({"hidden_size": 32, "num_attention_heads": 8}), 4
        )
        with self.assertRaises(ValueError):
            ref.head_dim_of({"hidden_size": 30, "num_attention_heads": 8})

    def test_validate_names_every_missing_key(self):
        _, config = _tiny_model()
        broken = {k: v for k, v in config.items() if k != "vocab_size"}
        with self.assertRaises(ValueError) as caught:
            ref.validate_config(broken)
        self.assertIn("vocab_size", str(caught.exception))

    def test_group_mismatch_is_refused(self):
        _, config = _tiny_model()
        broken = dict(config)
        broken["num_key_value_heads"] = 3
        with self.assertRaises(ValueError):
            ref.config_of_record(broken)

    def test_checkpoint_group_ratio_is_derived_from_head_counts(self):
        self.assertEqual(ref.key_value_group_count(32, 8), 4)
        with self.assertRaises(ValueError):
            ref.key_value_group_count(32, 0)
        with self.assertRaises(ValueError):
            ref.key_value_group_count(31, 8)

    def test_checkpoint_identity_reports_unreadable_files_honestly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
            identity = ref.checkpoint_identity(root, ["config.json", "absent.json"])
            self.assertEqual(len(identity["config.json"]), 64)
            self.assertTrue(identity["absent.json"].startswith("unreadable:"))

    def test_deployed_mistral_config_is_inspectable_and_matches_the_contract(self):
        config = ref.load_json(DEPLOYED_CONFIG / "config.json")
        quantization = ref.load_json(DEPLOYED_CONFIG / "quantization_config.json")
        generation = ref.load_json(DEPLOYED_CONFIG / "generation_config.json")
        record = ref.config_of_record(config, quantization, generation)
        self.assertEqual(record["hidden_size"], 5120)
        self.assertEqual(record["intermediate_size"], 32768)
        self.assertEqual(record["num_hidden_layers"], 40)
        self.assertEqual(record["num_attention_heads"], 32)
        self.assertEqual(record["num_key_value_heads"], 8)
        self.assertEqual(record["key_value_groups"], 4)
        self.assertEqual(record["head_dim"], 128)
        self.assertEqual(record["rms_norm_eps"], 1e-5)
        self.assertEqual(record["rope_theta"], 1_000_000_000.0)
        self.assertEqual(record["vocab_size"], 131072)
        self.assertEqual(record["eos_token_id"], 2)
        self.assertEqual(record["quantization"]["bits"], 4)
        self.assertEqual(record["quantization"]["group_size"], 128)
        self.assertEqual(record["quantization"]["quant_method"], "awq")
        self.assertEqual(record["quantization"]["version"], "gemm")
        self.assertTrue(record["quantization"]["sym"])


if __name__ == "__main__":
    unittest.main()
