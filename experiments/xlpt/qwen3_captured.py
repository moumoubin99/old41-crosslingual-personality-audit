#!/usr/bin/env python3
"""Project-owned Qwen3 causal forward used by the English XLPT generator.

The implementation intentionally exposes every formula-bearing operation in
Python.  PyTorch supplies tensor primitives only; no Transformers model forward,
attention implementation, or generation kernel is invoked.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from pathlib import Path


def rms_norm(hidden, weight, epsilon: float):
    import torch

    dtype = hidden.dtype
    normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(dtype) * weight


def rotate_half(value):
    midpoint = value.shape[-1] // 2
    return __import__("torch").cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def rotary_cos_sin(position_ids, head_dim: int, theta: float, dtype, device):
    import torch

    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / float(head_dim)
    inv_frequency = 1.0 / (float(theta) ** exponent)
    frequencies = position_ids.to(device=device, dtype=torch.float32)[..., None] * inv_frequency
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rotary(query, key, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def repeat_key_value(hidden, repetitions: int):
    if repetitions == 1:
        return hidden
    batch, key_heads, sequence, head_dim = hidden.shape
    expanded = hidden[:, :, None, :, :].expand(batch, key_heads, repetitions, sequence, head_dim)
    return expanded.reshape(batch, key_heads * repetitions, sequence, head_dim)


def causal_attention_mask(attention_mask, query_length: int, past_length: int, dtype, device):
    """Additive mask for causal attention plus any invisible key positions."""
    import torch

    key_length = past_length + query_length
    query_positions = torch.arange(past_length, key_length, device=device)[:, None]
    key_positions = torch.arange(key_length, device=device)[None, :]
    allowed = key_positions <= query_positions
    minimum = torch.finfo(dtype).min
    additive = torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), minimum, dtype=dtype, device=device))
    additive = additive[None, None, :, :]
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[1] != key_length:
            raise ValueError("attention mask must cover every cached and current key position")
        visible = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        additive = torch.where(visible, additive, torch.full_like(additive, minimum))
    return additive


def eager_grouped_attention(query, key, value, additive_mask, scaling: float, groups: int):
    import torch

    repeated_key = repeat_key_value(key, groups)
    repeated_value = repeat_key_value(value, groups)
    score = torch.matmul(query, repeated_key.transpose(2, 3)) * float(scaling)
    score = score + additive_mask
    probability = torch.softmax(score, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probability, repeated_value).transpose(1, 2).contiguous()
    return output, probability


def gated_silu_mlp(hidden, gate_weight, up_weight, down_weight):
    import torch.nn.functional as functional

    return functional.linear(functional.silu(functional.linear(hidden, gate_weight))
                             * functional.linear(hidden, up_weight), down_weight)


def qwen_decoder_layer(hidden, layer_weights, config, cos, sin, additive_mask, cached_key_value=None):
    import torch
    import torch.nn.functional as functional

    batch, sequence, _ = hidden.shape
    heads = int(config["num_attention_heads"])
    key_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    residual = hidden
    normalized = rms_norm(hidden, layer_weights["input_layernorm.weight"], config["rms_norm_eps"])
    query = functional.linear(normalized, layer_weights["self_attn.q_proj.weight"])
    key = functional.linear(normalized, layer_weights["self_attn.k_proj.weight"])
    value = functional.linear(normalized, layer_weights["self_attn.v_proj.weight"])
    query = query.view(batch, sequence, heads, head_dim)
    key = key.view(batch, sequence, key_heads, head_dim)
    value = value.view(batch, sequence, key_heads, head_dim).transpose(1, 2)
    query = rms_norm(query, layer_weights["self_attn.q_norm.weight"], config["rms_norm_eps"]).transpose(1, 2)
    key = rms_norm(key, layer_weights["self_attn.k_norm.weight"], config["rms_norm_eps"]).transpose(1, 2)
    query, key = apply_rotary(query, key, cos, sin)
    if cached_key_value is not None:
        key = torch.cat((cached_key_value[0], key), dim=2)
        value = torch.cat((cached_key_value[1], value), dim=2)
    new_cache = (key, value)
    attended, _ = eager_grouped_attention(
        query, key, value, additive_mask, head_dim ** -0.5, heads // key_heads,
    )
    attended = attended.reshape(batch, sequence, heads * head_dim)
    hidden = residual + functional.linear(attended, layer_weights["self_attn.o_proj.weight"])
    residual = hidden
    normalized = rms_norm(hidden, layer_weights["post_attention_layernorm.weight"], config["rms_norm_eps"])
    hidden = residual + gated_silu_mlp(
        normalized,
        layer_weights["mlp.gate_proj.weight"],
        layer_weights["mlp.up_proj.weight"],
        layer_weights["mlp.down_proj.weight"],
    )
    return hidden, new_cache


class CapturedQwen3ForCausalLM:
    """Minimal inference-only Qwen3 model with explicit immutable weights."""

    def __init__(self, config, weights, device):
        self.config = dict(config)
        self.weights = weights
        self.device = __import__("torch").device(device)

    @classmethod
    def from_state_dict(cls, config, state_dict, device="cpu"):
        """Fixture constructor used by exact full-logit conformance tests."""
        layers = []
        suffixes = (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        )
        for layer_index in range(int(config["num_hidden_layers"])):
            prefix = f"model.layers.{layer_index}."
            layers.append({suffix: state_dict[prefix + suffix].detach().to(device) for suffix in suffixes})
        weights = {
            "embed_tokens.weight": state_dict["model.embed_tokens.weight"].detach().to(device),
            "norm.weight": state_dict["model.norm.weight"].detach().to(device),
            "lm_head.weight": state_dict["lm_head.weight"].detach().to(device),
            "layers": layers,
        }
        return cls(config, weights, device)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, device="cuda"):
        """Load only the indexed safetensor weights required by this forward."""
        import torch
        from safetensors import safe_open

        root = Path(model_path).resolve()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3" or config.get("rope_scaling") is not None:
            raise ValueError("captured runtime only supports the frozen default-RoPE Qwen3 checkpoint")
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        required = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        suffixes = (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        )
        for layer_index in range(int(config["num_hidden_layers"])):
            required.extend(f"model.layers.{layer_index}.{suffix}" for suffix in suffixes)
        if any(name not in index for name in required):
            raise ValueError("checkpoint index is missing a captured Qwen3 weight")
        loaded = {}
        by_shard = {}
        for name in required:
            by_shard.setdefault(index[name], []).append(name)
        for shard_name, names in sorted(by_shard.items()):
            with safe_open(root / shard_name, framework="pt", device="cpu") as shard:
                for name in names:
                    loaded[name] = shard.get_tensor(name).to(device=device, non_blocking=False)
        layers = []
        for layer_index in range(int(config["num_hidden_layers"])):
            prefix = f"model.layers.{layer_index}."
            layers.append({suffix: loaded.pop(prefix + suffix) for suffix in suffixes})
        weights = {
            "embed_tokens.weight": loaded.pop("model.embed_tokens.weight"),
            "norm.weight": loaded.pop("model.norm.weight"),
            "lm_head.weight": loaded.pop("lm_head.weight"),
            "layers": layers,
        }
        if loaded:
            raise ValueError("unexpected captured weights remain after load")
        torch.cuda.empty_cache()
        return cls(config, weights, device)

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask, past_key_values=None, use_cache=True,
                 return_dict=True, last_token_only=True):
        import torch
        import torch.nn.functional as functional

        if not use_cache or not return_dict:
            raise ValueError("captured Qwen runtime requires cache and return_dict")
        input_ids = input_ids.to(self.device)
        batch, sequence = input_ids.shape
        past_length = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
        positions = torch.arange(past_length, past_length + sequence, device=self.device)[None, :].expand(batch, -1)
        hidden = functional.embedding(input_ids, self.weights["embed_tokens.weight"])
        cos, sin = rotary_cos_sin(
            positions, int(self.config["head_dim"]), float(self.config["rope_theta"]), hidden.dtype, self.device,
        )
        mask = causal_attention_mask(attention_mask, sequence, past_length, hidden.dtype, self.device)
        caches = []
        for layer_index, layer_weights in enumerate(self.weights["layers"]):
            cached = None if past_key_values is None else past_key_values[layer_index]
            hidden, new_cache = qwen_decoder_layer(hidden, layer_weights, self.config, cos, sin, mask, cached)
            caches.append(new_cache)
        hidden = rms_norm(hidden, self.weights["norm.weight"], self.config["rms_norm_eps"])
        if last_token_only:
            hidden = hidden[:, -1:, :]
        logits = functional.linear(hidden, self.weights["lm_head.weight"]).float()
        return SimpleNamespace(logits=logits, past_key_values=tuple(caches))
