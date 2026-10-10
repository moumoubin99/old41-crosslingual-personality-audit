#!/usr/bin/env python3
"""Captured reference implementation of the Qwen3 decoder forward mathematics.

Purpose
-------
The deployment reviews of `timing-qwen-tuning-v5` and `-v6` rejected the vLLM
generation packet because the executed forward path bottoms out in
`torch.ops.vllm.unified_attention` -> a compiled FlashAttention CUDA kernel.
That kernel is neither capturable Python source nor runnable under the probe's
CPU constraint, so the selected formulas could never cover the mathematics the
launch actually executes.  The deployment protocol's own branch for that
situation is to *split the run into a reviewable unit*, not to omit the
dependency.

This module is that split: every operation the decoder performs is written here
as plain `torch` tensor arithmetic with no external model class, no custom
kernel and no runtime dispatch.  Each function is small enough to be bound to a
single registered formula and checked against an independently computed
expected value on the CPU.

Scope limits (deliberate, and stated so no reviewer has to infer them)
---------------------------------------------------------------------
* Nothing here is a scientific estimator.  Every artefact produced by a launch
  that uses this module carries ``eligible_for_analysis: false``.
* This module does not authorize confirmatory generation.  The blockers in
  ``refine-logs/EXPERIMENT_CODE_REVIEW.md`` remain in force.
* Training is never performed.  All tensors are produced under
  ``torch.inference_mode`` by the callers; no gradient is requested anywhere.
* The two sealed 24-seed scientific blocks are refused by :func:`check_seed`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


# The evaluation and qualification seed blocks of record (R26-F5, R26-F6).
# Apparatus work never draws from them.
SEALED_SEED_RANGES = (range(20260811, 20260811 + 24), range(20260911, 20260911 + 24))

# Configuration keys this reference consumes.  A checkpoint that omits any of
# them is refused rather than defaulted.
REQUIRED_CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
)

# ``rope_theta`` is read through ``rope_theta_of``: the released Qwen3 configs
# carry it at the top level, while configs written by transformers 5.x nest it
# under ``rope_parameters``.  Both layouts describe the same base frequency.
_ROPE_NESTS = ("rope_parameters", "rope_scaling")


def check_seed(seed):
    """Refuse negative seeds and both sealed scientific seed blocks."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    for reserved in SEALED_SEED_RANGES:
        if seed in reserved:
            raise ValueError(
                f"seed {seed} lies in a sealed scientific seed block "
                f"[{reserved.start}, {reserved.stop - 1}] and may not be used for apparatus work"
            )
    return seed


# --------------------------------------------------------------------------
# Unit 1 kernels: normalization, rotary position embedding, linear, SwiGLU MLP
# --------------------------------------------------------------------------


def accumulation_dtype(x):
    """Dtype the reductions run in: at least float32, never below the input's."""
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def rms_norm(x, weight, eps):
    """Root-mean-square layer normalization over the trailing axis.

    ``y[..., d] = x[..., d] / sqrt(mean_d(x^2) + eps) * weight[d]``

    The reciprocal square root is taken at no less than float32 regardless of
    the input dtype -- the released reference upcasts bf16 to float32 for this
    reduction -- and the result is cast back to the input dtype before the
    per-channel gain is applied.
    """
    if eps <= 0.0:
        raise ValueError(f"rms_norm eps must be positive, got {eps}")
    if weight.shape[-1] != x.shape[-1]:
        raise ValueError(f"rms_norm weight width {weight.shape[-1]} != input width {x.shape[-1]}")
    dtype = x.dtype
    wide = x.to(accumulation_dtype(x))
    scale = torch.rsqrt(wide.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (wide * scale).to(dtype) * weight.to(dtype)


def rope_tables(head_dim, positions, theta, device=None, dtype=torch.float32):
    """Cosine/sine tables for rotary position embedding.

    ``inv_freq[j] = theta ** (-2j / head_dim)`` for ``j`` in ``[0, head_dim/2)``
    and the returned tables repeat that half twice along the last axis, which
    is the layout the half-split rotation in :func:`apply_rope` expects.
    Returned shape: ``(len(positions), head_dim)``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for rotary embedding, got {head_dim}")
    if theta <= 0.0:
        raise ValueError(f"rope theta must be positive, got {theta}")
    pos = torch.as_tensor(positions, dtype=torch.float64, device=device)
    if pos.ndim != 1:
        raise ValueError(f"positions must be one-dimensional, got shape {tuple(pos.shape)}")
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim
    inv_freq = torch.pow(torch.tensor(float(theta), dtype=torch.float64, device=device), -exponents)
    freqs = pos[:, None] * inv_freq[None, :]
    table = torch.cat((freqs, freqs), dim=-1)
    return table.cos().to(dtype), table.sin().to(dtype)


def rotate_half(x):
    """Half-split rotation: ``[a, b] -> [-b, a]`` over the trailing axis."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    """Apply rotary position embedding to ``x`` of shape ``(heads, tokens, head_dim)``."""
    if x.shape[-2] != cos.shape[0]:
        raise ValueError(f"rope table covers {cos.shape[0]} positions but input has {x.shape[-2]}")
    if x.shape[-1] != cos.shape[-1]:
        raise ValueError(f"rope table width {cos.shape[-1]} != head_dim {x.shape[-1]}")
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


def linear(x, weight):
    """Bias-free projection ``y = x @ weight.T`` (Qwen3 carries no QKV bias)."""
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(f"linear input width {x.shape[-1]} != weight input width {weight.shape[-1]}")
    return x @ weight.transpose(-1, -2).to(x.dtype)


def silu(x):
    """``x * sigmoid(x)``."""
    return x * torch.sigmoid(x)


def swiglu_mlp(x, gate_weight, up_weight, down_weight):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""
    gate = linear(x, gate_weight)
    up = linear(x, up_weight)
    if gate.shape != up.shape:
        raise ValueError(f"gate shape {tuple(gate.shape)} != up shape {tuple(up.shape)}")
    return linear(silu(gate) * up, down_weight)


# --------------------------------------------------------------------------
# Unit 2 kernels: grouped-query repetition and scaled causal attention
# --------------------------------------------------------------------------


def repeat_kv(x, n_rep):
    """Expand ``(kv_heads, tokens, head_dim)`` to ``(kv_heads*n_rep, tokens, head_dim)``.

    Head ``h`` of the expanded tensor reads key/value head ``h // n_rep``; the
    repetition is contiguous per source head, which is the grouping the query
    head order assumes.
    """
    if n_rep < 1:
        raise ValueError(f"n_rep must be >= 1, got {n_rep}")
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=0)


def causal_mask(q_len, kv_len, device=None):
    """Boolean keep-mask of shape ``(q_len, kv_len)``.

    Query ``i`` is the token at absolute position ``kv_len - q_len + i``, so it
    may read key ``j`` exactly when ``j <= kv_len - q_len + i``.  This is the
    single definition used by both the full-prefix and the cached incremental
    paths, which is why they agree.
    """
    offset = kv_len - q_len
    if offset < 0:
        raise ValueError(f"kv_len {kv_len} is shorter than q_len {q_len}")
    rows = torch.arange(q_len, device=device)[:, None] + offset
    cols = torch.arange(kv_len, device=device)[None, :]
    return cols <= rows


def scaled_causal_attention(q, k, v, scale=None):
    """Softmax attention with a causal mask.

    ``q``: ``(heads, q_len, head_dim)``; ``k``/``v``: ``(heads, kv_len, head_dim)``.
    Scores are computed and normalized in float32 and cast back to ``q``'s dtype.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("attention expects three-dimensional (head, token, feature) tensors")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("query, key and value must carry the same head count")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("key and value must cover the same token count")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key head_dim must match")
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    wide = accumulation_dtype(q)
    scores = (q.to(wide) @ k.to(wide).transpose(-1, -2)) * scale
    keep = causal_mask(q.shape[-2], k.shape[-2], device=q.device)
    scores = scores.masked_fill(~keep[None, :, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return (weights @ v.to(wide)).to(q.dtype), weights


# --------------------------------------------------------------------------
# Unit 3: decoder layer, model forward, incremental cache, samplers
# --------------------------------------------------------------------------


def head_dim_of(config):
    """Head width, taken from the config when present and derived otherwise."""
    if config.get("head_dim"):
        return int(config["head_dim"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    if hidden % heads != 0:
        raise ValueError(f"hidden_size {hidden} is not divisible by num_attention_heads {heads}")
    return hidden // heads


def rope_theta_of(config):
    """Rotary base frequency, from either config layout, never defaulted."""
    if config.get("rope_theta") is not None:
        return float(config["rope_theta"])
    for nest in _ROPE_NESTS:
        nested = config.get(nest)
        if isinstance(nested, dict) and nested.get("rope_theta") is not None:
            return float(nested["rope_theta"])
    raise ValueError("configuration declares no rope_theta at the top level or under "
                     f"{' / '.join(_ROPE_NESTS)}")


def config_of_record(config):
    """The configuration fields this reference consumed, for the launch receipt."""
    record = {key: config[key] for key in REQUIRED_CONFIG_KEYS if key != "rope_theta"}
    record["rope_theta"] = rope_theta_of(config)
    return record


def validate_config(config):
    """Reject a configuration this reference cannot faithfully execute."""
    missing = [key for key in REQUIRED_CONFIG_KEYS
               if key != "rope_theta" and key not in config]
    if missing:
        raise ValueError(f"configuration is missing required keys: {missing}")
    rope_theta_of(config)
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    if kv_heads < 1 or heads % kv_heads != 0:
        raise ValueError(f"num_attention_heads {heads} is not a multiple of num_key_value_heads {kv_heads}")
    head_dim_of(config)
    return config


def decoder_layer(hidden, params, config, cache=None, positions=None):
    """One pre-norm decoder block.

    ``hidden``: ``(tokens, hidden_size)``.  ``cache``, when given, is the
    ``(key, value)`` pair already accumulated for this layer, each of shape
    ``(kv_heads, past, head_dim)``; the returned cache appends this call's
    keys and values.  ``positions`` are the absolute positions of the incoming
    tokens and default to the contiguous block that follows the cache.
    """
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = head_dim_of(config)
    eps = float(config["rms_norm_eps"])
    theta = rope_theta_of(config)
    tokens = hidden.shape[-2]
    past = 0 if cache is None else cache[0].shape[-2]
    if positions is None:
        positions = list(range(past, past + tokens))
    if len(positions) != tokens:
        raise ValueError(f"{len(positions)} positions supplied for {tokens} tokens")

    normed = rms_norm(hidden, params["input_norm"], eps)
    q = linear(normed, params["q_proj"]).view(tokens, heads, head_dim).transpose(0, 1)
    k = linear(normed, params["k_proj"]).view(tokens, kv_heads, head_dim).transpose(0, 1)
    v = linear(normed, params["v_proj"]).view(tokens, kv_heads, head_dim).transpose(0, 1)

    # Qwen3 normalizes each query and key head before the rotary rotation.
    if params.get("q_norm") is not None:
        q = rms_norm(q, params["q_norm"], eps)
    if params.get("k_norm") is not None:
        k = rms_norm(k, params["k_norm"], eps)

    cos, sin = rope_tables(head_dim, positions, theta, device=hidden.device, dtype=hidden.dtype)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if cache is not None:
        k = torch.cat((cache[0].to(k.dtype), k), dim=-2)
        v = torch.cat((cache[1].to(v.dtype), v), dim=-2)
    new_cache = (k, v)

    n_rep = heads // kv_heads
    context, _ = scaled_causal_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep))
    context = context.transpose(0, 1).reshape(tokens, heads * head_dim)
    hidden = hidden + linear(context, params["o_proj"])

    normed = rms_norm(hidden, params["post_attn_norm"], eps)
    hidden = hidden + swiglu_mlp(normed, params["gate_proj"], params["up_proj"], params["down_proj"])
    return hidden, new_cache


def model_logits(params, config, token_ids, cache=None, positions=None):
    """Vocabulary logits for ``token_ids``.

    Returns ``(logits, cache)`` where ``logits`` has shape
    ``(len(token_ids), vocab_size)`` and ``cache`` is the per-layer key/value
    list after this call.  Passing the cache back in with the next token is the
    incremental path; passing ``cache=None`` with the whole prefix is the
    full-prefix path.  The two agree by construction because they share
    :func:`causal_mask` and the same absolute positions.
    """
    validate_config(config)
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=params["embed"].device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("token_ids must be a non-empty one-dimensional sequence")
    if int(ids.max()) >= params["embed"].shape[0] or int(ids.min()) < 0:
        raise ValueError("token id outside the embedding table")
    layers = int(config["num_hidden_layers"])
    if len(params["layers"]) != layers:
        raise ValueError(f"{len(params['layers'])} layer parameter sets for {layers} declared layers")
    if cache is not None and len(cache) != layers:
        raise ValueError(f"cache holds {len(cache)} layers for {layers} declared layers")

    hidden = params["embed"][ids]
    past = 0 if cache is None else cache[0][0].shape[-2]
    if positions is None:
        positions = list(range(past, past + ids.numel()))
    new_cache = []
    for index in range(layers):
        layer_cache = None if cache is None else cache[index]
        hidden, updated = decoder_layer(hidden, params["layers"][index], config, layer_cache, positions)
        new_cache.append(updated)
    hidden = rms_norm(hidden, params["final_norm"], float(config["rms_norm_eps"]))
    return linear(hidden, params["lm_head"]), new_cache


def greedy_next_token(logits):
    """Argmax over the final row's vocabulary axis."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be (tokens, vocab), got shape {tuple(logits.shape)}")
    return int(torch.argmax(logits[-1].to(torch.float32)).item())


def nucleus_next_token(logits, temperature, top_p, generator):
    """Temperature + nucleus sampling from the final row.

    The kept set is the shortest descending-probability prefix whose cumulative
    probability reaches ``top_p``; the first token is always kept, so the set is
    never empty.  ``temperature`` 0 degenerates to the greedy choice.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (tokens, vocab), got shape {tuple(logits.shape)}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must lie in (0, 1], got {top_p}")
    if temperature == 0.0:
        return greedy_next_token(logits)
    row = logits[-1].to(torch.float32) / temperature
    probs = torch.softmax(row, dim=-1)
    ordered, order = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(ordered, dim=-1)
    keep = cumulative - ordered < top_p
    keep[0] = True
    ordered = torch.where(keep, ordered, torch.zeros_like(ordered))
    ordered = ordered / ordered.sum()
    picked = int(torch.multinomial(ordered, 1, generator=generator).item())
    return int(order[picked].item())


def generate(params, config, prompt_ids, max_new_tokens, temperature=0.0, top_p=1.0,
             generator=None, stop_ids=()):
    """Autoregressive decode through the captured cache path.

    Returns ``(generated_ids, prompt_logits)``.  Only the captured functions in
    this module are executed; no external generation loop participates.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if temperature > 0.0 and generator is None:
        raise ValueError("stochastic decoding requires an explicit seeded generator")
    logits, cache = model_logits(params, config, prompt_ids)
    prompt_logits = logits
    produced = []
    for _ in range(max_new_tokens):
        if temperature > 0.0:
            token = nucleus_next_token(logits, temperature, top_p, generator)
        else:
            token = greedy_next_token(logits)
        produced.append(token)
        if token in tuple(stop_ids):
            break
        logits, cache = model_logits(params, config, [token], cache=cache)
    return produced, prompt_logits


# --------------------------------------------------------------------------
# Fixtures and checkpoint identity
# --------------------------------------------------------------------------


FIXTURE_CONFIG = {
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "intermediate_size": 24,
    "vocab_size": 11,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


def fixture_params(config, seed, device=None, dtype=torch.float32):
    """Deterministic small parameter set for the executable contracts.

    Weights are drawn from one CPU generator in a fixed order so an independent
    re-derivation can reproduce them exactly.  Norm weights are drawn around
    one, projections around zero with a small scale, which keeps the fixture's
    logits well away from float32 saturation.
    """
    check_seed(seed)
    validate_config(config)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = head_dim_of(config)
    inter = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])

    def draw(*shape, centre=0.0, scale=0.2):
        tensor = torch.rand(*shape, generator=generator, dtype=torch.float32) - 0.5
        return (tensor * (2.0 * scale) + centre)

    params = {
        "embed": draw(vocab, hidden),
        "layers": [],
        "final_norm": draw(hidden, centre=1.0, scale=0.05),
        "lm_head": draw(vocab, hidden),
    }
    for _ in range(int(config["num_hidden_layers"])):
        params["layers"].append({
            "input_norm": draw(hidden, centre=1.0, scale=0.05),
            "q_proj": draw(heads * head_dim, hidden),
            "k_proj": draw(kv_heads * head_dim, hidden),
            "v_proj": draw(kv_heads * head_dim, hidden),
            "q_norm": draw(head_dim, centre=1.0, scale=0.05),
            "k_norm": draw(head_dim, centre=1.0, scale=0.05),
            "o_proj": draw(hidden, heads * head_dim),
            "post_attn_norm": draw(hidden, centre=1.0, scale=0.05),
            "gate_proj": draw(inter, hidden),
            "up_proj": draw(inter, hidden),
            "down_proj": draw(hidden, inter),
        })
    return to_device(params, device, dtype)


def to_device(params, device=None, dtype=None):
    """Move a parameter tree to a device/dtype without changing its structure."""
    def move(tensor):
        if tensor is None:
            return None
        return tensor.to(device=device, dtype=dtype) if (device or dtype) else tensor

    return {
        "embed": move(params["embed"]),
        "layers": [{key: move(value) for key, value in layer.items()} for layer in params["layers"]],
        "final_norm": move(params["final_norm"]),
        "lm_head": move(params["lm_head"]),
    }


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(model_path):
    """Config fields and per-file digests that pin the checkpoint actually loaded.

    No expected digest is written into this repository: the released weights are
    not reachable from the authoring sandbox, so the launch records what it read
    and the reviewer compares it against the checkpoint on the deployment host.
    """
    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"no config.json under {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    weights = sorted(p for p in root.iterdir() if p.suffix in (".safetensors", ".bin"))
    return {
        "model_path": str(root),
        "config_sha256": sha256_file(config_path),
        "architectures": config.get("architectures"),
        "config_of_record": config_of_record(config),
        "head_dim": head_dim_of(config),
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
        "weight_files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in weights
        ],
    }


def load_checkpoint_params(model_path, device=None, dtype=torch.bfloat16):
    """Read the released weights into this reference's parameter tree.

    Only ``safetensors`` tensor reads are used; no model class is constructed,
    so nothing outside this module participates in the forward pass.
    """
    from safetensors.torch import load_file

    root = Path(model_path)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shard under {root}")
    tensors = {}
    for shard in shards:
        tensors.update(load_file(str(shard)))

    def get(name):
        if name not in tensors:
            raise ValueError(f"checkpoint is missing tensor {name}")
        return tensors[name].to(device=device, dtype=dtype)

    prefix = "model."
    params = {
        "embed": get(prefix + "embed_tokens.weight"),
        "layers": [],
        "final_norm": get(prefix + "norm.weight"),
        "lm_head": None,
    }
    params["lm_head"] = params["embed"] if config.get("tie_word_embeddings") else get("lm_head.weight")
    for index in range(int(config["num_hidden_layers"])):
        base = f"{prefix}layers.{index}."
        params["layers"].append({
            "input_norm": get(base + "input_layernorm.weight"),
            "q_proj": get(base + "self_attn.q_proj.weight"),
            "k_proj": get(base + "self_attn.k_proj.weight"),
            "v_proj": get(base + "self_attn.v_proj.weight"),
            "q_norm": tensors.get(base + "self_attn.q_norm.weight", None),
            "k_norm": tensors.get(base + "self_attn.k_norm.weight", None),
            "o_proj": get(base + "self_attn.o_proj.weight"),
            "post_attn_norm": get(base + "post_attention_layernorm.weight"),
            "gate_proj": get(base + "mlp.gate_proj.weight"),
            "up_proj": get(base + "mlp.up_proj.weight"),
            "down_proj": get(base + "mlp.down_proj.weight"),
        })
        for key in ("q_norm", "k_norm"):
            value = params["layers"][-1][key]
            if value is not None:
                params["layers"][-1][key] = value.to(device=device, dtype=dtype)
    return params, config
