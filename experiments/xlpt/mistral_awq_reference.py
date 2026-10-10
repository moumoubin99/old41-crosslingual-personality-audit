#!/usr/bin/env python3
"""Captured pure-torch reference for the frozen int4-AWQ Mistral rater.

Why this file exists
--------------------
The deployment review of ``english-b0-rater-v2`` rejected the rater packet for
the same reason the vLLM Qwen packet was rejected twice: the launched forward
bottoms out in fused CUDA kernels (AWQ GEMM, FlashAttention, fused RMSNorm and
rotary) that are not capturable Python source, so they can neither be bound to
a formula nor executed under the deployment probe's CPU constraint.  The
reviewer additionally found that the thirteen ``rater_*`` formulas bound in
``ENGLISH_EXECUTION_MATH_SPEC.json`` were tautological or wrong in six places,
and that five material operations of the run carried no formula at all.

This module is the capture side of that answer.  It implements the complete
int4-AWQ Mistral decode path in readable torch, with every operation the
reviewer named written out rather than delegated:

* packed AWQ extraction (bit unpacking, group indexing) and the equivalence of
  the dequantize-then-matmul path with the fused GEMM the runtime calls;
* embedding lookup defined from token ids and a table, not asserted;
* RoPE inverse frequencies, table construction, rotate-half and application,
  all position-dependent;
* QKV projection with its split and reshape axes made explicit;
* grouped-query repetition whose output head count is ``key_head * groups``;
* attention with the QK scaling and the causal mask inside the expression;
* KV-cache content and position updates;
* vocabulary-parallel logit concatenation;
* EOS-set membership stopping, not scalar equality.

Nothing here trains, fine-tunes or updates a weight, and nothing here produces
a rubric score.  It is apparatus mathematics: the reviewable statement of what
the rater path computes.  Whether the launched rater is then routed through
these functions or bound to the production symbols is a separate decision that
this file does not make and must not be read as having made.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


# --------------------------------------------------------------------------
# Sealed-seed refusal, shared with the Qwen apparatus.
# --------------------------------------------------------------------------

SEALED_SEED_RANGES = (
    range(20260811, 20260811 + 24),
    range(20260911, 20260911 + 24),
)


def check_seed(seed: int) -> int:
    """Refuse both sealed 24-seed blocks; apparatus work runs outside them."""
    value = int(seed)
    for block in SEALED_SEED_RANGES:
        if value in block:
            raise ValueError(
                "seed %d belongs to a sealed confirmatory block; apparatus "
                "verification may not read it" % value
            )
    return value


# --------------------------------------------------------------------------
# Packed int4 AWQ: extraction, dequantization, and fused-GEMM equivalence.
# --------------------------------------------------------------------------

# AutoAWQ/vLLM store eight 4-bit values per int32 column in this nibble order.
AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)

AWQ_BITS = 4
AWQ_PACK_FACTOR = 32 // AWQ_BITS


def awq_extract(packed: torch.Tensor) -> torch.Tensor:
    """Extract every int32 column to ``[rows, packed_cols, 8]``.

    ``result[r, c, k]`` is nibble ``AWQ_UNPACK_ORDER[k]`` of packed column
    ``c``.  Keeping extraction three-dimensional makes its formula/output
    contract exact; :func:`awq_unpack` owns the separately explicit flatten.
    """
    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError("packed AWQ values must be a rank-two int32 tensor")
    shifts = torch.tensor(
        AWQ_UNPACK_ORDER, dtype=torch.int64, device=packed.device
    ) * AWQ_BITS
    return (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF


def awq_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Flatten extracted nibbles to deployed ``[rows, packed_cols * 8]``."""
    nibbles = awq_extract(packed)
    return nibbles.reshape(packed.shape[0], packed.shape[1] * AWQ_PACK_FACTOR)


def awq_group_index(input_dim: int, group_size: int, device=None) -> torch.Tensor:
    """Row -> group map: row ``i`` is scaled by group ``i // group_size``."""
    if group_size <= 0:
        raise ValueError("AWQ group size must be positive")
    if input_dim % group_size:
        raise ValueError("AWQ group size must divide the input dimension")
    return torch.arange(input_dim, device=device) // group_size


def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """W[i, j] = (unpack(qweight)[i, j] - unpack(qzeros)[i // g, j]) * scales[i // g, j].

    Shapes follow the stored layout: ``qweight`` is ``[input_dim,
    output_dim / 8]`` int32, ``qzeros`` is ``[groups, output_dim / 8]`` int32,
    and ``scales`` is ``[groups, output_dim]``.  The returned weight is
    ``[input_dim, output_dim]`` in the scales' dtype, oriented for ``x @ W``.
    """
    weight = awq_unpack(qweight)
    zeros = awq_unpack(qzeros)
    input_dim, output_dim = weight.shape
    groups = input_dim // group_size if group_size else 0
    if zeros.shape != (groups, output_dim):
        raise ValueError("AWQ zero points do not cover every group and output")
    if tuple(scales.shape) != (groups, output_dim):
        raise ValueError("AWQ scales do not cover every group and output")
    index = awq_group_index(input_dim, group_size, device=qweight.device)
    centered = weight - zeros[index]
    return centered.to(scales.dtype) * scales[index]


def awq_linear(
    hidden: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The fused AWQ GEMM's mathematical content: ``x @ dequantize(...)``.

    The runtime calls one fused kernel; that kernel's contract is exactly this
    composition.  Stating it as a composition is what makes the claim testable:
    ``test_mistral_awq_quant.py`` checks that a hand-built packed weight
    dequantizes to a known matrix and that this product equals an independent
    scalar accumulation over the same numbers.
    """
    weight = awq_dequantize(qweight, scales, qzeros, group_size)
    if hidden.shape[-1] != weight.shape[0]:
        raise ValueError("hidden width does not match the AWQ input dimension")
    out = torch.matmul(hidden.to(weight.dtype), weight)
    if bias is not None:
        out = out + bias
    return out


# --------------------------------------------------------------------------
# Elementwise and projection kernels.
# --------------------------------------------------------------------------


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulate in float32 for reduced precision, but never DOWNCAST.

    The runtime keeps its float32 reductions over bfloat16 activations; a
    blanket ``.float()`` would also silently truncate a float64 reference
    computation, which is how a scalar-oracle test first caught this.
    """
    return dtype if dtype == torch.float64 else torch.float32


def embedding_lookup(token_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """E[b, t, :] = table[token_ids[b, t], :].

    Defined as a gather from the table, so the formula has content: the vocab
    axis of ``table`` is indexed by the id, and the hidden axis survives.
    """
    if table.ndim != 2:
        raise ValueError("embedding table must be [vocabulary, hidden]")
    ids = token_ids.to(torch.int64)
    if int(ids.min()) < 0 or int(ids.max()) >= table.shape[0]:
        raise ValueError("token id outside the embedding table")
    flat = table.index_select(0, ids.reshape(-1))
    return flat.reshape(*ids.shape, table.shape[1])


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """y = w * x / sqrt(mean(x^2) + eps), accumulated in float32."""
    if epsilon <= 0:
        raise ValueError("RMS epsilon must be positive")
    if weight.shape[-1] != hidden.shape[-1]:
        raise ValueError("RMS weight width does not match the hidden width")
    accumulated = hidden.to(_accumulation_dtype(hidden.dtype))
    scaled = accumulated * torch.rsqrt(
        accumulated.pow(2).mean(dim=-1, keepdim=True) + float(epsilon)
    )
    return scaled.to(hidden.dtype) * weight


def silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def gated_silu_mlp(
    hidden: torch.Tensor,
    gate: Mapping[str, object],
    up: Mapping[str, object],
    down: Mapping[str, object],
) -> torch.Tensor:
    """down( silu(gate(x)) * up(x) ), every projection an AWQ GEMM."""
    activated = silu(awq_linear(hidden, **gate))
    return awq_linear(activated * awq_linear(hidden, **up), **down)


def linear(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Unquantized projection ``x @ W^T`` for the norm-scaled head path."""
    if hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("linear input width does not match the weight width")
    return torch.matmul(hidden, weight.transpose(-1, -2))


# --------------------------------------------------------------------------
# Rotary position embedding: frequencies, tables, rotate-half, application.
# --------------------------------------------------------------------------


def rope_inverse_frequencies(
    head_dim: int, theta: float, device=None
) -> torch.Tensor:
    """inv_freq[k] = theta ** (-2k / head_dim) for k in [0, head_dim/2).

    Computed in float64: at head_dim 128 and theta 1e6 the float32 rounding of
    the exponent is large enough to move the last table entries.
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("rotary head dimension must be positive and even")
    if theta <= 0:
        raise ValueError("rotary theta must be positive")
    index = torch.arange(0, head_dim, 2, dtype=torch.float64, device=device)
    return 1.0 / (float(theta) ** (index / float(head_dim)))


def rope_angle_pairs(
    positions: torch.Tensor, head_dim: int, theta: float
) -> torch.Tensor:
    """Return duplicated rotary angles as ``[..., 2, head_dim / 2]``."""
    inverse = rope_inverse_frequencies(head_dim, theta, device=positions.device)
    half = positions.to(torch.float64).unsqueeze(-1) * inverse
    return torch.stack((half, half), dim=-2)


def rope_trig_tables(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unflattened cosine/sine pairs used by :func:`rope_tables`."""
    angles = rope_angle_pairs(positions, head_dim, theta)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rope_tables(
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) of shape ``[*positions.shape, head_dim]``.

    The angle at position ``p`` and pair ``k`` is ``p * inv_freq[k]``; the
    half-width table is duplicated so that it lines up with ``rotate_half``.
    Position dependence is the point: two different positions must produce two
    different rows, which the contract test checks directly.
    """
    cosine, sine = rope_trig_tables(positions, head_dim, theta, dtype)
    return cosine.flatten(-2), sine.flatten(-2)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """(-x2, x1) split at the head-dimension midpoint."""
    if value.shape[-1] % 2:
        raise ValueError("rotary head dimension must be even")
    midpoint = value.shape[-1] // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def apply_rope(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    """y = x * cos + rotate_half(x) * sin, broadcast over the head axis."""
    if cosine.shape != sine.shape:
        raise ValueError("rotary cosine and sine tables must have equal shapes")
    if cosine.shape[-1] != value.shape[-1]:
        raise ValueError("rotary table width does not match the head dimension")
    return value * cosine + rotate_half(value) * sine


# --------------------------------------------------------------------------
# Attention: QKV split, grouped-query repetition, scaled masked attention.
# --------------------------------------------------------------------------


def split_qkv(
    fused: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[batch, seq, (q + k + v) * head_dim]`` into head-major tensors.

    The reviewer asked for the split and reshape axes to be material rather
    than implied.  They are: the last axis is cut at ``num_heads * head_dim``
    and again ``num_key_value_heads * head_dim`` later; each piece is reshaped
    to ``[batch, seq, heads, head_dim]`` and transposed to
    ``[batch, heads, seq, head_dim]``.  Reshape before transpose -- the other
    order silently interleaves heads with positions.
    """
    if fused.ndim != 3:
        raise ValueError("fused QKV must be [batch, sequence, width]")
    q_width = num_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    if fused.shape[-1] != q_width + 2 * kv_width:
        raise ValueError("fused QKV width does not match the head configuration")
    query, key, value = fused.split([q_width, kv_width, kv_width], dim=-1)
    batch, sequence, _ = fused.shape

    def to_heads(part: torch.Tensor, heads: int) -> torch.Tensor:
        return part.reshape(batch, sequence, heads, head_dim).transpose(1, 2)

    return (
        to_heads(query, num_heads),
        to_heads(key, num_key_value_heads),
        to_heads(value, num_key_value_heads),
    )


def repeat_kv_head(
    hidden: torch.Tensor, groups: int, query_head_index: int
) -> torch.Tensor:
    """Select the source KV head for one flattened query-head index."""
    if groups <= 0:
        raise ValueError("grouped-query repetition factor must be positive")
    if hidden.ndim != 4:
        raise ValueError("key/value must be [batch, key_head, sequence, head_dim]")
    total = hidden.shape[1] * groups
    if query_head_index < 0 or query_head_index >= total:
        raise ValueError("query head index is outside the repeated head range")
    return hidden[:, query_head_index // groups]


def repeat_kv(hidden: torch.Tensor, groups: int) -> torch.Tensor:
    """Expand ``[batch, key_head, seq, head_dim]`` to ``key_head * groups`` heads.

    The output head count is ``key_head * groups`` and equals the query head
    count; the previous spec entry claimed ``key_head``, which is the input.
    """
    if groups <= 0:
        raise ValueError("grouped-query repetition factor must be positive")
    if hidden.ndim != 4:
        raise ValueError("key/value must be [batch, key_head, sequence, head_dim]")
    return torch.stack(
        [repeat_kv_head(hidden, groups, index)
         for index in range(hidden.shape[1] * groups)],
        dim=1,
    )


def causal_mask(
    query_len: int, key_len: int, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """Additive mask: 0 where key <= query (in absolute positions), else -inf.

    With a KV cache the query block sits at the end of the key block, so query
    ``i`` may attend to keys ``0 .. key_len - query_len + i``.
    """
    if query_len < 0 or key_len < query_len:
        raise ValueError("key length must be at least the query length")
    offset = key_len - query_len
    rows = torch.arange(query_len, device=device).unsqueeze(-1) + offset
    cols = torch.arange(key_len, device=device).unsqueeze(0)
    allowed = cols <= rows
    mask = torch.zeros(query_len, key_len, dtype=dtype, device=device)
    return mask.masked_fill(~allowed, float("-inf"))


def scaled_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    groups: int,
    scaling: float | None = None,
) -> torch.Tensor:
    """softmax((QK^T) * s + M) V, with the scale and the mask inside.

    ``s`` defaults to ``head_dim ** -0.5``.  The softmax accumulates in float32
    and the result returns to the query dtype.  Keeping the scale and mask in
    the expression is the correction the reviewer required: the previous entry
    softmaxed a score tensor supplied from outside, which asserted nothing.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention inputs must be [batch, head, sequence, head_dim]")
    head_dim = query.shape[-1]
    scale = float(head_dim) ** -0.5 if scaling is None else float(scaling)
    repeated_key = repeat_kv(key, groups)
    repeated_value = repeat_kv(value, groups)
    if repeated_key.shape[1] != query.shape[1]:
        raise ValueError("repeated key heads do not match the query head count")
    accumulate = _accumulation_dtype(query.dtype)
    scores = torch.matmul(
        query.to(accumulate), repeated_key.to(accumulate).transpose(-1, -2)
    ) * scale
    mask = causal_mask(
        query.shape[-2], repeated_key.shape[-2], dtype=scores.dtype, device=scores.device
    )
    probability = torch.softmax(scores + mask, dim=-1)
    return torch.matmul(probability, repeated_value.to(accumulate)).to(query.dtype)


# --------------------------------------------------------------------------
# KV cache: content and position updates.
# --------------------------------------------------------------------------


def cache_append(
    cached: torch.Tensor | None, fresh: torch.Tensor
) -> torch.Tensor:
    """Concatenate along the sequence axis; the cache grows at axis -2."""
    if fresh.ndim != 4:
        raise ValueError("cache entries must be [batch, head, sequence, head_dim]")
    if cached is None:
        return fresh
    if cached.shape[:-2] != fresh.shape[:-2] or cached.shape[-1] != fresh.shape[-1]:
        raise ValueError("cache and fresh entry disagree outside the sequence axis")
    return torch.cat((cached, fresh), dim=-2)


def cache_next_length(attention_length: int, appended: int = 1) -> int:
    """L' = L + appended: the cached key length after one decode step."""
    if attention_length < 0 or appended < 0:
        raise ValueError("cache lengths must be nonnegative")
    return int(attention_length) + int(appended)


def next_positions(past_length: int, count: int, device=None) -> torch.Tensor:
    """Absolute positions ``past_length .. past_length + count - 1``.

    The position update is what makes the cached path agree with the uncached
    one: a decode step at cache length L is at position L, not at position 0.
    """
    if past_length < 0 or count <= 0:
        raise ValueError("position updates need a nonnegative past and a positive count")
    return torch.arange(past_length, past_length + count, device=device)


# --------------------------------------------------------------------------
# Decoder layer, model, vocabulary-parallel head.
# --------------------------------------------------------------------------


def decoder_layer(
    hidden: torch.Tensor,
    params: Mapping[str, object],
    config: Mapping[str, object],
    positions: torch.Tensor,
    cached_key: torch.Tensor | None = None,
    cached_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One pre-norm Mistral block; returns (hidden, new_key, new_value).

    Residual structure: ``h = h + attn(norm1(h))`` then ``h = h + mlp(norm2(h))``.
    """
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    epsilon = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    groups = heads // kv_heads
    if heads % kv_heads:
        raise ValueError("query heads must be a multiple of key/value heads")

    normed = rms_norm(hidden, params["input_layernorm"], epsilon)
    fused = awq_linear(normed, **params["qkv_proj"])
    query, key, value = split_qkv(fused, heads, kv_heads, head_dim)

    cosine, sine = rope_tables(positions, head_dim, theta, dtype=query.dtype)
    cosine = cosine.unsqueeze(0).unsqueeze(0)
    sine = sine.unsqueeze(0).unsqueeze(0)
    query = apply_rope(query, cosine, sine)
    key = apply_rope(key, cosine, sine)

    key = cache_append(cached_key, key)
    value = cache_append(cached_value, value)

    attended = scaled_causal_attention(query, key, value, groups)
    batch, _, sequence, _ = attended.shape
    merged = attended.transpose(1, 2).reshape(batch, sequence, heads * head_dim)
    residual = hidden
    attention_branch = awq_linear(merged, **params["o_proj"])
    post_attention = residual + attention_branch

    normed = rms_norm(post_attention, params["post_attention_layernorm"], epsilon)
    mlp_branch = gated_silu_mlp(
        normed, params["gate_proj"], params["up_proj"], params["down_proj"]
    )
    hidden = compose_decoder_residual(residual, attention_branch, mlp_branch)
    return hidden, key, value


def compose_decoder_residual(
    residual: torch.Tensor,
    attention_branch: torch.Tensor,
    mlp_branch: torch.Tensor,
) -> torch.Tensor:
    """Compose both Mistral residual additions in their execution order."""
    if residual.shape != attention_branch.shape or residual.shape != mlp_branch.shape:
        raise ValueError("decoder residual branches must have equal shapes")
    return residual + attention_branch + mlp_branch


def vocabulary_parallel_logit(
    shards: Sequence[torch.Tensor], vocabulary_index: int
) -> torch.Tensor:
    """Return one full-vocabulary logit column from rank-ordered shards."""
    if not shards:
        raise ValueError("vocabulary-parallel gather needs at least one shard")
    leading = shards[0].shape[:-1]
    for shard in shards:
        if shard.shape[:-1] != leading:
            raise ValueError("logit shards disagree outside the vocabulary axis")
    total = sum(shard.shape[-1] for shard in shards)
    if vocabulary_index < 0 or vocabulary_index >= total:
        raise ValueError("vocabulary index is outside the gathered range")
    offset = int(vocabulary_index)
    for shard in shards:
        if offset < shard.shape[-1]:
            return shard[..., offset]
        offset -= shard.shape[-1]
    raise AssertionError("validated vocabulary index was not resolved")


def vocabulary_parallel_logits(shards: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-shard logits along the vocabulary axis, in rank order.

    The runtime splits the head across tensor-parallel ranks even at world size
    one; the gathered result is the concatenation in rank order, and rank order
    is load-bearing -- a permuted gather silently renames every token.
    """
    if not shards:
        raise ValueError("vocabulary-parallel gather needs at least one shard")
    total = sum(shard.shape[-1] for shard in shards)
    return torch.stack(
        [vocabulary_parallel_logit(shards, index) for index in range(total)],
        dim=-1,
    )


def model_logits(
    token_ids: torch.Tensor,
    params: Mapping[str, object],
    config: Mapping[str, object],
    past_length: int = 0,
    cache: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Full forward: embed, decode every layer, final-norm, project to vocab."""
    hidden = embedding_lookup(token_ids, params["embed_tokens"])
    positions = next_positions(past_length, token_ids.shape[-1], device=hidden.device)
    layers = params["layers"]
    if cache is not None and len(cache) != len(layers):
        raise ValueError("cache depth does not match the layer count")

    new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, layer in enumerate(layers):
        cached_key, cached_value = (None, None) if cache is None else cache[index]
        hidden, key, value = decoder_layer(
            hidden, layer, config, positions, cached_key, cached_value
        )
        new_cache.append((key, value))

    normed = rms_norm(hidden, params["norm"], float(config["rms_norm_eps"]))
    head = params["lm_head"]
    if isinstance(head, (list, tuple)):
        shards = [linear(normed, shard) for shard in head]
        logits = vocabulary_parallel_logits(shards)
    else:
        logits = linear(normed, head)
    return logits.float(), new_cache


# --------------------------------------------------------------------------
# Token selection and stopping.
# --------------------------------------------------------------------------


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Final-axis argmax; torch returns the first maximum, fixing ties."""
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have a nonempty vocabulary axis")
    return logits.argmax(dim=-1)


def eos_stop(token: int, eos_tokens: Iterable[int]) -> bool:
    """Stop when the token is a MEMBER of the EOS set.

    Mistral-Small ships several terminal ids; scalar equality against one of
    them under-stops, which is the defect the reviewer named.
    """
    members = {int(value) for value in eos_tokens}
    if not members:
        raise ValueError("the EOS set must contain at least one token")
    return int(token) in members


def greedy_generate(
    token_ids: torch.Tensor,
    params: Mapping[str, object],
    config: Mapping[str, object],
    max_new_tokens: int,
    eos_tokens: Iterable[int],
) -> torch.Tensor:
    """Cached greedy decode; stops on EOS-set membership or the token budget."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if token_ids.shape[0] != 1:
        raise ValueError("the reference decode loop is single-sequence")
    logits, cache = model_logits(token_ids, params, config)
    produced: list[int] = []
    past = token_ids.shape[-1]
    for _ in range(int(max_new_tokens)):
        token = int(greedy_next_token(logits[:, -1, :]).item())
        produced.append(token)
        if eos_stop(token, eos_tokens):
            break
        step = torch.tensor([[token]], device=token_ids.device)
        logits, cache = model_logits(step, params, config, past_length=past, cache=cache)
        past = cache_next_length(past)
    return torch.tensor([produced], dtype=torch.int64, device=token_ids.device)


# --------------------------------------------------------------------------
# Config of record: inspectable content, not only hashes.
# --------------------------------------------------------------------------

_CONFIG_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
)

_QUANT_FIELDS = ("bits", "group_size", "sym", "zero_point", "quant_method", "version")
_FROZEN_QUANTIZATION = {
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "zero_point": False,
    "quant_method": "awq",
    "version": "gemm",
}

# transformers 5.x nests the rotary base; released checkpoints keep it at the
# top level.  Both layouts are accepted, and disagreeing ones are refused.
_ROPE_NESTS = ("rope_parameters", "rope_scaling")


def rope_theta_of(config: Mapping[str, object]) -> float:
    found = []
    if "rope_theta" in config:
        found.append(float(config["rope_theta"]))
    for nest in _ROPE_NESTS:
        block = config.get(nest)
        if isinstance(block, Mapping) and "rope_theta" in block:
            found.append(float(block["rope_theta"]))
    if not found:
        raise ValueError("configuration declares no rotary base")
    if len(set(found)) > 1:
        raise ValueError("configuration declares disagreeing rotary bases")
    return found[0]


def head_dim_of(config: Mapping[str, object]) -> int:
    if config.get("head_dim"):
        return int(config["head_dim"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    if hidden % heads:
        raise ValueError("hidden size is not divisible by the head count")
    return hidden // heads


def text_config_of(config: Mapping[str, object]) -> Mapping[str, object]:
    """Mistral-Small 3.1 is multimodal; the rater reads its text tower."""
    inner = config.get("text_config")
    return inner if isinstance(inner, Mapping) else config


def config_of_record(
    config: Mapping[str, object],
    quantization: Mapping[str, object] | None = None,
    generation: Mapping[str, object] | None = None,
) -> dict:
    """Return the decode-relevant configuration CONTENT, for inspection.

    The reviewer asked for head counts, KV groups, head dimension, RMS epsilon,
    rotary base, vocabulary size and the quantization block to be readable in
    the evidence, not merely hashed.  This is that projection; it asserts
    nothing about the checkpoint beyond what the files say.
    """
    text = text_config_of(config)
    record: dict = {}
    for field in _CONFIG_FIELDS:
        if field == "head_dim":
            record[field] = head_dim_of(text)
        elif field == "rope_theta":
            record[field] = rope_theta_of(text)
        elif field in text:
            record[field] = text[field]
    heads = int(record["num_attention_heads"])
    kv_heads = int(record["num_key_value_heads"])
    record["key_value_groups"] = key_value_group_count(heads, kv_heads)
    eos = text.get("eos_token_id", config.get("eos_token_id"))
    if eos is None and generation is not None:
        eos = generation.get("eos_token_id")
    if eos is None:
        raise ValueError("configuration and generation configuration declare no EOS token")
    record["eos_token_id"] = list(eos) if isinstance(eos, (list, tuple)) else eos
    if quantization is not None:
        record["quantization"] = {
            field: quantization[field] for field in _QUANT_FIELDS if field in quantization
        }
    return record


def key_value_group_count(heads: int, key_value_heads: int) -> int:
    """Return the exact GQA ratio declared by a checkpoint configuration."""
    if heads <= 0 or key_value_heads <= 0 or heads % key_value_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    return int(heads) // int(key_value_heads)


def validate_quantization_config(
    config: Mapping[str, object], standalone: Mapping[str, object]
) -> Mapping[str, object]:
    """Require matching nested/standalone metadata for the frozen AWQ checkpoint."""
    if not isinstance(standalone, Mapping):
        raise ValueError("standalone quantization configuration is required")
    text = text_config_of(config)
    nested = text.get("quantization_config")
    if not isinstance(nested, Mapping):
        nested = config.get("quantization_config")
    if not isinstance(nested, Mapping):
        raise ValueError("checkpoint configuration has no nested quantization contract")
    for field, expected in _FROZEN_QUANTIZATION.items():
        if field not in standalone or field not in nested:
            raise ValueError(f"quantization configuration is missing {field}")
        if standalone[field] != nested[field]:
            raise ValueError(f"standalone and nested quantization disagree on {field}")
        if standalone[field] != expected:
            raise ValueError(f"frozen quantization contract rejects {field}={standalone[field]!r}")
    return standalone


def validate_config(config: Mapping[str, object]) -> Mapping[str, object]:
    """Refuse a configuration missing anything the decode path reads."""
    text = text_config_of(config)
    required = [f for f in _CONFIG_FIELDS if f not in ("head_dim", "rope_theta")]
    missing = [field for field in required if field not in text]
    if missing:
        raise ValueError("configuration is missing required keys: %s" % sorted(missing))
    head_dim_of(text)
    rope_theta_of(text)
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(model_path: Path, names: Sequence[str]) -> dict:
    """Digest the named checkpoint files, reporting unreadable ones honestly."""
    root = Path(model_path)
    identity: dict = {}
    for name in names:
        target = root / name
        try:
            identity[name] = sha256_file(target)
        except OSError as error:
            identity[name] = "unreadable: %s" % type(error).__name__
    return identity


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "SEALED_SEED_RANGES",
    "check_seed",
    "awq_unpack",
    "awq_group_index",
    "awq_dequantize",
    "awq_linear",
    "embedding_lookup",
    "rms_norm",
    "silu",
    "gated_silu_mlp",
    "linear",
    "rope_inverse_frequencies",
    "rope_angle_pairs",
    "rope_trig_tables",
    "rope_tables",
    "rotate_half",
    "apply_rope",
    "split_qkv",
    "repeat_kv_head",
    "repeat_kv",
    "causal_mask",
    "scaled_causal_attention",
    "cache_append",
    "cache_next_length",
    "next_positions",
    "compose_decoder_residual",
    "decoder_layer",
    "vocabulary_parallel_logit",
    "vocabulary_parallel_logits",
    "model_logits",
    "greedy_next_token",
    "eos_stop",
    "greedy_generate",
    "rope_theta_of",
    "head_dim_of",
    "text_config_of",
    "config_of_record",
    "key_value_group_count",
    "validate_quantization_config",
    "validate_config",
    "sha256_file",
    "checkpoint_identity",
    "load_json",
]
