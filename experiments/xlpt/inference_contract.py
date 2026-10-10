#!/usr/bin/env python3
"""Small, testable inference/scoring primitives used by the English XLPT jobs."""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path


def temperature_probabilities(logits, temperature: float):
    """Stable temperature softmax over the final vocabulary axis."""
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    import torch

    scaled = logits.float() / float(temperature)
    scaled = scaled - scaled.amax(dim=-1, keepdim=True)
    return torch.softmax(scaled, dim=-1)


def nucleus_probabilities(probabilities, top_p: float):
    """Keep the smallest descending-probability prefix reaching ``top_p``."""
    if not math.isfinite(float(top_p)) or not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    import torch

    ordered, indices = torch.sort(probabilities.float(), dim=-1, descending=True)
    cumulative_before = torch.cumsum(ordered, dim=-1) - ordered
    keep = cumulative_before < float(top_p)
    kept = torch.where(keep, ordered, torch.zeros_like(ordered))
    normalizer = kept.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise ValueError("nucleus has zero probability mass")
    kept = kept / normalizer
    result = torch.zeros_like(kept)
    return result.scatter(-1, indices, kept)


def gumbel_categorical(probabilities, uniform_draw):
    """Pure Gumbel-max transform for controlled-draw contract tests."""
    import torch

    if probabilities.ndim != 2 or torch.any(probabilities < 0):
        raise ValueError("probabilities must be a nonnegative [batch,vocabulary] tensor")
    mass = probabilities.sum(dim=-1)
    if not torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5):
        raise ValueError("probabilities must sum to one")
    if uniform_draw.shape != probabilities.shape or torch.any((uniform_draw <= 0) | (uniform_draw >= 1)):
        raise ValueError("uniform draws must match probabilities and lie strictly inside (0,1)")
    uniform = uniform_draw.to(device=probabilities.device, dtype=torch.float32)
    score = torch.where(
        probabilities > 0,
        torch.log(probabilities) - torch.log(-torch.log(uniform)),
        torch.full_like(probabilities, -torch.inf),
    )
    return score.argmax(dim=-1)


def seeded_categorical(probabilities, generator):
    """Draw a categorical token with a seeded Gumbel-max construction."""
    import torch

    uniform = torch.rand(
        probabilities.shape,
        dtype=torch.float32,
        device=probabilities.device,
        generator=generator,
    ).clamp_(1e-7, 1.0 - 1e-7)
    return gumbel_categorical(probabilities, uniform)


def greedy_token(logits):
    """Deterministic greedy token selection."""
    return logits.argmax(dim=-1)


def trait_score(item_scores):
    """Equal-weight mean of exactly six integer rubric scores in [1, 7]."""
    import torch

    values = torch.as_tensor(item_scores)
    if values.shape[-1] != 6:
        raise ValueError("trait score requires exactly six rubric items")
    if values.dtype == torch.bool or values.is_floating_point():
        raise ValueError("rubric item scores must be integers")
    if torch.any((values < 1) | (values > 7)):
        raise ValueError("rubric item scores must lie in [1, 7]")
    return values.to(torch.float64).mean(dim=-1)


def measured_rate(completed_calls: int, elapsed_seconds: float) -> float:
    if isinstance(completed_calls, bool) or completed_calls < 0:
        raise ValueError("completed_calls must be a nonnegative integer")
    if not math.isfinite(float(elapsed_seconds)) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    return float(completed_calls) / float(elapsed_seconds)


def substream_seed(root_seed: int, scenario_index: int, turn_index: int) -> int:
    """Injective registered mapping from a root seed to per-turn substreams."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0
           for x in (root_seed, scenario_index, turn_index)):
        raise ValueError("seed coordinates must be nonnegative integers")
    if turn_index >= 3 or scenario_index >= 1000:
        raise ValueError("substream coordinate exceeds the frozen three-turn/1000-scenario envelope")
    return root_seed * 10_000 + scenario_index * 3 + turn_index


def advance_attention_mask(attention_mask):
    """Append the single visible position used by the next autoregressive step."""
    import torch

    return torch.cat([
        attention_mask,
        torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
    ], dim=1)


def should_stop(sampled_token: int, eos_token_ids: set[int]) -> bool:
    return int(sampled_token) in eos_token_ids


def qwen_eager_attention_reference(query, key, value, attention_mask, scaling: float):
    """Independent no-dropout reference for the pinned Qwen3 eager attention."""
    import torch

    scores = torch.matmul(query, key.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


def validate_qwen_runtime(manifest_path: Path, model_path: Path) -> str:
    """Cryptographically bind packages and every loaded checkpoint/tokenizer artifact."""
    import importlib.metadata

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for package, expected in manifest["package_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"runtime package version mismatch: {package}")
    for name, expected in manifest["model_files"].items():
        path = Path(model_path) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Qwen checkpoint/tokenizer artifact changed: {name}")
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def autoregressive_generate_ids(
    model,
    input_ids,
    attention_mask,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    eos_token_ids: set[int],
):
    """Run a captured Python autoregressive loop around a causal model forward."""
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("reviewed autoregressive generator requires batch size one")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise ValueError("reviewed autoregressive attention mask requires batch size one")
    device = input_ids.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    generated = []
    past = None
    current_ids = input_ids
    current_mask = attention_mask
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = output.logits[:, -1, :]
            probs = temperature_probabilities(logits, temperature)
            probs = nucleus_probabilities(probs, top_p)
            token = seeded_categorical(probs, generator)
            token_id = int(token.item())
            generated.append(token_id)
            if should_stop(token_id, eos_token_ids):
                break
            past = output.past_key_values
            current_ids = token[:, None]
            current_mask = advance_attention_mask(current_mask)
    return generated
