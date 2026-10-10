"""Small executable reference contracts for the timing run's standard math."""

from __future__ import annotations

import numpy as np


def autoregressive_joint_probability(token_conditional_probability):
    probability = np.asarray(token_conditional_probability, dtype=float)
    if probability.ndim < 1 or np.any((probability < 0) | (probability > 1)):
        raise ValueError("conditional probabilities must lie in [0,1]")
    return np.prod(probability, axis=-1)


def temperature_softmax(logits, temperature):
    values = np.asarray(logits, dtype=float)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scaled = values / temperature
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    weights = np.exp(scaled)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def nucleus_renormalize(probability, nucleus_mask):
    probability = np.asarray(probability, dtype=float)
    mask = np.asarray(nucleus_mask)
    if probability.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("nucleus mask must be boolean and match probability shape")
    retained = np.where(mask, probability, 0.0)
    total = np.sum(retained, axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("every nucleus must retain positive mass")
    return retained / total


def exponential_race_sample(nucleus_probability, exponential_noise):
    probability = np.asarray(nucleus_probability, dtype=float)
    noise = np.asarray(exponential_noise, dtype=float)
    if noise.shape != probability.shape or np.any(noise <= 0):
        raise ValueError("positive exponential noise must match the probability tensor")
    return np.argmax(probability / noise, axis=-1)


def timing_throughput(generated_call_count, elapsed_seconds):
    if isinstance(generated_call_count, bool) or int(generated_call_count) != generated_call_count:
        raise ValueError("generated call count must be an integer")
    if generated_call_count < 0 or not np.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("timing inputs must be nonnegative calls and positive finite elapsed time")
    return int(generated_call_count) / float(elapsed_seconds)
