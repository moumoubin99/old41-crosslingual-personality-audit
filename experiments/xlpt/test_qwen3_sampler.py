#!/usr/bin/env python3
"""Executable contract for deployment unit U5 - the seeded nucleus sampler.

The deployment review rejected the earlier package partly because no trace showed
a sampler actually executing: "their traces contain no actual model forward,
Sampler.forward, Sampler.sample or run_generation.run execution."  This file
executes the captured sampler that the apparatus run dispatches through, and
checks it against values computed here in plain Python with `math`, never
against the sampler's own output.

Covered: the temperature rescaling, the nucleus truncation set, renormalization
over that set, the degenerate `temperature == 0` branch, seeded reproducibility
of the whole decode loop at the registered decoding settings (temperature 0.7,
top_p 0.95), every invalid-argument branch, and the sealed-seed guard that keeps
this apparatus off the evaluation seed blocks.
"""

from __future__ import annotations

import math

import torch

from qwen3_reference import (
    FIXTURE_CONFIG,
    SEALED_SEED_RANGES,
    check_seed,
    fixture_params,
    generate,
    greedy_next_token,
    nucleus_next_token,
    validate_config,
)

SEED = 17091701
TEMPERATURE = 0.7
TOP_P = 0.95
DRAWS = 4000


# --------------------------------------------------------------------------
# Independent pure-Python oracle for the truncated sampling distribution
# --------------------------------------------------------------------------


def oracle_distribution(logits, temperature, top_p):
    """The distribution the sampler must draw from, computed without torch.

    Returns ``(probabilities, kept_token_ids)`` where ``probabilities`` is a list
    over the whole vocabulary with zeros outside the nucleus.
    """
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    probs = [w / total for w in weights]

    order = sorted(range(len(probs)), key=lambda index: (-probs[index], index))
    kept = []
    prefix = 0.0
    for rank, token in enumerate(order):
        # The kept set is the shortest descending prefix whose cumulative mass
        # reaches top_p; the highest-probability token is always kept.
        if rank == 0 or prefix < top_p:
            kept.append(token)
        prefix += probs[token]

    mass = sum(probs[token] for token in kept)
    truncated = [0.0] * len(probs)
    for token in kept:
        truncated[token] = probs[token] / mass
    return truncated, sorted(kept)


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


# Deliberately distinct values so the descending order carries no ties.
ROW = [2.0, 0.5, -1.0, 3.5, 1.25, -2.5, 0.0, 2.75]


def as_logits(row):
    """A (tokens, vocab) tensor whose final row is the one under test."""
    filler = [0.0] * len(row)
    return torch.tensor([filler, row], dtype=torch.float32)


def check_truncation_set_is_respected():
    """Every drawn token lies in the independently computed nucleus."""
    logits = as_logits(ROW)
    for top_p in (0.3, 0.6, 0.95, 1.0):
        _, kept = oracle_distribution(ROW, TEMPERATURE, top_p)
        if not kept:
            raise AssertionError(f"the oracle kept no token at top_p {top_p}")
        generator = torch.Generator().manual_seed(SEED)
        seen = set()
        for _ in range(DRAWS):
            seen.add(nucleus_next_token(logits, TEMPERATURE, top_p, generator))
        outside = sorted(seen - set(kept))
        if outside:
            raise AssertionError(
                f"top_p {top_p}: sampler drew tokens {outside} outside the nucleus {kept}")
        # Truncation must be real: a tight top_p keeps strictly fewer tokens
        # than the vocabulary, and the sampler must reach every kept token.
        missing = sorted(set(kept) - seen)
        if top_p <= 0.95 and missing:
            raise AssertionError(
                f"top_p {top_p}: kept tokens {missing} were never drawn in {DRAWS} draws")

    _, tight = oracle_distribution(ROW, TEMPERATURE, 0.3)
    _, wide = oracle_distribution(ROW, TEMPERATURE, 1.0)
    if not len(tight) < len(wide) == len(ROW):
        raise AssertionError(
            f"top_p is not truncating: {len(tight)} kept at 0.3, {len(wide)} at 1.0")


def check_empirical_distribution():
    """Draw frequencies agree with the independently renormalized probabilities."""
    logits = as_logits(ROW)
    expected, kept = oracle_distribution(ROW, TEMPERATURE, TOP_P)
    generator = torch.Generator().manual_seed(SEED)
    counts = [0] * len(ROW)
    for _ in range(DRAWS):
        counts[nucleus_next_token(logits, TEMPERATURE, TOP_P, generator)] += 1

    for token in range(len(ROW)):
        observed = counts[token] / DRAWS
        want = expected[token]
        # Four standard errors of a binomial proportion, floored so that a
        # zero-probability token is held to exactly zero.
        tolerance = 4.0 * math.sqrt(max(want * (1.0 - want), 1e-12) / DRAWS)
        if abs(observed - want) > tolerance + 1e-12:
            raise AssertionError(
                f"token {token}: drawn {observed:.4f} of the time, oracle says {want:.4f} "
                f"(tolerance {tolerance:.4f}, nucleus {kept})")


def check_temperature_shapes_the_distribution():
    """Temperature rescales the logits before the softmax, and is load-bearing."""
    logits = as_logits(ROW)
    peak = max(range(len(ROW)), key=lambda index: ROW[index])

    frequencies = []
    for temperature in (0.2, 1.0, 4.0):
        generator = torch.Generator().manual_seed(SEED)
        hits = sum(nucleus_next_token(logits, temperature, 1.0, generator) == peak
                   for _ in range(DRAWS))
        frequencies.append(hits / DRAWS)
    if not frequencies[0] > frequencies[1] > frequencies[2]:
        raise AssertionError(
            f"raising the temperature did not flatten the distribution: {frequencies}")

    # The independently computed distribution moves the same way.
    cold, _ = oracle_distribution(ROW, 0.2, 1.0)
    hot, _ = oracle_distribution(ROW, 4.0, 1.0)
    if not cold[peak] > hot[peak]:
        raise AssertionError("the oracle disagrees that temperature flattens the row")
    if abs(frequencies[0] - cold[peak]) > 0.04 or abs(frequencies[2] - hot[peak]) > 0.04:
        raise AssertionError(
            f"empirical peak frequencies {frequencies[0]:.3f}/{frequencies[2]:.3f} do not match "
            f"the oracle {cold[peak]:.3f}/{hot[peak]:.3f}")


def check_zero_temperature_is_greedy():
    logits = as_logits(ROW)
    expected = max(range(len(ROW)), key=lambda index: ROW[index])
    if greedy_next_token(logits) != expected:
        raise AssertionError("greedy_next_token disagrees with the hand-computed argmax")
    # No generator is required and none is consulted on the degenerate branch.
    for top_p in (0.1, 1.0):
        if nucleus_next_token(logits, 0.0, top_p, None) != expected:
            raise AssertionError(f"temperature 0 at top_p {top_p} did not degenerate to greedy")


def check_seeded_reproducibility():
    logits = as_logits(ROW)

    def draw(seed, count=32):
        generator = torch.Generator().manual_seed(seed)
        return [nucleus_next_token(logits, TEMPERATURE, TOP_P, generator) for _ in range(count)]

    first = draw(SEED)
    if first != draw(SEED):
        raise AssertionError("the sampler is not reproducible for a fixed seed")
    if first == draw(SEED + 1):
        raise AssertionError("two different seeds produced an identical draw sequence")
    if len(set(first)) < 2:
        raise AssertionError(f"the draw sequence is degenerate: {first}")


def check_guards():
    logits = as_logits(ROW)
    generator = torch.Generator().manual_seed(SEED)
    for temperature, top_p, label in (
        (-0.1, 1.0, "negative temperature"),
        (1.0, 0.0, "top_p 0"),
        (1.0, -0.5, "negative top_p"),
        (1.0, 1.5, "top_p above 1"),
    ):
        try:
            nucleus_next_token(logits, temperature, top_p, generator)
        except ValueError:
            continue
        raise AssertionError(f"nucleus_next_token accepted {label}")

    for bad in (logits[-1], logits.unsqueeze(0)):
        try:
            nucleus_next_token(bad, 1.0, 1.0, generator)
        except ValueError:
            pass
        else:
            raise AssertionError(f"nucleus_next_token accepted logits of rank {bad.ndim}")
        try:
            greedy_next_token(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"greedy_next_token accepted logits of rank {bad.ndim}")


def check_decode_loop_at_registered_settings():
    """The whole decode loop is reproducible at temperature 0.7 / top_p 0.95."""
    config = dict(FIXTURE_CONFIG)
    validate_config(config)
    params = fixture_params(config, SEED, dtype=torch.float64)
    prompt = [3, 7, 0]

    def run(seed):
        return generate(params, config, prompt, max_new_tokens=6,
                        temperature=TEMPERATURE, top_p=TOP_P,
                        generator=torch.Generator().manual_seed(seed))[0]

    produced = run(SEED)
    if len(produced) != 6:
        raise AssertionError(f"decode produced {len(produced)} tokens")
    if produced != run(SEED):
        raise AssertionError(f"seeded decode is not reproducible: {produced} vs {run(SEED)}")
    if produced == run(SEED + 1):
        raise AssertionError("two different seeds produced an identical continuation")
    vocab = int(config["vocab_size"])
    if any(not 0 <= token < vocab for token in produced):
        raise AssertionError(f"decode emitted a token outside the vocabulary: {produced}")

    # Stochastic decoding without a seed is refused rather than silently seeded
    # from global state, and the greedy path stays available and deterministic.
    try:
        generate(params, config, prompt, max_new_tokens=2, temperature=TEMPERATURE, top_p=TOP_P)
    except ValueError:
        pass
    else:
        raise AssertionError("stochastic decoding ran without an explicit generator")
    if generate(params, config, prompt, max_new_tokens=4)[0] != \
            generate(params, config, prompt, max_new_tokens=4)[0]:
        raise AssertionError("greedy decoding is not deterministic")

    # A stop token ends the loop at the first occurrence.
    stopped, _ = generate(params, config, prompt, max_new_tokens=6,
                          temperature=TEMPERATURE, top_p=TOP_P,
                          generator=torch.Generator().manual_seed(SEED),
                          stop_ids=(produced[0],))
    if stopped != [produced[0]]:
        raise AssertionError(f"stop token was not honoured under sampling: {stopped}")
    return produced


def check_sampler_seed_is_not_sealed():
    """The apparatus seed is outside every sealed evaluation block."""
    check_seed(SEED)
    for reserved in SEALED_SEED_RANGES:
        if SEED in reserved:
            raise AssertionError(f"the sampler seed {SEED} lies in a sealed block {reserved}")
        for sealed in (reserved.start, reserved.stop - 1):
            try:
                check_seed(sealed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"check_seed accepted sealed seed {sealed}")


def main():
    check_truncation_set_is_respected()
    check_empirical_distribution()
    check_temperature_shapes_the_distribution()
    check_zero_temperature_is_greedy()
    check_seeded_reproducibility()
    check_guards()
    produced = check_decode_loop_at_registered_settings()
    check_sampler_seed_is_not_sealed()
    print("PASS: nucleus truncation against an independent pure-Python distribution, empirical "
          f"draw frequencies within four standard errors over {DRAWS} draws, temperature "
          "monotonicity, the greedy degeneracy, seeded decode reproducibility at "
          f"temperature {TEMPERATURE}/top_p {TOP_P} (continuation {produced}), the argument "
          "guards and the sealed-seed refusal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
