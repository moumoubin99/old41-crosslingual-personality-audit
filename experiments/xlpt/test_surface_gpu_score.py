#!/usr/bin/env python3
"""CPU contracts for the complete timing-only linear scorer mathematics.

Every selected surface_* formula is exercised against an independent expected
value computed here (pure Python / hand arithmetic), not against the deployed
implementation's own output. The frozen launch input is loaded and checked so
the contract covers the tensor the GPU command actually scores.
"""

import json
import math
from pathlib import Path

import torch

from run_surface_gpu_score import (
    SEALED_SEED_RANGES,
    linear_surface_score,
    measured_throughput,
    seeded_linear_parameters,
    standardize_features,
    surface_features,
)


ROOT = Path(__file__).resolve().parents[2]
LAUNCH_INPUT = ROOT / "data/xlpt-support/rendered_items_and_scenarios/rubric_items_en.jsonl"
CPU = torch.device("cpu")


def expected_row_features(text):
    """Independent re-statement of the three counted surface features."""
    characters = 0
    for _ in text:
        characters += 1
    words = 0
    inside = False
    for character in text:
        if character.isspace():
            inside = False
        elif not inside:
            inside = True
            words += 1
    lines = 0
    pending = False
    for character in text:
        if character == "\n":
            lines += 1
            pending = False
        else:
            pending = True
    if pending:
        lines += 1
    return [float(characters), float(words), float(lines)]


def check_surface_features():
    rows = [
        {"text": "one two\nthree"},
        {"text": "four five six\nseven eight"},
    ]
    features = surface_features(rows, CPU)
    assert features.dtype == torch.float32 and features.device.type == "cpu"
    torch.testing.assert_close(features, torch.tensor([[13.0, 3.0, 2.0], [25.0, 5.0, 2.0]]))

    # Axis contract: one row per item, three columns in the declared order.
    assert features.shape == (2, 3)

    # Whitespace and terminator edge cases pin the exact counting definitions.
    tricky = [
        {"text": "a  b\t c\n"},
        {"text": "\n\n"},
        {"text": "solo"},
    ]
    tricky_features = surface_features(tricky, CPU)
    expected = torch.tensor([expected_row_features(row["text"]) for row in tricky])
    torch.testing.assert_close(tricky_features, expected)
    torch.testing.assert_close(tricky_features, torch.tensor([[8.0, 3.0, 1.0], [2.0, 0.0, 2.0], [4.0, 1.0, 1.0]]))


def check_standardize():
    features = torch.tensor([[13.0, 3.0, 2.0], [25.0, 5.0, 2.0]])
    standardized = standardize_features(features)
    # Column 0: mean 19, deviations -6/+6, root-mean-square 6 -> -1/+1.
    # Column 1: mean 4, deviations -1/+1, root-mean-square 1 -> -1/+1.
    # Column 2: zero variance, scale clamped to 1e-6, numerator 0 -> 0.
    torch.testing.assert_close(standardized, torch.tensor([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]))
    assert standardized.shape == features.shape

    # Population (not sample) normalization, applied down the item axis.
    asymmetric = torch.tensor([[0.0, 10.0], [4.0, 10.0], [8.0, 10.0]])
    result = standardize_features(asymmetric)
    torch.testing.assert_close(result[:, 0], torch.tensor([-math.sqrt(1.5), 0.0, math.sqrt(1.5)]))
    torch.testing.assert_close(result[:, 1], torch.zeros(3))
    torch.testing.assert_close(result.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(result[:, 0].pow(2).mean(), torch.tensor(1.0))

    # A single item has zero variance in every column and must not divide by zero.
    single = standardize_features(torch.tensor([[7.0, -3.0]]))
    torch.testing.assert_close(single, torch.zeros(1, 2))


def check_seeded_parameters():
    weights, bias = seeded_linear_parameters(17091701, 3, CPU)
    assert weights.shape == (3,) and bias.shape == (1,)
    assert weights.dtype == torch.float32 and bias.dtype == torch.float32

    # Independent re-derivation of the declared draw: uniform[0,1) shifted by -1/2,
    # weights consumed before the bias from one CPU generator seeded with the seed.
    generator = torch.Generator(device="cpu").manual_seed(17091701)
    expected_weights = torch.rand(3, generator=generator, dtype=torch.float32) - 0.5
    expected_bias = torch.rand(1, generator=generator, dtype=torch.float32) - 0.5
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(bias, expected_bias)
    assert bool(torch.all((weights >= -0.5) & (weights < 0.5)))
    assert bool(torch.all((bias >= -0.5) & (bias < 0.5)))

    repeat_weights, repeat_bias = seeded_linear_parameters(17091701, 3, CPU)
    torch.testing.assert_close(weights, repeat_weights)
    torch.testing.assert_close(bias, repeat_bias)

    other_weights, other_bias = seeded_linear_parameters(17091702, 3, CPU)
    assert not torch.equal(weights, other_weights)
    assert not torch.equal(bias, other_bias)

    # Sealed scientific seed blocks are refused at both ends of both ranges.
    for sealed in (20260811, 20260811 + 23, 20260911, 20260911 + 23):
        assert any(sealed in reserved for reserved in SEALED_SEED_RANGES)
        try:
            seeded_linear_parameters(sealed, 3, CPU)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sealed seed {sealed} was accepted")
    for allowed in (20260810, 20260811 + 24, 20260910, 20260911 + 24, 0):
        seeded_linear_parameters(allowed, 3, CPU)
    try:
        seeded_linear_parameters(-1, 3, CPU)
    except ValueError:
        pass
    else:
        raise AssertionError("negative seed was accepted")


def check_linear_surface_score():
    standardized = torch.tensor([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])

    # Fixed parameters give hand-computable targets: 1 + 6 * sigmoid(-1) and 1 + 6 * sigmoid(1).
    weights = torch.tensor([1.0, 0.0, 0.0])
    bias = torch.tensor([0.0])
    scores = linear_surface_score(standardized, weights, bias)
    low = 1.0 + 6.0 / (1.0 + math.exp(1.0))
    high = 1.0 + 6.0 / (1.0 + math.exp(-1.0))
    torch.testing.assert_close(scores, torch.tensor([low, high]))
    assert scores.shape == (2,)
    assert abs(low - 2.6136484) < 1e-6 and abs(high - 5.3863516) < 1e-6

    # The contraction is over the feature axis: a weight on a zeroed column changes nothing.
    torch.testing.assert_close(
        linear_surface_score(standardized, torch.tensor([1.0, 0.0, 9.0]), bias), scores
    )

    # Independent pure-Python oracle for the seeded parameters actually used at launch.
    drawn_weights, drawn_bias = seeded_linear_parameters(17091701, 3, CPU)
    drawn = linear_surface_score(standardized, drawn_weights, drawn_bias)
    oracle = []
    for row in standardized.tolist():
        z = sum(x * w for x, w in zip(row, drawn_weights.tolist())) + drawn_bias.item()
        oracle.append(1.0 + 6.0 / (1.0 + math.exp(-z)))
    torch.testing.assert_close(drawn, torch.tensor(oracle), atol=1e-6, rtol=0)

    # Mixed-sign weights and a known nonzero bias catch constant output, an
    # omitted bias, a wrong contraction axis, or the wrong 1+6*sigmoid scaling.
    varied = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 0.75, 3.0]])
    mixed_weights = torch.tensor([0.4, -0.2, 0.1])
    nonzero_bias = torch.tensor([0.3])
    varied_scores = linear_surface_score(varied, mixed_weights, nonzero_bias)
    varied_oracle = []
    for row in varied.tolist():
        z = sum(x * w for x, w in zip(row, mixed_weights.tolist())) + 0.3
        varied_oracle.append(1.0 + 6.0 / (1.0 + math.exp(-z)))
    torch.testing.assert_close(varied_scores, torch.tensor(varied_oracle), atol=1e-6, rtol=1e-6)
    perturbed = varied.clone()
    perturbed[0, 0] += 1.0
    perturbed_scores = linear_surface_score(perturbed, mixed_weights, nonzero_bias)
    assert perturbed_scores[0] != varied_scores[0]
    torch.testing.assert_close(perturbed_scores[1], varied_scores[1])

    # The map is strictly increasing in the logit and centres the rubric range at 4.
    spread = linear_surface_score(
        torch.tensor([[-10.0], [-1.0], [0.0], [1.0], [10.0]]), torch.tensor([1.0]), torch.tensor([0.0])
    )
    assert bool(torch.all((spread > 1.0) & (spread < 7.0)))
    torch.testing.assert_close(spread[2], torch.tensor(4.0))
    assert bool(torch.all(spread[1:] > spread[:-1]))
    # Saturating logits stay inside the closed rubric range; float32 rounds to the bound.
    saturated = linear_surface_score(
        torch.tensor([[-400.0], [400.0]]), torch.tensor([1.0]), torch.tensor([0.0])
    )
    torch.testing.assert_close(saturated, torch.tensor([1.0, 7.0]))


def check_measured_throughput():
    assert measured_throughput(24, 2.0) == 12.0
    assert measured_throughput(0, 5.0) == 0.0
    assert measured_throughput(7, 0.5) == 14.0
    for items, elapsed in ((-1, 1.0), (5, 0.0), (5, -2.0)):
        try:
            measured_throughput(items, elapsed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid throughput input accepted: {items}, {elapsed}")


def check_launch_input():
    rows = [
        json.loads(line)
        for line in LAUNCH_INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 12, len(rows)
    assert all(row["surface"] == "rater_rubric_item" for row in rows)
    features = surface_features(rows, CPU)
    assert features.shape == (12, 3)
    expected = torch.tensor([expected_row_features(row["text"]) for row in rows])
    torch.testing.assert_close(features, expected)
    assert bool(torch.all(features > 0))

    standardized = standardize_features(features)
    torch.testing.assert_close(standardized.mean(dim=0), torch.zeros(3), atol=1e-5, rtol=0)
    weights, bias = seeded_linear_parameters(17091701, standardized.shape[1], CPU)
    scores = linear_surface_score(standardized, weights, bias)
    assert scores.shape == (12,)
    assert bool(torch.all((scores > 1.0) & (scores < 7.0)))


def main():
    check_surface_features()
    check_standardize()
    check_seeded_parameters()
    check_linear_surface_score()
    check_measured_throughput()
    check_launch_input()
    print(
        "PASS: surface features, population scaling, seeded parameters, bounded linear "
        "scores, throughput and the frozen 12-item launch input"
    )


if __name__ == "__main__":
    main()
