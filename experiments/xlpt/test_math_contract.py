#!/usr/bin/env python3
"""Independent numeric/shape checks for deployed eq1--eq7 implementations."""

from __future__ import annotations

import numpy as np

from math_impl import (
    directed_retention_matrix,
    normalized_shortfall,
    paired_difference_in_differences,
    target_resource_shortfall_gradient,
    three_way_factorial_decomposition,
    transfer_asymmetry_index,
    translation_ensemble_persona_effect,
)


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def main():
    # Eq1 uses a hand-computed paired mean, including nonconstant neutral values.
    persona = np.arange(2 * 2 * 3 * 2 * 2 * 3, dtype=float).reshape(2, 2, 3, 2, 2, 3) / 10
    neutral = np.flip(persona, axis=-1) * 0.25 - 0.7
    eq1_expected = np.empty((2, 2, 3, 2))
    for m in range(2):
        for s in range(2):
            for t in range(3):
                for k in range(2):
                    vals = [persona[m, s, t, k, v, r] - neutral[m, s, t, k, v, r]
                            for v in range(2) for r in range(3)]
                    eq1_expected[m, s, t, k] = sum(vals) / len(vals)
    effect = translation_ensemble_persona_effect(persona, neutral)
    assert_close(effect, eq1_expected)

    # Eq2/3/6 use explicit broadcasting expectations and preserve all axes.
    diagonal = np.array([[[2.0, 4.0], [5.0, 10.0]], [[3.0, 6.0], [8.0, 16.0]]])
    eligibility = np.ones(diagonal.shape, dtype=bool)
    ratio = directed_retention_matrix(effect, diagonal, eligibility)
    assert ratio.shape == effect.shape
    assert_close(ratio[:, :, 1, :], effect[:, :, 1, :] / diagonal)
    assert_close(normalized_shortfall(effect, diagonal, eligibility), 1.0 - ratio)
    assert_close(paired_difference_in_differences(diagonal, effect), diagonal[:, :, None, :] - effect)
    guarded_diagonal = diagonal.copy()
    guarded_diagonal[0, 0, 0] = 0.0
    guarded_eligibility = eligibility.copy()
    guarded_eligibility[0, 0, 0] = False
    guarded = directed_retention_matrix(effect, guarded_diagonal, guarded_eligibility)
    assert np.isnan(guarded[0, 0, :, 0]).all()
    try:
        directed_retention_matrix(effect, guarded_diagonal, eligibility)
    except ValueError:
        pass
    else:
        raise AssertionError("eq2 accepted an eligible zero denominator")

    # Eq4 double-counts ordered directions by design and divides by L(L-1).
    matrix = np.zeros((1, 3, 3, 1), dtype=float)
    matrix[0, 0, 1, 0], matrix[0, 1, 0, 0] = 0.2, 0.5
    matrix[0, 0, 2, 0], matrix[0, 2, 0, 0] = -0.1, 0.3
    matrix[0, 1, 2, 0], matrix[0, 2, 1, 0] = 0.7, 0.6
    expected_asymmetry = 2 * (0.3 + 0.4 + 0.1) / 6
    assert_close(transfer_asymmetry_index(matrix), np.array([[expected_asymmetry]]))

    # Eq5 is an independently computed scalar OLS slope for every model/trait.
    scarcity = np.array([-2.0, -1.0, 1.0, 2.0])
    slopes = np.array([[0.5, -0.25], [1.5, 0.75]])
    shortfall = slopes[:, None, :] * scarcity[None, :, None]
    assert_close(target_resource_shortfall_gradient(scarcity, shortfall), slopes)

    # Eq7 recovers known coefficients from orthogonal, sum-to-zero contrasts.
    levels = np.array([-1.0, 0.0, 1.0])
    p, d, i = np.meshgrid(levels, levels, levels, indexing="ij")
    contrasts = np.stack([np.ones_like(p), p, d, i, p * d, p * i, d * i, p * d * i])
    coefficients = np.array([0.2, -0.3, 0.4, 0.1, 0.5, -0.2, 0.7, -0.6])
    cube = np.einsum("t,tpdi->pdi", coefficients, contrasts)[None, ..., None]
    recovered = three_way_factorial_decomposition(cube, contrasts)
    assert_close(recovered[0, :, 0], coefficients)
    nonorthogonal = contrasts.copy()
    nonorthogonal[1] = contrasts[1] + 0.25 * contrasts[2]
    nonorthogonal_cube = np.einsum("t,tpdi->pdi", coefficients, nonorthogonal)[None, ..., None]
    try:
        three_way_factorial_decomposition(nonorthogonal_cube, nonorthogonal)
    except ValueError:
        pass
    else:
        raise AssertionError("eq7 accepted a nonorthogonal projection basis")
    print("PASS: deployed eq1-eq7 numeric, shape, axis and invariant checks")


if __name__ == "__main__":
    main()
