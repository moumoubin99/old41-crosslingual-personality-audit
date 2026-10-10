"""Numerical implementations bound to ``refine-logs/MATH_SPEC.json``.

The functions are deliberately small and side-effect free.  Generation and scoring
write unit-level records; this module is the only place that turns those records into
the registered eq1--eq7 derived quantities.
"""

from __future__ import annotations

import numpy as np


def translation_ensemble_persona_effect(persona_score, neutral_score):
    """Eq1: paired A-B difference, averaged over translation and repeat axes."""
    persona = np.asarray(persona_score, dtype=float)
    neutral = np.asarray(neutral_score, dtype=float)
    if persona.shape != neutral.shape or persona.ndim != 6:
        raise ValueError("eq1 inputs must have one identical 6-D shape")
    return np.mean(persona - neutral, axis=(4, 5))


def directed_retention_matrix(persona_effect, same_language_effect, eligibility):
    """Eq2: eligible cross-language effect divided by its diagonal reference.

    Eligibility is supplied by the registered two-part denominator test.  False
    cells are returned as NaN (``RATIO NOT DEFINED``); they are never imputed.
    """
    effect = np.asarray(persona_effect, dtype=float)
    diagonal = np.asarray(same_language_effect, dtype=float)
    eligible = np.asarray(eligibility)
    if effect.ndim != 4 or diagonal.shape != (effect.shape[0], effect.shape[1], effect.shape[3]):
        raise ValueError("eq2 shapes must be [m,s,t,k] and [m,s,k]")
    if eligible.shape != diagonal.shape or eligible.dtype != np.bool_:
        raise ValueError("eq2 eligibility must be a boolean [m,s,k] registered-gate mask")
    if np.any(eligible & (~np.isfinite(diagonal) | (diagonal == 0.0))):
        raise ValueError("eq2 eligibility cannot authorize a zero or non-finite denominator")
    valid_diagonal = eligible
    denominator = np.expand_dims(diagonal, axis=2)
    valid = np.broadcast_to(np.expand_dims(valid_diagonal, axis=2), effect.shape)
    result = np.full(effect.shape, np.nan, dtype=float)
    np.divide(effect, denominator, out=result, where=valid)
    return result


def normalized_shortfall(persona_effect, same_language_effect, eligibility):
    """Eq3: one minus the eligible-cell retention ratio."""
    return 1.0 - directed_retention_matrix(persona_effect, same_language_effect, eligibility)


def transfer_asymmetry_index(retention_shortfall):
    """Eq4: mean ordered-pair absolute asymmetry, excluding the zero diagonal."""
    shortfall = np.asarray(retention_shortfall, dtype=float)
    if shortfall.ndim != 4 or shortfall.shape[1] != shortfall.shape[2]:
        raise ValueError("eq4 input must have shape [model, language, language, trait]")
    language = shortfall.shape[1]
    if language < 2:
        raise ValueError("eq4 needs at least two languages")
    return np.sum(np.abs(shortfall - np.swapaxes(shortfall, 1, 2)), axis=(1, 2)) / (
        language * (language - 1)
    )


def target_resource_shortfall_gradient(target_scarcity_within_source, shortfall_within_source):
    """Eq5: within-source least-squares slope through the centered origin."""
    scarcity = np.asarray(target_scarcity_within_source, dtype=float)
    shortfall = np.asarray(shortfall_within_source, dtype=float)
    if scarcity.ndim != 1 or shortfall.ndim != 3 or shortfall.shape[1] != scarcity.shape[0]:
        raise ValueError("eq5 shapes must be [pair] and [model,pair,trait]")
    denominator = np.sum(scarcity * scarcity)
    if denominator <= 0:
        raise ValueError("eq5 centered scarcity must have positive sum of squares")
    return np.sum(shortfall * scarcity.reshape(1, -1, 1), axis=1) / denominator


def paired_difference_in_differences(same_language_effect, persona_effect):
    """Eq6: native-scale matched minus mismatched pole-aligned cell effect."""
    effect = np.asarray(persona_effect, dtype=float)
    diagonal = np.asarray(same_language_effect, dtype=float)
    if effect.ndim != 4 or diagonal.shape != (effect.shape[0], effect.shape[1], effect.shape[3]):
        raise ValueError("eq6 shapes must be [m,s,t,k] and [m,s,k]")
    return np.expand_dims(diagonal, axis=2) - effect


def three_way_factorial_decomposition(unnormalized_shortfall_factorial, contrast_matrix):
    """Eq7: normalized projections on an explicitly orthogonal factorial basis."""
    values = np.asarray(unnormalized_shortfall_factorial, dtype=float)
    contrasts = np.asarray(contrast_matrix, dtype=float)
    if values.ndim != 5 or contrasts.ndim != 4:
        raise ValueError("eq7 shapes must be [m,p,d,i,k] and [term,p,d,i]")
    if tuple(values.shape[1:4]) != tuple(contrasts.shape[1:4]):
        raise ValueError("eq7 factor axes must agree")
    flat = contrasts.reshape(contrasts.shape[0], -1)
    gram = flat @ flat.T
    denominator = np.diag(gram)
    if np.any(denominator <= 0):
        raise ValueError("eq7 contrast rows must have positive squared norm")
    off_diagonal = gram - np.diag(denominator)
    tolerance = 1e-10 * max(1.0, float(np.max(denominator)))
    if np.any(np.abs(off_diagonal) > tolerance):
        raise ValueError("eq7 normalized projections require an orthogonal contrast basis")
    numerator = np.einsum("tpdi,mpdik->mtk", contrasts, values)
    return numerator / denominator.reshape(1, -1, 1)
