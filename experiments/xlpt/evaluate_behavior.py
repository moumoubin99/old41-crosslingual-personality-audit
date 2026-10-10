#!/usr/bin/env python3
"""Validate scored XLPT records and compute registered descriptive math.

This evaluator never silently drops an ITT unit.  It refuses incomplete cells and
keeps confirmatory clipping separate from the unclipped descriptive eq1--eq7
quantities.  The empirical-Bernstein declaration layer is intentionally a later
freeze-dependent step; timing records are rejected here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from math_impl import (
    directed_retention_matrix,
    normalized_shortfall,
    paired_difference_in_differences,
    transfer_asymmetry_index,
)


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(args):
    rows = load(args.input_jsonl)
    if not rows or any(not row.get("eligible_for_analysis") for row in rows):
        raise ValueError("evaluation accepts only qualification/evaluation records, never timing output")
    required = {"checkpoint_id", "persona_language", "dialogue_language", "trait", "arm",
                "pole", "engine", "unit_id", "trait_score"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"record {index} missing {sorted(missing)}")

    values = defaultdict(list)
    for row in rows:
        key = (row["checkpoint_id"], row["persona_language"], row["dialogue_language"],
               row["trait"], row["arm"], row["pole"], row["engine"])
        values[key].append(float(row["trait_score"]))
    checkpoints = sorted({key[0] for key in values})
    languages = sorted({key[1] for key in values} | {key[2] for key in values})
    traits = sorted({key[3] for key in values})
    engines = sorted({key[6] for key in values})
    if len(engines) != 3:
        raise ValueError("registered cell effects require exactly three fixed engine strata")

    effect = np.empty((len(checkpoints), len(languages), len(languages), len(traits)))
    for mi, model in enumerate(checkpoints):
        for si, source in enumerate(languages):
            for ti, target in enumerate(languages):
                for ki, trait in enumerate(traits):
                    pole_effects = []
                    for pole, sign in (("high", 1.0), ("low", -1.0)):
                        engine_effects = []
                        for engine in engines:
                            persona = values[(model, source, target, trait, "persona", pole, engine)]
                            neutral = values[(model, source, target, trait, "neutral", pole, engine)]
                            if not persona or len(persona) != len(neutral):
                                raise ValueError(f"incomplete paired cell {(model, source, target, trait, pole, engine)}")
                            engine_effects.append(float(np.mean(np.asarray(persona) - np.asarray(neutral))))
                        pole_effects.append(sign * float(np.mean(engine_effects)))
                    effect[mi, si, ti, ki] = 0.5 * sum(pole_effects)
    diagonal = np.diagonal(effect, axis1=1, axis2=2).transpose(0, 2, 1)
    mismatch = paired_difference_in_differences(diagonal, effect)
    with np.errstate(divide="ignore", invalid="ignore"):
        retention = directed_retention_matrix(effect, diagonal)
        shortfall = normalized_shortfall(effect, diagonal)
    result = {
        "schema_version": 1,
        "checkpoint_order": checkpoints,
        "language_order": languages,
        "trait_order": traits,
        "cell_effect_eq1": effect.tolist(),
        "retention_eq2_descriptive": retention.tolist(),
        "shortfall_eq3_descriptive": shortfall.tolist(),
        "asymmetry_eq4_descriptive": transfer_asymmetry_index(shortfall).tolist(),
        "mismatch_eq6_unclipped_descriptive": mismatch.tolist(),
        "confirmatory_note": "No declaration is made here; the frozen pre-B1 family and unit-level R26-F4 bounds are required.",
    }
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output_json, "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    main(ap.parse_args())

