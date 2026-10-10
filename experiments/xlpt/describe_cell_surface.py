#!/usr/bin/env python3
"""Response-language rate and turn length of every pole generation cell (descriptive).

The plan (FOLLOWUP_PLAN_2026-09-28, BC) requires the response-language rate on BC outputs to be
reported beside the intention-to-treat rater contrast. This applies the Han-ratio turn rule of
``describe_zh_b4_generation.turn_indicators`` to the agent turns of all eight pole phases and
also reports mean agent-turn length, since a rater score can track verbosity.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from describe_zh_b4_generation import turn_indicators  # noqa: E402

PHASES = {
    "P=en,D=en": ("b4-extraversion-high", "b4-extraversion-low"),
    "P=en,D=zh": ("zh-b4-extraversion-high", "zh-b4-extraversion-low"),
    "P=zh,D=en": ("bc-pzhn-extraversion-high-den", "bc-pzhn-extraversion-low-den"),
    "P=zh,D=zh": ("bc-pzhn-extraversion-high-dzh", "bc-pzhn-extraversion-low-dzh"),
}


def describe(phase_root: Path) -> dict:
    out = {}
    for checkpoint_dir in sorted(p for p in phase_root.iterdir() if p.is_dir()):
        turns = []
        rows = 0
        for path in sorted(checkpoint_dir.glob("seed-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rows += 1
                turns.extend(json.loads(line)["agent_turns"])
        per = [turn_indicators(text) for text in turns]
        out[checkpoint_dir.name] = {
            "rows": rows,
            "agent_turns": len(turns),
            "chinese_turn_rate": round(sum(p["chinese"] for p in per) / len(per), 6),
            "no_han_turn_rate": round(sum(p["no_han"] for p in per) / len(per), 6),
            "mean_chars_per_turn": round(statistics.fmean(len(text) for text in turns), 2),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default="process/results/xlpt/XLPT_CELL_SURFACE.json")
    args = ap.parse_args(argv)
    project = Path(args.project).resolve()
    root = project / "process/results/xlpt"
    cells = {
        label: {"high": describe(root / high), "low": describe(root / low)}
        for label, (high, low) in PHASES.items()
    }
    out = {"schema_version": 1, "verdict": "DESCRIPTIVE_ONLY", "eligible_for_gate": False,
           "turn_rule": "Han >= 0.5 of Han + Latin letters", "cells": cells}
    (project / args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    for label, poles in cells.items():
        for pole, by_ckpt in poles.items():
            for ckpt, d in by_ckpt.items():
                print(f"{label} {pole:4s} {ckpt:24s} zh={d['chinese_turn_rate']:.3f} "
                      f"nohan={d['no_han_turn_rate']:.3f} chars={d['mean_chars_per_turn']:.0f} "
                      f"rows={d['rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
