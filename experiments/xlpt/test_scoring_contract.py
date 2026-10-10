#!/usr/bin/env python3
"""CPU contracts for English rubric request construction and score aggregation."""

from pathlib import Path

from inference_contract import measured_rate, trait_score
from run_rater_scoring import build_item_request, parse_item_score, rater_decoding_contract
from score_rater import rubric_items


for raw, expected in (("1", 1), (" 4\n", 4), ("7", 7)):
    if parse_item_score(raw) != expected:
        raise AssertionError("strict rater integer parser changed")
for invalid in ("", "0", "8", "score=3 because", "3 4"):
    try:
        parse_item_score(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"rater parser accepted invalid output: {invalid!r}")
if rater_decoding_contract() != {
    "temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_tokens": 8, "seed": 20260911,
}:
    raise AssertionError("rater greedy decoding contract changed")

rubric = Path("refine-logs/RUBRIC_BEHAVIOR.md")
for trait in ("extraversion", "agreeableness"):
    items = rubric_items(rubric, trait)
    if len(items) != 6 or len({item[0] for item in items}) != 6:
        raise AssertionError(f"{trait} rubric is not the exact frozen six-item surface")
    request = build_item_request(items[0], "CUSTOMER: hello\nAGENT: response")
    if "Return exactly one integer" not in request or "TRANSCRIPT" not in request:
        raise AssertionError("rater request lost its strict output or transcript contract")

if trait_score([1, 2, 3, 4, 5, 6]).item() != 3.5:
    raise AssertionError("rater six-item mean changed")
if measured_rate(12, 3.0) != 4.0:
    raise AssertionError("rater throughput contract changed")

print("PASS: English rater scoring contracts")
