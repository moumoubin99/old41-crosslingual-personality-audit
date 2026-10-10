#!/usr/bin/env python3
"""Score frozen transcripts with the deterministic English behavior rubric."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


ITEM_ROW = re.compile(r"^\|\s*([EA][1-6])\s*\|(.+)$")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rubric_items(path, trait):
    prefix = {"extraversion": "E", "agreeableness": "A"}[trait]
    items = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        match = ITEM_ROW.match(raw.strip())
        if match and match.group(1).startswith(prefix):
            cells = [x.strip() for x in match.group(2).split("|") if x.strip()]
            if len(cells) != 5:
                raise ValueError(f"malformed rubric row {match.group(1)}")
            items.append((match.group(1), *cells))
    if len(items) != 6:
        raise ValueError(f"expected six {trait} items, found {len(items)}")
    return items


def parse_scores(text, item_ids):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("rater output has no JSON object")
    payload = json.loads(match.group(0))
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(scores, dict) or set(scores) != set(item_ids):
        raise ValueError("rater JSON must contain exactly the six requested item ids")
    if any(type(value) is not int for value in scores.values()):
        raise ValueError("all rubric scores must be JSON integers, not coerced numeric values")
    clean = dict(scores)
    if any(value < 1 or value > 7 for value in clean.values()):
        raise ValueError("all rubric scores must be integers 1..7")
    return clean


def transcript(row):
    turns = []
    for customer, agent in zip(row["customer_turns"], row["agent_turns"]):
        turns.extend([f"CUSTOMER: {customer}", f"AGENT: {agent}"])
    return "\n".join(turns)


def main(args):
    project = Path(args.project).resolve()
    source = read_jsonl(project / args.input_jsonl)
    items = rubric_items(project / "refine-logs/RUBRIC_BEHAVIOR.md", args.trait)
    item_ids = [row[0] for row in items]
    rubric = "\n".join(f"{iid} | {facet} | {prompt} | 1={a1} | 4={a4} | 7={a7}"
                       for iid, facet, prompt, a1, a4, a7 in items)
    from vllm import LLM, SamplingParams
    engine = LLM(model=str(Path(args.rater_path).resolve()), trust_remote_code=True,
                 quantization="awq", dtype="auto", max_model_len=4096,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 enable_prefix_caching=False, enforce_eager=args.enforce_eager)
    tokenizer = engine.get_tokenizer()
    params = SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=8, seed=20260911)
    output_rows = []
    for row in source:
        prompt = (
            "Score only the AGENT behavior in the transcript. Use every item independently. "
            "Return strict JSON as {\"scores\":{\"ID\":integer,...}} with no omitted ids.\n\n"
            + rubric + "\n\nTRANSCRIPT\n" + transcript(row)
        )
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw = engine.generate([rendered], params, use_tqdm=False)[0].outputs[0].text
        scores = parse_scores(raw, item_ids)
        output_rows.append({
            "schema_version": 1, "source_scenario_id": row["scenario_id"],
            "checkpoint_id": row["checkpoint_id"], "setting": row["setting"],
            "seed": row["seed"], "trait": args.trait, "instrument_language": "English",
            "scores": scores, "trait_score": sum(scores.values()) / 6,
            "rater_raw": raw, "eligible_for_analysis": bool(row.get("eligible_for_analysis")),
        })
    out = (project / args.output_jsonl).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + f".tmp-{os.getpid()}")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    os.replace(temp, out)
    print(json.dumps({"output": str(out), "records": len(output_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--rater-path", required=True)
    ap.add_argument("--trait", choices=("extraversion", "agreeableness"), required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    ap.add_argument("--enforce-eager", action="store_true")
    sys.exit(main(ap.parse_args()) or 0)
