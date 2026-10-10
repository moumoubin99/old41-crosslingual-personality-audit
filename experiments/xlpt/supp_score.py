"""Supplementary descriptive scoring pass (namespace supp-20261004).

Exploratory only: nothing written here is a gate input, and no canonical score
shard is read for scoring or overwritten.  The pass re-scores FROZEN transcripts

* with the original rater under batched decoding (runtime replication),
* with two additional open-weight raters (same English item prompts),
* after deterministic surface interventions on the agent turns, and
* with a single unvalidated Chinese rendering of the Extraversion items.

Differences from the canonical scorer are deliberate and recorded in every row:
requests are batched across items/transcripts, and an unparseable rater output
is recorded as missing instead of aborting the shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
import time
from pathlib import Path

from experiments.xlpt.run_rater_scoring import (
    INTEGER,
    RATER_MAX_MODEL_LEN,
    RATER_MAX_TOKENS,
    build_item_request,
    load_rater_chat_template,
    rater_decoding_contract,
)
from experiments.xlpt.score_rater import rubric_items

NAMESPACE = "supp-20261004"
OUT_ROOT = f"process/results/xlpt/{NAMESPACE}"
# Declared before any supplementary score existed: the six lowest registered seeds.
SUPP_SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815, 20260816)
GENERATORS_ALL = ("qwen3-8b", "aya-expanse-8b", "Llama-3.1-8B-Instruct", "gemma-3-12b-it")
GENERATORS_PZH = ("qwen3-8b", "aya-expanse-8b", "Llama-3.1-8B-Instruct")
# cell -> (high phase, low phase, generators)
CELLS = {
    "Pen_Den": ("b4-extraversion-high", "b4-extraversion-low", GENERATORS_ALL),
    "Pen_Dzh": ("zh-b4-extraversion-high", "zh-b4-extraversion-low", GENERATORS_ALL),
    "Pzh_Den": ("bc-pzhn-extraversion-high-den", "bc-pzhn-extraversion-low-den", GENERATORS_PZH),
    "Pzh_Dzh": ("bc-pzhn-extraversion-high-dzh", "bc-pzhn-extraversion-low-dzh", GENERATORS_PZH),
}
VARIANTS = ("none", "strip_edges", "trunc2", "swap_edges")
# ASCII terminators end a sentence only before whitespace/end, so "29.99" stays whole.
SENTENCE = re.compile(
    r".*?(?:[。！？…]+[\"'”’)）】]*|[.!?]+[\"'”’)\]]*(?=\s|$)|\n|$)\s*", re.S)
LENIENT = re.compile(r"\s*([1-7])(?![0-9])")
ZH_REQUEST_HEAD = (
    "仅针对对话记录中 AGENT 的行为，就下面这一个条目评分。"
    "只返回一个 1 到 7 之间的整数，不要输出任何其他文字。\n"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sentences(text: str):
    parts = [m.group(0) for m in SENTENCE.finditer(text) if m.group(0).strip()]
    return parts or [text]


def strip_edges(agent_turns):
    """Drop the first sentence of the first turn and the last sentence of the last turn."""
    turns = list(agent_turns)
    changed = 0
    first = sentences(turns[0])
    if len(first) >= 2:
        turns[0] = "".join(first[1:]).strip()
        changed += 1
    last = sentences(turns[-1])
    if len(last) >= 2:
        turns[-1] = "".join(last[:-1]).strip()
        changed += 1
    return turns, changed


def trunc2(agent_turns):
    """Keep only the first two sentences of every agent turn."""
    turns, changed = [], 0
    for turn in agent_turns:
        parts = sentences(turn)
        if len(parts) > 2:
            changed += 1
        turns.append("".join(parts[:2]).strip())
    return turns, changed


def swap_edges(agent_turns, partner_turns):
    """Take the opening and closing sentence from the opposite-pole partner transcript."""
    turns = list(agent_turns)
    changed = 0
    own_first, other_first = sentences(turns[0]), sentences(partner_turns[0])
    if len(own_first) >= 2 and len(other_first) >= 2:
        turns[0] = (other_first[0].strip() + " " + "".join(own_first[1:]).strip()).strip()
        changed += 1
    own_last, other_last = sentences(turns[-1]), sentences(partner_turns[-1])
    if len(own_last) >= 2 and len(other_last) >= 2:
        turns[-1] = ("".join(own_last[:-1]).strip() + " " + other_last[-1].strip()).strip()
        changed += 1
    return turns, changed


def render(customer_turns, agent_turns) -> str:
    out = []
    for customer, agent in zip(customer_turns, agent_turns):
        out.extend([f"CUSTOMER: {customer}", f"AGENT: {agent}"])
    return "\n".join(out)


def zh_items(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(i["id"], i["facet"], i["prompt"], i["a1"], i["a4"], i["a7"]) for i in payload["items"]]


def build_zh_request(item, rendered: str) -> str:
    item_id, facet, prompt, a1, a4, a7 = item
    return (ZH_REQUEST_HEAD + f"条目 {item_id} | {facet} | {prompt}\n"
            f"锚点 1={a1} | 4={a4} | 7={a7}\n\n对话记录\n" + rendered)


def load_units(project: Path, seeds, variants, cells):
    """One unit per (cell, generator, pole, seed, scenario, variant)."""
    units = []
    for cell in cells:
        high_phase, low_phase, generators = CELLS[cell]
        for gen in generators:
            for seed in seeds:
                rows = {}
                for pole, phase in (("high", high_phase), ("low", low_phase)):
                    path = project / f"process/results/xlpt/{phase}/{gen}/seed-{seed}.jsonl"
                    for row in read_jsonl(path):
                        rows[(pole, row["scenario_id"])] = (row, phase)
                for (pole, scenario), (row, phase) in sorted(rows.items()):
                    agent = row.get("agent_turns") or []
                    customer = row.get("customer_turns") or []
                    if len(agent) != len(customer) or not agent or not all(
                            isinstance(t, str) and t.strip() for t in agent):
                        continue
                    for variant in variants:
                        if variant == "none":
                            turns, changed = list(agent), 0
                        elif variant == "strip_edges":
                            turns, changed = strip_edges(agent)
                        elif variant == "trunc2":
                            turns, changed = trunc2(agent)
                        elif variant == "swap_edges":
                            other = rows.get(("low" if pole == "high" else "high", scenario))
                            partner = (other[0].get("agent_turns") or []) if other else []
                            if len(partner) != len(agent) or not all(
                                    isinstance(t, str) and t.strip() for t in partner):
                                continue
                            turns, changed = swap_edges(agent, partner)
                        else:
                            raise ValueError(variant)
                        if not all(t.strip() for t in turns):
                            continue
                        units.append({
                            "cell": cell, "source_phase": phase, "checkpoint_id": gen,
                            "pole": pole, "seed": seed, "source_scenario_id": scenario,
                            "variant": variant, "turns_changed": changed,
                            "agent_chars": sum(len(t) for t in turns),
                            "rendered": render(customer, turns),
                        })
    return units


def load_public_units(project: Path, corpus: str):
    from experiments.xlpt.score_public_personality import render as public_render

    units = []
    for unit in read_jsonl(project / f"process/results/xlpt/h-public/units-{corpus}.jsonl"):
        units.append({
            "corpus": corpus, "unit_id": unit["unit_id"], "identity": unit["identity"],
            "cluster": unit["cluster"], "labels": unit["labels"], "variant": "full",
            "rendered": public_render(unit["lines"]),
        })
    return units


_POOL = {}


def _pool_init(rater_path, kwargs):
    from transformers import AutoTokenizer

    _POOL["tokenizer"] = AutoTokenizer.from_pretrained(rater_path)
    _POOL["kwargs"] = kwargs


def _pool_tokenize(content):
    ids = _POOL["tokenizer"].apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
        **_POOL["kwargs"])
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def build_engine(args):
    from vllm import LLM

    kwargs = dict(
        model=str(args.rater_path), trust_remote_code=False, dtype="auto",
        max_model_len=RATER_MAX_MODEL_LEN, gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False, limit_mm_per_prompt={"image": 0},
        max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs,
    )
    if args.rater == "mistral":
        # Same engine configuration as the canonical scorer, batching aside.
        kwargs.update(quantization="awq", enforce_eager=True, logits_processors=[
            "experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"])
    return LLM(**kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--rater", required=True, choices=("mistral", "gemma", "llama"))
    parser.add_argument("--rater-path", required=True)
    parser.add_argument("--task", required=True, choices=("transcripts", "public"))
    parser.add_argument("--tag", required=True, help="output file stem")
    parser.add_argument("--tok-workers", type=int, default=16)
    parser.add_argument("--variants", default="none")
    parser.add_argument("--traits", default="extraversion,agreeableness")
    parser.add_argument("--instrument", default="en", choices=("en", "zh"))
    parser.add_argument("--cells", default=",".join(CELLS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SUPP_SEEDS))
    parser.add_argument("--corpus", default="friendspersona")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args(argv)

    project = Path(args.project).resolve()
    out = project / OUT_ROOT / f"{args.tag}.jsonl"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing supplementary shard: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    traits = tuple(t for t in args.traits.split(",") if t)
    rubric_path = project / "refine-logs/RUBRIC_BEHAVIOR.md"
    zh_path = project / "experiments/xlpt/supp_rubric_zh.json"
    if args.instrument == "zh":
        if traits != ("extraversion",):
            raise SystemExit("the Chinese rendering covers the Extraversion items only")
        items = {"extraversion": zh_items(zh_path)}
        builder, instrument_sha = build_zh_request, sha256_file(zh_path)
    else:
        items = {t: rubric_items(rubric_path, t) for t in traits}
        builder, instrument_sha = build_item_request, sha256_file(rubric_path)

    if args.task == "transcripts":
        seeds = tuple(int(s) for s in args.seeds.split(","))
        units = load_units(project, seeds, tuple(args.variants.split(",")),
                           tuple(args.cells.split(",")))
    else:
        units = load_public_units(project, args.corpus)
    if args.limit:
        units = units[:: max(1, len(units) // args.limit)][: args.limit]

    template = load_rater_chat_template(Path(args.rater_path)) if args.rater == "mistral" else None
    kwargs = {"chat_template": template} if template else {}
    contents = [(index, item, builder(item, unit["rendered"]))
                for index, unit in enumerate(units) for trait in traits for item in items[trait]]
    # The server is shared and CPU-throttled, so prompts are tokenized by a worker pool started
    # before the engine exists; a sample is re-tokenized with the engine's own tokenizer below
    # and any difference discards the pooled result.
    pooled = None
    if args.tok_workers > 1:
        try:
            with multiprocessing.get_context("fork").Pool(
                    args.tok_workers, _pool_init, (str(args.rater_path), kwargs)) as pool:
                pooled = pool.map(_pool_tokenize, [c for _, _, c in contents], chunksize=128)
        except Exception as error:  # fall back to the engine tokenizer
            print(json.dumps({"tag": args.tag, "pooled_tokenization": "failed: %r" % (error,)}),
                  flush=True)
            pooled = None

    engine = build_engine(args)
    from vllm import SamplingParams

    tokenizer = engine.get_tokenizer()
    params = SamplingParams(**rater_decoding_contract())

    def tokenize(content):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            **kwargs)
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    if pooled is not None:
        sample = sorted(set(range(0, len(contents), 499)) | {len(contents) - 1})
        if any(pooled[k] != tokenize(contents[k][2]) for k in sample):
            print(json.dumps({"tag": args.tag, "pooled_tokenization": "mismatch, redone serially"}),
                  flush=True)
            pooled = None
        else:
            print(json.dumps({"tag": args.tag, "pooled_tokenization": "verified",
                              "sample": len(sample)}), flush=True)

    requests = []  # (unit index, item id, token ids)
    for unit in units:
        unit["max_prompt_tokens"] = 0
        unit["overflow"] = False
    for k, (index, item, content) in enumerate(contents):
        unit = units[index]
        if True:
            if True:
                ids = pooled[k] if pooled is not None else tokenize(content)
                unit["max_prompt_tokens"] = max(unit["max_prompt_tokens"], len(ids))
                if len(ids) + RATER_MAX_TOKENS >= RATER_MAX_MODEL_LEN:
                    unit["overflow"] = True
                    continue
                requests.append((index, item[0], ids))
    requests = [r for r in requests if not units[r[0]]["overflow"]]

    # Raw completions are checkpointed per chunk so an interrupted run resumes
    # instead of re-scoring; the request list is deterministic for fixed arguments.
    partial = out.with_suffix(".partial.jsonl")
    raws = {}
    if partial.exists():
        for row in read_jsonl(partial):
            raws[(row["i"], row["item"])] = row["raw"]
    todo = [r for r in requests if (r[0], r[1]) not in raws]
    resumed = len(requests) - len(todo)
    started = time.time()
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        outputs = engine.generate([{"prompt_token_ids": ids} for _, _, ids in chunk],
                                  params, use_tqdm=False)
        with open(partial, "a", encoding="utf-8") as handle:
            for (index, item_id, _), output in zip(chunk, outputs):
                raws[(index, item_id)] = output.outputs[0].text
                handle.write(json.dumps({"i": index, "item": item_id,
                                         "raw": raws[(index, item_id)]}) + "\n")
        done = start + len(chunk)
        print(json.dumps({"tag": args.tag, "done": resumed + done, "total": len(requests),
                          "calls_per_second": done / max(time.time() - started, 1e-9)}),
              flush=True)
    elapsed = time.time() - started

    meta = {
        "namespace": NAMESPACE, "rater": args.rater, "rater_path": Path(args.rater_path).name,
        "instrument_language": "Chinese" if args.instrument == "zh" else "English",
        "instrument_sha256": instrument_sha,
        "code_sha256": sha256_file(Path(__file__)),
        "rater_decoding": dict(rater_decoding_contract(), cross_request_batching=True),
        "eligible_for_gate": False, "exploratory": True,
    }
    records = []
    for index, unit in enumerate(units):
        scores, lenient, raw_by_item, errors = {}, {}, {}, []
        for trait in traits:
            for item in items[trait]:
                raw = raws.get((index, item[0]))
                if raw is None:
                    continue
                raw_by_item[item[0]] = raw
                strict = INTEGER.fullmatch(raw)
                loose = LENIENT.match(raw)
                if strict:
                    scores[item[0]] = int(strict.group(1))
                else:
                    errors.append(item[0])
                if loose:
                    lenient[item[0]] = int(loose.group(1))
        trait_scores = {}
        for trait in traits:
            ids = [item[0] for item in items[trait]]
            if all(i in scores for i in ids):
                trait_scores[trait] = sum(scores[i] for i in ids) / len(ids)
        record = {k: v for k, v in unit.items() if k not in ("rendered", "overflow")}
        record.update(meta)
        record.update({
            "status": "excluded_context" if unit["overflow"] else "scored",
            "scores": scores, "scores_lenient": lenient, "rater_raw_by_item": raw_by_item,
            "parse_errors": errors, "trait_scores": trait_scores,
            "transcript_sha256": hashlib.sha256(unit["rendered"].encode()).hexdigest(),
        })
        records.append(record)

    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out)
    receipt = {"output": str(out.relative_to(project)), "sha256": sha256_file(out),
               "records": len(records), "calls": len(requests),
               "calls_resumed_from_checkpoint": resumed, "elapsed_seconds": elapsed,
               "calls_per_second": len(todo) / max(elapsed, 1e-9), "args": vars(args)}
    out.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps(receipt, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
