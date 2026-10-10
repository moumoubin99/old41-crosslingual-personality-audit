#!/usr/bin/env python3
"""Block H (public-data version): score public scripted dialogue with the frozen rater.

Spec: refine-logs/followup/H_public_spec.md (revision 2).  Two public corpora that already
carry published human Big Five labels are rendered into the rater's AGENT/CUSTOMER transcript
form and scored with the unchanged instrument (E1-E6 and A1-A6 of RUBRIC_BEHAVIOR.md, the
Mistral AWQ rater, the project argmax processor, greedy batch-one decoding).  The result is a
descriptive label concordance; nothing here touches a registered gate.

    --build-only           CPU: write units-{corpus}.jsonl and H_PUBLIC_PROVENANCE.json
    --corpus X (default)   GPU: score one corpus into the immutable scores-{corpus}.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from .inference_contract import trait_score
    from .rater_frozen_inputs import validate_rater_frozen_inputs
    from .rater_runtime import measured_rate, validate_rater_runtime
    from .run_rater_scoring import (
        PROJECT_SOURCE_ROOT, RATER_MAX_MODEL_LEN, _sha256, atomic_jsonl, build_item_request,
        load_rater_chat_template, parse_item_score, rater_decoding_contract,
        validate_context_window, validate_project_binding,
    )
    from .score_rater import rubric_items
except ImportError:  # direct-script contract tests
    from inference_contract import trait_score
    from rater_frozen_inputs import validate_rater_frozen_inputs
    from rater_runtime import measured_rate, validate_rater_runtime
    from run_rater_scoring import (
        PROJECT_SOURCE_ROOT, RATER_MAX_MODEL_LEN, _sha256, atomic_jsonl, build_item_request,
        load_rater_chat_template, parse_item_score, rater_decoding_contract,
        validate_context_window, validate_project_binding,
    )
    from score_rater import rubric_items


DATA_DIR = "data/collected/public_personality"
OUT_DIR = "process/results/xlpt/h-public"
PROVENANCE = f"{OUT_DIR}/H_PUBLIC_PROVENANCE.json"
CORPORA = ("friendspersona", "cped")
REVISIONS = {
    "friendspersona": {"repository": "emorynlp/personality-detection",
                       "commit": "3ec08a58dc7c708c5dfc314b3bff8f5808786928", "licence": None},
    "cped": {"repository": "scutcyr/CPED",
             "commit": "1e4b81c28a123f22387e06664f37e5dc9322380f", "licence": "Apache-2.0"},
}
INPUT_FILES = {
    "friendspersona": ("friendspersona/friends-personality.csv",),
    "cped": ("cped/train_split.csv", "cped/valid_split.csv", "cped/test_split.csv"),
}
EXPECTED_COUNTS = {
    "friendspersona": {"units": 711},
    "cped": {"labelled_show_speakers": 334, "eligible_show_speakers": 333,
             "identities": 287, "units": 569},
}
FRIENDS_ALIASES = {
    "Joey": "Joey Tribbiani", "Rachel": "Rachel Green", "Phoebe": "Phoebe Buffay",
    "Monica": "Monica Geller", "Ross": "Ross Geller", "Chandler": "Chandler Bing",
}
FRIENDS_HEADER = re.compile(r"<b>(.*?) for (.*?)</b>")
FRIENDS_TURN = re.compile(r"<b>([^<]*?)</b>:")
CPED_EXCLUDED_SPEAKER = "其他"
CPED_MIN_UTTERANCES = 3
CPED_K = 2
SELECTION_SALT = "20260929"
FROZEN_CONFIG = "process/consumption/frozen/english_execution_v1.json"
FROZEN_RECEIPT = "process/consumption/frozen/english_execution_receipt_v1.json"
STAGE_PREFIX = "[STAGE DIRECTION] "
# Every score record binds the code that produced it (checked by reduce_h_public) and the docs
# it ran under (recorded only: status lines of the spec are edited after the run).
CODE_FILES = ("score_public_personality.py", "reduce_h_public.py", "test_h_public_contract.py",
              "inference_contract.py", "rater_frozen_inputs.py", "rater_runtime.py",
              "run_rater_scoring.py", "score_rater.py")
DOC_FILES = ("refine-logs/followup/H_public_spec.md", "refine-logs/RUBRIC_BEHAVIOR.md")
SHARED_KEYS = ("rater_decoding", "prompt_config_sha256", "freeze_receipt_sha256",
               "rater_runtime_manifest_sha256", "units_sha256", "provenance_sha256", "code_sha256",
               "doc_sha256", "max_model_len", "scoring_elapsed_seconds", "rater_calls",
               "rater_calls_per_second", "eligible_for_gate", "eligible_for_analysis")
RATER_DECODING_RECORD = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_new_tokens_per_item": 8,
                         "seed": 20260911, "batch_size": 1, "cross_request_batching": False}


def context_excluded(max_prompt_tokens: int) -> bool:
    """Exclusion rule of validate_context_window: prompt plus 8 new tokens must fit in 4096."""
    return max_prompt_tokens + 8 >= RATER_MAX_MODEL_LEN


def receipt_file(project: Path, corpus: str) -> Path:
    return project / OUT_DIR / f"scores-{corpus}.receipt.json"


def code_hashes():
    here = Path(__file__).resolve().parent
    return {f"experiments/xlpt/{name}": _sha256(here / name) for name in CODE_FILES}


# ---------------------------------------------------------------- rendering

def render(lines, *, strip_directions: bool = False) -> str:
    """One source turn per line; target = AGENT, everyone else = CUSTOMER, directions unroled."""
    out = []
    for role, text in lines:
        if role == "direction":
            if not strip_directions:
                out.append(STAGE_PREFIX + text)
        elif role in ("AGENT", "CUSTOMER"):
            out.append(f"{role}: {text}")
        else:
            raise ValueError(f"unknown line role {role!r}")
    return "\n".join(out)


def _clean_html(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", fragment)).split())


# ---------------------------------------------------------------- FriendsPersona

def build_friends(data_dir: Path):
    path = Path(data_dir) / INPUT_FILES["friendspersona"][0]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    units, counts = [], defaultdict(int)
    for row in rows:
        segments = row["text"].split("<br><br>")
        header = FRIENDS_HEADER.fullmatch(segments[0].strip())
        character = row["character"]
        if header is None or _clean_html(header.group(2)) != character:
            raise ValueError(f"FriendsPersona header/character mismatch in row {row['']}")
        lines = []
        for segment in segments[1:]:
            segment = segment.strip()
            if not segment:
                continue
            turn = FRIENDS_TURN.match(segment)
            if turn:
                speaker = _clean_html(turn.group(1))
                text = _clean_html(segment[turn.end():])
                lines.append(("AGENT" if speaker == character else "CUSTOMER", text))
                continue
            text = _clean_html(segment)
            if text[:1] not in "([":
                raise ValueError(f"FriendsPersona unattributed non-direction segment in row {row['']}")
            lines.append(("direction", text))
        if not any(role == "AGENT" for role, _ in lines):
            counts["excluded_no_target_turn"] += 1
            continue
        units.append({
            "unit_id": f"fp-{int(row['']):04d}",
            "corpus": "friendspersona",
            "identity": FRIENDS_ALIASES.get(character, character),
            "character": character,
            "scene": row["scene_id"],
            "cluster": row["scene_id"],
            "labels": {"EXT": "high" if row["cEXT"] == "1" else "low",
                       "AGR": "high" if row["cAGR"] == "1" else "low"},
            "lines": [list(line) for line in lines],
        })
    for flag in ("cEXT", "cAGR"):
        if any(row[flag] not in ("0", "1") for row in rows):
            raise ValueError(f"FriendsPersona {flag} is not binary")
    counts["rows"] = len(rows)
    counts["units"] = len(units)
    return units, dict(counts)


# ---------------------------------------------------------------- CPED

def _cped_rows(data_dir: Path):
    rows = []
    for relative in INPUT_FILES["cped"]:
        with (Path(data_dir) / relative).open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    seen = set()
    for row in rows:
        if row["Utterance_ID"] in seen:
            raise ValueError(f"CPED duplicate Utterance_ID {row['Utterance_ID']}")
        seen.add(row["Utterance_ID"])
    return [row for row in rows if row["Speaker"] != CPED_EXCLUDED_SPEAKER]


def cped_components(rows):
    """Connected components of TV_ID under exact shared Speaker names; named by the smallest TV_ID."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_tv = {}
    for row in rows:
        tv = int(row["TV_ID"])
        parent.setdefault(tv, tv)
        other = first_tv.setdefault(row["Speaker"], tv)
        a, b = find(tv), find(other)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {tv: f"c{find(tv):02d}" for tv in parent}


def _selection_key(component, speaker, tv_id, dialogue_id) -> str:
    raw = "|".join((SELECTION_SALT, component, speaker, tv_id, dialogue_id))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_cped(data_dir: Path):
    rows = _cped_rows(data_dir)
    component_of = cped_components(rows)
    counts = defaultdict(int)
    counts["components"] = len(set(component_of.values()))
    dialogues = defaultdict(list)
    for row in rows:
        dialogues[(row["TV_ID"], row["Dialogue_ID"])].append(row)
    for utterances in dialogues.values():
        utterances.sort(key=lambda r: r["Utterance_ID"])

    labels = defaultdict(set)
    show_speakers = defaultdict(set)
    for row in rows:
        identity = (component_of[int(row["TV_ID"])], row["Speaker"])
        labels[identity].add((row["Extraversion"], row["Agreeableness"]))
        show_speakers[identity].add(row["TV_ID"])

    eligible_dialogues = defaultdict(list)
    for (tv_id, dialogue_id), utterances in dialogues.items():
        per_speaker = defaultdict(int)
        for row in utterances:
            per_speaker[row["Speaker"]] += 1
        for speaker, n in per_speaker.items():
            if n >= CPED_MIN_UTTERANCES:
                eligible_dialogues[(component_of[int(tv_id)], speaker)].append((tv_id, dialogue_id))

    units = []
    eligible_show_speakers = set()
    for identity in sorted(labels):
        component, speaker = identity
        if len(labels[identity]) != 1:
            counts["excluded_inconsistent_labels"] += 1
            continue
        ext, agr = next(iter(labels[identity]))
        if ext not in ("high", "low"):
            continue
        counts["labelled_show_speakers"] += len(show_speakers[identity])
        candidates = eligible_dialogues.get(identity, [])
        eligible_show_speakers.update((tv, speaker) for tv, _ in candidates)
        if not candidates:
            counts["excluded_no_dialogue_with_3_utterances"] += 1
            continue
        counts["identities"] += 1
        chosen = sorted(candidates, key=lambda c: _selection_key(component, speaker, *c))[:CPED_K]
        for tv_id, dialogue_id in chosen:
            lines = [["AGENT" if r["Speaker"] == speaker else "CUSTOMER", " ".join(r["Utterance"].split())]
                     for r in dialogues[(tv_id, dialogue_id)]]
            units.append({
                "unit_id": f"cped-{dialogue_id}-{hashlib.sha256(speaker.encode('utf-8')).hexdigest()[:8]}",
                "corpus": "cped",
                "identity": f"{component}|{speaker}",
                "character": speaker,
                "tv_id": tv_id,
                "dialogue_id": dialogue_id,
                "cluster": component,
                "labels": {"EXT": ext, "AGR": agr if agr in ("high", "low") else "unknown"},
                "lines": lines,
            })
    counts["eligible_show_speakers"] = len(eligible_show_speakers)
    counts["units"] = len(units)
    units.sort(key=lambda u: u["unit_id"])
    if len({u["unit_id"] for u in units}) != len(units):
        raise ValueError("CPED unit ids are not unique")
    return units, dict(counts)


BUILDERS = {"friendspersona": build_friends, "cped": build_cped}


def units_bytes(units) -> bytes:
    return "".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n" for u in units).encode("utf-8")


def check_counts(corpus: str, counts: dict) -> None:
    for key, expected in EXPECTED_COUNTS[corpus].items():
        if counts.get(key) != expected:
            raise ValueError(f"{corpus} {key}={counts.get(key)} differs from the spec's {expected}")


def build(project: Path):
    project = Path(project).resolve()
    data_dir = project / DATA_DIR
    out_dir = project / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"schema_version": 1, "spec": "refine-logs/followup/H_public_spec.md",
                  "builder": "experiments/xlpt/score_public_personality.py",
                  "builder_sha256": _sha256(Path(__file__).resolve()), "corpora": {}}
    for corpus in CORPORA:
        units, counts = BUILDERS[corpus](data_dir)
        check_counts(corpus, counts)
        payload = units_bytes(units)
        target = out_dir / f"units-{corpus}.jsonl"
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"refusing to replace a different existing unit file: {target}")
        target.write_bytes(payload)
        provenance["corpora"][corpus] = {
            **REVISIONS[corpus],
            "inputs": {f"{DATA_DIR}/{rel}": _sha256(data_dir / rel) for rel in INPUT_FILES[corpus]},
            "units_file": f"{OUT_DIR}/units-{corpus}.jsonl",
            "units_sha256": hashlib.sha256(payload).hexdigest(),
            "counts": counts,
        }
    atomic_json(project / PROVENANCE, provenance)
    print(json.dumps({corpus: meta["counts"] for corpus, meta in provenance["corpora"].items()},
                     ensure_ascii=False))
    return provenance


def atomic_json(path: Path, payload) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_verified_units(project: Path, corpus: str):
    """Units must equal a fresh deterministic rebuild and the recorded provenance hash."""
    provenance = json.loads((project / PROVENANCE).read_text(encoding="utf-8"))
    if (provenance.get("builder") != "experiments/xlpt/score_public_personality.py"
            or provenance.get("builder_sha256") != _sha256(Path(__file__).resolve())):
        raise ValueError("H_PUBLIC_PROVENANCE.json was built by a different scorer; rerun --build-only")
    meta = provenance["corpora"][corpus]
    for relative, digest in meta["inputs"].items():
        if _sha256(project / relative) != digest:
            raise ValueError(f"public input changed since the build: {relative}")
    payload = (project / meta["units_file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != meta["units_sha256"]:
        raise ValueError(f"{corpus} unit file differs from its provenance hash")
    units, counts = BUILDERS[corpus](project / DATA_DIR)
    check_counts(corpus, counts)
    if units_bytes(units) != payload:
        raise ValueError(f"{corpus} unit file is not the deterministic rebuild")
    return units, meta["units_sha256"], _sha256(project / PROVENANCE)


# ---------------------------------------------------------------- scoring

def scoring_passes(corpus: str):
    """(rendering, traits): Friends also gets an E-only pass without stage directions."""
    passes = [("full", ("extraversion", "agreeableness"))]
    if corpus == "friendspersona":
        passes.append(("no_directions", ("extraversion",)))
    return passes


def run(args, *, llm_cls=None, sampling_params_cls=None,
        frozen_validator=validate_rater_frozen_inputs, runtime_validator=validate_rater_runtime,
        project_binding_validator=validate_project_binding, clock=time.time):
    project = Path(args.project).resolve()
    project_binding_validator(project, PROJECT_SOURCE_ROOT)
    _, config_sha, receipt_sha = frozen_validator(project, FROZEN_CONFIG, FROZEN_RECEIPT)
    corpus = args.corpus
    units, units_sha, provenance_sha = load_verified_units(project, corpus)
    out = project / OUT_DIR / f"scores-{corpus}.jsonl"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing immutable score shard: {out}")
    items = {trait: rubric_items(project / "refine-logs/RUBRIC_BEHAVIOR.md", trait)
             for trait in ("extraversion", "agreeableness")}

    rater_path = Path(args.rater_path).resolve()
    runtime_sha = runtime_validator(project / args.runtime_manifest, rater_path, PROJECT_SOURCE_ROOT)
    if llm_cls is None or sampling_params_cls is None:
        from vllm import LLM, SamplingParams

        llm_cls, sampling_params_cls = LLM, SamplingParams
    engine = llm_cls(
        model=str(rater_path),
        trust_remote_code=False,
        quantization="awq",
        dtype="auto",
        max_model_len=RATER_MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0},
        logits_processors=["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"],
    )
    tokenizer = engine.get_tokenizer()
    chat_template = load_rater_chat_template(rater_path)
    params = sampling_params_cls(**rater_decoding_contract())

    started = clock()
    calls = 0
    records = []
    for unit in units:
        for rendering, traits in scoring_passes(corpus):
            rendered = render(unit["lines"], strip_directions=rendering == "no_directions")
            # Every request is tokenized before any exclusion, so max_prompt_tokens is the true
            # maximum and the exclusion status is recomputable from it (context_excluded).
            requests = []
            overflow = False
            max_prompt_tokens = 0
            for trait in traits:
                for item in items[trait]:
                    ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": build_item_request(item, rendered)}],
                        chat_template=chat_template, tokenize=True, add_generation_prompt=True,
                    )
                    max_prompt_tokens = max(max_prompt_tokens, len(ids))
                    try:
                        ids = validate_context_window(ids)
                    except ValueError:
                        overflow = True
                        continue
                    requests.append((trait, item[0], ids))
            if overflow != context_excluded(max_prompt_tokens):
                raise AssertionError("context guard disagrees with the recorded token maximum")
            record = {"schema_version": 1, "unit_id": unit["unit_id"], "corpus": corpus,
                      "rendering": rendering, "identity": unit["identity"], "cluster": unit["cluster"],
                      "labels": unit["labels"], "max_prompt_tokens": max_prompt_tokens}
            if overflow:  # excluded and counted, never truncated
                record.update({"status": "excluded_context", "scores": {}, "rater_raw_by_item": {},
                               "parse_errors": [], "trait_scores": {}})
                records.append(record)
                continue
            scores, raw_outputs, errors = {}, {}, []
            for trait, item_id, ids in requests:
                raw = engine.generate([{"prompt_token_ids": list(ids)}], params,
                                      use_tqdm=False)[0].outputs[0].text.strip()
                calls += 1
                raw_outputs[item_id] = raw
                try:
                    scores[item_id] = parse_item_score(raw)
                except ValueError:
                    scores[item_id] = None
                    errors.append(item_id)
            trait_scores = {}
            for trait in traits:
                ordered = [scores[item[0]] for item in items[trait]]
                trait_scores[trait] = (None if any(v is None for v in ordered)
                                       else float(trait_score(ordered).item()))
            record.update({"status": "scored", "scores": scores, "rater_raw_by_item": raw_outputs,
                           "parse_errors": errors, "trait_scores": trait_scores})
            records.append(record)
    elapsed = clock() - started
    rate = measured_rate(calls, elapsed) if calls else 0.0
    shared = {
        "rater_decoding": dict(RATER_DECODING_RECORD),
        "prompt_config_sha256": config_sha, "freeze_receipt_sha256": receipt_sha,
        "rater_runtime_manifest_sha256": runtime_sha, "units_sha256": units_sha,
        "provenance_sha256": provenance_sha, "code_sha256": code_hashes(),
        "doc_sha256": {rel: _sha256(project / rel) for rel in DOC_FILES},
        "max_model_len": RATER_MAX_MODEL_LEN, "scoring_elapsed_seconds": elapsed,
        "rater_calls": calls, "rater_calls_per_second": rate,
        "eligible_for_gate": False, "eligible_for_analysis": True,
    }
    for record in records:
        record.update(shared)
    receipt_path = receipt_file(project, corpus)
    if out.exists() or receipt_path.exists():
        raise ValueError(f"refusing to overwrite existing immutable score shard or receipt: {out}")
    atomic_jsonl(out, records)
    # The receipt is written once, immediately after the shard; the chain copies its digest into
    # the completion ledger and the reducer requires both to match the shard.
    atomic_json(receipt_path, {
        "schema_version": 1, "corpus": corpus, "shard": f"{OUT_DIR}/scores-{corpus}.jsonl",
        "shard_sha256": _sha256(out), "records": len(records), "rater_calls": calls,
        "units_sha256": units_sha, "provenance_sha256": provenance_sha, "code_sha256": code_hashes(),
    })
    print(json.dumps({"output": str(out), "records": len(records), "calls": calls,
                      "excluded_context": sum(r["status"] == "excluded_context" for r in records),
                      "elapsed_seconds": elapsed, "calls_per_second": rate}, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--corpus", choices=CORPORA)
    ap.add_argument("--rater-path")
    ap.add_argument("--runtime-manifest", default="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    return ap


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        if parsed.build_only:
            build(Path(parsed.project))
            raise SystemExit(0)
        if not parsed.corpus or not parsed.rater_path:
            raise SystemExit("--corpus and --rater-path are required for scoring")
        raise SystemExit(run(parsed))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
