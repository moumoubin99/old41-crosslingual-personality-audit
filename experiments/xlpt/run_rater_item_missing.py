#!/usr/bin/env python3
"""Score one generation shard with the frozen English rubric, recording item parse failures.

Parse policy ``strict_item_missing_v1`` (FOLLOWUP_PLAN amendment of 2026-09-30). The frozen
scorer ``run_rater_scoring.py`` aborts a whole 100-row shard when one of its 600 greedy
rater outputs is not exactly one integer 1..7; four block-A shards died that way on outputs
such as ``'<LOCAL_PATH>'`` (a partial byte-level token decoded ahead of the digit). This scorer
keeps every frozen input identical -- engine, template, date freeze, rubric, request text,
decoding contract, context check and the unchanged ``parse_item_score`` -- and changes one
thing only: an item whose output fails that parser is recorded as missing instead of raising.

* The item score is ``null``; the raw decoded text (unstripped), the parser input, the parser
  error, the item id and the generated token ids go to ``parse_failures``. Nothing is retried.
* A trait score needs all six items (``trait_score`` unchanged); otherwise it is ``null``.
* ``--namespace canonical`` writes the phase's own score directory; ``--namespace rescue``
  writes ``{score_dir}-rescue`` with ``rescue_run`` true, for a shard whose canonical run
  crashed. Either refuses when the canonical or the rescue shard already exists, so a cell
  can never have two scored versions.

Rows are ``schema_version`` 3; everything the frozen scorer writes is kept under its key, and
``scorer_sha256`` / ``frozen_scorer_sha256`` bind this wrapper and the frozen module it imports.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

try:
    from .inference_contract import trait_score
    from .rater_frozen_inputs import validate_rater_frozen_inputs
    from .rater_runtime import measured_rate, validate_rater_runtime
    from .run_rater_scoring import (
        QUALIFICATION_SEEDS,
        RATER_MAX_MODEL_LEN,
        SOURCE_CHECKPOINTS,
        SOURCE_PHASES,
        _sha256,
        atomic_jsonl,
        build_item_request,
        contract_smoke,
        load_rater_chat_template,
        parse_item_score,
        rater_decoding_contract,
        read_jsonl,
        validate_context_window,
        validate_project_binding,
        validate_source_shard,
    )
    from .score_rater import rubric_items, transcript
except ImportError:  # direct-script contract tests
    from inference_contract import trait_score
    from rater_frozen_inputs import validate_rater_frozen_inputs
    from rater_runtime import measured_rate, validate_rater_runtime
    from run_rater_scoring import (
        QUALIFICATION_SEEDS,
        RATER_MAX_MODEL_LEN,
        SOURCE_CHECKPOINTS,
        SOURCE_PHASES,
        _sha256,
        atomic_jsonl,
        build_item_request,
        contract_smoke,
        load_rater_chat_template,
        parse_item_score,
        rater_decoding_contract,
        read_jsonl,
        validate_context_window,
        validate_project_binding,
        validate_source_shard,
    )
    from score_rater import rubric_items, transcript


SCHEMA_VERSION = 3
PARSE_POLICY = "strict_item_missing_v1"
NAMESPACES = ("canonical", "rescue")
RESCUE_SUFFIX = "-rescue"


def score_paths(project: Path, score_dir: str, checkpoint: str, seed: int) -> dict:
    """The canonical and rescue locations of one scored cell."""
    base = Path(project) / "process/results/xlpt"
    return {
        "canonical": base / score_dir / checkpoint / f"seed-{seed}.jsonl",
        "rescue": base / f"{score_dir}{RESCUE_SUFFIX}" / checkpoint / f"seed-{seed}.jsonl",
    }


def cell_lock_path(project: Path, score_dir: str, checkpoint: str, seed: int) -> Path:
    """One lock per scored cell, shared by the canonical and the rescue namespace."""
    return (Path(project) / "process/results/xlpt/.cell-locks"
            / f"{score_dir}__{checkpoint}__seed-{seed}.lock")


@contextlib.contextmanager
def cell_lock(path: Path):
    """Hold the cell's exclusive lock, or refuse at once if another scorer holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another scorer holds the cell lock: {path}") from None
        yield
    finally:
        handle.close()


def publish_no_replace(path: Path, rows) -> None:
    """Write the shard to a temp file and hard-link it into place; never replace a shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    try:
        os.link(temp, path)
    except FileExistsError:
        raise ValueError(f"refusing to overwrite a shard written during scoring: {path}") from None
    finally:
        temp.unlink()


def score_item(raw_decoded: str, token_ids, item_id: str):
    """Apply the frozen parser; return (score or None, failure record or None)."""
    parser_input = raw_decoded.strip()
    try:
        return parse_item_score(parser_input), None
    except ValueError as exc:
        return None, {
            "item_id": item_id,
            "raw_decoded": raw_decoded,
            "parser_input": parser_input,
            "error": str(exc),
            "token_ids": [int(token) for token in token_ids],
        }


def complete_trait_score(ordered):
    """Six parsed items give the frozen trait mean; any missing item gives None."""
    if len(ordered) != 6 or any(value is None for value in ordered):
        return None
    return float(trait_score(ordered).item())


def run(
    args,
    *,
    llm_cls=None,
    sampling_params_cls=None,
    frozen_validator=validate_rater_frozen_inputs,
    runtime_validator=validate_rater_runtime,
    project_binding_validator=validate_project_binding,
    clock=time.time,
):
    if args.seed not in QUALIFICATION_SEEDS:
        raise ValueError("rater seed selector must be in the registered qualification block")
    namespace = getattr(args, "namespace", None)
    if namespace not in NAMESPACES:
        raise ValueError(f"score namespace must be one of {NAMESPACES}, got {namespace!r}")
    project = Path(args.project).resolve()
    project_binding_validator(project, PROJECT_SOURCE_ROOT)
    _, config_sha, receipt_sha = frozen_validator(project, args.prompt_config, args.freeze_receipt)
    source_checkpoint = args.source_checkpoint
    if source_checkpoint not in SOURCE_CHECKPOINTS:
        raise ValueError(f"unregistered rater source checkpoint: {source_checkpoint}")
    source_phase = args.source_phase
    if source_phase not in SOURCE_PHASES:
        raise ValueError(f"unregistered rater source phase: {source_phase}")
    phase = SOURCE_PHASES[source_phase]
    if (args.prompt_config != phase["prompt_config"]
            or args.freeze_receipt != phase["freeze_receipt"]):
        raise ValueError(
            f"phase {source_phase} must be scored against {phase['prompt_config']} "
            f"and {phase['freeze_receipt']}"
        )
    paths = score_paths(project, phase["score_dir"], source_checkpoint, args.seed)
    # One lock covers both namespaces of the cell from the first existence check to publication.
    with cell_lock(cell_lock_path(project, phase["score_dir"], source_checkpoint, args.seed)):
        return _score_cell(args, project, phase, source_phase, source_checkpoint, namespace, paths,
                           config_sha, receipt_sha, llm_cls, sampling_params_cls,
                           runtime_validator, clock)


def _score_cell(args, project, phase, source_phase, source_checkpoint, namespace, paths,
                config_sha, receipt_sha, llm_cls, sampling_params_cls, runtime_validator, clock):
    for existing in paths.values():
        if existing.exists():
            raise ValueError(f"refusing to score a cell that already has a shard: {existing}")
    out = paths[namespace]

    source_path = (
        project / f"process/results/xlpt/{source_phase}/{source_checkpoint}/seed-{args.seed}.jsonl"
    )
    source = validate_source_shard(
        project, read_jsonl(source_path), args.seed, source_checkpoint, source_phase
    )
    source_sha = _sha256(source_path)
    scorer_sha = _sha256(Path(__file__).resolve())
    # The frozen module whose parser, request builder and decoding contract this wrapper imports.
    frozen_scorer_sha = _sha256(Path(sys.modules[parse_item_score.__module__].__file__).resolve())

    rater_path = Path(args.rater_path).resolve()
    runtime_sha = runtime_validator(project / args.runtime_manifest, rater_path, PROJECT_SOURCE_ROOT)

    if llm_cls is None or sampling_params_cls is None:
        from vllm import LLM, SamplingParams

        llm_cls = LLM
        sampling_params_cls = SamplingParams

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
    failures_total = 0
    output_rows = []
    for row in source:
        rendered_transcript = transcript(row)
        for trait in ("extraversion", "agreeableness"):
            items = rubric_items(project / "refine-logs/RUBRIC_BEHAVIOR.md", trait)
            scores = {}
            raw_outputs = {}
            failures = []
            for item_id, facet, prompt, anchor1, anchor4, anchor7 in items:
                request = build_item_request((item_id, facet, prompt, anchor1, anchor4, anchor7), rendered_transcript)
                messages = [{"role": "user", "content": request}]
                prompt_token_ids = tokenizer.apply_chat_template(
                    messages,
                    chat_template=chat_template,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                prompt_token_ids = validate_context_window(prompt_token_ids)
                output = engine.generate(
                    [{"prompt_token_ids": list(prompt_token_ids)}],
                    params,
                    use_tqdm=False,
                )[0].outputs[0]
                score, failure = score_item(output.text, output.token_ids, item_id)
                scores[item_id] = score
                raw_outputs[item_id] = output.text.strip()
                if failure is not None:
                    failures.append(failure)
                calls += 1
            ordered = [scores[item[0]] for item in items]
            failures_total += len(failures)
            output_rows.append({
                "schema_version": SCHEMA_VERSION,
                "parse_policy": PARSE_POLICY,
                "score_namespace": namespace,
                "rescue_run": namespace == "rescue",
                "source_phase": row["phase"],
                "source_scenario_id": row["scenario_id"],
                "checkpoint_id": row["checkpoint_id"],
                "setting": row["setting"],
                "seed": row["seed"],
                "trait": trait,
                "instrument_language": "English",
                "scores": scores,
                "items_parsed": sum(value is not None for value in ordered),
                "trait_score": complete_trait_score(ordered),
                "rater_raw_by_item": raw_outputs,
                "parse_failures": failures,
                "rater_decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                                     "max_new_tokens_per_item": 8, "seed": 20260911,
                                     "batch_size": 1, "cross_request_batching": False},
                "prompt_config_sha256": config_sha,
                "freeze_receipt_sha256": receipt_sha,
                "rater_runtime_manifest_sha256": runtime_sha,
                "source_shard_sha256": source_sha,
                "scorer_sha256": scorer_sha,
                "frozen_scorer_sha256": frozen_scorer_sha,
                "eligible_for_gate": phase["eligible_for_gate"],
                "eligible_for_analysis": phase["eligible_for_analysis"],
            })
    elapsed = clock() - started
    rate = measured_rate(calls, elapsed)
    for row in output_rows:
        row["scoring_elapsed_seconds"] = elapsed
        row["rater_calls"] = calls
        row["rater_calls_per_second"] = rate
    for existing in paths.values():
        if existing.exists():
            raise ValueError(f"refusing to overwrite a shard written during scoring: {existing}")
    publish_no_replace(out, output_rows)
    print(json.dumps({"output": str(out), "namespace": namespace, "records": len(output_rows),
                      "calls": calls, "item_parse_failures": failures_total,
                      "incomplete_traits": sum(row["trait_score"] is None for row in output_rows),
                      "elapsed_seconds": elapsed, "calls_per_second": rate}, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rater-path", required=True)
    ap.add_argument("--prompt-config", required=True)
    ap.add_argument("--freeze-receipt", required=True)
    ap.add_argument("--runtime-manifest", default="process/consumption/deployment/mistral-rater-runtime-v2-manifest.json")
    ap.add_argument("--source-checkpoint", required=True, choices=sorted(SOURCE_CHECKPOINTS))
    ap.add_argument("--source-phase", required=True, choices=sorted(SOURCE_PHASES))
    ap.add_argument("--namespace", required=True, choices=NAMESPACES)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--contract-smoke", action="store_true")
    return ap


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        raise SystemExit(
            contract_smoke(parsed, entrypoint=Path(__file__)) if parsed.contract_smoke else run(parsed)
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
