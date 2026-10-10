"""Executable contract for the frozen rater prompt date and the source registry.

Two properties are checked here, both CPU-only and model-free.

1. ``freeze_template_date`` removes the checkpoint template's wall-clock date call.
   The production scorer sends a user-only request, which is exactly the branch that
   renders the checkpoint's default system message with ``strftime_now("%Y-%m-%d")``
   inside it.  Scoring a second checkpoint on a later day would otherwise change every
   prompt by one line and silently break comparability with the Qwen shards, which were
   all scored on the host's 2026-09-18.  The freeze must also refuse to no-op: if the
   date call is absent, or present more than once, the drift would return unannounced.

2. ``SOURCE_CHECKPOINTS`` binds each scorable B0 arm to its own generation contract --
   checkpoint id, model path, runtime-manifest path, the row field carrying that
   manifest's digest, and the attention implementation the arm was generated with.  The
   Qwen arm came off the vLLM stack (``eager``) and the aya/llama/gemma arms off the HF
   stack (``hf-eager``); the registry must keep those apart rather than assume one value.

Run: ``python3 -m experiments.xlpt.test_rater_prompt_date_freeze`` from the project root.
"""

from __future__ import annotations

import json
from pathlib import Path

from .run_rater_scoring import (
    RATER_PROMPT_DATE,
    RATER_TEMPLATE_DATE_CALL,
    SOURCE_CHECKPOINTS,
    freeze_template_date,
)

ROOT = Path(__file__).resolve().parents[2]

REAL_DEFAULT_MESSAGE = (
    "{%- set default_system_message = 'You are Mistral Small 3. "
    "The current date is ' + " + RATER_TEMPLATE_DATE_CALL + " + '.' %}"
)


def expect_error(callable_, *args, reason):
    try:
        callable_(*args)
    except ValueError:
        return
    raise AssertionError(f"freeze_template_date accepted {reason}")


frozen = freeze_template_date(REAL_DEFAULT_MESSAGE)
if RATER_TEMPLATE_DATE_CALL in frozen:
    raise AssertionError("the wall-clock date call survived the freeze")
if json.dumps(RATER_PROMPT_DATE) not in frozen:
    raise AssertionError("the frozen date literal was not substituted into the template")
if frozen != REAL_DEFAULT_MESSAGE.replace(
    RATER_TEMPLATE_DATE_CALL, json.dumps(RATER_PROMPT_DATE)
):
    raise AssertionError("the freeze changed template bytes beyond the date call")

# The substitution is idempotent only in the sense that a frozen template no longer
# contains the call; running the freeze twice must fail loudly rather than pass silently.
expect_error(freeze_template_date, frozen, reason="an already-frozen template")
expect_error(freeze_template_date, "no date call here", reason="a template with no date call")
expect_error(
    freeze_template_date,
    REAL_DEFAULT_MESSAGE + REAL_DEFAULT_MESSAGE,
    reason="a template with two date calls",
)
expect_error(
    freeze_template_date,
    REAL_DEFAULT_MESSAGE,
    "18-09-2026",
    reason="a malformed frozen date",
)

if RATER_PROMPT_DATE != "2026-09-18":
    raise AssertionError(
        "the frozen prompt date must stay the day the Qwen rater shards were produced"
    )

# Registry contract.
if set(SOURCE_CHECKPOINTS) != {
    "qwen3-8b",
    "aya-expanse-8b",
    "Llama-3.1-8B-Instruct",
    "gemma-3-12b-it",
}:
    raise AssertionError("the scorable B0 arms changed without a contract update")

required = {
    "checkpoint_id",
    "model_path",
    "runtime_manifest",
    "runtime_sha_field",
    "attention",
}
seen_ids = set()
seen_paths = set()
for slug, contract in SOURCE_CHECKPOINTS.items():
    missing = required - set(contract)
    if missing:
        raise AssertionError(f"{slug} is missing registry fields {sorted(missing)}")
    if contract["checkpoint_id"] in seen_ids:
        raise AssertionError(f"{slug} reuses another arm's checkpoint id")
    if contract["model_path"] in seen_paths:
        raise AssertionError(f"{slug} reuses another arm's model path")
    seen_ids.add(contract["checkpoint_id"])
    seen_paths.add(contract["model_path"])
    if contract["attention"] not in ("eager", "hf-eager"):
        raise AssertionError(f"{slug} declares an unregistered attention implementation")
    if contract["runtime_sha_field"] not in (
        "qwen_runtime_manifest_sha256",
        "hf_runtime_manifest_sha256",
    ):
        raise AssertionError(f"{slug} reads an unregistered runtime digest field")

if SOURCE_CHECKPOINTS["qwen3-8b"]["attention"] != "eager":
    raise AssertionError("the Qwen arm was generated on the vLLM stack (eager)")
for slug in ("aya-expanse-8b", "Llama-3.1-8B-Instruct", "gemma-3-12b-it"):
    if SOURCE_CHECKPOINTS[slug]["attention"] != "hf-eager":
        raise AssertionError(f"{slug} was generated on the HF stack (hf-eager)")

# The generated shards on disk must actually agree with the registry, where present.
for slug, contract in SOURCE_CHECKPOINTS.items():
    shard = ROOT / f"process/results/xlpt/b0-competence/{slug}/seed-20260811.jsonl"
    if not shard.exists():
        continue
    row = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    if row.get("checkpoint_id") != contract["checkpoint_id"]:
        raise AssertionError(f"{slug} shards disagree with the registered checkpoint id")
    if row.get("model_path") != contract["model_path"]:
        raise AssertionError(f"{slug} shards disagree with the registered model path")
    if row.get("decoding", {}).get("attention") != contract["attention"]:
        raise AssertionError(f"{slug} shards disagree with the registered attention")
    if not isinstance(row.get(contract["runtime_sha_field"]), str):
        raise AssertionError(f"{slug} shards carry no {contract['runtime_sha_field']}")

print("PASS: rater prompt date frozen to " + RATER_PROMPT_DATE + " and source registry bound")
