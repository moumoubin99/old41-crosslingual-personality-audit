#!/usr/bin/env python3
"""Executable contract for the deployment-unit runner.

The runner is the argv the deployment review sees, so its guards are part of
what must be reviewed: the partition fail-closure that keeps this apparatus off
qualification and evaluation, the sealed-seed refusal, project-relative path
containment, the refusal to overwrite an existing receipt, and the fact that the
receipt it writes is marked ineligible for analysis.

Every expected value here is computed in this file, not read back from the
runner's own output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

import run_qwen_reference_probe as probe
from qwen3_reference import SEALED_SEED_RANGES, check_seed

SEED = 17091701


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return path


def row(record_id, partition, language="en", text="a customer writes in"):
    return {"record_id": record_id, "partition": partition, "language": language,
            "surface": "scenario", "text": text}


def check_digest_is_the_file_digest():
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "payload.bin"
        target.write_bytes(b"xlpt")
        expected = hashlib.sha256(b"xlpt").hexdigest()
        if probe.sha256_path(target) != expected:
            raise AssertionError("sha256_path does not reproduce an independent digest")
        target.write_bytes(b"xlpu")
        if probe.sha256_path(target) == expected:
            raise AssertionError("sha256_path returned the same digest for different bytes")


def check_path_containment():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "data").mkdir()
        inside = write_jsonl(root / "data" / "rows.jsonl", [row("a", "tuning")])
        if probe.resolve_inside(root, "data/rows.jsonl") != inside:
            raise AssertionError("a project-relative path was not resolved to itself")
        for escape in ("../outside.jsonl", "data/../../outside.jsonl"):
            try:
                probe.resolve_inside(root, escape)
            except ValueError:
                continue
            raise AssertionError(f"resolve_inside accepted {escape}")
        try:
            probe.resolve_inside(root, "data/missing.jsonl")
        except ValueError:
            pass
        else:
            raise AssertionError("resolve_inside accepted a path that does not exist")


def check_partition_fail_closure():
    """Only English TUNING rows are ever selected, and the sealed splits refuse."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mixed = write_jsonl(root / "mixed.jsonl", [
            row("xlpt-b", "tuning"),
            row("xlpt-a", "tuning"),
            row("xlpt-c", "qualification"),
            row("xlpt-d", "evaluation"),
            row("xlpt-e", "tuning", language="zh"),
        ])
        selected, excluded = probe.tuning_rows(mixed, limit=0)
        if [item["record_id"] for item in selected] != ["xlpt-a", "xlpt-b"]:
            raise AssertionError(
                f"selection is not the sorted English tuning rows: {selected}")
        if excluded != ["evaluation", "qualification"]:
            raise AssertionError(f"excluded partitions recorded as {excluded}")
        if probe.tuning_rows(mixed, limit=1)[0][0]["record_id"] != "xlpt-a":
            raise AssertionError("--max-scenarios did not truncate deterministically")

        sealed_only = write_jsonl(root / "sealed.jsonl", [
            row("xlpt-c", "qualification"), row("xlpt-d", "evaluation")])
        try:
            probe.tuning_rows(sealed_only, limit=0)
        except ValueError as error:
            if "tuning" not in str(error):
                raise AssertionError(f"refusal did not name the tuning partition: {error}")
        else:
            raise AssertionError("a qualification/evaluation-only corpus was accepted")

        empty = write_jsonl(root / "empty.jsonl", [])
        try:
            probe.tuning_rows(empty, limit=0)
        except ValueError:
            pass
        else:
            raise AssertionError("an empty corpus was accepted")


def check_sealed_seeds_are_refused():
    """The apparatus seed and every per-scenario sub-stream stay off sealed blocks."""
    check_seed(SEED)
    for reserved in SEALED_SEED_RANGES:
        if SEED in reserved:
            raise AssertionError(f"the default apparatus seed lies in {reserved}")
        for sealed in (reserved.start, reserved.stop - 1):
            try:
                check_seed(sealed)
            except ValueError:
                continue
            raise AssertionError(f"check_seed accepted sealed seed {sealed}")
    # The sub-stream the generation unit derives per scenario is checked too.
    for index in range(180):
        check_seed(SEED + index)


def check_device_selection():
    if probe.select_device("cpu").type != "cpu":
        raise AssertionError("--device cpu did not select the CPU")
    auto = probe.select_device("auto")
    if auto.type != ("cuda" if torch.cuda.is_available() else "cpu"):
        raise AssertionError(f"auto selected {auto} against torch.cuda.is_available()")
    if not torch.cuda.is_available():
        try:
            probe.select_device("cuda")
        except RuntimeError:
            pass
        else:
            raise AssertionError("--device cuda succeeded with no CUDA device visible")
    record = probe.device_record(torch.device("cpu"))
    if record["device"] != "cpu" or "cuda_available" not in record:
        raise AssertionError(f"device_record for CPU is {record}")


def check_unit_table():
    """Every declared unit is dispatchable and every contract module exists here."""
    if set(probe.UNITS) != set(probe.CONTRACT_UNITS) | {"generation"}:
        raise AssertionError(f"unit table {probe.UNITS} does not match the contract table")
    here = Path(probe.__file__).resolve().parent
    for unit, module in probe.CONTRACT_UNITS.items():
        source = here / f"{module}.py"
        if not source.is_file():
            raise AssertionError(f"unit {unit} names missing contract module {module}")


def run_unit(root, unit, output="receipts/{unit}-{seed}.json", seed=SEED):
    args = argparse.Namespace(
        unit=unit, project=str(root), output_json=output, seed=seed, device="cpu",
        model_path="/nonexistent/model", scenarios="scenarios.jsonl",
        max_scenarios=1, max_prompt_tokens=8, max_new_tokens=1)
    return probe.main(args)


def check_receipt_is_written_once_and_marked_ineligible():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        if run_unit(root, "kernels") != 0:
            raise AssertionError("the kernels unit did not return 0")
        receipt = root / "receipts" / f"kernels-{SEED}.json"
        record = json.loads(receipt.read_text(encoding="utf-8"))

        if record["eligible_for_analysis"] is not False:
            raise AssertionError("the receipt does not mark itself ineligible for analysis")
        if record["trains_or_updates_weights"] is not False:
            raise AssertionError("the receipt does not record that nothing is trained")
        if record["unit"] != "kernels" or record["kind"] != "contract":
            raise AssertionError(f"receipt identifies unit {record['unit']}/{record['kind']}")
        if record["result"] != "PASS":
            raise AssertionError(f"receipt result {record['result']}")
        expected = hashlib.sha256(
            (Path(probe.__file__).resolve().parent / "test_qwen3_kernels.py").read_bytes()
        ).hexdigest()
        if record["test_file_sha256"] != expected:
            raise AssertionError("the receipt pins a digest that is not the test file's")
        if record["reference_module_sha256"] != probe.sha256_path(
                Path(probe.__file__).resolve().parent / "qwen3_reference.py"):
            raise AssertionError("the receipt does not pin the captured reference module")

        # An existing receipt is never silently replaced, and no partial file is left.
        try:
            run_unit(root, "kernels")
        except ValueError as error:
            if "refusing to overwrite" not in str(error):
                raise AssertionError(f"unexpected refusal: {error}")
        else:
            raise AssertionError("the runner overwrote an existing receipt")
        leftovers = sorted(p.name for p in (root / "receipts").iterdir()
                           if ".tmp-" in p.name)
        if leftovers:
            raise AssertionError(f"temporary receipt files were left behind: {leftovers}")


def check_runner_guards():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        try:
            run_unit(root, "kernels", output="../escape.json")
        except ValueError:
            pass
        else:
            raise AssertionError("the runner wrote a receipt outside the project")
        try:
            run_unit(root, "no-such-unit")
        except ValueError:
            pass
        else:
            raise AssertionError("the runner accepted an unknown unit")
        try:
            run_unit(root / "absent", "kernels")
        except ValueError:
            pass
        else:
            raise AssertionError("the runner accepted a --project that is not a directory")
        # A sealed seed is refused before any generation work begins.
        sealed = SEALED_SEED_RANGES[0].start
        try:
            run_unit(root, "generation", seed=sealed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"the generation unit accepted sealed seed {sealed}")


def main():
    check_digest_is_the_file_digest()
    check_path_containment()
    check_partition_fail_closure()
    check_sealed_seeds_are_refused()
    check_device_selection()
    check_unit_table()
    check_receipt_is_written_once_and_marked_ineligible()
    check_runner_guards()
    print("PASS: runner path containment, tuning-only partition fail-closure, sealed-seed "
          "refusal on both the base seed and every scenario sub-stream, device selection, "
          "the unit dispatch table, and a single atomic receipt marked ineligible for analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
