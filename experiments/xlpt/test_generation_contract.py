#!/usr/bin/env python3
"""Boundary tests for the timing-only generation path; does not load a model."""

from run_generation import validate_execution_boundary


def must_fail(mode, partition, seed):
    try:
        validate_execution_boundary(mode, partition, seed)
    except ValueError:
        return
    raise AssertionError((mode, partition, seed))


def main():
    validate_execution_boundary("timing", "tuning", 0)
    validate_execution_boundary("timing", "tuning", 17091701)
    validate_execution_boundary("timing", "tuning", 17091702)
    must_fail("timing", "qualification", 17091701)
    must_fail("timing", "evaluation", 17091701)
    must_fail("timing", "tuning", 20260811)
    must_fail("timing", "tuning", 20260834)
    must_fail("timing", "tuning", 20260911)
    must_fail("timing", "tuning", 20260934)
    must_fail("timing", "tuning", -1)
    must_fail("qualification", "qualification", 20260811)
    print("PASS: timing boundary permits seed reuse but protects sealed partitions and seed blocks")


if __name__ == "__main__":
    main()
