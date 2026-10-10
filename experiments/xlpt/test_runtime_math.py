#!/usr/bin/env python3
"""Independent numeric checks for the timing run's standard mathematics."""

import hashlib
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import torch
import vllm.model_executor.models.qwen3 as deployed_qwen3
import vllm.v1.sample.ops.topk_topp_sampler as deployed_top_p
import vllm.v1.sample.sampler as deployed_sampler

from runtime_math import (
    autoregressive_joint_probability,
    exponential_race_sample,
    nucleus_renormalize,
    temperature_softmax,
    timing_throughput,
)


ROOT = Path(__file__).resolve().parents[2]
CAPTURED = ROOT / "process/consumption/deployment/vendor_source"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_captured(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    pairs = (
        (inspect.getsourcefile(deployed_sampler), CAPTURED / "vllm/sampler.py"),
        (inspect.getsourcefile(deployed_top_p), CAPTURED / "vllm/topk_topp_sampler.py"),
        (inspect.getsourcefile(deployed_qwen3), CAPTURED / "vllm/qwen3.py"),
    )
    for installed, captured in pairs:
        assert digest(installed) == digest(captured), (installed, captured)

    captured_sampler = load_captured("captured_sampler", CAPTURED / "vllm/sampler.py")
    captured_top_p = load_captured("captured_top_p", CAPTURED / "vllm/topk_topp_sampler.py")
    captured_qwen3 = load_captured(
        "vllm.model_executor.models.qwen3_captured", CAPTURED / "vllm/qwen3.py"
    )

    conditional = np.array([[0.5, 0.25, 0.8], [1.0, 0.4, 0.2]])
    np.testing.assert_allclose(autoregressive_joint_probability(conditional), [0.1, 0.08])
    logits = np.array([[2.0, 1.0, -1.0]])
    expected = np.exp(logits / 0.5 - np.max(logits / 0.5, axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(temperature_softmax(logits, 0.5), expected)
    probability = np.array([[0.55, 0.25, 0.15, 0.05]])
    nucleus = nucleus_renormalize(probability, np.array([[True, True, False, False]]))
    np.testing.assert_allclose(nucleus, [[0.6875, 0.3125, 0.0, 0.0]])
    np.testing.assert_array_equal(exponential_race_sample(
        nucleus, np.array([[2.0, 0.25, 1.0, 1.0]])
    ), [1])
    assert timing_throughput(6, 2.0) == 3.0

    sampler = captured_sampler.Sampler()
    torch_logits = torch.tensor([[2.0, 1.0, -1.0]], dtype=torch.float32)
    scaled = sampler.apply_temperature(torch_logits.clone(), torch.tensor([0.5]), True)
    torch.testing.assert_close(scaled, torch_logits / 0.5)
    masked = captured_top_p.apply_top_k_top_p(
        torch.tensor([[2.0, 1.0, 0.0, -1.0]]), None, torch.tensor([0.8])
    )
    finite = torch.isfinite(masked)
    torch.testing.assert_close(finite, torch.tensor([[True, True, False, False]]))
    renormalized = masked.softmax(dim=-1)
    expected_retained = torch.tensor([[2.0, 1.0]]).softmax(dim=-1)
    torch.testing.assert_close(renormalized[:, :2], expected_retained)
    torch.testing.assert_close(renormalized[:, 2:], torch.zeros((1, 2)))

    forward_sampler = captured_top_p.TopKTopPSampler()
    forward_sample, _ = forward_sampler.forward_native(
        torch.tensor([[2.0, 1.0, 0.0, -1.0]]),
        {0: torch.Generator().manual_seed(17091701)}, None, torch.tensor([0.8])
    )
    assert int(forward_sample[0]) in (0, 1)

    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    def draws(seed, count=64):
        generator = torch.Generator().manual_seed(seed)
        return [int(captured_top_p.random_sample(probs.clone(), {0: generator})[0]) for _ in range(count)]
    assert draws(17091701) == draws(17091701)
    assert draws(17091701) != draws(17091702)
    empirical = np.bincount(draws(17091703, 4096), minlength=4) / 4096
    np.testing.assert_allclose(empirical, [0.4, 0.3, 0.2, 0.1], atol=0.04)
    class FakeQwen:
        lm_head = None

        @staticmethod
        def logits_processor(lm_head, hidden_states):
            return hidden_states

    hidden = torch.tensor([[0.1, 0.2, 0.3]])
    torch.testing.assert_close(captured_qwen3.Qwen3ForCausalLM.compute_logits(FakeQwen(), hidden), hidden)
    print("PASS: captured vLLM/model bytes, inference math, throughput, and seeded sampling invariants")


if __name__ == "__main__":
    main()
