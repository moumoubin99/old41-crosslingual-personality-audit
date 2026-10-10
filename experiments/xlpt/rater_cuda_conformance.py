#!/usr/bin/env python3
"""Real-CUDA numerical witness for the pinned Mistral/AWQ rater runtime.

This is intentionally separate from the CUDA-hidden deployment probe.  It runs
only after the generation queue releases the worker GPU, exercises the actual
checkpoint-compatible AWQ kernels, CUDA RMSNorm/RoPE/SwiGLU operators, and the
vLLM paged FlashAttention prefill plus two cached-decode steps, and compares
them with independent PyTorch oracles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_contract_sha256(path: Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload.pop("cuda_witness", None)
    payload.pop("cuda_expected", None)
    return canonical_sha256(payload)


def max_error(actual, expected) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def independent_awq_dequantize(qweight, scales, qzeros, group_size: int):
    """Literal AWQ unpack/dequant oracle; independent of project constants."""
    import torch

    shifts = torch.tensor((0, 4, 1, 5, 2, 6, 3, 7), dtype=torch.int64) * 4
    weight = ((qweight.to(torch.int64).unsqueeze(-1) >> shifts) & 15).reshape(qweight.shape[0], -1)
    zeros = ((qzeros.to(torch.int64).unsqueeze(-1) >> shifts) & 15).reshape(qzeros.shape[0], -1)
    group = torch.arange(weight.shape[0], dtype=torch.int64) // group_size
    return (weight - zeros[group]).to(scales.dtype) * scales[group]


def awq_contract(model_dir: Path) -> dict:
    import torch
    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.awq import AWQConfig, AWQLinearMethod

    shard = model_dir / "model-00001-of-00004.safetensors"
    stem = "language_model.model.layers.0.self_attn.q_proj"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        qweight = handle.get_tensor(stem + ".qweight")[:, :16].contiguous()
        qzeros = handle.get_tensor(stem + ".qzeros")[:, :16].contiguous()
        scales = handle.get_tensor(stem + ".scales")[:, :128].contiguous()
    dequantized = independent_awq_dequantize(qweight, scales, qzeros, 128).cuda()
    layer = SimpleNamespace(
        qweight=qweight.cuda(), qzeros=qzeros.cuda(), scales=scales.cuda()
    )
    method = AWQLinearMethod(AWQConfig(4, 128, False))

    base = torch.linspace(-0.75, 0.875, qweight.shape[0], dtype=torch.float16, device="cuda")
    small = torch.stack((base, base.flip(0)))
    small_actual = method.apply(layer, small)
    small_expected = torch.matmul(small, dequantized)
    small_error = max_error(small_actual, small_expected)

    row_offsets = torch.arange(256, dtype=torch.float16, device="cuda")[:, None] / 2048.0
    large = base[None, :] + row_offsets
    large_actual = method.apply(layer, large)
    large_expected = torch.matmul(large, dequantized)
    large_error = max_error(large_actual, large_expected)
    if not torch.allclose(small_actual, small_expected, atol=0.2, rtol=0.02):
        raise AssertionError(f"real fused AWQ GEMM diverged from oracle: {small_error}")
    if not torch.allclose(large_actual, large_expected, atol=0.2, rtol=0.02):
        raise AssertionError(f"real AWQ dequantize/matmul diverged from oracle: {large_error}")
    return {
        "checkpoint_shard": shard.name,
        "checkpoint_shard_sha256": sha256(shard),
        "tensor_prefix": stem,
        "sliced_shapes": {
            "qweight": list(qweight.shape), "qzeros": list(qzeros.shape), "scales": list(scales.shape),
        },
        "group_size": 128,
        "fused_gemm_max_abs_error": small_error,
        "dequantize_matmul_max_abs_error": large_error,
    }


def native_cuda_contract(config: dict) -> dict:
    import torch
    import torch.nn.functional as functional
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    text = config["text_config"]
    hidden_size = int(text["hidden_size"])
    intermediate_size = int(text["intermediate_size"])
    head_dim = int(text["head_dim"])
    query_heads = int(text["num_attention_heads"])
    key_value_heads = int(text["num_key_value_heads"])
    epsilon = float(text["rms_norm_eps"])
    theta = float(text["rope_theta"])
    maximum_positions = int(text["max_position_embeddings"])

    hidden = torch.linspace(-2.0, 2.0, hidden_size, dtype=torch.float16, device="cuda")[None]
    residual = torch.linspace(1.5, -1.5, hidden_size, dtype=torch.float16, device="cuda")[None]
    norm = RMSNorm(hidden_size, eps=epsilon, dtype=torch.float16).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, hidden_size, device="cuda"))
    actual_norm, actual_residual = norm.forward_cuda(hidden.clone(), residual.clone())
    expected_residual = (hidden.float() + residual.float()).to(torch.float16)
    expected_norm = expected_residual.float()
    expected_norm = expected_norm * torch.rsqrt(expected_norm.pow(2).mean(-1, keepdim=True) + epsilon)
    expected_norm = expected_norm.to(torch.float16) * norm.weight
    norm_error = max_error(actual_norm, expected_norm)
    residual_error = max_error(actual_residual, expected_residual)
    if norm_error > 0.01 or residual_error != 0.0:
        raise AssertionError("real fused RMSNorm residual contract diverged")

    activation_input = torch.linspace(
        -3.0, 3.0, 2 * intermediate_size, dtype=torch.float16, device="cuda"
    )[None]
    actual_activation = SiluAndMul().forward_cuda(activation_input)
    expected_activation = (functional.silu(activation_input[..., :intermediate_size].float())
                           * activation_input[..., intermediate_size:].float())
    activation_error = max_error(actual_activation, expected_activation)
    if activation_error > 0.01:
        raise AssertionError("real CUDA gated-SiLU contract diverged")

    rotary = RotaryEmbedding(
        head_dim, head_dim, maximum_positions, theta, True, torch.float16
    ).cuda()
    positions = torch.tensor([0, 3], dtype=torch.long, device="cuda")
    query = torch.linspace(
        -1.0, 1.0, 2 * query_heads * head_dim, dtype=torch.float16, device="cuda"
    ).reshape(2, query_heads * head_dim)
    key = torch.linspace(
        1.0, -1.0, 2 * key_value_heads * head_dim, dtype=torch.float16, device="cuda"
    ).reshape(2, key_value_heads * head_dim)

    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda") / head_dim)
    )
    frequencies = torch.outer(positions.float(), inverse_frequency)
    cosine = torch.cat((frequencies, frequencies), dim=-1).cos()[:, None, :]
    sine = torch.cat((frequencies, frequencies), dim=-1).sin()[:, None, :]

    def independent_rotate(value, heads):
        shaped = value.float().reshape(2, heads, head_dim)
        half = head_dim // 2
        rotated = torch.cat((-shaped[..., half:], shaped[..., :half]), dim=-1)
        return (shaped * cosine + rotated * sine).to(value.dtype).reshape_as(value)

    expected_query = independent_rotate(query, query_heads)
    expected_key = independent_rotate(key, key_value_heads)
    actual_query, actual_key = rotary.forward_cuda(positions, query.clone(), key.clone())
    rope_error = max(max_error(actual_query, expected_query), max_error(actual_key, expected_key))
    if rope_error > 0.01:
        raise AssertionError("real CUDA RoPE contract diverged")
    return {
        "fused_rms_norm_max_abs_error": norm_error,
        "fused_residual_max_abs_error": residual_error,
        "gated_silu_max_abs_error": activation_error,
        "rope_max_abs_error": rope_error,
        "fixture": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "head_dim": head_dim,
            "num_attention_heads": query_heads,
            "num_key_value_heads": key_value_heads,
            "rms_norm_eps": epsilon,
            "rope_theta": theta,
            "max_position_embeddings": maximum_positions,
        },
    }


def attention_oracle(query, key, value):
    import torch

    repeated_key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    repeated_value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    scores = torch.einsum("thd,shd->hts", query.float(), repeated_key.float()) / math.sqrt(query.shape[-1])
    q_len, kv_len = query.shape[0], key.shape[0]
    context = kv_len - q_len
    allowed = torch.arange(kv_len, device=query.device)[None, :] <= (
        context + torch.arange(q_len, device=query.device)[:, None]
    )
    scores = scores.masked_fill(~allowed[None, :, :], -torch.inf)
    probability = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", probability, repeated_value.float()).to(query.dtype)


def flash_attention_contract(config: dict) -> dict:
    import torch
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata

    text = config["text_config"]
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    width, block_size = int(text["head_dim"]), 16
    implementation = FlashAttentionImpl(
        num_heads=heads, head_size=width, scale=1.0 / math.sqrt(width),
        num_kv_heads=kv_heads, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="auto",
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0, device="cuda"),
        _k_scale=torch.tensor(1.0, device="cuda"),
        _v_scale=torch.tensor(1.0, device="cuda"),
    )
    cache = torch.zeros((2, 1, block_size, kv_heads, width), dtype=torch.float16, device="cuda")
    all_key, all_value, errors = [], [], []

    def execute(query, key, value, slot, sequence_length, query_start):
        metadata = FlashAttentionMetadata(
            num_actual_tokens=query.shape[0], max_query_len=query.shape[0],
            query_start_loc=torch.tensor(query_start, dtype=torch.int32, device="cuda"),
            max_seq_len=sequence_length,
            seq_lens=torch.tensor([sequence_length], dtype=torch.int32, device="cuda"),
            block_table=torch.tensor([[0]], dtype=torch.int32, device="cuda"),
            slot_mapping=torch.tensor(slot, dtype=torch.int64, device="cuda"),
            use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
            prefix_kv_lens=None, suffix_kv_lens=None, causal=True,
        )
        output = torch.empty_like(query)
        implementation.forward(layer, query, key, value, cache, metadata, output)
        all_key.extend(key.unbind(0)); all_value.extend(value.unbind(0))
        expected = attention_oracle(
            query, torch.stack(all_key), torch.stack(all_value)
        )
        error = max_error(output, expected)
        if error > 0.03:
            raise AssertionError(f"real paged FlashAttention diverged: {error}")
        errors.append(error)

    torch.manual_seed(17091821)
    prefill_query = torch.randn((2, heads, width), dtype=torch.float16, device="cuda") / 4
    prefill_key = torch.randn((2, kv_heads, width), dtype=torch.float16, device="cuda") / 4
    prefill_value = torch.randn((2, kv_heads, width), dtype=torch.float16, device="cuda") / 4
    execute(prefill_query, prefill_key, prefill_value, [0, 1], 2, [0, 2])
    for slot in (2, 3):
        query = torch.randn((1, heads, width), dtype=torch.float16, device="cuda") / 4
        key = torch.randn((1, kv_heads, width), dtype=torch.float16, device="cuda") / 4
        value = torch.randn((1, kv_heads, width), dtype=torch.float16, device="cuda") / 4
        execute(query, key, value, [slot], slot + 1, [0, 1])
    cache_error = max(
        max_error(cache[0, 0, :4], torch.stack(all_key)),
        max_error(cache[1, 0, :4], torch.stack(all_value)),
    )
    if cache_error != 0.0:
        raise AssertionError("real paged KV cache contents changed")
    return {
        "prefill_and_decode_max_abs_errors": errors,
        "kv_cache_max_abs_error": cache_error,
        "decode_steps": 2,
        "gqa_groups": heads // kv_heads,
        "selected_fa_version": implementation.vllm_flash_attn_version,
        "fixture": {
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": width,
            "block_size": block_size,
        },
    }


def run(project: Path, model_dir: Path, runtime_manifest: Path, output: Path) -> dict:
    manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    for name in manifest.get("required_unset_environment", []):
        if os.environ.get(name) not in (None, ""):
            raise ValueError(f"CUDA witness backend override must be unset: {name}")
    import torch
    import vllm
    from vllm import _C
    from vllm.vllm_flash_attn import _vllm_fa2_C, _vllm_fa3_C
    from experiments.xlpt.rater_production_closure import (
        validate_production_dependency_closure,
    )
    from experiments.xlpt.rater_runtime import validate_runtime_sources

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA conformance requires the released worker GPU")
    if Path.cwd().resolve() != project.resolve():
        raise ValueError("CUDA conformance must execute from the deployment root")
    runtime_sources = validate_runtime_sources(runtime_manifest, project)
    production_dependencies = validate_production_dependency_closure(runtime_manifest)
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is None:
        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).splitlines()
        device_uuid = uuids[device].strip()
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["model_files"].get("config.json"):
        raise ValueError("checkpoint config is not bound by the runtime manifest")
    provenance_path = project / manifest["build_provenance"]["path"]
    result = {
        "version": 1,
        "status": "PASS",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cuda": {
            "device_name": properties.name,
            "device_uuid": str(device_uuid),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "packages": {"vllm": vllm.__version__},
        "dependency_binding": {
            "runtime_source_count": len(runtime_sources),
            "runtime_sources_sha256": canonical_sha256(list(runtime_sources)),
            "production_dependency_count": len(production_dependencies),
            "production_dependencies_sha256": canonical_sha256(
                list(production_dependencies)
            ),
        },
        "artifacts": {
            "conformance_source_sha256": sha256(Path(__file__)),
            "runtime_contract_sha256": runtime_contract_sha256(runtime_manifest),
            "checkpoint_config_sha256": sha256(config_path),
            "build_provenance_sha256": sha256(provenance_path),
        },
        "binaries": {
            "vllm._C": sha256(Path(_C.__file__)),
            "vllm.flash_attn.fa2": sha256(Path(_vllm_fa2_C.__file__)),
            "vllm.flash_attn.fa3": sha256(Path(_vllm_fa3_C.__file__)),
        },
        "awq": awq_contract(model_dir),
        "native_cuda": native_cuda_contract(config),
        "flash_attention": flash_attention_contract(config),
    }
    output = (project / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(project.resolve()):
        raise ValueError("CUDA witness must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--project", type=Path, required=True)
    argument_parser.add_argument("--model-dir", type=Path, required=True)
    argument_parser.add_argument(
        "--runtime-manifest", type=Path,
        default=Path("process/consumption/deployment/mistral-rater-runtime-v2-manifest.json"),
    )
    argument_parser.add_argument(
        "--output", type=Path,
        default=Path("process/consumption/deployment/mistral-rater-cuda-witness-v1.json"),
    )
    return argument_parser


if __name__ == "__main__":
    arguments = parser().parse_args()
    runtime_manifest = (arguments.project / arguments.runtime_manifest).resolve()
    report = run(
        arguments.project.resolve(), arguments.model_dir.resolve(), runtime_manifest, arguments.output
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
