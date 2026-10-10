#!/usr/bin/env python3
"""Auditable Mistral/AWQ inference contracts used by the English rater.

The production forward remains vLLM 0.11.0's frozen Mistral3/AWQ runtime.  Its
formula-bearing Python sources are captured under ``vendor/vllm_0_11`` and
their hashes are checked before model construction.  Final token selection is
owned by this project: the vLLM worker loads ``ProjectArgmaxLogitsProcessor``
and forces every row to the token returned by ``greedy_token``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path


AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def measured_rate(completed_calls: int, elapsed_seconds: float) -> float:
    """Strict calls-per-second contract for immutable rater outputs."""
    if (isinstance(completed_calls, bool) or not isinstance(completed_calls, int)
            or completed_calls < 0):
        raise ValueError("completed_calls must be a nonnegative integer")
    if (isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds)) or elapsed_seconds <= 0):
        raise ValueError("elapsed_seconds must be a finite positive number")
    return float(completed_calls) / float(elapsed_seconds)


def unpack_awq_int4(packed):
    """Unpack vLLM/AutoAWQ int32 columns in their registered interleave."""
    import torch

    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError("packed AWQ values must be a rank-two int32 tensor")
    shifts = torch.tensor(AWQ_UNPACK_ORDER, dtype=torch.int64, device=packed.device) * 4
    unpacked = (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF
    return unpacked.reshape(packed.shape[0], packed.shape[1] * 8)


def awq_dequantize(qweight, scales, qzeros, group_size: int):
    """Reference W=(Q-Z)*S for the frozen 4-bit AWQ GEMM dependency."""
    import torch

    if group_size <= 0 or qweight.shape[0] % group_size:
        raise ValueError("AWQ group size must positively divide the input dimension")
    weight = unpack_awq_int4(qweight)
    zeros = unpack_awq_int4(qzeros)
    groups = qweight.shape[0] // group_size
    if zeros.shape != (groups, weight.shape[1]) or scales.shape != zeros.shape:
        raise ValueError("AWQ scales and zero points do not cover every group/output")
    group_index = torch.arange(qweight.shape[0], device=qweight.device) // group_size
    return (weight - zeros[group_index]).to(scales.dtype) * scales[group_index]


def awq_linear(hidden, qweight, scales, qzeros, group_size: int):
    """Independent dense meaning of either deployed int4-AWQ kernel branch."""
    return hidden @ awq_dequantize(qweight, scales, qzeros, group_size)


def embedding_lookup(token_ids, embedding_weight):
    import torch.nn.functional as functional

    return functional.embedding(token_ids, embedding_weight)


def split_fused_qkv(fused, query_heads: int, key_value_heads: int, head_dim: int):
    """Split the deployed fused QKV width and expose canonical head axes."""
    for name, value in (
        ("query_heads", query_heads),
        ("key_value_heads", key_value_heads),
        ("head_dim", head_dim),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if fused.ndim != 3:
        raise ValueError("fused QKV projection must have [batch, sequence, width]")
    query_width = query_heads * head_dim
    key_value_width = key_value_heads * head_dim
    expected_width = query_width + 2 * key_value_width
    if fused.shape[-1] != expected_width:
        raise ValueError("fused QKV width does not match the checkpoint head layout")
    query, key, value = fused.split(
        (query_width, key_value_width, key_value_width), dim=-1
    )

    def expose_heads(tensor, heads):
        batch, sequence, _ = tensor.shape
        return tensor.reshape(batch, sequence, heads, head_dim).transpose(1, 2)

    return (
        expose_heads(query, query_heads),
        expose_heads(key, key_value_heads),
        expose_heads(value, key_value_heads),
    )


def build_rope_tables(absolute_positions, head_dim: int, theta: float):
    """Construct the float32 NeoX cosine/sine tables used at absolute positions."""
    import torch

    if (absolute_positions.ndim != 1
            or absolute_positions.dtype not in (torch.int32, torch.int64)
            or bool(torch.any(absolute_positions < 0))):
        raise ValueError("absolute_positions must be a nonnegative integer vector")
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0 or head_dim % 2:
        raise ValueError("rotary head_dim must be a positive even integer")
    if (isinstance(theta, bool) or not isinstance(theta, (int, float))
            or not math.isfinite(float(theta)) or theta <= 0):
        raise ValueError("rotary theta must be finite and positive")
    frequency_index = torch.arange(
        0, head_dim, 2, dtype=torch.float32, device=absolute_positions.device
    )
    inverse_frequency = torch.exp(
        -frequency_index * math.log(float(theta)) / float(head_dim)
    )
    phase = torch.outer(absolute_positions.to(torch.float32), inverse_frequency)
    cosine_half, sine_half = torch.cos(phase), torch.sin(phase)
    return (
        torch.cat((cosine_half, cosine_half), dim=-1),
        torch.cat((sine_half, sine_half), dim=-1),
    )


def absolute_cache_positions(past_length: int, current_length: int):
    """Return the absolute positions assigned to the current cache append."""
    import torch

    for name, value in (("past_length", past_length), ("current_length", current_length)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if past_length < 0 or current_length <= 0:
        raise ValueError("cache lengths require past>=0 and current>0")
    return torch.arange(past_length, past_length + current_length, dtype=torch.int64)


def append_key_value_cache(past_key, past_value, current_key, current_value):
    """Append key/value content on the sequence axis without changing other axes."""
    import torch

    if (past_key.shape != past_value.shape or current_key.shape != current_value.shape
            or past_key.ndim != 4 or current_key.ndim != 4
            or past_key.shape[:2] + past_key.shape[3:] != current_key.shape[:2] + current_key.shape[3:]):
        raise ValueError("past/current key-value caches have incompatible axes")
    return (
        torch.cat((past_key, current_key), dim=-2),
        torch.cat((past_value, current_value), dim=-2),
    )


def write_cache_slots(prior_cache, current_cache, keep_mask, append_selector):
    """Independent dense meaning of absolute-position paged-cache writes."""
    import torch

    if (prior_cache.ndim != 4 or current_cache.ndim != 4
            or keep_mask.ndim != 1 or append_selector.ndim != 2
            or prior_cache.shape[:2] + prior_cache.shape[3:]
            != current_cache.shape[:2] + current_cache.shape[3:]
            or keep_mask.shape[0] != prior_cache.shape[-2]
            or append_selector.shape != (prior_cache.shape[-2], current_cache.shape[-2])):
        raise ValueError("cache write tensors have incompatible axes")
    retained = prior_cache * keep_mask.to(prior_cache.dtype)[None, None, :, None]
    written = torch.einsum(
        "tc,bhcd->bhtd", append_selector.to(current_cache.dtype), current_cache
    )
    return retained + written


def cache_aware_causal_mask(query_positions, key_positions, dtype=None):
    """Build the additive mask from absolute query/key cache positions."""
    import torch

    if (query_positions.ndim != 1 or key_positions.ndim != 1
            or query_positions.dtype not in (torch.int32, torch.int64)
            or key_positions.dtype not in (torch.int32, torch.int64)
            or bool(torch.any(query_positions < 0)) or bool(torch.any(key_positions < 0))):
        raise ValueError("causal-mask positions must be nonnegative integer vectors")
    mask_dtype = dtype or torch.float32
    visible = key_positions[None, :] <= query_positions[:, None]
    return torch.where(
        visible,
        torch.zeros((), dtype=mask_dtype, device=query_positions.device),
        torch.full((), torch.finfo(mask_dtype).min, dtype=mask_dtype,
                   device=query_positions.device),
    )


def rms_norm(hidden, weight, epsilon: float):
    import torch

    if epsilon <= 0:
        raise ValueError("RMS epsilon must be positive")
    normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return normalized.to(hidden.dtype) * weight


def rotate_half(value):
    import torch

    if value.shape[-1] % 2:
        raise ValueError("rotary head dimension must be even")
    midpoint = value.shape[-1] // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def apply_rotary(value, cosine, sine):
    if value.shape != cosine.shape or value.shape != sine.shape:
        raise ValueError("rotary values and tables must have identical shapes")
    return value * cosine + rotate_half(value) * sine


def repeat_key_value(hidden, repetitions: int):
    if repetitions <= 0:
        raise ValueError("key/value repetitions must be positive")
    if repetitions == 1:
        return hidden
    batch, key_heads, sequence, head_dim = hidden.shape
    expanded = hidden[:, :, None, :, :].expand(batch, key_heads, repetitions, sequence, head_dim)
    return expanded.reshape(batch, key_heads * repetitions, sequence, head_dim)


def scaled_attention_probabilities(query, key, additive_mask, scaling: float, groups: int):
    import torch

    repeated_key = repeat_key_value(key, groups)
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * float(scaling)
    return torch.softmax(scores + additive_mask, dim=-1, dtype=torch.float32).to(query.dtype)


def attention_value(probability, value, groups: int):
    import torch

    return torch.matmul(probability, repeat_key_value(value, groups))


def linear_projection(hidden, weight):
    import torch.nn.functional as functional

    return functional.linear(hidden, weight)


def residual_update(residual, branch):
    return residual + branch


def gated_silu(hidden, gate_weight, up_weight, down_weight):
    import torch.nn.functional as functional

    gated = functional.silu(functional.linear(hidden, gate_weight))
    return functional.linear(gated * functional.linear(hidden, up_weight), down_weight)


def final_logits(hidden, norm_weight, lm_head_weight, epsilon: float):
    return linear_projection(rms_norm(hidden, norm_weight, epsilon), lm_head_weight).float()


def single_rank_logits(final_normalized, lm_head_weight, original_vocab_size: int):
    """TP=1 LM-head projection with the padded vocabulary tail removed."""
    if (isinstance(original_vocab_size, bool)
            or not isinstance(original_vocab_size, int)
            or original_vocab_size <= 0
            or original_vocab_size > lm_head_weight.shape[0]):
        raise ValueError("original_vocab_size must select a nonempty LM-head prefix")
    return linear_projection(final_normalized, lm_head_weight)[..., :original_vocab_size]


def cache_next_length(attention_length: int) -> int:
    if attention_length < 0:
        raise ValueError("attention length must be nonnegative")
    return attention_length + 1


def eos_stop(token: int, eos_tokens) -> bool:
    return int(token) in {int(value) for value in eos_tokens}


def token_budget_stop(generated_tokens: int, max_tokens: int) -> bool:
    """Return the exact length-cap condition used by the rater request."""
    for name, value in (("generated_tokens", generated_tokens), ("max_tokens", max_tokens)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if generated_tokens < 0 or max_tokens <= 0:
        raise ValueError("token counts must be nonnegative with a positive maximum")
    return generated_tokens >= max_tokens


def greedy_token(logits):
    """Final-axis argmax; PyTorch's first maximum fixes tie behavior."""
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have a nonempty vocabulary axis")
    return logits.argmax(dim=-1)


try:
    from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor
except ImportError:  # lets non-vLLM control hosts import the formula helpers
    BatchUpdate = object

    class LogitsProcessor:  # type: ignore[no-redef]
        pass


class ProjectArgmaxLogitsProcessor(LogitsProcessor):
    """Force vLLM's downstream sampler to the project-owned greedy token."""

    def __init__(self, vllm_config, device, is_pin_memory: bool) -> None:
        self.device = device

    def apply(self, logits):
        import torch

        selected = greedy_token(logits)
        forced = torch.full_like(logits, -torch.inf)
        forced.scatter_(-1, selected.unsqueeze(-1), 0.0)
        return forced

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _runtime_contract_sha256(manifest: dict) -> str:
    contract = dict(manifest)
    contract.pop("cuda_witness", None)
    contract.pop("cuda_expected", None)
    # CPU-only tokenizer/detokenizer/loader closure additions do not change
    # the already-recorded CUDA-kernel witness.  Reconstruct its original
    # kernel-relevant contract while every launch still validates the full
    # package and dependency closure separately.
    contract["package_versions"] = {
        name: version for name, version in contract["package_versions"].items()
        if name not in {"jinja2", "safetensors", "tokenizers"}
    }
    cuda_dependencies = []
    for entry in contract["dependency_closure"]:
        if not entry.get("cuda_bound", True):
            continue
        entry = dict(entry)
        if entry.get("module") == "vllm.model_executor.model_loader.weight_utils":
            entry["symbols"] = [
                symbol for symbol in entry.get("symbols", [])
                if symbol != "safetensors_weights_iterator"
            ]
        cuda_dependencies.append(entry)
    contract["dependency_closure"] = cuda_dependencies
    return _canonical_sha256(contract)


def validate_runtime_sources(manifest_path: Path, deployment_root: Path) -> tuple[str, ...]:
    """Verify installed dependencies equal hashes bound to reviewed source."""
    manifest_path = Path(manifest_path).resolve()
    deployment_root = Path(deployment_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = []
    for entry in manifest["runtime_sources"]:
        spec = importlib.util.find_spec(entry["module"])
        if spec is None or spec.origin is None:
            raise ValueError(f"missing rater runtime dependency: {entry['module']}")
        installed = Path(spec.origin).resolve()
        snapshot = (deployment_root / entry["snapshot"]).resolve()
        if not snapshot.is_relative_to(deployment_root):
            raise ValueError("runtime source-capture path escapes deployment root")
        installed_sha = _sha256(installed)
        # The bounded semantic-closure artifact carries the exact reviewed
        # excerpts. Full snapshots are capture-time inputs, not worker files.
        if installed_sha != entry["sha256"]:
            raise ValueError(f"rater runtime source changed: {entry['module']}")
        checked.append(entry["module"])
    return tuple(checked)


def validate_cuda_witness(manifest_path: Path, deployment_root: Path) -> str:
    """Validate the real-kernel witness against the pinned worker and artifacts."""
    manifest_path = Path(manifest_path).resolve()
    deployment_root = Path(deployment_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("cuda_witness")
    if not isinstance(entry, dict):
        raise ValueError("rater runtime manifest has no real-CUDA conformance witness")
    witness_path = (deployment_root / entry["path"]).resolve()
    if (not witness_path.is_relative_to(deployment_root)
            or not witness_path.is_file()
            or _sha256(witness_path) != entry["sha256"]):
        raise ValueError("real-CUDA conformance witness changed")
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    if witness.get("status") != "PASS" or witness.get("host") != manifest.get("worker"):
        raise ValueError("real-CUDA witness does not bind the pinned worker")
    if witness.get("packages", {}).get("vllm") != manifest["package_versions"]["vllm"]:
        raise ValueError("real-CUDA witness uses a different vLLM version")
    expected_cuda = manifest.get("cuda_expected")
    cuda = witness.get("cuda", {})
    if (not isinstance(expected_cuda, dict)
            or cuda.get("device_name") != expected_cuda.get("device_name")
            or cuda.get("device_uuid") != expected_cuda.get("device_uuid")
            or cuda.get("capability") != expected_cuda.get("capability")
            or cuda.get("torch") != manifest["package_versions"]["torch"]
            or cuda.get("torch_cuda") != expected_cuda.get("torch_cuda")):
        raise ValueError("real-CUDA witness device/runtime binding changed")
    provenance_entry = manifest.get("build_provenance", {})
    provenance_path = (deployment_root / provenance_entry.get("path", "")).resolve()
    if (not provenance_path.is_relative_to(deployment_root)
            or not provenance_path.is_file()
            or _sha256(provenance_path) != provenance_entry.get("sha256")):
        raise ValueError("CUDA witness build provenance changed")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    config_path = (
        deployment_root / manifest["checkpoint_config_dir"] / "config.json"
    ).resolve()
    if (not config_path.is_relative_to(deployment_root)
            or not config_path.is_file()
            or _sha256(config_path) != manifest["model_files"]["config.json"]):
        raise ValueError("CUDA witness checkpoint config changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts = witness.get("artifacts", {})
    conformance_source = deployment_root / "experiments/xlpt/rater_cuda_conformance.py"
    if (artifacts.get("conformance_source_sha256") != _sha256(conformance_source)
            or artifacts.get("runtime_contract_sha256") != _runtime_contract_sha256(manifest)
            or artifacts.get("checkpoint_config_sha256") != _sha256(config_path)
            or artifacts.get("build_provenance_sha256") != _sha256(provenance_path)):
        raise ValueError("real-CUDA witness artifact binding changed")
    dependency_binding = witness.get("dependency_binding", {})
    runtime_modules = [entry["module"] for entry in manifest["runtime_sources"]]
    production_modules = [
        entry["module"] for entry in manifest["dependency_closure"]
        if entry.get("cuda_bound", True)
    ]
    if (dependency_binding.get("runtime_source_count") != len(runtime_modules)
            or dependency_binding.get("runtime_sources_sha256")
            != _canonical_sha256(runtime_modules)
            or dependency_binding.get("production_dependency_count") != len(production_modules)
            or dependency_binding.get("production_dependencies_sha256")
            != _canonical_sha256(production_modules)):
        raise ValueError("real-CUDA witness dependency binding changed")
    binary_names = {
        "vllm._C": "vllm._C",
        "vllm.flash_attn.fa2": "vllm.vllm_flash_attn._vllm_fa2_C",
        "vllm.flash_attn.fa3": "vllm.vllm_flash_attn._vllm_fa3_C",
    }
    for witness_name, provenance_name in binary_names.items():
        expected = provenance["compiled_extensions"][provenance_name]["sha256"]
        if witness.get("binaries", {}).get(witness_name) != expected:
            raise ValueError(f"CUDA witness binary changed: {witness_name}")
    awq = witness.get("awq", {})
    if awq.get("checkpoint_shard_sha256") != manifest["model_files"].get(awq.get("checkpoint_shard")):
        raise ValueError("CUDA witness checkpoint shard changed")
    if (awq.get("group_size") != 128
            or awq.get("sliced_shapes") != {
                "qweight": [5120, 16], "qzeros": [40, 16], "scales": [40, 128],
            }):
        raise ValueError("real AWQ CUDA witness fixture changed")
    attention = witness.get("flash_attention", {})
    errors = attention.get("prefill_and_decode_max_abs_errors", [])
    if (attention.get("decode_steps") != 2 or attention.get("gqa_groups") != 4
            or attention.get("selected_fa_version") != expected_cuda.get("selected_fa_version")
            or len(errors) != 3 or max(errors, default=float("inf")) > 0.03
            or attention.get("kv_cache_max_abs_error") != 0.0):
        raise ValueError("real paged FlashAttention witness is incomplete")
    if (awq.get("fused_gemm_max_abs_error", float("inf")) > 0.2
            or awq.get("dequantize_matmul_max_abs_error", float("inf")) > 0.2):
        raise ValueError("real AWQ CUDA witness exceeds its numerical tolerance")
    native = witness.get("native_cuda", {})
    text = config["text_config"]
    expected_fixture = {
        "hidden_size": int(text["hidden_size"]),
        "intermediate_size": int(text["intermediate_size"]),
        "head_dim": int(text["head_dim"]),
        "num_attention_heads": int(text["num_attention_heads"]),
        "num_key_value_heads": int(text["num_key_value_heads"]),
        "rms_norm_eps": float(text["rms_norm_eps"]),
        "rope_theta": float(text["rope_theta"]),
        "max_position_embeddings": int(text["max_position_embeddings"]),
    }
    expected_attention_fixture = {
        "num_attention_heads": int(text["num_attention_heads"]),
        "num_key_value_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "block_size": 16,
    }
    if (native.get("fixture") != expected_fixture
            or attention.get("fixture") != expected_attention_fixture):
        raise ValueError("real-CUDA witness does not use checkpoint-compatible fixtures")
    if any(native.get(name, float("inf")) > 0.01 for name in (
        "fused_rms_norm_max_abs_error", "fused_residual_max_abs_error",
        "gated_silu_max_abs_error", "rope_max_abs_error",
    )):
        raise ValueError("real native-CUDA witness exceeds its numerical tolerance")
    return _sha256(witness_path)


def validate_rater_runtime(manifest_path: Path, model_path: Path, deployment_root: Path) -> str:
    """Bind packages, model artifacts, and installed sources to reviewed capture."""
    manifest_path = Path(manifest_path).resolve()
    model_path = Path(model_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest.get("required_unset_environment", []):
        if os.environ.get(name) not in (None, ""):
            raise ValueError(f"rater runtime backend override must be unset: {name}")
    for package, expected in manifest["package_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"rater runtime package mismatch: {package}")
    for name, expected in manifest["model_files"].items():
        if _sha256(model_path / name) != expected:
            raise ValueError(f"rater checkpoint/tokenizer artifact changed: {name}")
    validate_runtime_sources(manifest_path, deployment_root)
    validate_cuda_witness(manifest_path, deployment_root)
    try:
        from .rater_production_closure import run_production_cpu_contracts
    except ImportError:  # direct-script contract tests
        from rater_production_closure import run_production_cpu_contracts
    run_production_cpu_contracts(manifest_path, model_path)
    return _sha256(manifest_path)
