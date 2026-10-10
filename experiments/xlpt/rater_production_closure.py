#!/usr/bin/env python3
"""CPU-only conformance checks for the pinned production vLLM rater path.

The rater still executes the released vLLM 0.11.0 CUDA implementation.  This
module makes that dependency auditable without allocating the GPU: the launch
hash-checks every material Python and binary dependency, verifies reviewable
exact-source excerpts of every Python dependency, then executes the production Python
dispatch/native paths against the independently specified project reference.
The separately recorded CUDA witness covers the pinned compiled kernels.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import ast
import json
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ast_sha256(source: str) -> str:
    class StripDocstrings(ast.NodeTransformer):
        def _visit_body(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
            return node

        visit_Module = _visit_body
        visit_FunctionDef = _visit_body
        visit_AsyncFunctionDef = _visit_body
        visit_ClassDef = _visit_body

    tree = StripDocstrings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _semantic_fragment(source: str, qualified_name: str) -> tuple[str, str]:
    """Mirror the builder's exact AST selection on the installed full file."""
    tree = ast.parse(source)
    parts = qualified_name.split(".")
    top = parts[0]
    node = next(
        (item for item in tree.body
         if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
         and item.name == top),
        None,
    )
    if node is not None and len(parts) > 1:
        if not isinstance(node, ast.ClassDef):
            raise ValueError(f"{qualified_name}: dotted symbol is not rooted at a class")
        node = next(
            (item for item in node.body
             if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and item.name == parts[1]),
            None,
        )
    if node is not None:
        fragment = ast.get_source_segment(source, node)
        if fragment:
            return "definition", fragment
    for item in tree.body:
        if isinstance(item, ast.ImportFrom):
            for alias in item.names:
                if (alias.asname or alias.name) == top:
                    fragment = ast.get_source_segment(source, item)
                    if fragment:
                        return "import_binding", fragment
        elif isinstance(item, ast.Import):
            for alias in item.names:
                if (alias.asname or alias.name.split(".")[-1]) == top:
                    fragment = ast.get_source_segment(source, item)
                    if fragment:
                        return "import_binding", fragment
    for item in tree.body:
        if not isinstance(
            item,
            (ast.Assign, ast.AnnAssign, ast.If, ast.Try, ast.With, ast.For),
        ):
            continue
        binds_name = any(
            (isinstance(child, ast.Name) and child.id == top
             and isinstance(child.ctx, ast.Store))
            or (isinstance(child, ast.alias)
                and (child.asname or child.name.split(".")[-1]) == top)
            for child in ast.walk(item)
        )
        if binds_name:
            fragment = ast.get_source_segment(source, item)
            if fragment:
                return "conditional_binding", fragment
    raise ValueError(f"cannot locate semantic source for {qualified_name}")


def validate_semantic_dependency_closure(
    manifest: dict, deployment_root: Path
) -> tuple[str, ...]:
    """Bind every declared Python dependency symbol to reviewable exact source."""
    path = deployment_root / "process/consumption/deployment/mistral-rater-semantic-closure-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_entries = [entry for entry in manifest["dependency_closure"] if entry.get("snapshot")]
    rows = payload.get("modules")
    if (payload.get("version") != 1 or not isinstance(rows, list)
            or payload.get("python_dependency_count") != len(expected_entries)):
        raise ValueError("semantic production closure is incomplete")
    by_module = {row.get("module"): row for row in rows}
    if set(by_module) != {entry["module"] for entry in expected_entries}:
        raise ValueError("semantic production closure module set changed")
    checked = []
    for entry in expected_entries:
        row = by_module[entry["module"]]
        spec = importlib.util.find_spec(entry["module"])
        if spec is None or spec.origin is None:
            raise ValueError(f"installed semantic source missing: {entry['module']}")
        origin = Path(spec.origin).resolve()
        source = origin.read_text(encoding="utf-8")
        if row.get("file_sha256") != entry["sha256"] or _sha256(origin) != entry["sha256"]:
            raise ValueError(f"semantic closure file binding changed: {entry['module']}")
        fragments = row.get("symbols")
        declared = entry.get("symbols") or ["<module>"]
        if not isinstance(fragments, list) or [item.get("symbol") for item in fragments] != declared:
            raise ValueError(f"semantic closure symbol set changed: {entry['module']}")
        for item in fragments:
            stored = item.get("source")
            if not isinstance(stored, str) or _ast_sha256(stored) != item.get("ast_sha256"):
                raise ValueError(f"semantic closure excerpt changed: {entry['module']}")
            if item["symbol"] == "<module>":
                actual_kind, actual = "complete_module", source
            else:
                actual_kind, actual = _semantic_fragment(source, item["symbol"])
            if actual_kind != item.get("kind") or _ast_sha256(actual) != item.get("ast_sha256"):
                raise ValueError(
                    f"installed semantic source changed: {entry['module']}:{item['symbol']}"
                )
        checked.append(entry["module"])
    return tuple(checked)


def validate_production_dependency_closure(manifest_path: Path) -> tuple[str, ...]:
    """Hash and resolve every material dependency named by the runtime ledger."""
    manifest_path = Path(manifest_path).resolve()
    deployment_root = Path.cwd().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("dependency_closure")
    if not isinstance(entries, list) or not entries:
        raise ValueError("rater runtime manifest has no production dependency closure")
    provenance_entry = manifest.get("build_provenance")
    if not isinstance(provenance_entry, dict):
        raise ValueError("rater runtime manifest has no compiled build provenance")
    provenance = (deployment_root / provenance_entry["path"]).resolve()
    if (not provenance.is_relative_to(deployment_root)
            or not provenance.is_file()
            or _sha256(provenance) != provenance_entry["sha256"]):
        raise ValueError("compiled rater build provenance changed")
    provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
    if (provenance_payload.get("distribution") != "vllm"
            or provenance_payload.get("distribution_version") != "0.11.0"):
        raise ValueError("compiled rater build provenance does not identify vLLM 0.11.0")
    checked = []
    for entry in entries:
        module_name = entry["module"]
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise ValueError(f"missing production rater dependency: {module_name}")
        origin = Path(spec.origin).resolve()
        if not origin.is_file() or _sha256(origin) != entry["sha256"]:
            raise ValueError(f"production rater dependency changed: {module_name}")
        snapshot_relative = entry.get("snapshot")
        if origin.suffix == ".py":
            if not snapshot_relative:
                raise ValueError(f"reviewable Python capture metadata missing: {module_name}")
        elif snapshot_relative:
            raise ValueError(f"compiled dependency must use build provenance, not a source snapshot: {module_name}")
        elif entry.get("distribution"):
            distribution = entry["distribution"]
            expected_version = entry.get("distribution_version")
            if (not isinstance(expected_version, str)
                    or importlib.metadata.version(distribution) != expected_version
                    or entry.get("size_bytes") != origin.stat().st_size):
                raise ValueError(
                    f"compiled dependency distribution changed: {module_name}"
                )
        elif entry.get("build_provenance") != provenance_entry["path"]:
            raise ValueError(f"compiled dependency lacks pinned build provenance: {module_name}")
        if origin.suffix != ".py":
            if entry.get("distribution"):
                if entry.get("sha256") != _sha256(origin):
                    raise ValueError(f"compiled dependency hash changed: {module_name}")
            else:
                binary = provenance_payload.get("compiled_extensions", {}).get(module_name)
                if (not isinstance(binary, dict)
                        or binary.get("sha256") != entry["sha256"]
                        or binary.get("size_bytes") != origin.stat().st_size):
                    raise ValueError(f"compiled dependency provenance changed: {module_name}")
        # Symbol existence and exact AST content are checked against the
        # installed source by validate_semantic_dependency_closure below.
        # Avoid importing every module here: some CUDA backends initialize the
        # device at import time, while the deployment probe is CPU-isolated.
        checked.append(module_name)
    if len(checked) != len(set(checked)):
        raise ValueError("production dependency closure contains duplicate modules")
    semantic_checked = validate_semantic_dependency_closure(manifest, deployment_root)
    python_checked = tuple(
        entry["module"] for entry in entries if entry.get("snapshot")
    )
    if semantic_checked != python_checked:
        raise ValueError("semantic dependency closure order changed")
    return tuple(checked)


def _pack_awq(rows):
    """Literal AWQ interleave oracle; deliberately independent of project constants."""
    import torch

    interleave = (0, 4, 1, 5, 2, 6, 3, 7)
    packed_rows = []
    for row in rows:
        words = []
        for start in range(0, len(row), 8):
            word = 0
            for slot, shift_index in enumerate(interleave):
                word |= (int(row[start + slot]) & 0xF) << (4 * shift_index)
            words.append(word - (1 << 32) if word >= (1 << 31) else word)
        packed_rows.append(words)
    return torch.tensor(packed_rows, dtype=torch.int32)


def production_awq_dispatch_contract() -> None:
    """Execute vLLM's exact AWQLinearMethod.apply branch and shape logic on CPU."""
    import torch
    from vllm.model_executor.layers.quantization import awq as production
    from . import mistral_awq_reference as reference

    input_dim = output_dim = 8
    group_size = 4
    weight_rows = [[(3 * i + 5 * j) % 16 for j in range(output_dim)]
                   for i in range(input_dim)]
    zero_rows = [[8] * output_dim for _ in range(input_dim // group_size)]
    layer = SimpleNamespace(
        qweight=_pack_awq(weight_rows),
        scales=torch.full((input_dim // group_size, output_dim), 0.125),
        qzeros=_pack_awq(zero_rows),
    )
    method = production.AWQLinearMethod(
        production.AWQConfig(4, group_size, False)
    )
    original_gemm = production.ops.awq_gemm
    original_dequantize = production.ops.awq_dequantize
    calls = []

    def fake_gemm(x, qweight, scales, qzeros, pack_factor):
        if pack_factor != 8:
            raise AssertionError("production AWQ pack factor changed")
        calls.append("gemm")
        return reference.awq_linear(
            x, qweight, scales, qzeros, group_size
        )

    def fake_dequantize(qweight, scales, qzeros, *_workspace):
        calls.append("dequantize")
        return reference.awq_dequantize(
            qweight, scales, qzeros, group_size
        )

    production.ops.awq_gemm = fake_gemm
    production.ops.awq_dequantize = fake_dequantize
    try:
        small = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 8.0
        bias = torch.arange(8, dtype=torch.float32) / 10.0
        expected_small = reference.awq_linear(
            small, layer.qweight, layer.scales, layer.qzeros, group_size
        ) + bias
        actual_small = method.apply(layer, small, bias)
        if not torch.allclose(actual_small, expected_small, atol=1e-6):
            raise AssertionError("production fused-AWQ dispatch disagrees with the reference")

        large = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8) / 257.0
        expected_large = reference.awq_linear(
            large, layer.qweight, layer.scales, layer.qzeros, group_size
        )
        actual_large = method.apply(layer, large)
        if not torch.allclose(actual_large, expected_large, atol=1e-5):
            raise AssertionError("production dequantize-matmul branch disagrees with the reference")
    finally:
        production.ops.awq_gemm = original_gemm
        production.ops.awq_dequantize = original_dequantize
    if calls != ["gemm", "dequantize"]:
        raise AssertionError(f"both production AWQ branches were not executed: {calls}")


def production_native_kernel_contract() -> None:
    """Exercise production embedding, RMSNorm, RoPE, SiLU and dense LM-head math."""
    import torch
    import torch.nn.functional as functional
    from vllm.model_executor.layers import vocab_parallel_embedding as vocab_module
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from . import mistral_awq_reference as reference

    table = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    ids = torch.tensor([[2, 0, 1]])

    class EmbeddingMethod:
        @staticmethod
        def embedding(layer, token_ids):
            return functional.embedding(token_ids, layer.weight)

    fake_embedding = SimpleNamespace(tp_size=1, quant_method=EmbeddingMethod(), weight=table)
    original_reduce = vocab_module.tensor_model_parallel_all_reduce
    vocab_module.tensor_model_parallel_all_reduce = lambda value: value
    try:
        embedded = vocab_module.VocabParallelEmbedding.forward_native(fake_embedding, ids)
    finally:
        vocab_module.tensor_model_parallel_all_reduce = original_reduce
    if not torch.equal(embedded, reference.embedding_lookup(ids, table)):
        raise AssertionError("production embedding path disagrees with the reference")

    hidden = torch.tensor([[[1.0, -2.0, 0.5, 3.0]]])
    norm = RMSNorm(4, eps=1e-5, dtype=torch.float32)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 2.0, -1.0]))
    if not torch.allclose(
        norm.forward_native(hidden),
        reference.rms_norm(hidden, norm.weight.detach(), 1e-5),
        atol=1e-6,
    ):
        raise AssertionError("production native RMSNorm disagrees with the reference")

    gate = torch.tensor([[[1.0, -2.0, 0.5]]])
    up = torch.tensor([[[2.0, 3.0, -4.0]]])
    actual_activation = SiluAndMul().forward_native(torch.cat((gate, up), dim=-1))
    expected_activation = reference.silu(gate) * up
    if not torch.allclose(actual_activation, expected_activation, atol=1e-6):
        raise AssertionError("production native SiluAndMul disagrees with the reference")

    rotary = RotaryEmbedding(4, 4, 16, 10000.0, True, torch.float32)
    positions = torch.tensor([0, 3], dtype=torch.long)
    query = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    actual_query, _ = rotary.forward_native(positions, query.clone(), None)
    cosine, sine = reference.rope_tables(positions, 4, 10000.0)
    expected_query = reference.apply_rope(
        query.view(2, 1, 4).transpose(0, 1),
        cosine.unsqueeze(0), sine.unsqueeze(0),
    ).transpose(0, 1).reshape_as(query)
    if not torch.allclose(actual_query, expected_query, atol=1e-6):
        raise AssertionError("production native RoPE disagrees with the reference")

    dense_layer = SimpleNamespace(weight=torch.tensor([[1.0, 0.0], [0.5, -1.0]]))
    dense_input = torch.tensor([[2.0, 3.0]])
    dense = UnquantizedLinearMethod().apply(dense_layer, dense_input)
    expected_dense = reference.linear(dense_input, dense_layer.weight)
    if not torch.equal(dense, expected_dense):
        raise AssertionError("production dense LM-head projection disagrees with the reference")


def production_attention_contract() -> None:
    """Exercise production QKV splitting and prefill-to-cached-decode routing."""
    import torch
    from vllm.attention import layer as attention_module
    from vllm.model_executor.models.llama import LlamaAttention
    from . import mistral_awq_reference as reference

    fused = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)

    class Projection:
        def __call__(self, hidden):
            if hidden.shape[0] != 2:
                raise AssertionError("production QKV projection input changed")
            return fused.clone(), None

    class Rotary:
        def __call__(self, positions, query, key):
            if positions.tolist() != [0, 1]:
                raise AssertionError("production position routing changed")
            return query, key

    captured = {}

    class CaptureAttention:
        def __call__(self, query, key, value):
            captured.update(query=query.clone(), key=key.clone(), value=value.clone())
            return query

    class IdentityProjection:
        def __call__(self, hidden):
            return hidden, None

    fake_llama = SimpleNamespace(
        qkv_proj=Projection(), q_size=4, kv_size=2,
        rotary_emb=Rotary(), attn=CaptureAttention(), o_proj=IdentityProjection(),
    )
    hidden = torch.zeros(2, 4)
    llama_output = LlamaAttention.forward(fake_llama, torch.tensor([0, 1]), hidden)
    if not torch.equal(captured["query"], fused[:, :4]):
        raise AssertionError("production query split changed")
    if not torch.equal(captured["key"], fused[:, 4:6]) or not torch.equal(captured["value"], fused[:, 6:8]):
        raise AssertionError("production key/value split changed")
    if not torch.equal(llama_output, fused[:, :4]):
        raise AssertionError("production attention output projection changed")

    class ReferenceBackend:
        def forward(self, layer, query, key, value, cache, metadata, output=None):
            del layer, metadata
            cache["key"] = key.clone() if "key" not in cache else torch.cat((cache["key"], key), dim=0)
            cache["value"] = value.clone() if "value" not in cache else torch.cat((cache["value"], value), dim=0)
            q = query.transpose(0, 1).unsqueeze(0)
            k = cache["key"].transpose(0, 1).unsqueeze(0)
            v = cache["value"].transpose(0, 1).unsqueeze(0)
            attended = reference.scaled_causal_attention(q, k, v, groups=2)
            result = attended.squeeze(0).transpose(0, 1)
            output.copy_(result)
            return output

    cache = {}
    fake_layer = SimpleNamespace(
        calculate_kv_scales=False, query_quant=None, use_output=True,
        use_mla=False, use_direct_call=True, num_heads=2, num_kv_heads=1,
        head_size=2, impl=ReferenceBackend(), kv_cache=[cache], layer_name="contract",
    )
    original_context = attention_module.get_forward_context
    attention_module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={"contract": {}}, virtual_engine=0
    )
    try:
        prefill_q = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 1.0, -1.0]])
        prefill_k = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        prefill_v = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        prefill = attention_module.Attention.forward(fake_layer, prefill_q, prefill_k, prefill_v)
        decode_q = torch.tensor([[1.0, 1.0, -1.0, 0.5]])
        decode_k = torch.tensor([[0.5, 0.5]])
        decode_v = torch.tensor([[10.0, 12.0]])
        decoded = attention_module.Attention.forward(fake_layer, decode_q, decode_k, decode_v)
    finally:
        attention_module.get_forward_context = original_context
    if prefill.shape != (2, 4) or decoded.shape != (1, 4):
        raise AssertionError("production attention output shapes changed")
    if cache["key"].shape != (3, 1, 2) or cache["value"].shape != (3, 1, 2):
        raise AssertionError("production prefill/decode cache append changed")


def production_model_structure_contract() -> None:
    """Exercise production Mistral/Llama MLP and two-residual decoder ordering."""
    import torch
    from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP

    events = []

    class Projection:
        def __init__(self, name, transform):
            self.name, self.transform = name, transform

        def __call__(self, value):
            events.append(self.name)
            return self.transform(value), None

    fake_mlp = SimpleNamespace(
        gate_up_proj=Projection("gate_up", lambda value: torch.cat((value, value + 1), dim=-1)),
        act_fn=lambda value: value[..., : value.shape[-1] // 2] * value[..., value.shape[-1] // 2 :],
        down_proj=Projection("down", lambda value: value - 2),
    )
    mlp_input = torch.tensor([[1.0, 2.0]])
    mlp_output = LlamaMLP.forward(fake_mlp, mlp_input)
    if events != ["gate_up", "down"] or not torch.equal(mlp_output, mlp_input * (mlp_input + 1) - 2):
        raise AssertionError("production gated-MLP ordering changed")

    class FirstNorm:
        def __call__(self, hidden, residual=None):
            events.append("input_norm")
            if residual is None:
                return hidden + 1
            return hidden + residual + 1, hidden + residual

    class PostNorm:
        def __call__(self, hidden, residual):
            events.append("post_norm")
            return hidden + residual + 2, hidden + residual

    class Branch:
        def __init__(self, name, delta):
            self.name, self.delta = name, delta

        def __call__(self, **kwargs):
            events.append(self.name)
            value = kwargs.get("hidden_states")
            return value + self.delta

    fake_decoder = SimpleNamespace(
        input_layernorm=FirstNorm(), self_attn=Branch("attention", 3),
        post_attention_layernorm=PostNorm(), mlp=lambda value: events.append("mlp") or value + 4,
    )
    events.clear()
    hidden_out, residual_out = LlamaDecoderLayer.forward(
        fake_decoder, torch.tensor([0]), torch.tensor([[1.0, 2.0]]), None
    )
    if events != ["input_norm", "attention", "post_norm", "mlp"]:
        raise AssertionError(f"production decoder ordering changed: {events}")
    if hidden_out.shape != residual_out.shape or hidden_out.shape != (1, 2):
        raise AssertionError("production decoder residual shapes changed")


def production_checkpoint_group_contract(model_dir: Path) -> None:
    """Derive the deployed checkpoint's GQA ratio from hash-validated config."""
    from . import mistral_awq_reference as reference

    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    text_config = reference.text_config_of(config)
    query_heads = int(text_config["num_attention_heads"])
    key_value_heads = int(text_config["num_key_value_heads"])
    if reference.key_value_group_count(query_heads, key_value_heads) != 4:
        raise AssertionError("deployed checkpoint GQA ratio changed from 32/8=4")


def production_tokenizer_io_contract(model_dir: Path) -> None:
    """Exercise the pinned template, real tokenizer, vLLM input, and detokenizer."""
    from transformers import AutoTokenizer
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.detokenizer import (
        FastIncrementalDetokenizer,
        IncrementalDetokenizer,
    )
    from .run_rater_scoring import validate_context_window

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=False, local_files_only=True
    )
    template_payload = json.loads(
        (model_dir / "chat_template.json").read_text(encoding="utf-8")
    )
    template = template_payload.get("chat_template")
    if tokenizer.chat_template is not None or not isinstance(template, str):
        raise AssertionError("checkpoint chat-template storage contract changed")
    messages = [
        {"role": "system", "content": "contract-system"},
        {"role": "user", "content": "contract-user"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=True, add_generation_prompt=True
    )
    expected_rendered = (
        "<s>[SYSTEM_PROMPT]contract-system[/SYSTEM_PROMPT]"
        "[INST]contract-user[/INST]"
    )
    expected_ids = [1, 17, 58026, 59856, 18, 3, 58026, 24804, 4]
    if (rendered != expected_rendered or token_ids != expected_ids
            or tokenizer.encode(rendered, add_special_tokens=False) != expected_ids):
        raise AssertionError("real checkpoint prompt tokenization changed")
    validate_context_window(expected_ids)
    validate_context_window([1] * 4087)
    try:
        validate_context_window([1] * 4088)
    except ValueError:
        pass
    else:
        raise AssertionError("4096-token context-boundary request was accepted")

    preprocessor = InputPreprocessor(
        SimpleNamespace(is_encoder_decoder=False), tokenizer
    )
    processed = preprocessor.preprocess({"prompt_token_ids": expected_ids})
    if processed != {"type": "token", "prompt_token_ids": expected_ids}:
        raise AssertionError("vLLM real token-input preprocessing changed")
    request = SimpleNamespace(
        sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0, max_tokens=8
        ),
        request_id="rater-detokenizer-contract",
        prompt_token_ids=expected_ids,
        prompt_embeds=None,
    )
    detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)
    if not isinstance(detokenizer, FastIncrementalDetokenizer):
        raise AssertionError("vLLM no longer selected the pinned fast detokenizer")
    if (tokenizer.encode("7", add_special_tokens=False) != [1055]
            or detokenizer.update([1055], False) is not None
            or detokenizer.get_next_output_text(True, False) != "7"):
        raise AssertionError("real generated score detokenization changed")


def production_safetensors_loader_contract() -> None:
    """Load an independently encoded fixture through the exact production loader."""
    import torch
    from safetensors.torch import load_file
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    header = {
        "alpha": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "beta": {"dtype": "I64", "shape": [2], "data_offsets": [16, 32]},
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    payload = (
        struct.pack("<Q", len(encoded))
        + encoded
        + struct.pack("<4f2q", 1.25, -2.5, 3.75, 0.5, -3, 11)
    )
    with tempfile.TemporaryDirectory(prefix="rater-safetensors-contract-") as temporary:
        fixture = Path(temporary) / "known.safetensors"
        fixture.write_bytes(payload)
        loaded = load_file(str(fixture), device="cpu")
        loader = object.__new__(DefaultModelLoader)
        loader.load_config = SimpleNamespace(
            model_loader_extra_config={},
            load_format="auto",
            download_dir=None,
            use_tqdm_on_load=False,
            safetensors_load_strategy="lazy",
            pt_load_map_location="cpu",
        )
        loader.counter_before_loading_weights = 0.0
        loader._prepare_weights = lambda *_args, **_kwargs: (
            temporary, [str(fixture)], True
        )
        source = SimpleNamespace(
            model_or_path=temporary,
            revision=None,
            fall_back_to_pt=False,
            allow_patterns_overrides=None,
            prefix="",
        )
        production_loaded = dict(loader._get_weights_iterator(source))
    if set(loaded) != {"alpha", "beta"}:
        raise AssertionError("safetensors loader changed known tensor names")
    if (loaded["alpha"].shape != (2, 2)
            or loaded["alpha"].dtype != torch.float32
            or loaded["alpha"].tolist() != [[1.25, -2.5], [3.75, 0.5]]
            or loaded["beta"].shape != (2,)
            or loaded["beta"].dtype != torch.int64
            or loaded["beta"].tolist() != [-3, 11]
            or set(production_loaded) != {"alpha", "beta"}
            or not torch.equal(production_loaded["alpha"], loaded["alpha"])
            or not torch.equal(production_loaded["beta"], loaded["beta"])):
        raise AssertionError("safetensors loader changed known tensor shape or values")


def production_logits_stopping_contract(model_dir: Path) -> None:
    """Exercise production logits, first-index argmax and checkpoint EOS stopping."""
    import torch
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.utils import check_stop
    from vllm.v1.request import RequestStatus
    from vllm.v1.sample.sampler import Sampler
    from .rater_runtime import (
        ProjectArgmaxLogitsProcessor,
        single_rank_logits,
        token_budget_stop,
    )

    model_dir = Path(model_dir)
    generation = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    eos_id = int(generation["eos_token_id"])
    if tokenizer.get("eos_token") != "</s>" or eos_id != 2:
        raise AssertionError("checkpoint terminal-token configuration changed")

    head = SimpleNamespace(
        weight=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [9.0, 9.0]]),
        quant_method=UnquantizedEmbeddingMethod(),
    )
    head.cpu_linear = lambda hidden, weight, bias: torch.nn.functional.linear(
        hidden, weight, bias
    )
    processor = object.__new__(LogitsProcessor)
    processor.org_vocab_size = 3
    processor.use_all_gather = False
    from vllm.model_executor.layers import logits_processor as logits_module
    original_gather = logits_module.tensor_model_parallel_gather
    logits_module.tensor_model_parallel_gather = lambda value: value
    try:
        logits = LogitsProcessor._get_logits(
            processor, torch.tensor([[2.0, 3.0]]), head, None
        )
    finally:
        logits_module.tensor_model_parallel_gather = original_gather
    if logits.tolist() != [[2.0, 3.0, 5.0]]:
        raise AssertionError("production vocabulary projection/gather/truncation changed")
    expected_logits = single_rank_logits(
        torch.tensor([[2.0, 3.0]]), head.weight, processor.org_vocab_size
    )
    if not torch.equal(logits, expected_logits):
        raise AssertionError("production TP=1 logits disagree with the independent prefix oracle")

    ties = torch.tensor([[7.0, 7.0, 1.0], [-2.0, -1.0, -1.0]])
    expected = torch.tensor([0, 1])
    if not torch.equal(Sampler.greedy_sample(None, ties), expected):
        raise AssertionError("production sampler first-index argmax changed")
    project_processor = ProjectArgmaxLogitsProcessor(None, torch.device("cpu"), False)
    if not torch.equal(project_processor.apply(ties).argmax(dim=-1), expected):
        raise AssertionError("project-owned argmax adapter changed")

    params = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=8)
    if params.top_k != 0:
        raise AssertionError("vLLM greedy top_k normalization changed")
    params.update_from_generation_config(generation, model_eos_token_id=eos_id)
    if params.all_stop_token_ids != {eos_id}:
        raise AssertionError(f"production checkpoint EOS set changed: {params.all_stop_token_ids}")
    eos_request = SimpleNamespace(
        num_tokens=2, num_output_tokens=1, max_tokens=8, pooling_params=None,
        sampling_params=params, output_token_ids=[eos_id], eos_token_id=eos_id,
        status=RequestStatus.RUNNING, stop_reason=None,
    )
    if not check_stop(eos_request, 4096) or eos_request.status != RequestStatus.FINISHED_STOPPED:
        raise AssertionError("production scheduler did not stop on checkpoint EOS")
    nonterminal = SimpleNamespace(**{**eos_request.__dict__, "output_token_ids": [eos_id - 1],
                                     "status": RequestStatus.RUNNING})
    if check_stop(nonterminal, 4096):
        raise AssertionError("production scheduler stopped on a nonterminal token")
    length_capped = SimpleNamespace(
        **{
            **eos_request.__dict__,
            "num_output_tokens": 8,
            "output_token_ids": [eos_id - 1] * 8,
            "status": RequestStatus.RUNNING,
            "stop_reason": None,
        }
    )
    if (not token_budget_stop(8, 8)
            or token_budget_stop(7, 8)
            or not check_stop(length_capped, 4096)
            or length_capped.status != RequestStatus.FINISHED_LENGTH_CAPPED):
        raise AssertionError("production scheduler did not enforce max_tokens=8")


def production_entrypoint_processor_contract(model_dir: Path) -> None:
    """Execute vLLM's real LLM argument, request/EOS and logits-plugin routing."""
    import torch
    import vllm.entrypoints.llm as llm_module
    import vllm.platforms as platforms_module
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import processor as processor_module
    from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns
    from vllm.v1.sample.logits_processor.state import LogitsProcessors
    from .rater_runtime import ProjectArgmaxLogitsProcessor

    loaded = _load_logitsprocs_by_fqcns(
        ["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"]
    )
    if loaded != [ProjectArgmaxLogitsProcessor]:
        raise AssertionError("vLLM did not load the project logits processor by FQCN")
    project_processor = loaded[0](None, torch.device("cpu"), False)
    processor_state = LogitsProcessors(iter((project_processor,)))
    if processor_state.argmax_invariant or processor_state.non_argmax_invariant != [project_processor]:
        raise AssertionError("project argmax processor was classified as argmax-invariant")
    ties = torch.tensor([[4.0, 4.0, 1.0], [-1.0, 2.0, 2.0]])
    if processor_state.non_argmax_invariant[0].apply(ties).argmax(dim=-1).tolist() != [0, 1]:
        raise AssertionError("loaded project processor changed first-index tie behavior")

    generation = json.loads((Path(model_dir) / "generation_config.json").read_text(encoding="utf-8"))

    class InputPreprocessor:
        @staticmethod
        def preprocess(prompt, tokenization_kwargs=None, mm_uuids=None):
            del prompt, tokenization_kwargs, mm_uuids
            return {"type": "tokens", "prompt": "contract", "prompt_token_ids": [1, 2]}

        @staticmethod
        def get_eos_token_id():
            return 2

    processor = object.__new__(processor_module.Processor)
    processor._validate_lora = lambda value: None
    processor._validate_params = lambda value: None
    processor._validate_multi_modal_uuids = lambda value: None
    processor._validate_model_inputs = lambda encoder, decoder: None
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    processor.model_config = SimpleNamespace(multimodal_config=None, max_model_len=4096)
    processor.cache_config = SimpleNamespace(enable_prefix_caching=False)
    processor.input_preprocessor = InputPreprocessor()
    processor.generation_config_fields = generation
    processor.tokenizer = None
    original_platform = platforms_module.current_platform
    platforms_module.current_platform = SimpleNamespace(
        validate_request=lambda **kwargs: None
    )
    try:
        _, core_request = processor_module.Processor.process_inputs(
            processor,
            "contract-request",
            "contract prompt",
            SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=8),
        )
    finally:
        platforms_module.current_platform = original_platform
    if core_request.eos_token_id != 2 or core_request.sampling_params.all_stop_token_ids != {2}:
        raise AssertionError("production Processor did not apply checkpoint EOS configuration")
    if core_request.sampling_params.top_k != 0:
        raise AssertionError("production Processor changed normalized greedy top_k")

    captured = {}

    class FakeEngineArgs:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeEngine:
        model_config = SimpleNamespace(io_processor_plugin=None)
        vllm_config = SimpleNamespace()

        @staticmethod
        def get_supported_tasks():
            return {"generate"}

    class FakeLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args, usage_context):
            del cls, engine_args, usage_context
            return FakeEngine()

    saved = {
        "EngineArgs": llm_module.EngineArgs,
        "LLMEngine": llm_module.LLMEngine,
        "log_non_default_args": llm_module.log_non_default_args,
        "get_io_processor": llm_module.get_io_processor,
    }
    llm_module.EngineArgs = FakeEngineArgs
    llm_module.LLMEngine = FakeLLMEngine
    llm_module.log_non_default_args = lambda value: None
    llm_module.get_io_processor = lambda config, plugin: None
    try:
        instance = object.__new__(llm_module.LLM)
        llm_module.LLM.__init__(
            instance,
            model=str(Path(model_dir).resolve()),
            trust_remote_code=False,
            quantization="awq",
            dtype="auto",
            gpu_memory_utilization=0.88,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 0},
            logits_processors=["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"],
            max_model_len=4096,
            enable_prefix_caching=False,
        )
    finally:
        for name, value in saved.items():
            setattr(llm_module, name, value)
    expected = {
        "model": str(Path(model_dir).resolve()),
        "trust_remote_code": False,
        "quantization": "awq",
        "dtype": "auto",
        "gpu_memory_utilization": 0.88,
        "enforce_eager": True,
        "limit_mm_per_prompt": {"image": 0},
        "logits_processors": ["experiments.xlpt.rater_runtime:ProjectArgmaxLogitsProcessor"],
        "max_model_len": 4096,
        "enable_prefix_caching": False,
        "tensor_parallel_size": 1,
    }
    for name, value in expected.items():
        if captured.get(name) != value:
            raise AssertionError(f"production LLM argument forwarding changed: {name}")


def production_construction_selection_contract() -> None:
    """Bind the registered text model and AWQ selector used during loading."""
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.models.registry import ModelRegistry

    registered = ModelRegistry.models.get("MistralForCausalLM")
    if registered is None or vars(registered) != {
        "module_name": "vllm.model_executor.models.llama",
        "class_name": "LlamaForCausalLM",
    }:
        raise AssertionError("Mistral text-model registry route changed")
    quantization = get_quantization_config("awq")
    if (quantization.__module__
            != "vllm.model_executor.layers.quantization.awq"
            or quantization.__name__ != "AWQConfig"):
        raise AssertionError("AWQ quantization selector changed")


def run_production_cpu_contracts(manifest_path: Path, model_dir: Path) -> dict:
    """Run the exact CPU-safe closure invoked by every real rater launch."""
    checked = validate_production_dependency_closure(manifest_path)
    production_awq_dispatch_contract()
    production_native_kernel_contract()
    production_attention_contract()
    production_model_structure_contract()
    production_checkpoint_group_contract(model_dir)
    production_tokenizer_io_contract(model_dir)
    production_safetensors_loader_contract()
    production_logits_stopping_contract(model_dir)
    production_entrypoint_processor_contract(model_dir)
    production_construction_selection_contract()
    return {
        "dependencies": len(checked),
        "awq_branches": ["fused_gemm", "dequantize_matmul"],
        "attention": "prefill_plus_cached_decode",
        "eos_token_ids": [2],
        "checkpoint_gqa_groups": 4,
        "tokenizer_io": "real_template_tokens_and_score_text",
        "safetensors_loader": "known_names_shapes_values",
        "entrypoint": "llm_processor_logits_plugin",
        "construction": "registry_llama_plus_awq",
    }


__all__ = [
    "validate_production_dependency_closure",
    "production_awq_dispatch_contract",
    "production_native_kernel_contract",
    "production_attention_contract",
    "production_model_structure_contract",
    "production_checkpoint_group_contract",
    "production_logits_stopping_contract",
    "production_entrypoint_processor_contract",
    "production_construction_selection_contract",
    "run_production_cpu_contracts",
]
