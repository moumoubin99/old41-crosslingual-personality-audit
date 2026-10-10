#!/usr/bin/env python3
"""Executable contract for deployment unit U4 - checkpoint identity and parity.

Two things are established here, both of which the deployment review asked for
by name: that the launch records *which* checkpoint it actually read, and that
the captured forward agrees numerically with an independent implementation of
the same architecture.

The independent implementation is the released `transformers` `Qwen3ForCausalLM`
class. It is genuinely independent of `qwen3_reference` - different code, different
authors - and the two are compared on identical weights. The comparison runs on a
*small* Qwen3 configuration, not on the 8B release: loading 8B on CPU cannot
finish inside the reviewer's 300 s test budget. The released checkpoint is pinned
separately, by identity rather than by forward pass: `checkpoint_identity` is
exercised here against a checkpoint written to a temporary directory, and the real
path under the deployment host is asserted when it is present.

No expected digest for the released weights is hard-coded anywhere in this
repository: the weights are not reachable from the authoring sandbox, so the
launch records what it read and the reviewer compares it on the host.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from qwen3_reference import (
    REQUIRED_CONFIG_KEYS,
    checkpoint_identity,
    load_checkpoint_params,
    model_logits,
    rope_theta_of,
    sha256_file,
)

RELEASED_MODEL_PATH = Path("<LOCAL_PATH>/Qwen__Qwen3-8B")
PARITY_TOL = 1e-6

TINY = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 48,
    "vocab_size": 23,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "attention_bias": False,
}


def build_reference_model(seed):
    """A small real Qwen3ForCausalLM with deterministic weights."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(**TINY, attn_implementation="eager")
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model.to(torch.float64)


def write_checkpoint(model, directory):
    model.save_pretrained(directory, safe_serialization=True)
    return Path(directory)


def check_identity_is_read_not_assumed(root):
    identity = checkpoint_identity(root)

    if identity["model_path"] != str(root):
        raise AssertionError(f"identity records path {identity['model_path']}")
    if identity["architectures"] != ["Qwen3ForCausalLM"]:
        raise AssertionError(f"architectures {identity['architectures']}")
    if identity["head_dim"] != TINY["head_dim"]:
        raise AssertionError(f"head_dim {identity['head_dim']}")
    if identity["tie_word_embeddings"] is not False:
        raise AssertionError("tie_word_embeddings was not read from the config")

    for key in REQUIRED_CONFIG_KEYS:
        if key not in identity["config_of_record"]:
            raise AssertionError(f"config_of_record omits {key}")
        if identity["config_of_record"][key] != TINY[key]:
            raise AssertionError(
                f"config_of_record[{key}] = {identity['config_of_record'][key]!r}, "
                f"checkpoint declares {TINY[key]!r}")

    # Digests are recomputed here, independently of sha256_file's own bookkeeping.
    expected_config = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    if identity["config_sha256"] != expected_config:
        raise AssertionError("config_sha256 does not match an independent digest")
    shards = sorted(p for p in root.iterdir() if p.suffix in (".safetensors", ".bin"))
    if not shards:
        raise AssertionError("the written checkpoint has no weight shard to pin")
    if [entry["name"] for entry in identity["weight_files"]] != [p.name for p in shards]:
        raise AssertionError("weight_files does not list every shard on disk")
    for entry, path in zip(identity["weight_files"], shards):
        if entry["bytes"] != path.stat().st_size:
            raise AssertionError(f"{entry['name']}: recorded size {entry['bytes']}")
        if entry["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise AssertionError(f"{entry['name']}: recorded digest is not the file's digest")

    # The digest is load-bearing: a one-byte change must move it.
    scratch = root / "probe.bin"
    scratch.write_bytes(b"a")
    first = sha256_file(scratch)
    scratch.write_bytes(b"b")
    if sha256_file(scratch) == first:
        raise AssertionError("sha256_file returned the same digest for different bytes")
    scratch.unlink()

    return identity


def check_identity_guards(root):
    with tempfile.TemporaryDirectory() as empty:
        try:
            checkpoint_identity(empty)
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint_identity accepted a directory with no config.json")
        try:
            load_checkpoint_params(empty)
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError("load_checkpoint_params accepted an empty directory")

    # A config missing a required field is refused rather than silently defaulted.
    with tempfile.TemporaryDirectory() as broken:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        config.pop("num_key_value_heads")
        (Path(broken) / "config.json").write_text(json.dumps(config), encoding="utf-8")
        try:
            checkpoint_identity(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint_identity accepted a config missing num_key_value_heads")

    # Both published layouts of the rotary base are read, and neither is defaulted.
    flat = json.loads((root / "config.json").read_text(encoding="utf-8"))
    nested_value = rope_theta_of(flat)
    if nested_value != TINY["rope_theta"]:
        raise AssertionError(f"rope_theta read as {nested_value} from the written config")
    stripped = {key: value for key, value in flat.items()
                if key not in ("rope_parameters", "rope_scaling", "rope_theta")}
    if rope_theta_of(dict(stripped, rope_theta=5.0)) != 5.0:
        raise AssertionError("a top-level rope_theta was not honoured")
    try:
        rope_theta_of(stripped)
    except ValueError:
        pass
    else:
        raise AssertionError("rope_theta_of invented a default base frequency")


def check_weights_are_the_checkpoint_weights(model, root):
    params, config = load_checkpoint_params(root, dtype=torch.float64)
    state = model.state_dict()

    if int(config["num_hidden_layers"]) != len(params["layers"]):
        raise AssertionError("loaded layer count disagrees with the config")
    pairs = [("embed", "model.embed_tokens.weight"), ("final_norm", "model.norm.weight"),
             ("lm_head", "lm_head.weight")]
    for index in range(len(params["layers"])):
        base = f"model.layers.{index}."
        for key, suffix in (("input_norm", "input_layernorm.weight"),
                            ("q_proj", "self_attn.q_proj.weight"),
                            ("k_proj", "self_attn.k_proj.weight"),
                            ("v_proj", "self_attn.v_proj.weight"),
                            ("q_norm", "self_attn.q_norm.weight"),
                            ("k_norm", "self_attn.k_norm.weight"),
                            ("o_proj", "self_attn.o_proj.weight"),
                            ("post_attn_norm", "post_attention_layernorm.weight"),
                            ("gate_proj", "mlp.gate_proj.weight"),
                            ("up_proj", "mlp.up_proj.weight"),
                            ("down_proj", "mlp.down_proj.weight")):
            tensor = params["layers"][index][key]
            if tensor is None:
                raise AssertionError(f"layer {index} {key} was not loaded")
            reference = state[base + suffix].to(torch.float64)
            if not torch.equal(tensor, reference):
                raise AssertionError(f"layer {index} {key} differs from the checkpoint tensor")
    for key, name in pairs:
        if not torch.equal(params[key], state[name].to(torch.float64)):
            raise AssertionError(f"{key} differs from the checkpoint tensor {name}")
    return params, config


def check_forward_parity(model, params, config):
    tokens = [3, 17, 0, 9, 4, 22]
    ours, _ = model_logits(params, config, tokens)
    with torch.no_grad():
        theirs = model(input_ids=torch.tensor([tokens])).logits[0].to(torch.float64)

    if tuple(ours.shape) != tuple(theirs.shape):
        raise AssertionError(f"shape {tuple(ours.shape)} vs {tuple(theirs.shape)}")
    difference = float((ours - theirs).abs().max())
    scale = float(theirs.abs().max())
    if difference > PARITY_TOL * max(scale, 1.0):
        raise AssertionError(
            f"captured forward differs from transformers Qwen3ForCausalLM by {difference} "
            f"(logit scale {scale})")

    # Parity is not vacuous: the comparison moves when the input moves, and the
    # two implementations agree on the argmax that a greedy decode would take.
    moved = [3, 17, 1, 9, 4, 22]
    ours_moved, _ = model_logits(params, config, moved)
    if float((ours_moved - ours).abs().max()) < 1e-6:
        raise AssertionError("changing an input token did not change the logits")
    with torch.no_grad():
        theirs_moved = model(input_ids=torch.tensor([moved])).logits[0].to(torch.float64)
    if float((ours_moved - theirs_moved).abs().max()) > PARITY_TOL * max(scale, 1.0):
        raise AssertionError("the two implementations disagree on the perturbed sequence")
    if int(ours[-1].argmax()) != int(theirs[-1].argmax()):
        raise AssertionError("the two implementations disagree on the next-token argmax")

    # The cached decode path agrees with transformers at every generated position.
    logits, cache = model_logits(params, config, tokens)
    extended = list(tokens)
    for _ in range(3):
        token = int(logits[-1].argmax())
        extended.append(token)
        logits, cache = model_logits(params, config, [token], cache=cache)
        with torch.no_grad():
            reference = model(input_ids=torch.tensor([extended])).logits[0, -1].to(torch.float64)
        step = float((logits[-1] - reference).abs().max())
        if step > PARITY_TOL * max(scale, 1.0):
            raise AssertionError(
                f"cached decoding at length {len(extended)} differs from transformers by {step}")
    return difference


def check_released_checkpoint():
    """Pin the released checkpoint when the deployment host provides it.

    On the authoring sandbox the path is absent; the check reports that plainly
    instead of claiming a verification it did not perform.
    """
    try:
        present = (RELEASED_MODEL_PATH / "config.json").is_file()
    except OSError:
        # Unreadable from this account is the same as absent for this contract:
        # the identity is recorded on the deployment host that owns the weights.
        present = False
    if not present:
        return None
    identity = checkpoint_identity(RELEASED_MODEL_PATH)
    if identity["architectures"] != ["Qwen3ForCausalLM"]:
        raise AssertionError(
            f"released checkpoint declares architectures {identity['architectures']}")
    if not identity["weight_files"]:
        raise AssertionError("released checkpoint carries no weight shard")
    if identity["head_dim"] <= 0:
        raise AssertionError("released checkpoint has a non-positive head_dim")
    if float(identity["config_of_record"]["rope_theta"]) <= 0.0:
        raise AssertionError("released checkpoint has a non-positive rope_theta")
    return identity


def main():
    model = build_reference_model(seed=17091702)
    with tempfile.TemporaryDirectory() as directory:
        root = write_checkpoint(model, directory)
        check_identity_is_read_not_assumed(root)
        check_identity_guards(root)
        params, config = check_weights_are_the_checkpoint_weights(model, root)
        difference = check_forward_parity(model, params, config)

    identity = check_released_checkpoint()
    if identity is None:
        print(f"NOTE: {RELEASED_MODEL_PATH} is absent here; the released-checkpoint "
              "identity is recorded on the deployment host, not asserted from this sandbox")
    else:
        digests = ", ".join(f"{entry['name']}:{entry['sha256'][:12]}"
                            for entry in identity["weight_files"])
        print(f"released checkpoint pinned: config {identity['config_sha256'][:12]}, {digests}")

    print("PASS: checkpoint identity read from disk with independent digests, weights loaded "
          "byte-identically, and the captured forward agrees with transformers "
          f"Qwen3ForCausalLM to {difference:.2e} in full-prefix and cached decoding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
