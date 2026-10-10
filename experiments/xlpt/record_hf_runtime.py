#!/usr/bin/env python3
"""Write a host-local runtime manifest that hash-binds one HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


HASH_NAMES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_model_files(model_path: Path) -> dict[str, str]:
    files = {}
    for name in HASH_NAMES:
        path = model_path / name
        if path.is_file():
            files[name] = sha256_file(path)
    for path in sorted(model_path.glob("model-*.safetensors")):
        files[path.name] = sha256_file(path)
    if not any(name.startswith("model-") and name.endswith(".safetensors") for name in files):
        single = model_path / "model.safetensors"
        if single.is_file():
            files[single.name] = sha256_file(single)
    if "config.json" not in files:
        raise ValueError(f"checkpoint missing config.json: {model_path}")
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError(f"checkpoint missing safetensors weights: {model_path}")
    return files


def run(args) -> int:
    model_path = Path(args.model_path).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint_id": args.model_id,
        "model_path": str(model_path),
        "worker": Path("/etc/hostname").read_text(encoding="utf-8").strip() if Path("/etc/hostname").exists() else "",
        "interpreter": args.python,
        "package_versions": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "model_files": collect_model_files(model_path),
        "model_forward": "transformers.AutoModelForCausalLM eager bf16",
        "attention_implementation": "hf-eager",
        "prefix_caching": False,
        "note": "Host-local binding for English B0 remaining checkpoints. Not a formula PASS.",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "files": len(payload["model_files"])}, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default="<LOCAL_PATH>/python3.12")
    return ap


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
