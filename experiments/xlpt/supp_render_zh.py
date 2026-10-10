"""Further machine renderings of the six Extraversion rubric items into Chinese.

Extension of the supplementary round supp-20261004 (2026-10-06). The round had one
unvalidated Chinese rendering (supp_rubric_zh.json, here R1). This script asks open
models that are already on the host to translate the same English items, one field at
a time and with greedy decoding, so the Chinese-rubric condition can be repeated with
renderings that share no author with R1. Nothing here is checked by a person: the
only checks are mechanical (a non-empty answer in Han script, and a machine
back-translation that is stored for the reader, not scored). The request head and the
three structural labels of the Chinese request stay those of R1, so the items and
anchors are the only thing that differs between renderings.

Usage (CPU is enough, the texts are a few words long):
  python -m experiments.xlpt.supp_render_zh --project . --models-root <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from experiments.xlpt.score_rater import rubric_items

FIELDS = ("facet", "prompt", "a1", "a4", "a7")
INSTRUCTION = ("Translate the following English text into Simplified Chinese. "
               "Output only the translation, with no explanation.\n\n")
# rendering id -> (model directory, kind)
RENDERERS = {
    "r2": ("Qwen__Qwen3-8B", "chat"),
    "r3": ("CohereLabs__aya-expanse-8b", "chat"),
    "r4": ("facebook__nllb-200-3.3B", "nllb"),
}
BACK = "facebook__nllb-200-3.3B"
HAN = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = text.splitlines()[0].strip() if text else text
    return text.strip("\"'“”‘’ ").rstrip("。.")


def chat_translate(model_dir: Path, texts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.eval()
    extra = {"enable_thinking": False} if "Qwen3" in model_dir.name else {}
    out = []
    for text in texts:
        ids = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + text}],
                                      add_generation_prompt=True, return_tensors="pt", **extra)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=96, do_sample=False)
        out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    del model
    return out


def nllb_translate(model_dir: Path, texts, src: str, tgt: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, src_lang=src)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    out = []
    for text in texts:
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                 max_new_tokens=96, num_beams=1, do_sample=False)
        out.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    del model
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--only", default=",".join(RENDERERS))
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    project, root = Path(args.project), Path(args.models_root)
    rubric = project / "refine-logs/RUBRIC_BEHAVIOR.md"
    english = rubric_items(rubric, "extraversion")
    flat = [(item[0], field, item[1 + k]) for item in english for k, field in enumerate(FIELDS)]
    # Identical English strings (the facet names) are translated once per rendering.
    unique = list(dict.fromkeys(text for _, _, text in flat))

    for rid in args.only.split(","):
        name, kind = RENDERERS[rid]
        out_path = project / f"experiments/xlpt/supp_rubric_zh_{rid}.json"
        if out_path.exists():
            print(json.dumps({"rendering": rid, "status": "exists"}), flush=True)
            continue
        started = time.time()
        if kind == "chat":
            raw = chat_translate(root / name, unique)
        else:
            raw = nllb_translate(root / name, unique, "eng_Latn", "zho_Hans")
        zh = {text: clean(r) for text, r in zip(unique, raw)}
        back = dict(zip(unique, nllb_translate(root / BACK, [zh[t] for t in unique],
                                               "zho_Hans", "eng_Latn")))
        items, checks = [], []
        for item in english:
            row = {"id": item[0]}
            for k, field in enumerate(FIELDS):
                source = item[1 + k]
                row[field] = zh[source]
                checks.append({"id": item[0], "field": field, "english": source,
                               "chinese": zh[source], "raw": raw[unique.index(source)],
                               "back_translation_nllb": back[source],
                               "has_han": bool(HAN.search(zh[source])),
                               "has_latin_letters": bool(LATIN.search(zh[source]))})
            items.append(row)
        payload = {
            "note": ("Machine rendering %s of the six project-authored Extraversion rubric items, "
                     "made for the 2026-10-06 extension of the supplementary round. Not checked by "
                     "a person, not back-translated by a person, not a validated instrument." % rid),
            "rendering": rid, "renderer": name, "renderer_kind": kind,
            "instruction": INSTRUCTION if kind == "chat" else "eng_Latn -> zho_Hans",
            "decoding": "greedy, max_new_tokens 96, one field per request",
            "source_sha256": sha256_file(rubric),
            "code_sha256": sha256_file(Path(__file__)),
            "all_fields_have_han": all(c["has_han"] for c in checks),
            "fields_with_latin_letters": sum(c["has_latin_letters"] for c in checks),
            "seconds": round(time.time() - started, 1),
            "items": items, "checks": checks,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
        print(json.dumps({"rendering": rid, "renderer": name,
                          "all_fields_have_han": payload["all_fields_have_han"],
                          "seconds": payload["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
