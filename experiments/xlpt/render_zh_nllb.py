#!/usr/bin/env python3
"""Render the frozen English XLPT-Support scenario packets and the v2 prompt settings into
Simplified Chinese with NLLB-200-3.3B, and screen every rendered field by back-translation.

Registered procedure (design card, translation slot): three independent forward engines with
back-translation screening, block-allocated over (scenario, seed) units (rule H). Only engine C
(NLLB-200) is reachable from the worker, so this renders ONE stratum. The deviation is recorded
in the screen file this script writes and must be carried into the freeze receipt and the paper.

Deterministic: beam search (num_beams=4, no sampling), fp32, fixed sentence splitter, fixed
batch order. Forward: eng_Latn -> zho_Hans. Screen: zho_Hans -> eng_Latn back-translation, chrF
(sacrebleu, default chrF2) of the back-translation against the source field. No chrF threshold
is registered, so nothing is filtered: every item is kept and low-scoring fields are listed
descriptively for the manuscript.

Writes into --out-dir (default data/xlpt-support/rendered_items_and_scenarios/):
  scenarios_zh.jsonl                    140 rows: zh fields + back-translation + chrF + provenance
  zh_b0_competence_scenarios_v1.jsonl   50 rows, packet-shaped (same keys/order as the English v1 packet)
  zh_b1_control_scenarios_v1.jsonl      90 rows, packet-shaped
  zh_settings_render_v1.json            rendered settings for the v3 unit, with screen + provenance
  zh_render_screen_v1.json              chrF summary, flagged fields, deviation note, provenance
The frozen packets are NOT written into process/consumption/frozen by this script; freezing is a
separate, receipted step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

ENGINE_ID = "facebook/nllb-200-3.3B"
ENGINE_REVISION = "1a07f7d195896b2114afcb79b7b57ab512e7b43e"
SRC_LANG = "eng_Latn"
TGT_LANG = "zho_Hans"
NUM_BEAMS = 4
BATCH = 8

EN_SPLIT = re.compile(r"(?<=[.!?])\s+")
ZH_SPLIT = re.compile(r"(?<=[。！？])")

PACKETS = (
    ("english_b0_competence_scenarios_v1.jsonl", "zh_b0_competence_scenarios_v1.jsonl", "QUALIFICATION", 50),
    ("english_b1_control_scenarios_v1.jsonl", "zh_b1_control_scenarios_v1.jsonl", "EVALUATION", 90),
)
PACKET_KEYS = ("scenario_id", "domain", "intent", "variant", "title", "context",
               "customer_turns", "resolution_space", "prereg_partition")
SCENARIO_TEXT_FIELDS = ("title", "context", "customer_turns", "resolution_space")

V2_CONFIG = "process/consumption/frozen/english_execution_v2.json"
RENDERED_SETTINGS = ("NEUTRAL", "EXTRAVERSION_HIGH", "EXTRAVERSION_LOW")
EN_INSTRUCTION = "Respond to the customer's request in English."
ZH_INSTRUCTION_EN = "Respond to the customer's request in Chinese."
FACTS_LABEL_EN = "Account and policy facts:"
# Descriptive flag only (no registered threshold): fields whose back-translation chrF falls
# below this are listed in the screen file so the paper can name them.
FLAG_CHRF = 50.0


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_en(text: str):
    return [s for s in EN_SPLIT.split(text.strip()) if s]


def split_zh(text: str):
    return [s for s in ZH_SPLIT.split(text.strip()) if s]


class Translator:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok_en = AutoTokenizer.from_pretrained(str(model_dir), src_lang=SRC_LANG, local_files_only=True)
        self.tok_zh = AutoTokenizer.from_pretrained(str(model_dir), src_lang=TGT_LANG, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float32, local_files_only=True
        ).to(device).eval()
        self.bos = {
            TGT_LANG: self.tok_en.convert_tokens_to_ids(TGT_LANG),
            SRC_LANG: self.tok_en.convert_tokens_to_ids(SRC_LANG),
        }
        self.calls = 0

    def _run(self, tokenizer, sentences, target: str):
        out = []
        for start in range(0, len(sentences), BATCH):
            chunk = sentences[start:start + BATCH]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            longest = int(enc["input_ids"].shape[1])
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.torch.inference_mode():
                ids = self.model.generate(
                    **enc,
                    forced_bos_token_id=self.bos[target],
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    max_new_tokens=min(400, longest * 3 + 20),
                )
            out.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
            self.calls += len(chunk)
        return [s.strip() for s in out]

    def en_to_zh(self, sentences):
        return self._run(self.tok_en, sentences, TGT_LANG)

    def zh_to_en(self, sentences):
        return self._run(self.tok_zh, sentences, SRC_LANG)


def render_field(tr: Translator, chrf, text: str) -> dict:
    """Translate one English field sentence-by-sentence and screen it by back-translation."""
    src_sents = split_en(text)
    zh_sents = tr.en_to_zh(src_sents)
    zh = "".join(zh_sents)
    back_sents = tr.zh_to_en(split_zh(zh) or [zh])
    back = " ".join(back_sents)
    return {
        "source": text,
        "zh": zh,
        "back_translation": back,
        "chrf": round(chrf.sentence_score(back, [text]).score, 2),
        "source_sentences": src_sents,
        "zh_sentences": zh_sents,
    }


def render_scenario(tr, chrf, row: dict) -> dict:
    rendered = {
        "title": render_field(tr, chrf, row["title"]),
        "context": render_field(tr, chrf, row["context"]),
        "customer_turns": [render_field(tr, chrf, t) for t in row["customer_turns"]],
        "resolution_space": [render_field(tr, chrf, t) for t in row["resolution_space"]],
    }
    scores = [rendered["title"]["chrf"], rendered["context"]["chrf"]]
    scores += [r["chrf"] for r in rendered["customer_turns"]]
    scores += [r["chrf"] for r in rendered["resolution_space"]]
    return rendered, scores


def packet_row(row: dict, rendered: dict) -> dict:
    out = {}
    for key in PACKET_KEYS:
        if key == "title":
            out[key] = rendered["title"]["zh"]
        elif key == "context":
            out[key] = rendered["context"]["zh"]
        elif key == "customer_turns":
            out[key] = [r["zh"] for r in rendered["customer_turns"]]
        elif key == "resolution_space":
            out[key] = [r["zh"] for r in rendered["resolution_space"]]
        else:
            out[key] = row[key]
    return out


def dzh_setting(text: str) -> str:
    if text.count(EN_INSTRUCTION) != 1 or not text.endswith(EN_INSTRUCTION):
        raise ValueError("setting text does not end with the single registered English instruction")
    return text[: -len(EN_INSTRUCTION)] + ZH_INSTRUCTION_EN


def summarize(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "min": values[0],
        "p10": values[max(0, int(0.10 * n) - 1)],
        "median": values[n // 2] if n % 2 else round((values[n // 2 - 1] + values[n // 2]) / 2, 2),
        "mean": round(sum(values) / n, 2),
        "max": values[-1],
    }


def main(args) -> int:
    import sacrebleu
    import torch
    import transformers
    from sacrebleu.metrics import CHRF

    project = Path(args.project).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_dir = project / args.out_dir
    frozen = project / "process/consumption/frozen"
    for name, _, _, _ in PACKETS:
        if not (frozen / name).is_file():
            raise FileNotFoundError(frozen / name)
    config = json.loads((project / V2_CONFIG).read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN":
        raise ValueError("v2 prompt config is not FROZEN")
    for shard in ("pytorch_model-00001-of-00003.bin", "pytorch_model-00002-of-00003.bin",
                  "pytorch_model-00003-of-00003.bin", "sentencepiece.bpe.model", "config.json"):
        if not (model_dir / shard).is_file():
            raise FileNotFoundError(model_dir / shard)
    if list(model_dir.rglob("*.incomplete")):
        raise ValueError("engine directory still holds .incomplete blobs")

    chrf = CHRF()
    started = time.time()
    tr = Translator(model_dir, device=args.device)
    provenance = {
        "engine_id": ENGINE_ID,
        "engine_revision": ENGINE_REVISION,
        "engine_stratum": "C",
        "engine_dir": str(model_dir),
        "engine_index_sha256": sha256_file(model_dir / "pytorch_model.bin.index.json"),
        "engine_config_sha256": sha256_file(model_dir / "config.json"),
        "direction_forward": f"{SRC_LANG}->{TGT_LANG}",
        "direction_screen": f"{TGT_LANG}->{SRC_LANG}",
        "decoding": {"num_beams": NUM_BEAMS, "do_sample": False, "dtype": "float32",
                     "batch": BATCH, "max_new_tokens_rule": "min(400, 3*source_tokens+20)"},
        "sentence_split_en": EN_SPLIT.pattern,
        "sentence_split_zh": ZH_SPLIT.pattern,
        # sacrebleu's own signature object refuses to render before the metric has scored
        # anything (it needs num_refs), so the settings are read off the instance instead.
        "screen_metric": (
            f"sacrebleu {sacrebleu.__version__} chrF "
            f"(char_order={chrf.char_order}, word_order={chrf.word_order}, beta={chrf.beta}, "
            f"whitespace={chrf.whitespace}, eps_smoothing={chrf.eps_smoothing}, "
            f"lowercase={chrf.lowercase})"
        ),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "device": args.device,
        "source_prompt_config": V2_CONFIG,
        "source_prompt_config_sha256": sha256_file(project / V2_CONFIG),
        "source_packets": {name: sha256_file(frozen / name) for name, _, _, _ in PACKETS},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }

    scenario_rows, all_scores, flagged = [], [], []
    per_kind = {"title": [], "context": [], "customer_turns": [], "resolution_space": []}
    for src_name, dst_name, partition, limit in PACKETS:
        rows = read_jsonl(frozen / src_name)
        if len(rows) != limit or any(r["prereg_partition"] != partition for r in rows):
            raise ValueError(f"{src_name}: expected {limit} {partition} rows")
        packet = []
        for index, row in enumerate(rows):
            rendered, scores = render_scenario(tr, chrf, row)
            all_scores.extend(scores)
            per_kind["title"].append(rendered["title"]["chrf"])
            per_kind["context"].append(rendered["context"]["chrf"])
            per_kind["customer_turns"].extend(r["chrf"] for r in rendered["customer_turns"])
            per_kind["resolution_space"].extend(r["chrf"] for r in rendered["resolution_space"])
            for kind in SCENARIO_TEXT_FIELDS:
                entries = rendered[kind] if isinstance(rendered[kind], list) else [rendered[kind]]
                for pos, entry in enumerate(entries):
                    if entry["chrf"] < FLAG_CHRF:
                        flagged.append({"scenario_id": row["scenario_id"], "field": kind, "index": pos,
                                        "chrf": entry["chrf"], "source": entry["source"],
                                        "zh": entry["zh"], "back_translation": entry["back_translation"]})
            scenario_rows.append({
                "scenario_id": row["scenario_id"],
                "domain": row["domain"],
                "intent": row["intent"],
                "variant": row["variant"],
                "prereg_partition": partition,
                "source_packet": src_name,
                "source_index": index,
                "language": "zh",
                "rendering": rendered,
                "chrf_min": min(scores),
                "chrf_mean": round(sum(scores) / len(scores), 2),
                "provenance": {"engine_id": ENGINE_ID, "engine_revision": ENGINE_REVISION,
                               "engine_stratum": "C", "num_beams": NUM_BEAMS},
            })
            packet.append(packet_row(row, rendered))
            print(f"{partition} {index + 1}/{limit} {row['scenario_id']} chrf_min={min(scores):.1f}",
                  flush=True)
        atomic_jsonl(out_dir / dst_name, packet)

    settings = {}
    for name in RENDERED_SETTINGS:
        en_text = config["settings"][name]
        dzh_text = dzh_setting(en_text)
        pzh = render_field(tr, chrf, dzh_text)
        settings[f"{name}_DZH"] = {"text": dzh_text, "persona_language": "English",
                                   "dialogue_language": "Chinese", "derived_from": name,
                                   "derivation": "final-sentence language instruction swapped, no translation"}
        settings[f"{name}_PZH"] = {"text": pzh["zh"], "persona_language": "Chinese",
                                   "dialogue_language": "Chinese", "derived_from": f"{name}_DZH",
                                   "derivation": "NLLB eng_Latn->zho_Hans rendering of the _DZH text",
                                   "back_translation": pzh["back_translation"], "chrf": pzh["chrf"],
                                   "source_sentences": pzh["source_sentences"],
                                   "zh_sentences": pzh["zh_sentences"]}
    facts = render_field(tr, chrf, FACTS_LABEL_EN)
    provenance["elapsed_seconds"] = round(time.time() - started, 1)
    provenance["engine_calls"] = tr.calls
    settings_doc = {
        "version": 1,
        "purpose": "Chinese-surface settings for the v3 execution unit (P=en/D=zh and P=zh/D=zh).",
        "english_instruction": EN_INSTRUCTION,
        "chinese_instruction_in_english": ZH_INSTRUCTION_EN,
        "facts_label": {"en": FACTS_LABEL_EN, "zh": facts["zh"],
                        "back_translation": facts["back_translation"], "chrf": facts["chrf"]},
        "settings": settings,
        "provenance": provenance,
    }
    atomic_jsonl(out_dir / "scenarios_zh.jsonl", scenario_rows)
    atomic_write(out_dir / "zh_settings_render_v1.json",
                 json.dumps(settings_doc, ensure_ascii=False, indent=2) + "\n")
    screen = {
        "version": 1,
        "deviation": (
            "Registered translation slot: three independent forward engines with back-translation "
            "screening, block-allocated over (scenario, seed) units (rule H). Only engine C "
            "(NLLB-200-3.3B) was reachable from the worker, so this is a SINGLE-STRATUM rendering; "
            "rule H's block allocation collapses to one variant for every unit. No chrF threshold "
            "is registered; nothing was filtered or hand-edited, and low-scoring fields are listed "
            "below for descriptive reporting."
        ),
        "scenarios_rendered": len(scenario_rows),
        "fields_rendered": len(all_scores),
        "chrf_all_fields": summarize(all_scores),
        "chrf_by_field": {kind: summarize(vals) for kind, vals in per_kind.items()},
        "chrf_settings": {name: entry["chrf"] for name, entry in settings.items() if "chrf" in entry},
        "chrf_facts_label": facts["chrf"],
        "flag_rule": f"descriptive only: field chrF < {FLAG_CHRF}",
        "flagged_fields": flagged,
        "outputs": {
            name: sha256_file(out_dir / name)
            for name in ("scenarios_zh.jsonl", "zh_b0_competence_scenarios_v1.jsonl",
                         "zh_b1_control_scenarios_v1.jsonl", "zh_settings_render_v1.json")
        },
        "provenance": provenance,
    }
    atomic_write(out_dir / "zh_render_screen_v1.json", json.dumps(screen, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"ok": True, "scenarios": len(scenario_rows), "fields": len(all_scores),
                      "chrf": screen["chrf_all_fields"], "flagged": len(flagged),
                      "elapsed_seconds": provenance["elapsed_seconds"]}, ensure_ascii=False))
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default="data/xlpt-support/rendered_items_and_scenarios")
    ap.add_argument("--device", default="cuda")
    return ap


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        raise
