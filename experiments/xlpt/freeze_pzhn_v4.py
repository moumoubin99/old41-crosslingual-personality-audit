#!/usr/bin/env python3
"""Freeze the reviewed Chinese persona strings (``*_PZHN_*``) into execution unit v4.

v4 is v3 plus six new settings taken verbatim from ``refine-logs/followup/P_zh_persona_draft.md`` (Codex review
2026-09-29).  Every v3 field and setting string is carried byte-identical; v1-v3 and their receipts are not touched.
The strings are checked mechanically before anything is written: NFC, full-width punctuation, the 97-code-point
persona parity between the poles, and D = zh / D = en variants differing only in the final response-language
sentence.  Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

V3 = "process/consumption/frozen/english_execution_v3.json"
V3_RECEIPT = "process/consumption/frozen/english_execution_receipt_v3.json"
V4 = "process/consumption/frozen/english_execution_v4.json"
V4_RECEIPT = "process/consumption/frozen/english_execution_receipt_v4.json"
DRAFT = "refine-logs/followup/P_zh_persona_draft.md"
DESIGN = "refine-logs/design_card.json"
REVIEW = Path("<LOCAL_PATH>/review.md")

NAMES = ("EXTRAVERSION_HIGH_PZHN_DZH", "EXTRAVERSION_LOW_PZHN_DZH", "NEUTRAL_PZHN_DZH",
         "EXTRAVERSION_HIGH_PZHN_DEN", "EXTRAVERSION_LOW_PZHN_DEN", "NEUTRAL_PZHN_DEN")
SHARED = "遵循所提供的账户和政策事实。不要声称进行了事实不允许的操作。"
LANG = {"DZH": "用中文回复客户的请求。", "DEN": "用英文回复客户的请求。"}
PERSONA_LEN = 97
FULL_DZH_LEN = 138


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_signed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in payload.items() if k != "self_sha256"}
    if payload.get("status") != "FROZEN" or payload.get("self_sha256") != canonical_hash(unsigned):
        raise SystemExit(f"{path} is not a valid FROZEN self-hashed file")
    return payload


def extract(draft: str) -> dict:
    strings = {}
    for name in NAMES:
        match = re.search(r"^### `" + name + r"`[^\n]*\n\n```\n(.*?)\n```$", draft, re.M | re.S)
        if not match:
            raise SystemExit(f"{name} not found as a fenced block in {DRAFT}")
        strings[name] = match.group(1)
    return strings


def check(strings: dict) -> dict:
    for name, text in strings.items():
        if unicodedata.normalize("NFC", text) != text or "\n" in text or text != text.strip():
            raise SystemExit(f"{name}: not a single NFC line")
        if re.search(r"[.,;:!?]", text):
            raise SystemExit(f"{name}: ASCII punctuation present")
    counts = {}
    for d in ("DZH", "DEN"):
        neutral = strings[f"NEUTRAL_PZHN_{d}"]
        if neutral != SHARED + LANG[d]:
            raise SystemExit(f"NEUTRAL_PZHN_{d} is not the shared sentences plus the {d} language sentence")
        for pole in ("HIGH", "LOW"):
            text = strings[f"EXTRAVERSION_{pole}_PZHN_{d}"]
            if not text.endswith(neutral):
                raise SystemExit(f"EXTRAVERSION_{pole}_PZHN_{d} does not end with NEUTRAL_PZHN_{d}")
            persona = text[: -len(neutral)]
            if len(persona) != PERSONA_LEN:
                raise SystemExit(f"EXTRAVERSION_{pole}_PZHN_{d}: persona is {len(persona)} code points, not {PERSONA_LEN}")
            if d == "DZH" and len(text) != FULL_DZH_LEN:
                raise SystemExit(f"EXTRAVERSION_{pole}_PZHN_DZH is {len(text)} code points, not {FULL_DZH_LEN}")
            if strings[f"EXTRAVERSION_{pole}_PZHN_DEN"][: -len(LANG['DEN'])] != strings[f"EXTRAVERSION_{pole}_PZHN_DZH"][: -len(LANG['DZH'])]:
                raise SystemExit(f"{pole}: DZH and DEN differ outside the final language sentence")
            counts[f"EXTRAVERSION_{pole}_PZHN_{d}"] = {"persona_code_points": len(persona), "total_code_points": len(text)}
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    for rel in (V4, V4_RECEIPT):
        if (project / rel).exists():
            raise SystemExit(f"{rel} already exists; frozen files are never overwritten")

    v3 = load_signed(project / V3)
    v3_receipt = load_signed(project / V3_RECEIPT)
    for rel, expected in v3_receipt["files"].items():
        if file_sha(project / rel) != expected:
            raise SystemExit(f"v3 receipt file changed: {rel}")
    strings = extract((project / DRAFT).read_text(encoding="utf-8"))
    counts = check(strings)
    clash = set(strings) & set(v3["settings"])
    if clash:
        raise SystemExit(f"new identifiers collide with v3 settings: {sorted(clash)}")

    v4 = {k: v for k, v in v3.items() if k != "self_sha256"}
    v4["version"] = 4
    v4["scope"] = ("Adds reviewed Chinese persona wording (P = zh) to the version 3 execution unit: the Extraversion "
                   "poles and neutral prompt as *_PZHN_* settings, each with a Chinese (D = zh) and an English (D = en) "
                   "response-language variant; every version 3 setting is carried byte-identical")
    v4["extends"] = V3
    v4["settings"] = dict(v3["settings"]) | strings
    v4["pzhn_rendering_provenance"] = {
        "method": "single reviewed rendering of the v2 English poles, clause by clause; not machine-translated at run time",
        "reviewer": "Codex gpt-5.6-sol, reasoning effort xhigh, 2026-09-29 (model reviewer, not a native human reader)",
        "review_path": str(REVIEW),
        "review_sha256": file_sha(REVIEW) if REVIEW.is_file() else None,
        "source_text": DRAFT,
        "source_text_sha256_at_freeze": file_sha(project / DRAFT),
        "source_prompt_config": "process/consumption/frozen/english_execution_v2.json",
        "counting_convention": "NFC code points; each Han character and punctuation mark counts as one",
        "length_checks": counts,
        "variant_rule": "D = zh and D = en variants differ only in the final response-language sentence",
        "facts_label": "D = zh keeps facts_label.zh byte-identical; D = en keeps the v2 English scenario rendering",
        "reserved_names": "The v3 *_PZH* settings stay the frozen, never-run NLLB renderings; *_PZHN_* is the reviewed text",
    }
    v4["limitations"] = list(v3["limitations"]) + [
        "The *_PZHN_* persona text is a SINGLE reviewed rendering, not the card's three-engine translation ensemble. It was reviewed by a model reviewer whose secondary cross-check was same-family, not by a native human reader; any P effect is specific to this rendering.",
        "The Chinese scenario surface under *_PZHN_*_DZH is still the single-stratum NLLB rendering frozen in v3, with its limitations.",
        "No phase, queue entry or shard uses a *_PZHN_* setting at freeze time. Launching block BC remains a separate operator decision.",
    ]
    v4["self_sha256"] = canonical_hash(v4)

    receipt = {k: v for k, v in v3_receipt.items() if k != "self_sha256"}
    receipt["version"] = 4
    receipt["scope"] = ("Chinese persona surface (P = zh) with reviewed *_PZHN_* wording, for D = zh and D = en; "
                        "everything frozen by receipt v3 is carried unchanged")
    receipt["design_card_sha256"] = file_sha(project / DESIGN)
    receipt["prompt_config_self_sha256"] = v4["self_sha256"]
    v4_text = json.dumps(v4, ensure_ascii=False, indent=2) + "\n"
    files = {V4: hashlib.sha256(v4_text.encode("utf-8")).hexdigest()}
    files |= {rel: sha for rel, sha in v3_receipt["files"].items() if rel != V3}
    receipt["files"] = files
    receipt["limitations"] = [x for x in v3_receipt["limitations"]
                              if not x.startswith("This receipt governs the zh phases only")] + [
        "This receipt governs *_PZHN_* phases only. Every shard already generated stays bound to receipt v1, v2 or v3, all unchanged.",
        "The *_PZHN_* text is a single model-reviewed rendering; see pzhn_rendering_provenance in the config.",
    ]
    receipt["self_sha256"] = canonical_hash(receipt)

    with open(project / V4, "x", encoding="utf-8") as fh:
        fh.write(v4_text)
    with open(project / V4_RECEIPT, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"config": V4, "config_self_sha256": v4["self_sha256"], "receipt": V4_RECEIPT,
                      "receipt_self_sha256": receipt["self_sha256"], "length_checks": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
