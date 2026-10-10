# Not yet rendered: the self-report item pool and the target-language surfaces

This directory is the contamination scan's `test_path`
(`design_card.json` → `pico.controls.contamination_scan.test_path` =
`data/xlpt-support/rendered_items_and_scenarios/*.jsonl`). Two of the four surfaces that
scan must eventually cover are present; two are not. This file records exactly which, and why,
so that no absence is mistaken for a clean scan.

## Present

| File | Records | Content |
|---|---|---|
| `scenarios_en.jsonl` | 180 | the frozen XLPT-Support scenarios, English source surface, each tagged with its rule-E partition |
| `rubric_items_en.jsonl` | 12 | the rater facet items of `RUBRIC_BEHAVIOR.md`, prompt plus anchors at 1 / 4 / 7 |

Both are produced by `tools/render_items_and_scenarios.py`, which is deterministic and whose
`--check` mode reproduces them byte for byte.

## Absent — 1. the self-report item pool (BFI-2, 60 items; TIPI, 10 items)

The design card registers `BFI-2 translated item pool` (Soto and John 2017 item set, 60 items, with
TIPI as a convergent check) as the self-report instrument, and names the memorised-normative-response
risk as precisely what the 13-gram scan exists to detect. **The verbatim item text of both
instruments is therefore load-bearing: the scan is a scan of that exact wording.**

The item text is not present in this repository, and it could not be retrieved from this host — the
network egress proxy blocks the authoritative distribution page
(`www.colby.edu/academics_cs/departments/psychology/bfi2/`, `EGRESS_BLOCKED`).

**The items have not been reconstructed from memory, and must not be.** BFI-2 and TIPI are
third-party published instruments; an approximate reconstruction would (a) change the measuring
instrument while leaving its name unchanged, and (b) make the contamination scan meaningless, since
the scan would then compare invented 13-grams against the pretraining proxies rather than the
published ones it is supposed to test. A near-miss here is worse than an acknowledged gap.

**What unblocks it:** the verbatim item text supplied from the authoritative source — the Soto &
John (2017) BFI-2 item table (or the official BFI-2 site) and the Gosling, Rentfrow & Swann (2003)
TIPI item list — placed under `data/xlpt-support/instruments/` and rendered by extending
`tools/render_items_and_scenarios.py`. Either an unblocked egress route to those sources, or the
files provided directly, is sufficient.

## Absent — 2. the target-language renderings

The card renders `context`, `customer_turns` and `resolution_space`, plus the self-report items, into
six languages (English, Chinese, Spanish, German, Arabic, Swahili). Under the registered procedure
that rendering is produced by the **pinned three-engine translation ensemble** — Google Cloud
Translation v3, Microsoft Azure AI Translator v3.0 and `facebook/nllb-200-3.3B` — with the
target-language length band calibrated on the TUNING partition only, at M2, and never re-fitted.

Hand-translating these would substitute an unregistered translation source for the registered one
and would silently break the translation-ensemble control that the falsifier depends on, so no
target-language text is written here. Google and Azure require API credentials that are not
configured on this host; NLLB-200-3.3B is a local HF model and is the part that can run on the
pinned hardware.

The rubric is deliberately **not** in this second list: under R16-F1(a) the rater instrument is
English in every cell, so `rubric_items_en.jsonl` is complete as it stands and acquires no
target-language counterpart.
