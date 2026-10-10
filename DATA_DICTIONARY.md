# Data dictionary

All analyses are exploratory. A score row is a stored machine rating, not a validated measurement of human personality.

## Generation shards

`process/results/xlpt/<phase>/<checkpoint>/seed-*.jsonl` contains one generated scenario per row.

| Field | Meaning |
|---|---|
| `phase`, `setting` | Executed arm and neutral/compliance/high/low persona setting. |
| `scenario_id`, `seed`, `checkpoint_id` | Join keys for scenario, seed cluster and generator. Checkpoint capitalization follows historical records. |
| `persona_language`, `dialogue_language` | Persona instruction and requested dialogue language. |
| `customer_turns`, `agent_turns` | Authored customer stimuli and saved generated responses; email-like strings are masked in the publication copies. |
| `decoding`, `substream_seeds` | Recorded generation parameters and random-stream allocation. |
| `eligible_for_gate`, `eligible_for_analysis` | Historical eligibility flags; descriptive analyses do not change registered gates. |
| `*_sha256`, `prompt_hashes` | Original execution provenance. Public file digests are in `PROVENANCE.json`. |
| call/timing/token fields | Recorded run metadata, not a new runtime benchmark. |

Each high/low pole block is paired on seed and scenario. The main pole block has 23 seed clusters and 50 qualification scenarios. The four-cell Chinese-persona block lacks the gemma generator; it should not be interpreted as a complete four-generator design.

## Canonical rubric-score shards

`<phase>-rater/` and some `-rater-rescue/` directories contain trait/item ratings for generated scenarios. `source_scenario_id`, `seed`, `checkpoint_id` and `source_phase` join them to generation records.

| Field | Meaning |
|---|---|
| `trait`, `instrument_language` | Target rubric item set and language of the scoring instrument. |
| `scores` | Ordered item-ID to integer 1–7 or missing value; E1–E6 are Extraversion, A1–A6 Agreeableness. |
| `trait_score` | Complete item mean when present. A stored score is not imputed after a parser failure. |
| `rater_raw_by_item` | Stored model answer used by the score parser, retained for parse checks. |
| `schema_version`, `parse_policy`, `parse_failures`, `items_parsed` | Versioned parsing policy and missing-item provenance where the schema includes them. |
| original source/freeze/scorer hashes | Links to the historical generation, configuration and scoring artifacts. |

The current reducer re-applies the strict item parser and checks schema-3 missingness consistency. A null item is missing, not the scale minimum or a zero. Common-mask and worst-case analyses have separate estimands; do not mix their denominators.

## Supplementary score shards

`supp-20261004/*.jsonl` contains later rater, intervention or instrument-rendering observations. The filenames retain the historical namespace even where a file was completed later.

| Field | Meaning |
|---|---|
| `rater` | `mistral`, `llama` or `gemma`; the latter two also generated some transcripts. |
| `cell`, `pole`, `checkpoint_id`, `seed`, `source_scenario_id` | Design/join keys for generated-text analyses. |
| `variant` | Unedited text or the named text intervention. |
| `instrument_language`, `instrument_sha256` | Rubric wording language and original rendering identity. |
| `scores`, `scores_lenient`, `trait_scores` | Strict item scores, separately recorded lenient parses and complete trait means. Published reducers specify which set they use. |
| `status`, `parse_errors`, `max_prompt_tokens` | Scored/context-excluded state, parser failures and input-context size. |
| `code_sha256`, `transcript_sha256` | Original execution source/transcript provenance. |

The `s4-*-zh-r2/r3/r4` files hold additional Chinese renderings; `-b` files extend selected canonical-rater runs from three to six seeds. `SUPP_EXT_SUMMARY.json` records the all-rater/all-rendering common mask. No Chinese rendering in this snapshot was validated by a native human reader.

## Public-corpus numeric scores and metadata

`h-public/scores-*.jsonl` and `supp-20261004/pub-*.jsonl` contain numeric rater responses and original annotation mappings for FriendsPersona/CPED, with no script-turn text. `unit_id` is the join key; `identity` denotes a fictional character identity and `cluster` the corpus-specific grouping unit.

`h-public/unit-metadata-*.jsonl` contains one numeric metadata row per unit:

| Field | Meaning |
|---|---|
| `unit_id`, `corpus`, `identity`, `cluster` | Stable unit and grouping keys from the original extraction. |
| `labels` | Published high/low/unknown personality-label mapping. |
| `text_chars` | Sum of Unicode character counts of the original extracted script lines, before transcript formatting; used for the length baseline. |
| `turn_count` | Number of extracted source lines. |

There are 711 FriendsPersona units and 569 CPED units. Script lines, stage directions and original corpus files are omitted. Exact original unit-file digests, repository commits and input hashes are recorded in the corpus provenance. Label concordance on these out-of-domain television corpora does not validate or invalidate a rater for customer-support dialogue.

## Saved summaries and receipts

Uppercase `*_SUMMARY.json`, `XLPT_*.json` and related result files are archived outputs. CPU replay creates new summaries in a separate output directory. Receipt counts and original hashes describe completed historical score shards; their hashes can differ from sanitized publication file hashes. `PROVENANCE.json` and `SHA256SUMS` make the distinction explicit.

Bootstrap seeds and draw counts are recorded in each reducer/output. Replayed comparisons retain the seed/scenario clustering and masking rules used by the manuscript. All intervals are exploratory pointwise intervals under those recorded conditions.
