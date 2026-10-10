# Block H: sampling specification (v2)

> **SUPERSEDED 2026-09-29, not run.** Block H now uses public human-labelled corpora (`H_public_spec.md`).
> `H_package/` stays read-only and unused.

**Status: SPEC v2, 2026-09-29, Codex-reviewed. Drawn 2026-09-29** by `experiments/xlpt/build_h_package.py` into
`refine-logs/followup/H_package/` (read-only).

- **Selections.** 96 rated and 4 practice transcripts; no seed skipped; 44 candidate collisions logged.
- **Practice pool.** The QUALIFICATION fallback, because the phases have no TUNING records.
- **Source checks.** All 184 English b4 generation shards match their `source_shard_sha256`.
- **Wording check.** E1–E6 item text and anchors are byte-identical; facet labels differ only in capitalisation.
- **Manifest.** Sha256 in `H_SAMPLE_MANIFEST.json` → `spec_sha256`. That value is the file's hash before this status
  line was edited.

- v2 applies every sampling edit from the Codex review
  (`<LOCAL_PATH>/review.md`, DECISION 2).
- The draw is fixed here, before anyone looks at the selected transcripts, and it uses no score.
- The package is built once the English generation shards have been copied from bjb1 (see "Source shards").
- Whatever builds it must reproduce this spec exactly and record the outputs listed at the end.

## Frame

| Factor | Levels |
|---|---|
| Checkpoint *c* | `Llama-3.1-8B-Instruct`, `aya-expanse-8b`, `qwen3-8b`. gemma is excluded: it failed the prior zh competence screen (`ZH_G1_COMPETENCE_SCREEN.json`). |
| Dialogue language *D* | `en`, `zh` (P = en in both) |
| Pole *k* | `high`, `low` |
| Seed cluster *s* | the 23 retained qualification seeds 20260811–20260834, excluding 20260829 |

- **Size.** 3 × 2 × 2 = **12 cells**, 8 transcripts per cell, **96 transcripts** in all.
- **Shared seeds.** Within a checkpoint, all four cells use the **same 8 seed clusters**. Each checkpoint
  therefore contributes 8 seed blocks of 4 transcripts, and the bootstrap in the protocol resamples those blocks.

## Source shards (read only; none is modified)

- **D = zh.** `process/results/xlpt/zh-b4-extraversion-{high,low}/<c>/seed-<s>.jsonl`, which are local
  generation shards. Each record carries `customer_turns[3]` and `agent_turns[3]`.
- **D = en.** The English b4 generation shards (phase `b4-extraversion-{high,low}`).
  - Only the rater shards are local (`process/results/xlpt/b4-extraversion-{high,low}-rater/<c>/`).
  - The generation shards must be copied read-only from bjb1.
  - Each copied shard must match the `source_shard_sha256` carried by its rater records. On any mismatch, stop
    and do not substitute.

## Candidate units

A candidate is one generation record (*c*, *D*, *k*, `scenario_id`, *s*) that meets both conditions:

1. `eligible_for_analysis == true` and `partition == "QUALIFICATION"`.
2. It is present as generated.
   - **No filter** is applied for response language, length, refusal or rater score (ITT).
   - No response-language field is derived or stored. The analysis uses the target *D* and the humans'
     language-mismatch flags.

## Hash serialisation (used by every step below)

- **Fields.** Joined with a literal `|`, in the order given. Their forms:
  - *c*: the exact checkpoint directory name, as listed in Frame;
  - *D*: `en` or `zh`;
  - *k*: `high` or `low`;
  - *s*: a decimal integer with no padding;
  - `scenario_id`: the exact string from the record.
- **Encoding.** The string is encoded as UTF-8 with no BOM and hashed with SHA-256.
- **Comparison.** The digest is compared as a lower-case 64-character hex string. Ascending order means ascending
  string order, which is the same as numeric order.
- **Ties.** A tie is impossible in practice. If one occurs, the build stops.

## Draw (deterministic, score-blind)

### Step 1: seed blocks, per checkpoint *c*

1. **Qualifying seeds.** A seed qualifies only if **each** of the four cells (en,high), (en,low), (zh,high) and
   (zh,low) has at least one candidate for it.
2. **Ranking.** Rank the 23 seeds by `sha256("XLPT-H-v2-seed|" + c + "|" + str(s))`.
3. **Selection.** Walk the ranking and keep the first 8 qualifying seeds.
   - Every seed skipped as non-qualifying is logged.
   - If fewer than 8 seeds qualify, the build stops.

### Step 2: one record per seed block and cell

1. **Order of work.** Within checkpoint *c*, visit the 8 selected seeds in their Step 1 rank order. For each
   seed, visit the cells in the order (en,high), (en,low), (zh,high), (zh,low).
2. **Ranking.** For each (seed, cell), rank **every** candidate of that seed and cell by
   `sha256("XLPT-H-v2|" + c + "|" + D + "|" + k + "|" + str(s) + "|" + scenario_id)`.
3. **Selection.** Take the first candidate whose `scenario_id` is **not yet used anywhere in checkpoint *c***.
   - Every skipped candidate is logged, with the `scenario_id` it collided with.
   - This traversal handles collisions recursively. A replacement that itself collides is skipped in turn.
   - If the ranked list is exhausted, the build stops. No substitution comes from another seed or cell.
4. **Effect.**
   - A rater never sees one scenario twice for the same model. Seeing it twice could reveal the pole by
     contrast.
   - Repeats across checkpoints are allowed.

The draw reads only `scenario_id`, seed, checkpoint, phase and the eligibility fields. It reads no score from the
English rater shards or from block A.

## Practice set (4 transcripts, score-blind)

- **Pool.** For each *D*, the practice pool is the eligible `partition == "TUNING"` records of the same three
  checkpoints and phases.
  - If a *D* has no TUNING records, its pool is instead the QUALIFICATION candidates whose `scenario_id` is not
    used anywhere among the 96.
  - In both cases, a record whose `scenario_id` appears among the 96 is excluded.
- **Selection.** For each *D*, take the pool's first `high` record and first `low` record by
  `sha256("XLPT-H-v2-practice|" + D + "|" + c + "|" + k + "|" + str(s) + "|" + scenario_id)`.
  - This gives 2 en and 2 zh practice transcripts, each language covering both poles.
  - Which pool was used for each *D* is recorded.
- **Display.** Practice transcripts are shown in the same format as the 96. They are marked as practice, but
  their poles are not revealed.

## Blinding and display

- **Presentation ID.** `"H" + first 10 hex of sha256(salt_hex + "|" + c + "|" + D + "|" + k + "|" + scenario_id + "|" + str(s))`.
  - `salt` is 32 random bytes generated once at build time. It enters the hash as 64 lower-case hex characters
    and is stored **only** in the key file.
  - The build **asserts that all 96 presentation IDs, together with the 4 practice IDs, are unique**. If they are
    not, it stops.
- **Per-rater order.** Sort by `sha256(salt_hex + "|" + rater_id + "|" + presentation_id)`. Each rater has a
  different, mixed-language order.
- **What raters see.** The presentation ID, then the three customer turns and three agent turns interleaved as
  `CUSTOMER: …` / `AGENT: …`.
  - This is exactly the transcript the frozen scorer (`experiments/xlpt/score_rater.py`, `transcript()`) sends to
    the rater model.
  - The labels are the same for Chinese dialogues. **No facts block is shown**, because the scorer shows none.
- **Removed before distribution.** Checkpoint, model path, setting, phase, persona text, seed, substream seeds,
  every hash, and the `scenario_id`.
- **Key file** (`H_BLINDING_KEY.json`). The operator holds it; raters never receive it.
  - It contains the salt.
  - For each presentation ID it records *c*, *D*, *k*, *s*, `scenario_id`, and the source shard path with its
    sha256.

## Outputs of the build (immutable once written)

- **Rater packages.** `H_PACKAGE_R1.md` and `H_PACKAGE_R2.md`, each holding all 96 in its own rater order.
- **Practice.** `H_PRACTICE.md`, the 4 practice transcripts.
- **Key.** `H_BLINDING_KEY.json`.
- **Manifest.** `H_SAMPLE_MANIFEST.json`, which contains no salt. It records:
  - the spec version and the sha256 of this spec, the protocol and the form;
  - `agreeableness_items_included: false`;
  - the sha256 of `refine-logs/RUBRIC_BEHAVIOR.md` and of `rubric_items_en.jsonl`, with the result of the E1–E6
    wording check;
  - the source-shard hashes;
  - the 8 selected seeds per checkpoint and the log of skipped seeds;
  - the 96 selections and the log of candidate skips;
  - the practice selections and the pool used for each language;
  - the result of the presentation-ID uniqueness check.
- **R3 package (later).** `H_PACKAGE_R3.md` is built only after the R1 and R2 forms and flags are locked. It
  holds only the referred IDs, in an R3-specific order, and carries no referral reason.
- **Annotation receipt (later).** `H_ANNOTATION_RECEIPT.json` records the sha256 of every returned form. The
  manifest itself is never modified.

## What the sample supports

It supports **descriptive** statements only:

- the rater-model–human association per *D*, both raw and checkpoint × pole-adjusted, with seed-block bootstrap
  intervals;
- human pole separation per *D*.

The sample is far below G2's 720 double-annotated transcripts. It admits nothing, meets no gate, and writes
nothing to `QUALIFICATION_GATE.results`.
