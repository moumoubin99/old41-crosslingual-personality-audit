# Block H: annotation protocol (v2)

> **SUPERSEDED 2026-09-29, not run.** The user's instruction 「全部换成公开数据之类的，不要增加人工」
> replaced human annotation with public human-labelled corpora; see `H_public_spec.md`. No rater is recruited
> and this protocol is kept only as a record.

**Status: PROTOCOL v2, 2026-09-29, Codex-reviewed. Raters are not yet recruited; recruiting them is the user's
task.**

- v2 applies every protocol edit from the Codex review
  (`<LOCAL_PATH>/review.md`, DECISION 2).
- The sample is fixed by `H_sampling_spec.md` v2. Scores are entered on `H_scoring_form.md` v2.
- **Standing.** H is a post-preregistration descriptive precursor to G2. It is not G2, and it is not a new
  preregistration.
  - It was specified after the preregistered results were known. The paper is explicitly exploratory (user
    decision (b), 2026-09-28).
  - It admits nothing, meets no gate, and writes nothing to `QUALIFICATION_GATE.results`.

## Purpose

1. Check whether the rater model's Extraversion scores track human judgement on the pole transcripts, separately
   for English and Chinese dialogues.
2. Measure human pole separation per dialogue language.
   - Raters are **not shown** pole or assignment. The pole can still be inferred from behaviour, and that is what
     the item scores are meant to register.

## Agreeableness items: not included

- `agreeableness_items_included: false` is recorded in `H_SAMPLE_MANIFEST.json`.
- **Reason.** A1–A6 do not serve either purpose above. They would roughly double the scoring burden and add
  fatigue.
- **Consequence.** H provides **no human Agreeableness discriminant evidence**, and nothing in the paper may cite
  H for discriminant validity.

## Raters

Every rater must be bilingual.

| Rater | Profile |
|---|---|
| R1 | Native or dominant, fully literate reader of Simplified Chinese; English reading at least CEFR C1 (or documented equivalent) |
| R2 | Native or dominant English reader; Mandarin / Simplified-Chinese reading at least C1-equivalent |
| R3 | The same bilingual minimum; native or dominant in either language. Needed only under the third-rater rule |

- **Independence.** Raters must not be project authors. They must not have seen the persona prompts, the pole
  design, the hypotheses or any result.
- **What is stored.** Only de-identified language dominance, the qualification basis, and signed
  conflict-of-interest and exposure attestations. No name or contact detail is written to project files.
- **Package.** R1 and R2 each rate all 96 transcripts, mixed-language, in independently randomised orders (see
  the sampling spec).
- **Workload.** 96 × 4–6 minutes = **6.4–9.6 hours per rater**, before practice and breaks.
  - Budget about 7–10 hours of rater time.
  - Sessions last at most 90 minutes.

## Instrument

- **Every transcript is rated with the English items**, whichever language the dialogue is in.
  - This matches the rater model, which uses the same fixed English instrument on every row (R16-F1).
  - It also avoids an unvalidated rubric translation.
- **Source of the item wording.**
  - The rater model's runtime loads its items from `refine-logs/RUBRIC_BEHAVIOR.md`.
  - The same wording is recorded in `data/xlpt-support/rendered_items_and_scenarios/rubric_items_en.jsonl`.
  - At build time, the E1–E6 wording parsed from `RUBRIC_BEHAVIOR.md` is checked against the JSONL, and the sha256
    of both files is recorded in the manifest. If the wording differs, the build stops.
- **Scale.** Each item takes an integer from 1 to 7. Anchors are given at 1, 4 and 7; 2, 3, 5 and 6 are
  in-between values.
- **Administration difference (disclosed).** The item wording is shared, but the rater model receives one item
  per request, while human raters see all six items together. Human–model differences may partly reflect this
  difference in administration.

| Item | Facet | Item text | 1 | 4 | 7 |
|---|---|---|---|---|---|
| E1 | sociability | Initiates social contact with the customer beyond the task minimum | Answers only what is asked; no greeting or closing beyond the template | Greets and closes warmly, one brief social remark | Opens and sustains social contact throughout, several unforced personal remarks |
| E2 | sociability | Sustains conversational engagement across turns | Terse, transactional replies; lets the exchange lapse | Keeps the exchange going with occasional follow-up questions | Actively keeps the customer talking, invites elaboration in most turns |
| E3 | assertiveness | States a recommendation and stands behind it | Defers entirely; offers no recommendation | Recommends once, hedged | Recommends clearly, repeats the recommendation when the customer wavers |
| E4 | assertiveness | Takes the lead in structuring the resolution | Follows the customer's lead throughout | Proposes the next step about half the time | Sets the agenda, names the next step in nearly every turn |
| E5 | energy level | Conveys enthusiasm and pace in the wording | Flat, minimal phrasing | Moderately lively; some positive wording | Highly energetic phrasing, exclamations, visible momentum |
| E6 | energy level | Volunteers extra help or options unprompted | Never volunteers anything beyond the request | Volunteers one extra option | Repeatedly volunteers additional options and next steps |

## Instructions given to raters (verbatim)

> You will read short customer-support conversations between a customer and a support agent. Some conversations
> are in English and some in Chinese. Rate **only the agent's behaviour as written**. Do not rate whether the
> agent solved the problem correctly, and do not rate the customer.
>
> For each conversation, give every item a whole number from 1 to 7 using the anchors. 2, 3, 5 and 6 are
> in-between values. Rate each item on its own; a high score on one item does not imply a high score on
> another. Judge Chinese conversations by the same behavioural standard as English ones. For example, a warm
> greeting counts as social contact in either language, and conventional politeness formulas count only as
> much as their English equivalents would.
>
> Tick "language mismatch" if **most of the agent's turns** (two or more of the three) are written mainly in a
> language other than the customer's. Rate the behaviour anyway. If a conversation is unreadable or empty, tick
> "cannot rate" and leave the items blank.
>
> Apart from the practice conversations, work alone and do not discuss any conversation with anyone until all
> your scores have been submitted. Take breaks. Do not go back to change earlier conversations after moving on.

### Calibration

- **Practice set.** Raters first rate **4 practice transcripts**: 2 in English and 2 in Chinese.
  - They are drawn deterministically and score-blind, as set out in the sampling spec.
  - None of them is among the 96.
- **The only discussion allowed.** Discussing the practice transcripts is the sole exception to the no-discussion
  rule.
  - The discussion may only clarify the existing anchors. Anchors are not changed, and the rater model's scores
    are never shown.
- **After calibration**, the 96 are rated with no further discussion. Practice scores are not analysed.

## Scoring

- **Human trait score.** For each rater and transcript, the mean of E1–E6.
- **Criterion.** Rules are applied in this order:
  1. **Both primaries tick cannot_rate.** The transcript is excluded without R3.
  2. **Exactly one primary ticks cannot_rate.** R3 rates the transcript. The criterion is the mean of the two
     valid six-item means (the valid primary and R3). If R3 also ticks cannot_rate, fewer than two valid ratings
     exist and the transcript is excluded.
  3. **|mean(R1) − mean(R2)| > 1.0.** R3 rates the transcript. The criterion is the median of the three six-item
     means. If R3 ticks cannot_rate, the criterion is the mean of R1 and R2, and the event is counted.
  4. **Otherwise.** The criterion is the mean of R1 and R2.
- **Language-mismatch flag.**
  - Operational definition: the rater instruction above (the majority of agent turns are mainly not in the
    customer's language).
  - R1 and R2 each flag every transcript independently.
  - Concordant flags stand.
  - Discordant flags send the transcript to R3. R3's flag decides by majority. If R3 also rates the transcript
    under rule 2 or 3, the same R3 rating is used for both purposes.
  - Every transcript stays in the primary analysis, mismatched or not (ITT). Resolved-mismatch transcripts are
    also reported separately.
- **Exclusions.** Excluded transcripts are counted and reported per cell.
- **R3 blinding.**
  - R3 sees neither R1's nor R2's scores or flags, nor the reason a transcript was referred.
  - R3's package is built only after the R1 and R2 forms and flags are locked.

## Pre-specified descriptive analysis

All of the following are descriptive. None is gated, and no threshold is registered.

**Intervals.** All intervals come from a percentile bootstrap with 10,000 replicates and seed 20260928. Within
each checkpoint, it resamples the 8 shared seed blocks with replacement. Each block carries its 4 transcripts
(en/zh × high/low) together.

1. **Human agreement, per *D*.**
   - Absolute-agreement, two-way random-effects **ICC(2,1)** and **ICC(2,2)** on the E mean, R1 versus R2;
   - six item-specific quadratic-weighted κ (E1–E6);
   - the share of transcripts referred to R3, by reason;
   - intervals for every statistic.
2. **Rater model versus humans, per *D*.**
   - (a) **Raw** Pearson and Spearman correlations between the rater model's six-item E `trait_score` and the
     human criterion.
     - These are **confounded by pole and checkpoint**: both scores differ between cells.
     - They are reported as such, not as transcript-level tracking.
   - (b) **Checkpoint × pole-adjusted association.** Both scores are centred within each (checkpoint, pole) cell,
     and the Pearson correlation of the centred scores is taken. Only (b) may be read as transcript-level
     tracking.
   - The rater-model score for D = zh transcripts comes from block A, so this row waits for A.
3. **Pole separation, per *D* and checkpoint.**
   - sep_human = mean(high) − mean(low) of the criterion;
   - shown beside the rater model's sep on the same transcripts.
4. **Language contrast.**
   - The en − zh difference in 2(b) and in 3.
   - It is not a test of instrument invariance. Humans and the rater model both use the English instrument, so
     the contrast speaks only to whether the rater model tracks human judgement equally on the two dialogue
     surfaces.

### Limits stated with every H number

- **Size.** n = 96, far below G2's 720.
- **Coverage.** Three checkpoints and one trait.
- **Prior exposure.** The material is qualification-seed material that the project has already observed.
- **Rater pool.** The raters are bilingual (not native-monolingual) readers.
- **Administration.** Items are administered differently to humans and to the rater model (see Instrument).
- **No Agreeableness items.** There is no human discriminant evidence.
- **Standing.** H is a post-preregistration descriptive block. It **does not establish evaluator validity** and
  admits nothing.

## Data handling

- **What raters receive.** The rater instructions, the item and anchor sheet, the 4 practice transcripts, and
  their own package (presentation IDs and transcripts only). Nothing else.
- **Blinding key.** The operator keeps it until every score is locked, R3's included.
- **Returned forms.**
  - Each returned form is validated against `H_scoring_form.md` and then locked.
  - Its sha256 is recorded in a separate `H_ANNOTATION_RECEIPT.json`. The build manifest is never modified.
- **Rater identity.** Nothing beyond R1, R2, R3 and the de-identified profile fields above is stored in project
  files.
