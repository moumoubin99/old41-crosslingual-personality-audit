# Follow-up experiment plan — 2026-09-28 (rev. 2, after Codex review)

Status: **RUNNING (updated 2026-09-29).**

* Server 41 is on. A0 passed, and A is running: 10/184 at 14:33Z.
* BC (276 generation + 276 rater jobs) is chained after A, and H on public data is chained after BC.
* The server powers down only after every block is complete and pulled.
* This plan adds no confirmatory claim and runs no registered gate.

Every block is a **post-preregistration descriptive analysis** on the qualification partition, which has
23/24 seed clusters (s20260829 is not retried). Blocks may resemble a registered gate, but they borrow no
gate label, threshold verdict or pass/fail language.

Authority: `refine-logs/design_card.json` governs. This file schedules work and amends nothing.
Review: gpt-5.6-sol xhigh scored rev. 1 at 4/10. Its findings and the changes they led to are listed in §6.

## 0. Where the evidence stands

| Cell (P = persona language, D = dialogue language) | Generated | Rater-scored (English instrument) |
|---|---|---|
| P=en, D=en (b4 poles) | yes, 4 checkpoints × 23 clusters | yes — `B4_POLE_RATER_SUMMARY.json` |
| P=en, D=zh (zh-b4 poles) | yes, 184/184 | **no** |
| P=zh, D=zh | no. v3 `_PZH` is frozen and queued, but its persona text is a raw, disfluent NLLB rendering | no |
| P=zh, D=en | no. The cell does not exist in any frozen unit | no |

Gates G2–G5 are all unmet (manuscript §3 table) and remain unmet after this plan.

## 1. Blocks

| Block | Work | Reading |
|---|---|---|
| **A** | Rater-score zh-b4 poles: 8 phases of `zh_surface_rater_queue.json`, 184 jobs | sep(en,zh) per checkpoint and the D simple effect sep(en,en) − sep(en,zh). Descriptive, and runtime-matched within checkpoint |
| **D** | Chinese rendering of the behavior rubric; re-score 46 zh transcripts per checkpoint under it | A within-transcript instrument-language difference (English − Chinese). A descriptive analogue of G3, **not** G3 |
| **H** | *Replaced 2026-09-29:* the frozen rater scores public dialogue with the corpora's own human Big-Five labels (FriendsPersona en, CPED zh). No human raters, no new annotation. Spec: `refine-logs/followup/H_public_spec.md` | Descriptive label concordance of a frozen support-oriented score on two public scripted-dialogue corpora; **not** validation of the rater. Does its E score rank human-labelled high-Extraversion speakers above low ones, and more for E than for A? This is **not** G2 and **not** a criterion on our own transcripts |
| **P** | Native-reviewed Chinese persona text for both P = zh cells | Removes the generator-side disfluency confound before any P = zh generation |
| **BC** | One indivisible block: P=zh × D∈{zh,en} Extraversion poles on the 3 pre-selected checkpoints, then rater scoring | Completes an exploratory en/zh 2×2. The card sets L = 2 to VOID, so this is **never** H1 or Delta_mis_bar evidence |

**Dropped: rev. 1 block E** (round-trip-MT "perturbation pairs"). The two operations it compared,
en→fr→en and zh→en→zh, are different interventions. Round-tripping agent turns can also erase or create the
cues being rated, and chrF overlap does not match semantic or style damage across languages.

A faithful G4 needs behavior-preserving, severity-calibrated perturbations validated blind by bilingual
humans. That is deferred (§4). The manuscript §8 item 4 already says G4 is unmet and would not test
behavioral sensitivity even if run.

**Scale disclosure:** throughout, sep = mean(high) − mean(low) = 2·E*, where E* = ½(high − low) is the card's
pole-aligned separation. sep is never compared with card thresholds such as f_diag = 0.18.

## 2. Block specifications

### A — zh-b4 rater scoring (queue ready)

* **Slice.** The 8 `zh-b4-{high,low}-rater-<ckpt>` phases, unchanged scorer, frozen English rubric,
  per-item greedy requests, prompt date frozen at 2026-09-18. Both rubric traits are scored as the queue
  already does. The analysed score is the **own-setting Extraversion trait mean**, and Agreeableness is
  reported separately as a discriminant check. The two traits are never averaged.
* **A0 pre-flight (server CPU, minutes).** Tokenize every zh-b4 item request with the rater tokenizer
  against `RATER_MAX_MODEL_LEN = 4096`.
  * Overflow is plausible for gemma: its high-pole turns are 1 193 chars/turn longer than its low-pole turns.
  * **Stop rule:** on any overflow, do not truncate and do not raise the context. Report the counts per
    checkpoint × pole and ask the user.
* **gemma** is scored like the other checkpoints, because its shards exist. It is reported as an execution
  failure case (it failed the zh competence screen), not as a transfer estimate.
* **Reducer.** A new sibling of `summarize_pole_rater.py`.
  * Primary interval: a **joint paired seed-cluster bootstrap** that resamples the 23 seed clusters jointly
    across the b4 and zh-b4 cells of a checkpoint, with a fixed seed (20260928) and 9 999 draws, percentile
    interval.
  * It also reports the normal-approximation interval that `B4_POLE_RATER_SUMMARY.json` uses, for comparability.
  * Per-item blocks are reported as in manuscript §5.3.
* **Cost.** The English b4 rater shards took ≈ 12.3 GPU-h for 184 jobs. With longer Chinese inputs, budget
  **≈ 22 GPU-h**.
* **Parse failures.** Cells that crash on the frozen item parser are rescued under the dated amendment
  `strict_item_missing_v1` (after §2 BC).

### D — Chinese rubric rendering and within-transcript re-scoring

* **Rendering.** Same engine and decoding as the v3 unit: NLLB-200-3.3B, stratum C, beam 4, fp32,
  eng_Latn→zho_Hans.
  * Inputs: the 6 Extraversion item rows (item, facet, three anchors) and the fixed request frame.
  * The zho_Hans→eng_Latn chrF screen is descriptive only.
  * Nothing is hand-edited or re-rendered on the basis of the screen, and no BFI-2/TIPI text is involved.
  * Output: `rubric_items_zh_v1.jsonl` plus a hashed receipt, frozen before any Chinese-instrument score.
  * Disclosed: one engine, not the three forward translations the card registers for its instrument channel.
* **Scorer.** Add an `--instrument-language Chinese` path to `run_rater_scoring.py`.
  * It swaps only the item table and the frame, and records `instrument_language: "Chinese"`.
  * A contract test proves the English path is byte-identical.
* **Sample.** Per zh checkpoint, **46 transcripts**: 23 clusters × one high-pole and one low-pole transcript.
  They are drawn by a hashed seed stream before any Chinese-instrument score exists. Their English-instrument
  scores come from A.
* **Statistic.**
  * Per checkpoint, the mean within-transcript difference (English − Chinese instrument) with a
    seed-cluster bootstrap interval.
  * The same difference taken separately on the high and low poles, and therefore on sep.
  * No TOST verdict, no ι = 0.10 pass/fail and no q*. The number is reported against the registered ι only as
    a scale reference.
  * English cells are not re-scored: both instruments are identical bytes under greedy decoding.
* **Reading.** A large difference **narrows claim scope**. Every D-contrast stays an operational
  English-instrument rater contrast. A small difference does not establish calibration or equal sensitivity;
  G2 and the language-dependent scoring slope stay open.
* **Cost.** ≈ 184 transcripts × 6 items ≈ **1–2 GPU-h**, plus ≈ 0.5 h of NLLB.
* **Dependency.** Runs after A.

### H — public human-annotated data (replaces the human-rating subset, 2026-09-29)

* **Decision.** The user, 2026-09-29: 「全部换成公开数据之类的，不要增加人工」. The human-rating subset
  below is **superseded and not run**, and no rater is recruited. Its package stays as a read-only record.
* **Spec.** `refine-logs/followup/H_public_spec.md` rev. 2. Codex round 1 raised 2 CRITICAL and 4 MAJOR
  findings, all fixed in rev. 2. Rounds 2 and 3 (spec plus code) and their fixes are recorded in the spec's §8.
* **Data.**
  * FriendsPersona: 711 scene units. The target character's own turns are `AGENT`, everyone else is
    `CUSTOMER`, and stage directions are unroled lines.
  * CPED: 569 dialogue units from 287 identities in 32 franchise components, at most K = 2 dialogues per
    identity by salted hash.
  * Labels are the corpora's own human annotations. The builder is `experiments/xlpt/score_public_personality.py`
    `--build-only`, which is deterministic and hash-bound in `H_PUBLIC_PROVENANCE.json`.
* **Rater.** The frozen Mistral rater with the main study's exact per-item request, decoding and 4096-token
  guard. A unit that overflows is excluded, never truncated. All 12 E + A items are scored, and Friends gets an
  extra E-only pass without stage directions. That is ≤ 19,626 calls, ≈ 2.6 GPU-h in 2 jobs
  (`h_public_queue.json`).
* **Reading.** `experiments/xlpt/reduce_h_public.py`:
  * the E→EXT AUC:
    * Friends: the primary is the pair-weighted within-identity AUC, with a crossed identity × scene
      bootstrap;
    * CPED: the primary is the identity-level AUC over identity means, with a franchise-component bootstrap;
  * the 2×2 trait-specificity matrix with its difference-in-differences;
  * per-item AUCs;
  * missingness by label, item and cluster, and the maximum prompt-token count.
  * Before anything is reduced, `--validate-only` checks each shard fail-closed against the verified units
    and the current code hashes. `chain_h_public.sh` requires this check to pass before it records completion.
* **Standing.** This is descriptive label concordance of a frozen support-oriented score on two public
  scripted-dialogue corpora. It is **not** validation of the rater, and it is not G2.
  * Every record carries `eligible_for_gate = false`, and nothing is written to `QUALIFICATION_GATE.results`.
  * Limitations stated with every H number:
    * The rubric is support-oriented and the dialogue is scripted.
    * Friends labels are scene-level; CPED labels are character-level.
    * 12,511 of the 12,565 informative Friends EXT pairs (99.57%) come from the six principal characters.
* **Order.** Runs after BC on the same single 4090 (`chain_h_public.sh`). The power-off is held by
  `.hold_poweroff` until the H shards are pulled and hash-checked.

### H (superseded 2026-09-29, not run) — blinded bilingual human-rating subset

* **Sample.** 96 transcripts: per checkpoint (excluding gemma), 32 transcripts, i.e. {en, zh} × {high, low} ×
  8, stratified by seed cluster with a hashed draw.
  * *Corrected 2026-09-28:* rev. 2 read "16 … × 4", which gives 48, not the stated 96. The total of 96 is kept.
* **Package.** The transcripts in random order, stripped of every arm, checkpoint and condition field. Also:
  the 6 Extraversion items with anchors, a scoring form, and a blinding key held separately.
  * **Changed 2026-09-28 (local build):** raters use the **English** items for both en and zh transcripts. This
    matches the rater model's fixed English instrument (R16-F1) and removes H's dependence on D's NLLB rubric
    rendering, which needs the GPU.
  * Built documents: `refine-logs/followup/H_annotation_protocol.md`, `H_scoring_form.md` and
    `H_sampling_spec.md`.
* **Raters.** Two independent bilingual raters. A third rater scores any transcript whose two six-item means
  differ by more than 1.0, and the criterion is then the median. Recruitment is the user's.
* **Reading.** Rater–human correlation per language with a bootstrap interval. It is descriptive, and its
  size is far below G2's 720.
* **v2, 2026-09-29 (Codex decisions, delegated by the user; review in
  `<LOCAL_PATH>/review.md`).**
  * **No Agreeableness items.** `agreeableness_items_included: false`, so H gives no human A discriminant
    evidence.
  * **Rater profile.**
    * R1: native Simplified-Chinese reader, English ≥ C1.
    * R2: native English reader, Chinese ≥ C1.
    * R3: the same bilingual minimum.
    * Only de-identified profile and attestation fields are stored.
  * **Standing.** H is post-preregistration and descriptive. Raters are *not shown* pole or assignment.
  * **Workload.** 6.4–9.6 h per rater.
  * **Draw.**
    * The 8 seed clusters are shared across the 4 cells of each checkpoint, and the bootstrap resamples these
      seed blocks jointly.
    * Collisions are handled by a full hash-ranked traversal that logs every skip, and the build stops if a list
      is exhausted.
    * The hash serialisation is exact, and presentation-ID uniqueness is asserted.
  * **Rater input.**
    * Transcript-only `CUSTOMER:`/`AGENT:` turns and no facts block, exactly as `score_rater.py` sends them.
    * Each rater gets one combined package of all 96 in their own order (`H_PACKAGE_R1/R2.md`).
    * The 4 practice transcripts (2 en and 2 zh) are hashed and score-blind, and they are the only material that
      may be discussed.
  * **Missingness and flags.**
    * Explicit missingness and cannot-rate rules.
    * The language-mismatch flag is operationalised, discordant flags go to R3, and all transcripts are kept as
      ITT.
  * **Statistics.**
    * Absolute-agreement ICC(2,1) and ICC(2,2), and six item-level quadratic κ.
    * Raw correlations are labelled pole/checkpoint-confounded. The checkpoint × pole-centred association is the
      transcript-level reading.
  * **Provenance.** The runtime rubric `RUBRIC_BEHAVIOR.md` and the JSONL are both hashed, and the E1–E6 wording
    is checked. The one-item-per-request administration difference is disclosed.
  * **Receipts.** The build manifest is immutable. Returned-form hashes go to `H_ANNOTATION_RECEIPT.json`.

### P — native-reviewed Chinese persona (prerequisite of BC)

* The v3 `_PZH` persona strings are raw NLLB output and visibly disfluent (e.g. "你是一个支持的代理人谁是保留").
* A fluent Chinese rendering of the two v2 pole personas and the facts/constraint sentences is drafted (the
  project routes terminology/style edits through Codex). The **user, as a native reader, reviews and approves**
  it. It is then frozen before any generation.
* Only the final response-language sentence differs between the D = zh and D = en variants.
* Disclosed: a human-reviewed single translation, not the card's translation ensemble.
* **Draft, 2026-09-28:** `refine-logs/followup/P_zh_persona_draft.md`. The draft is not frozen. It notes that the
  NLLB `EXTRAVERSION_LOW_PZH` string drops the "Keep each reply short and plain" clause, so the v3 P = zh LOW
  pole lost content as well as fluency.
* **Final text, 2026-09-29 (Codex decision, delegated by the user).** The four decision points were resolved as
  follows:
  * 「不加多余的问候」;
  * 「主导处理过程，推动问题解决」;
  * 「进行了……操作」;
  * exact length parity: the persona portions are 97/97 characters and the full D = zh strings 138/138 (NFC code
    points, rechecked).
* The facts label `账户和政策事实:` is kept byte-identical.
* **Canonical identifiers.** The new strings are `*_PZHN_*`. The v3 `*_PZH*` names stay with the frozen, never-run
  NLLB strings.
* Acceptance is provisional: the reviewer's secondary check was same-family. The next step is the v4 freeze with a
  receipt. The BC Tier-2 go-ahead remains the user's.

### BC — P = zh poles, both dialogue languages, one indivisible block

* **Unit.** New frozen unit `english_execution_v4.json` extending v3 with
  `EXTRAVERSION_{HIGH,LOW}_PZHN_DZH` (Chinese scenarios v1) and `EXTRAVERSION_{HIGH,LOW}_PZHN_DEN` (English
  scenarios as in v2).
  * Both use the P text.
  * The receipt, contract tests and `run_rater_scoring.py` source entries are frozen before generation.
  * The v3 `_PZH` phases are not run.
* **Checkpoints.** Qwen, aya and Llama, pre-selected by the **earlier, independent zh competence screen**
  (`ZH_G1_COMPETENCE_SCREEN.json`, "23-cluster screen-positive"). gemma is excluded on that same prior screen.
* **No outcome-dependent attrition.** Every generated BC output is rater-scored as produced, and the analysis
  is intention-to-treat. The response-language rate on BC outputs is reported. Compliance-conditioned results
  appear only as a sensitivity analysis.
* **B and C run back-to-back with no stop between them.** A half 2×2 is not reported as a result.
* **Reading.** The exploratory interaction [sep(en,en) + sep(zh,zh)] − [sep(en,zh) + sep(zh,en)] per
  checkpoint, with the joint seed-cluster bootstrap from A.
  * Disclosed scope: one trait (Extraversion), one persona rendering, one scenario translation engine,
    23 clusters, and qualification material already observed.
* **Cost.** 276 generation jobs × 12.7 min ≈ 58 h, plus 276 rater jobs ≈ 30 h, for **≈ 88 GPU-h**.

### Amendment 2026-09-30 — rater parse policy `strict_item_missing_v1` (post-failure, dated)

*Written after the failure it responds to, and labelled as such. Decided by Codex (gpt-5.6-sol xhigh,
option B, delegated by the user); no confirmation is self-signed.*

* **The failure.** In block A, 4 of the 23 `zh-b4-high-rater-qwen3-8b` cells (seeds 20260822, 20260823,
  20260828, 20260832) stopped with the frozen parser's error `rater must return exactly one integer`. The
  greedy rater output was, for example, `'�7'`. `run_rater_scoring.py` aborts the whole shard on one
  such item, so these cells have no shard; the queue marks them `stuck` after one attempt. As of
  2026-09-29 16:17 UTC every other finished A cell (21) is complete, and no other checkpoint or pole has failed.
* **Policy (new scorer path, frozen scorer unchanged).** `run_rater_item_missing.py` wraps the frozen
  scorer. It uses the same v3 freeze, engine, chat template, prompt date, decoding and the unchanged
  `parse_item_score` (`INTEGER.fullmatch`). On a strict-parse failure:
  * the item is `null`, and a `parse_failures` record keeps the item ID, the exact decoded text, the parser
    input, the error and the generated token IDs;
  * the item is not retried, and the remaining items are still scored;
  * `trait_score` is `null` unless all six items of that trait parse: no imputation, no 5-of-6 mean.
  Rows are `schema_version` 3 with `parse_policy = strict_item_missing_v1`, `items_parsed`,
  `score_namespace` and `rescue_run`. Nothing strips leading characters, and generation is not constrained.
* **A rescue.** After the A queue manager exits, `build_a_rescue_queue.py --select` schedules only cells
  that are not completed, have no canonical and no rescue shard, and whose log shows the frozen parser's
  error. The cell must also be in state `stuck` and have a scheduled block-A seed; 20260829 is never
  selected. Each such cell gets one `run_rater_item_missing --namespace rescue` run; its shard goes to
  `{score_dir}-rescue/` with `rescue_run = true`. The A queue, its state and its logs are not modified, and
  the selection with its reasons is written to `a_rescue_selection.json`. An empty selection removes any
  older rescue queue. The rescue runs in `chain_h_public.sh` step 4b, only after H has been validated and
  `.h_chain_done` written, so it cannot delay or block H. It runs under a hard `timeout` (default 4 h),
  leftover rescue scorers are killed, and `.a_rescue_done` records the selected count, the rescue shards
  on disk and the manager's exit status.
* **Single writer per cell.** Canonical and rescue scoring of one cell share an exclusive, non-blocking
  cell lock. The shard is published with a no-replace link, so a shard written by another run during
  scoring is never overwritten.
* **BC.** All 276 BC rater jobs use `run_rater_item_missing --namespace canonical` from job 1, so the
  policy is uniform across the block.
* **H.** `score_public_personality.py` already applies the same rule (per-item parse errors recorded, not
  retried, a trait mean only with all six items; `H_public_spec.md` §4). H is bound to this amendment.
* **Reduction** (`summarize_pole_rater.py`, summary `schema_version` 2):
  * A cell has exactly one shard across the canonical and rescue directories. A collision is refused, and so
    is a rescue row that is not schema 3 or whose namespace fields disagree with its directory.
  * The cell key is the phase, the case-folded checkpoint directory and the seed. The file name must match
    the row seed, and one directory holds a single `checkpoint_id`, so spelling or seed-metadata variants
    cannot enter as a second cell.
  * BC phases accept only schema-3 canonical rows; a BC rescue directory is refused.
  * Descriptives use complete trait scores.
  * Both paired contrasts use common-complete units: a (seed, scenario) enters a trait's high − low
    contrast only when both poles have that trait's score, and the seed mean is taken over those same
    scenarios on both sides. With no incomplete trait this equals the earlier reduction: the English b4
    summary reproduces exactly.
  * Reported: parse failures and incomplete traits in total and per shard, missingness per pole cell, the
    matched/common-complete/dropped denominators per checkpoint × trait, and the list of rescue cells.
* **Reading.** Missingness can be language-conditioned and informative, and sensitivity analyses cannot
  rule that out. If failures are appreciable or differ by cell, the A and BC contrasts are qualified as
  instrument-fragile. Permitted sensitivities: 5-of-6 available-item means; deleting only U+FFFD and then
  reapplying the strict parser; bounds that set missing items to 1 and 7.
* **Tests.** `test_rater_item_missing.py` checks strict-parser invariance, parity on valid output, failure
  containment, 6-of-6 propagation, path immutability, collision refusal, and the cell lock with
  no-replace publication. `test_pole_summary_contract.py` (cases 12–19) checks the schema-3 reduction,
  rescue acceptance, and refusal of collisions, namespace mismatches, unnamed nulls, checkpoint-spelling
  and seed-metadata variants, and non-schema-3 or rescue rows in BC. `test_a_rescue_selection.py` checks
  the selection: only stuck parse-error cells with a scheduled seed and no shard, never 20260829, and
  removal of a stale queue on an empty selection.
* **Code review (gpt-5.6-sol xhigh) 5/10, not ready → fixes.** (1) Selection now requires `stuck` and a
  scheduled seed. (2) Cell lock and no-replace publication. (3) Case-folded cell key, file-name/row seed
  check, one checkpoint per directory. (4) BC schema 3 enforced. (5) Empty selection clears stale
  queue files, and the chain reads the count from `a_rescue_selection.json`. The rescue also moved after
  H (H review round 4).
* **Combined review round 5 (not ready) → fixes.** (1) `chain_h_public.sh` is now the sole power-off
  owner. It stops the waiting `chain_a_bc.sh` once BC is done and waits for both pull
  acknowledgements. It shuts down only through a fail-closed `guard_idle`, which refuses on a queue
  manager, a rescue scorer, a failing `nvidia-smi` or any GPU process. (2) The summarizer re-verifies
  schema-3 rows with `check_item_missing_evidence`: frozen re-parse, exact failure records, and
  lowercase-hex scorer digests that are uniform across shards and equal to the files on disk.
* **Combined review round 6 (5/10, not ready) → fixes.** (1) H takes over only if `.bc_chain_done`
  records exactly `bc_generation=276 bc_scores=276`, and only once `chain_a_bc.sh` is gone; otherwise
  it STOPs with the instance up. `guard_idle` repeats both checks before shutdown. `a_scores=184` is
  not required. (2) The summarizer requires ordered `E1`–`E6`/`A1`–`A6` ids and finite trait scores. It
  also requires the (calls, elapsed, rate) triple to be valid and shard-uniform, with 6 calls per row
  (test 23). (3) Scorer docstring: 600 calls.
* **Combined review round 7: 10/10, READY.** All three round-6 findings are fixed, and there are no new
  findings (no dash `set -eu` or `pgrep`/`pkill` self-match defect).

## 3. Order, tiers and decision points

| Tier | Blocks | GPU-h | Wall-clock (one job per card) |
|---|---|---|---|
| 1 | A0 → A → D; H and P prepared locally in parallel | ≈ 24 | ~1 day |
| 2 | BC | ≈ 88 | ~3.7 days |
| 2b | H on public data, after BC | ≈ 2.6 | ~3 h |

*2026-09-29: the user gave the Tier-2 go-ahead (「开跑」). BC is chained after A on server 41 by
`chain_a_bc.sh`, and H is chained after BC by `chain_h_public.sh`.*

1. **At power-on:**
   * check the SSH transport and GPU;
   * export `ARIS_MAX_TOTAL_GPU_HOURS=600`;
   * run A0, then launch A through `pilot_budget.py`.
2. **While A runs, on CPU/local:**
   * build D (rendering script, scorer path, contract test), the H package, the P draft, and the v4 unit and
     queue;
   * for each new GPU entrypoint, run the Codex code review and the deployment formula check.
3. **After A, D:** D takes about 2 h.
4. **Tier-2 decision, the user's:** BC launches only when both hold:
   * (i) A shows a D simple effect worth completing the 2×2 for, reported with its interval;
   * (ii) the P persona text is approved.
   D's result sets the claim scope. It does not veto BC.

## 4. Deferred, not compute

* **G2 in full** (720 transcripts, double annotation). H is a separate out-of-domain descriptive check that does not address G2.
* **G4 in full:** behavior-preserving, severity-calibrated perturbation pairs validated blind by bilingual
  humans, across the registered 12 cells.
* **Swahili** (L = 3). Without it every interaction stays exploratory.
* **A confirmatory H1 test** needs all of the following, because the 90 EVALUATION scenarios were read out of order by B1-neutral and B2 (`METHOD_DESIGN.md` execution status):
  * a new preregistration, frozen first;
  * a replacement never-read evaluation corpus;
  * rule-F gates met on qualification;
  * power re-sized for both support and refutation near β₀.

  Blocks A, D, H and BC work on the qualification-seed pole surfaces, and none of them makes a confirmatory claim.

  The design-kill re-review of 2026-09-28 (gpt-5.6-sol; `.aris/traces/design-kill/adjudicator_1790562739.json`) keeps this exposure as the one **unresolved fatal** item. It adds four obligations for the replacement preregistration:
  * justify the β₀ = 0.09 SESOI, and size power for both support and refutation near it;
  * simulate power over pair- and pole-heterogeneous configurations, or label MDE 0.33 as conditional on the homogeneous vector;
  * give the 70-item self-report a tested parseable format with an adequate token allowance, or budget 70 separate calls;
  * justify the clip at c₀ = 1 with a saturation sensitivity analysis.

  **Decision (b), user, 2026-09-28.** None of this is built in this project. The paper stays explicitly exploratory, and the design revision is not submitted for renewed confirmation. The four obligations above stay on record as requirements for any future confirmatory study.
* **Runtime-matched rater pass** (vLLM vs HF), needed only for cross-checkpoint statements.
* **Seed s20260829:** not retried (user instruction).

## 5. Standing constraints

* One job per card, and GPU 17 is untouched.
* Nothing is written to `QUALIFICATION_GATE.results`, and no confirmation is self-signed.
* No rerun of Qwen English B1/B2.
* No BFI-2/TIPI text.
* SLOT_DIFF stays EMPTY_COLUMN.
* Credentials go only through SSHPASS in the environment, never into files or Beads.

## 6. Codex review of rev. 1 (gpt-5.6-sol xhigh, 4/10) → changes

| Finding | Severity | Change in rev. 2 |
|---|---|---|
| D's n_inv = 48 is impossible with 23 clusters; not registered G3 | critical | 46 per cell, relabelled a descriptive analogue, no gate verdict |
| E's round-trip MT does not identify response to quality loss | critical | E dropped; faithful G4 deferred |
| D/E borrow TOST thresholds and labels | major | no TOST/q* verdicts; ι only as a scale reference |
| B/C skip scoring after their own G1 outcome (attrition) | major | checkpoints pre-selected by the prior screen; ITT scoring of every output |
| disfluent Chinese persona confounds P | major | block P: native-reviewed persona, approved by the user before freeze |
| A's "same bootstrap as B4 summary" is wrong (it is a normal approximation) | major | joint paired seed-cluster bootstrap specified; normal interval kept for comparison |
| decision point overstated D/E validity | major | D sets claim scope; the Tier-2 decision is A plus persona approval |
| trait analysed undefined | minor | own-setting Extraversion; Agreeableness separate |
| sep = 2·E* | minor | disclosed in §1 |
| B and C must not be split | advice | one indivisible block |
