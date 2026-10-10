# Method design: cross-lingual transfer matrix of induced personality

Design-time record for the confirmed idea `xling-induced-personality-transfer-matrix`. The machine-readable counterpart is `design_card.json`, authoritative where the two differ. The run plan is `EXPERIMENT_PLAN.md`.

**Execution status (2026-09-28).** This design was confirmed on 2026-09-17 (`.aris/PRE_RUN_CONFIRMED`, with Operator A's recorded design-kill WARN waiver). The approved order was then broken during execution.

- On 2026-09-18, operator instruction (`.aris/OPERATOR_SKIP_RULE_F`) let English B1-neutral and the B2 compliance-only control start without the rule-F admission freeze.
- Those runs read the **EVALUATION partition**: 90 scenarios × 24 seed clusters, before G2–G5 were met and before the R22-F5 renewal.
- B3's indicator contrast is computed on those transcripts. `EXPERIMENT_TRACKER.md` records this as a protocol deviation.

**Consequences, binding on every later reading:**

1. The 90 evaluation scenarios are **exposed**. No result drawn from them, or from any later run on them, is confirmatory evidence for H1/C1. Everything produced so far is **exploratory**.
2. A confirmatory test of H1 needs all of the following:
   - (a) a **new preregistration**, frozen before any of its evaluation data is generated;
   - (b) a **replacement evaluation corpus** of never-read scenarios, cut and hashed under rule E;
   - (c) the rule-F gates met on the qualification partition first;
   - (d) power re-sized for both lower-bound support and upper-bound refutation near β₀.
3. None of (a)–(d) exists yet. The 2026-09-28 follow-up (`FOLLOWUP_PLAN_2026-09-28.md`) adds descriptive blocks only, on the qualification-seed pole surfaces. None of them carries a confirmatory reading.
4. **Decision (b), user, 2026-09-28.** The study stays **explicitly exploratory**.
   - (a)–(d) will **not** be built in this project: no replacement preregistration and no replacement corpus.
   - This design revision is **not** submitted for renewed design confirmation. The two design-kill FAILs on the exposure item (`adjudicator_1790562131.json`, `adjudicator_1790562739.json`) stand unresolved and are disclosed, not waived.
   - The manuscript reports every number as exploratory and descriptive.

At the original drafting, this line read: "design drafted, deterministic checks passed, not yet approved by the human confirmation point. No generation, accelerator time or data collection has started."

This is revision **R12 as amended by R15–R35**, the **design of record**: the preregistered rule is one self-contained specification, sections **A–N plus R15-F1…R35-F2**, of `pico.preregistration.decision_rule`; nothing outside the card binds. Precedence: **R35 > R34 > R33 > R32 > R31 > R30 > R29 > R28 > R27 > R26 > R25 > R24 > R23 > R22 > R21 > R20 > R18 > R17 > R16 > R15 > R12**. Citations **rule X** point at section X. The language factorial runs **in full** as the gated second level (rule L) and carries every **operational R17-F1** contrast and **no mechanism claim** (**R25-F4**).

---

## 0. Module map (pasted from `SLOT_DIFF.md`)

---

## 1. Hypothesis and the falsifier of record

**Named labels** — **(C)**, **(LangFac)**, **(M) retired**, **(Elig)**, **(R) withdrawn** (**R25-F4**), **(A)/(B)** — are declared once in `pico.definitions`, with their scope and estimands, and not here.

**Falsifier of record (rule C).** The ONE rule is `pico.hypothesis.falsifier`: three outcomes per targeted trait over the admitted checkpoints, read off the **R26-F4** region with the **R29-F4** support quantifier and its exact complement for refutation (**R27-F5**). This section is a pointer to that field and states **no** independent rule.

**Rivals** are registered once each in `pico.hypotheses[]` — estimand, family and kill mechanism: **H2** is an observable instrument-administration-pipeline effect in its own family (**rule L**), never a confirmatory read for H1; **H2b** rater insensitivity is **not excluded**, since the R16-F1 family compares identical transcripts and establishes **no cross-language calibration** (**R17-F1**), and is reported beside the four rater-free indicators; **H3** is declarable only by the card rule-D TOST region and defaults to **UNRESOLVED**. All are tested against the *same* numbers.

---

## 2. Comparison and baselines

**Budget matching** is stated once in `pico.controls.held_fixed_nuisances`. The innovated module's column in the paradigm map is empty (EMPTY_COLUMN), so comparators are sourced externally, as `pico.comparison.baseline_source` records.

---

## 3. Single-variable property and declared adaptations

The independent variable is (C) and nothing else; `pico.intervention.variables` declares exactly one `independent` entry, everything else `controlled` or `derived`.

**The three modules marked `adapted: true`** (`design_lint` flags), justified once here: the *self-report instrument* (ref-108) is rendered into six languages, item content, scoring keys and trait assignment unchanged; *behavior-probe scoring* (ref-032) **adds** four rater-free indicators beside the published scoring, which runs unchanged in parallel; *translation and rendering* (ref-019) block-allocates the engine over (scenario, seed) units (rule H). None changes the induction mechanism, model, decoding or dialogue content, so attribution to (C) holds.

---

## 4. Controls, leakage and confounders

Nine leakage categories, ten confounders and the evaluator-validity protocol (rules F2/G) are executable in the card, including the three `risk` categories **L3.2** non-independence, **L3.3** sampling bias and **L4** contamination. Evaluator validity is **established before B1 on the qualification partition** — 720 blind double-annotated transcripts, the sample map in **R32-F1**, the trait score in **R33-F1**, the evaluation-tensor neutral scores in **R34-F1** and the statistic in **R31-F1** — or the behavior arm is demoted to descriptive.

---

## 5. Ablations

Six one-variable ablations and four load-bearingness drops are enumerated once in `pico.ablations`.

**The factorial core (LangFac) — P × D × I, 3 × 3 × 3 over {English, Chinese, Swahili}** — is the design's identification, not an optional ablation: three factors crossed rather than nested, contrasts within scenario and seed (eq6), decomposed sum-to-zero (eq7). **No main effect is a confirmatory endpoint for H1**; the confirmatory estimand is the **R15-F1** balanced match-minus-mismatch interaction contrast **clipped ONCE at the R29-F1 location** over the L × L subtable admitted by **R15-F3**. The factorial **runs in full**: the gated second-level family on the behavior channel and H2's self-report family on the frozen transcripts (rule L). **Switch × resource tier** is descriptive only: six languages cannot identify tier as a cause.

---

## 6. Outcome, power and variance

**One canonical specification, stated once in `design_card.json` (R20-F1).** Where any wording differs, the card wins.

The dispersion prior only seeds the simulation and never changes a run count (R20-F1). The **interval and declaration of record are the R26-F4 betting bounds** on unit contrasts clipped ONCE at the **R29-F1** location, read on the **R26-F1** budget at the **n_U = 144** R27-F2 units (no escalation, **R26-F5**); the boundary is the registered constant β₀ of **R27-F3**. The bootstrap-t and max-T regions are **reported agreement checks** and the twelve envelope points a **power study**. Certificates of record: `POWER_SIM_R30_FINAL.json` (**R30-F1**) and `power/GATE_POWER_R30_FINAL.json` (**R30-F2**); every figure lives once in the card.

---

## 7. Scope

The **operational** claim (**R25-F4**: no mechanism and no behavioral-transfer reading anywhere) is stated in the **R17-F1** words of `pico.hypothesis.statement`: it is made only for the three factorial languages, **for each trait at least three admitted checkpoints** (**R29-F4**), **conditional on the 90 fixed evaluation scenarios (rule E)**, about **behavior as scored** in a fixed ENGLISH instrument language (**R16-F1**) whose invariance is a gating limb family of **R22-F1**. The remaining apparatus rivals — **language-dependent rater sensitivity included** — stay **live and untested** (**R15-F4**). The six-language grid **(C) is descriptive** (**R29-F3**). Neither extends to resource tier as a cause.

---

## 8. Preregistration

Metric and split are locked before any evaluation generation. **The analysis plan, decision rule, falsifier of record and predicted result live once in `design_card.json`; this section is a pointer and states no independent rule.**

**Decision rule — pointer only.** Card rule B, as amended by R22, R23 and R26: the five conditions, the **R22-F1** admission gate, the **R26-F4** region at the **n_U = 144** R27-F2 units, the boundary β₀ (**R27-F3**) and α_j over K are stated there once. **No main effect carries a confirmatory read**: the language factorial runs in full as the gated second-level and self-report families, never as support for H1.

---

## Named models and the competence screen

| Role | Model | Why this one |
|---|---|---|
| Graded checkpoint 1 | `google/gemma-3-12b-it` (12B) | Swahili documented; Google lineage |
| Graded checkpoint 2 | `Qwen/Qwen3-8B` (8B, non-thinking pinned) | Swahili documented; Alibaba lineage |
| Graded checkpoint 3 | `meta-llama/Llama-3.1-8B-Instruct` (8B) | Swahili undocumented; exercises the screen |
| Graded checkpoint 4 | `CohereLabs/aya-expanse-8b` (8B) | multilingual-alignment lineage; Swahili undocumented |
| Rater | `OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym` (24B, int4 AWQ of the pinned base; bf16 `mistralai/Mistral-Small-3.1-24B-Instruct-2503` permitted at ≥80 GB) | fifth, independent lineage; serves in ~22 GB on the 48 GB card of record (**R35-F1**) |
| Translation engine A | Google Cloud Translation v3 (NMT) | all six languages |
| Translation engine B | Microsoft Azure AI Translator v3.0 | all six languages |
| Translation engine C | `facebook/nllb-200-3.3B`, pinned commit, local | reproducible after commercial engines change; DeepL excluded (no Swahili) |

Every checkpoint is loaded by an exact Hugging Face commit revision, resolved once by `tools/pin_checkpoints.sh` and written into the preregistration.

**Why naming forces a new control.** Two checkpoints do not document Swahili; without a screen a model that cannot hold a Swahili conversation would enter the matrix as one whose induced trait fails to transfer. The pre-registered competence screen (**R16-F5**, on its own hashed 50-prompt **qualification** sample per checkpoint × language, never tuning material) runs before evaluation; under rule F3 a checkpoint enters confirmation only if it clears **all three L₃ languages**.
