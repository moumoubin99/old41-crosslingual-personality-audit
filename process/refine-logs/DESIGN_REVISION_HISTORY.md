

# Archived from METHOD_DESIGN.md at revision R5 (2026-09-11)

## 9. R3 repairs — what each design-kill finding required and what answers it

The R2 design-kill (reviewer `gpt-5.6-sol`, fresh attack + adjudication threads) returned **FAIL** on card
`cab14725…` with four `unresolved_fatal` and three `partially_controlled`. Each is answered here, and the
answer is in the card, not only in this file.

**F1 — the intervention bundles five things (fatal).** "The language-switch intervention jointly changes
generation language, persona–dialogue congruence, translated measurement, tokenization and pragmatics", so
no result could isolate the representational mechanism. The reviewer offered two exits: factorially
manipulate the languages, or narrow the claim to an omnibus pipeline effect with no mechanism claim. **The
user chose the factorial.** (LangFac) crosses persona-instruction, dialogue-generation and instrument
language over {English, Chinese, Swahili}; translation *procedure* is held constant by the same
three-forward + back-translation screen everywhere; generation length is held constant by a per-language
band calibrated on the development split. The bundled grid survives as (C), relabelled an omnibus effect
and stripped of every mechanism claim. §1, §5, `pico.definitions (LangFac)`, `ablations.factorial[0]`, eq7.

**F2 — the retention ratio is undefined or unstable for weak diagonal effects, not sign-safe, not bounded
(fatal).** The confirmatory metric moves to the unnormalized paired difference-in-differences (M), which has
no denominator, is defined in every cell and is sign-safe; δ_raw and ε_raw are expressed on that scale. The
ratio and shortfall survive as interpretable secondaries under (Elig), with Fieller intervals rather than
bootstrap percentiles because the denominator is itself estimated, and **no clipping** anywhere. Ineligible
cells are reported as *ratio not defined* and counted, never imputed or dropped. §1, §6, eq2/eq3/eq6.

**F3 — failure to reject a shortfall was read as full transfer (fatal).** H3 becomes an equivalence claim
with a justified margin (ε = 0.075 of the induced shift = half the minimum audit-relevant loss) tested by
Holm-adjusted TOSTs that must hold **simultaneously** across pairs and models. Anything else is *H3
unresolved*. The R2 clause "failure of (1) reports H3 as supported" is deleted. §1 (H3 row), §8.

**F4 — six languages cannot identify pretraining resource tier, and two resource statistics were
registered (fatal).** The gradient is demoted to a descriptive six-language association; the rank
correlation is deleted so only the eq5 within-source centered slope remains; censored cells get worst-case
and best-case bounds instead of being dropped; and no hypothesis condition depends on it. §5, §6, eq5.

**F5 — power was read off a single-contrast paired formula while the decision is hierarchical
(partial).** The registered count now comes from a simulation of the complete decision — dialogue-language
main effect at δ_raw, family-wide BH over the primary family, required in ≥ 3 of 4 checkpoints — driven by
development-split variance components, with the script and seed frozen alongside the preregistration.
Multiplicity is applied **once** family-wide rather than separately within each model. §6.

**F6 — rater agreement was treated as equal calibration and equal sensitivity (partial).** Two additions
qualify the rater per language before B1: **multilingual anchor pairs** (frozen transcripts carrying known
injected trait deltas, matched in content and length, rendered into every language through the same
pipeline; the rater must recover the delta with slope in 0.85–1.15 and intercept within ±0.1 in *every*
language, and the slopes must pass a cross-language equivalence test) and **perturbation pairs** (minimally
edited pairs differing by one trait-diagnostic behavior, certifying equal *sensitivity* under the same
margin). For the four rule-based rater-free indicators, trait-specific measurement invariance is established
per language; an indicator failing invariance for a trait is descriptive for that trait and leaves the
confirmatory hierarchy. Failing qualification triggers the pre-fixed channel substitution. §4, `controls`.

**F7 — the B3 ablation subset could be chosen after B1 exposed the evaluation scenarios (partial).** The B3
scenario IDs, the selection algorithm, the contrasts and the reuse rules are frozen and **hashed before B1
starts**; the hash is recorded in the plan. `preregistration.analysis_plan`, EXPERIMENT_PLAN §B3.

**What R3 does not fix, stated rather than hidden.** The factorial identifies mismatch versus instrument
translation; it does not identify *why* a mismatch costs expression — tokenization, pragmatic register and
representation quality still move together within a language. Three languages is the smallest set that
crosses script and resource tier, and the mechanism claim is scoped to them. And the resource gradient
remains an association over six points: no design at this scale can make it causal.

## 10. R4 repairs — the DESIGN_KILL R3 findings and the design change that answers each

R3's adversarial round returned **five `unresolved_fatal` points and two `partially_controlled`**. Each is
answered by a change to the design, not by rewording; two of them remove a claim rather than defend it.

**R3-F1 — the confirmatory dialogue-language main effect is not a persona–dialogue mismatch estimand.** A main
effect of D averages over persona languages, so it mixes the mismatch of interest with the difficulty of
conversing in D; the predicted pattern could appear with no mismatch effect at all. R4 makes the confirmatory
estimand the **pre-registered pair-specific mismatch contrasts**
Δ_mis(p,d) = M(P=p, D=p) − M(P=p, D=d), d ≠ p — six ordered contrasts per checkpoint per targeted trait,
which are exactly the **signed P × D interaction contrasts**, because the neutral arm entering M is generated
in the *same* dialogue language and so differences D's own effect out before the contrast is formed. The
confirmatory read is their registered average Δ̄_mis; the six are also reported individually under the same
family-wide BH with a registered **4-of-6 consistency requirement** so no single language pair can carry the
result. Multiplicity and the power simulation are defined over exactly this family and no other.

**R3-F2 — instrument language is not a manipulable factor for the transcript-behavior outcome.** The
transcript is produced under P and D and frozen; the inventory is administered afterwards, so I cannot cause
anything in the behavior channel. R4 splits the factorial **by outcome channel**: the behavior channel is a
**P × D factorial only** (9 cells over L3, one transcript set per cell); the self-report channel crosses **I on
the frozen transcripts** — each core transcript re-administered once per instrument language, paired within
transcript, every analysis clustered at transcript level — and carries H2 together with the three-way
decomposition eq7, which is computed on that channel only; and a separate pre-registered **rater-language**
factor is applied post hoc to **identical** frozen transcripts on a 20% stratified subsample, carrying H2b as
a pure measurement control.

**R3-F3 — the factorial does not identify a language-specific representation.** R4 **withdraws the mechanism
claim** rather than dressing it. (R) is no longer a hypothesis, enters no decision-rule condition, and no
result is reported as evidence for or against it; H1 claims the **observable** mismatch effect and nothing
more. Dialogue-language competence, tokenization, pragmatic conventions and translation naturalness of the
persona text are named as live alternative explanations, partially bounded by the competence screen, the
length band and the three-translation ensemble, and otherwise disclosed as untested. Distinguishing them would
need orthogonal manipulations that this design does not run; they are named as the next experiment.

**R3-F4 — compliance handling left the estimand and the sampling rule undefined.** R4 locks an
**intention-to-treat** rule on the **first planned generation**, with pre-defined outcomes for wrong-language
responses (analysed as produced, rate reported as a secondary outcome), unparseable items (scale midpoint,
conservative toward the null; >30% unparseable ⇒ the whole administration at midpoint, never dropped) and
empty transcripts (scored as produced). Replacement is forbidden except for prespecified infrastructure
failures, re-run with the same seed, logged and capped at 2% per cell; **no regeneration may depend on any
outcome value**. Per-protocol estimates are secondary throughout.

**R3-F5 — the run, inventory, compute and power arithmetic omitted load-bearing factors.** R4 enumerates the
**full run tensor**: 2 targeted traits × 2 poles plus a shared neutral arm = 5 generation settings; behavior
core 9 × 5 × 4 × 90 × 8 = **129 600 transcripts**; descriptive omnibus grid 27 × 5 × 4 × 30 × 4 = **64 800**;
self-report **388 800 administrations** on the core (70 items each = 27 216 000 **item responses**) plus
129 600 perturbation and 64 800 omnibus administrations; rater scoring of 194 400 transcripts plus 51 840
rater-language re-scorings. The scenario counts are reconciled once and for all: **120 = 30 development + 90
evaluation**, with the omnibus grid on a registered 30-scenario subset of the 90. Compute is recomputed from
that tensor at **≈240–400 GPU-hours** on one A100 80GB for evaluation plus ≈90 for development; **one 80GB
card is required, for the 24B rater only** — the four graded checkpoints run at bf16 on 48GB. Power is
simulated from the exact hierarchical sampling plan for the exact R4 decision rule. The three untargeted Big
Five traits are measured in every administration and reported as **descriptive off-target outcomes**, which is
how the trait factor is kept inside the envelope without hiding it.

**R3-P6 — the round-trip dummy does not isolate instrument-translation degradation.** Demoted to a
**sensitivity analysis**; the hard tie rule that let it reject H1 is **deleted**. H2 is carried instead by the
within-transcript instrument-language contrasts and by a **matched text-quality perturbation** that varies
rendering quality at **fixed final instrument language**, separating "in another language" from "worse text".

**R3-P7 — the universal H3 equivalence decision is not powered by the registered simulation.** With
ε_raw ≈ 0.045 against a δ_raw = 0.09 the study is sized for, a *simultaneous* Holm-TOST across every pair and
model is not adequately powered at any run count this envelope can buy. R4 **removes H3 from the powered
confirmatory claims**: per-pair TOSTs are reported descriptively with their simulated power stated beside
them, full language-generality is never declared, failure to reject a shortfall is never read as full
transfer, and the pre-registered default reading of an inconclusive pattern is **H3 UNRESOLVED**.

**What R4 does not fix.** The mechanism behind the mismatch effect is unidentified and unclaimed. The omnibus
grid stays bundled and descriptive. Three Big Five traits are off-target descriptive outcomes only. And the
self-report channel's instrument contrast, while causally coherent, measures a *measurement* effect on a
frozen state — it cannot speak to behavior, and the design says so instead of implying otherwise.


# Archived from EXPERIMENT_PLAN.md at revision R5 (2026-09-11)

## 8. R4 — what each DESIGN_KILL R3 finding changed

**Fatal 1 — the dialogue-language main effect is not a mismatch estimand.** Replaced everywhere by the
pre-registered **pair-specific mismatch contrasts** Δ_mis(p,d), i.e. the signed P × D interaction contrasts,
with their registered average as the confirmatory read, family-wide BH over exactly that family, a 4-of-6
consistency requirement, and power simulated for exactly that rule.

**Fatal 2 — instrument language is not manipulable for the behavior outcome.** The factorial is split by
outcome channel: the behavior channel is **P × D only**, one transcript set per cell; instrument language is
crossed on the **frozen transcripts** in the self-report channel with transcript-level clustering; and a
separate **rater-language** factor is applied post hoc to identical transcripts for H2b.

**Fatal 3 — the factorial does not identify a language-specific representation.** H1 is narrowed to the
**observable** mismatch effect. (R) is withdrawn as a claim and retained only as background motivation;
competence, tokenization, pragmatics and translation naturalness are named as live, untested alternatives.

**Fatal 4 — compliance handling left the estimand undefined.** §6 locks an **intention-to-treat** rule on the
first planned generation, with pre-defined outcomes for wrong-language responses (analysed as produced; rate
reported), unparseable items (scored at the scale midpoint, conservative toward the null), and empty
transcripts; replacement forbidden except for prespecified infrastructure failures capped at 2% per cell;
per-protocol estimates secondary.

**Fatal 5 — the run, inventory, compute and power arithmetic omitted load-bearing factors.** §4 and the card's
`outcome.n_runs` now enumerate the **full run tensor** — traits and poles as explicit generation settings, the
shared neutral arm, both channels, administrations distinguished from item responses — and reconcile
**120 = 30 development + 90 evaluation**, with the omnibus grid on a registered 30-scenario subset of the 90.
Compute and power are recomputed from that tensor.

**Partial 6 — the round-trip dummy is non-diagnostic.** Demoted to a sensitivity analysis; the hard tie rule
is deleted; H2 is carried instead by outcome-specific instrument-language contrasts plus a matched
text-quality perturbation at fixed final language.

**Partial 7 — the universal H3 equivalence decision is not powered.** H3 is **removed from the powered
confirmatory claims**. Per-pair Holm-adjusted TOSTs at ε_raw ≈ 0.045 are reported descriptively with their
simulated power beside them; the default reading of an inconclusive pattern is **H3 UNRESOLVED**.

**What R4 does not fix.** The mechanism behind the mismatch effect stays unidentified and unclaimed. The
descriptive omnibus grid remains bundled. Three of the Big Five are measured only as off-target descriptive
outcomes, because the confirmatory persona manipulation is restricted to two traits so that the run tensor
stays inside the compute envelope.


<!-- archived from METHOD_DESIGN.md at R6 -->
## 10. R5 repairs — the DESIGN_KILL R4 findings and the design change that answers each

R4's adversarial gate returned FAIL on four `unresolved_fatal` findings. **R5 is the design of record**;
where any earlier wording survives elsewhere and conflicts with this section, this section governs.

**R5-F1 — the mismatch estimand had incompatible definitions and a reversed sign (fatal).** The letter `M`
named both a within-cell effect and a difference-in-differences, so the registered sign flipped between
readings. R5 retires `M` and fixes one algebra used everywhere: cell effect `E_q(p,d)` (eq1) → pole-aligned
cell effect `E*(p,d) = ½·[E_high − E_low]` → confirmatory mismatch contrast
`Δ_mis(p,d) = E*(p,p) − E*(p,d)`, which is eq6 with `s = p`, `t = d`. Positive `Δ_mis` = the switch **cost**
induced expression. Every equation, hypothesis, family and power rule in the card and in this document now
uses that one chain (§1 (M), §5, §6, §8).

**R5-F2 — obsolete R3 main-effect endpoints and the round-trip-tie rule were still live (fatal).** R5 is a
single authoritative revision: the confirmatory family is the 6 pair-specific mismatch contrasts and their
registered average `Δ̄_mis`, and nothing else. Every confirmatory read on a **main effect**, and the
round-trip-tie condition, are deleted from H1's rule, its falsifier, the MDE metric, the power target and
the analysis plan. **The language factorial is still run in full, as directed:** the behavior channel's
P × D factorial supplies the persona-language main effect, the dialogue-language main effect and the P × D
interaction as the *gated second-level* family; the self-report channel's P × D × I factorial on the same
frozen transcripts carries H2; the rater-language factor carries H2b. Those terms are estimated and
reported — they are simply not the confirmatory read, because a main effect of D averages over persona
languages and mixes the mismatch of interest with the difficulty of conversing in D.

**R5-F3 — power was not established for the complete conjunctive rule, whose variability threshold was
itself undefined (fatal).** `V̂` now has an explicit population definition and a frozen estimator (§8), the
threshold is the single number `max(δ_raw, 2·V̂)`, and the frozen Monte-Carlo simulation encodes the whole
conjunction — BH significance ∧ threshold (with `V̂` drawn from its own sampling distribution) ∧ 4-of-6 pair
consistency ∧ ≥ 3-of-4 checkpoints — run on development-split components before any evaluation generation,
with a pre-registered deterministic branch (raise seeds to a cap of 12, else drop the `2·V̂` limb and record
it) decided at that moment and never after.

**R5-F4 — the confirmatory estimand omitted the persona pole and left its aggregation unspecified
(fatal).** The pole index `q ∈ {high, low}` is now a registered dimension of the estimand: frozen signs
`σ_high = +1`, `σ_low = −1`; the neutral arm (B) shared by both poles inside a cell, so the pole enters only
through the persona arm; and the confirmatory pooling is the **equal-weight** mean of the two sign-aligned
poles — no data-dependent weighting, no pole selection, no pole dropped. A pre-specified **pole × language
heterogeneity test** (`Δ_mis^high − Δ_mis^low`, per checkpoint, trait and ordered pair) is registered in its
own secondary family under its own BH at 0.05. It never gates H1; if it survives correction, the pooled
confirmatory value is still the claim but must be reported beside the two per-pole values and stated as
holding on the pole-averaged scale.

**R5-P4 — scorer validation was not independent and the item set could change through outcome-dependent DIF
screening (partial).** B0's 30 development scenarios are partitioned once, before any tuning, into 18
scorer-tuning and 12 disjoint scorer-qualification scenarios; the rater is tuned only on the tuning half,
qualified once on the held-out half and then frozen by hash. All inventory items and rater-free indicators
are frozen before the first evaluation administration, and DIF screening on evaluation data is a
**sensitivity analysis only** — no item is added, removed, re-weighted or re-scored on the basis of
evaluation-derived DIF.

**R5-P5 — the uncertainty analysis conditioned on realized seeds and aliased translation, wording and order
nuisances with seed (partial).** One pre-registered randomization schedule, frozen before any generation,
assigns the sampling seed, forward-translation variant, persona wording variant, item order and option
order for every (cell, setting, scenario, seed). Translation and wording variants are **balanced-completely
crossed** with seed — the R4 rotation of translation across seeds is deleted — item and option order are
independently randomized, and uncertainty comes from a **multiway cluster bootstrap** over scenarios, seeds
and translation variants jointly.

**R5-P6 — the primary contrast estimated assignment, not realized conversation language (partial).** H1 is
restated as an **intention-to-treat** claim about *assignment* to a requested dialogue language. Every
confirmatory number uses the first planned generation as produced, including wrong-language output;
per-cell compliance rates are a registered secondary outcome reported beside every primary result; and
compliance-conditioned per-protocol estimates are secondary and never substitute for the ITT estimate. No
IV or encouragement estimand is claimed, so no exclusion or monotonicity assumption is needed.

The (Elig) rule, translation ensemble, invariance screen, competence screen and contamination reporting are
unchanged and remain in force.


<!-- archived from EXPERIMENT_PLAN.md at R6 -->
## 8. R5 — what this revision changed

`METHOD_DESIGN.md` §10 states each of the four `unresolved_fatal` and three `partially_controlled` R4
design-kill findings and the repair that answers it; revisions R2→R4 are archived verbatim in
`DESIGN_REVISION_HISTORY.md`. What the **plan** executes differently: §1 carries the corrected estimand
`Δ_mis = E*(p,p) − E*(p,d)`, the frozen pole signs and equal-weight pole pooling, the threshold
`max(δ_raw, 2·V̂)` with `V̂`'s estimator, and the ITT statement; §6 item 1 demotes the round-trip tie to a
sensitivity signal and §6 item 2 is rewritten on `Δ̄_mis`; the §7 checklist adds the B0 tuning/qualification
split, the frozen item set with DIF as sensitivity only, the single frozen randomization schedule with
translation and wording crossed with seed, and the complete-conjunction power simulation before any
evaluation generation. B1b still runs the language factorial in full — as the gated second-level family, not
as a confirmatory read. No block, budget line or run count changes unless the pre-registered power branch of
§1 raises the seed count, which is decided and recorded before any evaluation generation.


<!-- archived from EXPERIMENT_PLAN.md §1 at R6 -->
**R5 fixes the symbol and the sign (design-kill R4 fatal 1).** The letter `M` is retired: `E_q(p,d)` is the
persona-minus-neutral **cell effect** for pole `q` (eq1), `E*(p,d) = ½·[E_high(p,d) − E_low(p,d)]` is the
**pole-aligned** cell effect under the frozen signs `σ_high = +1`, `σ_low = −1` against the shared neutral
arm, and `Δ_mis` is eq6 on `E*`. Positive `Δ_mis` = the switch **cost** induced expression.

**R5 puts the persona pole in the estimand (fatal 4).** The confirmatory pooling over poles is the
equal-weight mean above — no data-dependent weighting, no pole dropped — and a pole × language heterogeneity
test (`Δ_mis^high − Δ_mis^low`) is registered in its **own** secondary family under its own BH at 0.05. It
never gates C1; if it survives correction the pooled value is still the claim, reported beside both per-pole
values and stated as holding on the pole-averaged scale.


<!-- archived from METHOD_DESIGN.md §1 label block at R6 -->
- **(C)** — the **bundled** cross-lingual pipeline switch: the persona instruction is given in the
  induction language while the dialogue, the inventory items and the response-scale labels are all rendered
  in the assessment language. R3 states plainly what this is: it moves persona/dialogue language mismatch,
  instrument language, tokenization and pragmatics *at the same time*, so it identifies an **omnibus
  language-pipeline effect** and nothing finer. No mechanism claim is read off (C).
- **(LangFac)** — the **three-way language factorial core**, the design's actual identification from R3.
  Persona-instruction language P × dialogue-generation language D × instrument language I, fully crossed
  over L₃ = {English, Chinese, Swahili} — one high-resource Latin-script, one high-resource non-Latin-script,
  one low-resource — giving 27 cells per checkpoint per arm, of which 18 are new (the 3 all-equal cells and
  the 6 cells with P = s, D = I = t already exist in the (C) grid and are reused run-for-run). Rendering is
  matched: every persona text, scenario seed and inventory item passes the same three-forward-translation
  plus back-translation screen, so translation *procedure* is constant across factors; and generation length
  is matched by a per-language length band calibrated on the development split (identical turn count, mean
  turn length within ±10%), with realized token counts still reported per cell.
- **(M) — retired symbol, R5.** Through R4 the letter `M` named both a within-cell effect and a
  difference-in-differences, and the registered sign was reversed under one of the two readings
  (design-kill R4 fatal 1). R5 names the two levels separately and uses them everywhere:
  the **cell effect** `E_q(p,d)` (eq1) is the persona-minus-neutral induced shift for pole `q` inside the
  single cell (P = p, D = d), averaged over translation variants and repeats — *not* a
  difference-in-differences; the **pole-aligned cell effect** is
  `E*(p,d) = ½·[E_high(p,d) − E_low(p,d)]`, with frozen signs `σ_high = +1`, `σ_low = −1` and the same
  neutral arm (B) shared by both poles inside the cell; and the **confirmatory mismatch contrast** is
  `Δ_mis(p,d) = E*(p,p) − E*(p,d)` for `d ≠ p`, which *is* eq6 with `s = p`, `t = d`. It has no
  denominator, is defined in every cell, is sign-safe — positive means the language switch **cost** induced
  expression — and is never clipped, which is why it is the confirmatory quantity. The
  interpretable effect-retention ratio ρ = Δ(s,t)/Δ(s,s) and the shortfall `S = 1 − ρ` are **secondary**
  summaries, reported with Fieller intervals and only on cells passing (Elig).
- **(Elig)** — the denominator-eligibility rule. A cell's ratio and shortfall are reported only if its
  diagonal reference effect Δ(s,s) is bounded away from zero on **both** tests: its two-sided bootstrap
  interval excludes zero after the same Benjamini–Hochberg correction as the primary family, **and**
  |Δ(s,s)| ≥ max(0.30 scale points, 4·SE). The 0.30 floor is where the σ = 0.11 relative dispersion prior
  stops distinguishing a diagonal effect from noise at 8 seeds; the 4·SE condition caps the ratio's relative
  standard error near 0.25, which is where Fieller intervals stay finite. Cells failing (Elig) are reported
  as **ratio not defined** — never imputed, never clipped, never silently dropped — and still contribute
  their unnormalized `Δ_mis` difference, which needs no denominator.
- **(R)** — the claimed mechanism: persona instructions act partly through language-specific
  representations, so a **mismatch between the language of the persona instruction and the language the
  model actually converses in** should cost induced trait expression over and above the cost of rendering
  the instrument in another language. **R5 rebinds (R):** it is supported only by the pair-specific mismatch
  contrasts `Δ_mis(p,d)` of (Mismatch) — never by any **main effect** (a main effect of D averages over
  persona languages and mixes the mismatch of interest with the sheer difficulty of conversing in D),
  never by the bundled switch (C), and never by the six-language resource gradient, which is descriptive.
- **(A)** / **(B)** — the persona arm and the content-matched neutral arm. Every cell reports the
  difference (A) − (B), never an absolute trait score.



<!-- archived from METHOD_DESIGN.md at R7 -->
## 10. Revision R5 → R6

R5's repairs (the four R4 `unresolved_fatal` and three `partially_controlled` findings) are archived verbatim
in `DESIGN_REVISION_HISTORY.md`; their rules remain in force in `design_card.json`. **R6 is the design of
record**, and where any earlier wording conflicts with it, R6 governs. The authoritative text of every R6
repair is `design_card.json` → `pico.preregistration.decision_rule`, blocks R6-F1…F5 and R6-P1. This section
states what changed and why; it adds no rule of its own.

- **R5 fatal 1 — the mandatory bootstrap resampled a three-cluster dimension (R6-F1).** The three forward
  translation engines are a frozen exhaustive **fixed set**, so cluster-level inference over them is
  impossible. Uncertainty is now a **two-way cluster bootstrap over scenarios and seeds only**, computed
  *within* each engine stratum and averaged over the three strata with equal weight. Scope is narrowed in the
  claim itself to the registered three-engine ensemble, and engine invariance becomes a reported
  **sign-concordance check**, not an inference; discordance narrows the headline claim to name the engine.
- **R5 fatal 2 — 0.8 power was claimed at a true effect equal to the decision threshold (R6-F2).** That is
  unattainable: the frozen simulator returns **0.050** there. `δ_raw = 0.09` is now the SESOI and threshold
  only, and the **powered alternative** `δ_alt` is the smallest true effect above `max(δ_raw, 2·V̂)` reaching
  0.80 for the *complete* conjunction. `power/power_sim_r6.py` (sha256 33aad3b6…2b2a, 100 000 draws, seed
  20260911) gives **δ_alt = 0.14** at the registered planning values (0.498 at 0.12, 0.924 at 0.15); the same
  frozen script is re-run on development-split components before any evaluation generation. The deterministic
  branch raises seeds to the cap of 12 and then raises `δ_alt`, declaring the SESOI-to-`δ_alt` band explicitly
  underpowered; the `2·V̂` limb is **never** removed to buy power (the R5 removal branch is deleted).
- **R5 fatal 3 — competence exclusions left the six-pair average and the 4-of-6 rule undefined (R6-F3).** A
  checkpoint enters confirmation only if it clears the competence screen in **all three L3 languages**, so
  every admitted checkpoint carries the identical six ordered pairs. If fewer than three checkpoints qualify,
  **H1 is untestable** — the ≥ 3-of-4 requirement is never re-scaled.
- **R5 fatal 4 — the rule had no quantifier across the two targeted traits (R6-F4).** The complete rule must
  hold **independently for each** targeted trait. Holding for exactly one is reported as a trait-specific
  result naming both traits, not as support for H1.
- **R5 fatal 5 — the rater-failure fallback switched the endpoint (R6-F5).** The substitution to the
  self-report channel plus rule-based indicators is **deleted**. If the rater fails its registered validity
  bar, confirmatory H1 is declared **untestable**; the self-report channel and the rater-free indicators stay
  pre-specified secondary analyses.
- **R5 partial 1 — the shared neutral arm cancels from the pole-averaged endpoint (R6-P1).** The endpoint is
  unchanged, but H1 now carries a registered **pole-concordance** condition (5): both pole-specific
  persona-minus-neutral mismatch contrasts must be positive, and its cost is carried in the power simulation.
  Where they disagree the result is reported as a pole-separation effect, not a persona-versus-neutral shift.

The (Elig) rule, translation ensemble, invariance screen, competence screen, ITT sampling rule, scorer
tuning/qualification split and contamination reporting are unchanged and remain in force.


<!-- archived from EXPERIMENT_PLAN.md at R7 -->
## 8. R6 — what this revision changed

`METHOD_DESIGN.md` §10 states each R5 design-kill finding and the repair that answers it; R2→R5 are archived
verbatim in `DESIGN_REVISION_HISTORY.md` and the authoritative rules live in
`design_card.json` → `pico.preregistration.decision_rule` (R6-F1…F5, R6-P1). What the **plan** executes
differently under R6: uncertainty is a two-way cluster bootstrap over scenarios and seeds *within* fixed
translation-engine strata (never resampling engines), with an engine sign-concordance check reported beside
every primary result; the power run before any evaluation generation now targets the **powered alternative**
`δ_alt` (0.14 at planning values) rather than the SESOI `δ_raw = 0.09`, using the frozen
`power/power_sim_r6.py`; only checkpoints clearing all three L3 languages enter confirmation, and fewer than
three qualifying checkpoints ends the study with an untestable verdict; the complete rule must pass for both
targeted traits; and there is no endpoint substitution if the rater fails qualification. B1b still runs the
language factorial in full — as the gated second-level family, not as a confirmatory read. No block or budget
line changes unless the pre-registered power branch raises the seed count, which is decided and recorded
before any evaluation generation.


<!-- archived from METHOD_DESIGN.md at R8 -->
## 9. Revision history

Revisions R2→R4 and the design-kill findings each answered are archived verbatim in
`DESIGN_REVISION_HISTORY.md`. Only the current revision is stated here.

## 10. Revisions R6 → R7
Full R5 → R6 text archived in `DESIGN_REVISION_HISTORY.md`. R7 answers the R6 design-kill FAIL:
- **R7-F1 (fatal 1) — one executable run tensor.** Translation engine and persona wording are **block-allocated**, never crossed with seed: within each (checkpoint, cell, setting) the 90 × 8 = 720 generation units split into the 9 (engine, wording) blocks at exactly 80 units each (240 per engine, 240 per wording), orthogonal to seed and scenario stratum. 9 × 5 × 4 × 720 = 129 600 transcripts, exactly `outcome.n_runs`; the indexed allocation is frozen and hashed as `NUISANCE_SCHEDULE.json`. eq1's cell effect is the equal-weight (1/3) combination over the three fixed engine strata of the within-stratum matched-pair mean — no separate variant dimension.
- **R7-F2 (fatal 2) — one falsifier.** The R7 falsifier of record in §1 is stated in identical words in `hypothesis.falsifier` and `preregistration.decision_rule`; every earlier falsifier wording is void. Only the equivalence branch (B) disproves H1; failure of the conjunction alone is (C) inconclusive.
- **R7-F3 (fatal 3) — rater attenuation bounded and gating.** Contrasts are formed on the anchor-calibrated scale; anchor slope 0.94–1.06, intercept ±0.03, cross-language slope equivalence ±0.05, so worst-case attenuation of the 0.60 diagonal effect is ≤ 0.036, well under δ_raw = 0.09. Any anchor, perturbation or rater-language artifact failure deterministically removes that checkpoint-language from the admitted set; below 3 of 4 admitted checkpoints the behavior channel is not admitted and H1 falls to (C), never to refutation.


<!-- archived from design_card.json decision_rule at R8 -->
R5-F2 MAKES THIS THE SINGLE AUTHORITATIVE CONFIRMATORY SPECIFICATION, governing wherever an earlier revision conflicts. The confirmatory family is the 6 pair-specific mismatch contrasts and their registered average, and NOTHING else: no confirmatory read on the dialogue-language main effect, no round-trip-tie rule, no R3 main-effect family. THE LANGUAGE FACTORIAL IS STILL RUN IN FULL - the behavior P x D factorial over L3 (both main effects and the interaction) as a GATED SECOND-LEVEL family after the primary family, the self-report P x D x I factorial on the same frozen transcripts as H2's instrument-language family, and the rater-language factor as H2b - estimated and reported, simply not the confirmatory read, because a main effect of D averages over persona languages and mixes the mismatch of interest with the difficulty of conversing in D. The confirmatory estimand is Delta_mis(p,d) = E*(p,p) - E*(p,d) (eq1 -> eq6); the overloaded symbol M is retired; full definition in outcome.primary_metric.
R6-F1 (design-kill R5 fatal 1) STOPS RESAMPLING A THREE-CLUSTER DIMENSION. The three forward-translation engines are a FROZEN, EXHAUSTIVE, FIXED SET, not a sample from a population of engines, so no cluster-level inference over them is possible and none is claimed. The confirmatory uncertainty procedure is a TWO-WAY CLUSTER BOOTSTRAP over SCENARIOS and SEEDS ONLY (10000 resamples, Cameron-Gelbach-Miller construction), computed CONDITIONAL ON the engine ensemble: the engine is a FIXED STRATUM, resampling is done WITHIN each engine stratum, and the confirmatory statistic is the equal-weight (1/3 each) average of the three within-engine estimates, exactly as the balanced-crossed schedule allocates them. Engine variation therefore enters the point estimate as a fixed design average and never as a resampled random effect. SCOPE IS NARROWED ACCORDINGLY, in the claim itself: every confirmatory result is stated as holding UNDER THE REGISTERED THREE-ENGINE TRANSLATION ENSEMBLE (Google Cloud Translation v3, Microsoft Azure AI Translator v3.0, NLLB-200-3.3B at the pinned revision), and generalisation to arbitrary translation systems is NOT claimed. ENGINE INVARIANCE IS A REGISTERED, REPORTED CHECK, NOT AN INFERENCE: Delta_mis_bar is re-estimated separately within each of the three engine strata and the three point estimates must be SIGN-CONCORDANT; if they are not, the headline claim is narrowed in the title, abstract and discussion to name the discordant engine and to state that the effect is engine-dependent. The R5 multiway bootstrap that resampled translation variants jointly is DELETED from every artifact; the scenario-only bootstrap of R4 remains a labelled sensitivity view.


<!-- archived from design_card.json outcome.primary_metric at R8 -->
R5-F4 (design-kill R4 fatal 4) PUTS THE PERSONA POLE INTO THE CONFIRMATORY ESTIMAND. The pole index q in {high, low} is a registered dimension of the estimand, not a nuisance. Frozen sign transformation: sigma_high = +1, sigma_low = -1. The neutral arm B is shared by the two poles within a cell, so the pole enters only through the persona arm. CONFIRMATORY POOLING is the EQUAL-WEIGHT mean over the two sign-aligned poles, i.e. E* above - no data-dependent weighting, no pole selection, no pole dropped. A PRE-SPECIFIED POLE-BY-LANGUAGE HETEROGENEITY TEST is registered: the pole contrast Delta_mis^high(p,d) - Delta_mis^low(p,d) per checkpoint, trait and ordered pair, tested in its OWN registered secondary family under Benjamini-Hochberg at 0.05, never pooled with the primary family. It does not gate H1: if pole heterogeneity survives correction the pooled value is still the claim, but it MUST then be reported beside the two per-pole values wherever it appears and the claim stated as holding on the pole-averaged scale. 


<!-- archived at R9 -->
### R8-F2 original algorithm (seed-first ordering), superseded by R9-F3

R8-F2 (design-kill R7 fatal 2) MAKES THE CONFIRMATORY ANALYSIS UNIQUELY EXECUTABLE. THE ONE ALGORITHM, in this order and no other: (1) per (cell, setting, scenario, seed) compute the channel score; (2) aggregate over the 8 SEEDS by ORDINARY MEAN (the interquartile mean is a sensitivity view only); (3) aggregate over scenarios by ordinary mean within each fixed translation-engine stratum; (4) form the pair-specific mismatch contrast Delta_mis(p,d) within stratum; (5) combine the three engine strata with EQUAL WEIGHT 1/3 (eq1); (6) average the ELIGIBLE ordered pairs (R8-F1) to Delta_mis_bar; (7) interval and test: WILD CLUSTER BOOTSTRAP-t over the 8 SEED CLUSTERS within each engine stratum, Webb 6-point weights, 9999 replications, NULL IMPOSED, studentized by the cluster-robust standard error, engine strata combined at the same 1/3 weights inside every replication, with the registered bootstrap seed; (8) Benjamini-Hochberg at 0.05 family-wide over the primary family of multiplicity_correction, and the reported interval is taken at the BH-adjusted level. Every alternative named anywhere - the two-way scenario x seed cluster bootstrap of R6-F1, METHOD_DESIGN section 6 and EXPERIMENT_PLAN C1/B1b, the scenario-only cluster bootstrap of R4, the 'scenario-cluster interval' of condition (1), and the interquartile-mean aggregator of variance_protocol - IS A LABELLED SENSITIVITY VIEW AND DECIDES NOTHING. Condition (1) reads on this interval.

### R6-F2 full text, condensed at R9

R6-F2 (design-kill R5 fatal 2) SEPARATES THE SESOI FROM THE POWERED ALTERNATIVE. delta_raw = 0.09 is the SESOI and the decision THRESHOLD; it is NOT the alternative the study is powered at. At a true effect exactly equal to a threshold the rule can be declared at most about half the time - the frozen simulator returns 0.023 for the complete conjunction there - so the R5 claim of 0.8 power at delta_raw was unattainable and is WITHDRAWN. THE POWERED ALTERNATIVE delta_alt IS DEFINED AS THE SMALLEST TRUE AVERAGE MISMATCH EFFECT, on a 0.01 grid strictly above max(delta_raw, 2*V_hat), at which the frozen Monte Carlo of the COMPLETE conjunction reaches power 0.80. The simulator is refine-logs/power/power_sim_r6.py (sha256 42c8064c0892ed37eb4e5364799e21c6708cff260b329cc4d522b111f542391e, 100000 simulations, generator seed 20260911); it encodes family-wide BH significance AND Delta_mis_bar >= max(delta_raw, 2*V_hat) with V_hat drawn from its own sampling distribution AND the 4-of-6 pair consistency requirement AND the >= 3-of-4 admitted-checkpoint requirement AND, per R6-F4, the whole rule holding INDEPENDENTLY FOR BOTH targeted traits AND, per R6-P1, pole concordance. At the registered PLANNING variance values (sigma_pair 0.050 at 8 seeds x 90 scenarios, rho_pair 0.40, tau_pair 0.020, tau_ckpt 0.020, sigma_pole 0.060, V 0.050 on 42 df - V set at the value this design itself anticipates) the filed output refine-logs/power/POWER_SIM_R6.json gives DELTA_ALT = 0.15 WITH POWER 0.824 AT THE REGISTERED 8 SEEDS (0.023 at 0.09, 0.332 at 0.12, 0.985 at 0.18); 12 seeds would give 0.907 at the same delta_alt and do not move delta_alt on the grid, so the seed count stays at 8 unless the development-split components require the branch below. THE STUDY IS THEREFORE POWERED FOR EFFECTS OF 0.15 SCALE POINTS OR MORE, AND THIS IS STATED IN THE PAPER RATHER THAN HIDDEN: the pre-registered point prediction of 0.12 (preregistration.predicted_result) lies BELOW delta_alt, its simulated power is 0.332, and consequently a failure to declare H1 anywhere in the 0.09-0.15 band is pre-registered as UNINFORMATIVE - never as evidence that persona traits transfer across languages. Because of that band the study's primary deliverable is stated as the ESTIMATE AND INTERVAL of Delta_mis_bar for every admitted checkpoint and trait, reported in full whatever the conjunction returns, with the conjunctive rule as the confirmatory declaration on top of it. THE PLANNING VALUES ARE PLACEHOLDERS: the same frozen script is re-run on the development-split variance components BEFORE any evaluation generation and the delta_alt it returns then is the registered one. DETERMINISTIC BRANCH, decided at that moment and never after: if power at that delta_alt is below 0.80 at 8 seeds, the seed count rises to the smallest sufficient value up to the registered cap of 12; the scenario lever does not exist, because the corpus holds exactly 90 evaluation scenarios; if 12 seeds still do not suffice, delta_alt is RAISED to the smallest simulated value reaching 0.80 and the enlarged underpowered band is declared in the abstract. The 2*V_hat limb is never removed to buy power; the R5 removal branch is DELETED. No condition may be added, removed or re-thresholded once evaluation generation has started.


### R8-F1 full text, superseded by R9-F2 (archived at R9)

R8-F1 (design-kill R7 fatal 1) ADDS THE DIAGONAL-INDUCIBILITY GATE THAT MAKES 'RETENTION LOSS' MEAN WHAT IT SAYS. E*(p,d) = 0.5[E_high - E_low] cancels the shared neutral, so a NULL diagonal followed by pole reversal could previously satisfy every H1 condition without any same-language effect to lose. Registered gate, evaluated BEFORE any mismatch contrast may support H1 and inside the primary BH family of multiplicity_correction: an ordered pair (p,d) may contribute to Delta_mis_bar ONLY IF, in the SAME checkpoint, (i) both pole-versus-neutral diagonal effects at the source language are ALIGNED in the registered direction - E_high(p,p) - E_neutral(p,p) > 0 and E_neutral(p,p) - E_low(p,p) > 0 - each with a BH-adjusted lower confidence bound above 0, and (ii) the diagonal pole-aligned effect E*(p,p) has a BH-adjusted LOWER confidence bound above the pre-registered PRACTICAL INDUCIBILITY FLOOR f_diag = 0.18 raw scale points, fixed from the development split before any evaluation generation and equal to 2 x delta_raw, so a mismatch shortfall of delta_raw can exist as a fraction of a real effect. Pairs failing (i) or (ii) are NOT ELIGIBLE and are excluded from Delta_mis_bar and from the consistency count; if fewer than 4 of the 6 ordered pairs are eligible in a checkpoint, that checkpoint cannot count toward condition (1). If fewer than 3 checkpoints remain, outcome (C) INCONCLUSIVE of pico.hypothesis.falsifier applies. (Elig) is amended accordingly: it no longer admits diagonal effects that are not bounded away from zero into the unnormalized confirmation.

### R7-F1 full text, condensed at R9 (archived at R9)

R7-F1 (design-kill R6 fatal 1) FIXES ONE EXECUTABLE RUN TENSOR. Translation engine and persona wording are NOT crossed with seed and add NO runs: every statement that they are 'balanced-completely crossed with seed' is VOID. Within each (checkpoint, cell, setting) the 90 evaluation scenarios x 8 seeds = 720 generation units are partitioned by the single frozen randomization schedule into the 9 (engine, wording) blocks at exactly 80 units each, so each engine and each wording carries exactly 240 units, engine is orthogonal to wording, to seed and to scenario stratum, and within every scenario the 8 seeds receive engines in counts 3/3/2 with the deficient engine rotated across scenarios (wording independently, same rule). 9 cells x 5 settings x 4 checkpoints x 720 = 129 600 transcripts, exactly outcome.n_runs. The full indexed allocation is generated before any generation and frozen as refine-logs/NUISANCE_SCHEDULE.json (sha256 67133fbe220f0e890fdf31171a9bb048186da058327ccf1927603b1f2d855c88, 129 600 units) by refine-logs/schedule/make_nuisance_schedule.py (sha256 ac5cea281392c1451d8092c96027994308c1c8e8661a074666776f2d36d2c238), both hashes recorded in the preregistration. ESTIMATOR: eq1's cell effect is the equal-weight (1/3) combination over the three FIXED engine strata of the within-stratum mean of the matched (persona arm - neutral arm) difference, the neutral arm matched by (scenario, seed, engine); there is no separate variant dimension to average over. Wording is a nuisance balanced inside each engine stratum and is resampled with its scenario and seed. Inference is unchanged: two-way cluster bootstrap over scenarios and seeds WITHIN fixed engine strata, engines never resampled, plus the engine sign-concordance check.


## Archived from METHOD_DESIGN §10 — R8 — answering the R7 design-kill FAIL

1. **R8-F1 (fatal 1)** — a multiplicity-controlled **diagonal-inducibility gate**: a pair may support H1 only
   if both aligned pole-vs-neutral diagonal effects are positive with BH-adjusted lower bounds above 0 and
   `E*(p,p)`'s BH-adjusted lower bound exceeds the practical floor f_diag = 0.18 = 2·δ_raw. "Retention loss"
   can no longer be declared without an established same-language persona effect to lose. (Elig) is amended.
2. **R8-F2 (fatal 2)** — ONE uniquely executable estimator-and-interval algorithm (§6); the interquartile
   mean, the two-way scenario×seed bootstrap and the scenario-only bootstrap are labelled sensitivities that
   decide nothing, and every artifact now states the same procedure.
3. **R8-F3 (fatal 3)** — every admission-affecting rater/language check moves to an **independent pre-B1
   qualification set**; admitted checkpoints and the primary family freeze before any confirmatory transcript
   exists; the confirmatory-core subsample is non-gating.
4. **R8-F4 (fatal 4)** — one falsifier of record; §1, §2, `hypotheses[0].falsifier`, `dummy_baseline` and
   EXPERIMENT_PLAN B2 point at it, the round-trip tie is deleted as a refutation path, and
   `refine-logs/consistency_check.py` fails the pre-registration if a competing path reappears.
5. **R8-P1 / R8-P2 / R8-P3** — H1 narrowed everywhere to effects ≥ 0.15; the practical threshold becomes a
   statistical claim on both limbs with V̂'s uncertainty propagated; the 12 development-touched BFI-2 items
   are excluded from the confirmatory H2 score.

---



## R11 consolidation (2026-09-11) — the superseded R4→R10 amendment stack, archived verbatim

The text below was pico.preregistration.decision_rule immediately before the R11 consolidation. It is retained for provenance only and has NO force; the R11 consolidated specification in the card governs.

~~~
R4 rewrites the rule around the pair-specific mismatch estimand, an intention-to-treat sampling rule, outcome-specific channels, and the removal of H3 from the powered claims.
H1 is supported only if ALL of (conditions as amended by R8-F1, R8-F2 and R8-P2 below, which govern wherever the older wording differs): (1) the pre-registered average mismatch contrast Delta_mis_bar = mean over the R8-F1-ELIGIBLE ordered pairs of Delta_mis(p,d) is POSITIVE with a family-wide BH-adjusted interval excluding zero - the interval being the one and only R8-F2 wild-cluster-bootstrap-t interval - in at least 3 of the 4 checkpoints ADMITTED BEFORE B1 UNDER R8-F3; (2) and (4) TOGETHER, as amended by R8-P2: the BH-adjusted LOWER bounds of BOTH Delta_mis_bar - delta_raw and Delta_mis_bar - 2*V_hat exceed 0, where delta_raw = 0.15 x median|Delta(s,s)| = 0.09 on the unnormalized scale, fixed from the development split before any evaluation generation, and V_hat is recomputed inside every bootstrap replication; the point comparison Delta_mis_bar >= max(delta_raw, 2*V_hat) is retained as a descriptive summary only; (3) CONSISTENCY: in each checkpoint that counts toward (1), at least 4 of the 6 ordered pair-specific mismatch contrasts - counted among ELIGIBLE pairs only - are positive in point estimate, so the result cannot be carried by a single language pair; and (5) POLE CONCORDANCE (R6-P1). THE R3 TIE RULE IS DELETED: the round-trip translation control no longer rejects H1 and is a sensitivity analysis only.
H2 is supported when the WITHIN-TRANSCRIPT instrument-language contrast on the self-report channel carries the loss under the same standard within its own family while the behavior-channel mismatch contrasts do not, AND the matched text-quality perturbation at fixed final language does not by itself reproduce the contrast. H2b is a measurement control on identical frozen transcripts and declares nothing about behavior.
H3 IS NOT A POWERED CONFIRMATORY CLAIM (design-kill R3 partial 7). Per-pair Holm-adjusted TOSTs at epsilon_raw ~ 0.045 are reported descriptively with their simulated power stated beside them; full language-generality is never declared; failure to reject a shortfall is never read as full transfer; the default reading of an inconclusive pattern is H3 UNRESOLVED.
INTENTION-TO-TREAT SAMPLING RULE (design-kill R3 finding 4), locked before any generation. The analysed unit is the FIRST PLANNED GENERATION for every (cell, setting, scenario, seed) of the registered tensor; there is no second attempt and no substitution. Pre-defined outcomes for non-conforming responses: (a) a response in the WRONG LANGUAGE is analysed AS PRODUCED — wrong-language output is an outcome of the manipulation, not a failure of it — and its rate is reported per cell as a registered secondary outcome; (b) an UNPARSEABLE inventory item is scored at the scale midpoint, which is conservative toward the null, and the per-cell unparseable rate is reported; an administration with more than 30% unparseable items is scored entirely at the midpoint rather than dropped; (c) an EMPTY or truncated transcript is scored by the rater as produced. REPLACEMENT IS FORBIDDEN except for PRESPECIFIED INFRASTRUCTURE FAILURES — API error, timeout, or a zero-length return caused by the serving stack — which are re-run with the SAME seed and the same prompt, logged with their cause, and capped at 2% of any cell; a cell exceeding that cap is reported as degraded. No regeneration may depend on the value of any outcome. Compliance-conditioned (per-protocol) estimates, restricted to responses in the intended language, are reported as SECONDARY beside every primary result and never substitute for it.
The omnibus grid (C) and the resource-rank slope enter no hypothesis condition. A cell failing (Elig) contributes its unnormalized difference and no ratio. If the behavior channel fails rater qualification, confirmatory H1 is UNTESTABLE (R6-F5); there is no endpoint substitution. R5-F3 (design-kill R4 fatal 3) DEFINES THE REPEATED-CONDITION VARIABILITY V AND POWERS THE COMPLETE RULE. V_{m,k} is the REPEATED-CONDITION STANDARD DEVIATION of the pole-aligned cell effect: across the 8 registered seeds - repeated identical-condition measurements differing only in the registered nuisance randomization - of the scenario-averaged E*(p,p) in the three DIAGONAL cells of L3, pooled over cells and over the two targeted traits' persona settings. Estimator, frozen with the pre-registration: V_hat = sqrt( (1/(C*(S-1))) * sum over the C pooled cell-settings and the S = 8 seeds of (E*_hat_{c,s} - E*_bar_c)^2 ), computed on the evaluation split from the same generations the confirmatory contrasts use, reported with its chi-square 95% interval. The decision threshold is Delta_mis_bar >= max(delta_raw, 2 * V_hat) - one number produced by the frozen analysis code - and the rule's sensitivity to the endpoints of V_hat's 95% interval is reported beside it. The frozen Monte Carlo encodes the COMPLETE conjunction and its deterministic branch: stated once in R6-F2 below. R5-F2 + R6-F1, CONDENSED AT R8 (full text archived in refine-logs/DESIGN_REVISION_HISTORY.md): the confirmatory family is the pair-specific mismatch contrasts and their average over ELIGIBLE pairs and NOTHING else - no confirmatory read on any main effect, no round-trip-tie rule, no R3 main-effect family - while THE LANGUAGE FACTORIAL IS STILL RUN IN FULL as the gated second-level (behavior P x D), the self-report P x D x I family (H2) and the rater-language factor (H2b). The three forward-translation engines are a FROZEN, EXHAUSTIVE, FIXED SET: they are never resampled, resampling happens WITHIN each engine stratum, and the confirmatory statistic is the equal-weight 1/3 average of the three within-stratum estimates (R8-F2 states the one complete algorithm; the R6-F1 two-way scenario x seed bootstrap is now a sensitivity view). SCOPE IS NARROWED ACCORDINGLY: every confirmatory result holds UNDER THE REGISTERED THREE-ENGINE TRANSLATION ENSEMBLE (Google Cloud Translation v3, Microsoft Azure AI Translator v3.0, NLLB-200-3.3B at the pinned revision) and generalisation to arbitrary translation systems is NOT claimed. ENGINE INVARIANCE IS A REPORTED CHECK, NOT AN INFERENCE: Delta_mis_bar is re-estimated within each engine stratum and the three point estimates must be SIGN-CONCORDANT, else the headline claim is narrowed in the title, abstract and discussion to name the discordant engine. The R5 multiway bootstrap over translation variants is DELETED from every artifact.
R6-F2 (design-kill R5 fatal 2), CONDENSED AT R9 (full text archived in refine-logs/DESIGN_REVISION_HISTORY.md). delta_raw = 0.09 is the SESOI and, with 2*V_hat, the DECISION THRESHOLD AND THE TESTED BOUNDARY - not the alternative the study is powered at: at a true effect exactly on a threshold the complete conjunction declares about 0.023 of the time, so the R5 claim of 0.8 power at delta_raw is WITHDRAWN. delta_alt is defined as the smallest true average mismatch effect, on a 0.01 grid strictly above max(delta_raw, 2*V_hat), at which the frozen Monte Carlo of the COMPLETE conjunction reaches power 0.80. The simulator refine-logs/power/power_sim_r6.py (sha256 42c8064c0892ed37eb4e5364799e21c6708cff260b329cc4d522b111f542391e, 100000 simulations, generator seed 20260911) encodes family-wide BH significance AND the max(delta_raw, 2*V_hat) threshold with V_hat drawn from its own sampling distribution AND the 4-of-6 pair consistency side-condition AND the >= 3-of-4 admitted-checkpoint requirement AND (R6-F4) the whole rule holding independently for BOTH targeted traits AND (R6-P1) pole concordance. At the registered planning values (sigma_pair 0.050 at 8 seeds x 90 scenarios, rho_pair 0.40, tau_pair 0.020, tau_ckpt 0.020, sigma_pole 0.060, V 0.050 on 42 df) refine-logs/power/POWER_SIM_R6.json gives delta_alt = 0.15 with power 0.824 at 8 seeds (0.023 at 0.09, 0.332 at 0.12, 0.985 at 0.18); 12 seeds give 0.907 and do not move delta_alt on the grid, so the seed count stays at 8. HONEST-POWER STATEMENT, printed in the abstract: the tested boundary is max(delta_raw, 2*V_hat) (R9-F1), the study attains 0.80 power only at effects of about 0.15 or more, the descriptive prior expectation of 0.12 has simulated power 0.332, and a non-declaration anywhere in the 0.09-0.15 UNINFORMATIVE BAND is pre-registered as underpowered non-support - never as evidence that persona traits transfer across languages. Whatever the conjunction returns, the primary deliverable is the ESTIMATE AND INTERVAL of Delta_mis_bar for every admitted checkpoint and trait. THE PLANNING VALUES ARE PLACEHOLDERS: the frozen script, extended per R9-F2, is re-run on the development-split variance components BEFORE any evaluation generation and the delta_alt it returns then is the registered one. DETERMINISTIC BRANCH, decided at that moment and never after: if power at that delta_alt is below 0.80 at 8 seeds, the seed count rises to the smallest sufficient value up to the registered cap of 12; the scenario lever does not exist (the corpus holds exactly 90 evaluation scenarios); if 12 seeds still do not suffice, delta_alt is RAISED to the smallest simulated value reaching 0.80 and the enlarged uninformative band is declared in the abstract. The 2*V_hat limb is never removed to buy power; the R5 removal branch is DELETED. No condition may be added, removed or re-thresholded once evaluation generation has started.
R6-F3 (design-kill R5 fatal 3) FIXES THE PAIR SET AND THE ADMISSION RULE UNDER COMPETENCE EXCLUSIONS. A checkpoint is ADMITTED TO CONFIRMATION only if it clears the pre-registered target-language competence screen in ALL THREE L3 languages; a checkpoint failing any one is reported descriptively and contributes to NO confirmatory condition. Every admitted checkpoint is therefore evaluated on the SAME complete set of 6 ordered pairs, so Delta_mis_bar and the 4-of-6 consistency rule are defined identically everywhere and no two checkpoints are compared on different pair sets. The R5 clauses that dropped individual checkpoint x language cells from condition (1), or removed a language while leaving the six-pair average in place, are DELETED. IF FEWER THAN THREE CHECKPOINTS QUALIFY, H1 IS DECLARED UNTESTABLE: the >= 3-of-4 requirement is not re-scaled and no reduced-checkpoint variant is substituted. The screen runs once, on development data only, frozen before any evaluation generation.
R6-F4 (design-kill R5 fatal 4) SUPPLIES THE MISSING QUANTIFIER ACROSS TARGETED TRAITS. THE COMPLETE H1 RULE - conditions (1) through (4) with the >= 3-of-4 checkpoint requirement - MUST HOLD INDEPENDENTLY FOR EACH OF THE TWO PRE-REGISTERED TARGETED TRAITS; the R5 rule quantified only over checkpoints. No cross-trait pooling, no equal-weight cross-trait estimand and no 1-of-2 allowance exists. If the rule holds for exactly one trait, H1 is NOT supported: the study reports a TRAIT-SPECIFIC result naming the trait it holds for and the trait it fails for, in the title, abstract and contribution list. Power is computed for this conjunction (R6-F2), which is why the powered alternative exceeds the single-trait value.
R6-F5 (design-kill R5 fatal 5) REMOVES THE ENDPOINT-SWITCHING FALLBACK: the confirmatory endpoint is the behavior-channel trait score of outcome.primary_metric and nothing may be substituted for it; the self-report channel and the rater-free indicators remain pre-specified secondary analyses. SUPERSEDED IN ITS GATE BY R7-F3 BELOW, which states the single admission rule for the behavior channel.
R6-P1 (design-kill R5 partial 1) STOPS THE NEUTRAL ARM FROM CANCELLING SILENTLY. Because E*(p,d) = 0.5*[E_high - E_low] shares one neutral arm B across both poles, B cancels algebraically and the pole-averaged endpoint measures HIGH-VERSUS-LOW POLE SEPARATION rather than a persona-versus-neutral shift. The endpoint is unchanged but the claim is gated on evidence that it is a persona effect: for every admitted checkpoint and targeted trait, BOTH pole-specific persona-minus-neutral mismatch contrasts - [E_high(p,p) - B(p,p)] - [E_high(p,d) - B(p,d)] and its low-pole mirror at sigma_low = -1 - must be POSITIVE in point estimate (POLE CONCORDANCE); this is registered condition (5) of H1 and its cost is carried in the power simulation. Where they disagree the result is reported as a pole-separation effect, not a persona-versus-neutral shift, and the claim is narrowed. The R5 pole-heterogeneity family is retained as a separate secondary family.
FALSIFIER OF RECORD - NOT RESTATED HERE. The one exhaustive three-way rule lives in pico.hypothesis.falsifier and governs every artifact: (A) supported iff this whole conjunction holds; (B) REFUTED only by the equivalence branch; (C) inconclusive otherwise, including any non-declaration in the 0.09-0.15 band. R8-F4: this field states NO independent falsifier and NO competing refutation path.
R7-F1 (design-kill R6 fatal 1) FIXES ONE EXECUTABLE RUN TENSOR. Translation engine and persona wording are NOT crossed with seed and add NO runs; every statement that they are 'balanced-completely crossed with seed' is VOID. Within each (checkpoint, cell, setting) the 90 scenarios x 8 seeds = 720 generation units are partitioned by the single frozen randomization schedule into the 9 (engine, wording) blocks at exactly 80 units each, so each engine and each wording carries exactly 240 units, engine is orthogonal to wording, to seed and to scenario stratum, and within every scenario the 8 seeds receive engines in counts 3/3/2 with the deficient engine rotated across scenarios (wording independently, same rule). 9 cells x 5 settings x 4 checkpoints x 720 = 129 600 transcripts, exactly outcome.n_runs. The full indexed allocation is frozen before any generation as refine-logs/NUISANCE_SCHEDULE.json (sha256 67133fbe220f0e890fdf31171a9bb048186da058327ccf1927603b1f2d855c88, 129 600 units) by refine-logs/schedule/make_nuisance_schedule.py (sha256 ac5cea281392c1451d8092c96027994308c1c8e8661a074666776f2d36d2c238), both hashes recorded in the preregistration. THE ENGINE LABEL IS A PROPERTY OF THE GENERATION UNIT and is carried through every step of R9-F3; eq1's cell effect is the equal-weight (1/3) combination over the three FIXED engine strata of the within-stratum mean of the matched (persona - neutral) difference, the neutral arm matched by (scenario, seed, engine). Wording is a nuisance balanced inside each stratum. Inference: R9-F3.
R7-F3 (design-kill R6 fatal 3) BOUNDS RATER ATTENUATION BELOW THE SESOI AND MAKES THE ARTIFACT CONTROLS A DETERMINISTIC GATE. (i) SCORE CONSTRUCTION: every confirmatory contrast is formed on the ANCHOR-CALIBRATED rater scale - each language's rater output affinely re-calibrated by the anchor slope and intercept estimated on the multilingual anchor set before B1, the coefficients frozen and hashed before any evaluation transcript is scored. (ii) MARGINS: the anchor slope must lie in 0.94-1.06 and the intercept within +-0.03 in EVERY language, with a cross-language slope-equivalence margin of +-0.05 (was 0.85-1.15 / +-0.1 / unspecified), so worst-case language-dependent attenuation of the assumed 0.60 diagonal effect is at most 0.036, far below delta_raw = 0.09. (iii) DETERMINISTIC GATE, replacing all 'qualifies the interpretation' language and superseding the narrower R6-F5 gate: failure of ANY of the anchor bounds, the cross-language slope equivalence, the perturbation-pair sensitivity equivalence, or the H2b rater-language artifact check, in a given checkpoint-language, REMOVES that checkpoint from the admitted set for the behavior channel; if fewer than 3 of the 4 checkpoints remain admitted the behavior channel is NOT ADMITTED and H1 falls to outcome (C) INCONCLUSIVE - never reported as refuted on that ground. No calibration or margin may be changed after any evaluation transcript is scored.
R8-F1 (design-kill R7 fatal 1) ADDED THE DIAGONAL-INDUCIBILITY GATE so that 'retention loss' always means the loss of an ESTABLISHED same-language effect: an ordered pair (p,d) may contribute only if, in the same checkpoint, both pole-versus-neutral diagonal effects at the source language are aligned in the registered direction with a lower confidence bound above 0, and E*(p,p) has a lower confidence bound above the inducibility floor f_diag = 0.18 = 2 x delta_raw, fixed from the development split. ITS EVALUATION IS MOVED WHOLESALE TO THE PRE-B1 QUALIFICATION SET BY R9-F2 BELOW, which is the operative statement; the R8-F1 clauses placing the gate on evaluation-split estimates and inside the primary BH family are VOID. Fewer than 4 eligible pairs in a checkpoint: it cannot count toward condition (1); fewer than 3 checkpoints: outcome (C). (Elig) is amended accordingly.
R8-F2 AS REWRITTEN BY R9-F3 (design-kill R8 fatal 3) MAKES THE CONFIRMATORY ANALYSIS UNIQUELY EXECUTABLE AND CONSISTENT WITH THE R7-F1 RUN TENSOR. The earlier ordering - average the 8 seeds first, stratify by engine afterwards - is VOID: it discarded the unit-level engine label that eq1 requires. THE ONE ALGORITHM, in this order and no other: (1) UNIT SCORE: for every generation unit u = (cell, setting, scenario, seed) compute the channel score, carrying u's frozen engine label e(u) and wording label w(u) from refine-logs/NUISANCE_SCHEDULE.json; no aggregation of any kind precedes this step. (2) MATCHED UNIT DIFFERENCE: form the persona-minus-neutral difference at UNIT level, the neutral arm matched to the persona arm on (scenario, seed, engine) - the schedule assigns the same engine to both arms of a matched unit by construction - giving one difference per (cell, setting, scenario, seed) with its engine label intact. (3) WITHIN-STRATUM AGGREGATION: within each of the three FIXED engine strata e, average the matched differences over the (scenario, seed) units assigned to e (240 per cell-setting), giving the within-stratum cell effect E_q^(e); wording is balanced inside the stratum and is never aggregated over separately. (4) POLE ALIGNMENT AND CONTRAST WITHIN STRATUM: form E*^(e)(p,d) = 0.5[E_high^(e) - E_low^(e)] and then Delta_mis^(e)(p,d) = E*^(e)(p,p) - E*^(e)(p,d), both inside the same stratum. (5) ENGINE COMBINATION: Delta_mis(p,d) = (1/3) sum over the three strata of Delta_mis^(e)(p,d) - this and only this is eq1's equal-weight combination. (6) PAIR AVERAGE: average the pairs that were declared ELIGIBLE BEFORE B1 by R9-F2 to Delta_mis_bar. (7) INTERVAL AND TEST: WILD CLUSTER BOOTSTRAP-t over the 8 SEED CLUSTERS within each engine stratum, Webb 6-point weights, 9999 replications, NULL IMPOSED, studentized by the cluster-robust standard error, the three strata recombined at weight 1/3 inside every replication, with the registered bootstrap seed; the 90 evaluation scenarios are FIXED, NOT RESAMPLED (R9-P4). (8) Benjamini-Hochberg at 0.05 family-wide over the primary family of multiplicity_correction, frozen before B1; the reported interval is taken at the BH-adjusted level. Every alternative named anywhere - the two-way scenario x seed cluster bootstrap of R6-F1, METHOD_DESIGN section 6 and EXPERIMENT_PLAN C1/B1b, the scenario-only cluster bootstrap of R4, the 'scenario-cluster interval' of condition (1), the interquartile-mean aggregator of variance_protocol and the seed-first ordering of the original R8-F2 - IS A LABELLED SENSITIVITY VIEW AND DECIDES NOTHING. Condition (1) reads on this interval.
R8-P2 (design-kill R7 partial 2) MAKES THE PRACTICAL THRESHOLD A STATISTICAL CLAIM, NOT A POINT COMPARISON. Conditions (2) and (4) are replaced by ONE joint requirement: the BH-adjusted LOWER confidence bounds of BOTH Delta_mis_bar - delta_raw AND Delta_mis_bar - 2*V_hat must exceed 0. V_hat's own uncertainty is PROPAGATED, not ignored: V_hat is recomputed inside EVERY wild-bootstrap replication from that replication's resampled seed clusters, so the two limbs are bootstrapped jointly and their dependence is preserved. The point-estimate comparison with max(delta_raw, 2*V_hat) is retained only as a descriptive summary.
R8-F3 (design-kill R7 fatal 3) TAKES CHECKPOINT ADMISSION OFF EVALUATION-DERIVED RESULTS. Every admission-affecting rater and language check runs on an INDEPENDENT PRE-B1 QUALIFICATION SET - fresh transcripts from the development scenario pool at the same checkpoints, settings and engines, disjoint from the confirmatory core in scenarios, items and seeds, and hashed into the pre-registration before B1 opens. The ADMITTED CHECKPOINT SET and the PRIMARY FAMILY are frozen at that moment and CANNOT CHANGE after any confirmatory transcript exists. The 20% rater-language subsample of the confirmatory core (H2b) survives ONLY as a NON-GATING sensitivity: a failure there is disclosed wherever the headline result is stated and narrows the claim, but never removes a checkpoint, changes the family or changes an admission. R7-F3's deterministic admission gate reads on the qualification set alone.
R8-F4 (design-kill R7 fatal 4) LEAVES EXACTLY ONE REFUTATION PATH, IN EVERY FIELD, AND ENFORCES IT MECHANICALLY. pico.hypothesis.falsifier (the R7 three-way exhaustive rule) is the falsifier of record; pico.hypotheses[0].falsifier and comparison.dummy_baseline are REWRITTEN to point at it, METHOD_DESIGN section 2 and EXPERIMENT_PLAN B2 are rewritten, and the round-trip tie rule is DELETED as a refutation path everywhere (it narrows the claim instead). A blanket 'earlier language is void' sentence is not sufficient and is not relied on. refine-logs/consistency_check.py (sha256 d429eb8fbd9dd22f) scans design_card.json, METHOD_DESIGN.md and EXPERIMENT_PLAN.md for any competing refutation path - any 'falsif'/'refut'/'rejects H1' string outside pico.hypothesis.falsifier that is not an explicit pointer to it - and EXITS NONZERO if one is found; it must pass before the pre-registration is frozen and again before analysis.
R8-P1 (design-kill R7 partial 1) abandoned the 0.12 expectation as a confirmatory target, retaining it as a descriptive prior expectation with no registered role. ITS 'claim about effects of at least delta_alt = 0.15' WORDING IS VOID: R9-F1 states the claim at the tested boundary. Non-declaration in the 0.09-0.15 band remains UNDERPOWERED NON-SUPPORT and is never reported as evidence of no effect.
R8-P3 (design-kill R7 partial 6) MAKES THE ITEM QUARANTINE REAL. The 12 development-touched BFI-2 items are EXCLUDED from the confirmatory H2 self-report score, which is computed on the remaining 48 items only; the 12 are reported separately as a labelled sensitivity. comparison.datasets, controls.leakage_arguments, preregistration.analysis_plan, METHOD_DESIGN and EXPERIMENT_PLAN all state the same disjoint split, and the executable scoring code reads the frozen 48-item list.
R9-F1 (design-kill R8 fatal 1) STATES THE CLAIM AT THE BOUNDARY THE RULE ACTUALLY TESTS. The confirmatory claim of H1 is that the average mismatch shortfall Delta_mis_bar EXCEEDS THE TESTED BOUNDARY max(delta_raw = 0.09, 2*V_hat), and nothing larger. EVERY sentence in EVERY artifact that made 'effects of at least delta_alt = 0.15' the confirmatory claim - hypothesis.statement, hypotheses[0], preregistration.predicted_result, title, abstract, contribution list, METHOD_DESIGN and EXPERIMENT_PLAN - IS VOID AND IS REWRITTEN. delta_alt = 0.15 is a POWER STATEMENT ONLY: the effect size at which the complete conjunction attains 0.80 power (R6-F2). It is never a threshold, never a claim boundary and never appears in a decision. Conditions (1) and (2)/(4)-as-amended-by-R8-P2 ARE the test of the boundary: the BH-adjusted lower bounds of both Delta_mis_bar - delta_raw and Delta_mis_bar - 2*V_hat must exceed 0. Conditions (3) (4 of 6 eligible pair contrasts positive in point estimate) and (5) (pole concordance) are ROBUSTNESS SIDE-CONDITIONS that can only make declaration harder; they set no threshold, carry no claim and are never described as what the study tests. NO REPOWERING IS REQUIRED because the frozen simulator already encodes exactly this conjunction at exactly this boundary; power at the registered alternative is unchanged (0.824 at 0.15) and the honest-power statement of R6-F2 stands unchanged. A claim at 0.15 would require the BH-adjusted lower bound of Delta_mis_bar - 0.15 to exceed 0 in every counting checkpoint and trait plus a complete repowering of the enlarged rule; that is NOT DONE and NOT REGISTERED, so the 0.15 claim is WITHDRAWN everywhere.
R9-F2 (design-kill R8 fatal 2) TAKES PAIR ELIGIBILITY OFF THE CONFIRMATORY DATA. The R8-F1 diagonal-inducibility gate is a SELECTION rule and it may not be computed from the same evaluation outcomes that enter Delta_mis(p,d). It is therefore evaluated ONLY on the INDEPENDENT PRE-B1 QUALIFICATION PARTITION of R10-F3 - fresh transcripts from its 60 scenarios at the same checkpoints, settings, engines and seeds, disjoint from the confirmatory core and from the tuning partition in scenarios, items and seeds. Its two limbs are unchanged: (i) both pole-versus-neutral diagonal effects aligned in the registered direction with a lower confidence bound above 0, and (ii) the diagonal pole-aligned effect E*(p,p) with a lower confidence bound above the practical inducibility floor f_diag = 0.18 = 2 x delta_raw. The resulting ELIGIBLE PAIR SET, per checkpoint and targeted trait, is FROZEN AND HASHED into refine-logs/ELIGIBLE_PAIRS.json BEFORE B1 OPENS, together with the admitted checkpoint set and the primary BH family, and the estimand Delta_mis_bar is fixed at that moment. THE DEFAULT ESTIMAND IS THE FIXED ALL-SIX-PAIR AVERAGE; a pair is dropped only by this pre-B1 rule and never afterwards. Diagonal effects estimated on the CONFIRMATORY split are reported descriptively and can NEVER change eligibility, estimand composition, the BH family or the admitted set; the R8-F1 clause placing the gate tests inside the primary BH family is VOID, because a pre-B1 selection step is not an inference on the confirmatory data (multiplicity_correction is amended accordingly). If fewer than 4 of the 6 ordered pairs are eligible in a checkpoint that checkpoint cannot count toward condition (1), and if fewer than 3 checkpoints remain, outcome (C) INCONCLUSIVE of pico.hypothesis.falsifier applies. GATE POWER IS DEMONSTRATED, NOT ASSUMED: refine-logs/power/power_sim_r9.py extends the frozen simulator with (a) this pre-B1 gate as an explicit stage on qualification-set units and (b) the resulting eligible-pair-count distribution feeding the confirmatory conjunction; it must show at least 0.90 per-pair gate power at the anticipated diagonal effect of 0.60 with the registered sigma_pair, and it reports BOTH the conditional power given the frozen eligible set AND the unconditional power including gate failure. R10-F4 replaces the R9 expansion branch (VOID): at the registered Q = 60 qualification scenarios, per-pair gate power below 0.90 makes H1 UNTESTABLE on the behavior channel, and no corpus is enlarged. Its filed output refine-logs/power/POWER_SIM_R9.json is produced before B1 together with POWER_SIM_R6.json's re-run on development-split variance components, and the delta_alt those two return is the registered one under the R6-F2 deterministic branch.
R9-F4 (design-kill R8 fatal 4) DEFINES THE BEHAVIOR RUBRIC AND SCALE AND MAKES THEIR VALIDITY A DEMONSTRATED, POWERED PRE-B1 GATE. (i) RUBRIC AND SCALE, frozen and hashed as refine-logs/RUBRIC_BEHAVIOR.md before any generation: for each targeted trait the rater scores the transcript on SIX facet items derived from that trait's BFI-2 facet definitions and written in customer-support-dialogue terms - Extraversion: sociability, assertiveness, energy level; Agreeableness: compassion, respectfulness, trust; two items per facet - each on an INTEGER 1-7 scale with written behavioral anchors at 1, 4 and 7, rendered into the transcript language through the same frozen translation pipeline as every other instrument. The trait score is the EQUAL-WEIGHT MEAN of its six items on the native 1-7 scale, before the anchor calibration of R7-F3(i). The item list IS the mapping from scored cues to the trait: no weight, no item and no cue is estimated from data, nothing outside the six items enters the score, and the rater returns the six item scores so every aggregate is reconstructible. (ii)-(iii) HUMAN CRITERION AND POWERED VALIDITY THRESHOLDS ARE RESTATED ONCE IN R10-F2, WHICH GOVERNS: the criterion is collected on the 60-scenario QUALIFICATION partition, one transcript per scenario in each of the 12 checkpoint-language cells (720 transcripts, double-annotated blind on this frozen rubric by annotators fluent in the language and blind to persona language, dialogue language, checkpoint and pole, a third adjudicating, none of them seeing the injected-delta anchor transcripts), and every convergent, discriminant and anti-stereotype threshold is estimated INSIDE its checkpoint-language cell by a scenario-cluster BCa bootstrap. The R9 transcript-level Fisher-z standard errors are VOID. (iv) GATE: failure of ANY threshold in a language removes that checkpoint-language from the admitted set under the deterministic gate of R7-F3(iii); if fewer than 3 of the 4 checkpoints remain admitted the behavior channel is NOT ADMITTED and H1 takes outcome (C) INCONCLUSIVE of pico.hypothesis.falsifier - never outcome (B). No rubric item, anchor, cue list or threshold may be changed after any confirmatory transcript is scored. The rater-free indicators of controls.evaluator_validity remain a descriptive convergence check and substitute for nothing (R6-F5).
R9-P4 (design-kill R8 partial 4) MAKES THE INFERENCE CONDITIONAL ON THE FIXED SCENARIO CORPUS AND SAYS SO EVERYWHERE. Every confirmatory interval is CONDITIONAL ON THE 90 FIXED EVALUATION SCENARIOS: scenarios are a frozen purposive corpus, they are NEVER resampled, and the only resampled cluster is the seed within an engine stratum (R9-F3 step 7). Consequently the confirmatory claim is scoped to this corpus (scope.claim_population) and NO domain-general claim about customer-support dialogue is made. Every 'scenario-clustered', 'scenario-resampled' or two-way scenario x seed wording in controls.leakage_arguments.L3_distribution_match, outcome.effect_size_reported, outcome.variance_protocol, METHOD_DESIGN and EXPERIMENT_PLAN is VOID as an inferential statement and survives only as a labelled sensitivity view; scenario-level heterogeneity is reported descriptively.
R9-P5 (design-kill R8 partial 6) NARROWS H2 TO WHAT THE FACTORIAL IDENTIFIES. Changing the instrument language changes item semantics, tokenization, comprehension, option interpretation and the stochastic response process together, so LangFac cannot isolate a passive response-scale shift or certify an unchanged latent trait state. H2 is therefore an OBSERVABLE INSTRUMENT-ADMINISTRATION-PIPELINE EFFECT: the within-transcript instrument-language contrast on the self-report channel, on frozen transcripts, with the matched text-quality perturbation arm bounding one translation-quality component. Every 'response-scale artifact' and 'underlying induced trait state is unchanged' reading in hypothesis.anti_claim, METHOD_DESIGN and EXPERIMENT_PLAN is VOID; the language factorial still runs in full as the gated second-level (behavior P x D), the self-report P x D x I family and the rater-language factor H2b.
R9-P6 (design-kill R8 partial 7) DECLARES THE SENSITIVITY LIMIT PROSPECTIVELY. The tested boundary is max(delta_raw, 2*V_hat) and the complete conjunction reaches 0.80 power only near 0.15; the descriptive prior expectation of 0.12 has simulated power 0.332 and CANNOT be reached by any lever this design owns - the scenario corpus is fixed at 90 by construction and the registered seed cap of 12 raises power at 0.12 only marginally. The study therefore registers itself prospectively as a BOUNDARY TEST WITH ~0.15 SENSITIVITY whose primary deliverable is the estimate and interval of Delta_mis_bar for every admitted checkpoint and trait, states that limit in the abstract, and pre-registers that a non-declaration in the 0.09-0.15 uninformative band is reported as such and never as evidence of language-general transfer. Raising delta_alt is not treated as a gain in sensitivity anywhere.
R10 REPAIRS THE DESIGN-KILL R9 FAIL. R10-F1 THROUGH R10-F6 GOVERN WHEREVER ANY EARLIER WORDING DIFFERS.
R10-F1 (R9 fatal 1) RESTORES THE OPPOSITE-DIRECTION REFUTATION BRANCH INSIDE THE ONE RULE. pico.hypothesis.falsifier, the falsifier of record (R8-F4), now refutes H1 through EITHER of two limbs, each applied per targeted trait over the admitted checkpoints with the SAME BH family, the SAME R10-F6 intervals and the SAME at-least-3-of-4 quantifier: (B1) EQUIVALENCE - the adjusted interval for Delta_mis_bar lies entirely inside +-delta_raw = 0.09; or (B2) REVERSAL - the adjusted interval lies entirely BELOW -delta_raw. A practically large mismatch effect in the direction opposite to the registered one therefore refutes the directional claim instead of being absorbed into inconclusive, and the rule is symmetric in sign. Outcome (C) INCONCLUSIVE keeps every remaining case, including non-declaration in the 0.09-0.15 band. The power simulator reports the (B2) branch beside (B1). No other artifact states an independent path (consistency_check.py).
R10-F2 (R9 fatal 2) MAKES THE EVALUATOR-VALIDITY GATE A SCENARIO-CLUSTER-LEVEL, PER-CHECKPOINT-LANGUAGE ESTIMATE. R9-F4 (ii) and (iii) are amended: the human criterion is collected on the R10-F3 QUALIFICATION PARTITION of Q = 60 scenarios, and each of the 12 checkpoint-language cells (4 checkpoints x 3 dialogue languages) receives ONE transcript per qualification scenario, so a cell carries 60 INDEPENDENT SCENARIO CLUSTERS and no transcript is shared between two cells' gates: 720 qualification transcripts, each double-annotated blind with a third adjudicating. EVERY GATE IS ESTIMATED INSIDE ITS OWN CELL by a BCa bootstrap that RESAMPLES SCENARIOS (9999 replications, generator seed 20260911): convergent - the cell's rater-versus-criterion correlation must have a 95% cluster-bootstrap LOWER BOUND above 0.40; pooled across cells, with each scenario counted once by averaging within scenario, the convergent lower bound must exceed 0.60; discriminant and the anti-stereotype criterion keep the R9-F4(iii) thresholds but are computed by that same scenario-level bootstrap, paired within scenario. TRANSCRIPT-LEVEL FISHER-z STANDARD ERRORS (SE_z = 0.053 at n = 360, 0.132 at n = 60) ARE VOID EVERYWHERE: they pseudoreplicated shared scenarios and pooled checkpoints under a checkpoint-language-specific admission rule. A cell failing any threshold leaves the admitted set under R7-F3(iii); below 3 admitted checkpoints the behavior channel is NOT ADMITTED and outcome (C) applies.
R10-F3 (R9 fatal 3) PARTITIONS THE SUPPORT CORPUS BEFORE ANY TUNING AND CONFINES EVERY DEVELOPMENT CHOICE TO THE TUNING PARTITION. The support corpus is partitioned ONCE, before any prompt is written and before any pipeline is run, into TUNING = 30 scenarios and QUALIFICATION = 60 scenarios, mutually disjoint and disjoint from the 90 EVALUATION scenarios, frozen and hashed as refine-logs/SCENARIO_PARTITION.json. ALL development-time selection - prompt and persona wording, parsing rules, scorer and rubric configuration, translation-pipeline settings, the dispersion re-estimate and the R6-F2 variance components - runs on the TUNING partition ONLY. The QUALIFICATION partition is untouched until it is read once, before B1, for exactly two gates: the R9-F2 pair-eligibility gate and the R10-F2 evaluator-validity gate. The R5-P4 18/12 scorer-tuning/scorer-qualification quarantine is VOID: it restricted scorer tuning alone while prompt and pipeline selection still touched the whole development pool, so diagonal inducibility was not independently established.
R10-F4 (R9 fatal 4) REPLACES THE IMPOSSIBLE EXPANSION BRANCH WITH A FIXED, BUDGETED CORPUS AND A DETERMINISTIC UNTESTABLE OUTCOME. The registered corpus is exactly 180 support scenarios = 30 TUNING + 60 QUALIFICATION + 90 EVALUATION; outcome.n_runs budgets the qualification generations and the 720 human annotations. The R9-F2 clause raising qualification scenarios 'from 30 toward the registered cap of 60' is VOID - no scenario source, run tensor or budget ever supported it. If power_sim_r9.py returns per-pair gate power below 0.90 at the registered Q = 60, the consequence is DETERMINISTIC AND REGISTERED HERE: H1 is UNTESTABLE on the behavior channel and takes outcome (C); the corpus is NOT enlarged, no gate threshold is relaxed and no scenario is reused.
R10-F5 (R9 fatal 5) REGISTERS THE ENTIRE SEED SCHEDULE BEFORE M1. The seed list is 20260911 + {0..11}. For EVERY allowed count S in {8, 9, 10, 11, 12} the run tensor and the nuisance allocation are generated by refine-logs/schedule/make_nuisance_schedule.py BEFORE M1 and hashed into NUISANCE_SCHEDULE.json as five indexed blocks, each satisfying the R7-F1 orthogonality rule at its own count and each NESTED in the next, so the first eight seeds' allocation is identical in all five: S = 8 is 129 600 confirmatory transcripts and every further seed adds 9 x 5 x 4 x 90 = 16 200, to 194 400 at S = 12. M2 SELECTION IS DETERMINISTIC: S is the SMALLEST count in that list whose simulated power at delta_alt reaches 0.80 under R6-F2, evaluated once on the tuning-partition variance components before any evaluation generation; if no count reaches it, S stays 8 and delta_alt is raised exactly as R6-F2 registers. type_ext.llm_eval.decoding_config lists the twelve seeds and marks the first eight the default block; outcome.compute_budget states the S = 8 budget and the S = 12 ceiling.
R10-F6 (R9 fatal 6) MAKES THE INTERVAL AND THE MULTIPLICITY ALGORITHM UNIQUELY EXECUTABLE, AMENDING R9-F3. (i) The resampling unit is the SEED CLUSTER (scenario, seed) and its wild weight is drawn ONCE PER BOOTSTRAP REPLICATION AND SHARED GLOBALLY: the same Rademacher weight multiplies that cluster's residual in every engine stratum, cell, setting, checkpoint, trait and contrast in which the cluster appears, so dependence across contrasts that share clusters is preserved by construction; 9999 replications, generator seed 20260911. (ii) Every statistic is studentized by the cluster-robust standard error computed on those same clusters, and V_hat is recomputed inside each replication from the resampled clusters, so the threshold's sampling variability travels jointly with the estimate. (iii) THE FAMILY is the set of (admitted checkpoint x targeted trait) mismatch tests, enumerated and frozen before B1 in ELIGIBLE_PAIRS.json. The three shifted nulls of a member - Delta_mis_bar equal to 0, to delta_raw and to 2*V_hat - are tested by that same studentized bootstrap and combined INSIDE the member by their MAXIMUM p-value (an intersection-union across the shifts), so BH at level 0.05 runs over exactly one p-value per member. (iv) THE INTERVAL IS DEFINED BY INVERSION: the reported set for a member is every theta on a 0.001 grid whose studentized bootstrap test of Delta_mis_bar = theta is not rejected at that member's BH-adjusted level, which gives BY-style FCR-controlled simultaneous intervals over the selected members; the unadjusted per-member interval is reported beside it as descriptive. Every earlier phrase asking for 'an interval at the BH-adjusted level' without this inversion rule is VOID.
~~~


## R11 -> R12: text superseded by the R11 design-kill repairs (archived verbatim, NO FORCE)

Repairs: (1) the pre-B1 eligibility gate now has an executable, separately budgeted 28 800-unit qualification run tensor (H2), distinguished from the 720-transcript F2 annotation subsample; (2) the confirmatory resampling unit is the SEED cluster alone, G = S; (3) the primary family is one member per admitted checkpoint x targeted trait with Holm and max-T simultaneous intervals, the 6 pair contrasts forming their own gated family. Superseded text follows.

### superseded decision_rule line 27 (R11)

F1. PAIR ELIGIBILITY (diagonal inducibility). Evaluated on fresh transcripts from the 60 QUALIFICATION scenarios at the same checkpoints, settings, engines and seeds, disjoint from the confirmatory core and from the tuning partition. Two limbs: (i) both pole-versus-neutral diagonal effects at the source language aligned in the registered direction with a lower confidence bound above 0; (ii) the diagonal pole-aligned effect E*(p,p) with a lower confidence bound above f_diag = 0.18. THE DEFAULT ESTIMAND IS THE FIXED ALL-SIX-PAIR AVERAGE; a pair is dropped only by this pre-B1 rule and never afterwards. A pre-B1 selection step is not an inference on the confirmatory data, so these gate tests are NOT members of the primary BH family. If fewer than 4 of the 6 ordered pairs are eligible in a checkpoint, that checkpoint cannot count toward condition (1); if fewer than 3 checkpoints remain, outcome (C).


### superseded decision_rule line 39 (R11)

I. THE INTERVAL AND THE MULTIPLICITY ALGORITHM ARE UNIQUELY EXECUTABLE. (i) THE RESAMPLING UNIT is the SEED CLUSTER (scenario, seed) and its wild weight is drawn ONCE PER BOOTSTRAP REPLICATION AND SHARED GLOBALLY: the same Rademacher weight multiplies that cluster's residual in EVERY engine stratum, cell, setting, checkpoint, trait and contrast in which the cluster appears, so dependence across contrasts that share clusters is preserved by construction and no shared-cluster dependence is broken by resampling contrasts separately; 9999 replications, generator seed 20260911, null imposed, the three engine strata recombined at weight 1/3 inside every replication, the 90 evaluation scenarios FIXED and not resampled. (ii) Every statistic is STUDENTIZED by the cluster-robust standard error computed on those same clusters, and V_hat is recomputed inside each replication from the resampled clusters, so the threshold's sampling variability travels jointly with the estimate. (iii) THE FAMILY is the set of (admitted checkpoint x targeted trait) mismatch tests, enumerated and frozen before B1 in ELIGIBLE_PAIRS.json. The three shifted nulls of a member - Delta_mis_bar equal to 0, to delta_raw and to 2*V_hat - are tested by that same studentized bootstrap and combined INSIDE the member by their MAXIMUM p-value (an intersection-union across the shifts), so Benjamini-Hochberg at level 0.05 runs over EXACTLY ONE p-value per member. (iv) THE INTERVAL IS DEFINED BY INVERSION: the reported set for a member is every theta on a 0.001 grid whose studentized bootstrap test of Delta_mis_bar = theta is not rejected at that member's BH-adjusted level, which gives BY-style FCR-controlled simultaneous intervals over the selected members; the unadjusted per-member interval is reported beside it as descriptive. Condition (1) and both limbs of condition (2) read on these intervals and on no other.



## Archived verbatim: METHOD_DESIGN.md §10 as it stood at R11 (NO FORCE)

## 10. R12 — the consolidated design of record

`design_card.json` → `pico.preregistration.decision_rule` is now **one self-contained specification, sections
A–N**, with **no amendment stack**; the superseded R4→R10 chain is archived verbatim in
`DESIGN_REVISION_HISTORY.md` and has **no force**. Citations of the form **rule X** here point at section X of
that specification; section N lists every void inference. The six repairs the last design-kill forced:

1. **Sign-symmetric refutation (rule C).** The falsifier of record — `pico.hypothesis.falsifier`, the sole
   rule, pointed at and never restated here — now carries two limbs, (B1) equivalence inside ±δ_raw and
   (B2) reversal entirely below −δ_raw, so a large opposite-direction effect is not protected inside
   "inconclusive". The power simulator reports the (B2) branch beside (B1).
2. **Evaluator validity estimated where it is used (rule F2).** One qualification transcript per scenario in
   each of the 12 checkpoint-language cells: 60 independent scenario clusters per cell, no transcript shared
   between two cells' gates, every threshold bootstrapped **by resampling scenarios inside its own cell**.
   Transcript-level Fisher-z standard errors, which pseudoreplicated shared scenarios and pooled checkpoints
   under a checkpoint-specific admission rule, are void (720 transcripts total).
3. **Qualification material never touched by development (rule E).** The corpus is partitioned 30 / 60 / 90
   **before any prompt is written**; all tuning, wording, scorer, threshold and variance-component choices run
   on the 30 only. The narrower scorer-only quarantine is void.
4. **No impossible contingency (rule F4).** At the registered Q = 60, per-pair gate power below 0.90 makes H1
   UNTESTABLE on the behavior channel and takes outcome (C). The corpus is **not** enlarged, no threshold is
   relaxed, no scenario is reused; the "30 toward the cap of 60" expansion branch is void.
5. **The whole seed schedule exists before M1 (rule H).** Five nested, pre-hashed allocation blocks, one per
   allowed S ∈ {8…12}; M2 picks the smallest sufficient S once, on tuning-partition variance components. No
   adaptive or unscheduled seed count exists.
6. **One executable interval and multiplicity algorithm (rule I).** The resampling unit is the **seed alone**,
   G = S clusters (8 at the default count); one globally shared Webb 6-point weight per seed per replication
   preserves dependence across every contrast that shares seeds; studentization uses the cluster-robust SE
   with the G/(G−1) correction and V̂ is recomputed inside each replication on G − 1 = 7 df. The **primary
   family is one member per admitted checkpoint × targeted trait** (≤ 8 members), its three shifted nulls
   combined inside the member by their maximum p-value, and the procedure is **Holm at 0.05 with max-T
   simultaneous bootstrap-t intervals**; the six pair-specific contrasts form their **own gated family** under
   the same procedure. BH, BH-adjusted intervals and BY-style FCR inversion are **VOID** here — BH controls a
   false-discovery rate, not the simultaneous coverage rule B reads.



## Archived verbatim: EXPERIMENT_PLAN.md §8 as it stood at R11 (NO FORCE)

## 8. R12 — the consolidated design of record

The preregistered rule is now **one self-contained specification, sections A–N** of
`design_card.json` → `pico.preregistration.decision_rule`, with **no amendment stack**; the superseded
R4→R10 chain is archived verbatim in `DESIGN_REVISION_HISTORY.md` and has **no force**. **rule X** citations
here point at section X of it; section N lists every void inference. The six repairs the last design-kill
forced are enumerated once, in `METHOD_DESIGN.md` §10. In this plan they land as §6's two-limb pointer to the
falsifier of record (rule C), B0's opening 30 / 60 / 90 partition (E), B0's per-cell scenario-cluster evaluator-validity gate on
720 blind double-annotated qualification transcripts (F2), B0's deterministic UNTESTABLE branch with no
corpus expansion (F4), §3/§4's five pre-hashed nested seed blocks (H), and §5's globally shared cluster
weights, maximum-p combination and inversion-defined intervals (I). Added cost: the 720-transcript blind
double annotation and the qualification generations on the 60-scenario partition, inside the existing B0
envelope plus annotation effort.



## Superseded verbatim: EXPERIMENT_PLAN.md block B0 before the R12 pointer-first compression (no force)

### B0 — Instrument and rater qualification, and the pre-B1 qualification gates

Establish that the apparatus works in all six languages and decide **all** checkpoint admissions, pair
eligibility and family membership *before* any evaluation generation.

**The corpus is partitioned first, and the partition is what makes B0 independent (rule E).** The 180
support scenarios are cut ONCE — **before any prompt is written and before any pipeline is run** — into
**TUNING = 30**, **QUALIFICATION = 60** and **EVALUATION = 90**, mutually disjoint, frozen and hashed as
`SCENARIO_PARTITION.json`. All development-time selection — prompt and persona wording, parsing rules,
scorer and rubric configuration, translation-pipeline settings, the dispersion re-estimate, δ_raw, f_diag,
the anchor coefficients and the variance components — runs on the **TUNING 30 only**. The QUALIFICATION 60
are untouched until they are read **once**, before B1, for exactly the two gates below; the EVALUATION 90
are untouched until B1 opens. Because no design choice was ever fitted to a qualification scenario, the
eligibility and validity gates are established on genuinely independent material.

**Rule F — the gates and their freeze.** Every admission-affecting check runs on that QUALIFICATION
partition at the same 4 checkpoints, 5 settings and three engines, hashed into the pre-registration at M1
before B1 opens; the admitted checkpoint set, the eligible pair set and the primary family freeze at that
moment, and no admission decision ever reads an evaluation-split result.

- Render BFI-2/TIPI into the six languages through the three-translation ensemble; back-translation screen;
  record between-translation disagreement per item. Internal consistency, invariance/DIF screen, BFI-2–TIPI
  convergence per trait per language.
- Double-annotated transcripts with adjudication; automatic rater must reach Krippendorff α ≥ 0.7 in
  every language, with cross-language equivalence at a 0.05 margin.
- Two-cell re-estimate of the dispersion on the **unnormalized** confirmatory scale against the σ = 0.066
  prior (0.11 relative × the 0.60 median diagonal effect).
- **Per-language length band** (frozen at M1): fix the mean-turn-length target keeping every (LangFac) cell within ±10%
  at identical turn count, so length is matched by design, not merely disclosed.
- **Rater qualification beyond agreement** (frozen at M1): multilingual **anchor pairs** recovered with slope
  **0.94–1.06**, intercept within **±0.03**, slopes cross-language equivalent at **±0.05** — every confirmatory
  contrast is formed on that anchor-calibrated scale (rule F3); **perturbation pairs** must show equivalent
  detection rates across languages; trait-specific invariance established for the four rater-free indicators.
- **Freeze and hash the B3 ablation subset** (frozen at M1): IDs, selection algorithm, contrasts and reuse rules written
  and hashed here, before B1, recorded at M1.
- **Re-estimate the variance components** driving power and rerun the frozen simulators (`power_sim_r6.py`
  and the gate-aware `power_sim_r9.py` of rule F4).
- **rule F1 — decide pair eligibility here and freeze it** into hashed `ELIGIBLE_PAIRS.json` with the admitted
  checkpoint set and the primary Holm family. It is estimated on its **own executable qualification run
  tensor (rule H2): 12 checkpoint-language cells × 60 qualification scenarios × 5 settings × S seeds =
  28 800 transcripts at S = 8** (+3 600 per further seed), allocated by the sixth indexed block of
  `NUISANCE_SCHEDULE.json` under the same engine × wording orthogonality rule and scored once each by the
  frozen 24B rater at I = D — **not** on the 720-transcript rule-F2 annotation subsample drawn from it.
  Confirmatory-split diagonal effects are descriptive only.
- **rule G — qualify the behavior rubric as a construct.** Freeze `RUBRIC_BEHAVIOR.md`: six behaviorally
  anchored 1–7 facet items per targeted trait, equal-weight trait score, nothing estimated from data.
- **rule F2 — establish evaluator validity where it is used, at scenario-cluster level.** Each of the **12
  checkpoint-language cells** (4 checkpoints × 3 dialogue languages) receives **one transcript per
  qualification scenario**, so a cell carries **60 independent scenario clusters and shares no transcript
  with any other cell's gate**: **720 qualification transcripts**, each double-annotated blind on the frozen
  rubric by annotators fluent in the language and **blind to persona language, dialogue language, checkpoint
  and pole**, a third adjudicating, none of them shown the injected-delta anchors. **Every threshold is
  estimated inside its own cell** by a BCa bootstrap that **resamples scenarios** (9999 replications, seed
  20260911): cell convergent lower bound > 0.40; pooled across cells with each scenario counted once > 0.60;
  discriminant and anti-stereotype by the same paired scenario bootstrap. **Transcript-level Fisher-z
  standard errors are VOID** — they pseudoreplicated shared scenarios and pooled checkpoints under a
  checkpoint-language-specific admission rule.

**Gate (rule F, on the qualification set alone).** Failure of α ≥ 0.7, of the anchor slope/intercept bounds, of
cross-language slope equivalence at ±0.05, of perturbation-pair sensitivity equivalence, or of the
rater-language artifact check, in a checkpoint-language, **removes that checkpoint from the admitted set for
the behavior channel**. If fewer than 3 of 4 checkpoints remain admitted the behavior channel is **not
admitted** and H1 falls to outcome (C) INCONCLUSIVE of the falsifier of record — never to refutation. The 20%
rater-language subsample of the confirmatory core (B1b-rater) is a **non-gating sensitivity**: a failure there
is disclosed wherever the headline is stated and narrows the claim, but never removes a checkpoint and never
changes the family or an admission. The self-report channel and the rater-free indicators stay pre-specified
secondary analyses; there is no endpoint substitution. An indicator failing trait-specific invariance is
descriptive for that trait and leaves the confirmatory hierarchy. The seed count is **not** renegotiated here: under rule H the allocation
for every allowed S ∈ {8…12} already exists, hashed, as five nested blocks from before M1, and M2 picks the
smallest sufficient S deterministically on tuning-partition variance components. Under **rule F4**, per-pair
gate power below 0.90 at the registered Q = 60 makes H1 **UNTESTABLE** on the behavior channel — outcome
(C) — and **the corpus is not enlarged, no threshold is relaxed and no scenario is reused**. Cost:
≈110–130 GPU hours (including the 28 800-transcript rule-H2 qualification tensor), plus the annotation effort.



## Superseded verbatim: METHOD_DESIGN.md section 6 before the R12 pointer compression (no force)

## 6. Outcome, power and variance

**One canonical specification, stated once in `design_card.json`.** This section is a pointer, not a second
copy; where any wording here differs, the card wins.

| Item | Value |
| --- | --- |
| Primary metric | `Δ_mis(p,d) = E*(p,p) − E*(p,d)` (eq6 on eq1's pole-aligned cell effect), behavior arm; the confirmatory read is the pair-specific contrasts and their equal-weight average `Δ̄_mis` over eligible pairs. Full text: `outcome.primary_metric` |
| Eligibility | **rule F1** — the diagonal-inducibility gate (aligned pole-vs-neutral diagonal effects with lower bounds above 0; `E*(p,p)` lower bound above f_diag = 0.18 = 2·δ_raw) is evaluated **only on the 28 800-transcript qualification run tensor of rule H2, built on the 60-scenario QUALIFICATION partition of rule E** — a partition cut before any prompt was written, so no development-time selection ever touched it — and frozen into `ELIGIBLE_PAIRS.json` **before B1**; default estimand is the fixed all-six-pair average and the gate enters **no** confirmatory family. < 4 eligible pairs ⇒ the checkpoint cannot count; < 3 checkpoints ⇒ outcome (C) |
| Tested boundary / powered alternative | **rule C: the tested boundary — and the only thing claimed against — is max(δ_raw, 2·V̂)**, δ_raw = 0.09. **δ_alt = 0.15 is a power statement only**: 0.824 power there, 0.023 at δ_raw (`power/power_sim_r6.py`, sha256 42c8064c…391e, 100 000 draws, seed 20260911). The 0.09–0.15 band is pre-registered UNINFORMATIVE, the 0.12 expectation registers nothing, and the ≥0.15 claim wording of rule J is **VOID** (rule J) |
| Practical-significance test | **rule B(2)** — max-T simultaneous lower bounds of BOTH `Δ̄_mis − δ_raw` and `Δ̄_mis − 2·V̂` exceed 0, V̂ recomputed inside every replication |
| Equivalence margin (H3) | ε = 0.075 of the induced shift; ε_raw ≈ 0.045; simultaneous Holm-adjusted TOSTs |
| α / power / tails | 0.05 / 0.80 at δ_alt / two-tailed, paired |
| Corpus partition | **rule E** — 180 support scenarios cut ONCE, before any prompt is written, into TUNING 30 / QUALIFICATION 60 / EVALUATION 90, hashed as `SCENARIO_PARTITION.json`. All development-time selection (prompt and persona wording, parsing, scorer and rubric configuration, translation settings, δ_raw, f_diag, anchor coefficients, variance components) runs on TUNING only; QUALIFICATION is read once, before B1, for the two gates of rule F; EVALUATION is untouched until B1. **The corpus is fixed at 30 + 60 + 90 and is never enlarged** (rule F4) |
| Evaluator-validity gate | **rule F2** — one qualification transcript per scenario in each of the 12 checkpoint-language cells (720 transcripts, double-annotated blind, a third adjudicating), so each cell carries **60 independent scenario clusters and shares no transcript with another cell's gate**; every threshold is estimated **inside its own cell** by a scenario-resampling BCa bootstrap (9999, seed 20260911): cell convergent lower bound > 0.40, pooled (each scenario counted once) > 0.60, discriminant and anti-stereotype by the same paired scenario bootstrap. Transcript-level Fisher-z standard errors are **VOID** |
| Registered run count and seed schedule | **rule H** — seeds 20260911 + {0…11}. For **every** allowed S ∈ {8…12} the run tensor and nuisance allocation are hashed into `NUISANCE_SCHEDULE.json` before M1 as **five nested blocks**, the first eight seeds' allocation identical in all five: 129 600 confirmatory transcripts at S = 8, +16 200 per seed, 194 400 at S = 12. M2 picks the smallest sufficient S **once**, deterministically, on tuning-partition variance components; if none reaches 0.80 power at δ_alt, S stays 8 and δ_alt is raised. **No adaptive or unscheduled seed count exists.** Dispersion prior 0.11 (ref-010) → σ = 0.066, re-estimated on the tuning partition |
| Estimator and interval | **ONE algorithm — decision_rule A** (the rule A seed-first ordering is VOID: it discarded the unit-level engine label eq1 needs). Unit score with frozen engine label → matched persona-minus-neutral difference at unit level → within-stratum average → pole alignment and pair contrast inside the stratum → 1/3 combination → average over pre-B1 eligible pairs → **wild cluster bootstrap-t over the G = S seed clusters** (Webb 6-point, one globally shared weight per seed per replication, 9999, null imposed, studentized with the G/(G−1) correction, V̂ recomputed inside each replication on G − 1 = 7 df), **90 scenarios fixed, never resampled (rule E)** → **Holm at 0.05 with max-T simultaneous intervals**. The full executable statement is `decision_rule` rule I and is not restated here; every other aggregator and bootstrap named anywhere is a **sensitivity view that decides nothing** |
| Multiplicity | **Holm with max-T simultaneous intervals, family-wide, never BH** (`outcome.multiplicity_correction`, rule I(iv)–(v)): the primary family is **one member per admitted checkpoint × targeted trait** (≤ 8 members, three shifted nulls combined inside the member by maximum p-value); the 6 pair contrasts are a **separate gated family**; **rule F1 removes the gate statistics from both**. Pole heterogeneity is its own family; the language factorial the gated second level; the 30 off-diagonal grid pairs the third, descriptive level, the only place BH survives |
| Randomness varied | sampling seed, item order, option order, persona wording variant, scenario-to-seed assignment; translation engine balanced in the schedule but a **fixed stratum** in inference |
| Compute | Stated once in `pico.outcome.compute_budget`; not restated here |

The dispersion prior is literature-derived, not an in-house pilot — this host has no accelerator and no
reachable model provider — and only seeds the simulation. Whatever the conjunction returns, the estimate and
interval of `Δ̄_mis` are reported in full for every admitted checkpoint and trait.

---



## Superseded verbatim: EXPERIMENT_PLAN.md section 1 before the R12 compression (no force)

## 1. Claims to defend

**This section is a pointer.** The confirmatory specification lives once in `design_card.json` →
`pico.hypothesis`, `pico.outcome`, `pico.preregistration.decision_rule` and `pico.hypothesis.falsifier`;
where any wording here differs, the card wins.

**Primary claim (C1).** Holding the persona language fixed, making the model **converse** in a different
language costs part of the induced Big Five shift. Evidence required — stated in full in
`decision_rule` — is the pre-registered average Δ̄_mis over the pairs declared **eligible before B1**
(rule F1), positive with the family's max-T simultaneous interval excluding zero under the single **rule A**
algorithm, in ≥ 3 of 4 checkpoints **admitted before B1 on the qualification set** (rule F), both **rule B(2)**
limbs' lower bounds above 0, the 4-of-6 pair consistency and pole-concordance side-conditions holding, and
the whole rule holding independently for each targeted trait. **rule C: the tested boundary — and the only
thing claimed — is max(δ_raw, 2·V̂); δ_alt = 0.15 is a power statement only and the rule J "claim about
effects ≥ 0.15" wording is VOID**; the 0.09–0.15 band is pre-registered uninformative (rule J). Inference is
conditional on the 90 fixed evaluation scenarios (rule E).

**rule F1 in one line.** The diagonal-inducibility gate — both aligned pole-vs-neutral diagonal effects with
lower bounds above 0 and `E*(p,p)` above f_diag = 0.18 = 2·δ_raw — is evaluated **only on the independent
pre-B1 qualification set** and the eligible pair set is frozen into `ELIGIBLE_PAIRS.json` before B1, so
eligibility is never selected on the confirmatory outcomes. Default estimand: the fixed all-six-pair average.
Fewer than 4 eligible pairs disqualifies the checkpoint; fewer than 3 checkpoints gives outcome (C).

**Estimand, poles, ITT, `V̂` and power** are defined once in `pico.outcome.primary_metric`,
`pico.outcome.variance_protocol` and `decision_rule` (rule D, rule J, rule A) and are not restated here. C1 is
an **intention-to-treat** claim about *assignment* (§5's ITT rule). `power/power_sim_r6.py` encodes the whole
conjunction; `power/power_sim_r9.py` adds the pre-B1 eligibility gate as an explicit stage and must show
≥ 0.90 per-pair gate power, reporting power conditional on the frozen eligible set and unconditionally. Both
run on development-split components **before** any evaluation generation, with the rule J branch decided at
that moment and never after. The `2·V̂` limb is never dropped to buy power.

**No main effect is confirmatory, and the language factorial still runs in full:** a main effect of D
averages over persona languages and mixes the mismatch with the difficulty of conversing in D. The behavior
channel's P × D factorial is the *gated second-level* family, the self-report P × D × I factorial carries
C2/H2, and the rater-language factor carries H2b — estimated and reported, never a confirmatory read for C1.
The mismatch contrast is immune by construction: the neutral arm entering `E*` is generated in the same
dialogue language, so D's own effect is differenced out first. **C1 says nothing about why** — competence,
tokenization, pragmatics and translation naturalness remain live alternatives, bounded only partially by the
competence screen, the length band and the three-translation ensemble, and are disclosed as untested. The
bundled 6×6 matrix is an omnibus pipeline effect and the resource gradient a descriptive association; neither
enters a hypothesis condition.

**Supporting claim (C2).** The loss is not an artifact of the measuring apparatus. Instrument language is
administered *after* the transcript exists, so it is not a factor of the behavior channel: C2's evidence is
the **within-transcript instrument-language contrast on the self-report channel** (same frozen conversation
state re-administered in each instrument language, paired, transcript-clustered), plus a **matched
text-quality perturbation** arm at fixed final instrument language, narrowed by rule L to an observable
instrument-administration-pipeline effect. **rule E: the 12 development-touched BFI-2 items are excluded from
the confirmatory H2 score (48 items scored) and reported separately.** Also required: between-translation
variance small relative to the shortfall; interrater agreement equivalent across the six languages **on the
qualification set**; rater-free indicators reproducing the direction of the primary comparison.

**Anti-claims that must be ruled out** are registered as rival hypotheses H2, H2b and H3 in
`design_card.json` → `pico.hypotheses[]`, each with its own estimand, family and kill mechanism, and are not
restated here — including that the round-trip control is a sensitivity that narrows, never a refutation path
(the falsifier of record is `pico.hypothesis.falsifier`, rule C), that H3 is removed from the powered
confirmatory claims with **H3 UNRESOLVED** as its default reading, and that the compliance confounder is
handled by the compliance-only control instruction with per-cell compliance rates reported and conditioned on.

Two primary claims, at the cap. No third claim is defended.

---



## Superseded verbatim: METHOD_DESIGN.md section 1 before the R12 compression (no force)

## 1. Hypothesis and the falsifier of record

**Named labels** are declared once in `design_card.json` → `pico.definitions` and referenced by label from
here on: **(C)** the bundled cross-lingual pipeline switch (omnibus only); **(LangFac)** the three-way
P × D × I factorial over L₃ = {English, Chinese, Swahili}; **(M) retired** in favour of `E_q(p,d)` (eq1),
`E*(p,d) = ½·[E_high − E_low]` and the confirmatory `Δ_mis(p,d) = E*(p,p) − E*(p,d)` (eq6, unclipped,
positive = the switch **cost** induced expression), with ρ and S secondary and Fieller-interval; **(Elig)**
the denominator-eligibility rule, amended by the diagonal-inducibility gate now run **before B1** (rule F1);
**(R)** the claimed mechanism; **(A)/(B)** the persona and content-matched neutral arms.

**H1.** A mismatch between the language of the persona instruction and the language the model
converses in reduces the persona-induced shift in expressed Big Five trait level — measured as a positive
`Δ_mis(p,d)`, averaged over the eligible ordered pairs as `Δ̄_mis`. This is the term (R) predicts and the one
instrument translation cannot manufacture, because the neutral arm is generated in the *same* dialogue
language it is differenced against: the main effect of D is removed inside `E*` before the contrast is
formed, and `Δ_mis` is exactly the signed P × D interaction contrast. **No confirmatory read is taken on
the dialogue-language main effect** (rule L). H1 is an **intention-to-treat** claim about *assignment* to a requested
dialogue language: every confirmatory number is computed on the first planned generation as produced,
including wrong-language output, with per-cell compliance rates beside it and per-protocol estimates
secondary. The bundled grid (C) is reported alongside as an omnibus pipeline effect — informative about
deployment, silent about mechanism — and the pretraining-resource gradient is descriptive only. The factorial
is what separates "the instruction is encoded language-specifically" from "the questionnaire reads
differently in Swahili"; the claim is exactly as wide as the factorial supports and, per **rule E**,
conditional on the 90 fixed evaluation scenarios. The tested boundary is `max(δ_raw, 2·V̂)` (rule C).

**Falsifier of record (rule C).** The ONE falsifier is `design_card.json` ->
`pico.hypothesis.falsifier`, applied per targeted trait over the admitted checkpoints with the same Holm
family, the same rule-I intervals and the same ≥ 3-of-4 quantifier: (A) supported iff the full conjunction
of `preregistration.decision_rule` holds; (B) REFUTED through **either** limb — **(B1) equivalence**, the
adjusted interval for `Δ̄_mis` lies entirely inside ±δ_raw = 0.09, **or (B2) reversal**, the adjusted
interval lies entirely **below −δ_raw**, so a practically large mismatch effect in the direction *opposite*
to the registered one refutes the directional claim instead of being absorbed into inconclusive and the
rule is symmetric in sign; (C) INCONCLUSIVE otherwise, including any non-declaration in
the 0.09–0.15 uninformative band and every UNTESTABLE branch. Only (B) disproves H1. This section states NO independent falsifier
and NO competing refutation path; the round-trip tie of §2 narrows the claim and never refutes it, and
`refine-logs/consistency_check.py` fails the pre-registration if any competing path reappears.

**Rivals** are registered once each in `pico.hypotheses[]` — estimand, family and kill mechanism — and are
not restated here. In one line each: **H2** is, per **rule L**, an observable instrument-administration-
pipeline effect, measured in its own family by the instrument-language main effect of the self-report
P × D × I factorial and never a confirmatory read for H1; **H2b** (rater insensitivity in lower-resource
languages) is killed by the qualification-set agreement equivalence test and the four rater-free indicators;
**H3** (full language-generality) is an equivalence claim declarable only by simultaneous Holm-adjusted TOSTs
at ε = 0.075, is removed from the powered confirmatory claims, and defaults to **UNRESOLVED** — failure to
reject a shortfall is never reported as full transfer.

Every hypothesis is tested against the *same* numbers under the preregistered analysis plan; none of the
readings is selected after seeing the data.

---



## Superseded verbatim: EXPERIMENT_PLAN.md section 1 before the R12 pointer reduction (no force)

## 1. Claims to defend

**This section is a pointer.** The confirmatory specification lives once in `design_card.json` →
`pico.hypothesis`, `pico.outcome`, `pico.preregistration.decision_rule` and `pico.hypothesis.falsifier`;
where any wording here differs, the card wins.

**Primary claim (C1).** Holding the persona language fixed, making the model **converse** in a different
language costs part of the induced Big Five shift. Evidence required — stated in full in `decision_rule` —
is the pre-registered average Δ̄_mis over the pairs declared **eligible before B1** (rule F1), positive with
the family's max-T simultaneous interval excluding zero under the single **rule A** algorithm, in ≥ 3 of 4
checkpoints **admitted before B1 on the qualification set** (rule F), both **rule B(2)** limbs' lower bounds
above 0, the 4-of-6 pair consistency and pole-concordance side-conditions holding, and the whole rule holding
independently for each targeted trait. **rule C: the tested boundary — and the only thing claimed — is
max(δ_raw, 2·V̂); δ_alt = 0.15 is a power statement only and the rule J "claim about effects ≥ 0.15" wording
is VOID**; the 0.09–0.15 band is pre-registered uninformative. Inference is conditional on the 90 fixed
evaluation scenarios (rule E).

**rule F1 in one line.** The diagonal-inducibility gate — both aligned pole-vs-neutral diagonal effects with
lower bounds above 0 and `E*(p,p)` above f_diag = 0.18 = 2·δ_raw — is evaluated **only on the independent
pre-B1 qualification run tensor (rule H2)** and the eligible pair set is frozen into `ELIGIBLE_PAIRS.json`
before B1, so eligibility is never selected on the confirmatory outcomes. Default estimand: the fixed
all-six-pair average. Fewer than 4 eligible pairs disqualifies the checkpoint; fewer than 3 checkpoints gives
outcome (C).

**Estimand, poles, ITT, `V̂` and power** are defined once in `pico.outcome.primary_metric`,
`pico.outcome.variance_protocol` and `decision_rule` (rules D, J, A) and are not restated here. C1 is an
**intention-to-treat** claim about *assignment* (§5's ITT rule). `power/power_sim_r6.py` encodes the whole
conjunction; `power/power_sim_r9.py` adds the pre-B1 eligibility gate as an explicit stage and must show
≥ 0.90 per-pair gate power, reporting power conditional on the frozen eligible set and unconditionally. Both
run on development-split components **before** any evaluation generation, the rule J branch decided at that
moment and never after. The `2·V̂` limb is never dropped to buy power.

**No main effect is confirmatory, and the language factorial still runs in full:** a main effect of D averages
over persona languages and mixes the mismatch with the difficulty of conversing in D. The behavior channel's
P × D factorial is the *gated second-level* family, the self-report P × D × I factorial carries C2/H2, and the
rater-language factor carries H2b — estimated and reported, never a confirmatory read for C1. The mismatch
contrast is immune by construction: the neutral arm entering `E*` is generated in the same dialogue language,
so D's own effect is differenced out first. **C1 says nothing about why** — competence, tokenization,
pragmatics and translation naturalness remain live alternatives, bounded only partially by the competence
screen, the length band and the three-translation ensemble, and are disclosed as untested. The bundled 6×6
matrix is an omnibus pipeline effect and the resource gradient a descriptive association; neither enters a
hypothesis condition.

**Supporting claim (C2).** The loss is not an artifact of the measuring apparatus. Instrument language is
administered *after* the transcript exists, so it is not a factor of the behavior channel: C2's evidence is the
**within-transcript instrument-language contrast on the self-report channel** (same frozen conversation state
re-administered in each instrument language, paired, transcript-clustered), plus a **matched text-quality
perturbation** arm at fixed final instrument language, narrowed by rule L to an observable
instrument-administration-pipeline effect. **rule E: the 12 development-touched BFI-2 items are excluded from
the confirmatory H2 score (48 items scored) and reported separately.** Also required: between-translation
variance small relative to the shortfall; interrater agreement equivalent across the six languages **on the
qualification set**; rater-free indicators reproducing the direction of the primary comparison.

**Anti-claims that must be ruled out** are registered as rival hypotheses H2, H2b and H3 in `design_card.json`
→ `pico.hypotheses[]`, each with its own estimand, family and kill mechanism, and are not restated here —
including that the round-trip control is a sensitivity that narrows, never a refutation path (the falsifier of
record is `pico.hypothesis.falsifier`, rule C), that H3 is removed from the powered confirmatory claims with
**H3 UNRESOLVED** as its default reading, and that the compliance confounder is handled by the compliance-only
control instruction with per-cell compliance rates reported and conditioned on.

Two primary claims, at the cap. No third claim is defended.

---



## Superseded verbatim: EXPERIMENT_PLAN.md section 7 before the R12 compression (no force)

## 7. Final checklist before any run

- [ ] Design confirmation point approved by the human reviewer (not self-signed)
- [ ] Downstream permission granted for accelerator work
- [ ] Preregistration written and locked; metric and split locked
- [ ] Checkpoint revisions, prompt templates, item renderings, scenario set, seed list, harness version pinned and hashed
- [ ] **Corpus partitioned 30 TUNING / 60 QUALIFICATION / 90 EVALUATION and hashed as `SCENARIO_PARTITION.json` before any prompt is written (rule E); qualification read once before B1, evaluation untouched until B1**
- [ ] B0 apparatus gates resolved and recorded: per-language anchor-slope and perturbation-sensitivity rater
      qualification, the trait-specific invariance screen for the rater-free indicators, and the **rule G** rubric
      freeze with its convergent / discriminant / anti-stereotype construct-validity thresholds
- [ ] **Every development-time choice — prompts, wording, parsing, scorer and rubric configuration, translation
      settings, δ_raw, f_diag, anchor coefficients, variance components — made on the TUNING 30 alone, the rater
      then frozen by hash (rule E); all inventory items and rater-free indicators frozen before the first
      evaluation administration, with DIF screening registered as a sensitivity analysis only**
- [ ] **The single randomization schedule — sampling seed, translation variant, persona wording variant, item
      order, option order per (cell, setting, scenario, seed) — frozen before any generation from one recorded RNG
      seed, translation and wording variants **block-allocated** over the 720 (scenario, seed) units of each cell
      and setting (rule H), the indexed allocation frozen as `refine-logs/NUISANCE_SCHEDULE.json` with its sha256
      recorded**
- [ ] **The qualification-partition transcripts generated, hashed and their admission decisions recorded — the
      720 blind double-annotated cell transcripts of rule F2 among them; the admitted checkpoint set, the eligible
      pair set (`refine-logs/ELIGIBLE_PAIRS.json`) and the primary family all frozen before B1 opens (rule F)**
- [ ] **`refine-logs/consistency_check.py` exits 0 — one refutation path only (rule C) — before the
      pre-registration is frozen and again before analysis**
- [ ] **The eligibility floor f_diag = 0.18, the rule A estimator, `V̂` and the tested boundary
      `max(δ_raw, 2·V̂)` encoded in the frozen analysis code; the 12 development-touched BFI-2 items excluded from
      the confirmatory H2 score (rule E); `power_sim_r6.py` and the gate-aware `power_sim_r9.py` run and recorded
      before any evaluation generation (rule D, rule F1)**
- [ ] Per-language length band calibrated and frozen
- [ ] B3 ablation scenario IDs, selection algorithm and contrasts frozen and hashed **before B1 starts**
- [ ] **Variance components re-estimated on the TUNING partition and the frozen simulators rerun before any
      evaluation generation; S selected once and deterministically from the five pre-hashed nested blocks of
      rule H, or δ_alt raised if none reaches 0.80 — no adaptive or unscheduled seed count**
- [ ] One JSON Lines record per generation, so analysis re-runs offline without regenerating
- [ ] Decision rule applied exactly as written, before any exploratory analysis



## Superseded verbatim: METHOD_DESIGN.md section 8 before the R12 de-BH correction (no force)

## 8. Preregistration

Metric and split are locked before any evaluation generation. **The analysis plan, decision rule, falsifier
of record and predicted result live once in `design_card.json` → `pico.preregistration` and
`pico.hypothesis.falsifier`; this section is a pointer and states no independent rule.** Where any wording
here differs, the card wins.

**Decision rule — head only.** H1 is supported only if, independently for each targeted trait: (1) `Δ̄_mis`
over the pairs declared **eligible before B1** (rule F1) is positive with a family-wide BH-adjusted interval
excluding zero, formed by the single **rule A** algorithm; (2) **rule B(2)** — the BH-adjusted lower bounds of
both `Δ̄_mis − δ_raw` and `Δ̄_mis − 2·V̂` exceed 0; (3) at least 4 of the 6 eligible pair contrasts positive
(a robustness side-condition, not a threshold); (4) (1)–(3) hold in ≥ 3 of the 4 checkpoints **admitted
before B1 on the independent qualification set** (rule F); (5) pole concordance. **rule C: the tested boundary
is max(δ_raw, 2·V̂) and that is the whole claim; δ_alt = 0.15 is a power statement only**, so the 0.09–0.15
band is pre-registered uninformative and the estimate and interval are reported regardless (rule J). The
registered branch (seeds to the cap of 12, then δ_alt raised) is decided before evaluation generation and
never after; the `2·V̂` limb is never removed to buy power; no condition may be added, removed or
re-thresholded once generation has started. There is **no confirmatory read on any main effect and no
round-trip-tie rule**: the language factorial runs in full as the gated second-level and self-report
families, never as support for H1. H2 is supported, inside its own family, when the instrument-language main
effect of the self-report factorial carries the loss under the same standard while the dialogue-language
effect does not. **H3** is supported only by simultaneous Holm-adjusted TOSTs at ε; failure of (1) does not
support H3, and an intermediate pattern is **H3 unresolved**.

Neither the bundled grid (C) nor the resource-rank slope enters any hypothesis condition. A cell failing
(Elig) contributes its unnormalized difference and no ratio. A shortfall present on self-report but with an
interval covering zero on the rater-free indicators reports H2b as unresolved and demotes the behavior arm to
descriptive; if the behavior channel fails the **qualification-set** rater, construct-validity (rule G) or
admission checks, confirmatory H1 is untestable and there is no endpoint substitution. The admission gate of rule F3 runs on the
qualification partition alone; the 20% confirmatory-core rater-language subsample is **non-gating**.

**Predicted result** is registered once in `pico.preregistration.predicted_result`; per **rule J** its 0.12
head is descriptive and registers nothing. **rule E**: the 12 development-touched BFI-2 items are excluded
from the confirmatory H2 self-report score (48 items scored) and reported separately.

**Every outcome is informative and each reading is bound in advance to the pattern that selects it**
(enumerated once in `EXPERIMENT_PLAN.md` §6): self-report-only shortfall → H2 as narrowed by rule L;
shortfall vanishing on rater-free indicators → H2b; full transfer → the positive language-general finding;
only the compliance rate moving → instruction following rather than trait expression (ref-031, ref-032).

---



## Superseded verbatim: EXPERIMENT_PLAN.md section 4 before the R12 compression (no force)

## 4. Budget

Every number is recomputed from the full run tensor of `pico.outcome.n_runs`; the card's `compute_budget` is authoritative.

| Item | Amount |
| --- | --- |
| Development and qualification phase (B0) | ≈110–130 GPU hours; TUNING 30 for every development choice, QUALIFICATION 60 for the two pre-B1 gates, **including the 28 800-transcript rule-H2 qualification run tensor and its rater scoring (≈11–20 GPU hours)** |
| **B1b-behavior (confirmatory core)** | **129 600 transcripts** at the default S = 8 — 9 P×D cells × 5 settings × 4 checkpoints × 90 scenarios × 8 seeds; +16 200 per further seed to 194 400 at the S = 12 ceiling (rule H) |
| **Omnibus grid (C), descriptive** | **64 800 transcripts** — 27 cells × 5 settings × 4 checkpoints × 30-scenario subset × 4 seeds |
| **B1b-selfreport** | **388 800 administrations** (27 216 000 item responses) + 129 600 perturbation + 64 800 omnibus = **583 200 administrations** |
| Generation compute | ≈155M prefill + 100M decode tokens → **≈40–70 GPU hours** on one A100 80GB |
| Self-report compute | ≈760M prefill + 82M decode → **≈30–50 GPU hours** |
| Rater scoring (194 400 transcripts + 51 840 rater-language re-scorings, 24B rater) | ≈295M tokens → **≈35–60 GPU hours** |
| Repeat arm, ablations, 1.5× infrastructure-failure allowance | included below |
| Scoring, contamination scan, analysis | ≈60 CPU hours |
| **Registered total** | **≈240–400 GPU hours evaluation (≈150–250 on a Hopper-class 80GB) + ≈110–130 GPU hours development and qualification + ≈60 CPU hours** |
| Human annotation | **720 qualification transcripts** (one per qualification scenario in each of the 12 checkpoint-language cells) double-annotated blind with a third adjudicating — 1440 primary annotations plus adjudications; this single sample carries both the agreement check and the rule-F2 criterion gate |

The 1.5× allowance covers **prespecified infrastructure failures only** — API error, timeout, zero-length return — re-run with the same seed and capped at 2% of any cell; **no regeneration may depend on the value of any outcome** (the ITT rule). Inference only. Budget is matched across the method and every comparator on dialogues, turns and generations, not on tokens.

**Host constraint.** This host has no accelerator and no reachable model provider, so no pilot could run and no number here is an in-house measurement; the dispersion prior is literature-derived (ref-010) and re-estimated on the tuning partition at M2 before it is relied on. **A card of at least 80GB is required** — for the rater model, not for generation, which fits a 48GB L40S or A6000.

---



## Superseded verbatim: METHOD_DESIGN.md named-models section before the R12 budget correction (no force)

## Named models and the competence screen

Every role is named and pinned, so the envelope is derivable rather than asserted.

| Role | Model | Why this one |
|---|---|---|
| Graded checkpoint 1 | `google/gemma-3-12b-it` (12B) | Swahili+Arabic documented; Google lineage |
| Graded checkpoint 2 | `Qwen/Qwen3-8B` (8B, non-thinking pinned) | Swahili documented; Alibaba lineage; strongest Chinese |
| Graded checkpoint 3 | `meta-llama/Llama-3.1-8B-Instruct` (8B) | most-studied open lineage; Swahili undocumented, so it exercises the screen |
| Graded checkpoint 4 | `CohereLabs/aya-expanse-8b` (8B) | multilingual-alignment lineage; Arabic documented, Swahili not |
| Rater | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (24B) | fifth, independent lineage — no model grades its own family; fits one 80GB card |
| Translation engine A | Google Cloud Translation v3 (NMT) | all six languages |
| Translation engine B | Microsoft Azure AI Translator v3.0 | all six languages |
| Translation engine C | `facebook/nllb-200-3.3B`, pinned commit, local | keeps translation reproducible after commercial engines change; DeepL excluded (no Swahili) |

Every checkpoint is loaded by an exact Hugging Face commit revision, resolved once by `tools/pin_checkpoints.sh` and written into the preregistration.

**Why naming forces a new control.** Two checkpoints do not document Swahili; without a screen a model that cannot hold a Swahili conversation would enter the matrix as one whose induced trait fails to transfer into Swahili — precisely the inference the study exists to make. The pre-registered competence screen (development split, 50 neutral prompts per checkpoint × language, response-language accuracy ≥ 0.95, instruction compliance ≥ 0.90) runs before evaluation; under rule F3 a checkpoint enters confirmation only if it clears **all three L₃ languages**, and if fewer than three qualify H1 is untestable.

**Compute.** The registered envelope is stated once, in `design_card.json` → `pico.outcome.compute_budget`
(≈240–400 GPU-hours on one A100 80GB, ≈150–250 on one Hopper-class 80GB, plus ≈60 CPU-hours and ≈90 GPU-hours
for the development split), and is not restated here. Generation and self-report fit a 48GB L40S/A6000; **one
80GB card is required** only to hold the 24B rater with its KV cache at usable batch sizes.

---



## Superseded verbatim: METHOD_DESIGN.md section 10 before the R12 compression (no force)

## 10. R12 — the consolidated design of record

`design_card.json` → `pico.preregistration.decision_rule` is **one self-contained specification, sections
A–N**, with **no amendment stack**; R11 and the superseded R4→R10 chain are archived verbatim in
`DESIGN_REVISION_HISTORY.md` and have **no force**, as are the narratives of the six R11 repairs. Citations of
the form **rule X** point at section X of that specification; section N lists every void inference. R12 repairs
the three fatals the R11 design-kill returned:

1. **The pre-B1 eligibility gate now has an executable run tensor (rules F1, H2).** rule F1 is estimated on
   **12 checkpoint-language cells × 60 qualification scenarios × 5 settings × S seeds = 28 800 fresh
   transcripts** at S = 8 (+3 600 per further seed), allocated by a sixth indexed block of
   `NUISANCE_SCHEDULE.json` under the same engine × wording orthogonality rule and scored once each by the
   frozen 24B rater at I = D. The **720-transcript rule-F2 sample is a stratified subsample of it** — one
   transcript per cell × qualification scenario at the first scheduled setting and seed — and carries the
   human annotations for evaluator validity only. Both are budgeted separately in `outcome.n_runs`, and the
   development-and-qualification envelope rises to ≈110–130 GPU hours.
2. **One resampling unit (rule I(i)–(iii)).** The **seed alone** is the cluster, G = S (8 at the default
   count); a cluster holds every unit-level matched difference generated at that seed. One globally shared
   Webb 6-point weight per seed per replication preserves dependence across every contrast that shares seeds;
   residuals are taken against the (engine stratum × cell × setting) fit under the imposed null;
   studentization uses the cluster-robust SE with the G/(G−1) correction and the bootstrap's own reference
   law; V̂ is recomputed inside each replication on G − 1 = 7 df. The (scenario, seed) unit is void, and both
   power simulators are re-run under this cluster.
3. **One enumerated family with a matching simultaneous interval (rule I(iv)–(v)).** The **primary family is
   one member per admitted checkpoint × targeted trait** (≤ 8 members), its three shifted nulls combined
   inside the member by their maximum p-value; the procedure is **Holm at 0.05 with max-T simultaneous
   bootstrap-t intervals** (c* = the 0.95 quantile of the family-wide max |t| in the same replications), so
   the intervals rule B reads cover simultaneously and agree with the Holm decisions. The **six pair-specific
   contrasts are a separate gated family** under the same procedure. BH, BH-adjusted intervals and BY-style
   FCR inversion are **VOID** in every confirmatory family — BH controls a false-discovery rate, not coverage.


## Superseded verbatim: EXPERIMENT_PLAN.md section 8 before the R12 compression (no force)

## 8. R12 — the consolidated design of record

The preregistered rule is **one self-contained specification, sections A–N** of
`design_card.json` → `pico.preregistration.decision_rule`, with **no amendment stack**; R11 and the
superseded R4→R10 chain are archived verbatim in `DESIGN_REVISION_HISTORY.md` and have **no force**.
**rule X** citations here point at section X of it; section N lists every void inference. The repairs are
enumerated once, in `METHOD_DESIGN.md` §10, and are not restated here. In this plan they land as §6's
two-limb pointer to the falsifier of record (rule C), B0's opening 30 / 60 / 90 partition (E), B0's per-cell
scenario-cluster evaluator-validity gate on 720 blind double-annotated transcripts drawn from the
28 800-transcript qualification run tensor (F1, F2, H2), B0's deterministic UNTESTABLE branch with no corpus
expansion (F4), §3/§4's pre-hashed nested seed blocks (H), and §5's seed-cluster bootstrap with globally
shared Webb weights, maximum-p combination and Holm max-T simultaneous intervals (I). Added cost: the
qualification run tensor and its rater scoring (≈11–20 GPU hours, inside the ≈110–130 GPU-hour B0 envelope)
and the 720-transcript blind double annotation.


## Superseded verbatim: EXPERIMENT_PLAN.md block B1b before the R12 scenario-count correction (no force)

### B1b — The language factorial, split by outcome channel (main result, evaluation split)

The instrument factor I cannot causally affect the behavior channel, because the transcript is produced under P
and D and *frozen* before the inventory is administered, so the plan runs two outcome-specific designs on the same
generations. The full run tensor — every cell, setting, checkpoint, scenario and seed count — is enumerated
once in `design_card.json` → `pico.outcome.n_runs` and is **not restated here**.

- **B1b-behavior — P × D over L₃, 9 cells (129 600 transcripts).** Instrument language is not a factor.
  Confirmatory read: the ordered pair-specific mismatch contrasts and their registered average, per admitted
  checkpoint and targeted trait, over the pairs declared **eligible before B1 (rule F1)**, with the single
  **rule A** algorithm — unit-level matched differences carrying their engine label → within-stratum
  aggregation over (scenario, seed) → pole alignment and pair contrast inside the stratum → equal-weight 1/3
  stratum combination → **wild cluster bootstrap-t over the G = S seed clusters** — the seed is the only
  resampling unit, one globally shared Webb 6-point weight per seed per replication, 9999, null imposed,
  studentized with the G/(G−1) correction and V̂ recomputed inside each replication on G − 1 = 7 df — the 90
  scenarios **fixed and never resampled (rule E)** → **Holm at 0.05 over the primary family of one member per
  admitted checkpoint × targeted trait, reported as max-T simultaneous intervals**; the six pair contrasts are
  a separate gated family under the same procedure. The seed-first ordering of rule A, the two-way scenario×seed bootstrap of rule N, the
  scenario-only bootstrap and the interquartile-mean aggregator are **labelled sensitivities that decide
  nothing**. The persona- and dialogue-language main effects and the P × D terms are the descriptive second
  level.
- **B1b-selfreport — I crossed on the frozen transcripts (388 800 paired administrations).** Carries H2 and
  the three-way decomposition eq7, computed on this channel only, clustered at the transcript level; the
  matched text-quality perturbation adds one degraded-rendering administration per core transcript.
- **B1b-rater — the measurement factor for H2b, non-gating (rule F).** A registered 20% stratified subsample
  of the core is re-scored in each of the two non-native L₃ languages, post hoc on identical frozen
  transcripts. It enters no confirmatory family and **cannot affect admission** — every admission-affecting
  rater check already ran on the pre-B1 qualification set of B0.
- **The omnibus grid (C) is descriptive and runs at reduced resolution** on the registered 30-scenario subset
  at 4 seeds, one administration each at I = D.

**Scenario counts, reconciled:** 120 support scenarios = 30 development (B0 only, never used for any
evaluation number) + 90 evaluation. The confirmatory core uses all 90; the omnibus grid uses a registered
30-scenario subset of those 90. No other scenario count appears anywhere in this plan.

Cost: see §4. **Feeds C1 and separates H1 from H2.**



## Superseded verbatim: METHOD_DESIGN.md deterministic-checks section (no force)

## Deterministic checks run at design time

| Check | Result |
| --- | --- |
| `design_lint.py refine-logs/design_card.json` | 0 errors, 6 warnings |
| Schema conformance (`design_card_schema.json`, jsonschema 4.26.0) | valid |
| `power_runs.py` consistency | n_runs 8 ≥ required 5 |
| `term_lint.py --refs refs` | 2 flags, both unavoidable (see below) |

The `design_lint` warnings and their disposition: the three `adapted: true` modules are justified individually in §3; the `llm_eval` critical-controls and metric-failure reminders are met by the frozen, reported decoding configuration, the 13-gram contamination scan (§4), the frozen prompt template and zero-shot count, the block-allocated persona wordings (rule H) and the qualification-set agreement, calibration and rule G construct-validity thresholds plus the rater-free replication; reproducibility level R2 (the paradigm-map level, unrelated to a design revision) is met by pinning checkpoint revisions, template hashes, the seed list and the harness version and writing one JSON Lines record per generation.

The two `term_lint` flags are `baseline_source: "websearch"` (a fixed schema enumeration value) and the absence of `refs/paradigm/glossary.json`, which is built automatically from `refs/paradigm/cards/` the next time the term check runs.



## Superseded verbatim: EXPERIMENT_PLAN.md section 6 and the section 5 open-obligation notes (no force)

## 6. Stopping and falsification

The plan stops and reports a negative or reinterpreted result — it does not extend, resample or re-scope — in
any of these cases, all fixed before data collection:

1. The round-trip dummy baseline ties the real switched condition ⇒ a **sensitivity signal favouring the
   apparatus explanation H2**, which **narrows** C1. **rule C: this tie is not a decision rule, not a
   refutation path, and cannot by itself reject C1** — a round trip returns text to its original language
   while degrading it, so it moves degradation and final language together. The earlier tie rule is deleted.
2. Outcomes for H1 are exactly the three of the **falsifier of record** (`pico.hypothesis.falsifier`,
   rule C): (A) supported; (B) **refuted through either limb — (B1) the adjusted interval for Δ̄_mis lies
   entirely inside ±δ_raw = 0.09, or (B2) it lies entirely below −δ_raw**, so a practically large effect in
   the direction opposite to the registered one refutes the directional claim rather than being absorbed
   into inconclusive, and the rule is symmetric in sign; (C) inconclusive — including any non-declaration
   in the 0.09–0.15 band and any case where the behavior channel is not admitted. **(C) alone does not report
   H3**: H3 is reported only if the simultaneous Holm-adjusted TOSTs also place every pre-specified pair
   inside (−ε, +ε) at ε = 0.075; otherwise **H3 unresolved**.
2b. The bundled 6×6 grid shows a loss but no (LangFac) term survives ⇒ an **omnibus cross-lingual pipeline
   effect whose mechanism is unresolved** — a real result about deployment, not a failure.
3. The shortfall appears on self-report but its interval covers zero on the rater-free indicators ⇒ H2b is
   unresolved and the behavior channel is demoted to descriptive.
4. Only the compliance rate moves ⇒ evidence that questionnaire-based transfer measures instruction
   following rather than trait expression.

No outcome here is a failed experiment; each is a specific, preregistered reading.

---



**Open literature obligation (carried, not silently closed).** A reviewer reminder names Amidei et al. (COLING 2025) on interrater-agreement reporting. It is **not** in `refs/refs.json` and could not be verified from this host, so it is used as neither baseline nor support; it is an outstanding reading item, and if it changes the agreement protocol, B0's gate is revised before M2 runs.

**Open method-map obligation.** `paradigm_map.py place` returned **DANGLING_REF**, exposing one real defect — the aggregation/interval module cited ref-007, which performs no interval estimation — repaired by re-sourcing to ref-010 with `adapted: true`. The innovated slot has no literature peers (**EMPTY_COLUMN**), so comparators are sourced externally. Open: `place` could not be re-run after the repair, so the derivation in `refine-logs/SLOT_DIFF/` is pre-repair and must be regenerated once the gate permits. See `SLOT_DIFF.md`.

## Superseded verbatim: METHOD_DESIGN.md sections 4-5 before the R12 compression (no force)

## 4. Controls, leakage and confounders

All nine leakage categories carry an explicit status in `pico.controls.leakage_checklist`. Three are marked
`risk`: **L3.2 non-independence** — the same scenario appears in every cell by design (that is what makes the
pairing valid), handled by clustering and by never treating cells as independent observations, the one
uncertainty procedure being **rule A/I**'s wild cluster bootstrap-t over the G = S seed clusters (the seed is
the only resampling unit) within fixed engine
strata, the 90 scenarios fixed and never resampled (rule E); **L3.3 sampling bias** — six languages are not a
random sample and support dialogue is a narrow domain, so the resource-tier relation describes six points and
is not a fitted law; **L4 contamination** — BFI-2 and TIPI items are near-certainly in pretraining data, so a
13-gram scan reports per-item overlap, within-cell (A) − (B) differencing cancels arm-constant memorization,
and the rater-free indicators give a contamination-free replication. **rule E** additionally excludes the 12
development-touched BFI-2 items from the confirmatory H2 score.

Ten named confounders are addressed in `pico.controls.confounders_addressed`; the load-bearing ones are
translation quality, translation degradation as such (the round-trip control — a sensitivity that narrows and
never refutes, rule C), instruction-following transfer rather than trait change, the cross-lingual baseline
difference (the neutral arm in every cell), within-condition measurement variability, and language-dependent
rater sensitivity.

**Evaluator validity is established, not assumed, and established *before* B1 (rule F2):** on the independent
pre-B1 qualification partition, **720 transcripts** — one per-cell block — are blind double-annotated with
third-party adjudication; the rater must reach Krippendorff α ≥ 0.7 against adjudicated labels in **every**
language, and the per-cell scenario-cluster bootstrap must clear the cross-language equivalence margin, or the
behavior arm is demoted to descriptive. The **rule G** rubric freeze and construct-validity thresholds gate on
the same 720. Self-report side: per-trait per-language internal consistency, measurement invariance with DIF
flags, BFI-2/TIPI convergence. Full text: `pico.controls.evaluator_validity`.

---

## 5. Ablations

Six one-variable ablations (switch removed; single translation instead of the ensemble; each measurement
channel alone; compliance-only instruction in place of the persona; item-order randomization disabled) and
four drops showing which parts are load-bearing (drop the neutral arm; the diagonal reference; the
multi-checkpoint requirement; the repeated-measurement arm).

**The factorial core (LangFac) — P × D × I, 3 × 3 × 3 over {English, Chinese, Swahili}** — is the design's
identification, not an optional ablation: three factors crossed rather than nested, each estimand a
within-scenario, within-seed contrast against the all-source cell on the unnormalized `Δ_mis` scale (eq6),
decomposed by sum-to-zero contrasts (eq7). **No term of this decomposition is a confirmatory endpoint for
H1 — no main effect is.** The confirmatory estimand is the pair-specific mismatch contrast over the pairs
declared eligible before B1 (**rule F1**); **the factorial runs in full** as the gated second-level family
(behavior channel, P × D) and H2's self-report family (P × D × I on the frozen transcripts), the
instrument-language main effect being confirmatory for H2 inside H2's own family only (rule L).

Readings fixed in advance, all *interpretive*: under (R) the mismatch contrasts are positive and the
instrument-language main effect small; under H2 the instrument-language effect carries the loss; under H3
every term lies inside the equivalence margin. **A large P × D interaction with small main effects is the
mismatch reading proper.** Two subordinate crossed factorials: **switch × measurement channel**, and
**switch × pretraining resource tier**, **descriptive only** — six languages cannot identify resource tier
as a cause; the tier association uses the single preregistered eq5 statistic, censoring-aware with worst-
and best-case bounds for cells failing (Elig), and enters no hypothesis condition.

---



## Superseded verbatim: METHOD_DESIGN.md deterministic-check dispositions (no force)

## Deterministic checks run at design time

`design_lint.py` 0 errors / 6 warnings; schema valid (`design_card_schema.json`, jsonschema 4.26.0);
`power_runs.py` n_runs 8 ≥ required 5; `term_lint.py --refs refs` 2 flags, both unavoidable.

Disposition: the three `adapted: true` modules are justified individually in §3; the `llm_eval`
critical-controls and metric-failure reminders are met by the frozen reported decoding configuration, the
13-gram contamination scan (§4), the frozen prompt template and zero-shot count, the block-allocated persona
wordings (rule H) and the qualification-set agreement, calibration and rule G construct-validity thresholds
plus the rater-free replication; reproducibility level R2 is met by pinning checkpoint revisions, template
hashes, the seed list and the harness version and writing one JSON Lines record per generation. The two
`term_lint` flags are `baseline_source: "websearch"` (a fixed schema enumeration value) and the absence of
`refs/paradigm/glossary.json`, built automatically from `refs/paradigm/cards/` at the next term check.



## Superseded verbatim: card I(iv)-(v) -> R13 step-down max-T (no force)

(iv) THE PRIMARY FAMILY IS ENUMERATED AND FROZEN BEFORE B1 IN ELIGIBLE_PAIRS.json AND HOLDS EXACTLY ONE MEMBER PER (ADMITTED CHECKPOINT x TARGETED TRAIT): the member's statistic is Delta_mis_bar, its three shifted nulls (0, delta_raw, 2*V_hat) are combined INSIDE the member by their MAXIMUM p-value (an intersection-union across the shifts), giving exactly one p-value per member, at most 4 x 2 = 8 members. The procedure is HOLM at 0.05 over the members and the reported set is a SIMULTANEOUS MAX-T BOOTSTRAP-t INTERVAL: within each replication the maximum over members of |t| is recorded, its 0.95 quantile c* is taken, and each member's interval is estimate +/- c* x its cluster-robust SE, so the family's intervals cover simultaneously at 0.95 and agree with the Holm max-T decisions by construction. Benjamini-Hochberg is NOT used in this family and no 'BH-adjusted level' or 'BY-style FCR' interval exists anywhere in this design (section N): BH controls a false-discovery rate, not simultaneous coverage, and conditions (1) and (2) read coverage. (v) THE PAIR-LEVEL FAMILY IS SEPARATE AND GATED: the 6 ordered pair-specific Delta_mis(p,d) of each admitted checkpoint and trait form their OWN family, reported only after the primary family, with Holm at 0.05 and its own max-T simultaneous intervals computed from the same replications. Condition (3) reads point estimates in that family and can only make declaration harder. Conditions (1) and both limbs of (2) read the PRIMARY family's simultaneous intervals and no others.

## Superseded verbatim: card C falsifier outcomes -> one region (no force)

Its three outcomes, applied per targeted trait over the admitted checkpoints with the SAME primary family, section-I intervals and at-least-3-of-4 quantifier: (A) SUPPORTED iff the whole section-B conjunction holds; (B) REFUTED through EITHER limb - (B1) EQUIVALENCE, the simultaneous interval for Delta_mis_bar lies entirely inside +-delta_raw = 0.09; or (B2) REVERSAL, it lies entirely BELOW -delta_raw. A practically large mismatch effect OPPOSITE to the registered direction therefore REFUTES the directional claim instead of being absorbed into inconclusive; the rule is sign-symmetric. 

## Superseded verbatim: card B(2) -> step-down region (no force)

(2) BOUNDARY: the adjusted LOWER confidence bounds of BOTH Delta_mis_bar - delta_raw AND Delta_mis_bar - 2*V_hat exceed 0. This joint statistical requirement IS the test; the point comparison Delta_mis_bar >= max(delta_raw, 2*V_hat) is a descriptive summary only. 

## Superseded verbatim: card B(1) wording (no force)

with its section-I interval excluding zero in 

## Superseded verbatim: card J placeholders -> R13 (no force)

Holm with max-T simultaneous intervals over the one-member-per-(checkpoint x trait) primary family - 

## Superseded verbatim: card N(7) void list + Holm (no force)

(section I(iv) replaces all of them with Holm and max-T simultaneous intervals)

## Superseded verbatim: METHOD practical-significance cell -> R13 (no force)

| Practical-significance test | **rule B(2)** — max-T simultaneous lower bounds of BOTH `Δ̄_mis − δ_raw` and `Δ̄_mis − 2·V̂` exceed 0, V̂ recomputed inside every replication |


## Superseded verbatim: METHOD estimator cell -> pointer (no force)

| Estimator and interval | **ONE algorithm — decision_rule A** (rule A's seed-first ordering is VOID: it discarded the unit-level engine label eq1 needs). Unit score with frozen engine label → matched persona-minus-neutral difference → within-stratum average → pole alignment and pair contrast inside the stratum → 1/3 combination → average over pre-B1 eligible pairs → **wild cluster bootstrap-t over the G = S seed clusters** (Webb 6-point, one globally shared weight per seed per replication, 9999, null imposed, studentized with G/(G−1), V̂ recomputed per replication on 7 df), **90 scenarios fixed, never resampled (rule E)** → **Holm at 0.05 with max-T simultaneous intervals**. The executable statement is rule I and is not restated here; every other aggregator named anywhere is a **sensitivity view that decides nothing** |


## Superseded verbatim: METHOD multiplicity cell -> R13 (no force)

| Multiplicity | **Holm with max-T simultaneous intervals, family-wide, never BH** (`outcome.multiplicity_correction`, rule I(iv)–(v)): the primary family is **one member per admitted checkpoint × targeted trait** (≤ 8 members, three shifted nulls combined inside the member by maximum p-value); the 6 pair contrasts are a **separate gated family**; **rule F1 removes the gate statistics from both**. Pole heterogeneity is its own family; the language factorial the gated second level; the 30 off-diagonal grid pairs the third, descriptive level — the only place BH survives |


## Superseded verbatim: METHOD head-only rule -> pointer (no force)



## Superseded verbatim: METHOD repair item 3 -> R13 (no force)

3. **One enumerated family with a matching simultaneous interval (rule I(iv)–(v))** — the primary family is
   **one member per admitted checkpoint × targeted trait** (≤ 8 members, three shifted nulls combined inside
   the member by maximum p-value) under **Holm at 0.05 with max-T simultaneous bootstrap-t intervals**
   (c* = the 0.95 quantile of the family-wide max |t| in the same replications), the six pair contrasts a
   separate gated family. BH, BH-adjusted intervals and BY-style FCR inversion are **VOID** in every

## Superseded: METHOD_DESIGN.md 'Decision rule - head only' paragraph of R12 (no force)

Replaced by the R13 pointer paragraph; the R12 text restated card rule B(1)-(5) verbatim with 'max-T simultaneous interval' wording and is superseded by the R13 step-down max-T region.


## Superseded verbatim: METHOD eligibility cell -> pointer (no force)

| Eligibility | **rule F1** — the diagonal-inducibility gate (aligned pole-vs-neutral lower bounds above 0; `E*(p,p)` above f_diag = 0.18 = 2·δ_raw) is evaluated **only on the 28 800-transcript qualification tensor of rule H2, built on rule E's 60-scenario QUALIFICATION partition** — cut before any prompt was written — and frozen into `ELIGIBLE_PAIRS.json` **before B1**; it enters **no** confirmatory family. < 4 eligible pairs ⇒ the checkpoint cannot count; < 3 checkpoints ⇒ outcome (C) |


## Superseded verbatim: METHOD tested-boundary cell -> pointer (no force)

| Tested boundary / powered alternative | **rule C: the tested boundary — and the only thing claimed against — is max(δ_raw, 2·V̂)**, δ_raw = 0.09. **δ_alt = 0.15 is a power statement only**: 0.824 power there, 0.023 at δ_raw (`power/power_sim_r6.py`, sha256 42c8064c…391e, 100 000 draws, seed 20260911). The 0.09–0.15 band is pre-registered UNINFORMATIVE and the ≥0.15 claim wording of rule J is **VOID** |


## Superseded verbatim: METHOD corpus cell -> pointer (no force)

| Corpus partition | **rule E** — 180 scenarios cut ONCE, before any prompt is written, into TUNING 30 / QUALIFICATION 60 / EVALUATION 90, hashed as `SCENARIO_PARTITION.json`; all development-time selection runs on TUNING only, QUALIFICATION is read once before B1 for rule F's gates, EVALUATION is untouched until B1. **The corpus is fixed at 30 + 60 + 90 and never enlarged** (rule F4) |


## Superseded verbatim: METHOD evaluator cell -> pointer (no force)

| Evaluator-validity gate | **rule F2** — the 720-transcript subsample of the H2 tensor gives each of the 12 checkpoint-language cells **60 independent scenario clusters sharing no transcript with another cell's gate**; every threshold is estimated **inside its own cell** by a scenario-resampling BCa bootstrap (9999, seed 20260911). Transcript-level Fisher-z standard errors are **VOID** |


## Superseded verbatim: METHOD run-count cell -> pointer (no force)

| Registered run count and seed schedule | **rule H** — seeds 20260911 + {0…11}. For **every** allowed S ∈ {8…12} the tensor and nuisance allocation are hashed into `NUISANCE_SCHEDULE.json` before M1 as **five nested confirmatory blocks** plus H2's sixth qualification block, the first eight seeds' allocation identical in all: 129 600 confirmatory transcripts at S = 8, +16 200 per seed, 194 400 at S = 12. M2 picks the smallest sufficient S **once**, deterministically, on tuning-partition variance components; if none reaches 0.80 power at δ_alt, S stays 8 and δ_alt is raised. **No adaptive or unscheduled seed count exists.** Dispersion prior 0.11 (ref-010) → σ = 0.066 |


## Superseded verbatim: METHOD decision-rule head paragraph -> R13 pointer (no force)

**Decision rule — head only; the executable text is the card.** H1 is supported only if, independently for each targeted trait, all five conditions of card rule B hold: (1) `Δ̄_mis` over the pairs declared **eligible before B1** (rule F1) is positive with its R13 step-down interval excluding zero; (2) **rule B(2)**; (3) at least 4 of the 6 eligible pair contrasts positive (a robustness side-condition, not a threshold); (4) in ≥ 3 of the 4 checkpoints **admitted before B1 on the independent qualification set** (rule F); (5) pole concordance. **rule C: the tested boundary is max(δ_raw, 2·V̂) and that is the whole claim; δ_alt = 0.15 is a power
statement only**, so the 0.09–0.15 band is pre-registered uninformative and the estimate and interval are
reported regardless (rule J). The registered branch (seeds to the cap of 12, then δ_alt raised) is decided
before evaluation generation and never after; the `2·V̂` limb is never removed to buy power; no condition may
be added, removed or re-thresholded once generation has started. There is **no confirmatory read on any main
effect and no round-trip-tie rule**: the language factorial runs in full as the gated second-level and
self-report families, never as support for H1. H2 is supported, inside its own family, when the
instrument-language main effect of the self-report factorial carries the loss under the same standard while
the dialogue-language effect does not. **H3** is supported only by simultaneous Holm-adjusted TOSTs at ε;
failure of (1) does not support H3, and an intermediate pattern is **H3 unresolved**.

## Superseded verbatim: PLAN §5 estimator bullet -> R13 pointer (no force)

  **rule A** algorithm and the **rule I** inference — unit-level matched differences carrying their engine
  label → within-stratum aggregation → pole alignment and pair contrast inside the stratum → equal-weight 1/3
  stratum combination → **wild cluster bootstrap-t over the G = S seed clusters** (the seed the only
  resampling unit, one globally shared Webb 6-point weight per seed per replication, 9999, null imposed,
  studentized with G/(G−1), V̂ recomputed per replication on 7 df), the 90 scenarios **fixed and never
  resampled (rule E)** → **Holm at 0.05 over the primary family of one member per admitted checkpoint ×
  targeted trait, reported as max-T simultaneous intervals**, the six pair contrasts a separate gated family.

## Superseded verbatim: PLAN multiplicity row -> R13 (no force)

| Multiplicity | **Holm at 0.05 with max-T simultaneous bootstrap-t intervals**, applied **once, family-wide** over the primary family — one member per admitted checkpoint × targeted trait (≤ 8) — the six pair contrasts in their **own gated family**, the pre-B1 eligibility gate **excluded** from both (rule F1). BH is void in every confirmatory family, surviving only in the descriptive omnibus grid |

## Superseded verbatim: PLAN §6 outcomes -> R13 one region (no force)

   rule C): (A) supported; (B) **refuted through either limb — (B1) the simultaneous interval for Δ̄_mis lies
   entirely inside ±δ_raw = 0.09, or (B2) it lies entirely below −δ_raw**, so a practically large effect
   opposite to the registered direction refutes the directional claim rather than being absorbed into
   inconclusive, the rule being sign-symmetric; (C) inconclusive — including any non-declaration in the
   0.09–0.15 band and any case where the behavior channel is not admitted. **(C) alone does not report H3**:
   H3 is reported only if the simultaneous Holm-adjusted TOSTs also place every pre-specified pair inside
   (−ε, +ε) at ε = 0.075; otherwise **H3 unresolved**.


## Superseded verbatim: METHOD header revision paragraph -> R13 (no force)

This is revision **R12**, the **consolidated design of record**: the preregistered rule is now one
self-contained specification in sections **A–N** of `pico.preregistration.decision_rule`, with **no
amendment stack**, and every earlier revision narrative — the whole superseded R4→R10 chain — is archived
verbatim in `DESIGN_REVISION_HISTORY.md` and has no force. Citations of the form **rule X** point at
section X of that specification. The load-bearing decision made on the user's instruction of 2026-09-11 to
**run the language factorial** still stands and is unconditional: three language factors —
persona-instruction, dialogue-generation and instrument — are manipulated separately and fully crossed, the
factorial runs **in full** as the gated second level (rule L), and every mechanism claim is carried by it.
§10 lists the six repairs the last design-kill forced and where each now lives.

## Superseded verbatim: METHOD §10 header -> R13 (no force)

## 10. R12 — the consolidated design of record

`design_card.json` → `pico.preregistration.decision_rule` is **one self-contained specification, sections
A–N**, with **no amendment stack**; R11 and the superseded R4→R10 chain are archived verbatim in
`DESIGN_REVISION_HISTORY.md` and have **no force**, as are the narratives of the six R11 repairs. Citations of
the form **rule X** point at section X of that specification; section N lists every void inference. R12 repairs
the three fatals the R11 design-kill returned; the executable text is in the card and is not restated here:

## Superseded verbatim: PLAN header revision paragraph -> R13 (no force)

This is revision **R12**, the consolidated design of record: the
preregistered rule is one self-contained specification (sections A–N) with no amendment stack, and
citations of the form **rule X** point at section X of it. The load-bearing decision — the three-way
**language factorial** (LangFac), block **B1b**, run **in full** as the gated second level — stands
unconditionally and is what carries every mechanism claim. 

## Superseded verbatim: PLAN §8 header -> R13 (no force)

## 8. R12 — the consolidated design of record

The preregistered rule is **one self-contained specification, sections A–N** of `design_card.json` →
`pico.preregistration.decision_rule`, with **no amendment stack**; R11 and the superseded R4→R10 chain are
archived verbatim in `DESIGN_REVISION_HISTORY.md` and have **no force**. **rule X** citations point at section
X of it; section N lists every void inference; the repairs are enumerated once in `METHOD_DESIGN.md` §10. In

## Superseded verbatim: PLAN §7 checklist -> pointer-ized (no force)

- [ ] **Corpus partitioned 30 TUNING / 60 QUALIFICATION / 90 EVALUATION and hashed as `SCENARIO_PARTITION.json` before any prompt is written (rule E); qualification read once before B1, evaluation untouched until B1**
- [ ] B0 apparatus gates resolved and recorded: per-language anchor-slope and perturbation-sensitivity rater
      qualification, the trait-specific invariance screen for the rater-free indicators, and the **rule G** rubric
      freeze with its construct-validity thresholds
- [ ] **Every development-time choice — prompts, wording, parsing, scorer and rubric configuration, translation
      settings, δ_raw, f_diag, anchor coefficients, variance components — made on the TUNING 30 alone, the rater
      then frozen by hash (rule E); inventory items and rater-free indicators frozen before the first evaluation
      administration, DIF screening registered as a sensitivity analysis only**
- [ ] **The single randomization schedule (rule H) frozen before any generation from one recorded RNG seed and
      hashed as `NUISANCE_SCHEDULE.json`, with translation and wording variants block-allocated, in six indexed
      blocks: the five nested confirmatory blocks for S ∈ {8…12} and the rule-H2 qualification block**
- [ ] **The rule-H2 qualification tensor (28 800 transcripts at S = 8) generated, scored and hashed, its 720
      blind double-annotated rule-F2 transcripts adjudicated, and the admitted checkpoint set, the eligible pair
      set (`ELIGIBLE_PAIRS.json`) and the primary family all frozen before B1 opens (rules F, F1, F2)**
- [ ] **`consistency_check.py` exits 0 — one refutation path only (rule C) — before the pre-registration is
      frozen and again before analysis**
- [ ] **f_diag = 0.18, the rule A estimator, the rule I seed-cluster bootstrap with the R13 step-down max-T region,
      `V̂` and the boundary `max(δ_raw, 2·V̂)` encoded in the frozen analysis code; the 12 development-touched
      BFI-2 items excluded from the confirmatory H2 score (rule E); `power_sim_r6.py` and the gate-aware
      `power_sim_r9.py` run and recorded before any evaluation generation (rules D, F1, F4)**


## Superseded verbatim: PLAN §7 variance bullet -> pointer (no force)

- [ ] **Variance components re-estimated on TUNING and the frozen simulators rerun before any evaluation
      generation; S selected once and deterministically from the five pre-hashed nested blocks, or δ_alt raised
      if none reaches 0.80 — no adaptive or unscheduled seed count**


## Superseded verbatim: METHOD rivals paragraph -> R13 (no force)

**Rivals** are registered once each in `pico.hypotheses[]` — estimand, family and kill mechanism — and are not
restated here. In one line each: **H2** is, per **rule L**, an observable instrument-administration-pipeline
effect, measured in its own family by the instrument-language main effect of the self-report P × D × I
factorial and never a confirmatory read for H1; **H2b** (rater insensitivity in lower-resource languages) is
killed by the qualification-set agreement equivalence test and the four rater-free indicators; **H3** (full
language-generality) is an equivalence claim declarable only by simultaneous Holm-adjusted TOSTs at ε = 0.075,
is removed from the powered confirmatory claims, and defaults to **UNRESOLVED** — failure to reject a shortfall
is never reported as full transfer.

## Superseded verbatim: METHOD §10 item 1 -> trimmed (no force)

1. **The pre-B1 eligibility gate now has an executable run tensor (rules F1, H2)** — 12 checkpoint-language
   cells × 60 qualification scenarios × 5 settings × S seeds = **28 800 fresh transcripts** at S = 8, allocated
   by a sixth indexed block of `NUISANCE_SCHEDULE.json` and scored by the frozen rater at I = D; the
   720-transcript rule-F2 sample is a stratified **subsample** of it carrying the human annotations for
   evaluator validity only. Both are budgeted in `outcome.n_runs`; B0 rises to ≈110–130 GPU hours.

## Superseded verbatim: PLAN open obligations -> trimmed (no force)

**Open obligations, carried not silently closed.** (i) A reviewer reminder names Amidei et al. (COLING 2025) on interrater-agreement reporting; it is **not** in `refs/refs.json` and could not be verified from this host, so it is neither baseline nor support but an outstanding reading item — if it changes the agreement protocol, B0's gate is revised before M2. (ii) `paradigm_map.py place` returned **DANGLING_REF**, exposing one real defect — the aggregation/interval module cited ref-007, which performs no interval estimation — repaired by re-sourcing to ref-010 with `adapted: true`; the innovated slot has no literature peers (**EMPTY_COLUMN**), so comparators are sourced externally, and `place` could not be re-run after the repair, so `SLOT_DIFF/` is pre-repair and must be regenerated once the gate permits.

## Superseded verbatim: PLAN round-trip item -> trimmed (no force)

1. The round-trip dummy baseline ties the real switched condition ⇒ a **sensitivity signal favouring the
   apparatus explanation H2**, which **narrows** C1. **rule C: this tie is not a decision rule, not a
   refutation path, and cannot by itself reject C1** — a round trip returns text to its original language
   while degrading it, so it moves degradation and final language together. The earlier tie rule is deleted.

## Superseded verbatim: card pico.outcome.multiplicity_correction (R12, no force)

One confirmatory outcome hierarchy, family-wide, and the procedure is HOLM WITH MAX-T SIMULTANEOUS INTERVALS, never Benjamini-Hochberg: BH controls a false-discovery rate, while conditions (1) and (2) of decision_rule rule B read INTERVAL COVERAGE, so only a simultaneous procedure is admissible. PRIMARY FAMILY (enumerated and frozen before B1 in ELIGIBLE_PAIRS.json, decision_rule rule I(iv)): EXACTLY ONE MEMBER PER (ADMITTED CHECKPOINT x TARGETED TRAIT), at most 4 x 2 = 8 members, whose statistic is the registered average Delta_mis_bar over the pre-B1 eligible pairs and whose single p-value is the MAXIMUM over its three shifted nulls (0, delta_raw, 2*V_hat). Holm at 0.05 over the members; the reported interval is the max-T bootstrap-t interval estimate +/- c* x cluster-robust SE, c* being the 0.95 quantile of the family-wide maximum |t| in the same wild bootstrap, so coverage is simultaneous at 0.95 and agrees with the Holm decisions by construction. PAIR-LEVEL FAMILY (rule I(v)), SEPARATE AND GATED on the primary family: the 6 ordered pair-specific Delta_mis(p,d) per admitted checkpoint and trait, Holm at 0.05 with their own max-T simultaneous intervals from the same replications. Rule B(3) reads point estimates here and can only make declaration harder. The average and the pair contrasts are NEVER members of one family. SECOND LEVEL, gated on the primary family and reported after it: the remaining terms of the behavior channel's P x D language factorial, in their own enumerated family under the same Holm and max-T procedure, never a confirmatory endpoint. SEPARATE SELF-REPORT FAMILY (H2): the within-transcript instrument-language contrasts and the matched text-quality perturbation contrasts, Holm at 0.05 with max-T simultaneous intervals within that family and transcript-level clustering, never pooled with the behavior family. POLE-HETEROGENEITY FAMILY: the pole contrasts over the same 6 x 4 x 2 cells, Holm within their own family, never pooled with the primary family and never gating H1. THIRD LEVEL, DESCRIPTIVE, THE ONLY PLACE BH SURVIVES: the 30 off-diagonal pairs of the omnibus grid (C) and the three off-target traits, BH at 0.05 within the grid and labelled omnibus and descriptive. The primary family spans BOTH targeted traits and the complete rule must hold independently for each (decision_rule rule B). THE DIAGONAL-INDUCIBILITY GATE STATISTICS ARE IN NO CONFIRMATORY FAMILY (rule F1): the gate is a PRE-B1 SELECTION step on the independent qualification run tensor of rule H2, not an inference on confirmatory data. Checkpoint admission AND the eligible pair set are decided BEFORE B1 on that tensor, never on confirmatory-core results; the primary family is frozen at that moment. Every other analysis named descriptive or exploratory in pico.definitions or outcome.n_runs enters no family and carries unadjusted intervals.

## Superseded verbatim: card pico.outcome.variance_protocol.significance_test (R12, no force)

Rule A NAMES ONE CONFIRMATORY ESTIMATOR-AND-INTERVAL ALGORITHM, stated once in decision_rule rule A; every other procedure in every artifact is a labelled sensitivity view that decides nothing - in particular the paired two-way scenario x seed cluster bootstrap, the scenario-only cluster bootstrap, the seed-first ordering of earlier revisions and the interquartile-mean aggregator. In one line: unit-level scores keep their frozen engine label, matched differences are formed FIRST at unit level, aggregated over (scenario, seed) WITHIN each fixed engine stratum, pole-aligned and contrasted inside the stratum, then combined at weight 1/3; interval by WILD CLUSTER BOOTSTRAP-t over the G = S SEED CLUSTERS - the seed is the ONLY resampling unit, G = 8 at the default count, one Webb 6-point weight per seed per replication shared globally across every stratum, cell, setting, checkpoint, trait and contrast, 9999 replications, null imposed, studentized by the cluster-robust SE with the G/(G-1) finite-cluster correction and V_hat recomputed inside each replication on G - 1 = 7 df - then HOLM at 0.05 over the frozen primary family of one member per (admitted checkpoint x targeted trait), the reported interval being the family's MAX-T SIMULTANEOUS bootstrap-t interval. The (scenario, seed) cluster, any BH-adjusted interval and any grid-inversion interval are VOID (decision_rule rule I, section N). THE 90 EVALUATION SCENARIOS ARE FIXED AND NEVER RESAMPLED; all inference is conditional on that corpus (rule E). Ratio-scale summaries use Fieller intervals on cells passing (Elig), unclipped. Equivalence claims (H3, rater agreement, translation ensemble) use two one-sided tests at their registered margins, Holm-adjusted and holding simultaneously. Nuisance allocation is stated once in preregistration.analysis_plan.

## Superseded verbatim: card pico.hypothesis.falsifier (R12, no force)

FALSIFIER OF RECORD - one exhaustive three-way rule, stated ONCE here; METHOD_DESIGN section 1, preregistration.decision_rule, hypotheses[0].falsifier, comparison.dummy_baseline and EXPERIMENT_PLAN sections 1/6/B2 carry POINTERS to it and no independent rule (rule C, enforced by refine-logs/consistency_check.py). EVERY earlier falsifier wording in every artifact is VOID and every competing refutation path is DELETED, not merely superseded. Per targeted trait, over the admitted checkpoints (rule F3): (A) H1 IS SUPPORTED if and only if the full conjunction of preregistration.decision_rule holds. (B) H1 IS REFUTED if and only if, in at least 3 of the 4 admitted checkpoints, the family's max-T simultaneous interval for Delta_mis_bar (rule I(iv), Holm at 0.05) satisfies EITHER limb: (B1) EQUIVALENCE - it lies entirely inside +-delta_raw = 0.09, positive evidence that no practically meaningful mismatch effect exists; or (B2) REVERSAL (rule C) - it lies entirely BELOW -delta_raw = 0.09, a practically large mismatch effect in the direction OPPOSITE to the registered one. The rule is symmetric in sign: a precise negative effect refutes the directional claim and is never absorbed into (C). (C) THE RESULT IS INCONCLUSIVE in every other case, including any non-declaration in the 0.09-0.15 band, which is UNDERPOWERED NON-SUPPORT, and any case in which the behavior channel is not admitted. Only (B) disproves H1; failure of the conjunction alone is (C), reported as inconclusive in title, abstract and discussion, never as disproof.

## Superseded verbatim: card multiplicity_correction head, first R13 draft (no force)

One confirmatory outcome hierarchy, family-wide, and the procedure is ONE INVERTIBLE PROCEDURE - THE R13 STUDENTIZED STEP-DOWN MAX-T TEST TOGETHER WITH THE SIMULTANEOUS REGION THAT IS ITS INVERSION - never Holm with separately constructed intervals and never Benjamini-Hochberg: BH controls a false-discovery rate, while conditions (1) and (2) of decision_rule rule B read INTERVAL COVERAGE, so only a simultaneous procedure is admissible. PRIMARY FAMILY (enumerated and frozen before B1 in ELIGIBLE_PAIRS.json, decision_rule rule I(iv)): TWO COMPONENTS PER (ADMITTED CHECKPOINT x TARGETED TRAIT) - Delta_mis_bar and Delta_mis_bar - 2*V - at most 2 x 4 x 2 = 16 components. At step j the critical value c_j is the 0.95 quantile of the bootstrap maximum |t| over the components still active at step j; each component's reported interval is estimate +/- c_j x its cluster-robust SE; a null value is REJECTED IF AND ONLY IF it lies outside that interval, so decisions and intervals cannot disagree. The random boundary is inverted JOINTLY with the fixed margin: the second component is studentized by the cluster-robust SE of the difference Delta_mis_bar_hat - 2*V_hat taken from the joint covariance of (Delta_mis_bar_hat, V_hat) over the same seed clusters and recomputed inside every replication. Every rule B and rule C decision - support, equivalence and reversal - reads that ONE region. PAIR-LEVEL FAMILY (rule I(v)), SEPARATE AND GATED on the primary family: the 6 ordered pair-specific Delta_mis(p,d) per admitted checkpoint and trait under the SAME step-down procedure and its inverted intervals from the same replications. Rule B(3) reads point estimates here and can only make declaration harder. The average and the pair contrasts are NEVER members of one family. HOLM, SINGLE-STEP MAX-T INTERVALS and the three-shifted-null maximum-p-value combination are VOID (decision_rule section N). 

## Superseded verbatim: card rule I(iv)-(v) R13 step-down (no force)

(iv) THE PRIMARY FAMILY IS ENUMERATED AND FROZEN BEFORE B1 IN ELIGIBLE_PAIRS.json AND HOLDS EXACTLY TWO COMPONENTS PER (ADMITTED CHECKPOINT x TARGETED TRAIT) MEMBER - Delta_mis_bar and Delta_mis_bar - 2*V - at most 2 x 4 x 2 = 16 components. R13 REPLACES THE SUPERSEDED AND VOID R12 COMBINATION OF HOLM DECISIONS WITH SINGLE-STEP INTERVALS BY ONE INVERTIBLE PROCEDURE: a STUDENTIZED STEP-DOWN MAX-T (Romano-Wolf) test at FWER 0.05 over those components, whose INVERSION IS the reported simultaneous 95% confidence region - at step j the critical value c_j is the 0.95 quantile of the bootstrap maximum of |t| over the components still active at step j, and each component's interval is its estimate +- c_j x its cluster-robust SE. A null value for a component is REJECTED IF AND ONLY IF IT LIES OUTSIDE THAT COMPONENT'S REPORTED INTERVAL: one event, one procedure, nothing left that could disagree. THE FIXED AND THE RANDOM BOUNDARY ARE INVERTED JOINTLY: the second component is studentized by the cluster-robust SE of the DIFFERENCE Delta_mis_bar_hat - 2*V_hat, read from the joint covariance of (Delta_mis_bar_hat, V_hat) over the same G seed clusters and recomputed inside every replication, so the 2*V boundary is a coordinate of the estimand vector and never a known constant. EVERY DECISION IN THIS DESIGN READS THIS ONE REGION AND NOTHING ELSE, so rules B and C cannot classify incompatibly: B(1) is lower bound of Delta_mis_bar > 0; B(2) is lower bound of Delta_mis_bar > delta_raw AND lower bound of Delta_mis_bar - 2*V > 0; C(B1) equivalence is the Delta_mis_bar interval lying wholly inside +-delta_raw; C(B2) reversal is its upper bound < -delta_raw. These are statements about ONE interval per member, so at most one can hold, and B(2) implies B(1) because delta_raw > 0. Holm, single-step max-T intervals and the three-shifted-null maximum-p-value combination are VOID (section N). Benjamini-Hochberg is NOT used in either confirmatory family and no 'BH-adjusted level' or 'BY-style FCR' interval exists anywhere in this design (section N): BH controls a false-discovery rate, not simultaneous coverage, and every rule above reads coverage. (v) THE PAIR-LEVEL FAMILY IS SEPARATE AND GATED: the 6 ordered pair-specific Delta_mis(p,d) of each admitted checkpoint and trait form their OWN family, reported only after the primary family, under the SAME step-down max-T procedure and its inverted intervals, computed from the same replications. Condition (3) reads point estimates in that family and can only make declaration harder. Conditions (1) and both limbs of (2), and outcomes (B1) and (B2), read the PRIMARY family's step-down region and no other interval.

## Superseded verbatim: PLAN B1b-behavior confirmatory read -> rule A/I pointer (no force)

  Confirmatory read: the ordered pair-specific mismatch contrasts and their registered average, per admitted
  checkpoint and targeted trait, over the pairs declared **eligible before B1 (rule F1)**, under the single
  **rule A** algorithm and the **rule I** inference, executable in the card and not restated here: unit-level
  matched differences carrying their engine label → within-stratum aggregation → pole alignment and pair
  contrast → 1/3 stratum combination → **wild cluster bootstrap-t over the G = S seed clusters** (the seed the
  only resampling unit, Webb 6-point, 9999, null imposed, studentized with G/(G−1), V̂ recomputed per
  replication on 7 df), the 90 scenarios **fixed and never resampled (rule E)** → the **R14 studentized
  single-step max-T region** over the primary family's two components per admitted checkpoint × targeted trait
  (`Δ̄_mis` and `Δ̄_mis − 2·V`), its inversion the reported simultaneous interval set, the six pair contrasts a
  separate gated family under the same procedure.




## Superseded verbatim: PLAN S6 item 2 -> falsifier pointer (no force)

2. Outcomes for H1 are exactly the three of the **falsifier of record** (`pico.hypothesis.falsifier`,
   rule C): (A) supported; (B) **refuted through either limb — (B1) the R14 single-step interval for Δ̄_mis
   lies entirely inside ±δ_raw = 0.09, or (B2) its upper bound lies below −δ_raw**, so a practically large
   effect opposite to the registered direction refutes the directional claim rather than being absorbed into
   inconclusive, the rule being sign-symmetric; every limb reads that one region, so no two outcomes can hold
   at once; (C) inconclusive — including any non-declaration in the 0.09–0.15 band and any case where the
   behavior channel is not admitted. **(C) alone does not report H3**: H3 is reported only if the single-step
   max-T TOST region also places every pre-specified pair inside (−ε, +ε) at ε = 0.075; otherwise **H3
   unresolved**.


## R15 superseded verbatim (METHOD 10)

R14 repairs the invalid-inversion fatal the R13 design-kill returned,
on top of the R12 repairs; the executable text is in the card and is not restated here:

1. **The pre-B1 eligibility gate now has an executable run tensor (rules F1, H2)** — 12 checkpoint-language
   cells × 60 qualification scenarios × 5 settings × S seeds = **28 800 fresh transcripts** at S = 8, in a
   sixth indexed block of `NUISANCE_SCHEDULE.json`, scored by the frozen rater at I = D; the 720-transcript
   rule-F2 sample is a stratified **subsample** of it carrying the human annotations for evaluator validity
   only. Both are budgeted in `outcome.n_runs`; B0 rises to ≈110–130 GPU hours.
2. **One resampling unit (rule I(i)–(iii))** — the **seed alone**, G = S, one globally shared Webb 6-point
   weight per seed per replication, residuals against the (engine stratum × cell × setting) fit under the
   imposed null, studentization with the G/(G−1) cluster-robust SE, V̂ recomputed inside each replication on
   G − 1 = 7 df. The (scenario, seed) unit is void and both power simulators are re-run under this cluster.
3. **One invertible procedure, one region (rule I(iv)-(v), R14)** - the primary family holds **two
   components, `Δ̄_mis` and `Δ̄_mis − 2·V`, per admitted checkpoint × targeted trait** (≤ 16) under a
   **studentized SINGLE-STEP max-T test at FWER 0.05 over that FIXED family, whose inversion IS the
   reported simultaneous 95 % region** (one critical value c = the 0.95 quantile of the bootstrap max |t|
   over all components, independent of the observed ordering and of which null is tested), so a null is
   rejected iff it lies outside that component's interval and rules B and C cannot disagree. R13's
   step-down critical values were read off observed-ordering active sets and reused as interval
   half-widths for every candidate null — not a valid inversion — and are **VOID**; the price of the
   repair is power, not validity. The random
   `2·V` boundary is a coordinate of the estimand vector, studentized by the cluster-robust SE of the
   difference and inverted jointly with δ_raw. Holm, step-down max-T, the three-shifted-null
   maximum-p-value combination, BH and BY-style FCR inversion are **VOID** in every
   confirmatory family — BH controls a false-discovery rate, not coverage.


## R15 superseded verbatim (card)

### A estimand
The confirmatory estimand is the average pair-specific mismatch shortfall Delta_mis_bar, per admitted checkpoint and per targeted trait, over the ordered (persona-language p, dialogue-language d) pairs declared ELIGIBLE BEFORE B1 by section F1.

### B3
at least 4 of the 6 ordered pair-specific mismatch contrasts - counted among ELIGIBLE pairs only -

### B4
(4) POLE CONCORDANCE: because E*(p,d) = 0.5*[E_high - E_low] shares one neutral arm B across both poles, B cancels and the pole-averaged endpoint measures HIGH-VERSUS-LOW POLE SEPARATION, not a persona-versus-neutral shift. The endpoint is unchanged but the claim is gated on evidence that it is a persona effect: for every

### F1 tail
A pre-B1 selection step is not an inference on the confirmatory data, so these gate tests are members of NO confirmatory family. If fewer than 4 of the 6 ordered pairs are eligible in a checkpoint, that checkpoint cannot count toward condition (1); if fewer than 3 checkpoints remain, outcome (C).

### L
Second-level status means only that the confirmatory family is the mismatch contrasts and their eligible-pair average AND NOTHING ELSE

## R15 superseded verbatim (METHOD 1)

### labels
**(M) retired** in favour of `E_q(p,d)` (eq1),
`E*(p,d) = ½·[E_high − E_low]` and the confirmatory `Δ_mis(p,d) = E*(p,p) − E*(p,d)` (eq6, unclipped,
positive = the switch **cost** induced expression), with ρ and S secondary and Fieller-interval; **(Elig)**
the denominator-eligibility rule, amended by the diagonal-inducibility gate now run **before B1** (rule F1);

### METHOD H1
**H1.** A mismatch between the language of the persona instruction and the language the model converses in
reduces the persona-induced shift in expressed Big Five trait level — measured as a positive `Δ_mis(p,d)`,
averaged over the eligible ordered pairs as `Δ̄_mis`. This is the term (R) predicts and the one instrument
translation cannot manufacture: the neutral arm is generated in the *same* dialogue language it is differenced
against, so D's main effect is removed inside `E*` before the contrast is formed and `Δ_mis` is exactly the
signed P × D interaction contrast. **No confirmatory read is taken on the dialogue-language main effect**
(rule L). H1 is an **intention-to-treat** claim about *assignment* to a requested dialogue language: every
confirmatory number is computed on the first planned generation as produced, including wrong-language output,
with per-cell compliance rates beside it and per-protocol estimates secondary. The bundled grid (C) is reported
alongside as an omnibus pipeline effect — informative about deployment, silent about mechanism — and the
pretraining-resource gradient is descriptive only. The factorial is what separates "the instruction is encoded
language-specifically" from "the questionnaire reads differently in Swahili"; the claim is exactly as wide as
the factorial supports and, per **rule E**, conditional on the 90 fixed evaluation scenarios. The tested
boundary is `max(δ_raw, 2·V̂)` (rule C).



### METHOD falsifier
**Falsifier of record (rule C).** The ONE falsifier is `design_card.json` → `pico.hypothesis.falsifier`,
applied per targeted trait over the admitted checkpoints with the same primary family, the same rule-I
intervals and the same ≥ 3-of-4 quantifier: (A) supported iff the full conjunction of
`preregistration.decision_rule` holds; (B) REFUTED through **either** limb — **(B1) equivalence**, the
simultaneous interval for `Δ̄_mis` lies entirely inside ±δ_raw = 0.09, **or (B2) reversal**, it lies entirely
**below −δ_raw**, so a practically large mismatch effect in the direction *opposite* to the registered one
refutes the directional claim instead of being absorbed into inconclusive and the rule is sign-symmetric;
(C) INCONCLUSIVE otherwise, including any non-declaration in the 0.09–0.15 uninformative band and every
UNTESTABLE branch. Only (B) disproves H1. This section states NO independent falsifier and NO competing
refutation path; the round-trip tie of §2 narrows the claim and never refutes it, and `consistency_check.py`
fails the pre-registration if any competing path reappears.



## R15 superseded verbatim (METHOD 6/7)

### m1
| Primary metric | `Δ_mis(p,d) = E*(p,p) − E*(p,d)` (eq6 on eq1's pole-aligned cell effect), behavior arm; the confirmatory read is the pair-specific contrasts and their equal-weight average `Δ̄_mis` over eligible pairs. Full text: `outcome.primary_metric` |

### m2
| Eligibility | **rule F1**, not restated here — the diagonal-inducibility gate is evaluated **only on the 28 800-transcript qualification tensor of rule H2** (rule E's 60 QUALIFICATION scenarios) and frozen into `ELIGIBLE_PAIRS.json` **before B1**; it enters **no** confirmatory family |

### m3
| Estimator and interval | **ONE algorithm — decision_rule A, executable there and not restated here** (rule A's seed-first ordering is VOID: it discarded the unit-level engine label eq1 needs): unit score with frozen engine label → matched difference → within-stratum average → pole alignment and pair contrast → 1/3 combination → average over pre-B1 eligible pairs → the rule I wild cluster bootstrap-t over the G = S seed clusters → the **R14 single-step max-T region**. Every other aggregator named anywhere is a **sensitivity view that decides nothing** |

### m4
The 6 pair contrasts are a **separate gated family** under the same procedure; **rule F1 removes the gate statistics from both**. Pole heterogeneity is its own family; the language factorial the gated second level; the 30 off-diagonal grid pairs the third, descriptive level — the only place BH survives |

### m5
The **mechanism** claim — a persona/dialogue language mismatch costs induced trait expression, at effects
exceeding the tested boundary max(δ_raw, 2·V̂) (**rule C**; 0.80 power only near 0.15, rule J) — is made only
for the three factorial languages and the four checkpoints, **conditional on the 90 fixed evaluation scenarios,
never resampled (rule E)**.

## ARCHIVED BY R16 - F1 pair-level eligibility, superseded by R15-F3 and amended by R16-F2/F3 (NO FORCE)

F1. PAIR ELIGIBILITY (diagonal inducibility), ESTIMATED ON THE EXECUTABLE QUALIFICATION TENSOR OF SECTION H2 AND ON NOTHING ELSE. The tensor is 12 checkpoint-language cells x 60 QUALIFICATION scenarios x 5 settings x S seeds = 28 800 fresh transcripts at S = 8, budgeted in outcome.n_runs, disjoint from the confirmatory core and the tuning partition. IT IS NOT THE F2 SAMPLE: F2's 720 transcripts are a registered stratified SUBSAMPLE of it (one per checkpoint-language cell x qualification scenario at the first scheduled setting and seed) carrying the human annotations; they estimate evaluator validity, never F1. Two limbs, both computed by the section-A algorithm and the section-I interval on the QUALIFICATION seed clusters: (i) both pole-versus-neutral diagonal effects at the source language aligned in the registered direction with a lower confidence bound above 0; (ii) the diagonal pole-aligned effect E*(p,p) with a lower confidence bound above f_diag = 0.18. THE DEFAULT ESTIMAND IS THE FIXED ALL-SIX-PAIR AVERAGE; a pair is dropped only by this pre-B1 rule and never afterwards. A pre-B1 selection step is not an inference on the confirmatory data, so these gate tests are members of NO confirmatory family. R15-F3 REPLACES THIS SECTION'S PAIR-LEVEL ELIGIBILITY WITH ITS ENUMERATED GATE FAMILY AND LANGUAGE-LEVEL ADMISSION MAP AND GOVERNS: read every later "eligible pair" as "ordered pair inside the admitted L x L subtable", and the admission and UNTESTABLE branches from R15-F3(v)-(vi).


## ARCHIVED BY R16 - F3 admission text superseded by R16-F1/F4/F5 (anchor-calibrated endpoint and tuning-data competence screen VOID) (NO FORCE)

F3. ADMISSION, DETERMINISTIC. A checkpoint is ADMITTED only if it clears the target-language competence screen in ALL THREE dialogue languages (run once on tuning data, frozen before any evaluation generation) AND, in each of its three cells, the anchor slope lies in 0.94-1.06, the intercept within +-0.03, the cross-language slope-equivalence margin within +-0.05, the perturbation-pair sensitivity equivalence holds and every section-F2 threshold passes. Every confirmatory contrast is formed on the ANCHOR-CALIBRATED rater scale - each language's rater output affinely re-calibrated by the anchor slope and intercept estimated on the multilingual anchor set before B1, frozen and hashed before any evaluation transcript is scored - so worst-case language-dependent attenuation of the anticipated 0.60 diagonal effect is at most 0.036, far below delta_raw = 0.09. Failure of ANY of these in a checkpoint-language REMOVES that checkpoint from the admitted set; if fewer than 3 of the 4 remain admitted the behavior channel is NOT ADMITTED, H1 takes outcome (C) and no endpoint is substituted. No calibration, margin, rubric item, anchor, cue list or threshold may change after any transcript is scored. The 20% rater-language subsample of the confirmatory core (H2b) is NON-GATING: a failure there is disclosed wherever the headline result is stated and narrows the claim, but never removes a checkpoint, changes the family or changes an admission.



## ARCHIVED (R16 byte budget) - pico.outcome.n_runs - run tensor now enumerated once in decision_rule E/H/H2/F1/F2 and R16-F4

8 paired seeds per confirmatory cell per setting (4 in the descriptive omnibus grid). This field enumerates the FULL RUN TENSOR, reconciles the scenario counts, separates the two channels and distinguishes administrations from item responses.
SCENARIO COUNTS (rule E/rule F4): the corpus holds exactly 180 support scenarios = 30 TUNING (block B0 only, never an evaluation number) + 60 QUALIFICATION (untouched until the two pre-B1 gates) + 90 EVALUATION, with no expansion branch. The confirmatory core uses all 90; the descriptive omnibus grid uses a registered 30-scenario subset of the 90 fixed before any evaluation generation. No other scenario count appears anywhere.
PERSONA CONDITIONS: 2 TARGETED TRAITS x 2 POLES = 4 persona conditions (Extraversion and Agreeableness, high and low), plus 1 NEUTRAL arm shared across traits because it carries no persona text - 5 generation settings per language cell. The inventory measures all five Big Five traits every administration, so the other three are DESCRIPTIVE off-target outcomes at no extra generation cost and enter no confirmatory family.
BEHAVIOR CHANNEL (transcripts). Confirmatory core: 9 cells x 5 settings x 4 checkpoints x 90 scenarios x 8 seeds = 129 600 transcripts. Descriptive omnibus grid: 27 cells x 5 settings x 4 checkpoints x 30 scenarios x 4 seeds = 64 800. Total 194 400. Instrument language is not a factor here.
SELF-REPORT CHANNEL (administrations). Each core transcript is a frozen conversation state on which the inventory is administered once per instrument language I in L3: 129 600 x 3 = 388 800 ADMINISTRATIONS of 70 items = 27 216 000 ITEM RESPONSES, the three administrations of a transcript PAIRED and analysed with transcript-level clustering. The matched text-quality perturbation arm adds 129 600 administrations; omnibus-grid transcripts receive one each at I = D: 64 800.
SCORING. All 194 400 transcripts are scored once by the frozen 24B rater. The rater-language factor re-scores a registered 20% stratified subsample of the core (25 920 transcripts) in each of the two non-native L3 rater languages: 51 840 extra scorings. Rule F: THIS CORE SUBSAMPLE IS A NON-GATING SENSITIVITY - it never removes a checkpoint or changes the primary family or an admission; every admission-affecting check AND the pair-eligibility gate run on QUALIFICATION, where the admitted checkpoint set, eligible pair set and primary family freeze.
SEEDS. 8 is not a closed-form number: it is the smallest count at which the frozen simulation of the COMPLETE decision rule reaches 0.8 power at delta_alt (decision_rule section J), and section H registers, hashes and budgets the run tensor and nuisance allocation for EVERY allowed S in {8, 9, 10, 11, 12} before M1 (129 600 confirmatory transcripts at S = 8, +16 200 per further seed, 194 400 at S = 12), with M2 selecting the smallest sufficient S deterministically.
QUALIFICATION CHANNEL, TWO DISTINCT OBJECTS, BOTH BUDGETED (sections F1, F2, H2). (a) THE F1 QUALIFICATION RUN TENSOR: 12 checkpoint-language cells x 60 qualification scenarios x 5 settings x S seeds = 28 800 transcripts at S = 8, +3 600 per further seed to 43 200 at S = 12, allocated by the sixth indexed block of NUISANCE_SCHEDULE.json under the same engine-wording orthogonality rule and scored once each by the frozen 24B rater at I = D. This tensor, and only this tensor, estimates the pair-eligibility gate. (b) THE F2 ANNOTATION SUBSAMPLE: a registered stratified subsample of (a) - one transcript per checkpoint-language cell x qualification scenario at the first scheduled setting and seed, 720 transcripts - each double-annotated blind with a third adjudicator (1440 primary annotations plus adjudications). It estimates evaluator validity only. Both are complete and frozen before evaluation generation; no confirmatory transcript is used for either.


## ARCHIVED (R16 byte budget) - pico.outcome.primary_metric - estimand and scale now stated once in decision_rule A / R15-F1 / R16-F1-F4

Mismatch INTERACTION on the XLPT-Support BEHAVIOR channel, native trait scale, defined in every admitted cell and never clipped; the authoritative rule is preregistration.decision_rule sections A-N plus R15-F1 to R15-F4, which govern and are NOT restated here. (i) CELL EFFECT (eq1): E_q(p,d) is the equal-weight (1/3) combination over the three fixed engine strata of the within-stratum mean of the matched (Y^A_q - Y^B) inside the single cell (P = p, D = d) for pole q. (ii) POLE-SEPARATION CELL EFFECT: E*(p,d) = 0.5*[E_high(p,d) - E_low(p,d)], frozen signs sigma_high = +1, sigma_low = -1; the shared neutral arm B CANCELS EXACTLY, so E* is the persona-induced HIGH-VERSUS-LOW POLE SEPARATION in that cell and NOT a persona-versus-neutral shift (R15-F2), the matched neutral difference being retained for variance reduction and for the rule F3 gates only. (iii) CONFIRMATORY ESTIMAND (R15-F1): Delta_mis_bar, the BALANCED MATCH-MINUS-MISMATCH CONTRAST over the admitted language subtable A of size L >= 2, diagonal weight 1/L and off-diagonal weight -1/(L(L-1)), whose weights sum to zero along every row and every column, so both additive main effects are annihilated algebraically and the quantity is a PURE P x D INTERACTION contrast; at L = 3 with the full table retained it is identically the equal-weight mean of the six ordered pair contrasts. The sign is single-valued: POSITIVE means the language switch COST induced expression. (iv) The six ordered Delta_mis(p,d) = E*(p,p) - E*(p,d) are SIMPLE EFFECTS at fixed p, carrying the additive dialogue-language term, reported with the section-I region as SECONDARY DESCRIPTIVE quantities and never confirmatory. rule G DEFINES THE SCALE: the equal-weight mean of the six frozen, behaviorally anchored 1-7 facet items of refine-logs/RUBRIC_BEHAVIOR.md for the targeted trait, on the native scale, before the rule F3(i) anchor calibration; the item list is the only mapping from scored cues to the trait, nothing is estimated from data, and its convergent, discriminant and anti-stereotype validity against a blind human criterion is a POWERED PRE-B1 GATE. Admission is language-level and deterministic (R15-F3). H1 is an INTENTION-TO-TREAT claim about assignment (rule K). Instrument language is NOT a factor of this channel: the transcript is produced under P and D before any inventory is administered. Intervals per outcome.variance_protocol.significance_test; the shortfall S = 1 - rho is a secondary summary with Fieller intervals. The authoritative confirmatory rule is preregistration.decision_rule; no main effect is a confirmatory endpoint, the language factorial runs in full as the gated second-level family, and the apparatus-artifact exclusion is EXPLORATORY (R15-F4)


## ARCHIVED (R16 byte budget) - pico.outcome.multiplicity_correction - family map now enumerated once in decision_rule I(iv)-(v) and R16-F4(c)

One confirmatory outcome hierarchy, family-wide, under ONE INVERTIBLE PROCEDURE: the R14 STUDENTIZED SINGLE-STEP MAX-T TEST and the simultaneous region that IS its inversion, executable in decision_rule rule I(iv)-(v) and not restated here. Never Holm with separately constructed intervals and never Benjamini-Hochberg: BH controls a false-discovery rate while rule B reads INTERVAL COVERAGE. PRIMARY FAMILY (frozen before B1 in ELIGIBLE_PAIRS.json): TWO COMPONENTS PER (ADMITTED CHECKPOINT x TARGETED TRAIT) - Delta_mis_bar and Delta_mis_bar - 2*V - at most 16; a null is REJECTED IF AND ONLY IF it lies outside that component's reported interval, and the random boundary is inverted JOINTLY with the fixed margin, so every rule B and rule C decision - support, equivalence, reversal - reads that ONE region and none can disagree. PAIR-LEVEL FAMILY (rule I(v)) SEPARATE AND GATED on it, same procedure, same replications; rule B(3) reads point estimates there. The average and the pair contrasts are NEVER members of one family. HOLM and STEP-DOWN (Romano-Wolf) MAX-T CONSTRUCTIONS and the three-shifted-null maximum-p-value combination are VOID (section N). SECOND LEVEL, gated on the primary family and reported after it: the remaining terms of the behavior channel's P x D language factorial, in their own enumerated family under the same Holm and max-T procedure, never a confirmatory endpoint. SEPARATE SELF-REPORT FAMILY (H2): the within-transcript instrument-language contrasts and the matched text-quality perturbation contrasts, Holm at 0.05 with max-T simultaneous intervals within that family and transcript-level clustering, never pooled with the behavior family. POLE-HETEROGENEITY FAMILY: the pole contrasts over the same 6 x 4 x 2 cells, Holm within their own family, never pooled with the primary family and never gating H1. THIRD LEVEL, DESCRIPTIVE, THE ONLY PLACE BH SURVIVES: the 30 off-diagonal pairs of the omnibus grid (C) and the three off-target traits, BH at 0.05 within the grid and labelled omnibus and descriptive. The primary family spans BOTH targeted traits and the complete rule must hold independently for each (decision_rule rule B). THE DIAGONAL-INDUCIBILITY GATE STATISTICS ARE IN NO CONFIRMATORY FAMILY (rule F1): the gate is a PRE-B1 SELECTION step on the independent qualification run tensor of rule H2, not an inference on confirmatory data. Checkpoint admission AND the eligible pair set are decided BEFORE B1 on that tensor, never on confirmatory-core results; the primary family is frozen at that moment. Every other analysis named descriptive or exploratory in pico.definitions or outcome.n_runs enters no family and carries unadjusted intervals.


## ARCHIVED (R16 byte budget) - pico.preregistration.analysis_plan - analysis algorithm now stated once in decision_rule A/B/C/I/K as amended

Locked before the evaluation phase; the B3 ablation scenario IDs, selection algorithm, contrasts and reuse rules are frozen and hashed before B1 starts, so no ablation subset can be chosen after B1 exposes the evaluation scenarios. For each checkpoint, trait and cell, compute the persona-minus-neutral induced shift (eq1), form E* and then Delta_mis(p,d) = E*(p,p) - E*(p,d). ESTIMATION AND INTERVALS FOLLOW ONE ALGORITHM AND ONE ONLY - decision_rule rule A; every alternative aggregator or bootstrap named anywhere is listed in decision_rule section N and IS A LABELLED SENSITIVITY VIEW THAT DECIDES NOTHING. rule B(2): both practical-threshold limbs are tested as single-step max-T lower bounds of the one region with V_hat recomputed inside every replication. rule C: the tested boundary is max(delta_raw, 2*V_hat). The PRIMARY family is the ordered pair-specific mismatch contrasts and their average over the pairs declared eligible before B1 (rule F1; the gate statistics are NOT in the family), over the ADMITTED checkpoints x 2 targeted traits; no main effect is in it. Gated on it and reported afterwards: the behavior channel's P x D language factorial and, on the self-report channel's frozen transcripts, the eq7 P x D x I decomposition carrying H2, whose score EXCLUDES the 12 development-touched BFI-2 items (rule E). The omnibus grid (C) is descriptive with its own BH correction; ratios and shortfalls only on cells passing (Elig), with Fieller intervals and censored cells listed; the asymmetry index and the single eq5 resource statistic censoring-aware with worst- and best-case bounds; H3 by simultaneous Holm-adjusted TOSTs at epsilon. All four hypotheses are read against the same numbers, each bound in advance to the pattern that selects it: a loss carried by the instrument-language effect reads as H2; one vanishing on rater-free indicators as H2b; equivalence throughout as H3, and only through the equivalence tests; a loss carried by the pair-specific mismatch contrasts as H1. A bundled-grid loss with no factorial term surviving is an omnibus pipeline effect whose mechanism is unresolved - a real result, not a failure. Rule E (the earlier 18/12 scorer-only quarantine is VOID): TUNING, QUALIFICATION AND EVALUATION ARE DISJOINT SCENARIO PARTITIONS, cut before any tuning and hashed in SCENARIO_PARTITION.json - TUNING 30, QUALIFICATION 60, EVALUATION 90. Prompt and persona wording, parsing, scorer and rubric configuration, translation settings and every variance re-estimate are chosen on TUNING alone; the rater is then frozen by hash and QUALIFICATION is read once, before B1, for the rule F1 and F2 gates only. All inventory items and rater-free indicators are validated and FROZEN before the first evaluation administration. Differential-item-functioning screening on evaluation data is a SENSITIVITY ANALYSIS ONLY: no item is added, removed, re-weighted or re-scored on evaluation-derived DIF; flagged items are reported and the primary analysis repeated with them excluded as a labelled sensitivity view. NUISANCES ARE BLOCK-ALLOCATED, NOT ALIASED WITH SEED, under rule H: one pre-registered schedule, frozen before any generation, assigns for every (cell, setting, scenario, seed) the sampling seed, forward-translation engine, persona wording variant, item order and option order; engine and wording are block-allocated inside each cell-setting and never rotated across seeds, item and option order independently randomized. Uncertainty: the single rule-I interval. See outcome.primary_metric for the estimand and decision_rule for the confirmatory rule.


## ARCHIVED (R16 byte budget) - pico.preregistration.predicted_result - prediction demotion; power certificate now R16-F6

R7-Q1 (design-kill R6 partial 1) RESOLVES THE UNDERPOWERED-PREDICTION PROBLEM BY DEMOTION, NOT BY HOPE. rule C REWRITES THE CLAIM SENTENCE: the confirmatory claim is that Delta_mis_bar exceeds the TESTED BOUNDARY max(delta_raw = 0.09, 2*V_hat); delta_alt = 0.15 is only the alternative at which the complete conjunction reaches 0.80 power (0.824 at the registered 8 seeds; decision_rule rule J), it is never claimed as the tested effect, and the abstract states both the boundary and the sensitivity limit (rule J). The earlier point prediction of Delta_mis_bar ~ 0.12 is DEMOTED to a descriptive prior expectation with NO registered role: it sizes nothing, it gates nothing, and no declaration or non-declaration is interpreted against it. Directional expectations retained, all descriptive: Delta_mis_bar positive in every admitted checkpoint, at least 5 of the 6 ordered pair-specific contrasts positive, largest for a Swahili dialogue language and smallest between English and Chinese; the self-report instrument-language main effect about 0.04 and the pole contrast about 0.02, i.e. not detectable at the registered run count; the round-trip control shortfall below 0.05; between-engine variance under 0.04; interrater agreement above 0.7 in all six languages. Every pre-R5 prediction stated on the deleted main effects or on the retired 0.25 shortfall scale is VOID. Whatever the conjunction returns, the primary deliverable is the ESTIMATE AND INTERVAL of Delta_mis_bar for every admitted checkpoint and trait, and a non-declaration in the 0.09-0.15 band is reported as UNINFORMATIVE (outcome (C) of the R7 falsifier of record), never as evidence that persona traits transfer across languages.


## ARCHIVED (R16 byte budget) - pico.controls.evaluator_validity - validity gates now stated once in decision_rule F2-F4 as amended by R16

The behavior arm is scored by an automatic rater whose validity is established before use, not assumed. Agreement is established on the SAME 720 blind double-annotated qualification transcripts that carry the rule-F2 criterion gate (one per qualification scenario in each of the 12 checkpoint-language cells, a third annotator adjudicating), so no separate annotation sample exists: the rater must reach Krippendorff alpha >= 0.7 against the adjudicated labels in EVERY language and pass a cross-language agreement equivalence test at a 0.05 margin, or the behavior arm is descriptive only. Independently, four rule-based rater-free indicators (escalation rate, refusal rate, mean turn length, apology-token rate) reproduce the primary comparison with no scoring model. Self-report validity is screened by per-trait per-language internal consistency, measurement invariance with differential item functioning flags, and BFI-2/TIPI convergence; items failing invariance are reported separately and excluded under the preregistered rule, and rule E additionally excludes the 12 development-touched BFI-2 items from the confirmatory H2 score. Agreement is never read as equal calibration or equal sensitivity. Rule F: EVERY ADMISSION-AFFECTING CHECK IN THIS FIELD - agreement threshold, cross-language equivalence test, anchor pairs, perturbation pairs, rater-language artifact check - RUNS ON THE 60-SCENARIO QUALIFICATION PARTITION, disjoint from the tuning partition and the confirmatory core, the admitted checkpoint set freezing before B1; the 20% rater-language subsample of the core is a NON-GATING sensitivity. Two pre-B1 qualifications per language: (i) MULTILINGUAL ANCHOR PAIRS - frozen transcripts carrying known injected trait deltas, content- and length-matched and rendered into every language through the same pipeline; the rater must recover the delta with slope in 0.94-1.06 and intercept within +-0.03 in EVERY language, slopes passing cross-language equivalence at +-0.05, or that checkpoint-language leaves the admitted set (decision_rule rule F3); (ii) PERTURBATION PAIRS - minimally edited pairs differing by one trait-diagnostic behavior, certifying equal detection sensitivity across languages within the same margin. For the four rater-free indicators, trait-specific configural and metric invariance is established per language; one failing invariance for a trait is descriptive for that trait and leaves the confirmatory hierarchy. rule G: the behavior rubric, its six anchored 1-7 facet items per trait and the trait score are frozen in RUBRIC_BEHAVIOR.md, and rater CONSTRUCT validity - convergent, discriminant and anti-stereotype criterion validity, estimated INSIDE each checkpoint-language cell on its 60 independent scenario clusters by the rule-F2 scenario-resampling bootstrap - is a pre-B1 admission gate. The agreement, calibration and sensitivity checks here do not by themselves establish construct validity and are never read as doing so.



## ARCHIVED BY R16 - rule G transcript-language rendering clause, VOID under R16-F1(a)

G. THE BEHAVIOR RUBRIC AND SCALE, frozen and hashed as RUBRIC_BEHAVIOR.md before any generation. For each targeted trait the rater scores the transcript on SIX facet items derived from that trait's BFI-2 facet definitions in customer-support-dialogue terms - Extraversion: sociability, assertiveness, energy level; Agreeableness: compassion, respectfulness, trust; two items per facet - each on an INTEGER 1-7 scale with written behavioral anchors at 1, 4 and 7, rendered into the transcript language by the same frozen translation pipeline as every other instrument. The trait score is the EQUAL-WEIGHT MEAN of its six items on the native 1-7 scale, before anchor calibration. The item list IS the cue-to-trait mapping: no weight, item or cue is estimated from data, nothing outside the six items enters the score, and the rater returns all six item scores so every aggregate is reconstructible. The rater-free indicators of controls.evaluator_validity are a descriptive convergence check and substitute for nothing.


## ARCHIVED BY R16 - section J optimistic planning values, replaced by the R16-F6 conservative pre-B0 certificate

J. POWER, AND THE HONEST-POWER STATEMENT. delta_alt is the smallest true average mismatch effect, on a 0.01 grid strictly above max(delta_raw, 2*V_hat), at which the frozen Monte Carlo of the COMPLETE conjunction reaches power 0.80. The simulator power/power_sim_r6.py (sha256 42c8064c0892ed37eb4e5364799e21c6708cff260b329cc4d522b111f542391e, 100000 simulations, generator seed 20260911) encodes family-wide BH significance AND the max(delta_raw, 2*V_hat) boundary with V_hat drawn from its own sampling distribution AND the 4-of-6 pair consistency side-condition AND the >= 3-of-4 admitted-checkpoint requirement AND the whole rule holding independently for BOTH targeted traits AND pole concordance, and it reports the (B2) reversal branch beside the (B1) equivalence branch. At the registered planning values (sigma_pair 0.050 at 8 seeds x 90 scenarios, rho_pair 0.40, tau_pair 0.020, tau_ckpt 0.020, sigma_pole 0.060, V 0.050 on 42 df) power/POWER_SIM_R6.json gives delta_alt = 0.15 with power 0.824 at 8 seeds (0.023 at 0.09, 0.332 at 0.12, 0.985 at 0.18); 12 seeds give 0.907 and do not move delta_alt on the grid.


## ARCHIVED BY R16 - section J placeholder-figure paragraph, superseded by R16-F6

THE PLANNING VALUES ARE PLACEHOLDERS: power_sim_r6.py and power_sim_r9.py are re-run on the TUNING-partition variance components BEFORE any evaluation generation, and the delta_alt they return then is the registered one. BOTH SIMULATORS ARE RE-RUN UNDER SECTION I AS IT NOW STANDS - the G = S seed-cluster Webb bootstrap, V_hat on G - 1 df, the R14 single-step max-T region over the two-component-per-(checkpoint x trait) primary family - so the V_hat distribution, the degrees of freedom and every power number are computed under the SAME cluster and family the analysis uses; every figure in this section obtained under family-wide BH or any other cluster is a PLANNING PLACEHOLDER and is VOID as a certificate. DETERMINISTIC BRANCH, decided at that moment and never after: if power at that delta_alt is below 0.80 at 8 seeds, S rises to the smallest sufficient value in {8,...,12} under section H; the scenario lever does not exist; if 12 seeds still do not suffice, delta_alt is RAISED to the smallest simulated value reaching 0.80 and the enlarged uninformative band is declared in the abstract. Raising delta_alt is never treated as a gain in sensitivity.


## ARCHIVED (R16 byte budget) - pico.outcome.variance_protocol.significance_test

Rule A NAMES ONE CONFIRMATORY ESTIMATOR-AND-INTERVAL ALGORITHM, stated once in decision_rule rule A; every other procedure in every artifact is a labelled sensitivity view that decides nothing - in particular the paired two-way scenario x seed cluster bootstrap, the scenario-only cluster bootstrap, the seed-first ordering of earlier revisions and the interquartile-mean aggregator. In one line: unit-level scores keep their frozen engine label, matched differences are formed FIRST at unit level, aggregated over (scenario, seed) WITHIN each fixed engine stratum, pole-aligned and contrasted inside the stratum, then combined at weight 1/3; interval by WILD CLUSTER BOOTSTRAP-t over the G = S SEED CLUSTERS - the seed is the ONLY resampling unit, G = 8 at the default count, one Webb 6-point weight per seed per replication shared globally across every stratum, cell, setting, checkpoint, trait and contrast, 9999 replications, null imposed, studentized by the cluster-robust SE with the G/(G-1) finite-cluster correction and V_hat recomputed inside each replication on G - 1 = 7 df - then the R14 STUDENTIZED SINGLE-STEP MAX-T PROCEDURE over the frozen primary family's two components per (admitted checkpoint x targeted trait), the reported intervals being the inversion of that same test. The (scenario, seed) cluster, any BH-adjusted interval and any grid-inversion interval are VOID (decision_rule rule I, section N). THE 90 EVALUATION SCENARIOS ARE FIXED AND NEVER RESAMPLED; all inference is conditional on that corpus (rule E). Ratio-scale summaries use Fieller intervals on cells passing (Elig), unclipped. Equivalence claims (H3, rater agreement, translation ensemble) use two one-sided tests at their registered margins, Holm-adjusted and holding simultaneously. Nuisance allocation is stated once in preregistration.analysis_plan.



## ARCHIVED (R16 byte budget) - definitions[5] language factorial core - trimmed, instrument language now fixed by R16-F1

Three-way language factorial core. The three factors that (C) bundles are manipulated SEPARATELY over L3 = {English, Chinese, Swahili} - one high-resource Latin-script, one high-resource non-Latin-script, one low-resource - fully crossed: P x D x I, 27 cells per checkpoint per arm, of which 18 are new (the 3 all-equal cells and the 6 cells with P = s, D = I = t are reused run-for-run from (C)). Matched controls: every rendering passes the same three-forward-translation plus back-translation screen, so translation procedure is constant across factors; generation length is matched by a per-language band calibrated on the tuning partition (identical turn count, mean turn length within +-10%), realized token counts reported per cell. Estimands, each a within-scenario, within-seed contrast against P = D = I = s: the three main effects, the three two-way interactions and the three-way term, on the unnormalized scale by eq6 and decomposed by eq7. NONE is a confirmatory endpoint for H1. THE FACTORIAL IS SPLIT BY OUTCOME CHANNEL: the transcript is produced under P and D and THEN frozen, so instrument language cannot be a factor for it. The BEHAVIOR channel is a P x D factorial only (9 cells over L3), read confirmatorily as the pair-specific mismatch contrasts of (Mismatch); the SELF-REPORT channel crosses I on those same frozen transcripts (each core transcript re-administered once per I, paired within transcript, transcript-level clustering) and carries H2; a separate pre-registered RATER-LANGUAGE factor applied post hoc to identical frozen transcripts on a 20% stratified subsample carries H2b. The 27-cell P x D x I read is withdrawn for the behavior channel. The authoritative confirmatory rule is preregistration.decision_rule: no main effect is a confirmatory endpoint, THE LANGUAGE FACTORIAL RUNS IN FULL as the gated second-level and self-report families (section L), and only pairs declared eligible before B1 (section F1) enter Delta_mis_bar.


## ARCHIVED (R16 byte budget) - effect_size_reported - trimmed to the rule A pointer

Every confirmatory contrast is reported on the unnormalized trait scale as a point estimate with its 95 percent rule A interval - wild cluster bootstrap-t over seed clusters within fixed engine strata, CONDITIONAL ON THE 90 FIXED EVALUATION SCENARIOS, never resampled (rule E) - alongside the paired standardized effect size and the number of contributing generations. The earlier "scenario-clustered bootstrap interval" wording is VOID as an inferential statement and survives only as a labelled sensitivity view. Ratio-scale summaries are reported only for cells passing (Elig), with Fieller intervals, unclipped, and censored cells are listed rather than omitted. No significance claim is reported without its interval. R5: every confirmatory value is the pole-averaged Delta_mis, reported beside its two per-pole values whenever the registered pole-heterogeneity test survives correction.


## ARCHIVED (R16 byte budget) - mde - trimmed

0.09 scale points is the confirmatory minimum detectable effect: delta_raw = 0.15 x median|Delta(s,s)|, the audit-relevant loss on the unnormalized scale, fixed per model and trait from the development split before any evaluation generation, so the confirmatory test never depends on a per-cell denominator. With a median diagonal effect near 0.60 scale points the working value is delta_raw = 0.09; the H3 equivalence margin is epsilon = 0.075 of the induced shift, epsilon_raw near 0.045. Ratio-scale quantities keep the 0.15 shortfall threshold for interpretation only. R5: the confirmatory threshold is max(delta_raw, 2*V_hat), V_hat defined in preregistration.decision_rule. rule C: max(delta_raw, 2*V_hat) is THE TESTED BOUNDARY and the only threshold the study claims against; delta_alt = 0.15 is the powered alternative (rule J, rule J) and is never an MDE claim, a threshold or a claim boundary.



## ARCHIVED (R16 byte budget) - hypothesis.statement - R16 harmonisation and trim

CONFIRMATORY H1 (observable, no representation claim): holding the persona language fixed, making the model converse in a DIFFERENT language than the persona instruction was written in reduces the PERSONA-INDUCED HIGH-VERSUS-LOW POLE SEPARATION in expressed Big Five trait level. The estimand is Delta_mis_bar, the BALANCED MATCH-MINUS-MISMATCH INTERACTION CONTRAST over the admitted L x L language subtable (R15-F1): diagonal weight 1/L, off-diagonal weight -1/(L(L-1)), summing to zero along every row and every column, so both additive main effects cancel algebraically and the quantity is a PURE P x D interaction. The neutral arm carries ZERO weight in it and NO persona-versus-neutral magnitude is claimed (R15-F2). The admitted languages and checkpoints are settled BEFORE B1 by the enumerated qualification-gate family and deterministic language-level admission map of R15-F3, so no retained subset can unbalance the contrast. rule C: the confirmatory claim is that Delta_mis_bar EXCEEDS max(delta_raw = 0.09, 2*V_hat); delta_alt = 0.15 is a POWER statement only (0.80 power there), never a claim or threshold. The six ordered pair contrasts Delta_mis(p,d) are SIMPLE EFFECTS and are reported as secondary descriptive quantities under the same region. Full definition in outcome.primary_metric; the authoritative rule (single family, the rule B(2) two-limb threshold, 4-of-6 simple-effect consistency, >= 3 of 4 ADMITTED checkpoints, pole concordance, both traits independently) is preregistration.decision_rule; the falsifier of record is hypothesis.falsifier. No main effect is confirmatory; the language factorial is run IN FULL and reported as the gated second-level family, and the apparatus-artifact exclusion is EXPLORATORY (R15-F4)


## ARCHIVED (R16 byte budget) - intervention.change - R16 harmonisation and trim

Prior work measures induced trait expression inside one language at a time, so induction and assessment language are always identical and their effects cannot be separated. This design crosses them: the persona instruction is applied in language s while the dialogue, the inventory items and the response-scale labels are rendered in language t, over the full directed 6x6 grid of English, Chinese, Spanish, German, Arabic and Swahili. Each cell is a difference between a persona arm (A) and a content-matched neutral arm (B), so the cell value is a persona-induced shift rather than an absolute trait score, and the diagonal cell supplies the same-language reference. The toggled component (C) is the assessment language relative to the induction language; model, persona semantics, dialogue content, decoding configuration, inventory version and translation procedure are identical across cells. On the user's instruction of 2026-09-11 to RUN THE LANGUAGE FACTORIAL: the bundled switch is retained as the omnibus grid (C) and a THREE-WAY LANGUAGE FACTORIAL CORE (LangFac) crossing P x D x I over {English, Chinese, Swahili}, with matched translation procedure and length band, runs in full (rule L). No confirmatory claim is carried by any main effect - the confirmatory read is the observable P x D mismatch contrast of (Mismatch) over the pairs declared eligible before B1 (rule F1); the factorial's main effects and interactions are the gated second-level and self-report families, and the grid carries no mechanism claim.


## ARCHIVED (R16 byte budget) - compute_budget - trimmed

Inference only, no training, computed from the full run tensor in n_runs at the default S = 8 block, the 28 800-transcript qualification tensor of decision_rule H2 included (it adds roughly 11-20 GPU-hours of generation and rater scoring inside the development-and-qualification envelope, now 110-130 GPU-hours); the S = 12 ceiling of decision_rule section H raises the confirmatory transcript count from 129 600 to 194 400 and scales this envelope by at most 1.5x. Generation: 194 400 transcripts at roughly 800 prompt / 512 output tokens = about 155M prefill and 100M decode over four checkpoints, roughly 40-70 GPU-hours under continuous batching (vLLM) on one A100 80GB. Self-report: 583 200 administrations, each carrying its transcript in context at roughly 1300 prompt / 140 output tokens = about 760M prefill and 82M decode, roughly 30-50 GPU-hours. Rater scoring: 194 400 transcripts plus 51 840 rater-language re-scorings through the 24B rater at about 1200 prompt / 40 output tokens = about 295M tokens, roughly 35-60 GPU-hours. With the identical-condition repeat arm, the ablation subset and a 1.5x allowance for PRESPECIFIED INFRASTRUCTURE FAILURES ONLY (never outcome-dependent regeneration - see the ITT rule in decision_rule), the registered envelope is APPROXIMATELY 240-400 GPU-HOURS on one A100 80GB, or 150-250 on one Hopper-class 80GB, plus roughly 60 CPU-hours. HARDWARE: the four graded checkpoints fit a 48GB card (L40S or A6000) at bf16; ONE 80GB CARD IS REQUIRED to hold the 24B rater with its KV cache at usable batch sizes.



## ARCHIVED (R16 byte budget) - METHOD_DESIGN.md §10 R14/R15 repair narrative (NO FORCE)

R14 repaired the invalid-inversion fatal the R13 design-kill
returned, **R15** repaired the four the R14 design-kill returned and **R16** repairs the four the R15
design-kill returned; **the executable text is in the card (`decision_rule` sections A–N plus R15-F1 to
R15-F4 and R16-F1 to R16-F7), which governs and is NOT restated here**. In one line
each: R14 gave the pre-B1 gate an executable run tensor (F1, H2), fixed ONE resampling unit — the seed alone,
G = S, V̂ recomputed inside every replication (I(i)–(iii)) — and replaced R13's step-down critical values,
which were read off observed-ordering active sets and reused as interval half-widths (**VOID**), with ONE
studentized **single-step max-T** test at FWER 0.05 over the fixed ≤ 16-component primary family whose
inversion **is** the reported simultaneous 95 % region (I(iv)–(v)); R15 makes the confirmatory endpoint the
**balanced match-minus-mismatch interaction** over the admitted language subtable, narrows the claim to
**pole separation**, enumerates the pre-B1 **qualification-gate family** and its deterministic admission
mapping, and demotes the apparatus-artifact exclusion to **exploratory** (R15-F1 to R15-F4). 


## ARCHIVED (R16 byte budget) - EXPERIMENT_PLAN.md §8 R15 landing list (NO FORCE)

In this plan they land as §1's balanced interaction endpoint on pole separation (R15-F1/F2), B0's 30 / 60 / 90
partition (E) and language-level admission gate (R15-F3) on the 28 800-transcript qualification tensor with its
720 blind double-annotated transcripts (F2, H2), §6's UNTESTABLE branch with no corpus expansion (F4, R15-F3),
§3/§4's pre-hashed nested seed blocks (H), and B1b's seed-cluster bootstrap with the R14 single-step max-T
region (I). C2 is an exploratory read only (R15-F4). 


## ARCHIVED (R16 byte budget) - hypotheses[1] H2 statement - trimmed, behavior instrument fixed by R16-F1

rule L NARROWS H2 TO WHAT THE DESIGN CAN OBSERVE: the shortfall is an INSTRUMENT-ADMINISTRATION-PIPELINE effect - administering the inventory in another language changes item semantics, tokenization, comprehension, option interpretation and the response process together. No passive response-scale artifact and no unchanged-latent-state claim is identified by this design or asserted. R4 gives H2 an estimand that respects the causal order: instrument language I is administered AFTER the transcript exists, so it cannot affect the behavior channel and is NOT a factor there. H2's estimand is the WITHIN-TRANSCRIPT instrument-language contrast on the SELF-REPORT channel - the same frozen conversation state re-administered with the inventory rendered in each I in L3, paired and analysed with transcript-level clustering, scored on the 48 items that rule E leaves after excluding the 12 development-touched BFI-2 items. Alongside it, a MATCHED TEXT-QUALITY PERTURBATION arm varies rendering quality at FIXED final instrument language, separating 'the instrument is in another language' from 'the instrument text is worse'. The round-trip translation control is a SENSITIVITY only and its R3 tie rule is DELETED: a round trip returns text to the ORIGINAL language while degrading it, so it varies degradation and final language together.


## ARCHIVED (R16 byte budget) - definitions[6] eligibility rule - trimmed

Denominator-eligibility rule for every ratio-scale quantity per DESIGN_KILL R2 finding 2. A cell's ratio ρ and shortfall S are reported only if its diagonal reference effect Δ(s,s) for that model, source language and trait is bounded away from zero on BOTH of two pre-registered tests: its two-sided bootstrap interval excludes zero under the same R14 single-step max-T procedure as the primary family, AND |Δ(s,s)| ≥ max(0.30 scale points, 4·SE(Δ(s,s))). The absolute floor is set at 0.30 because the registered dispersion prior σ = 0.11 makes a smaller diagonal effect indistinguishable from measurement noise at 8 seeds, and the 4·SE condition caps the ratio's relative standard error near 0.25, which is where Fieller intervals stay finite. Cells that fail (Elig) are reported as RATIO NOT DEFINED — never imputed, never clipped, never dropped silently — their count is reported per language and per model, and they still contribute their unnormalized (M) difference to the confirmatory analysis, which needs no denominator.



## ARCHIVED (R16 byte budget) - pilot_std_provenance - trimmed, R16-F6 certificate named

Between-seed dispersion of repeated identical-condition induced-shift measurements. The literature prior is a relative dispersion of 0.11 of the induced shift (ref-010, repeated-measurement instability of induced trait scores under fixed prompts). R3 carries it onto the confirmatory unnormalized scale by multiplying through the median diagonal effect of 0.60: sigma = 0.066 scale points per seed for the paired within-scenario contrast. It is a literature-derived prior, not an in-house pilot, and only seeds the power simulation: the variance components driving the registered seed count are re-estimated on the development split before any evaluation generation, and the seed count is recomputed and recorded if they exceed the prior.


## ARCHIVED (R16 byte budget) - confounders_addressed[10] competence screen - moved off tuning by R16-F5

Pre-registered target-language competence screen. Run before the evaluation phase, on the development split only, for every checkpoint x target language: 50 held-out neutral support prompts scored for response-language accuracy (language identifier at >=0.9 confidence) and instruction compliance under the frozen decoding configuration. A cell enters the grid only at accuracy >= 0.95 and compliance >= 0.90; failing cells are reported OUT OF COMPETENCE and excluded, because a model that cannot answer in Swahili would otherwise register as a model whose induced trait does not transfer to Swahili - the exact confound the study is about. Run once, frozen, never re-run after seeing evaluation data. rule F3/rule F: a checkpoint enters confirmation only if it clears ALL THREE L3 languages, decided before B1, so every admitted checkpoint carries the same 6 candidate pairs; fewer than three qualifying makes H1 UNTESTABLE.


## ARCHIVED (R16 byte budget) - leakage L3 - trimmed

Development and evaluation scenarios are drawn from the same schema pool and rendered by the same translation procedure, so the two splits match in domain, length and register; near-duplicates across the splits are removed. The mismatch that does exist is between the six studied languages and languages in general, and between support dialogue and open-ended interaction; both are stated as scope limits rather than treated as covered. rule E: the 90 evaluation scenarios are a FIXED purposive corpus and are never resampled; all confirmatory inference is conditional on them, dependence across units is handled by resampling seed clusters within fixed engine strata (decision_rule rule A), and no domain-general claim about support dialogue is made.


## ARCHIVED (R16 byte budget) - metric_discriminative - trimmed

The effect-retention shortfall discriminates the competing hypotheses because each predicts a distinct pattern in the same numbers: H1 a shortfall above the within-condition measurement variability, larger for lower-resource target languages and asymmetric between the two directions of a pair; H2 a shortfall that shrinks toward that variability once between-translation variance is removed and concentrates on DIF-flagged items; H2b a shortfall on the rater-scored arm that disappears on the rater-free indicators; H3 a shortfall indistinguishable from zero everywhere. The metric is interpretable on its native scale and the control condition occupies a known position on it, so a tie with the control is directly readable.


## ARCHIVED (R16 byte budget) - held_fixed_nuisances[6] - trimmed

Translation procedure and its three forward-translation engines pinned by version for the whole study: Google Cloud Translation v3, Microsoft Azure AI Translator v3.0, and facebook/nllb-200-3.3B run locally at a pinned commit revision. All three cover all six study languages including Swahili, which is why DeepL was not used; the open-weight leg keeps the translation step reproducible after the study ends. Every rendered item records its engine, all three renderings are retained, and engine invariance is a reported sign-concordance check across the three FIXED strata, not a resampled inference (decision_rule rule N, rule A).



## ARCHIVED (R16 byte budget) - the design-kill rationales of the R15/R16 repair clauses (NO FORCE; each clause itself governs)

- fatal 1: a fixed-p mismatch contrast is a SIMPLE EFFECT in which an additive dialogue-language term survives
- fatal 2: the shared neutral arm cancels from E*
- fatal 3: F1 invoked section I, which enumerates only the post-gate families, so the gate was not uniquely executable
- fatal 4: C2 had no equivalence estimand, no margin, no multiplicity rule and no power certificate
- fatal 1: rule G rendered the rubric into the transcript language, so the instrument changed with the dialogue language D and a scoring change alone could satisfy H1
- fatal 2: R15-F3(iv
- fatal 3: the L = 2 branch left a subtable with two ordered mismatch pairs where rule B(3
- fatal 4: F3 conditioned the endpoint on estimated anchor slopes and intercepts without propagating their error, and the gating and calibration samples were never enumerated
- residual: an admission-affecting screen ran on the material used to select prompts and pipelines
- residual: the only filed power figures were placeholders, and the variance inputs arrived only after the B0 phase


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule R15-F2, filed in full and binding

R15-F2 THE CLAIM IS POLE SEPARATION, NEVER A PERSONA-VERSUS-NEUTRAL MAGNITUDE.. E*(p,d) = 0.5*[E_high(p,d) - E_low(p,d)] = 0.5*[A_high - A_low] EXACTLY: the neutral arm B carries ZERO WEIGHT in the confirmatory endpoint. The matched persona-minus-neutral unit difference of A(2) is retained ONLY as variance reduction and as the input to the F3 gates; B is a CALIBRATION ANCHOR, not part of the estimand. H1, the title, the abstract, pico.hypothesis, outcome.primary_metric, METHOD_DESIGN.md and EXPERIMENT_PLAN.md therefore state the claim as the PERSONA-INDUCED HIGH-VERSUS-LOW POLE SEPARATION and its loss under a dialogue-language switch; every persona-versus-neutral MAGNITUDE claim is VOID. Rule B(4) survives as a DESCRIPTIVE side-condition with no threshold and no claim.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule R15-F3, filed in full and binding

R15-F3 THE PRE-B1 QUALIFICATION GATE IS ITS OWN FULLY ENUMERATED FAMILY WITH A DETERMINISTIC LANGUAGE-LEVEL ADMISSION MAP.. The gate runs ONLY on the independent 60-scenario QUALIFICATION partition of rule E and the section-H2 qualification tensor, is decided BEFORE B1, and is a SEPARATE family at its own FWER 0.05 sharing no critical value with any confirmatory family. (i) FAMILY, fixed and enumerated in advance: ONE limb per (checkpoint c, language p, targeted trait t), at most 4 x 3 x 2 = 24 limbs, statistic the diagonal pole separation E*(p,p) computed by the section-A algorithm on the qualification partition. (ii) NULL: H0: E*(p,p) <= f_diag = 0.18 native-scale points, one-sided, one null per limb. (iii) RESAMPLING: the I(i)-(iii) seed-cluster bootstrap, G = S clusters, one globally shared Webb 6-point weight per seed per replication, residuals against the (engine stratum x cell x setting) fit under the imposed null, studentized by the G/(G-1) cluster-robust SE. (iv) CRITICAL VALUE: REPLACED IN FULL BY R16-F2, WHICH GOVERNS - one studentized single-step max-T value over the SIGNED t_j, rejecting iff t_j > c; the |t| reading is VOID. (v) ADMISSION MAP, deterministic, AS AMENDED BY R16-F3, WHICH GOVERNS: language p is ADMITTED at checkpoint c iff its limb rejects for BOTH targeted traits; checkpoint c is ADMITTED iff ALL THREE of its languages are admitted, and its confirmatory table is the complete 3 x 3 subtable, which keeps the R15-F1 weights margin-free; the L = 2 branch is VOID. ADMISSION IS NEVER PAIR-LEVEL, so no retained subset can unbalance the contrast; the pair-level eligibility of F1 is VOID and 'eligible pair' means, everywhere, an ordered pair inside the admitted subtable. (vi) If fewer than 3 of the 4 checkpoints are admitted the study is UNTESTABLE, outcome (C) of rule C, and no corpus is expanded. The 24 t-statistics, the critical value, the admitted language and checkpoint sets and the resulting estimand composition are filed and hashed into ELIGIBLE_PAIRS.json before B1 opens and cannot change afterwards.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule R15-F4, filed in full and binding

R15-F4 THE APPARATUS-ARTIFACT EXCLUSION IS EXPLORATORY.. The text-quality perturbation arm, the within-transcript instrument-language contrast and the bundled 6 x 6 pipeline matrix RUN IN FULL and are reported DESCRIPTIVELY with the section-I region, but C2 IS DEMOTED TO EXPLORATORY: no artifact claims that an apparatus or instrument-administration-pipeline rival has been EXCLUDED, no equivalence or noninferiority declaration is read off them, and the scope sentence states instead that the claim is about BEHAVIOR AS SCORED by the frozen human-validated rubric of rule G with the F2 and F3 gates passed, the apparatus rivals remaining LIVE AND UNTESTED. hypotheses[H2] keeps its estimand and family as a descriptive second-level read. The study defends ONE claim.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule R16-F2, filed in full and binding

R16-F2 THE ADMISSION TEST REJECTS IN THE UPPER TAIL ONLY. rejected on |t| against a one-sided alternative, so a sufficiently NEGATIVE pole separation admitted a language). R15-F3(iv) IS REPLACED IN FULL: the critical value is ONE studentized SINGLE-STEP MAX-T value, the 0.95 quantile of the bootstrap distribution of the maximum of the SIGNED t_j over all enumerated limbs, and limb j rejects iff t_j > c - equivalently iff its simultaneous one-sided 95% LOWER bound on E*(p,p) exceeds f_diag = 0.18. No absolute value, no lower tail and no two-sided quantity exists anywhere in the admission family; every |t| reading of the gate is VOID. F1's limb (ii) inherits the same one-sided form.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule R16-F3, filed in full and binding

R16-F3 EVERY H1 CHECKPOINT CARRIES ALL THREE LANGUAGES., section I(v) and the section-J power model require six contrasts and at least four positives). R15-F3(v) IS AMENDED: checkpoint c is ADMITTED iff ALL THREE of its languages are admitted, so L = 3 always and its confirmatory table is the complete 3 x 3 subtable with exactly SIX ordered mismatch pairs. THE L = 2 BRANCH IS VOID. R15-F3(vi) stands unchanged - fewer than 3 of the 4 checkpoints admitted means UNTESTABLE, outcome (C). The 4-of-6 consistency rule of B(3), the I(v) composition and the six-pair power model of J therefore apply verbatim at every admitted checkpoint and no allowed L needs its own power run.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule R16-F5, filed in full and binding

R16-F5 THE COMPETENCE SCREEN LEAVES THE TUNING PARTITION.. The target-language competence screen runs on a SEPARATELY HASHED 50-prompt QUALIFICATION sample, disjoint from the 30 tuning scenarios and from the F2 and R16-F1(b) subsamples, enumerated in QUAL_BUDGET.json with its counts, allocation, thresholds and its own FWER 0.05. F3's "run once on tuning data" is VOID, and controls.evaluator_validity's rule that EVERY admission-affecting check is estimated on qualification material governs everywhere.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule H, the S in {8..12} seed-selection machinery

THE SEED SCHEDULE IS COMPLETE AND RUNNABLE FOR EVERY ALLOWED COUNT. The seed list is 20260911 + {0..11}. For EVERY allowed count S in {8, 9, 10, 11, 12} the run tensor and the nuisance allocation are generated by schedule/make_nuisance_schedule.py BEFORE M1 and hashed into NUISANCE_SCHEDULE.json as FIVE INDEXED BLOCKS (and the sixth, qualification block of H2), each satisfying the orthogonality rule above at its own count and each NESTED in the next, so the first eight seeds' allocation is identical in all five: S = 8 is 9 x 5 x 4 x 720 = 129 600 confirmatory transcripts and every further seed adds 9 x 5 x 4 x 90 = 16 200, to 194 400 at S = 12. M2 SELECTION IS DETERMINISTIC AND HAPPENS ONCE: S is the SMALLEST count in that list whose simulated power at delta_alt reaches 0.80 under section J, evaluated on tuning-partition variance components BEFORE any evaluation generation; if none reaches it, S stays 8 and delta_alt is raised exactly as section J registers. No adaptive branch, no unscheduled seed, no post-hoc count: whichever S is selected, its allocation already exists, hashed, from before M1. type_ext.llm_eval.decoding_config lists the twelve seeds and marks the first eight the default block; outcome.compute_budget states the S = 8 budget and the S = 12 ceiling.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule R15-F1, filed in full and binding

R15-F1 THE CONFIRMATORY ENDPOINT IS A BALANCED, MARGIN-FREE INTERACTION CONTRAST AND THE PAIR QUANTITIES ARE SIMPLE EFFECTS.. Write the cell effect as E*(p,d) = a_p + b_d + g_pd, additive persona-language and dialogue-language terms plus an interaction term. (i) Delta_mis(p,d) = E*(p,p) - E*(p,d) = (b_p - b_d) + (g_pp - g_pd) CONTAINS the additive term b_p - b_d: it is a SIMPLE EFFECT AT FIXED p, NOT an interaction contrast, and the earlier claim that the shared neutral arm removes d's main effect from it is VOID - that arm removes d's effect on the SCORE LEVEL, not its effect on the SIZE OF THE PERSONA EFFECT. (ii) THE CONFIRMATORY ESTIMAND Delta_mis_bar IS, FROM R15 ON, THE BALANCED MATCH-MINUS-MISMATCH CONTRAST over the ADMITTED language set A of size L >= 2: Delta_mis_bar = (1/L)*sum_{p in A} E*(p,p) - (1/(L(L-1)))*sum_{p != d in A} E*(p,d). Its weights sum to ZERO along EVERY ROW AND EVERY COLUMN of the L x L table, so both additive terms are ANNIHILATED ALGEBRAICALLY and Delta_mis_bar equals (1/L)*sum g_pp - (1/(L(L-1)))*sum g_pd, a PURE INTERACTION CONTRAST. At L = 3 with the full table retained it is IDENTICALLY the equal-weight mean of the six ordered pair contrasts, so no registered number, power figure or threshold moves; the DEFINITION as that mean is VOID because it is balanced only when the table is complete. (iii) The six ordered Delta_mis(p,d) and the three 2 x 2 pairwise interactions [E*(p,p) - E*(p,d)] - [E*(d,p) - E*(d,d)] are reported with the section-I region as SECONDARY DESCRIPTIVE quantities: no simple effect enters the primary family, gates H1 or carries a claim, and every artifact calls them simple effects. Rule B(3)'s 4-of-6 consistency requirement survives unchanged as a descriptive side-condition that can only make declaration harder.


## ARCHIVED BY R17 (filed in full; binding where it says so) - decision_rule J, the pre-R17 power section with its S in {8..12} branch

J. POWER, AND THE HONEST-POWER STATEMENT, AS AMENDED BY R16-F6, WHICH GOVERNS. delta_alt is the smallest true average mismatch effect, on a 0.01 grid strictly above max(delta_raw, 2*V_hat), at which the frozen Monte Carlo of the COMPLETE conjunction reaches power 0.80. The simulator power/power_sim_r6.py (sha256 42c8064c0892ed37eb4e5364799e21c6708cff260b329cc4d522b111f542391e, 100000 simulations, generator seed 20260911) encodes the boundary max(delta_raw, 2*V_hat) with V_hat drawn from its own sampling distribution AND the 4-of-6 pair consistency side-condition AND the >= 3-of-4 admitted-checkpoint requirement AND the whole rule holding independently for BOTH targeted traits AND pole concordance, and reports the (B2) reversal branch beside the (B1) equivalence branch; its family-wide BH significance layer is VOID and is replaced by the R14 single-step max-T region (section I(iv)).
EVERY PLANNING FIGURE PREVIOUSLY FILED IN THIS SECTION IS VOID AS A CERTIFICATE AND IS NOT RESTATED: the certificate of record is the CONSERVATIVE PRE-B0 RUN R16-F6 REQUIRES (POWER_SIM_R16_PRE.json, filed before B0 at its registered conservative variance components), and power_sim_r6.py and power_sim_r9.py are re-run once more on the TUNING-partition components before any evaluation generation, both under section I AS IT NOW STANDS - the G = S seed-cluster Webb bootstrap, V_hat on G - 1 df, the R14 single-step max-T region - so every power number is computed under the SAME cluster and family the analysis uses; the delta_alt that run returns is the registered one. DETERMINISTIC BRANCH, decided at that moment and never after: if power at that delta_alt is below 0.80 at 8 seeds, S rises to the smallest sufficient value in {8,...,12} under section H; the scenario lever does not exist; if 12 seeds still do not suffice, delta_alt is RAISED to the smallest simulated value reaching 0.80 and the enlarged uninformative band is declared in the abstract. Raising delta_alt is never treated as a gain in sensitivity.
HONEST-POWER STATEMENT, printed in the abstract: the tested boundary is max(delta_raw, 2*V_hat); THIS STUDY REGISTERS ITSELF PROSPECTIVELY AS A BOUNDARY TEST WITH ABOUT 0.15 SENSITIVITY; the descriptive prior expectation of 0.12 has simulated power 0.332 and CANNOT be reached by any lever this design owns - the corpus is fixed at 90 evaluation scenarios and the seed cap of 12 raises power at 0.12 only marginally; a non-declaration anywhere in the 0.09-0.15 UNINFORMATIVE BAND is pre-registered as UNDERPOWERED NON-SUPPORT and is never reported as evidence that persona traits transfer across languages. Whatever the conjunction returns, THE PRIMARY DELIVERABLE IS THE ESTIMATE AND INTERVAL OF Delta_mis_bar for every admitted checkpoint and trait. No condition may be added, removed or re-thresholded once evaluation generation has started.


## ARCHIVED BY R17 (filed in full; binding word for word) - decision_rule F2, the evaluator-validity gate in full

F2. EVALUATOR VALIDITY, ESTIMATED AT THE SCENARIO-CLUSTER LEVEL INSIDE EACH CHECKPOINT-LANGUAGE CELL. The human criterion is collected on the F1 tensor's registered subsample: each of the 12 checkpoint-language cells (4 checkpoints x 3 dialogue languages) contributes ONE transcript per qualification scenario, so a cell carries 60 INDEPENDENT SCENARIO CLUSTERS and NO TRANSCRIPT IS SHARED BETWEEN TWO CELLS' GATES - 720 transcripts, double-annotated blind on the frozen rubric by annotators fluent in the language and blind to persona language, dialogue language, checkpoint and pole, a third adjudicating, none seeing the injected-delta anchors. EVERY THRESHOLD IS ESTIMATED INSIDE ITS OWN CELL by a BCa bootstrap that RESAMPLES SCENARIOS (9999 replications, generator seed 20260911): convergent - the cell's rater-versus-criterion correlation needs a 95% cluster-bootstrap LOWER BOUND above 0.40; pooled across cells, each scenario counted once by averaging within scenario, that lower bound must exceed 0.60; discriminant and anti-stereotype keep their registered thresholds under the same scenario-level bootstrap, paired within scenario. This gate is a validity estimate, not a confirmatory inference, and its scenario resampling is confined to it. Being a scenario-cluster estimate inside one checkpoint-language cell, it pseudoreplicates nothing and licenses exactly the admission decision it is used for.


## ARCHIVED BY R17 (filed in full; binding word for word) - decision_rule F4, the gate-power clause in full

F4. GATE POWER IS DEMONSTRATED, NOT ASSUMED, AND FAILURE HAS A DETERMINISTIC CONSEQUENCE; ITS FIGURES ARE CERTIFIED UNDER R16-F6 AND NOWHERE ELSE. power/power_sim_r9.py extends the frozen simulator with (a) the F1 gate as an explicit stage on the H2 tensor's units and (b) the resulting eligible-pair-count distribution feeding the confirmatory conjunction; it must show at least 0.90 per-pair gate power at the anticipated diagonal effect of 0.60 with the registered sigma_pair, reporting BOTH the conditional power given the frozen eligible set AND the unconditional power including gate failure. Its filed output is power/POWER_SIM_R9.json, produced before B1 under the R16-F6 inputs. IF PER-PAIR GATE POWER IS BELOW 0.90 AT THE REGISTERED Q = 60, THE DETERMINISTIC CONSEQUENCE IS: H1 is UNTESTABLE on the behavior channel and takes outcome (C); THE CORPUS IS NOT ENLARGED, no gate threshold is relaxed and no scenario is reused: it is fixed at 30 + 60 + 90 by construction and no source or budget exists that could enlarge it; outcome.n_runs and QUAL_BUDGET.json budget that tensor and its human annotations separately. It simulates those limbs ON THAT TENSOR with the section-I seed-cluster bootstrap at G = S; no earlier gate-power figure computed under another cluster, family or interval survives.


## ARCHIVED BY R17 (filed in full; binding word for word) - decision_rule I as it stood at R16

I. THE INTERVAL AND THE MULTIPLICITY ALGORITHM ARE UNIQUELY EXECUTABLE. (i) THE RESAMPLING UNIT IS THE SEED, AND ONLY THE SEED. There are G = S = 24 INDEPENDENT seed clusters (R17-F2, which governs); cluster g holds EVERY unit-level matched difference generated at seed g, across all scenarios, cells, settings, engine strata, checkpoints and traits. The 90 evaluation scenarios are FIXED and NEVER resampled (rule E), so no scenario, no (scenario, seed) pair and no transcript is ever a confirmatory resampling unit; every two-way scenario x seed and scenario-only wording is VOID (section N). (ii) RESIDUALS AND WEIGHTS: the residual of a unit is its matched difference minus the fitted value of its (engine stratum x cell x setting) mean under the imposed null; one WEBB 6-POINT weight is drawn PER SEED CLUSTER PER REPLICATION and SHARED GLOBALLY, multiplying every residual of that cluster in every stratum, cell, setting, checkpoint, trait and contrast at once, so dependence across contrasts that share seeds is preserved by construction. Webb's 6-point law is registered because at G = 24 it supports 6^24 sign vectors, so the bootstrap p-value has no discreteness floor; 9999 replications, generator seed 20260911, null imposed, the three engine strata recombined at weight 1/3 inside every replication. (iii) STUDENTIZATION AND V_hat: every statistic is divided by the cluster-robust standard error computed over those same G seed clusters with the finite-cluster correction G/(G-1); the reference law is the bootstrap distribution of that studentized statistic itself and NO t or normal approximation is used anywhere. V_hat is the repeated-condition SD across seed clusters on G - 1 = 23 degrees of freedom and is RECOMPUTED inside every replication from the resampled clusters, so the boundary's sampling variability travels jointly with the estimate. (iv) THE PRIMARY FAMILY IS ENUMERATED AND FROZEN BEFORE B1 IN ELIGIBLE_PAIRS.json AND HOLDS EXACTLY TWO COMPONENTS PER (ADMITTED CHECKPOINT x TARGETED TRAIT) MEMBER - Delta_mis_bar and Delta_mis_bar - 2*V - at most 2 x 4 x 2 = 16 components. R14 REPLACES R13's STEP-DOWN CONSTRUCTION, WHICH IS SUPERSEDED AND VOID, BY THE ONE FIXED-FAMILY PROCEDURE THAT IS ITS OWN INVERSION: a STUDENTIZED SINGLE-STEP MAX-T TEST at FWER 0.05 over the FIXED, PRE-ENUMERATED family of at most 16 components. R13's step-down critical values c_j were read off ACTIVE SETS DETERMINED BY THE OBSERVED ORDERING and were then reused as interval half-widths for every candidate null, which is not a valid inversion - a candidate-null-specific reordering, constrained resampling or closed-test inversion would have been required, and none is registered - so no step-down quantity survives anywhere in this design. THE REGION OF RECORD: ONE critical value c = the 0.95 quantile of the bootstrap distribution of the maximum of |t| over ALL components of the fixed family, computed from the rule I(i)-(iii) seed-cluster Webb bootstrap with the null imposed and c NOT depending on the observed ordering or on which null is being tested; each component's REPORTED INTERVAL is its estimate +- c x its cluster-robust SE, and the 16 intervals TOGETHER are one simultaneous 95% confidence region with asymptotic simultaneous coverage. A null value for a component is REJECTED IF AND ONLY IF IT LIES OUTSIDE THAT COMPONENT'S REPORTED INTERVAL: rejection and non-coverage are THE SAME EVENT for every candidate null at once, so nothing in this design can disagree with anything else. The price is power, not validity, and it is paid deliberately. THE FIXED AND THE RANDOM BOUNDARY ARE INVERTED JOINTLY: the second component is studentized by the cluster-robust SE of the DIFFERENCE Delta_mis_bar_hat - 2*V_hat, read from the joint covariance of (Delta_mis_bar_hat, V_hat) over the same G seed clusters and recomputed inside every replication, so the 2*V boundary is a coordinate of the estimand vector and never a known constant. EVERY DECISION IN THIS DESIGN READS THIS ONE REGION AND NOTHING ELSE: B(1) is lower bound of Delta_mis_bar > 0; B(2) is lower bound of Delta_mis_bar > delta_raw AND lower bound of Delta_mis_bar - 2*V > 0; C(B1) equivalence is the Delta_mis_bar interval lying wholly inside +-delta_raw; C(B2) reversal is its upper bound < -delta_raw. These are statements about ONE interval per member, so at most one can hold, and B(2) implies B(1) because delta_raw > 0. Holm, step-down max-T in either family, and the three-shifted-null maximum-p-value combination are VOID (section N). Benjamini-Hochberg is NOT used in either confirmatory family and no 'BH-adjusted level' or 'BY-style FCR' interval exists anywhere in this design (section N): BH controls a false-discovery rate, not simultaneous coverage, and every rule above reads coverage. (v) THE PAIR-LEVEL FAMILY IS SEPARATE AND GATED: the 6 ordered pair-specific Delta_mis(p,d) of each admitted checkpoint and trait form their OWN fixed family, reported only after the primary family, under the SAME single-step max-T procedure with its own single critical value and its own inverted intervals, computed from the same replications. Condition (3) reads point estimates in that family and can only make declaration harder. Conditions (1) and both limbs of (2), and outcomes (B1) and (B2), read the PRIMARY family's single-step region and no other interval.


## ARCHIVED BY R17 - METHOD_DESIGN.md section 6 outcome table as it stood at R16

| Item | Value |
| --- | --- |
| Primary metric | `Δ̄_mis`, the **balanced match-minus-mismatch interaction contrast** over the admitted **3 × 3** subtable (L = 3 mandatory, **R16-F3**) on the pole-separation cell effect `E*` (**R15-F1/F2**), on the **raw native 1–7 rubric mean with no anchor calibration** (**R16-F4(a)**); the six `Δ_mis(p,d)` are **simple effects**, descriptive |
| Admission | **R15-F3 as amended by R16-F2/F3** — one enumerated ≤ 24-limb family (null `E*(p,p) ≤ f_diag = 0.18`, seed-cluster bootstrap, ONE **one-sided signed** single-step max-T critical value at its own FWER 0.05: admit only when the simultaneous **lower** bound exceeds 0.18), a deterministic **language-level** map, a checkpoint admitted only with **all three** languages, frozen into `QUALIFICATION_GATE.json` / `ELIGIBLE_PAIRS.json` **before B1**; in **no** confirmatory family. Pair-level eligibility and the `|t|` reading are **VOID** |
| Behavior instrument | **R16-F1** — the rubric and its anchors are **ENGLISH in every cell**; rule G's rendering clause is **VOID**. Invariance is a **gating** 12-limb equivalence family on 360 cross-scored qualification transcripts at ±ι = 0.10, one max-\|t\| critical value, powered before B0 |
| Tested boundary / powered alternative | **rule C**: the only thing claimed against is max(δ_raw, 2·V̂), δ_raw = 0.09. **δ_alt = 0.15 is a power statement only** (rule J). The 0.09–0.15 band is pre-registered UNINFORMATIVE; the ≥0.15 claim wording is **VOID** |
| Practical-significance test | **rule B(2)** — in the R14 single-step max-T region the lower bound of `Δ̄_mis` exceeds δ_raw and that of `Δ̄_mis − 2·V` exceeds 0, V̂ recomputed inside every replication |
| Equivalence margin (H3) | ε = 0.075 of the induced shift; ε_raw ≈ 0.045; the R14 single-step max-T TOST region |
| α / power / tails | 0.05 / 0.80 at δ_alt / two-tailed, paired. The certificate of record is the **conservative pre-B0** `POWER_SIM_R16_PRE.json` (**R16-F6**); every earlier planning figure is **VOID as a certificate** |
| Corpus partition | **rule E** — 180 scenarios cut ONCE into TUNING 30 / QUALIFICATION 60 / EVALUATION 90, hashed as `SCENARIO_PARTITION.json` and `RUN_TENSOR.json`, **never enlarged** (rule F4); no other scenario count exists |
| Evaluator-validity gate | **rule F2** — the 720-transcript stratified subsample, 60 independent scenario clusters per cell, every threshold estimated **inside its own cell**; transcript-level Fisher-z standard errors are **VOID**. The competence screen runs on its own hashed 50-prompt qualification sample (**R16-F5**), never on tuning material; every gating sample and its human cost is enumerated in `QUAL_BUDGET.json` (**R16-F4(b)**) |
| Run count and seed schedule | **rule H** — seeds 20260911 + {0…11}, every allowed S ∈ {8…12} hashed into `NUISANCE_SCHEDULE.json` before M1 (129 600 confirmatory transcripts at S = 8), S chosen once at M2. **No adaptive seed count exists** |
| Estimator and interval | **ONE algorithm — rule A**, ending in the balanced R15-F1 contrast over the admitted subtable, the rule I wild cluster bootstrap-t over the G = S seed clusters and the **R14 single-step max-T region**. Every other aggregator is a **sensitivity view that decides nothing** |
| Multiplicity | **ONE invertible procedure — the R14 studentized SINGLE-STEP max-T region at FWER 0.05, family-wide; Holm, step-down max-T and BH are VOID here** (rule I(iv)–(v)): the primary family holds two components per admitted checkpoint × targeted trait (≤ 16), so every rule B and rule C decision reads one region. The five separate FWER-0.05 families and their gating order are fixed in **R16-F4(c)**; BH survives only in the descriptive omnibus grid |
| Randomness varied | sampling seed, item order, option order, persona wording variant, scenario-to-seed assignment; engine balanced but a **fixed stratum** in inference |
| Compute | `pico.outcome.compute_budget`; not restated here |


## ARCHIVED BY R17 - METHOD_DESIGN.md section 10 R16 narrative as it stood at R16

**R16**, in one
line each: **F1** fixes the behavior instrument to **ENGLISH** in every cell, voids rule G's clause rendering
the rubric into the transcript language, and makes instrument-language invariance a **gating** 12-limb
equivalence family (n = 360 cross-scored qualification transcripts, ±ι = 0.10, one max-|t| critical value,
powered before B0); **F2** replaces the R15-F3(iv) `|t|` rejection with a **one-sided signed max-T** — admit
only when the simultaneous lower bound exceeds f_diag = 0.18 — so a negative effect can no longer admit;
**F3** requires **all three languages** at every H1 checkpoint, voiding the L = 2 branch that was
incompatible with B(3)'s 4-of-6 rule, I(v) and J's six-pair power model; **F4** removes estimated anchor
calibration from the confirmatory endpoint (raw native 1–7 equal-weight mean), enumerates every gating
sample and its human cost in `QUAL_BUDGET.json`, and fixes the five FWER-0.05 families and their gating
order; **F5** moves the competence screen onto a separately hashed 50-prompt qualification sample;
**F6** requires a **conservative pre-B0 power certificate** (`POWER_SIM_R16_PRE.json`) and voids every
placeholder figure as a certificate; **F7** fixes the label set and extends `consistency_check.py`, whose
filed PASS is a precondition of B0.


## ARCHIVED BY R17 - EXPERIMENT_PLAN.md section 4 budget table as it stood at R16

| Development and qualification (B0) | ≈110–130 GPU hours, **including the 28 800-transcript rule-H2 qualification tensor and its rater scoring (≈11–20 GPU hours)** |
| **B1b-behavior (confirmatory core)** | **129 600 transcripts** at S = 8; +16 200 per further seed to 194 400 at the S = 12 ceiling (rule H) |
| **Omnibus grid (C), descriptive** | **64 800 transcripts** — 27 cells × 5 settings × 4 checkpoints × 30-scenario subset × 4 seeds |
| **B1b-selfreport** | **583 200 administrations** = 388 800 core (27 216 000 item responses) + 129 600 perturbation + 64 800 omnibus |
| Generation / self-report / rater scoring | ≈40–70 + ≈30–50 + ≈35–60 GPU hours on one A100 80GB (194 400 transcripts plus 51 840 rater-language re-scorings) |
| Scoring, contamination scan, analysis | ≈60 CPU hours, incl. the repeat arm, ablations and the 1.5× failure allowance |
| **Registered total** | **≈240–400 GPU hours evaluation (≈150–250 on a Hopper-class 80GB) + ≈110–130 GPU hours development and qualification + ≈60 CPU hours** |
| Human annotation (`QUAL_BUDGET.json`, **R16-F4(b)**) | **720 transcripts** — the rule-F2 stratified subsample — blind double-annotated with a third adjudicating (1440 primary annotations plus adjudications), **plus 720 cross-scored ratings on the 360 R16-F1 invariance transcripts, 288 anchored ratings and 360 perturbation pairs** |


## ARCHIVED BY R17 - EXPERIMENT_PLAN.md section 8 R15/R16 landing-point narrative

The R15 landing points in this plan (§1's balanced interaction endpoint on pole separation, B0's 30 / 60 / 90
partition and language-level admission gate on the 28 800-transcript qualification tensor, §6's UNTESTABLE
branch with no corpus expansion, §3/§4's pre-hashed nested seed blocks, B1b's seed-cluster bootstrap with the
R14 single-step max-T region, C2 exploratory only) are enumerated once in `METHOD_DESIGN.md` §10 and are not
restated here. **R16** lands as B0's ENGLISH instrument and its gating
invariance family (F1), the one-sided admission test and mandatory L = 3 (F2, F3), the raw-scale endpoint
with every gating sample enumerated in `QUAL_BUDGET.json` and the five FWER-0.05 families fixed (F4), the
competence screen off tuning material (R16-F5), the conservative pre-B0 certificate (F6) and the extended
`consistency_check.py` whose PASS gates B0 (F7). Added cost: that tensor and its rater scoring (≈11–20 GPU
hours, inside the ≈110–130 GPU-hour B0 envelope) and the human annotation of `QUAL_BUDGET.json` — 720 blind
double-annotated qualification transcripts, 720 cross-scored ratings, 288 anchored ratings, 360 perturbation
pairs.


## ARCHIVED BY R17 - EXPERIMENT_PLAN.md section 1 claims as they stood at R16

**Primary claim (C1).** Holding the persona language fixed, making the model **converse** in a different
language costs part of the induced Big Five **pole separation**. The evidence required is the full
conjunction of `decision_rule` rules A, B, C, F, I and **R15-F1/F2/F3 as amended by R16**, on the **balanced
match-minus-mismatch interaction contrast** `Δ̄_mis` over the admitted **3 × 3** subtable (L = 3 mandatory,
R16-F3), per targeted trait, over ≥ 3 of the 4 checkpoints admitted before B1. Its weights sum to zero along
every row and column, so both additive language main effects are annihilated and the six pair quantities are
**simple effects**, descriptive only. The tested boundary is `max(δ_raw, 2·V̂)`; δ_alt = 0.15 is a power
statement only; inference is conditional on the 90 fixed evaluation scenarios (rule E). C1 is an
**intention-to-treat** claim about *assignment*, and it is pole separation, **never a persona-versus-neutral
magnitude** (R15-F2).

**Exploratory read (C2), per R15-F4.** Whether the loss is an artifact of the measuring apparatus — the
within-transcript instrument-language contrast plus the matched text-quality perturbation arm, narrowed by
rule L to an observable instrument-administration-pipeline effect. It carries **no apparatus-exclusion
claim** and enters no confirmatory family. The **behavior** instrument is no longer part of that rival: it is
fixed ENGLISH and its invariance is gated (**R16-F1**).

**What is NOT confirmatory, and still runs in full.** No main effect of D is a confirmatory read (rule L).
The behavior channel's P × D language factorial is the **gated second-level family**; the self-report
P × D × I factorial carries H2; the rater-language factor H2b; the bundled 6×6 matrix is an omnibus pipeline
effect and the resource gradient a descriptive association. All are estimated and reported. **C1 says nothing
about why** — competence, tokenization, pragmatics and translation naturalness stay live, bounded only
partially by the competence screen, the length band and the ensemble, and are disclosed as untested.

**Anti-claims** are registered as H2, H2b and H3 in `pico.hypotheses[]`, each with its own estimand, family
and kill mechanism: the round-trip control narrows and never refutes (the falsifier of record is
`pico.hypothesis.falsifier`), H3 defaults to **UNRESOLVED**, and compliance is handled by the compliance-only
control instruction with per-cell rates reported.

One confirmatory claim, under the cap. No other claim is defended.


## ARCHIVED BY R17 - METHOD_DESIGN.md section 8 preregistration body as it stood at R16

**Decision rule — head only; the executable text is card rule B and is not restated here.** H1 is supported
only if, independently for each targeted trait, all five conditions of rule B hold, in ≥ 3 of the 4
checkpoints admitted before B1 by the **rule R15-F3 qualification-gate family**, each read off the **single
R14 single-step max-T region** of rule I(iv) and no other interval; the boundary is max(δ_raw, 2·V̂) (rule C)
and δ_alt = 0.15 is a power statement only, so the 0.09–0.15 band is pre-registered uninformative and the
estimate and interval are reported regardless (rule J). No condition may be added, removed or re-thresholded
once generation has started. There is **no confirmatory read on any main effect and no round-trip-tie rule**:
the language factorial runs in full as the gated second-level and self-report families, never as support for
H1. H2 is supported, inside its own family, when the instrument-language main effect of the self-report
factorial carries the loss under the same standard while the dialogue-language effect does not; **H3** only
by the single-step max-T TOST region at ε, an intermediate pattern being **H3 unresolved**.

Neither the bundled grid (C) nor the resource-rank slope enters any hypothesis condition. A shortfall on
self-report whose interval covers zero on the rater-free indicators reports H2b as unresolved and demotes the
behavior arm to descriptive; if the behavior channel fails the **qualification-set** rater, construct-validity
(rule G) or admission checks, confirmatory H1 is untestable and there is no endpoint substitution. The 20%
confirmatory-core rater-language subsample is **non-gating**.

**Predicted result** is registered once in `pico.preregistration.predicted_result`; per **rule J** its 0.12
head is descriptive and registers nothing. **rule E**: the 12 development-touched BFI-2 items are excluded
from the confirmatory H2 self-report score (48 items scored) and reported separately.

**Every outcome is informative and each reading is bound in advance to the pattern that selects it**
(enumerated once in `EXPERIMENT_PLAN.md` §6): self-report-only shortfall → H2 as narrowed by rule L;
shortfall vanishing on rater-free indicators → H2b; full transfer → the positive language-general finding;
only the compliance rate moving → instruction following rather than trait expression (ref-031, ref-032).


## ARCHIVED BY R17 - METHOD_DESIGN.md section 7 out-of-scope tail as it stood at R16

**Out of scope** (8 items in the card): closed-weight models; languages outside the six; speech and
multimodal interaction; other domains; any claim that the model *has* a personality; non-Big-Five constructs;
fine-tuning or activation-level induction; safety-critical deployment recommendations. Results are broken
down per directed pair, checkpoint, trait, channel, resource tier, direction and compliance status.


## ARCHIVED BY R17 - METHOD_DESIGN.md section 1 H1 paragraph as it stood at R16

**H1.** A mismatch between the language of the persona instruction and the language the model converses in
reduces the **persona-induced high-versus-low pole separation** in expressed Big Five trait level. The
confirmatory estimand is `Δ̄_mis`, the **balanced match-minus-mismatch interaction contrast** over the admitted
3 × 3 language subtable (**R15-F1**; L = 3 is mandatory under **R16-F3**, so a checkpoint keeps all three
languages or is not admitted): diagonal weight 1/L, off-diagonal −1/(L(L−1)), zero along every row and column, so both additive main effects cancel algebraically and the quantity is a pure P × D interaction. The
six `Δ_mis(p,d)` are **simple effects at fixed p** — they retain the additive dialogue-language term — and are
secondary descriptive quantities. The neutral arm cancels from `E*` and carries **zero weight**, so **no
persona-versus-neutral magnitude is claimed** (**R15-F2**), and **no confirmatory read is taken on any main
effect** (rule L). Behavior is scored by the **fixed ENGLISH rubric on its raw native 1–7 scale** (**R16-F1**,
**R16-F4(a)**). H1 is an **intention-to-treat** claim about *assignment*: every confirmatory number is computed
on the first planned generation as produced, including wrong-language output, with per-cell compliance rates
beside it. The bundled grid (C) is an omnibus pipeline effect and the pretraining-resource gradient
descriptive only; per **rule E** inference is conditional on the 90 fixed evaluation scenarios, and the tested
boundary is `max(δ_raw, 2·V̂)` (rule C).


## ARCHIVED BY R17 - EXPERIMENT_PLAN.md section 7 checklist bullets as they stood at R16

- [ ] **One randomization schedule frozen from one recorded RNG seed and hashed as `NUISANCE_SCHEDULE.json` in
      six indexed blocks — five nested confirmatory blocks for S ∈ {8…12} plus the rule-H2 qualification block**
- [ ] **The rule-H2 qualification tensor (28 800 transcripts at S = 8) generated, scored and hashed, its 720
      blind double-annotated rule-F2 transcripts adjudicated, and the admitted checkpoint set, the
      language-level admission map (`QUALIFICATION_GATE.json`, `ELIGIBLE_PAIRS.json`) and the primary family all
      frozen before B1 opens (rules F, F2, R15-F3 as amended by R16-F2/F3: one-sided admission, all three
      languages per admitted checkpoint)**
- [ ] **`consistency_check.py` exits 0 — including the R16-F7 rules — before the pre-registration is frozen,
      its PASS filed as a precondition of B0, and again before analysis**
- [ ] **The conservative pre-B0 power certificate `POWER_SIM_R16_PRE.json` filed before B0 opens (R16-F6)**
- [ ] **f_diag = 0.18, the rule A estimator ending in the R15-F1 balanced contrast, the rule I seed-cluster
      bootstrap with the R14 single-step max-T region and the boundary `max(δ_raw, 2·V̂)` encoded in the frozen
      analysis code; `power_sim_r6.py` and `power_sim_r9.py` rerun under that region and endpoint, variance
      components re-estimated on TUNING, and S selected once from the pre-hashed nested blocks or δ_alt raised
      — all before any evaluation generation (rules D, F4, J)**


## ARCHIVED BY R17 - decision_rule C, the restated three-outcome rule

The one exhaustive rule lives in pico.hypothesis.falsifier and governs every artifact; this field states NO independent falsifier and NO competing refutation path, and neither does any other field, section or document. Its three outcomes, applied per targeted trait over the admitted checkpoints with the SAME primary family, the SAME section-I(iv) single-step region and the at-least-3-of-4 quantifier: (A) SUPPORTED iff the whole section-B conjunction holds; (B) REFUTED through EITHER limb - (B1) EQUIVALENCE, the region's interval for Delta_mis_bar lies entirely inside +-delta_raw = 0.09; or (B2) REVERSAL, its upper bound lies BELOW -delta_raw. A practically large mismatch effect OPPOSITE to the registered direction therefore REFUTES the directional claim instead of being absorbed into inconclusive; the rule is sign-symmetric. (C) INCONCLUSIVE keeps every remaining case: non-declaration in the 0.09-0.15 uninformative band, every UNTESTABLE branch of sections F and J, and every failure of the behavior channel to be admitted - an unadmitted channel is NEVER reported as outcome (B).


## ARCHIVED BY R17 - definitions[5] three-way language factorial core as it stood at R16

Three-way language factorial core. The three factors that (C) bundles are manipulated SEPARATELY over L3 = {English, Chinese, Swahili} - one high-resource Latin-script, one high-resource non-Latin-script, one low-resource - fully crossed: P x D x I, 27 cells per checkpoint per arm, of which 18 are new (the 3 all-equal cells and the 6 cells with P = s, D = I = t are reused run-for-run from (C)). Matched controls: every rendering passes the same three-forward-translation plus back-translation screen, so translation procedure is constant across factors; generation length is matched by a per-language band calibrated on the tuning partition (identical turn count, mean turn length within +-10%). Estimands, each a within-scenario, within-seed contrast against P = D = I = s: the three main effects, the three two-way interactions and the three-way term, on the unnormalized scale by eq6, decomposed by eq7. NONE is a confirmatory endpoint for H1. THE FACTORIAL IS SPLIT BY OUTCOME CHANNEL: the transcript is produced under P and D and THEN frozen, so instrument language cannot be a factor for it. The BEHAVIOR channel is a P x D factorial only (9 cells over L3), scored by the FIXED ENGLISH rubric (R16-F1); the SELF-REPORT channel crosses I on those same frozen transcripts (paired within transcript, transcript-level clustering) and carries H2; a pre-registered RATER-LANGUAGE factor on a 20% stratified subsample carries H2b and gates nothing. The 27-cell P x D x I read is withdrawn for the behavior channel. The authoritative rule is preregistration.decision_rule: no main effect is a confirmatory endpoint, THE LANGUAGE FACTORIAL RUNS IN FULL as the gated second-level and self-report families (section L), and admission is language-level under R15-F3 as amended by R16-F2 and R16-F3.


## ARCHIVED BY R17 - decision_rule pointer paragraph 3

R15-F2 THE CLAIM IS POLE SEPARATION, NEVER A PERSONA-VERSUS-NEUTRAL MAGNITUDE. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule R15-F2' AND BINDING WORD FOR WORD: the neutral arm carries ZERO weight in every confirmatory quantity, no persona-versus-neutral magnitude is an endpoint anywhere, and every sentence reading one is VOID.


## ARCHIVED BY R17 - decision_rule pointer paragraph 4

R15-F3 THE PRE-B1 QUALIFICATION GATE IS ITS OWN FULLY ENUMERATED FAMILY WITH A DETERMINISTIC CONSEQUENCE. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule R15-F3' AND BINDING WORD FOR WORD: every gate of section F is enumerated before B1, runs only on the qualification partition, carries its own FWER-0.05 correction inside its own family, and its failure has a deterministic, preregistered consequence with no discretion. AS AMENDED BY R16-F1, R16-F5 AND R17-F2, WHICH GOVERN.


## ARCHIVED BY R17 - decision_rule pointer paragraph 5

R15-F4 THE APPARATUS-ARTIFACT EXCLUSION IS EXPLORATORY. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule R15-F4' AND BINDING WORD FOR WORD: the text-quality perturbation arm and every apparatus rival are EXPLORATORY, gate nothing and falsify nothing; no confirmatory sentence may claim an apparatus rival has been excluded.


## ARCHIVED BY R17 - decision_rule pointer paragraph 22

R16-F2 THE ADMISSION TEST REJECTS IN THE UPPER TAIL ONLY. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule R16-F2' AND BINDING WORD FOR WORD: admission rejects iff the ONE-SIDED signed max-T statistic exceeds its critical value, equivalently iff the simultaneous LOWER bound exceeds f_diag = 0.18; the |t| reading is VOID.


## ARCHIVED BY R17 - decision_rule pointer paragraph 23

R16-F3 EVERY H1 CHECKPOINT CARRIES ALL THREE LANGUAGES. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule R16-F3' AND BINDING WORD FOR WORD: L = 3 is mandatory in every admitted checkpoint and the L = 2 admission branch is VOID.


## ARCHIVED BY R17 - decision_rule pointer paragraph 25

R16-F5 THE COMPETENCE SCREEN LEAVES THE TUNING PARTITION. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule R16-F5' AND BINDING WORD FOR WORD: the target-language competence screen runs on its own hashed 50-prompt QUALIFICATION sample and never on tuning material.


## ARCHIVED BY R17 - decision_rule F2 pointer

F2. EVALUATOR VALIDITY, ESTIMATED AT THE SCENARIO-CLUSTER LEVEL INSIDE EACH CHECKPOINT-LANGUAGE CELL. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule F2' AND BINDING WORD FOR WORD. Operative content: the human criterion is 720 transcripts - one per qualification scenario in each of the 12 checkpoint-language cells, so a cell carries 60 INDEPENDENT SCENARIO CLUSTERS and no transcript is shared between two cells' gates - double-annotated blind on the frozen rubric with a third adjudicating; EVERY THRESHOLD IS ESTIMATED INSIDE ITS OWN CELL by a BCa bootstrap RESAMPLING SCENARIOS (9999 replications, generator seed 20260911), the cell rater-versus-criterion correlation needing a 95% lower bound above 0.40 and the pooled bound above 0.60, with discriminant and anti-stereotype limbs at their registered thresholds; this is a validity estimate, not a confirmatory inference, and its scenario resampling is confined to it.


## ARCHIVED BY R17 - decision_rule F4 pointer

F4. GATE POWER IS DEMONSTRATED, NOT ASSUMED, AND FAILURE HAS A DETERMINISTIC CONSEQUENCE. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule F4' AND BINDING WORD FOR WORD; ITS FIGURES ARE CERTIFIED UNDER R16-F6 AND R17-F2 AND NOWHERE ELSE. Operative content: power/power_sim_r9.py adds the F1 gate as an explicit stage on the H2 tensor and feeds the eligible-pair-count distribution into the confirmatory conjunction, reporting BOTH conditional and unconditional power and filing power/POWER_SIM_R9.json before B1; it must show at least 0.90 per-pair gate power at the anticipated diagonal effect of 0.60, simulated on that tensor with the section-I seed-cluster bootstrap AT G = 24 (R17-F2), and no earlier figure under another cluster, family or interval survives. IF GATE POWER IS BELOW 0.90 AT THE REGISTERED Q = 60: H1 is UNTESTABLE on the behavior channel and takes outcome (C); THE CORPUS IS NOT ENLARGED, no threshold is relaxed and no scenario is reused.


## ARCHIVED BY R17 - decision_rule J pointer

J. POWER, AS AMENDED BY R16-F6 AND R17-F2 AND R17-F4(b), WHICH GOVERN. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule J'. Operative content, binding word for word: delta_alt is the smallest true average mismatch effect, on a 0.01 grid strictly above max(delta_raw, 2*V_hat), at which the frozen Monte Carlo of the COMPLETE conjunction reaches power 0.80; the simulator power/power_sim_r6.py (sha256 42c8064c0892ed37eb4e5364799e21c6708cff260b329cc4d522b111f542391e, 100000 simulations, generator seed 20260911) encodes the boundary max(delta_raw, 2*V_hat) with V_hat drawn from its own sampling distribution, the 4-of-6 pair-consistency side-condition, the >= 3-of-4 admitted-checkpoint requirement, BOTH targeted traits and pole concordance, and its BH layer is VOID and replaced by the section I(iv) single-step max-T region. EVERY PLANNING FIGURE PREVIOUSLY FILED HERE IS VOID AS A CERTIFICATE: the certificate of record is the conservative pre-B0 run of R16-F6 (POWER_SIM_R16_PRE.json), re-run on TUNING-partition components before any evaluation generation under section I as it now stands - the G = 24 seed-cluster Webb bootstrap of R17-F2, V_hat on 23 df, the single-step max-T region - and the delta_alt it returns is the registered one. THE S-ESCALATION BRANCH IS VOID: S = 24 is fixed by R17-F2 and no seed count, scenario count or threshold may move after any evaluation datum is seen; if power at delta_alt is below 0.80 at S = 24, delta_alt is RAISED, before B0, to the smallest simulated value reaching 0.80 and the enlarged uninformative band is declared in the abstract, which is never treated as a gain in sensitivity. HONEST-POWER STATEMENT, printed in the abstract: the tested boundary is max(delta_raw, 2*V_hat); this study registers itself prospectively as a BOUNDARY TEST with about 0.15 sensitivity; the descriptive prior expectation of 0.12 is out of reach of any lever this design owns; a non-declaration anywhere in the 0.09-0.15 UNINFORMATIVE BAND is pre-registered as UNDERPOWERED NON-SUPPORT and is never reported as evidence that persona traits transfer across languages; and EVERY power figure is CONDITIONAL ON EVERY ADMISSION GATE PASSING (R17-F4(b)). Whatever the conjunction returns, THE PRIMARY DELIVERABLE IS THE ESTIMATE AND INTERVAL OF Delta_mis_bar for every admitted checkpoint and trait.


## ARCHIVED BY R17 - decision_rule R15-F1 pointer

R15-F1 THE CONFIRMATORY ENDPOINT IS A BALANCED, MARGIN-FREE INTERACTION CONTRAST AND THE PAIR QUANTITIES ARE SIMPLE EFFECTS. FILED IN FULL IN DESIGN_REVISION_HISTORY.md UNDER 'decision_rule R15-F1' AND BINDING WORD FOR WORD. Its operative content: with E*(p,d) = a_p + b_d + g_pd, the estimand from R15 on is the BALANCED match-minus-mismatch contrast over the admitted set A of size L >= 2, Delta_mis_bar = (1/L)*sum_{p in A} E*(p,p) - (1/(L(L-1)))*sum_{p != d in A} E*(p,d), whose weights sum to zero along every row and column, so both additive terms are annihilated algebraically and it is a PURE INTERACTION CONTRAST; the six ordered Delta_mis(p,d) and the three 2 x 2 pairwise interactions are SECONDARY DESCRIPTIVE simple effects that gate nothing; the reading of Delta_mis(p,d) as an interaction contrast, and the definition of Delta_mis_bar as the mean of the six ordered pair contrasts, are VOID.


## ARCHIVED BY R17 - decision_rule E third paragraph

Scenarios are a FROZEN PURPOSIVE CORPUS and are NEVER RESAMPLED in confirmatory inference: every confirmatory interval is CONDITIONAL ON THE 90 FIXED EVALUATION SCENARIOS, the only resampled confirmatory cluster is the seed cluster of section I, and scenario-level heterogeneity is reported descriptively. The QUALIFICATION gates of sections F1 and F2 are the exception and resample scenarios explicitly, because there the scenario IS the sampling unit.
ITEM QUARANTINE: the 12 development-touched BFI-2 items are EXCLUDED from the confirmatory H2 self-report score (48 items scored) and reported separately as a labelled sensitivity; every artifact states the same disjoint split and the scoring code reads the frozen 48-item list.


## ARCHIVED BY R17 - decision_rule A engine paragraph

The three forward-translation engines (Google Cloud Translation v3, Microsoft Azure AI Translator v3.0, NLLB-200-3.3B at the pinned revision) are a FROZEN, EXHAUSTIVE, FIXED SET, never resampled; resampling happens WITHIN a stratum. ENGINE INVARIANCE IS A REPORTED CHECK, NOT AN INFERENCE: the three within-stratum estimates of Delta_mis_bar must be SIGN-CONCORDANT, else the headline claim is narrowed in the title, abstract and discussion to name the discordant engine.


## ARCHIVED BY R17 - controls.evaluator_validity as it stood at R16

The behavior arm is scored by an automatic rater whose validity is established before use, not assumed. THE ADMISSION-AFFECTING FORM OF EVERY CHECK NAMED HERE IS decision_rule F2, F3 AND F4 AS AMENDED BY R16-F1, R16-F4 AND R16-F5, WHICH GOVERN. Agreement is established on the SAME 720 blind double-annotated qualification transcripts that carry the rule-F2 criterion gate (one per qualification scenario in each of the 12 checkpoint-language cells, a third annotator adjudicating), so no separate annotation sample exists: the rater must reach Krippendorff alpha >= 0.7 against the adjudicated labels in EVERY language and pass the registered cross-language agreement equivalence test, or the behavior arm is descriptive only. Independently, four rule-based rater-free indicators (escalation rate, refusal rate, mean turn length, apology-token rate) reproduce the primary comparison with no scoring model; trait-specific configural and metric invariance is established per language and a trait failing it is descriptive for that trait. Self-report validity is screened by per-trait per-language internal consistency, measurement invariance with DIF flags and BFI-2/TIPI convergence, and rule E additionally excludes the 12 development-touched BFI-2 items from the confirmatory H2 score. AGREEMENT IS NEVER READ AS EQUAL CALIBRATION OR EQUAL SENSITIVITY. Rule F: every admission-affecting check runs on the 60-scenario QUALIFICATION partition, disjoint from tuning and from the confirmatory core, the admitted checkpoint set freezing before B1; the 20% rater-language subsample of the core is a NON-GATING sensitivity. Two pre-B1 qualifications per language survive: MULTILINGUAL ANCHOR PAIRS, whose recovered slope and intercept are DESCRIPTIVE DIAGNOSTICS ONLY because R16-F4(a) removes estimated calibration from the endpoint, and PERTURBATION PAIRS, whose equal-detection-sensitivity equivalence test remains GATING at its registered margin. Rater CONSTRUCT validity - convergent, discriminant and anti-stereotype criterion validity, estimated INSIDE each checkpoint-language cell on its 60 independent scenario clusters - is a powered pre-B1 admission gate. The agreement, calibration and sensitivity checks here do not by themselves establish construct validity and are never read as doing so.


## ARCHIVED BY R17 - preregistration.predicted_result as it stood at R16

0.09 IS THE TESTED BOUNDARY, NOT A PREDICTION. rule C fixes the confirmatory claim sentence: Delta_mis_bar exceeds max(delta_raw = 0.09, 2*V_hat). delta_alt = 0.15 is only the alternative at which the complete conjunction reaches 0.80 power (decision_rule J as amended by R16-F6, whose conservative pre-B0 certificate POWER_SIM_R16_PRE.json governs and is not restated here); it is never claimed as the tested effect, and the abstract states both the boundary and the sensitivity limit. The earlier point prediction of about 0.12 is DEMOTED to a descriptive prior expectation with NO registered role: it sizes nothing, it gates nothing, and no declaration or non-declaration is interpreted against it. Directional expectations retained, ALL DESCRIPTIVE: Delta_mis_bar positive in every admitted checkpoint; at least 5 of the 6 ordered pair contrasts positive, largest for a Swahili dialogue language and smallest between English and Chinese; the self-report instrument-language main effect about 0.04 and the pole contrast about 0.02, i.e. not detectable at the registered run count; interrater agreement above 0.7 in every language. Every pre-R5 prediction stated on the deleted main effects or on the retired 0.25 shortfall scale is VOID. Whatever the conjunction returns, the primary deliverable is the ESTIMATE AND INTERVAL of Delta_mis_bar for every admitted checkpoint and trait, and a non-declaration in the 0.09-0.15 band is reported as UNINFORMATIVE (outcome (C) of the R7 falsifier of record), never as evidence that persona traits transfer across languages.


## ARCHIVED BY R17 - hypothesis.anti_claim as it stood at R16

(H2, AS NARROWED BY rule L) The observed effect-retention shortfall is an OBSERVABLE INSTRUMENT-ADMINISTRATION-PIPELINE effect: administering the inventory in a different language changes item semantics, tokenization, comprehension, option interpretation and the stochastic response process together, and the self-report shortfall tracks that pipeline rather than the behavior channel. No passive response-scale artifact and no claim about an unchanged underlying induced trait state is identified by this design; both readings are VOID wherever they appear. (H2b) The observed effect-retention shortfall on the behavior arm is a rater artifact: the scoring pipeline that labels customer-support transcripts is less sensitive in lower-resource target languages, so measured trait expression falls without any change in the model's behavior. (H3) Persona induction is fully language-general: the effect-retention ratio equals one within measurement error for every directed language pair, and all apparent variation is within-condition measurement variability of repeated identical measurements. R3: H2 is now tested by its own estimand, the instrument-language main effect of the factorial core, rather than only by the channel contrast; and H3, full language-general transfer, is declared only by simultaneous equivalence tests at the registered margin, never by failure to reject a shortfall.


## ARCHIVED BY R17 - decision_rule L as it stood at R16

L. THE LANGUAGE FACTORIAL, H2, H2b AND H3. THE LANGUAGE FACTORIAL RUNS IN FULL as the gated second-level design: the behavior-channel P x D factorial that produces the confirmatory mismatch contrasts of section A, the self-report P x D x I factorial carrying H2, and the rater-language factor carrying H2b. Second-level status means only that the confirmatory family is Delta_mis_bar as redefined by R15-F1 AND NOTHING ELSE, C2 and H2 being EXPLORATORY by R15-F4 - no confirmatory read on any main effect, no round-trip-tie rule, no main-effect family - and never that a factor is dropped, collapsed or left ungenerated.
H2 IS AN OBSERVABLE INSTRUMENT-ADMINISTRATION-PIPELINE EFFECT, not a latent-state claim: changing the instrument language changes item semantics, tokenization, comprehension, option interpretation and the stochastic response process together, so the factorial cannot isolate a passive response-scale shift or certify an unchanged latent trait state. H2 is therefore the WITHIN-TRANSCRIPT instrument-language contrast on the self-report channel, on frozen transcripts, with the matched text-quality perturbation arm at fixed final language bounding one translation-quality component; it is supported when that contrast carries the loss under the same standard within its own family while the behavior-channel mismatch contrasts do not, AND the perturbation arm does not by itself reproduce the contrast. H2b is a measurement control on identical frozen transcripts and declares nothing about behavior.
H3 IS NOT A POWERED CONFIRMATORY CLAIM. Per-pair TOSTs at epsilon_raw ~ 0.045 inside the R14 single-step max-T region are reported descriptively with their simulated power stated beside them; full language-generality is never declared; failure to reject a shortfall is never read as full transfer; the default reading of an inconclusive pattern is H3 UNRESOLVED. The omnibus grid and the resource-rank slope enter no hypothesis condition.


## ARCHIVED BY R17 - hypotheses[0].statement as it stood at R16

CONFIRMATORY H1 - STATED ONCE IN pico.hypothesis.statement, WHICH GOVERNS AND IS NOT DUPLICATED HERE. In one line: holding the persona language fixed, making the model converse in a different language than the persona instruction was written in reduces the persona-induced trait shift, measured as the pair-specific mismatch contrast Delta_mis(p,d) = E*(p,p) - E*(p,d) on the pole-aligned behavior-channel trait score and averaged over the pre-B1 eligible ordered pairs as Delta_mis_bar. The complete rule, the tested boundary max(delta_raw, 2*V_hat), the estimator and the three-outcome falsifier are decision_rule sections A-C and pico.hypothesis.falsifier.


## ARCHIVED BY R17 - hypotheses[2].statement as it stood at R16

The observed shortfall on the behavior arm is a rater artifact: the scoring pipeline is less sensitive in lower-resource target languages, so measured trait expression falls without any change in the model's behavior. R4 gives H2b its own measurement factor: a pre-registered RATER-LANGUAGE factor applied POST HOC to IDENTICAL frozen transcripts - a registered 20% stratified subsample of the confirmatory core scored by the rater pipeline in each of the three L3 languages - so any difference is attributable to the scoring pipeline alone and cannot be a behavior effect. It is a measurement control, enters no confirmatory family, and its result qualifies the behavior-channel reading.


## ARCHIVED BY R17 - EXPERIMENT_PLAN.md B0 partition and gate paragraphs as they stood at R16

**Partition first (rule E).** The 180 support scenarios are cut ONCE — before any prompt is written — into
**TUNING 30 / QUALIFICATION 60 / EVALUATION 90**, disjoint, frozen and hashed as `SCENARIO_PARTITION.json`.
Every development-time choice — prompt and persona wording, parsing, scorer and rubric configuration,
translation settings, the dispersion re-estimate, δ_raw, f_diag and the variance components — runs on the
TUNING 30 only. The QUALIFICATION 60 are read **once**, before B1, for the gates below; the EVALUATION 90 stay
sealed until B1. No design choice is ever fitted to a qualification scenario.

**Rule F — the gates and their freeze.** Every admission-affecting check runs on the QUALIFICATION partition
at the same 4 checkpoints, 5 settings and three engines, hashed into the pre-registration at M1 before B1
opens; the admitted checkpoint set, the admitted language set and the primary family freeze at that moment,
and no admission decision ever reads an evaluation-split result. The checks themselves are executable in card
rules F, F2, F3, F4 and G and are **not restated here**: instrument rendering, back-translation, invariance
and convergence per language; automatic-rater agreement (α ≥ 0.7 per language, cross-language equivalent,
double-annotated with adjudication); the **R16-F1 instrument-language invariance family** — the rubric is
ENGLISH in every cell and 360 cross-scored qualification transcripts must show equivalence at ±ι = 0.10 on
one max-|t| critical value, **gating**; the **R16-F5** competence screen on its own hashed 50-prompt
qualification sample; the dispersion re-estimate; the per-language length band (±10% at identical turn
count); the rule G freeze of `RUBRIC_BEHAVIOR.md`; the hashed B3 ablation subset; and the re-estimated
variance components. **Anchor calibration is DESCRIPTIVE ONLY (R16-F4(a)): every confirmatory contrast is
formed on the raw native 1–7 equal-weight rubric mean.** The **R16-F6** conservative certificate
`POWER_SIM_R16_PRE.json` and the **R16-F7** `consistency_check.py` PASS are filed *before* B0 opens.


## ARCHIVED BY R17 - EXPERIMENT_PLAN.md section 5 multiplicity risk row as it stood at R16

| Multiplicity | **ONE invertible procedure — the R14 studentized single-step max-T region at FWER 0.05**, applied once, family-wide, over the primary family's two components per admitted checkpoint × targeted trait (≤ 16), a null rejected **iff** it falls outside that component's reported interval. The five families and their gating order are fixed in **R16-F4(c)**; Holm and BH are void in every confirmatory family |


## FILED IN FULL - R16-F6 and R16-F7, binding word for word as amended by R17

R16-F6 A CONSERVATIVE POWER CERTIFICATE EXISTS BEFORE B0.. Before any B0 GPU hour is spent, power_sim_r6.py and power_sim_r9.py are re-run under section I as it now stands with CONSERVATIVE PRE-REGISTERED VARIANCE INPUTS - sigma_pair 0.065, rho_pair 0.25, tau_pair 0.030, tau_ckpt 0.030, sigma_pole 0.078, V 0.065 - and the resulting single-step max-T certificate is filed as POWER_SIM_R16_PRE.json. planned_seeds' eight-seed power assertion is VOID until that file exists, and the deterministic escalation branch of section J reads S from it. The tuning-partition re-run of J still fixes the registered delta_alt, and any certificate obtained under family-wide BH or any cluster other than the section-I seed cluster stays VOID.

R16-F7 ENFORCEMENT. Every artifact carries the label set R12 AS AMENDED BY R15 AND R16 and no other. consistency_check.py is EXTENDED to exit BLOCK on (i) a label set stopping short of R16, (ii) any surviving rubric-rendered-into-the-transcript-language, anchor-calibrated-endpoint, |t| admission or L = 2 reading outside an explicit void context, (iii) any admission-affecting check placed on tuning material, and (iv) any power, delta_alt or seed-count assertion not read from POWER_SIM_R16_PRE.json or the section-J tuning re-run. A consistency_check.py PASS receipt is a precondition of B0.


## ARCHIVED BY R17-F2(b) calibration pass (restated elsewhere, no force) - METHOD_DESIGN §10 the R16 one-line-each narrative

**R16**, in one line each: **F1** fixes the behavior instrument to **ENGLISH** in every cell and makes
instrument-language invariance a gating 12-limb equivalence family; **F2** makes admission a **one-sided
signed max-T** test; **F3** requires **all three languages** at every H1 checkpoint; **F4** removes estimated
anchor calibration from the confirmatory endpoint and enumerates every gating sample in `QUAL_BUDGET.json`;
**F5** moves the competence screen onto its own hashed 50-prompt qualification sample; **F6** requires a
conservative pre-B0 power certificate; **F7** fixes the label set and extends `consistency_check.py`. The
full R16 narrative is archived in `DESIGN_REVISION_HISTORY.md`.


## FILED IN FULL BY THE R17-F2(b) CALIBRATION PASS (void with the same force) - SECTION N VOID LIST, ITEMS (8) TO (16)

(8) Placing the section-F gate tests inside the primary BH family. (9) Any scorer-tuning/scorer-qualification quarantine narrower than section E's three-way partition. (10) 'Response-scale artifact' and 'the underlying induced trait state is unchanged' readings of H2. (11) Every 'scenario-clustered', 'scenario-resampled' or two-way scenario x seed wording, as an inferential statement about confirmatory intervals. (12) Any claim that the 0.12 prior expectation is a powered target. (13) Any statement that engine and wording are 'balanced-completely crossed with seed'. (14) Any dropping of individual checkpoint x language cells from condition (1) after B1, or removal of a language while leaving the six-pair average in place. (15) The (scenario, seed) resampling unit and any cluster count other than G = S seed clusters (section I(i)). (16) Any reading in which the admission gate is estimated on the 720-transcript F2 annotation sample, or in which the H2 qualification tensor is unbudgeted or shared with the confirmatory core.


## ARCHIVED BY R18 (filed in full; binding word for word except where R18-F1 supersedes it) - decision_rule R17-F2(b)-(d) as it stood at R17

(b) A CALIBRATION CERTIFICATE IS A BLOCKING PRECONDITION OF B0, filed as power/CLUSTER_CALIB_R17.json by power/cluster_calib_r17.py, hashed, and produced from tuning-partition variance components only: under the imposed null it simulates THE COMPLETE max-T VECTOR OVER THE FULL SIXTEEN-COMPONENT PRIMARY FAMILY WITH V_hat RE-ESTIMATED IN EVERY REPLICATION, at 10 000 simulation draws each carrying the registered 9 999 Webb replications, and files (i) the ATTAINED NULL FWER with its Monte-Carlo s.e. and one-sided 95% UPPER bound, which must not exceed 0.05, and (ii) the SIMULTANEOUS COVERAGE of the inverted region with its one-sided 95% LOWER bound, which must reach 0.95, both at G = 24 and both under the registered heterogeneity of section J. (c) IF EITHER BOUND FAILS, THE STUDY IS NOT RUN AS DESIGNED and is re-preregistered; no alpha adjustment, no cluster-count increase and no procedure substitution may be made after seeing an evaluation datum. (d) NO ASYMPTOTIC JUSTIFICATION IS CLAIMED ANYWHERE: rule I(iv)'s asymptotic-coverage sentence is VOID and is replaced by the certificate of (b), and every artifact naming coverage names CLUSTER_CALIB_R17.json as its source.




## ARCHIVED BY R18 (filed in full; binding word for word) - decision_rule R17-F3 as it stood at R17

R17-F3 ONE RUN TENSOR, ONE FAMILY REGISTRY, AND A DETERMINISTIC CHECK THAT THE DOCUMENTS MATCH THEM (answering fatal 3: materially incompatible execution specifications). (a) TWO MACHINE-READABLE FILES ARE THE SINGLE SOURCE OF EVERY COUNT, SPLIT, GATE AND FAMILY IN THIS DESIGN: RUN_TENSOR.json, generated by schedule/make_run_tensor.py, enumerating every block, its partition, its scenario count, its seed count, its cells, its settings and its transcript total; and FAMILY_REGISTRY.json, an EXHAUSTIVE ORDERED registry of the FWER-controlled families with each family's members, its alpha, its gating predecessor and its resampling unit. Both are hashed into PREREG_OF_RECORD before block B0 and no artifact may state a count, a split, a gate or a family that is not read from them. (b) THE THREE CONTRADICTIONS ARE RESOLVED AT SOURCE, NOT POINTED AWAY FROM: the competence screen runs ONLY on its own hashed 50-prompt QUALIFICATION sample (R16-F5) and METHOD_DESIGN's development-split sentence is DELETED; THE CONFIRMATORY CORE IS THE ADMITTED 3 x 3 SUBTABLE ON THE 90 EVALUATION SCENARIOS AT S = 24, while the 6 x 6 directed grid is the DESCRIPTIVE omnibus block (C) at a 30-scenario subset and 4 seeds and is never confirmatory - every sentence placing the 6 x 6 grid, the 30-scenario subset or a 4-seed count inside a confirmatory quantity is VOID; and THE FWER-CONTROLLED FAMILY INVENTORY HAS EXACTLY SIX MEMBERS in the order registered in FAMILY_REGISTRY.json - primary, pair-level, the gated P x D language factorial, the self-report H2 family, the pole-heterogeneity family and the R16-F1 instrument-language invariance family - and every 'five families' wording, in R16-F4(c), in outcome.multiplicity_correction and anywhere else, is VOID. (c) refine-logs/consistency_check.py IS EXTENDED TO VERIFY THE DOCUMENTS AGAINST THE TWO FILES - every split, every scenario and seed count, every gate and every family membership - and to BLOCK on a label set stopping short of R17, on any trait-retention reading of Delta_mis_bar, on any surviving five-family wording and on any eight-cluster or seven-degrees-of-freedom quantity. ITS PASS RECEIPT IS A BLOCKING PRECONDITION OF B0, filed beside the R16-F7 receipt.




## ARCHIVED BY R18 (filed in full; binding word for word) - decision_rule R16-F4 as it stood at R17

R16-F4 ESTIMATED CALIBRATION LEAVES THE CONFIRMATORY ENDPOINT, AND EVERY GATING SAMPLE IS ENUMERATED AND BUDGETED.. (a) The confirmatory contrast is formed on the RAW NATIVE 1-7 EQUAL-WEIGHT RUBRIC MEAN. The anchor slope and intercept, the cross-language slope-equivalence margin and every other estimated affine re-calibration are REMOVED from the endpoint and survive as DESCRIPTIVE cell diagnostics; F3's clause forming every confirmatory contrast on the anchor-calibrated rater scale is VOID, and with it the 0.036 attenuation argument. NO ESTIMATED COEFFICIENT ENTERS ANY CONFIRMATORY STATISTIC, so there is no calibration uncertainty left to propagate. (b) THE PRE-B1 GATING BUDGET IS ENUMERATED ONCE, filed and hashed as QUAL_BUDGET.json before B0 and restated in outcome.n_runs: 86 400 qualification transcripts (section-H2 tensor at S = 24, R17-F2, the 28 800 figure being VOID); 720 blind double-annotated F2 transcripts; the 360 cross-scored transcripts of R16-F1(b), carrying 720 ratings; 288 anchored ratings over 12 anchor sets of 24, DESCRIPTIVE under (a); 360 perturbation pairs, 30 per cell, for the descriptive apparatus arm of R15-F4; and the 50-prompt competence sample of 


## ARCHIVED BY R18 (filed in full; binding word for word) - decision_rule R17-F4 as it stood at R17

R17-F4 TWO REPORTING CLAIMS ARE WITHDRAWN (answering the R16 design-kill's partially-controlled findings 3 and 7). (a) comparison.dummy_baseline's EXACT-ISOLATION claim is DELETED AND VOID: forward-plus-back translation in the source language is not treatment-matched to target-language interaction with one-way rendering, so the round-trip arm is an EXPLORATORY sensitivity that isolates nothing (R15-F4). A SECOND EXPLORATORY CONTROL IS REGISTERED: original versus one-way-rendered input at THE SAME FINAL LANGUAGE, run on the qualification partition, reported with its interval and gating nothing. THE STUDY STATES PLAINLY, IN comparison.baselines and in METHOD_DESIGN section 0, THAT NO SAME-SLOT LITERATURE COMPARATOR EXISTS and that the listed baselines are pipeline components. (b) outcome.power_target AND EVERY POWER SENTENCE ARE RELABELLED: the filed power figure is CONDITIONAL ON EVERY ADMISSION GATE PASSING - the diagonal admission gate, the competence screen, the perturbation-sensitivity gate, every rule F2 validity gate in all three languages and the R16-F1 invariance family - whose correlated conjunction is NOT simulated end to end; no artifact may state an unconditional power for the complete rule.




## ARCHIVED BY R18 (filed in full; binding word for word) - decision_rule R16-F1 to R16-F3 as they stood at R17

R16-F1 THE BEHAVIOR-MEASUREMENT LANGUAGE IS HELD FIXED AND ITS INVARIANCE IS A POWERED GATING TEST.. (a) Every confirmatory behavior score is produced with the rubric, its six facet items and its 1/4/7 anchors in ONE FIXED INSTRUMENT LANGUAGE, ENGLISH, applied unchanged to a transcript left in its own dialogue language, by raters certified bilingual in English and that language; the transcript is never translated. Rule G's clause rendering the rubric into the transcript language is VOID. (b) The instrument language is FULLY CROSSED ON IDENTICAL TRANSCRIPTS in the qualification partition: a registered stratified subsample of n_inv = 360 qualification transcripts - 30 per checkpoint-language cell, the first 30 qualification scenarios of the F2 subsample - is scored TWICE, once under the English instrument and once under the transcript-language rendering, by disjoint certified raters, blind exactly as in F2. (c) INVARIANCE IS CONFIRMATORY AND GATING, never descriptive: the family is ONE limb per (checkpoint, language) = 12 limbs, statistic the within-transcript mean instrument-language difference on the native 1-7 scale, resampled by the I(i)-(iii) seed-cluster bootstrap at its own FWER 0.05, read as an equivalence test against a TWO-SIDED single-step max-|t| critical value: invariance HOLDS for a cell iff the simultaneous 95% interval of that difference lies inside +- iota = 0.10 native points. A cell whose invariance fails is NOT ADMITTED under F3. (d) power/power_sim_r16.py certifies at least 0.80 power for that 12-limb equivalence family at a true difference of 0 under the conservative variance inputs of R16-F6 and files POWER_SIM_R18.json BEFORE B0; until it does, n_inv rises under the section-H2 schedule. (e) The H2b 20% rater-language subsample of the confirmatory core stays NON-GATING and descriptive and substitutes for nothing in (b)-(d).




## ARCHIVED BY R19 (filed in full; binding word for word) - pico.controls.evaluator_validity as it stood at R18

The behavior arm is scored by an automatic rater whose validity is established before use, not assumed. THE ADMISSION-AFFECTING FORM OF EVERY CHECK NAMED HERE IS decision_rule F2, F3 AND F4 AS AMENDED BY R16-F1, R16-F4, R16-F5 AND R17-F2, WHICH GOVERN. Agreement is established on the SAME 720 blind double-annotated qualification transcripts that carry the rule-F2 criterion gate, so no separate annotation sample exists: the rater must reach Krippendorff alpha >= 0.7 against the adjudicated labels in EVERY language and pass the registered cross-language agreement equivalence test, or the behavior arm is descriptive only. Independently, four rule-based rater-free indicators reproduce the primary comparison with no scoring model, and trait-specific configural and metric invariance is established per language. Self-report validity is screened by per-trait per-language internal consistency, measurement invariance with DIF flags and BFI-2/TIPI convergence, and rule E excludes the 12 development-touched BFI-2 items from the confirmatory H2 score. AGREEMENT IS NEVER READ AS EQUAL CALIBRATION OR EQUAL SENSITIVITY, AND NO CROSS-LANGUAGE CALIBRATION IS ESTABLISHED ANYWHERE IN THIS DESIGN (R17-F1): a language-dependent scoring slope remains an unexcluded rival of every reported contrast. Rule F: every admission-affecting check runs on the 60-scenario QUALIFICATION partition, disjoint from tuning and from the confirmatory core, the admitted checkpoint set freezing before B1; the 20% rater-language subsample of the core is a NON-GATING sensitivity. Two pre-B1 qualifications per language survive: MULTILINGUAL ANCHOR PAIRS, DESCRIPTIVE DIAGNOSTICS ONLY because R16-F4(a) removes estimated calibration from the endpoint, and PERTURBATION PAIRS, whose equal-detection-sensitivity equivalence test remains GATING. Rater CONSTRUCT validity, estimated INSIDE each checkpoint-language cell on its 60 independent scenario clusters, is a powered pre-B1 admission gate; the agreement, calibration and sensitivity checks here do not by themselves establish construct validity and are never read as doing so.


## ARCHIVED BY R19 (filed in full; binding word for word) - pico.preregistration.analysis_plan rule E and rule H as they stood at R18

RULE E (the earlier 18/12 scorer-only quarantine is VOID): TUNING 30, QUALIFICATION 60 and EVALUATION 90 are DISJOINT partitions cut before any tuning and hashed in SCENARIO_PARTITION.json; all development-time selection happens on TUNING alone, the rater is then frozen by hash, and QUALIFICATION is read once, before B1, for the registered gates only - including, under R16-F5, the competence screen on its separately hashed 50-prompt qualification sample. Differential-item-functioning screening on evaluation data is a SENSITIVITY ANALYSIS ONLY: no item is added, removed, re-weighted or re-scored on evaluation-derived DIF. NUISANCES ARE BLOCK-ALLOCATED, NOT ALIASED WITH SEED, under rule H: one pre-registered schedule, frozen before any generation, assigns for every (cell, setting, scenario, seed) the sampling seed, forward-translation engine, persona wording variant, item order and option order; engine and wording are block-allocated inside each cell-setting and never rotated across seeds.


## ARCHIVED BY R19 - pico.definitions[5] matched controls and estimands as they stood at R18

Matched controls: every rendering passes the same three-forward-translation plus back-translation screen, and generation length is matched by a per-language band calibrated on tuning. Estimands, each a within-scenario, within-seed contrast against P = D = I = s: three main effects, three two-way interactions and the three-way term, on the unnormalized scale by eq6, decomposed by eq7; NONE is a confirmatory endpoint for H1.


## ARCHIVED BY R19 - pico.controls.confounders_addressed[10], the competence screen as it stood at R18

Pre-registered target-language competence screen, AS AMENDED BY R16-F5, WHICH GOVERNS: it runs on its own SEPARATELY HASHED 50-prompt QUALIFICATION sample, never on the 30 tuning scenarios. For every checkpoint x target language, 50 held-out neutral support prompts are scored for response-language accuracy (language identifier at >= 0.9 confidence) and instruction compliance under the frozen decoding configuration. A cell enters the grid only at accuracy >= 0.95 and compliance >= 0.90; failing cells are reported OUT OF COMPETENCE and excluded, because a model that cannot answer in Swahili would otherwise register as a model whose induced trait does not transfer to Swahili - the exact confound the study is about. Run once, frozen, never re-run after seeing evaluation data. A checkpoint enters confirmation only if it clears ALL THREE L3 languages (R16-F3), decided before B1; fewer than three qualifying checkpoints makes H1 UNTESTABLE.


## ARCHIVED BY R20 - METHOD_DESIGN section 10 as it stood at R19

## 10. R17 — the design of record

`design_card.json` → `pico.preregistration.decision_rule` is **one self-contained specification, sections
A–N**, with **no amendment stack**; every superseded revision and repair narrative is archived verbatim in
`DESIGN_REVISION_HISTORY.md` and has **no force**. **rule X** citations point at section X; section N lists
every void inference. The repair history of R13, R14 and R15 is archived in `DESIGN_REVISION_HISTORY.md` and has **no force**;
**the executable text is in the card (`decision_rule` sections A–N plus R15-F1 to R15-F4, R16-F1 to R16-F7, R17-F1 to R17-F4, R18-F1 and R19-F1), which governs and is NOT restated here; under R19-F1 it is normalized once, with every filed-out clause inlined verbatim, into the generated `PREREG_OF_RECORD.json`, and no clause binds from anywhere else**. **R16**, in one line each (F1 English instrument and the gating invariance family; F2 one-sided signed max-T
admission; F3 all three languages at every checkpoint; F4 no estimated anchor calibration in the endpoint, gating
samples in `QUAL_BUDGET.json`; F5 the separately hashed 50-prompt competence sample; F6 the conservative pre-B0
power certificate; F7 the label set and `consistency_check.py`), **is filed in full in
`DESIGN_REVISION_HISTORY.md`** and is not restated here.

**R17 — the four repairs of record**, answering the R16 design-kill's three fatals and two of its
partially-controlled findings. **F1** narrows H1 everywhere to an **operational rater-score claim**:
assigning a dialogue language that differs from the persona-instruction language lowers the pole separation
of the score assigned by the frozen ENGLISH-instrument rater ensemble, as a **composite effect of assignment
to that exact pipeline including execution failures**; cross-language intercept-and-slope equivalence is
**not established and not claimed**, a language-dependent scoring slope is a registered **unexcluded rival**
named beside every effect, and every trait-retention reading is **VOID**. **F2** raises the confirmatory
resampling to **S = G = 24 independent seed clusters** (V̂ on 23 df) for every family, voids the {8…12} list
and rule J's escalation branch, and makes a **blocking pre-B0 calibration certificate** of
attained null FWER (upper bound ≤ 0.05) and simultaneous coverage (lower bound ≥ 0.95) for the complete
max-T vector with V̂ re-estimated; failure means the study is **not run as designed**. **R18-F1** replaces its single
exported level: the certificate of record is `power/CLUSTER_CALIB_R18.json`, which fixes **one level per family**,
calibrated to the **worst point of a registered adversarial DGP envelope** (E01–E12, bounded at skewness 2.0, t₃ tails,
9:1 heteroskedasticity, 5:1 imbalance, ρ ∈ [0, 0.9]) — primary **q\* = 0.998** (FWER 0.0379, upper 0.0412; coverage
0.9625, lower 0.9592 at corner A), pair-level 0.999, language-factorial 0.997, self-report-H2 0.996,
pole-heterogeneity 0.997, instrument-language invariance 0.996, diagonal admission 0.96, evaluator-validity F2 0.96.
The old single level **q\* = 0.965 is VOID** (FWER 0.153 under the envelope), and `POWER_SIM_R18.json` at the primary
level fixes **δ_alt = 0.38**. **Validity is claimed only inside the envelope.** **F3** makes
`RUN_TENSOR.json` and `FAMILY_REGISTRY.json`, generated by `schedule/make_run_tensor.py` before M1, the
**single source** of every count, split, gate and family: the competence screen is on the qualification
partition only, the confirmatory core is the admitted 3 × 3 subtable on **90** evaluation scenarios at
**24** seeds, the 6 × 6 grid is explicitly **descriptive** (30-scenario subset, 4 seeds), and the family
inventory has exactly **six** members — every "five families" wording is VOID; the extended
`consistency_check.py` PASS is a precondition of B0. **F4** deletes the round-trip control's
**exact-isolation** claim, adds an original-versus-one-way-rendered control at the same final language,
states that **no same-slot comparator exists**, and relabels every power figure as **conditional on every
admission gate passing**.

