# Experiment plan: cross-lingual transfer matrix of induced personality

Companion to `METHOD_DESIGN.md` and `design_card.json`, which is authoritative for
every number; this document fixes the claim map, the run order and the stopping conditions.

**Execution status (2026-09-28): partially executed, out of the preregistered order. The evaluation split is exposed, and the confirmatory status of this corpus is void.**

- **Qualification seed block (20260811+).** B0 was generated here. The English and zh (P = en) Extraversion pole blocks were also generated on the 23 retained qualification seeds.
- **EVALUATION partition, read out of order.** B1-neutral and the B2 compliance-only control read the EVALUATION partition (90 scenarios × 24 seed clusters). They ran under operator instruction `.aris/OPERATOR_SKIP_RULE_F`, before the rule-F admission freeze, before G2–G5 were met, and before the R22-F5 renewal. B3's indicator contrast uses those transcripts.
- **The deviation and its binding consequences** are stated once in `METHOD_DESIGN.md`:
  - every result is exploratory;
  - the 90 scenarios can never carry an H1/C1 confirmatory reading;
  - confirmation needs a new preregistration frozen before a replacement, never-read evaluation corpus is cut, with rule-F gates met first and power re-sized for support and refutation near β₀.
- **Remaining gaps.** No human annotation exists, and no registered gate G2–G5 is met.
- **Decision (b), user, 2026-09-28.** The study stays explicitly exploratory. No replacement preregistration or replacement corpus is built, and this revision is not submitted for renewed design confirmation (`METHOD_DESIGN.md`, consequence 4).
- **Follow-up schedule.** `FOLLOWUP_PLAN_2026-09-28.md` contains descriptive blocks only, and it amends nothing below. The milestones below describe the design as registered; they are not a record of what ran.

*At freeze, this line read:* "not started — no generation, accelerator time, annotation or data collection has occurred." Revision **R12 as amended by R15–R35**, the design of record: the ONE self-contained specification is `design_card.json` itself, and the later-numbered repair governs (**R35 > R34 > R33 > … > R12**). There is one **fixed** seed count (**S = 24**, no escalation, **R26-F5**), one run tensor and one inferential unit (**the R27-F2 subsample cluster, n_U = 144**) per family; the qualification-and-gate channel runs on the **disjoint** seed block 20260811 + {0..23} (**R26-F6**). A PASS receipt of both `normalize_prereg.py` and `consistency_check.py` is a precondition of B0. The **language factorial** (LangFac), block **B1b**, runs **in full** as the gated second level (**R25-F4**). Admission is decided before B1 and frozen into hashed `QUALIFICATION_GATE.json`. Nothing below runs before the design confirmation point, which under **R22-F5** must be renewed against the operational R17-F1 question; accelerator phases additionally require downstream permission.

---

## 1. Claims to defend

**Primary claim (C1), in the narrowed R17-F1 words** of `pico.hypothesis.statement`. Cross-language calibration of the rater is **not established and not claimed**; a **language-dependent scoring slope is a registered unexcluded rival** named beside every reported effect; no trait-retention or psychometric reading exists. The evidence required is stated once in that field and `decision_rule` rules A, B, C, F, I with **R15-F1/F2/F3 as amended by R16–R35**. Inference is conditional on the 90 fixed evaluation scenarios (rule E). C1 is an **intention-to-treat** claim about *assignment*, and it is pole separation, **never a persona-versus-neutral magnitude** (R15-F2).

**Exploratory read (C2), per R15-F4 and R17-F4(a).** Whether the loss is an artifact of the measuring apparatus — the within-transcript instrument-language contrast, the matched text-quality perturbation arm and the two translation controls, none of which **isolates** anything. No apparatus-exclusion claim, no confirmatory family.

**What is NOT confirmatory, and still runs in full.** No main effect of D is a confirmatory read (rule L). The behavior channel's P × D language factorial is the **gated second-level family** (q* = 0.997, budget 0.003); the self-report factorial carries H2, the rater-language factor H2b, and the 6 × 6 matrix is a **descriptive** omnibus pipeline effect (R17-F3). **C1 says nothing about why** — competence, tokenization, pragmatics, translation naturalness and rater calibration stay untested.

**Anti-claims** are registered as H2, H2b and H3 in `pico.hypotheses[]`, each with its own estimand, family and kill mechanism: the round-trip control narrows and never refutes (the falsifier of record is `pico.hypothesis.falsifier`), H3 defaults to **UNRESOLVED**, compliance is handled by the compliance-only control instruction.

One confirmatory claim, under the cap. No other claim is defended.

---

## 2. Experiment blocks

Five core blocks, each with its purpose, cost and the decision it feeds.

### B0 — Instrument and rater qualification, and the pre-B1 qualification gates

Establish that the apparatus works in all six languages and decide **all** checkpoint admissions, language admissions and family membership *before* any evaluation generation. The governing text is `decision_rule` sections E, F, F1–F4, G, H, H2, I with **R15-F3** and **R31-F1**–**R34-F2**.

**Partition first (rule E).** The 180 support scenarios are cut ONCE — before any prompt is written — into **TUNING 30 / QUALIFICATION 60 / EVALUATION 90**, disjoint and hashed as `SCENARIO_PARTITION.json`; development runs on TUNING only, QUALIFICATION is read **once** before B1 for the gates below on the **disjoint 20260811 seed block (R26-F6)**, and EVALUATION stays sealed.

**Rule F — the gates and their freeze.** Every admission-affecting check runs on the QUALIFICATION partition at the same 4 checkpoints, 5 settings and three engines, hashed into the pre-registration at M1 before B1 opens; the admitted checkpoints, the admitted languages and the primary family freeze then, and no admission decision reads an evaluation-split result. The executable text is card rules F, F2, F3, F4, G with **R22-F1**, the **R31-F1**–**R34-F1** evaluator-validity specification and the **R34-F2** perturbation family; **anchor calibration is descriptive only (R16-F4(a))**. The **sole** power certificate is `POWER_SIM_R30_FINAL.json` and the gate certificate `power/GATE_POWER_R30_FINAL.json` (**R30-F1**/**R30-F2**); `power/CLUSTER_CALIB_R18.json` is the nominal-level file. Both, with the `consistency_check.py` PASS, are filed at the close of B0, *before* B1 opens (**R28-F1**); `RATER_DETERMINISM.json` is filed after B0 generation and before any gate scoring (**R22-F2**).

### B1 — The transfer matrix (main result, evaluation split)

The **confirmatory core** is read from `RUN_TENSOR.json` (**R17-F3**) and enumerated once in `pico.outcome.n_runs`, not here; inference runs on the **n_U = 144 R27-F2 units** and the full 6 × 6 directed grid **descriptively only**. Produces per-cell (A) − (B) differences, the diagonal reference, the unnormalized `Δ_mis` per off-diagonal pair, the ratio and shortfall with Fieller intervals on cells passing (Elig), and the asymmetry index. Cost: ≈200–320 GPU hours plus ≈40 CPU hours. **Feeds the descriptive omnibus report — no claim (R29-F3), and not C1's operational claim (R25-F4).**

### B1b — The language factorial, split by outcome channel (main result, evaluation split)

The instrument factor I cannot causally affect the behavior channel: the transcript is produced under P and D and *frozen* before the inventory is administered, so the plan runs two outcome-specific designs on the same generations. The run tensor and the scenario counts are enumerated once in `pico.outcome.n_runs` (rule E).

Cost: see §4. **Feeds C1 and separates H1 from H2.**

### B2 — Apparatus controls

Run alongside B1 on the same scenarios and seeds:

**Feeds C2.** A tie between the round-trip dummy baseline and the real switched condition **NARROWS** the C1 claim and is a sensitivity signal favouring H2; it **NEVER** rejects C1. Only clause (B) of the falsifier of record (rule C, as amended by **R26-F7**) can refute H1.

### B3 — Ablations

The ablations and drops of `pico.ablations`, plus the subordinate crossed factorial and the **descriptive** switch × resource-tier breakdown, on a 30-scenario subset of the evaluation split with the same seeds. **The subset is frozen and hashed at M1, before B1 starts**, so it cannot be chosen after the evaluation scenarios are seen. **Feeds the subordinate readings; the primary mechanism separation lives in B1b.**

### B4 — Contamination and scope reporting

13-gram overlap scan of BFI-2 and TIPI item text against the available pretraining corpus proxies, per-item, with per-factor breakdowns and token counts per cell. Cost: CPU only. **Feeds the leakage and scope sections.**

---

## 3. Run order and milestones

The evaluation split is touched **once**, at M3. All tuning, prompt selection and threshold setting happens on the development split at M2 or earlier. B3 reuses M3's generations where possible.

---

## 4. Budget

Every number below is read from `pico.outcome.n_runs` and `RUN_TENSOR.json` (R20-F1); the card's `compute_budget` is authoritative.

The 1.5× allowance covers **prespecified infrastructure failures only**, re-run at the same seed; **no regeneration may depend on any outcome value** (the ITT rule). Budget is matched across arms on dialogues, turns and generations, not tokens.

**Host constraint.** The throughput figures in `pico.outcome.compute_budget` are measured on the 48 GB card of record (**R35-F2**); no *scientific* quantity is measured in house. The dispersion prior is literature-derived (ref-010) and re-estimated on TUNING at M2.

---

## 5. Risks and how each is handled

**Open obligations, carried not silently closed.** (i) Amidei et al. (COLING 2025) on interrater-agreement reporting is **not** in `refs/refs.json` and is unverified here; if it changes the agreement protocol, B0's gate is revised before M2. (ii) `paradigm_map.py place` returned **DANGLING_REF** for the aggregation/interval module, repaired by re-sourcing to ref-010.

---

### Intention-to-treat sampling rule — locked before any generation

## 6. Stopping and falsification

No outcome here is a failed experiment; each is a specific, preregistered reading. **The ONE falsifier of record is `pico.hypothesis.falsifier` (rule C, as amended by R23-F3); this section is a pointer and states no independent refutation path.**

---

## 7. Final checklist before any run

Each item is the *ordering* obligation only; the executable rule is the cited card section.

## Models, engines and hardware

Named once in `METHOD_DESIGN.md` and in the card. The **target-language competence screen** (**R16-F5**) runs before B1 on its own hashed 50-prompt QUALIFICATION sample; failing cells never enter the matrix.

**Hardware to rent: 1× 48 GB accelerator** (hardware of record: RTX 4090 48 GB, **R35-F2**) — the four graded checkpoints serve at bf16 and the 24B rater serves in its int4 AWQ build at ~22 GB (**R35-F1**), so no 80 GB card is required. Envelope: `pico.outcome.compute_budget`.

## 8. R15–R35 — the design of record