# XLPT-Support v1.0 — authoring protocol and rule-E partition rule

Corpus of record for `refine-logs/design_card.json` (`pico.comparison.datasets[0]`, name `XLPT-Support`),
written under **rule E** of `pico.preregistration.decision_rule`. This file is the *one* authoring
protocol referred to by the card's phrase "authored in ONE pass under one authoring protocol and the
v1.0 inventory". Where this file and the design card differ, **the card governs**.

Status: source corpus authored; nothing generated, translated, scored or run. No accelerator time used.

---

## 1. What a scenario is

A scenario is a **written customer-support situation in English** that fixes, for every cell of the
run tensor, exactly what the customer says and what the agent is allowed to know. The model under
test always plays the **support agent**; the customer turns are corpus text and are never generated.

Each scenario is therefore a *stimulus*, not a dialogue: the transcript that enters scoring is the
scenario's customer turns interleaved with the checkpoint's generated agent turns.

**Trait neutrality (load-bearing).** No scenario field describes, requests or rewards any agent
personality, tone, warmth, assertiveness, energy or politeness level. The only source of induced
style in the experiment is the persona instruction, which lives in the frozen prompt templates and
never in the corpus. A scenario that named an agent manner would confound the persona contrast with
scenario content and is out of protocol.

**Instrument independence.** No scenario contains BFI-2 or TIPI item text, trait vocabulary, Likert
labels or response-scale wording. This keeps the 13-gram contamination scan of the instrument
separable from the scan of the dialogue material.

## 2. The 180 and how they are structured

The corpus is exactly **180** scenarios, laid out as a complete crossing:

**6 support domains × 5 request intents × 6 variants = 180.**

| Code | Domain |
|---|---|
| `telecom` | mobile and broadband service |
| `retail` | consumer e-commerce orders |
| `banking` | retail banking, cards and payments |
| `travel` | airline and hotel bookings |
| `saas` | software subscriptions and accounts |
| `utilities` | home utilities and field services |

| Code | Request intent |
|---|---|
| `billing_dispute` | an unexpected, duplicated or disputed charge |
| `status_delay` | an order, delivery, refund or appointment that has not arrived |
| `technical_fault` | a service or device that does not work |
| `cancel_refund` | a cancellation, return or refund request |
| `account_access` | login, identity, profile or plan-change problems |

The **six variants** inside each (domain, intent) stratum differ in the situational particulars —
the customer's constraint, the evidence available, the account state and how much room the agent has
to resolve the request. They are authored to be *equally difficult in kind*, not ranked; the variant
index carries no ordering and no severity scale.

There are exactly **30 strata** of 6, which is what makes the rule-E cut exactly balanced (§5).

## 3. Turn structure and the length band

Every scenario has **exactly three customer turns**, in a fixed communicative order:

1. **Opening** — states the problem and one concrete identifier (order number, last four digits,
   appointment window, plan name).
2. **Complication** — adds a constraint, a correction or a piece of evidence that rules out the
   obvious first answer.
3. **Decision request** — asks the agent to commit to a specific outcome or to say plainly that it
   cannot.

Each customer turn is authored at **22–32 English words**, and each scenario at **70–84 English
words in total**. This is the source-side half of the card's per-language length band
(`pico.intervention.change`: identical turn count, mean turn length within ±10 %); the
target-language half is calibrated on the TUNING partition only, at M2, and is never re-fitted
afterwards.

The band is **uniform across every domain and intent, and is enforced per scenario, not only per
turn**. Mean turn length is itself one of the registered rater-free behavioural indicators, so any
systematic difference in authored length between strata would enter the outcome as a corpus
artefact rather than as model behaviour. The per-scenario total is what removes that confound: a
turn-only band still permits whole strata to sit at opposite ends of it. `tools/build_scenario_partition.py`
checks both bands and refuses to write the corpus if either is violated.

The agent's reply length is not constrained by the corpus — it is bounded only by the frozen
decoding configuration (`max_new_tokens` 512).

## 4. Record schema

One JSON object per line, UTF-8, keys in this order:

| Field | Meaning |
|---|---|
| `scenario_id` | `xlpt-<domain>-<intent>-v<N>`, unique, stable, never reused |
| `domain` | one of the six domain codes |
| `intent` | one of the five intent codes |
| `variant` | integer 1–6 within the stratum; an index only |
| `title` | short English label for logs and tables; never shown to a model |
| `context` | facts the agent is given in the dialogue framing: account state, system limits, what the agent may and may not do |
| `customer_turns` | list of exactly three strings, in order |
| `resolution_space` | the closed list of outcomes the agent may offer, so that task success and escalation are decidable by a rule rather than by taste |

`title`, `domain`, `intent` and `variant` are **metadata**: they are written to logs and never placed
in any model's context, in either channel. Only `context`, `customer_turns` and `resolution_space`
are rendered into the six languages and reach a model.

## 5. Rule-E partition — the cut of record was made before this corpus existed

The corpus is cut **once**, into **TUNING = 30, QUALIFICATION = 60, EVALUATION = 90**, mutually
disjoint. **The cut of record is `refine-logs/SCENARIO_PARTITION.json`**, which is filed with the
pre-registration and hashed alongside `RUN_TENSOR.json`, `NUISANCE_SCHEDULE.json` and
`RUBRIC_BEHAVIOR.md`. It is not recomputed here and may not be recomputed at all.

That file was written by `refine-logs/schedule/make_scenario_partition.py` **before any scenario was
authored**, as a seeded permutation (`numpy` `default_rng(20260911)`) of the 180 pre-registered
placeholder ids `sc001 … sc180`: the first 30 drawn are TUNING, the next 60 QUALIFICATION, the last
90 EVALUATION, each list sorted. Because the ids carried no content when the permutation was drawn,
the assignment of any individual scenario was unknowable while it was being written — which is the
property rule E requires.

The concrete corpus binds to that cut by **canonical corpus order**: line *i* of
`scenarios_en_v1.0.jsonl` is `sc{i:03d}`. The corpus order is fixed by the v1.0 inventory (six
domain shards in the frozen domain order, §7), is content-blind, and re-cuts nothing. Every rendered
scenario record carries its placeholder as `prereg_id` and its block as `partition`, so the binding
is auditable record by record. `tools/render_items_and_scenarios.py` performs it; re-running the
tool must reproduce the rendered files byte for byte, which is the reproduction check.

**Void.** An earlier draft of this protocol registered a second, competing cut — a per-stratum
ranking by `sha256("XLPT-Support|v1.0|rule-E|" + scenario_id)` computed over the real scenario ids
by `tools/build_scenario_partition.py`, written to `data/xlpt-support/SCENARIO_PARTITION.json`.
That rule and that file are **void**: they would have re-cut a corpus that the pre-registration had
already cut. The file has been removed and the tool no longer emits a partition; it assembles and
structurally checks the corpus only.

**Use discipline (rule E, restated, not amended).** All development-time selection — prompt and
persona wording, parsing rules, scorer and rubric configuration, translation-pipeline settings, the
dispersion re-estimate, `f_diag`, the anchor coefficients and the section-J variance components —
runs on **TUNING only**. QUALIFICATION is untouched until read once, before B1, for the section F1
and F2 gates on the 20260811 seed block. EVALUATION is untouched until B1 opens. The corpus is never
enlarged, re-cut or re-used across partitions, and confirmatory inference is conditional on the 90
fixed evaluation scenarios.

## 6. Provenance

The scenarios are **original English text written for this study**, composed against the situation
schemas that public task-oriented dialogue and support-intent resources make standard — the
schema-guided / multi-domain booking-and-service frame (domains, slots, transactional intents) and
the published customer-support intent taxonomies (billing, order status, technical fault,
cancellation and refund, account access). No sentence is copied from any corpus, transcript,
knowledge base or product document, and no real person, account, order or company appears: all
names, identifiers, amounts and dates are invented and internally consistent only within a scenario.

This matters for the contamination scan: because the text is new, a 13-gram hit against a
pretraining shard is evidence about incidental phrasing, not about the corpus having been published.

## 7. Files

| Path | Role |
|---|---|
| `data/xlpt-support/source/scenarios_en_v1.0.part{1..6}.jsonl` | the authored shards, one per domain |
| `data/xlpt-support/scenarios_en_v1.0.jsonl` | canonical corpus: the six shards concatenated in domain order, byte-identical to their contents |
| `refine-logs/SCENARIO_PARTITION.json` | the frozen rule-E cut of record, filed with the pre-registration (§5); **not** under `data/` |
| `refine-logs/RUBRIC_BEHAVIOR.md` | the twelve-item English rater instrument required by decision-rule section G, hashed into the pre-registration |
| `data/xlpt-support/rendered_items_and_scenarios/scenarios_en.jsonl` | the 180 scenarios as the English source surface, partition-tagged; part of the contamination scan's `test_path` |
| `data/xlpt-support/rendered_items_and_scenarios/rubric_items_en.jsonl` | the twelve rater items with their 1 / 4 / 7 anchors; English in every cell under R16-F1(a), so it has no target-language counterpart |
| `data/xlpt-support/rendered_items_and_scenarios/PENDING_SELF_REPORT_ITEMS.md` | what the scan directory does **not** yet contain, and why |

`tools/render_items_and_scenarios.py` writes the two rendered files from the canonical corpus, the
partition and the rubric; `--check` reproduces them byte for byte.

Two surfaces the contamination scan must eventually cover are **still absent**: the verbatim BFI-2
and TIPI item text, which must come from the authoritative published source and is not reconstructed
here, and the target-language renderings into Chinese, Spanish, German, Arabic and Swahili, which are
produced by the pinned three-engine translation ensemble and calibrated on TUNING at M2 — never by
hand. `PENDING_SELF_REPORT_ITEMS.md` records both gaps so that an incomplete scan is not mistaken for
a clean one.
