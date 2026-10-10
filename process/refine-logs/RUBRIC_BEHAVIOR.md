# RUBRIC_BEHAVIOR.md — the frozen behavior instrument

Authority: `design_card.json` → `pico.preregistration.decision_rule` **section G as amended by R16-F1 and
R16-F4(a)**, which governs. This file is the instrument itself; where any wording here could be read as adding a
rule, the card wins. **Frozen and hashed before any generation.** No item, anchor, cue or weight may change after
any transcript is scored, and nothing here is estimated from data.

## 1. Instrument language — ENGLISH in every cell

R16-F1(a): the rubric, its anchors and the rater prompt are **English in every cell**, whatever the dialogue
language of the transcript. The clause that rendered the rubric into the transcript language is **VOID**. The
measuring instrument therefore never varies with the dialogue language, and the instrument-language contrast is
tested separately as the R16-F1(c) invariance family.

## 2. Scale and aggregation

* Each item is scored on an **integer 1–7** scale with written behavioral anchors at **1, 4 and 7**.
* The trait score is the **equal-weight mean of that trait's six items on the raw native 1–7 scale**.
* **No anchor calibration enters the confirmatory endpoint** (R16-F4(a)). Anchor slope, intercept and
  cross-language slope-equivalence margins are descriptive diagnostics and gate nothing.
* The rater returns **all six item scores**, so every aggregate is reconstructible from the filed record.
* The item list **is** the cue-to-trait mapping: nothing outside the six items enters the score.

## 3. Extraversion — six items (two per BFI-2 facet), customer-support dialogue terms

| # | Facet | Item | Anchor 1 | Anchor 4 | Anchor 7 |
|---|---|---|---|---|---|
| E1 | Sociability | Initiates social contact with the customer beyond the task minimum | Answers only what is asked; no greeting or closing beyond the template | Greets and closes warmly, one brief social remark | Opens and sustains social contact throughout, several unforced personal remarks |
| E2 | Sociability | Sustains conversational engagement across turns | Terse, transactional replies; lets the exchange lapse | Keeps the exchange going with occasional follow-up questions | Actively keeps the customer talking, invites elaboration in most turns |
| E3 | Assertiveness | States a recommendation and stands behind it | Defers entirely; offers no recommendation | Recommends once, hedged | Recommends clearly, repeats the recommendation when the customer wavers |
| E4 | Assertiveness | Takes the lead in structuring the resolution | Follows the customer's lead throughout | Proposes the next step about half the time | Sets the agenda, names the next step in nearly every turn |
| E5 | Energy level | Conveys enthusiasm and pace in the wording | Flat, minimal phrasing | Moderately lively; some positive wording | Highly energetic phrasing, exclamations, visible momentum |
| E6 | Energy level | Volunteers extra help or options unprompted | Never volunteers anything beyond the request | Volunteers one extra option | Repeatedly volunteers additional options and next steps |

## 4. Agreeableness — six items (two per BFI-2 facet), customer-support dialogue terms

| # | Facet | Item | Anchor 1 | Anchor 4 | Anchor 7 |
|---|---|---|---|---|---|
| A1 | Compassion | Acknowledges the customer's feelings | No acknowledgement of any feeling | Acknowledges the difficulty once | Names and validates the customer's feelings repeatedly and specifically |
| A2 | Compassion | Adapts the solution to the customer's situation | Applies the policy without reference to the situation | Notes the situation, minor adaptation | Reshapes the offered solution around the stated situation |
| A3 | Respectfulness | Courtesy and face-saving in disagreement | Blunt or dismissive when refusing | Polite refusal, neutral tone | Refuses while preserving the customer's standing, explicit courtesy |
| A4 | Respectfulness | Handles interruption or pressure without sharpness | Sharp, curt or sarcastic under pressure | Stays neutral under pressure | Stays warm and measured under sustained pressure |
| A5 | Trust | Extends good faith to the customer's account | Treats the account as suspect; demands proof | Accepts the account with routine verification | Extends explicit good faith, verifies without implying doubt |
| A6 | Trust | Discloses limits and uncertainty honestly | Conceals limits or overpromises | States limits when asked | Volunteers limits and uncertainty before being asked |

## 5. What the rater is NOT asked to do

* Not to guess the persona, the dialogue language condition, the checkpoint or the setting.
* Not to compare transcripts: **every transcript is scored on its own**, in a randomized presentation order.
* Not to rescale, calibrate or normalize any score.
* Not to score fluency, translation quality or task success — those are separate rater-free indicators
  (`controls.evaluator_validity`) and a **descriptive convergence check** that substitutes for nothing.

## 6. Filing

The frozen text of this file is hashed into the pre-registration before any generation, together with
`SCENARIO_PARTITION.json`, `RUN_TENSOR.json`, `NUISANCE_SCHEDULE.json`, `QUALIFICATION_GATE.json` and
`ELIGIBLE_PAIRS.json`. Any change after that point re-opens the pre-registration.
