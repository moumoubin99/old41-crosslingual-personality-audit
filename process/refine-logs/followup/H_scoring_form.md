# Block H: scoring form (template v2)

> **SUPERSEDED 2026-09-29, not used.** No human rater is recruited; see `H_public_spec.md`.

**v2, 2026-09-29.** This version applies the form edits from the Codex review. The Agreeableness page is removed
(`agreeableness_items_included: false`).

- The build step fills in one copy per rater.
- The rater completes one row per presentation ID, in the order given.
- A CSV with the same columns is accepted.

**Rater:** R__  **Order file:** ______  **Date started:** ______  **Date finished:** ______

Scale: an integer from 1 to 7 for each item. The anchors are in `H_annotation_protocol.md`; keep the anchor sheet
open while rating.

| # | Presentation ID | E1 | E2 | E3 | E4 | E5 | E6 | Language mismatch (y/n) | Cannot rate (y/n) | Note (optional, ≤ 1 line) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | H__________ | | | | | | | | | |
| 2 | H__________ | | | | | | | | | |
| … | … | | | | | | | | | |
| 96 | H__________ | | | | | | | | | |

## CSV header (machine form)

```
rater_id,order_index,presentation_id,E1,E2,E3,E4,E5,E6,language_mismatch,cannot_rate,note
```

## Row rules

- **`cannot_rate`.** Always `y` or `n`.
- **When `cannot_rate = n`.**
  - E1–E6 are all required, and each is an integer from 1 to 7.
  - `language_mismatch` is required (`y` or `n`).
- **When `cannot_rate = y`.** E1–E6 must be blank. `language_mismatch` may be left blank.

## Validation before the key is opened

A returned form is **rejected and returned to the rater** if any of these holds:

- a row breaks a rule in "Row rules";
- a score is not an integer or lies outside 1–7;
- a flag is not `y`/`n`;
- a presentation ID is missing, duplicated or unknown;
- the rows are not in the rater's order file (`order_index` must match).

The key is opened only after every form passes validation.

## Locking

- A form that passes validation is **locked**, and its sha256 is written to `H_ANNOTATION_RECEIPT.json`.
- After locking, substantive scores cannot change.
- The only change allowed is a documented clerical correction: a transcription error proven against the rater's
  original sheet. It is logged in the receipt with the before and after values and a reason.

## R3 form

- **Columns and rules.** The same as above.
- **What R3 sees.** Only the referred presentation IDs, in an R3-specific order.
- **When it is built.** Only after the R1 and R2 forms and flags are locked.
- **Blinding.** R3 sees neither R1's nor R2's scores or flags, nor the reason a transcript was referred.
