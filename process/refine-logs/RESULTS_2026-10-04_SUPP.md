# Results record: supplementary round `supp-20261004`

Written 2026-10-05. This is a new dated record; `RESULTS_2026-10-03.md` is unchanged. All results
here are exploratory and descriptive. No qualification gate was read or written, no canonical score
shard was overwritten, no new dialogue was generated, and no human annotation was collected.
Reducer verdict: `DESCRIPTIVE_ONLY`, `eligible_for_gate: false`.

## What was run

Re-scoring of frozen transcripts on the first six registered seeds (20260811–20260816) × 50 scenarios
× 2 poles × 14 cells = 8,400 transcripts (three generators with four cells, gemma-3-12b-it with the
two P=en cells). Rubric: six Extraversion and six Agreeableness items, 1–7, greedy decoding
(`temperature 0.0, top_p 1.0, top_k 1, max_tokens 8, seed 20260911`).

| Block | Content | Raters and coverage |
|---|---|---|
| S0 | Batched re-score against the canonical one-request-at-a-time scores | Mistral-Small-3.1-24B, one seed (1,400 transcripts, 16,800 calls) |
| S1 | Second and third rater, E + A items | Llama-3.1-8B-Instruct and gemma-3-12b-it, six seeds (100,800 calls each) |
| pub | FriendsPersona and CPED | Llama and Gemma (8,532 and 6,828 calls each) |
| S2 | Edited transcripts: `strip_edges`, `swap_edges`, `trunc2`, E items | Llama and Gemma six seeds (151,200 calls each); Mistral three seeds (75,600 calls) |
| S3 | Chinese rendering of the six E items (`experiments/xlpt/supp_rubric_zh.json`) | Llama and Gemma six seeds (50,400 calls each); Mistral three seeds (25,200 calls) |
| CPU | Crossed intervals on common masks, item-removed separations, missing-transcript bounds, length baseline, leave-one-identity-out | existing scores |

## Code and outputs (SHA-256)

- `experiments/xlpt/supp_score.py`: `6b66f5e36e6f5db00fbaf5c15c80876bbe6722a4fedb57b4a729a8fdebfefa14`
  for every Gemma and Mistral shard. The Llama shards were produced by the earlier version, recorded
  in their receipts as `code_sha256` beginning `6a633450`. The difference is prompt tokenization in a
  worker pool before the engine starts, with a sampled re-tokenization by the engine tokenizer
  (`pooled_tokenization: verified`); request text, model and decoding are the same.
- `experiments/xlpt/reduce_supp.py`: `d964c781f60ebd78461d6b6b191204501373f2a7659a81898738db349831fe3e`
- `experiments/xlpt/supp_chain.sh`: `72b5cefc2caeb1f123796d9f5d5dcc012249a81683c866bf46359e41d9c9c7b6`
- `experiments/xlpt/supp_chain2.sh`: `6fd40d4800c0c3f8794094fb62383c216ac8637d732eee4e325f198600864adc`
- `experiments/xlpt/supp_rubric_zh.json`: `4599e30c7396c98b878cd0ad03f2d48390f23d7755a017deb8299fa17c465182`
- `process/results/xlpt/supp-20261004/SUPP_SUMMARY.json`: `80bcec228093be22d1c31e34d0e321ed964357b52d5637b73f9a10f018d95afb`
- `paper/supp_reading.json`: `70b96f2b5d898ed9b750665fe8b91744901842fa6c61715a3199814ea1a71e8f`

Shards in `process/results/xlpt/supp-20261004/` (each with a `.receipt.json`; local, server and
receipt digests are equal):

| Shard | Records | Calls | SHA-256 |
|---|---|---|---|
| `s0-mistral-none.jsonl` | 1,400 | 16,800 | `6c28367aa1f2e6999a86a48003be46a62976cbe17c680adf6799d7034cb1448b` |
| `s1-llama-none.jsonl` | 8,400 | 100,800 | `c76cd1b5150c192f57fcbaa01a6908c3d915e8d7d1621aacb2ec6ddb5ae63cd9` |
| `s1-gemma-none.jsonl` | 8,400 | 100,800 | `bdfa8f9980b47465c654c2ab00a9b9f38e7617cdb94b9763aab9273fff30d3a5` |
| `pub-llama-friendspersona.jsonl` | 711 | 8,532 | `1dc18244536f6836cac794de615f61eda3fcde59848d2a015c1ce085b90af783` |
| `pub-llama-cped.jsonl` | 569 | 6,828 | `d908febfa3ab8b48e1b7e23848763f93f5bd3245a48959a8ee7c0aa5597776bd` |
| `pub-gemma-friendspersona.jsonl` | 711 | 8,532 | `1ff0ec7a785d86063f3764a56abd1170de50e02df3b41557e83ab9c6f3edf845` |
| `pub-gemma-cped.jsonl` | 569 | 6,828 | `d63320c3ea8e0fe3a9355c46662650a98dfa16c92edc0d7493db77b119227859` |
| `s2-llama-surface.jsonl` | 25,200 | 151,200 | `8cd0dc850ec957da64931dfa4baaa80647ca6edb209c9d827164ff5800da9763` |
| `s2-gemma-surface.jsonl` | 25,200 | 151,200 | `4748fc2e6c6d4976777d9de8438aa717bcccbbc8b5a71836709aa4af28380397` |
| `s2-mistral-surface.jsonl` | 12,600 | 75,600 | `97731d8a920739c879f2a2dbcaa3bbd397309eb2d379fa666eddd44a82ce005f` |
| `s3-llama-zh.jsonl` | 8,400 | 50,400 | `99d217f229379792577d5abb5f37a6001fbca2748a3a99669eaeecc1bc3e2389` |
| `s3-gemma-zh.jsonl` | 8,400 | 50,400 | `85d0ab3d5cf59d9892ba0449507dd29369ac9f9aec00052a8bb7de8ef6b8329a` |
| `s3-mistral-zh.jsonl` | 4,200 | 25,200 | `20a5b729a9eb317efa63714f1fff59f639230205834ab52885e3a6f9fd267f59` |

## Execution incidents

1. First chain (2026-10-04 03:00 CST): the Llama jobs finished. Every Gemma and Mistral job failed at
   engine start because the KV cache did not fit the GPU memory fraction; the fractions were raised
   and the chain relaunched at 12:19 CST. No score was produced by the failed starts.
2. Single-threaded prompt tokenization took close to an hour per job on a CPU quota shared with
   another project, so the scorer was changed to pooled tokenization (hash change above). The stalled
   Gemma job was stopped before it produced any score.
3. 2026-10-05 01:39 CST: `s2-mistral-surface` stopped with ENOSPC when the system disk of the host
   filled. The checkpoint held 50,297 completed calls and one truncated last line. The truncated line
   was dropped; the damaged copy is kept on the server as
   `logs/s2-mistral-surface.partial.damaged-20261005.jsonl`. The job was resumed from the checkpoint
   by `supp_chain2.sh` at 02:01 CST with GPU memory fraction 0.45 and `TMPDIR` on the data disk; only
   the 25,303 missing calls were issued (receipt: `calls_resumed_from_checkpoint` 50,297). It ended
   with rc=0 at 03:59 CST and `s3-mistral-zh` with rc=0 at 06:12 CST.
4. The GPU and CPU were shared with another project of the user throughout; that project was not
   touched. The sharing slowed both.

## Key numbers (from `SUPP_SUMMARY.json`)

Rater order: Mistral (canonical) / Llama / Gemma. Intervals are 95% crossed seed × scenario
bootstrap intervals.

- Extraversion separation, range over cells and generators: 1.04–1.24 / 1.17–1.57 / 2.02–2.15.
  Agreeableness separation: 0.47–0.69 / 0.90–1.25 / 2.09–2.30.
- Ratio of the all-Chinese to the all-English cell (E separation): 0.83 [0.77, 0.91] / 0.75
  [0.67, 0.83] (0.68 [0.61, 0.76] without Llama's own transcripts) / 1.00 [0.95, 1.05].
- Agreement between rater pairs on transcript E means: Pearson 0.57–0.78 overall, 0.18–0.58 within
  a pole; mean differences 0.96–1.99 rubric points; quadratic-weighted kappa on items 0.07–0.59.
- Batching check (S0, 1,400 transcripts): Pearson 0.999 with the canonical scores, mean difference
  −0.001, exact item agreement 99.4–100%.
- Edited transcripts, share of the unedited separation retained: `strip_edges` 98% / 122% / 102%;
  `swap_edges` 42% / 83% / 87%; `trunc2` 45% / 14% / 14%; `trunc2` without item E1 −2% / −8% / 1%.
  Under Llama, `trunc2` reverses the sign in the D=zh cells (−31%, −54%).
- Chinese rubric rendering, ratio of the all-Chinese to the all-English cell: 0.85 → 0.90 (Mistral,
  three seeds; separations 1.03–1.22 → 0.54–0.63), 0.75 → 1.66 (Llama; 1.17–1.57 → 0.93–1.55),
  1.00 → 1.06 (Gemma; 2.02–2.15 → 1.76–1.86).
- Public corpora, AUC against human Extraversion labels: FriendsPersona 0.443 [0.367, 0.526] / 0.415
  [0.330, 0.500] / 0.403 [0.309, 0.514]; CPED 0.494 [0.436, 0.555] / 0.544 [0.478, 0.607] / 0.510
  [0.442, 0.576]. Length-only baseline 0.496 (FriendsPersona) and 0.509 (CPED); leave-one-identity-out
  on FriendsPersona 0.373–0.454.
- Missing: qwen3-8b masks 10 seed-scenario positions, as in the canonical analysis.

## Limits of this round

- Llama and Gemma are also generators, so they rate their own transcripts; the Llama ratio is given
  with and without them.
- The canonical rater covers one seed for S0 and three of the six seeds for S2 and S3.
- The Chinese rubric is one rendering with no validation and no human bilingual review.
- gemma-3-12b-it contributes P=en cells only (it failed the Chinese response-language screen).
- Agreement between raters is low within a pole and the Gemma rater shows no cell ordering, so the
  cell ordering is rater-dependent.
- The public corpora are scripted television dialogue, outside the customer-support domain.
- No human criterion exists for the customer-support dialogues.

## Addendum, 2026-10-05 (after review 6)

The prose generated from the summary was reworded after the Codex review (scope and wording only; no
number in `SUPP_SUMMARY.json` changed and the reducer was not re-run). `paper/supp_reading.json` is now
`902cdba1a71899f5d64710e556b2bee1ffa27b67f3c33ff8d79c7ee74ab44be9`. Two points of definition: the
"retained" share over cells is the edited separation averaged over the four cells divided by the unedited
one, per generator, and is not the mean of the four cell ratios; and the reducer counts 1,532 turns with
lower-case tone-marked letters in `gemma-3-12b-it|en_zh|high`, while `paper/checks/gemma_script_recount.py`
also counts upper-case ones and gives 1,534.

## Addendum, 2026-10-05 (after review 7)

Correction of a description in "Key numbers" above: the ranges 1.04-1.24, 1.17-1.57 and 2.02-2.15 are
ranges over the four cells after averaging the three generators with all four cells; they are not ranges
"over cells and generators". The numbers are unchanged. The generated prose was reworded again after Codex
review 7 (scope and wording only; `SUPP_SUMMARY.json` unchanged, reducer not re-run);
`paper/supp_reading.json` is now `95100cd0ade4994cab677e126a54e95cf688fca5ad1d1f91731a75da981aedb7`. The edge-removal sentence now reports that single generators
differ: for aya-expanse-8b transcripts the retained share is 93% [89%, 98%] under the canonical rater and
135% [126%, 147%] under Llama-3.1-8B-Instruct, so "does not reduce under any rater" held for the
three-generator average only.

## Addendum, 2026-10-06 (post hoc common-mask reductions; no model call)

A second reducer, `experiments/xlpt/reduce_supp_common.py`
(sha256 `a047bf065430e6cc9db88bba7eabc2b32d81823539fe2da6df3bc4fe24c042fb`), imports `reduce_supp.py`
unchanged and re-reduces the existing score shards. It ran on the server CPU on 2026-10-06 (no GPU, no model
call, no new score). Output: `process/results/xlpt/supp-20261004/SUPP_COMMON_SUMMARY.json`
(sha256 `e02d537033e8e5c99a1e37127d24fbdddae3e9908c45b554060c78a9005ce88d`), verdict `DESCRIPTIVE_ONLY`,
`eligible_for_gate: false`. The reductions were specified after the earlier results and review 7 had been
read. They are not registered and decide no gate. Nothing above is changed.

- Item attribution on one mask (units complete under both scorings; rounding error 0.0001). After truncation to two
  sentences, the five items without E1 retain -0.02 [-0.08, 0.05] / -0.08 [-0.19, 0.02] / 0.01 [-0.02, 0.04]
  (canonical / Llama / Gemma) and E1 alone 0.93 [0.91, 0.95] / 0.97 [0.87, 1.09] / 0.75 [0.66, 0.84]. After the
  exchange of opening and closing, E1 alone retains 0.04 [0.01, 0.06] / 0.50 [0.46, 0.55] / 0.54 [0.49, 0.59].
- Same three seeds (20260811-20260813), common transcripts, 4197 units: ratio zh_zh/en_en 0.84 [0.75, 0.93] /
  0.76 [0.66, 0.86] / 1.00 [0.94, 1.06] with the English rubric and 0.89 [0.80, 1.00] / 1.68 [1.47, 1.93] /
  1.07 [1.01, 1.12] with the Chinese rendering. Three seed clusters: intervals have poor coverage.
- Agreement (Pearson, all / within high / within low) without self-rated transcripts: canonical-Llama
  0.59 / 0.10 / 0.30 (n 5998); canonical-Gemma 0.77 / 0.43 / 0.43 (n 7194); Llama-Gemma 0.76 / 0.15 / 0.62
  (n 4797). Over generator-by-cell strata the canonical-Llama value ranges 0.28-0.88 (all) and -0.01-0.24
  (within high). In the gemma en-zh block that fails the language screen: canonical-Gemma 0.93 all, 0.05 within high.
- Public-corpus unit counts per rater are in S6 Appendix (S6 Table 11).

Decision of the project owner, 2026-10-06: no human rating and no human check will be collected. The missing
human criterion for customer-support dialogue and the unvalidated Chinese rendering therefore stay stated
limitations of the manuscript. No open corpus of customer-support dialogue with human Extraversion ratings
of the agent was identified; this is a statement about what was found, not a systematic search.

## Addendum 2026-10-08 — extension of the round (E1 six seeds for the canonical rater; E2 four renderings)

Status: `DESCRIPTIVE_ONLY`; re-scoring of frozen transcripts with open models on the rented GPU; no
generation, no change to an existing shard, nothing written to `QUALIFICATION_GATE.results`, no gate decided.
The rules were fixed in `.aris/context/NOTES.md` on 2026-10-06 (17:22 and 17:33 CST) before any new score
was read. No human check of a rendering was made, by decision of the project owner.

Runs (all finished with rc=0; 29 receipts in `process/results/xlpt/supp-20261004/`, 12 of them `s4-*`):
- E1: `s2-mistral-surface-b`, `s3-mistral-zh-b` (canonical rater, seeds 20260814–20260816).
- E2: `s4-{llama,gemma}-zh-{r2,r3,r4}` (six seeds) and `s4-mistral-zh-{r2,r3,r4}` with `-b` halves
  (three seeds each). Renderings: `experiments/xlpt/supp_rubric_zh_{r2,r3,r4}.json` (Qwen3-8B,
  aya-expanse-8b, NLLB-200-3.3B). r4 is rough (incomplete high anchor of item E1) and was kept as produced.
- Reduction on the server CPU: `experiments/xlpt/reduce_supp_ext.py --project .` →
  `SUPP_EXT_SUMMARY.json`, `missing_shards` empty, finished 2026-10-07 23:27 CST.

Execution incidents (no score shard was written by a failed start; logs kept as `logs/*.failed-*.log`):
1. `s4-mistral-zh-r3` rc=1 on 2026-10-07 14:00: the KV cache did not fit because the GPU memory of the
   previous job had not yet been released. Rerun by `experiments/xlpt/supp_chain5.sh`, rc=0 at 18:41.
2. `s4-gemma-zh-{r2,r3,r4}` rc=1 at 18:37–18:38: started by chain 4 while the rerun in (1) still held the
   card. Rerun by chain 5 after the card was free: rc=0 at 20:11, 21:40 and 23:09. Chain 5 ended 23:21 with
   `missing=0`.

Values (mean over the three generators with all four cells; canonical rater / Llama-3.1-8B-Instruct /
gemma-3-12b-it; 95% crossed seed-and-scenario bootstrap; six seed clusters):
- Retained share after truncation to two sentences: 0.44 [0.41, 0.47] / 0.14 [0.05, 0.22] / 0.14 [0.11, 0.17];
  item E1 alone 0.93 [0.91, 0.95] / 0.97 [0.87, 1.09] / 0.75 [0.66, 0.84]; five items without E1
  −0.02 [−0.06, 0.03] / −0.08 [−0.19, 0.02] / 0.01 [−0.02, 0.04].
- Opening and closing sentences exchanged: 0.41 [0.38, 0.44] / 0.83 [0.79, 0.88] / 0.87 [0.85, 0.88];
  removed: 0.97 [0.94, 1.00] / 1.22 [1.17, 1.28] / 1.02 [1.00, 1.04].
- One common mask of 8,380 transcripts (complete under every rater, the English rubric and all four
  renderings), ratio of the all-Chinese to the all-English separation:
  - canonical: English rubric 0.83 [0.77, 0.90]; r1–r4 0.90 [0.83, 0.99], 0.92 [0.84, 1.01],
    0.90 [0.83, 0.98], 0.91 [0.82, 1.00].
  - Llama: English rubric 0.74 [0.67, 0.82]; r1–r4 1.66 [1.48, 1.88], 1.44 [1.30, 1.62],
    1.41 [1.27, 1.58], 1.72 [1.46, 2.07].
  - gemma: English rubric 1.00 [0.95, 1.05]; r1–r4 1.06 [1.01, 1.11], 1.02 [0.96, 1.08],
    0.98 [0.94, 1.03], 1.03 [0.97, 1.09].
- Effect of the dialogue language at an English persona: canonical 0.15 (English rubric), 0.04 to 0.07
  (renderings); Llama 0.38, −0.67 to −0.49; gemma 0.10, −0.01 to 0.12.
- Pearson correlation of transcript Extraversion means, rendering against English rubric: 0.82–0.83
  (canonical), 0.79–0.83 (Llama), 0.95–0.96 (gemma).

Reading: the reversal of the cell ordering under Llama-3.1-8B-Instruct occurs with all four machine
renderings, so it is not specific to one rendering; it can still come from properties that machine
renderings share. Under the canonical rater no rendering gives a ratio above one. The three-seed limitation
of the canonical rater for the edited-text and rubric checks is removed; the earlier three-seed tables stay
in the manuscript as produced, and the six-seed values are reported next to them. These results do not
validate any rater and decide no gate.


## Addendum of 2026-10-08 — receipts in the result directory

The result directory holds 29 receipts and 27 score files enter the reductions. The count is: 13 score
files of the first pass (`SUPP_SUMMARY.json`), 14 of the extension (`s2-mistral-surface-b`,
`s3-mistral-zh-b` and the 12 `s4-*` files; `SUPP_EXT_SUMMARY.json`), and 2 receipts of smoke tests of
the scoring client (`smoke2-mistral`, `smoke3-mistral`), which enter no table. No result changed. The
14 files of the extension are now listed in S6 Appendix with their receipt checks.
