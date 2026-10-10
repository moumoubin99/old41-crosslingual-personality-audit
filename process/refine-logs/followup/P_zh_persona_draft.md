# Block P: Chinese persona text (P = zh)

**Status: FROZEN, 2026-09-29, in `process/consumption/frozen/english_execution_v4.json` (config self_sha256
`b02068de…`) with `english_execution_receipt_v4.json` (receipt self_sha256 `3304381a…`).** The freeze was run by
`experiments/xlpt/freeze_pzhn_v4.py`. It took the six strings below verbatim, checked NFC, full-width
punctuation, 97/97 persona and 138 full lengths, and D-variant parity, and carried every v3 setting byte-identical.
The text below is unchanged from the reviewed version. The receipt records this file's sha256 as of the moment of
freezing, taken before this status line was edited.

- **Decided.** The user delegated the four decision points and the style pass to Codex. The review is
  `<LOCAL_PATH>/review.md` (gpt-5.6-sol, xhigh, 2026-09-29).
  - The text below is Codex's output, verbatim.
  - Its acceptance is provisional: the reviewer notes that its secondary cross-check was same-family, not
    cross-family.
- **Freeze done.** The text is frozen in `english_execution_v4.json` with its receipt. The BC phases and queues
  are built.
- **Go-ahead given 2026-09-29.** The user answered 「开跑」 to the Tier-2 decision.
  - Block BC uses this text through `bc_generation_queue.json` (276 jobs) and `bc_rater_queue.json` (276 jobs).
  - Both queues are chained after block A on server 41 by `process/experiment_queue/chain_a_bc.sh`.

## Why the v3 `_PZH` strings are replaced

The v3 strings (`process/consumption/frozen/english_execution_v3.json`, FROZEN and unchanged) are raw
`facebook/nllb-200-3.3B` output. They have three kinds of defect:

- **Content loss in LOW.**
  - `EXTRAVERSION_LOW_PZH` drops the English clause "Keep each reply short and plain".
  - The v3 P = zh LOW pole is therefore a weaker instruction than its English source. This is a content change,
    not only a fluency problem.
- **Content loss in HIGH.**
  - "talkative" appears as "说话" ("speaking").
  - "keep the exchange lively" appears as "保持交换活", which is truncated.
  - "speak first" loses its contrast with waiting.
- **Disfluency throughout.**
  - The strings keep English syntax ("支持的代理人谁是保留") and use ASCII punctuation.
  - "Do not claim actions" is rendered as "不要要求采取…行动" ("do not demand to take actions"). This changes the
    constraint's meaning.

Any v3 P = zh result must carry these defects as a limitation. The v3 `_PZH` phases were never run, so no
existing result is affected.

## Design constraints for the new text

- **Fidelity.** The text follows the v2 English poles clause for clause. No persona content is added or removed.
- **Facets.** It keeps the same three Extraversion facets (sociability, assertiveness, energy) and adds no
  Agreeableness instruction.
- **No inventory wording.** Nothing from BFI-2, TIPI or any other inventory.
- **Register.** Full-width Chinese punctuation, in a register natural for a customer-service system prompt.
- **Parity.** HIGH and LOW are parallel in structure and equal in length, so that length gives no pole cue of its
  own.
- **Variants.** The D = zh and D = en variants differ only in the final response-language sentence.

## Codex decisions on the four open points

1. **「不加多余的问候」, not 寒暄客套.** It excludes only greeting embellishment, while 寒暄客套 could be read as
   forbidding any greeting, and it overlaps with 不闲聊.
2. **「主导处理过程，推动问题解决」.** It keeps both leading and driving toward resolution, without adding "the
   entire process" and without promising success.
3. **「操作」, in the collocation 「进行了……操作」.** It is the natural word for account and service operations.
   行动 is too general, and 采取操作 is awkward.
4. **Exact length parity.** The persona portion before 「遵循」 is **97 characters** in both HIGH and LOW. Each full
   D = zh string is **138**.
   - Counting convention: NFC code points, with every Han character and punctuation mark counting as one.
   - The counts were rechecked mechanically on 2026-09-29.

## Canonical identifiers

- The new strings are named `*_PZHN_*`. The v3 `*_PZH*` names stay reserved for the frozen, never-run NLLB
  strings, so the two cannot be confused.
- Mapping from the Codex review labels to these identifiers:
  - `HIGH_PZH_DZH` → `EXTRAVERSION_HIGH_PZHN_DZH`
  - `LOW_PZH_DZH` → `EXTRAVERSION_LOW_PZHN_DZH`
  - `NEUTRAL_PZH_DZH` → `NEUTRAL_PZHN_DZH`
  - The `_DEN` labels map the same way.

## Final strings

### `EXTRAVERSION_HIGH_PZHN_DZH`

```
你是一名外向、健谈、充满活力的客服人员。热情地开场并称呼客户的名字，先开口而不是等客户提问；清楚直接地给出建议，并主导处理过程，推动问题解决；让对话保持轻快活跃；主动提供客户未要求的进一步帮助。遵循所提供的账户和政策事实。不要声称进行了事实不允许的操作。用中文回复客户的请求。
```

### `EXTRAVERSION_LOW_PZHN_DZH`

```
你是一名内敛、安静、低调的客服人员。每次回复都应简短平实；让客户主导对话并掌握节奏；只回答客户问到的内容，除非客户征求意见，否则不发表意见；不加多余的问候，不闲聊，也不提供客户未要求的额外帮助。遵循所提供的账户和政策事实。不要声称进行了事实不允许的操作。用中文回复客户的请求。
```

### `NEUTRAL_PZHN_DZH` (reference only; BC runs the poles)

```
遵循所提供的账户和政策事实。不要声称进行了事实不允许的操作。用中文回复客户的请求。
```

### `EXTRAVERSION_HIGH_PZHN_DEN`

```
你是一名外向、健谈、充满活力的客服人员。热情地开场并称呼客户的名字，先开口而不是等客户提问；清楚直接地给出建议，并主导处理过程，推动问题解决；让对话保持轻快活跃；主动提供客户未要求的进一步帮助。遵循所提供的账户和政策事实。不要声称进行了事实不允许的操作。用英文回复客户的请求。
```

### `EXTRAVERSION_LOW_PZHN_DEN`

```
你是一名内敛、安静、低调的客服人员。每次回复都应简短平实；让客户主导对话并掌握节奏；只回答客户问到的内容，除非客户征求意见，否则不发表意见；不加多余的问候，不闲聊，也不提供客户未要求的额外帮助。遵循所提供的账户和政策事实。不要声称进行了事实不允许的操作。用英文回复客户的请求。
```

### `NEUTRAL_PZHN_DEN`

```
遵循所提供的账户和政策事实。不要声称进行了事实不允许的操作。用英文回复客户的请求。
```

### Facts label

- **D = zh:** keep the v3 label `账户和政策事实:` byte-identical, ASCII colon included. The Chinese scenario
  surface then matches the surface that A already scores.
- **D = en:** the English scenario rendering stays exactly as in v2.

## Clause-by-clause back-gloss (from the Codex review)

### HIGH

| English source (v2) | Final Chinese | Literal back-gloss |
|---|---|---|
| You are a support agent who is outgoing, talkative and energetic. | 你是一名外向、健谈、充满活力的客服人员。 | You are an outgoing, talkative, energetic customer-service agent. |
| Open warmly and by name, | 热情地开场并称呼客户的名字， | Open warmly and address the customer by name, |
| speak first rather than waiting to be asked, | 先开口而不是等客户提问； | speak first rather than wait for the customer to ask; |
| state your recommendation plainly and take the lead in driving the case to a resolution, | 清楚直接地给出建议，并主导处理过程，推动问题解决； | state your recommendation clearly and directly, lead the handling process, and drive the problem toward resolution; |
| keep the exchange lively, | 让对话保持轻快活跃； | keep the exchange brisk and lively; |
| and volunteer further help the customer has not requested. | 主动提供客户未要求的进一步帮助。 | volunteer further help the customer did not request. |

### LOW

| English source (v2) | Final Chinese | Literal back-gloss |
|---|---|---|
| You are a support agent who is reserved, quiet and low-key. | 你是一名内敛、安静、低调的客服人员。 | You are a reserved, quiet, low-key customer-service agent. |
| Keep each reply short and plain, | 每次回复都应简短平实； | Keep each reply short and plain; |
| let the customer lead the exchange and set its pace, | 让客户主导对话并掌握节奏； | let the customer lead the exchange and control its pace; |
| answer only what was asked and offer no opinion unless the customer asks for one, | 只回答客户问到的内容，除非客户征求意见，否则不发表意见； | answer only what the customer asked; unless the customer solicits an opinion, express none; |
| and add no greeting flourish, small talk or unrequested extra help. | 不加多余的问候，不闲聊，也不提供客户未要求的额外帮助。 | add no extra greeting, engage in no small talk, and provide no unrequested extra help. |

### Shared sentences

| English source (v2) | Final Chinese | Literal back-gloss |
|---|---|---|
| Follow the supplied account and policy facts. | 遵循所提供的账户和政策事实。 | Follow the supplied account and policy facts. |
| Do not claim actions that the facts do not permit. | 不要声称进行了事实不允许的操作。 | Do not claim to have performed operations the facts do not permit. |
| Respond to the customer's request in Chinese. / in English. | 用中文回复客户的请求。/ 用英文回复客户的请求。 | Respond to the customer's request in Chinese. / in English. |

## Disclosure carried into any BC result

- The P = zh surface uses a **single reviewed rendering** of the persona. It does not use the card's three-engine
  translation ensemble.
- That rendering was reviewed by a model reviewer, not by a native human reader.
- The Chinese scenario surface is still the single-stratum NLLB rendering (a v3 limitation).
- Any P effect is therefore specific to this rendering.
