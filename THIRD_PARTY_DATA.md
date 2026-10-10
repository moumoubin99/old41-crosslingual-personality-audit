# Third-party television dialogue sources

No original television-script text is included in this repository. Saved numeric ratings, published labels/group mappings and derived integer length metadata support the statistical replay without corpus text.

## FriendsPersona

- Original source: https://github.com/emorynlp/personality-detection
- Recorded revision: `3ec08a58dc7c708c5dfc314b3bff8f5808786928`.
- Source CSV: `CSV/friends-personality.csv` in the original repository.
- Recorded source-file SHA-256: `89c9d0785eb21570899f1d9202284aee48be1698ce89664f1283c73c4a6b7cd3`.
- The local import provenance records the license as unspecified (`null`). This publication copy does not infer permission to redistribute the source script text.
- Extracted sample: 711 `(scene, character)` units, with fictional-character identity and scene clustering retained for the numerical analysis.

Acquire the data directly from the original source and check its applicable terms. Do not treat this repository's publication of numeric derivatives as a license to copy the television script.

## CPED

- Original source: https://github.com/scutcyr/CPED
- Recorded revision: `1e4b81c28a123f22387e06664f37e5dc9322380f`.
- Fixed source: https://github.com/scutcyr/CPED/tree/1e4b81c28a123f22387e06664f37e5dc9322380f
- Original license: https://github.com/scutcyr/CPED/blob/1e4b81c28a123f22387e06664f37e5dc9322380f/LICENSE
- Original repository records Apache-2.0. The source dialogue is nevertheless not mirrored in this release.
- Extracted sample: 569 units; 287 retained fictional identities. The source-file SHA-256 values for `train_split.csv`, `valid_split.csv` and `test_split.csv` are in `process/results/xlpt/h-public/H_PUBLIC_PROVENANCE.json`.

## Rebuilding script units after independent acquisition

The historical extraction implementation is `experiments/xlpt/score_public_personality.py`, specifically the corpus builders and the `--build-only` path. It expects separately acquired files under:

```text
data/collected/public_personality/friendspersona/friends-personality.csv
data/collected/public_personality/cped/train_split.csv
data/collected/public_personality/cped/valid_split.csv
data/collected/public_personality/cped/test_split.csv
```

Check the source revision and input hashes before extraction. Historical inference/binding checks assume the original deployment environment; this source is provided to document the extraction rule, not as a tested portable generation/scoring runner. The CPU replay in `reproduce.py` does not need these files or `units-*.jsonl`.

The publication files `unit-metadata-friendspersona.jsonl` and `unit-metadata-cped.jsonl` expose only unit/group keys, labels, character counts and line counts. Their publication hashes and the original private unit-file digests are listed separately in `PROVENANCE.json`.
