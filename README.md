# Cross-lingual induced personality: saved-score analysis materials

Materials for **Auditing a prompt, generator and large language model rater pipeline for cross-lingual induced personality: an exploratory English-Chinese case study**, by Jibin Zhang and Hui Gui.

This is an exploratory audit of one prompt–generator–rater pipeline. The pre-specified confirmatory transfer analysis was not run. Positive high-minus-low rubric separation does not establish personality construct validity or cross-lingual transfer.

## Data availability and scope

The repository provides author-written customer-support scenarios, persona settings and rubric items; saved author-generated dialogues; item-level numeric LLM scores, parse outcomes and execution receipts; frozen scientific configurations and protocol records; archived numerical summaries; and the CPU analysis programs used for the reported diagnostics. The supplementary namespace includes all saved score shards needed for the later multi-rater, edited-text and four Chinese-rubric-rendering analyses.

The saved scores and transcript features can be reanalysed without loading a model. This release supports statistical reproduction from completed runs. It does not claim to reproduce the original GPU inference environment, to rerun generation or scoring, to repeat human evaluation, or to execute the confirmatory analysis that was not run.

FriendsPersona and CPED television-script text is **not redistributed**. Numeric score/label/cluster mappings and integer text-length metadata are provided for their statistical analyses. Obtain the corpora directly from their original repositories if their script text is needed; see [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). This separation does not prevent the published AUC, exclusion-count, leave-one-group-out or text-length-baseline calculations from being recomputed from the saved numeric files.

## CPU reproduction

Run commands from the repository root. The pinned analysis environment requires Python 3.11 or later, NumPy and SciPy. Plotting additionally requires Matplotlib and Pillow. The publication replay was tested on Linux through WSL with Python 3.13; native Windows execution was not separately validated.

```sh
python -m pip install -r requirements-analysis.txt
python reproduce.py --verify-only
python reproduce.py --output-dir recomputed
```

The entry point verifies publication-file hashes and runs the analysis jobs on CPU. It writes new summaries to `recomputed/` and keeps archived inputs and results unchanged. It makes no network calls and does not load model weights. `python reproduce.py --list` lists jobs; `--only supplementary supplementary_common supplementary_renderings` runs the supplementary analyses alone. Temporary output under `recomputed/.working/` satisfies the historical reducers' restriction that an output must remain inside the project, then each result is moved to the requested output directory.

The four numeric plots can be redrawn from the saved result JSONs with the program under `paper/figures/`. The original removed pipeline illustration is not included. Plot numbering and submission-specific font/output formatting should follow the accompanying final manuscript files.

## Verification performed for this publication copy

The CPU replay and numeric comparison are documented in [verification/README.md](verification/README.md). Saved result JSONs remain archived outputs. Recomputed hashes, path metadata and schema versions may differ; these changes are not statistical discrepancies. For example, the current pole reducer produces schema 2 and extra missingness fields for the original English block whose saved summary has schema 1. Its common statistical values agree with the saved summary.

## Publication-copy transformations

Local machine/model paths, host identifiers and interpreter-path fields have been removed or replaced in the selected publication copies. Detected email-like strings in the generated customer-support material have been replaced with non-contact masks of exactly the same Unicode character length: letters become `x`, digits become `0`, `@` becomes `#`, and other characters remain. There were 260 replacements in 114 files. The replacement preserves the length, Han/Latin script counts, question/exclamation counts and sentence-terminal punctuation pattern used by the replayed diagnostics. Saved rating values have not been changed or re-scored.

[PROVENANCE.json](PROVENANCE.json) distinguishes the SHA-256 of privately retained original bytes from the SHA-256 of each sanitized publication copy. Original execution hashes still recorded in a score row or historical receipt identify original artifacts; they must not be treated as hashes of the public file. The replay entry point separately checks public bytes and the recorded original configuration/scorer provenance. Derived third-party unit metadata has its own public hash.

## Historical inference-source boundary

Two versions of `supp_score.py` were recovered and matched against the historical digests recorded in the supplementary score shards: the pre-pool version (`6a633450228849962eed2cdbb88b42bfaa2fdac20bbb569ba3c21e1b9d700d62`) and the pool version (`6b66f5e36e6f5db00fbaf5c15c80876bbe6722a4fedb57b4a729a8fdebfefa14`). The pre-pool publication copy is under `experiments/xlpt/versions/` and is byte-identical to its recovered original. Other files that needed path redaction have separate original/public hashes in the provenance map.

The 100 records in the historical test file `smoke2-mistral.jsonl` record scoring digest `2df227c7dd7a9c69fc29ccd30c8fc498742e991913382936c10ca08d37f74d54`; an exact source file for that test digest was not recovered. The main and supplementary production reducers do not load this test file. The production supplementary score rows instead refer to the recovered pre-pool/pool versions above. The test-source gap therefore does not affect the saved-score statistical replay, but exact GPU re-execution of that historical smoke test remains unavailable. Inference programs also retain environment-specific historical assumptions and are not advertised as a portable GPU runner.

Historical protocol documents are versioned records of what was planned at their writing date. Statements such as “not yet run” in those records do not describe the publication snapshot as a whole. The protocol was an internal project record, not a publicly registered protocol.

## Files and interpretation

- `data/xlpt-support/`: authored scenario bank and rendered scenario/rubric materials.
- `process/consumption/frozen/`: versioned prompt surfaces and frozen scientific inputs.
- `process/results/xlpt/`: generation shards, canonical item-score shards and saved summaries.
- `process/results/xlpt/supp-20261004/`: later rater, text-edit, public-label and Chinese-rubric score shards and summaries.
- `process/results/xlpt/h-public/`: saved public-corpus numeric scores and non-text unit metadata.
- `process/refine-logs/`: selected scientific protocol, partition and deviation records. Historical paths beginning `refine-logs/` refer to this selected record directory; the public replay entry point resolves it explicitly.
- `experiments/xlpt/`: generation/scoring source and CPU reducers; vendor fragments retain upstream notices.
- `DATA_DICTIONARY.md`: record fields and how incomplete scores are represented.
- `SHA256SUMS`: publication file digests.

## Rights

This snapshot does not assign a new license to the authors' materials. Third-party software fragments retain the upstream terms listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Third-party corpus acquisition and reuse remain subject to their original terms. No television script text is covered by an author-material license in this repository.
