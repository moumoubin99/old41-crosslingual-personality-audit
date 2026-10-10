# Licensing for the v1.0.1 publication snapshot

Copyright (where applicable) 2026 Jibin Zhang and Hui Gui.

The authors selected the following licenses on 10 October 2026. Grants apply to the extent that the authors hold the applicable rights. They do not assert that every machine output or numeric fact attracts copyright.

## Author-written software: MIT

The root `LICENSE` contains the full MIT license. It applies to `reproduce.py`, author-written Python programs under `experiments/xlpt/` outside `vendor/`, and author-written plotting/check programs under `paper/`. MIT permits software reuse under its standard notice and disclaimer conditions.

All of `experiments/xlpt/vendor/` is excluded from this author MIT grant. Embedded copies of third-party source in deployment/runtime JSONs retain their upstream terms, even when surrounding author metadata is CC BY 4.0. See `THIRD_PARTY_NOTICES.md` and the supplied license texts.

## Author-created data and research materials: CC BY 4.0

Author-created scenario banks, persona settings, rubric materials, frozen author prompt surfaces, saved generated customer-support dialogues, their machine ratings and execution records, author protocol documents, statistical summaries, figures and publication documentation are licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The full standard legal text is in `licenses/CC-BY-4.0.txt`.

Credit Jibin Zhang and Hui Gui, link to the repository and CC BY 4.0 license, retain the relevant source attribution and indicate modifications. Suggested dataset attribution:

> Zhang J, Gui H. Cross-lingual induced personality: saved-score analysis materials. Version 1.0.1 (2026). https://github.com/moumoubin99/old41-crosslingual-personality-audit/releases/tag/v1.0.1. CC BY 4.0 for author-created data/materials.

## Mixed files and third-party exclusions

`process/results/xlpt/h-public/` and `process/results/xlpt/supp-20261004/pub-*` combine author-produced model ratings, parsing/provenance fields or statistical outputs with corpus-derived identifiers and annotations. The author-produced contributions may be reused under CC BY 4.0. Original FriendsPersona/CPED annotation labels, fictional-character identifiers, unit/scene/group/cluster mappings and acquired corpus elements are excluded from the author grant and retain their original-source terms. This applies wherever those fields are repeated, including supplementary summaries and corpus provenance. The full third-party script text is not redistributed.

Deployment/runtime records can contain embedded third-party function source. That embedded source retains its original software license; the author grant covers only the authors' own metadata or additions. Do not apply the author MIT or CC BY license to the embedded upstream code. Model weights and original television-script files are not included.

`LICENSE_MAP.json` classifies every publication file. It records the author license and third-party exceptions for mixed files; its rules are not a replacement for the full legal texts or upstream source terms. The licenses, upstream notices and exceptions above take precedence over a filename extension.

Original corpus acquisition instructions remain in `THIRD_PARTY_DATA.md`. Third-party code retains Apache-2.0 or BSD-3-Clause as identified in `THIRD_PARTY_NOTICES.md`.
