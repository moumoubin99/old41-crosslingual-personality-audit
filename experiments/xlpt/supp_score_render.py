"""Chinese-rubric condition with another machine rendering of the items.

supp_score.py is the scoring script of the supplementary round and is not edited. It
reads the Chinese items from one fixed file. This wrapper runs that script unchanged
and only substitutes which rendering file is read, so the request head, the labels,
the transcripts, the rater and the decoding are those of the S3 condition and the
rendering of the items and anchors is the single thing that differs.

Usage:
  python -m experiments.xlpt.supp_score_render --rendering r2 -- <supp_score arguments>

The receipt written by supp_score.py then carries the hash of the rendering file in
instrument_sha256; a sidecar <tag>.render.json names the rendering and this wrapper.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from experiments.xlpt import supp_score

DEFAULT_NAME = "supp_rubric_zh.json"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "--rendering" or argv[2] != "--":
        raise SystemExit("usage: supp_score_render --rendering <id> -- <supp_score arguments>")
    rendering, rest = argv[1], argv[3:]
    if "--instrument" not in rest or rest[rest.index("--instrument") + 1] != "zh":
        raise SystemExit("this wrapper is for --instrument zh only")
    project = Path(rest[rest.index("--project") + 1]).resolve()
    tag = rest[rest.index("--tag") + 1]
    alternative = project / f"experiments/xlpt/supp_rubric_zh_{rendering}.json"
    if not alternative.exists():
        raise SystemExit(f"missing rendering file {alternative}")

    original_items, original_sha = supp_score.zh_items, supp_score.sha256_file

    def redirect(path):
        path = Path(path)
        return alternative if path.name == DEFAULT_NAME else path

    supp_score.zh_items = lambda path: original_items(redirect(path))
    supp_score.sha256_file = lambda path: original_sha(redirect(path))

    code = supp_score.main(rest)
    out_dir = project / supp_score.OUT_ROOT
    (out_dir / f"{tag}.render.json").write_text(json.dumps({
        "tag": tag, "rendering": rendering,
        "rendering_file": str(alternative.relative_to(project)),
        "rendering_sha256": hashlib.sha256(alternative.read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_script_sha256": original_sha(Path(supp_score.__file__)),
        "request_head_and_labels": "those of the first rendering (supp_score.ZH_REQUEST_HEAD)",
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
