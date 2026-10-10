"""Recount of script statistics for gemma-3-12b-it turns in the P = en, D = zh pole phases.

Reads the generation shards, writes checks/gemma_script_recount.json. A turn passes the
registered rule when Han characters are at least half of its Han plus Latin letters.
"""
import glob, json, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[2] / "process/results/xlpt"
HAN = re.compile(r"[一-鿿㐀-䶿]")
LAT = re.compile(r"[A-Za-zÀ-ɏ]")
TONE = re.compile(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈŌÓǑÒ]")
out = {}
for pole in ("high", "low"):
    n = tone = zero = ok = 0
    files = sorted(glob.glob(str(ROOT / f"zh-b4-extraversion-{pole}/gemma-3-12b-it/seed-*.jsonl")))
    for f in files:
        for line in open(f, encoding="utf8"):
            for a in json.loads(line)["agent_turns"]:
                a = a if isinstance(a, str) else a.get("content", "")
                h, l = len(HAN.findall(a)), len(LAT.findall(a))
                n += 1; tone += bool(TONE.search(a)); zero += h == 0
                ok += (h + l > 0 and h / (h + l) >= 0.5)
    out[pole] = {"seed_files": len(files), "turns": n, "turns_with_pinyin_tone_marks": tone,
                 "turns_with_zero_han": zero, "turns_passing_rule": ok, "pass_rate": round(ok / n, 4)}
(pathlib.Path(__file__).parent / "gemma_script_recount.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
