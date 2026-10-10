#!/usr/bin/env python3
"""Figs 2-5 for the PLOS ONE manuscript, drawn from the result JSONs (no hand-typed numbers)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "process" / "results" / "xlpt"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

GENS = ["Llama-3.1-8B-Instruct", "Qwen3-8B", "aya-expanse-8b"]
CELLS = ["P=en,D=en", "P=en,D=zh", "P=zh(v4 reviewed),D=en", "P=zh(v4 reviewed),D=zh"]
CELL_LBL = ["P=en\nD=en", "P=en\nD=zh", "P=zh\nD=en", "P=zh\nD=zh"]
KEYS = ["sep_en_en", "sep_en_zh", "sep_zh_en", "sep_zh_zh"]
COL = {"Llama-3.1-8B-Instruct": "#0072B2", "Qwen3-8B": "#D55E00", "aya-expanse-8b": "#009E73",
       "gemma-3-12b-it": "#666666"}


def save(fig, name):
    fig.savefig(HERE / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(HERE / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(HERE / (name + ".png")).convert("RGB")
    im.save(HERE / (name + ".tif"), compression="tiff_lzw", dpi=(300, 300))
    print(name, im.size, "%.2f x %.2f in" % (im.size[0] / 300, im.size[1] / 300))


def fig2():
    rob = json.loads((RES / "XLPT_2X2_ROBUSTNESS.json").read_text())["blocks"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    off = {g: (i - 1) * 0.2 for i, g in enumerate(GENS)}
    for g in GENS:
        est = rob[g + "|extraversion"]["estimates"]
        y = [est[k]["all_cell_mask_estimate"] for k in KEYS]
        ci = [est[k]["ci95_crossed_seed_scenario_bootstrap"] for k in KEYS]
        lo = [a - c[0] for a, c in zip(y, ci)]
        hi = [c[1] - a for a, c in zip(y, ci)]
        x = np.arange(4) + off[g]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=4, capsize=2.5, lw=1, color=COL[g], label=g)
    gem = rob["gemma-3-12b-it|extraversion"]["estimates"]
    gy = [gem[k].get("all_cell_mask_estimate", gem[k].get("cellwise_estimate")) if k in gem else None
          for k in KEYS[:2]]
    gy = [v if v is not None else gem[k]["cellwise_estimate"] for v, k in zip(gy, KEYS[:2])]
    ax.plot(np.arange(2) + 0.4, gy, "D", ms=4, mfc="none", color=COL["gemma-3-12b-it"],
            label="gemma-3-12b-it (P=en only)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(4), CELL_LBL)
    ax.set_ylabel("High − low separation, Extraversion items\n(rubric points)")
    ax.set_ylim(0, 3.6)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    save(fig, "Fig2")


def fig3():
    aud = json.loads((RES / "XLPT_2X2_AUDIT.json").read_text())["items_paired"]
    items = ["E%d" % i for i in range(1, 7)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, g in zip(axes, GENS):
        cells = aud[g + "|extraversion"]["cells"]
        m = np.array([[cells[c]["share_of_summed_item_separation"][it] for c in CELLS] for it in items])
        if np.nanmax(np.abs(m)) <= 1.5:
            m = m * 100
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=65, aspect="auto")
        for i in range(6):
            for j in range(4):
                ax.text(j, i, "%.0f" % m[i, j], ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] < 40 else "black")
        ax.set_xticks(range(4), CELL_LBL, fontsize=6.5)
        ax.set_yticks(range(6), items)
        ax.set_title(g, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Share of separation (%)")
    save(fig, "Fig3")


def fig4():
    ln = json.loads((RES / "XLPT_LENGTH_SENSITIVITY.json").read_text())["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
    for ax, g in zip(axes, GENS):
        stats, pos, cols = [], [], []
        for j, c in enumerate(CELLS):
            q = ln[c][g + "|extraversion"]["length_chars"]
            for k, pole in enumerate(["low", "high"]):
                s = q[pole]
                stats.append({"med": s["p50"], "q1": s["p25"], "q3": s["p75"], "whislo": s["p05"],
                              "whishi": s["p95"], "fliers": []})
                pos.append(j + (k - 0.5) * 0.36)
                cols.append("#56B4E9" if pole == "low" else "#E69F00")
        bp = ax.bxp(stats, positions=pos, widths=0.3, showfliers=False, patch_artist=True,
                    medianprops={"color": "k", "lw": 1})
        for b, col in zip(bp["boxes"], cols):
            b.set_facecolor(col)
            b.set_linewidth(0.6)
        ax.set_yscale("log")
        ax.set_xticks(range(4), CELL_LBL, fontsize=6.5)
        ax.set_title(g, fontsize=8)
    axes[0].set_ylabel("Agent characters per transcript (log scale)")
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(fc="#56B4E9", ec="k", lw=0.6, label="low pole"),
                             Patch(fc="#E69F00", ec="k", lw=0.6, label="high pole")],
                    frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "Fig4")


def fig5():
    hp = json.loads((RES / "h-public" / "H_PUBLIC_SUMMARY.json").read_text())["corpora"]
    pairs = [("E_to_EXT", "E items → EXT label"), ("E_to_AGR", "E items → AGR label"),
             ("A_to_EXT", "A items → EXT label"), ("A_to_AGR", "A items → AGR label")]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i, (key, name, col) in enumerate([("friendspersona", "FriendsPersona", "#0072B2"),
                                          ("cped", "CPED", "#D55E00")]):
        c = hp[key]
        mat = c["trait_specificity"]["matrix"]
        for j, (p, _) in enumerate(pairs):
            e = c["E_to_EXT_all_parsed"]["primary"] if p == "E_to_EXT" else mat[p]
            est, ci = e["estimate"], e["ci95"]
            ax.errorbar(est, j + (i - 0.5) * 0.3, xerr=[[est - ci[0]], [ci[1] - est]], fmt="o", ms=4,
                        capsize=2.5, lw=1, color=col, label=name if j == 0 else None)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_yticks(range(4), [n for _, n in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0.3, 0.7)
    ax.set_xlabel("AUC (95% bootstrap interval)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "Fig5")


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5()
