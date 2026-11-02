#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Figures for the paper (docs/PAPER.md). Every number is copied from the result named beside it; change it there first.

    python3 make_figures.py            # writes fig_*.png and fig_*.svg next to this file

Colors: the dataviz reference palette (slots 1-2 categorical, blue ramp for ordered series), light print surface.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal: lightest step that clears 2:1 on light, to darkest

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220)
    plt.close(fig)
    print("wrote", name)


def ladder():
    """Figure 2: TORGO test WER, device engine (int8 + LM), one change per step. PAPER.md table 1."""
    steps = ["shipped", "fine-tune\n(TORGO+LS)", "24 layers", "export\ngrid", "full-context\ntraining", "26 %\ndysarthric", "+ simulated\n(big7)"]
    live_all = {0: 37.37, 1: 30.87, 2: 26.23, 3: 25.14, 5: 23.86, 6: 23.31}  # E8, E10, E13, E15 (step 0 without LM)
    live_dys = {0: 68.4, 1: 59.6, 2: 52.9, 3: 51.17, 5: 46.92, 6: 45.99}
    fin_all = {3: 19.75, 4: 19.51, 5: 17.74, 6: 17.03}                       # E12, E13, E15
    fin_dys = {3: 42.15, 4: 41.55, 5: 36.56, 6: 35.43}
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for d, c, ls, mk, lab in ((live_dys, ORANGE, "--", "s", "dysarthric, live text"), (fin_dys, ORANGE, "-", "o", "dysarthric, final text"),
                              (live_all, BLUE, "--", "s", "all speakers, live text"), (fin_all, BLUE, "-", "o", "all speakers, final text")):
        x, y = list(d), list(d.values())
        # no line across a step that was not measured: break it into runs of consecutive steps
        runs, cur = [], [x[0]]
        for a, b in zip(x, x[1:]):
            if b == a + 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        for k, r in enumerate(runs):
            ax.plot(r, [d[i] for i in r], ls=ls, color=c, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5, label=lab if k == 0 else None)
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points", va="center", color=INK2, fontsize=8)
    ax.annotate(f"{live_dys[0]:.1f}", (0, live_dys[0]), xytext=(-7, 0), textcoords="offset points", va="center", ha="right", color=INK2, fontsize=8)
    ax.annotate(f"{live_all[0]:.1f}", (0, live_all[0]), xytext=(-7, 0), textcoords="offset points", va="center", ha="right", color=INK2, fontsize=8)
    ax.set_xticks(range(len(steps)), steps, fontsize=7.5)
    ax.set_xlim(-0.6, len(steps) - 0.3)
    ax.set_ylim(0, 75)
    ax.set_ylabel("word error rate, %")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2, handlelength=3)
    ax.set_title("TORGO test (unseen speakers), on-device engine", fontsize=9, loc="left", color=INK)
    ax.annotate("live text\nnot measured", (4, 24.5), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
    save(fig, "fig_ladder")


def confidence():
    """Figure 4: confidence levels, big7 + LM, TORGO test, held-out half of the clips. E17."""
    lev = ["sure", "unsure", "probably wrong"]
    share = [53.6, 22.7, 23.7]
    right = [94.6, 65.0, 26.7]
    wer = [3.9, 14.5, 49.3]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    x = range(3)
    ax.bar(x, right, width=0.5, color=BLUE)
    for i in x:
        ax.annotate(f"{right[i]:.0f} %", (i, right[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x), [f"{l}\n{s:.0f} % of clips\nWER {w:.0f} %" for l, s, w in zip(lev, share, wer)], fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("transcripts fully right, %")
    ax.grid(axis="x", visible=False)
    ax.set_title("Confidence mark vs. correctness", fontsize=9, loc="left")
    save(fig, "fig_confidence")


def simulator():
    """Figure 3: how hard simulated speech is for the model (big6 + LM, full context, 1947 simulated test-clean clips), by calibration round,
    against the real test speakers with the same model. E15 (rounds 1-3); real speakers: big6_q121_full_lm, per speaker."""
    sev = ["mild", "moderate", "severe"]
    rounds = {"round 1 (hand-set)": [5.8, 10.7, 16.9], "round 2 (pauses, tilt)": [7.1, 13.6, 21.0], "round 3 (+ sound errors)": [8.3, 16.3, 24.1]}
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for (lab, y), c in zip(rounds.items(), RAMP):
        ax.plot(range(3), y, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=lab)
    ax.annotate(f"{rounds['round 3 (+ sound errors)'][-1]:.1f}", (2, 24.1), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    for v, lab in ((29.77, "real moderate speaker (F03)"), (49.40, "real severe speaker (M05)")):
        ax.axhline(v, color=INK2, lw=1, ls=":")
        ax.annotate(f"{lab}: {v:.1f}", (0, v), xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), [f"{s}" for s in sev])
    ax.set_xlabel("simulated severity")
    ax.set_xlim(-0.2, 2.4)
    ax.set_ylim(0, 55)
    ax.set_ylabel("word error rate, %")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 0.86))
    ax.set_title("Simulated vs. real dysarthric speech", fontsize=9, loc="left")
    save(fig, "fig_simulator")


def phrases():
    """Figure 5: phrase mode, big7 + LM, TORGO test, share of on-list utterances fully right on the screen, by list size. E18."""
    size = [10, 25, 50, 100, 398]
    dys = [88.5, 83.7, 79.0, 73.3, 58.3]
    ctl = [97.1, 96.2, 95.3, 94.3, 89.3]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for y, c, lab, free in ((ctl, BLUE, "control speakers", 83.5), (dys, ORANGE, "dysarthric speakers", 49.7)):
        ax.plot(range(5), y, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=lab)
        ax.axhline(free, color=c, lw=1, ls=":")
        ax.annotate(f"free decoding: {free:.0f} %", (4.2, free), xytext=(0, 3), textcoords="offset points", va="bottom", fontsize=7, color=INK2)
        ax.annotate(f"{y[0]:.0f}", (0, y[0]), xytext=(-7, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(5), [str(s) for s in size])
    ax.set_xlabel("phrases on the list")
    ax.set_xlim(-0.5, 5.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("utterances fully right, %")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("Phrase mode (5 % of off-list speech wrongly matched)", fontsize=9, loc="left")
    save(fig, "fig_phrases")


if __name__ == "__main__":
    ladder()
    confidence()
    simulator()
    phrases()
