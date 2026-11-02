#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Does the engine's confidence (tasr_nemo_last_confidence) tell right transcripts from wrong ones, and where should the screen's levels go?

    python3 conf_calib.py big7_q121_full_lm_conf_utts.jsonl

Input: eval_cli.py utterance file with conf_mean, conf_min (log posteriors of the emitted tokens) and edits. Prints, for each measure, the
area under the curve for "this transcript has no error", then three levels and what they mean on this data: share of clips, share fully
right, word error rate. Thresholds are chosen on every other clip and reported on the rest, so the levels are not judged on the clips
they were fitted to (the same speakers are on both sides: TORGO test has only four).
"""
import argparse, json
import numpy as np


def auc(score, good):
    order = np.argsort(score)
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    n1, n0 = good.sum(), (~good).sum()
    return (r[good].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("utts")
    ap.add_argument("--hi", type=float, default=0.95, help="'sure' level: at least this share of its clips fully right on the fitting half")
    ap.add_argument("--lo", type=float, default=0.5, help="'probably wrong' level: at most this share fully right")
    a = ap.parse_args()
    U = [json.loads(l) for l in open(a.utts, encoding="utf-8")]
    U = [u for u in U if "conf_mean" in u and u["ref"]]
    good = np.array([u["edits"] == 0 for u in U])
    words = np.array([len(u["ref"].split()) for u in U])
    edits = np.array([u["edits"] for u in U])
    none = np.array([u.get("tokens", 1) == 0 for u in U])  # nothing emitted: no confidence at all, the lowest score (as in mic.c)
    meas = {"mean": np.where(none, -99.0, [u["conf_mean"] for u in U]), "min": np.where(none, -99.0, [u["conf_min"] for u in U])}
    meas["mix"] = 0.5 * meas["mean"] + 0.5 * meas["min"]
    print(f"{len(U)} clips, {100 * good.mean():.1f} % fully right, WER {100 * edits.sum() / words.sum():.2f} %")
    for k, v in meas.items():
        print(f"  AUC of '{k}' for a fully right transcript: {auc(v, good):.3f}")
    fit = np.arange(len(U)) % 2 == 0
    best = max(meas, key=lambda k: auc(meas[k][fit], good[fit]))
    v = meas[best]
    # 'sure': the lowest threshold whose clips above it are at least hi fully right; 'probably wrong': the highest below which at most lo
    qs = np.quantile(v[fit], np.linspace(0.02, 0.98, 97))
    t_hi = next((q for q in qs if good[fit & (v >= q)].mean() >= a.hi), qs[-1])
    t_lo = next((q for q in qs[::-1] if (fit & (v < q)).any() and good[fit & (v < q)].mean() <= a.lo), qs[0])
    t_lo = min(t_lo, t_hi)
    print(f"measure '{best}', thresholds fitted on {fit.sum()} clips: sure >= {t_hi:.3f}, probably wrong < {t_lo:.3f}")
    for name, m in (("held-out clips", ~fit), ("all clips", np.ones(len(U), bool))):
        print(f" {name}:")
        for lvl, sel in (("sure", v >= t_hi), ("unsure", (v < t_hi) & (v >= t_lo)), ("probably wrong", v < t_lo)):
            s = sel & m
            if s.any():
                print(f"   {lvl:15s} {100 * s.sum() / m.sum():5.1f} % of clips, {100 * good[s].mean():5.1f} % fully right, WER {100 * edits[s].sum() / max(1, words[s].sum()):5.1f} %")
    grp = np.array([u.get("group", "?") for u in U])
    for g in sorted(set(grp)):
        s = grp == g
        print(f" {g}: sure {100 * (s & (v >= t_hi)).sum() / s.sum():.0f} %, probably wrong {100 * (s & (v < t_lo)).sum() / s.sum():.0f} % of clips")
    print(json.dumps({"measure": best, "sure": round(float(t_hi), 4), "wrong": round(float(t_lo), 4)}))


if __name__ == "__main__":
    main()
