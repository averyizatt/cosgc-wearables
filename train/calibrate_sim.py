# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""How hard is simulated speech compared with real dysarthric speech, for a model that has heard neither?

    python calibrate_sim.py REAL_utts.jsonl SIM_utts.jsonl [SIM2_utts.jsonl ...] [--mode full] [--conditions all]

Both files are eval_manifest.py outputs (*_utts.jsonl) from the SAME model, one that has not been trained on dysarthric speech (base24x, the
starting point of every big run), so neither side is helped by memorized speakers or prompts. REAL: TORGO's dysarthric TRAINING speakers (never
the test speakers F03 and M05, which only score finished models). SIM: a dysim.py validation set (made from LibriSpeech test-clean, which no
model trains on).

Prints each real speaker's WER, the simulated WER by severity and by condition, the severity at which the simulation is as hard as each real
speaker, and a severity range for training that spans the real speakers. Note the two sides read different material (TORGO: single words and
short prompts; LibriSpeech: long read sentences), so equal WER means equally hard for this model, not identical speech.

The severity curve uses only the dysarthria conditions by default: TORGO's speakers are dysarthric, and averaging in the voice conditions
(whisper, electrolarynx, dysphonia, esophageal, 4-15 % WER at any severity) and stutter and apraxia flattened the curve, which made the
simulation look far too mild (E16, Finding 43). --conditions all restores the old behaviour.
"""
import argparse, collections, json
import numpy as np
import jiwer

DYSARTHRIA = "flaccid,spastic,ataxic,hypokinetic,hyperkinetic,uumn,als,cp"  # as in dysim.py


def wer(rows, mode):
    return 100 * jiwer.wer([r["text"] for r in rows], [r["hyp_" + mode] or "<empty>" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real")
    ap.add_argument("sim", nargs="+")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--kind", default="", help="only real clips of this kind (word or sentence)")
    ap.add_argument("--conditions", default=DYSARTHRIA, help="simulated conditions in the severity curve (comma list, or 'all')")
    a = ap.parse_args()
    real = [json.loads(l) for l in open(a.real, encoding="utf-8")]
    real = [r for r in real if r.get("group") == "dysarthric" and (not a.kind or r.get("kind") == a.kind)]
    sim_all = [json.loads(l) for f in a.sim for l in open(f, encoding="utf-8")]
    keep = None if a.conditions == "all" else set(a.conditions.split(","))
    sim = [r for r in sim_all if keep is None or r.get("condition") in keep]
    spk = collections.defaultdict(list)
    for r in real:
        spk[r["speaker"]].append(r)
    print(f"real dysarthric training speakers ({a.mode}):")
    real_w = {}
    for s, rs in sorted(spk.items()):
        real_w[s] = wer(rs, a.mode)
        print(f"  {s:6s} {real_w[s]:6.1f} %  ({len(rs)} clips)")

    top = max([r.get("severity", 0) for r in sim] + [0.2])
    bins = np.arange(0.0, top + 0.2001, 0.2)
    curve = []
    print(f"simulated, by severity ({'all conditions' if keep is None else 'conditions: ' + a.conditions}):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        rs = [r for r in sim if lo <= r.get("severity", -1) < hi]
        if len(rs) >= 20:
            w = wer(rs, a.mode)
            curve.append(((lo + hi) / 2, w))
            print(f"  {lo:.1f}-{hi:.1f} {w:6.1f} %  ({len(rs)} clips)")
    print("simulated, by condition (all conditions):")
    by_c = collections.defaultdict(list)
    for r in sim_all:
        by_c[r.get("condition", "?")].append(r)
    for c, rs in sorted(by_c.items(), key=lambda kv: -wer(kv[1], a.mode)):
        print(f"  {c:14s} {wer(rs, a.mode):6.1f} %  ({len(rs)} clips)")

    if len(curve) >= 2:
        sev, w = np.array(curve).T
        w = np.maximum.accumulate(w)  # WER should rise with severity; flatten noise
        print("severity at which the simulation is as hard as each real speaker:")
        need = {}
        for s, rw in sorted(real_w.items()):
            if rw > w[-1]:
                need[s] = None
                print(f"  {s:6s} {rw:6.1f} %  beyond the hardest simulated bin ({w[-1]:.1f} %): the simulation is too mild for this speaker")
            else:
                need[s] = float(np.interp(rw, w, sev))
                print(f"  {s:6s} {rw:6.1f} %  severity {need[s]:.2f}")
        got = [v for v in need.values() if v is not None]
        if got:
            print(f"suggested --severity {max(0.1, min(got) - 0.1):.1f},{max(got) + 0.1:.1f}"
                  + ("  (and stronger traits: some speakers are beyond reach)" if None in need.values() else ""))


if __name__ == "__main__":
    main()
