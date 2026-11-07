#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Pool `eval_cli.py --grid` results over several dev sets and pick an LM weight and token bonus (docs/EXPERIMENTS.md, E25).

    python3 lm_grid.py ~/lmtest/e25_foldB_big7_full ~/lmtest/e25_foldC_big7_full [--keep 0.3:0.5 --margin 0.1]

Each prefix is the --out of one eval_cli.py --grid run without ".json"; its <prefix>_w<w>_b<b>_utts.jsonl files are read. WER is
corpus-level over all clips of all prefixes (edits summed / reference words summed). The chosen setting is the lowest; --keep stays
if it is within --margin points of the lowest. Also prints the dysarthric and control rows of the chosen and kept settings.
Standard library only.
"""
import argparse, collections, glob, json, os, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--keep", default="0.3:0.5")
    ap.add_argument("--margin", type=float, default=0.1)
    a = ap.parse_args()
    tot = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))  # (w, b) -> group -> edits, words, clips
    seen = collections.defaultdict(set)
    for p in a.prefixes:
        for f in glob.glob(glob.escape(p) + "_w*_b*_utts.jsonl"):
            m = re.search(r"_w([-0-9.]+)_b([-0-9.]+)_utts\.jsonl$", f)
            key = (float(m.group(1)), float(m.group(2)))
            seen[key].add(p)
            for line in open(f, encoding="utf-8"):
                u = json.loads(line)
                for g in ("all", u.get("group", "?"), "spk/" + u.get("speaker", "?")):
                    v = tot[key][g]
                    v[0] += u["edits"]; v[1] += len(u["ref"].split()); v[2] += 1
    keys = sorted(k for k in tot if len(seen[k]) == len(a.prefixes))
    if not keys:
        raise SystemExit("no setting is present for every prefix")
    wer = lambda k, g="all": 100 * tot[k][g][0] / max(1, tot[k][g][1])
    ws, bs = sorted({k[0] for k in keys}), sorted({k[1] for k in keys})
    print(f"{tot[keys[0]]['all'][2]} clips over {len(a.prefixes)} dev sets; WER, all clips\n")
    print("| weight \\ bonus | " + " | ".join(f"{b:g}" for b in bs) + " |")
    print("|---|" + "---|" * len(bs))
    for w in ws:
        print(f"| {w:g} | " + " | ".join(f"{wer((w, b)):.2f}" if (w, b) in tot else "" for b in bs) + " |")
    best = min(keys, key=wer)
    keep = tuple(map(float, a.keep.split(":")))
    choice = keep if keep in tot and wer(keep) - wer(best) <= a.margin else best
    print(f"\nlowest: weight {best[0]:g} bonus {best[1]:g}, {wer(best):.2f}")
    if keep in tot:
        print(f"kept setting {keep[0]:g}/{keep[1]:g}: {wer(keep):.2f} ({wer(keep) - wer(best):+.2f}; margin {a.margin})")
    print(f"CHOSEN: weight {choice[0]:g} bonus {choice[1]:g}")
    for k in dict.fromkeys([choice, keep, best]):
        if k in tot:
            spk = " ".join(f"{g[4:]} {wer(k, g):.2f}" for g in sorted(tot[k]) if g.startswith("spk/"))
            print(f"  {k[0]:g}/{k[1]:g}: all {wer(k):.2f}, dysarthric {wer(k, 'dysarthric'):.2f}, control {wer(k, 'control'):.2f}; {spk}")


if __name__ == "__main__":
    main()
