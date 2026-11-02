#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
# WER on TORGO test by whether the prompt is in TORGO training: python score unseen.py big4,big6 (docs/EXPERIMENTS.md, E13). Reads eval_manifest.py results in G:\cosgcxpval.
import collections, json, sys
E = r"G:\cosgc\exp\eval"
train = {json.loads(l)["text"] for l in open(r"G:\cosgc\data\packed\torgo_train.jsonl", encoding="utf8")}
vocab = {w for t in train for w in t.split()}


def ed(r, h):
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[len(h)]


def wer(rows, mode):
    e = n = 0
    for r in rows:
        x = r["text"].split()
        e += ed(x, r["hyp_" + mode].split())
        n += len(x)
    return round(100 * e / max(1, n), 2), len(rows), n


names = sys.argv[1].split(",")
U = {n: [json.loads(l) for l in open(rf"{E}\{n}_torgo_test_utts.jsonl", encoding="utf8")] for n in names}
subs = {
    "all": lambda r: True,
    "seen prompt": lambda r: r["text"] in train,
    "unseen prompt": lambda r: r["text"] not in train,
    "unseen sentence": lambda r: r["text"] not in train and r["kind"] == "sentence",
    "unseen word": lambda r: r["text"] not in train and r["kind"] == "word",
}
grp = {"all": lambda r: True, "dys": lambda r: r["group"] == "dysarthric", "ctrl": lambda r: r["group"] == "control"}
for sn, sf in subs.items():
    for gn, gf in grp.items():
        print(sn, gn, *[f"{n} full {wer([r for r in U[n] if sf(r) and gf(r)], 'full')} stream {wer([r for r in U[n] if sf(r) and gf(r)], 'stream')[0]}" for n in names])
