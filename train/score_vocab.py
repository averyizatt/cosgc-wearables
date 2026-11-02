#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
# UASpeech WER by whether the word is in TORGO training, and prompt overlap counts: python score vocab.py big4,big6 (docs/EXPERIMENTS.md, E13). Reads eval_manifest.py results in G:\cosgcxpval.
import json, sys
E = r"G:\cosgc\exp\eval"
train = [json.loads(l) for l in open(r"G:\cosgc\data\packed\torgo_train.jsonl", encoding="utf8")]
dys = [r for r in train if r["group"] == "dysarthric"]
V_all = {w for r in train for w in r["text"].split()}
V_dys = {w for r in dys for w in r["text"].split()}
test = [json.loads(l) for l in open(r"G:\cosgc\data\torgo_test.jsonl", encoding="utf8")]
P_test = {r["text"] for r in test}
P_train = {r["text"] for r in train}
print("TORGO train prompts", len(P_train), "clips", len(train), "| test prompts", len(P_test), "| train clips whose prompt is in test", sum(r["text"] in P_test for r in train), "| train hours with such prompts", round(sum(float(r["duration"]) for r in train if r["text"] in P_test) / 3600, 2), "of", round(sum(float(r["duration"]) for r in train) / 3600, 2))
print("train vocab", len(V_all), "dysarthric-train vocab", len(V_dys))


def wer(rows, mode="full"):
    e = n = 0
    for r in rows:
        x, h = r["text"].split(), r["hyp_" + mode].split()
        d = list(range(len(h) + 1))
        for i, a in enumerate(x, 1):
            p, d[0] = d[0], i
            for j, b in enumerate(h, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
        e += d[len(h)]
        n += len(x)
    return round(100 * e / max(1, n), 2), n


names = sys.argv[1].split(",")
U = {n: [json.loads(l) for l in open(rf"{E}\{n}_uaspeech_test_b2_utts.jsonl", encoding="utf8")] for n in names}
subs = {
    "all": lambda r: True,
    "in TORGO train vocab": lambda r: all(w in V_all for w in r["text"].split()),
    "not in TORGO train vocab": lambda r: not all(w in V_all for w in r["text"].split()),
}
for sn, sf in subs.items():
    for g in ("control", "dysarthric"):
        for m in ("full", "stream"):
            print("UASpeech", sn, g, m, *[f"{n} {wer([r for r in U[n] if sf(r) and r['group'] == g], m)}" for n in names])
