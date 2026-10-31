#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""WER on the test clips whose transcript never occurs in training, from eval_cli.py's per-clip output.

    python3 overlap.py train.jsonl res_utts.jsonl [more_utts.jsonl ...]

TORGO's speakers read the same prompt lists, so most test transcripts also occur in training (docs/EXPERIMENTS.md,
threats to validity). This splits each result into clips with a seen and an unseen transcript.
"""
import json, re, sys


def norm(t):
    t = t.lower().replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t)).strip()


seen = {norm(json.loads(l)["text"]) for l in open(sys.argv[1], encoding="utf-8") if l.strip()}
for p in sys.argv[2:]:
    acc = {True: [0, 0, 0], False: [0, 0, 0]}
    for l in open(p):
        u = json.loads(l)
        v = acc[u["ref"] in seen]
        v[0] += u["edits"]; v[1] += len(u["ref"].split()); v[2] += 1
    print(p, " | ".join(f"{'seen' if k else 'unseen'} {100 * e / max(1, n):.1f} % ({c} clips)" for k, (e, n, c) in acc.items()))
