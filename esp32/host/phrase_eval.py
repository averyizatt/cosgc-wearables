#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""How well does phrase mode work, and how often does it pick a phrase for speech that is not on the list?

    python3 phrase_eval.py split torgo_test.jsonl in_list.txt          # half of the test prompts go on the list (fixed by a hash)
    python3 ../tools/make_phrases.py in_list.txt phrases.tsv
    python3 eval_cli.py model.tnm torgo_test.jsonl --out res.json --lm ../../models/nemo_lm.tlm --phrases phrases.tsv
    python3 phrase_eval.py score res_utts.jsonl phrases.tsv [--sizes 10,25,50,100,0]

Clips whose prompt is on the list measure the right-pick rate; clips whose prompt is not on the list stand for speech the user did not
put on the list, and measure false picks. For a list of N phrases, each clip gets N - 1 other phrases drawn at random (plus its own when
it is on the list). A phrase is shown only when its score per frame is within a threshold of free decoding's; otherwise the free
text is shown. The threshold is set so that at most 5 % of off-list utterances are replaced by a phrase.
"""
import argparse, collections, hashlib, json, re
import numpy as np


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def split(a):
    prompts = sorted({norm(json.loads(l)["text"]) for l in open(a.manifest, encoding="utf-8") if l.strip()})
    on = [p for p in prompts if p and int(hashlib.md5(p.encode()).hexdigest(), 16) % 2 == 0]
    open(a.out, "w", encoding="utf-8", newline="\n").writelines(p + "\n" for p in on)
    print(len(on), "of", len(prompts), "prompts on the list")


def score(a):
    phr = [l.split("\t")[0] for l in open(a.phrases, encoding="utf-8") if l.strip()]
    idx = {p: i for i, p in enumerate(phr)}
    U = [json.loads(l) for l in open(a.utts, encoding="utf-8")]
    U = [u for u in U if "phr" in u and u["ref"]]
    rng = np.random.default_rng(0)
    P = len(phr)
    print(f"{len(U)} clips, {P} phrases on the full list; free decoding fully right on {100 * np.mean([u['edits'] == 0 for u in U]):.1f} %")
    for N in [int(s) for s in a.sizes.split(",")]:
        N = N or P
        gaps, right, onlist, grp, free_ok = [], [], [], [], []
        for u in U:
            s = np.array(u["phr"])
            own = idx.get(u["ref"])
            others = [k for k in range(P) if k != own]
            pick = list(rng.choice(others, min(N - (own is not None), len(others)), replace=False))
            if own is not None:
                pick.append(own)
            b = max(pick, key=lambda k: s[k])
            gaps.append((s[b] - u["phr_free"]) / max(1, u["phr_frames"]))  # how far the best phrase is from free decoding, per frame
            right.append(own is not None and b == own)
            onlist.append(own is not None)
            grp.append(u.get("group", "?"))
            free_ok.append(u["edits"] == 0)
        gaps, right, onlist, grp, free_ok = map(np.array, (gaps, right, onlist, grp, free_ok))
        thr = np.quantile(gaps[~onlist], 0.95)  # at most 5 % of off-list speech replaced by a phrase
        show = gaps > thr
        print(f"\nlist of {N} phrases; threshold {thr:.3f} per frame (5 % false picks on off-list speech)")
        for g in ["all"] + sorted(set(grp)):
            m = onlist & ((grp == g) if g != "all" else True)
            if not m.any():
                continue
            print(f"  on-list {g:10s} best phrase right {100 * right[m].mean():5.1f} % | shown {100 * show[m].mean():5.1f} %, "
                  f"of which right {100 * right[m & show].mean() if (m & show).any() else 0:5.1f} % | "
                  f"final screen right {100 * np.where(show, right, free_ok)[m].mean():5.1f} % (free decoding alone {100 * free_ok[m].mean():5.1f} %)")
        mo = ~onlist
        print(f"  off-list: phrase shown {100 * show[mo].mean():.1f} % (always wrong then); free decoding right {100 * free_ok[mo].mean():.1f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("manifest"); s.add_argument("out")
    c = sub.add_parser("score"); c.add_argument("utts"); c.add_argument("phrases"); c.add_argument("--sizes", default="10,25,50,100,0")
    a = ap.parse_args()
    split(a) if a.cmd == "split" else score(a)
