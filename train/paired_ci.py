# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Paired comparison of two models on the same clips: B minus A in WER points, with a 95 % bootstrap interval over clips.

    python paired_ci.py A_utts.jsonl B_utts.jsonl [--mode full|stream]

Reads either kind of per-clip output:
- the device engine's (esp32/host/eval_cli.py): "wav", "ref", "edits";
- eval_manifest.py's (*_utts.jsonl): "text", "hyp_full" / "hyp_stream", "speaker", "group" (--mode picks the hypothesis).
Prints all clips, then each group (dysarthric / control) and each speaker. For engine output without "speaker" and "group" fields, the
speaker comes from the clip's TORGO path and its name gives the group (FC.., MC.. are controls). Clips are resampled with replacement within the set being reported;
the same resample is applied to both models, so the interval is of the difference, not of either WER.
"""
import argparse, json, random, re

TORGO = re.compile(r"(?<![A-Z])([FM]C?0\d)(?![0-9])")


def load(path, mode):
    out = {}
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        if "edits" in r:  # engine output
            m = TORGO.search(r["wav"])
            spk = r.get("speaker") or (m.group(1) if m else "?")
            grp = r.get("group") or ("control" if "C" in spk else ("dysarthric" if spk != "?" else "?"))
            out[r["wav"]] = (r["edits"], len(r["ref"].split()), spk, grp)
        else:  # eval_manifest.py output
            import jiwer
            o = jiwer.process_words(r["text"], r["hyp_" + mode] or "<empty>")
            key = r.get("wav", r.get("audio", i))
            out[key] = (o.substitutions + o.deletions + o.insertions, len(r["text"].split()), r.get("speaker", "?"), r.get("group", "?"))
    return out


def wer(keys, d):
    return 100 * sum(d[k][0] for k in keys) / max(1, sum(d[k][1] for k in keys))


def ci(keys, a, b, n=1000):
    rng = random.Random(1)
    diffs = sorted(wer(s, b) - wer(s, a) for s in ([rng.choice(keys) for _ in keys] for _ in range(n)))
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--mode", default="full", help="eval_manifest.py output only: full or stream")
    args = ap.parse_args()
    a, b = load(args.a, args.mode), load(args.b, args.mode)
    keys = sorted(set(a) & set(b), key=str)
    print(f"{len(keys)} clips in both (A {len(a)}, B {len(b)})")
    sets = [("all", keys)]
    for field, label in ((3, "group"), (2, "speaker")):
        for v in sorted({a[k][field] for k in keys}):
            sets.append((f"{label} {v}", [k for k in keys if a[k][field] == v]))
    for name, ks in sets:
        lo, hi = ci(ks, a, b)
        print(f"  {name:20s} {len(ks):5d} clips  A {wer(ks, a):6.2f}  B {wer(ks, b):6.2f}  CI of B-A {lo:+.2f}..{hi:+.2f}")


if __name__ == "__main__":
    main()
