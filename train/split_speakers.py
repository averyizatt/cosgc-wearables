# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Keep or drop whole speakers from a manifest (packed or not), and report what is left.

    python split_speakers.py in.jsonl out.jsonl --drop F03,M07     training without held-out speakers
    python split_speakers.py in.jsonl out.jsonl --keep F03,M07     a test set of only those speakers

Exits with an error if a named speaker is not in the input, so a typo cannot silently leave a test speaker in training.
"""
import argparse, collections, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--drop", default="")
    g.add_argument("--keep", default="")
    a = ap.parse_args()
    names = set(filter(None, (a.drop or a.keep).split(",")))
    rows = [l for l in open(a.inp, encoding="utf-8") if l.strip()]
    spk = [json.loads(l)["speaker"] for l in rows]
    missing = sorted(names - set(spk))
    if missing:
        sys.exit(f"not in {a.inp}: {missing}")
    keep = [(l, s) for l, s in zip(rows, spk) if (s in names) == bool(a.keep)]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(l if l.endswith("\n") else l + "\n" for l, _ in keep)
    secs, groups = collections.Counter(), collections.Counter()
    for l, s in keep:
        d = json.loads(l)
        secs[d.get("group", "?")] += d.get("duration", 0.0)
        groups[d.get("group", "?")] += 1
    print(f"{len(keep)} of {len(rows)} clips, {sum(secs.values()) / 3600:.2f} h, speakers {sorted({s for _, s in keep})}; "
          + ", ".join(f"{g} {groups[g]} clips {secs[g] / 3600:.2f} h" for g in sorted(groups)))


if __name__ == "__main__":
    main()
