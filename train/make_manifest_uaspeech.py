#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Manifest (one JSON line per clip) for the UASpeech dysarthric speech database (see docs/DATA.md for its terms).

    python make_manifest_uaspeech.py G:\\cosgc\\data\\UASpeech out.jsonl [--version noisereduce] [--mics M5]

The download unpacks into UASpeech_<version>_<C|FM>/UASpeech/ folders; audio is under audio/<version>/<speaker>/ and
the words under mlf/<speaker>/*.mlf (HTK label files: a quoted "*/<file>.lab" line, then the word, then "."). Files are
named <speaker>_<block>_<word id>_<mic>.wav. Word ids: D1-D10 digit, LA-LZ letter (radio alphabet), C = command,
CW = common word, UW = uncommon word (different in each block).

The 7 array microphones record the same utterance, so by default only one is used (--mics M5, roughly central;
a comma list keeps several). Each line records speaker, group (dysarthric / control), sex, block, word kind and mic.
Clips that cannot be read or are shorter than 0.3 s are dropped and counted in out_report.txt.
"""
import argparse, collections, glob, json, os, re
import soundfile as sf

KINDS = {"D": "digit", "L": "letter", "C": "command", "CW": "common", "UW": "uncommon"}


def read_mlf(paths):
    """Word per <speaker>_<block>_<word id> from the *_word.mlf files. The same utterance is listed under several of the
    7 microphones (which all recorded it) and every release folder has a copy, so the first entry is kept; the
    *_aligned_phones.mlf files are skipped (phone lines). Concatenating repeated entries once gave "word word ... word"
    (10 times) as the transcript of a one-word clip (docs/EXPERIMENTS.md, E7)."""
    words = {}
    for p in paths:
        if not os.path.basename(p).endswith("_word.mlf"):
            continue
        key, entry = None, []
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.startswith('"'):
                key, entry = os.path.splitext(os.path.basename(line.strip('"')))[0].rsplit("_", 1)[0], []
            elif line == "." or line.startswith("#!"):
                if key and entry:
                    words.setdefault(key, " ".join(entry))
                key = None
            elif key and line:
                entry.append(line)
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--version", default="noisereduce", choices=["original", "normalized", "noisereduce"])
    ap.add_argument("--mics", default="M5")
    ap.add_argument("--blocks", default="", help="e.g. B1,B3 (the usual training blocks) or B2 (test); default all")
    a = ap.parse_args()
    mics = set(a.mics.split(","))
    a.blocks = set(a.blocks.split(",")) - {""}
    dirs = sorted(glob.glob(os.path.join(a.root, f"UASpeech_{a.version}_*", "UASpeech")))
    assert dirs, f"no UASpeech_{a.version}_* folders under {a.root}"
    words = read_mlf([p for d in dirs for p in glob.glob(os.path.join(d, "mlf", "**", "*.mlf"), recursive=True)
                      if not os.path.basename(p).startswith("._")])
    # every speaker reads the same word for a given block and word id; speakers without .mlf files (CF05, CM01, F05 in
    # the 2020 release) get their words from this table, built from the others (an id is used only if at least 90 % of
    # the speakers agree on its word)
    by_id = collections.defaultdict(collections.Counter)
    for key, w in words.items():
        p = key.split("_")
        if len(p) == 3:
            by_id[(p[1], p[2])][w.lower()] += 1
    table = {k: c.most_common(1)[0][0] for k, c in by_id.items() if c.most_common(1)[0][1] >= 0.9 * sum(c.values())}
    print(f"word table: {len(table)} of {len(by_id)} (block, word id) pairs agree across speakers")
    why, per_spk, out, from_table = collections.Counter(), collections.Counter(), [], collections.Counter()
    for d in dirs:
        for wav in sorted(glob.glob(os.path.join(d, "audio", a.version, "*", "*.wav"))):
            name = os.path.splitext(os.path.basename(wav))[0]
            if name.startswith("._"):
                continue
            parts = name.split("_")
            if len(parts) != 4:
                why["unexpected file name"] += 1
                continue
            spk, block, wid, mic = parts
            if mic not in mics or (a.blocks and block not in a.blocks):
                continue
            w = words.get(f"{spk}_{block}_{wid}")
            if not w and (block, wid) in table:
                w = table[(block, wid)]
                from_table[spk] += 1
            text = re.sub(r"[^a-z' ]", " ", (w or "").lower()).strip()
            if not text:
                why["no transcript"] += 1
                continue
            try:
                info = sf.info(wav)
            except Exception:
                why["unreadable audio file"] += 1
                continue
            dur = info.frames / info.samplerate
            if dur < 0.3:
                why["shorter than 0.3 s"] += 1
                continue
            out.append({"wav": os.path.abspath(wav), "text": text, "duration": round(dur, 3), "source": "uaspeech",
                        "speaker": spk, "group": "control" if spk.startswith("C") else "dysarthric",
                        "sex": "female" if spk.lstrip("C").startswith("F") else "male", "block": block,
                        "kind": "letter" if re.fullmatch(r"L[A-Z]", wid) else KINDS.get(re.sub(r"\d", "", wid), "?"),
                        "mic": mic, "sr": info.samplerate})
            per_spk[spk] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    h = sum(i["duration"] for i in out) / 3600
    lines = [f"{len(out)} clips, {h:.2f} h, version {a.version}, mics {sorted(mics)}",
             "dropped: " + (", ".join(f"{n} {r}" for r, n in why.most_common()) or "none"),
             "words from the shared table (no .mlf for the speaker): " + (", ".join(f"{s} {n}" for s, n in sorted(from_table.items())) or "none"),
             "clips per speaker: " + ", ".join(f"{s} {n}" for s, n in sorted(per_spk.items())),
             f"sample rates: {dict(collections.Counter(i['sr'] for i in out))}",
             f"kinds: {dict(collections.Counter(i['kind'] for i in out))}",
             f"words per transcript: {dict(collections.Counter(len(i['text'].split()) for i in out))} (UASpeech: 1)"]
    open(os.path.splitext(a.out)[0] + "_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
