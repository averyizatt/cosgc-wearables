#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Manifests (one JSON line per clip) for the TORGO dysarthric speech database (see docs/DATA.md for its terms).

    python make_manifest_torgo.py G:\\cosgc\\data\\TORGO out_prefix --heldout F03,M05,FC03,MC04

Writes out_prefix_train.jsonl (all speakers except the held-out ones) and out_prefix_test.jsonl (the held-out speakers),
plus out_prefix_report.txt with what was kept and dropped and why.

TORGO layout: <speaker>/Session*/prompts/NNNN.txt holds the prompt for wav_headMic/NNNN.wav and wav_arrayMic/NNNN.wav.
Dropped: instructions in brackets ("[say Ah-P-Eee repeatedly]"), picture-description prompts (a path to an image, no
transcript), prompts with nothing left after cleaning, and audio files that cannot be read (some TORGO files are
damaged) or are shorter than 0.3 s. Each line records speaker, group (dysarthric / control), sex, mic and prompt kind
(word / sentence) so results can be split per speaker.
"""
import argparse, collections, glob, json, os, re
import soundfile as sf


def clean(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out_prefix")
    ap.add_argument("--heldout", default="F03,M05,FC03,MC04", help="speakers kept out of training, for testing")
    a = ap.parse_args()
    held = set(a.heldout.split(","))
    out = {"train": [], "test": []}
    why = collections.Counter()
    per_spk = collections.Counter()
    for spk in sorted(os.listdir(a.root)):
        sdir = os.path.join(a.root, spk)
        if not os.path.isdir(sdir):
            continue
        group = "control" if spk[1] == "C" else "dysarthric"
        sex = "female" if spk[0] == "F" else "male"
        for ptxt in sorted(glob.glob(os.path.join(sdir, "Session*", "prompts", "*.txt"))):
            raw = open(ptxt, encoding="utf-8", errors="replace").read().strip()
            if not raw or raw.startswith("["):
                why["instruction or empty prompt"] += 1
                continue
            if "/" in raw or "\\" in raw or raw.lower().endswith((".jpg", ".png")):
                why["picture description (no transcript)"] += 1
                continue
            text = clean(raw)
            if not text or text == "xxx":
                why["nothing left after cleaning"] += 1
                continue
            sess = os.path.dirname(os.path.dirname(ptxt))
            uid = os.path.splitext(os.path.basename(ptxt))[0]
            for mic in ("headMic", "arrayMic"):
                wav = os.path.join(sess, f"wav_{mic}", uid + ".wav")
                if not os.path.exists(wav):
                    why[f"no {mic} recording"] += 1
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
                split = "test" if spk in held else "train"
                out[split].append({"wav": os.path.abspath(wav), "text": text, "duration": round(dur, 3),
                                   "source": "torgo", "speaker": spk, "group": group, "sex": sex, "mic": mic,
                                   "kind": "word" if len(text.split()) == 1 else "sentence",
                                   "session": os.path.basename(sess), "sr": info.samplerate})
                per_spk[(spk, mic)] += 1
    lines = []
    for k, items in out.items():
        with open(f"{a.out_prefix}_{k}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        h = sum(i["duration"] for i in items) / 3600
        lines.append(f"{k}: {len(items)} clips, {h:.2f} h, speakers {sorted({i['speaker'] for i in items})}")
    lines.append("dropped: " + ", ".join(f"{n} {r}" for r, n in why.most_common()))
    lines.append("clips per speaker and mic: " + ", ".join(f"{s}/{m} {n}" for (s, m), n in sorted(per_spk.items())))
    srs = collections.Counter(i["sr"] for v in out.values() for i in v)
    lines.append(f"sample rates: {dict(srs)}")
    open(f"{a.out_prefix}_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
