#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Write a training manifest (one JSON line per clip: wav, text, duration, source) for a LibriSpeech split.

    python make_manifest_librispeech.py <LibriSpeech/dev-clean> out_prefix [--val_speakers 4]

Writes out_prefix_train.jsonl and, with --val_speakers N, out_prefix_val.jsonl holding every clip of N speakers that are
not in the training file (so validation measures new voices). Audio stays FLAC; data.py reads it with soundfile.
"""
import argparse, glob, json, os
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out_prefix")
    ap.add_argument("--val_speakers", type=int, default=0)
    ap.add_argument("--source", default="librispeech")
    a = ap.parse_args()
    speakers = sorted(os.listdir(a.root), key=int)
    val_spk = set(speakers[:a.val_speakers])
    out = {"train": [], "val": []}
    for txt in sorted(glob.glob(os.path.join(a.root, "*", "*", "*.trans.txt"))):
        spk = os.path.basename(os.path.dirname(os.path.dirname(txt)))
        for line in open(txt, encoding="utf-8"):
            uid, text = line.strip().split(" ", 1)
            wav = os.path.join(os.path.dirname(txt), uid + ".flac")
            info = sf.info(wav)
            out["val" if spk in val_spk else "train"].append(
                {"wav": os.path.abspath(wav), "text": text.lower(), "duration": round(info.frames / info.samplerate, 3),
                 "source": a.source, "speaker": spk})
    for k, items in out.items():
        if not items:
            continue
        with open(f"{a.out_prefix}_{k}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"{a.out_prefix}_{k}.jsonl: {len(items)} clips, {sum(i['duration'] for i in items) / 3600:.2f} h")


if __name__ == "__main__":
    main()
