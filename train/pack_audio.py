#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Pack a manifest's audio into one file so training reads a hard drive sequentially instead of seeking.

    python pack_audio.py G:\\cosgc\\data\\ls100_train.jsonl G:\\cosgc\\data\\packed\\ls100_train

writes <out>.pack (16 kHz mono int16 samples, all clips back to back) and <out>.jsonl (the input lines plus "pack",
"off" and "n": file, first sample and number of samples). Train with <out>.jsonl in place of the original manifest;
data.py reads packed items from the .pack file and everything else from "wav" as before.

Why: the trainer's batches hold clips of similar duration drawn from the whole set, so with one file per clip every
step opens dozens of files scattered over the disk; on a hard drive that keeps it 100 % busy at 1-2 MB/s. Here clips
are written sorted by duration, so a batch is mostly one contiguous range of the .pack file.

The pack is about 115 MB per hour of audio (LibriSpeech train-clean-100: about 11.5 GB). "wav" is kept, so evaluation
and the feature statistics still read the original files.
"""
import argparse, json, os
import numpy as np, soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out", help="output path without extension")
    a = ap.parse_args()
    lines = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    lines.sort(key=lambda d: d.get("duration", 0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pack = os.path.abspath(a.out + ".pack")
    off, bad = 0, 0
    with open(pack + ".tmp", "wb") as f, open(a.out + ".jsonl.tmp", "w", encoding="utf-8") as m:
        for i, d in enumerate(lines):
            try:
                x, sr = sf.read(d["wav"], dtype="int16", always_2d=True)
            except Exception as e:
                print("skipped", d["wav"], e)
                bad += 1
                continue
            x = x[:, 0]
            if sr != 16000:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(16000, sr)
                x = np.clip(resample_poly(x.astype(np.float32), 16000 // g, sr // g), -32768, 32767).astype(np.int16)
            f.write(x.astype("<i2").tobytes())
            m.write(json.dumps({**d, "pack": pack, "off": off, "n": len(x)}) + "\n")
            off += len(x)
            if i % 5000 == 0:
                print(f"{i}/{len(lines)}", flush=True)
    os.replace(pack + ".tmp", pack)
    os.replace(a.out + ".jsonl.tmp", a.out + ".jsonl")
    print(f"packed {len(lines) - bad} clips, {off / 16000 / 3600:.1f} h, {off * 2 / 1e9:.2f} GB -> {pack}; {bad} skipped")


if __name__ == "__main__":
    main()
