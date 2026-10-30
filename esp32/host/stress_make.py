#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Make degraded copies of 16 kHz mono PCM16 test clips, crude stand-ins for conditions the models were not trained on.

    python stress_make.py clips.json out_dir          # clips.json: [[wav, reference text], ...] as esp32/firmware/bench_clips/clips.json

Writes out_dir/<condition>__<clip>.wav and out_dir/refs.json ({file: reference}). Run stress_run.py on the result.

These are SIMULATIONS. They show how a model degrades when the speech is quiet, slow, fast, noisy, interrupted by pauses or
has repeated syllables, but real dysarthric or stuttered speech differs in ways none of these captures. Use them to compare
models and settings against each other, never as an estimate of the error rate for a person.
"""
import json, os, sys, wave
import numpy as np

SR = 16000


def read(path):
    with wave.open(path, "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR), path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)


def write(path, x):
    x = np.clip(np.round(x), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x.tobytes())


def gain_db(x, db):
    return x * 10 ** (db / 20)


def resample(x, factor):
    """factor > 1 speeds up (and raises pitch): a crude stand-in for fast / slow speech."""
    n = int(len(x) / factor)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def noise(x, snr_db, kind, rng):
    n = rng.standard_normal(len(x)).astype(np.float32)
    if kind == "pink":  # 1/f shaping in the frequency domain
        f = np.fft.rfft(n)
        f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
        n = np.fft.irfft(f, len(n)).astype(np.float32)
    ps = float(np.mean(x ** 2)) + 1e-9
    n *= np.sqrt(ps / 10 ** (snr_db / 10) / (float(np.mean(n ** 2)) + 1e-12))
    return x + n


def active_start(x, frac=0.02):
    """Index where the speech starts: first 10 ms block above 2 % of the peak envelope."""
    blk = 160
    e = np.array([np.sqrt(np.mean(x[i:i + blk] ** 2)) for i in range(0, len(x) - blk, blk)])
    th = frac * e.max()
    return int(np.argmax(e > th)) * blk


def stutter(x, seg_ms=120, reps=2):
    """Repeat the first syllable-sized piece of the speech `reps` extra times, as if the speaker had repeated it."""
    s = active_start(x)
    seg = x[s:s + int(seg_ms * SR / 1000)]
    gap = np.zeros(int(0.08 * SR), np.float32)
    out = [x[:s]]
    for _ in range(reps):
        out += [seg, gap]
    out.append(x[s:])
    return np.concatenate(out)


def pause(x, ms=500):
    """A mid-utterance pause (a block or a word-finding pause) at the quietest point near the middle."""
    blk = 160
    mid = len(x) // 2
    lo, hi = mid - int(0.8 * SR), mid + int(0.8 * SR)
    e = [(np.mean(x[i:i + blk] ** 2), i) for i in range(max(lo, 0), min(hi, len(x) - blk), blk)]
    i = min(e)[1]
    return np.concatenate([x[:i], np.zeros(int(ms * SR / 1000), np.float32), x[i:]])


def lowpass(x, cutoff=3400):
    """Telephone-like band limit: a stand-in for a thin, muffled microphone."""
    f = np.fft.rfft(x)
    f[int(cutoff / (SR / 2) * len(f)):] = 0
    return np.fft.irfft(f, len(x)).astype(np.float32)


CONDITIONS = [
    ("clean", lambda x, r: x),
    # turned down, then the microphone's own noise floor added (about 0.6 LSB rms)
    ("quiet-40dB", lambda x, r: gain_db(x, -40) + r.standard_normal(len(x)).astype(np.float32) * 0.6),
    ("quiet-50dB", lambda x, r: gain_db(x, -50) + r.standard_normal(len(x)).astype(np.float32) * 0.6),
    ("slow-0.8x", lambda x, r: resample(x, 0.8)),
    ("slow-0.65x", lambda x, r: resample(x, 0.65)),
    ("fast-1.25x", lambda x, r: resample(x, 1.25)),
    ("noise-20dB", lambda x, r: noise(x, 20, "white", r)),
    ("noise-10dB", lambda x, r: noise(x, 10, "pink", r)),
    ("pause-0.5s", lambda x, r: pause(x, 500)),
    ("pause-1.5s", lambda x, r: pause(x, 1500)),
    ("stutter-2x", lambda x, r: stutter(x, 120, 2)),
    ("stutter-4x", lambda x, r: stutter(x, 100, 4)),
    ("muffled-3.4k", lambda x, r: lowpass(x, 3400)),
]


def main(clips_json, out_dir):
    clips = json.load(open(clips_json))
    base = os.path.dirname(os.path.abspath(clips_json))
    root = os.path.abspath(os.path.join(base, "..", "..", ".."))  # clips.json holds repo-relative paths
    os.makedirs(out_dir, exist_ok=True)
    refs = {}
    for wav, ref in clips:
        path = wav if os.path.isabs(wav) else os.path.join(root, wav)
        x = read(path)
        name = os.path.splitext(os.path.basename(wav))[0]
        for cond, fn in CONDITIONS:
            rng = np.random.default_rng(sum(map(ord, cond + name)) * 7919)  # same noise on every run
            out = f"{cond}__{name}.wav"
            write(os.path.join(out_dir, out), fn(x, rng))
            refs[out] = ref
    json.dump(refs, open(os.path.join(out_dir, "refs.json"), "w"), indent=1)
    print(f"{len(refs)} files in {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
