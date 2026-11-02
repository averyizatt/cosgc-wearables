#!/usr/bin/env python3
"""Room simulation with the board's measured noise and candidate noise reduction (E14). Windows python with numpy.
    python room2.py CLIPS_DIR SRC_WAV_DIR manifest.jsonl OUT_ROOT
The noise spectrum comes from the quietest 20 ms blocks of the board's raw recordings. Each clip is scaled to a speech level, then gets stationary
noise of that spectrum at -58 dBFS, then one treatment, then a fixed gain to the AGC's target (the settled AGC: E14 shows the level control barely matters).
Fixed before any scoring. A stationary noise with a known spectrum is the best case for noise reduction, so a loss here is a loss on the board too.
"""
import glob, json, math, os, sys, wave
import numpy as np

SR, TARGET, NOISE_DB, NFFT, HOP = 16000, 0.1, -58.0, 512, 256


def rd(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768


def wr(p, x):
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.clip(np.round(x * 32768), -32768, 32767).astype("<i2").tobytes())


def noise_psd(clips):  # mean power per FFT bin of the quietest blocks across the recordings
    P, win = [], np.hanning(NFFT)
    for p in sorted(glob.glob(os.path.join(clips, "*.wav"))):
        x = rd(p)
        fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::160]
        e = (fr ** 2).mean(1)
        P.append(np.abs(np.fft.rfft(fr[e <= np.quantile(e, 0.1)] * win, axis=1)) ** 2)
    return np.concatenate(P).mean(0)


def shaped(n, psd, rng):  # Gaussian noise with this spectrum, at NOISE_DB
    k = np.fft.rfftfreq(n, 1 / SR)
    S = np.interp(k, np.fft.rfftfreq(NFFT, 1 / SR), np.sqrt(psd))
    y = np.fft.irfft((rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k))) * S, n)
    return y * 10 ** (NOISE_DB / 20) / (np.sqrt((y ** 2).mean()) + 1e-12)


def voiced_rms(x):  # as quiet.sh: 20 ms blocks within 30 dB of the loudest
    h = 320
    e = np.array([(x[i:i + h] ** 2).mean() for i in range(0, max(1, len(x) - h + 1), h)])
    return math.sqrt(e[e > e.max() / 1000].mean()) if e.max() > 0 else 1e-6


def wiener(x, N, alpha, floor_db):  # spectral subtraction with the noise spectrum known, 32 ms frames, causal smoothing: a filter the board could run
    win = np.sqrt(np.hanning(NFFT + 1)[:-1])
    pad = np.concatenate((np.zeros(NFFT), x, np.zeros(NFFT)))
    out, norm = np.zeros(len(pad)), np.zeros(len(pad))
    floor, P, g = 10 ** (floor_db / 20), None, None
    for i in range(0, len(pad) - NFFT, HOP):
        X = np.fft.rfft(pad[i:i + NFFT] * win)
        p = np.abs(X) ** 2
        P = p if P is None else 0.6 * P + 0.4 * p
        G = np.maximum(floor, 1 - alpha * N / np.maximum(P, 1e-14))
        g = G if g is None else 0.5 * g + 0.5 * G
        out[i:i + NFFT] += np.fft.irfft(X * g, NFFT) * win
        norm[i:i + NFFT] += win ** 2
    return (out / np.maximum(norm, 1e-6))[NFFT:NFFT + len(x)]


def highpass(x, fc=200.0):  # second-order Butterworth, zero extra phase work: the board would run the same biquad
    w = math.tan(math.pi * fc / SR)
    q = math.sqrt(2)
    d = 1 + q * w + w * w
    b0, b1, b2 = 1 / d, -2 / d, 1 / d
    a1, a2 = 2 * (w * w - 1) / d, (1 - q * w + w * w) / d
    y, z1, z2 = np.zeros(len(x)), 0.0, 0.0
    for i, v in enumerate(x):
        o = b0 * v + z1
        z1 = b1 * v - a1 * o + z2
        z2 = b2 * v - a2 * o
        y[i] = o
    return y


def main():
    clips, src, man, root = sys.argv[1:5]
    stem = os.path.splitext(os.path.basename(man))[0]
    psd = noise_psd(clips)
    rng0 = np.random.default_rng(7)
    idle = shaped(SR * 2, psd, rng0)  # the board always has idle audio before a press to learn the noise from
    fr = np.lib.stride_tricks.sliding_window_view(idle, NFFT)[::HOP]
    N = (np.abs(np.fft.rfft(fr * np.sqrt(np.hanning(NFFT + 1)[:-1]), axis=1)) ** 2).mean(0)
    conds = [("near_base", -35, None), ("mid_base", -42, None), ("mid_nr1", -42, (1.0, -10.0)), ("mid_nr2", -42, (2.0, -20.0)), ("mid_hp", -42, "hp"),
             ("far_base", -50, None), ("far_nr1", -50, (1.0, -10.0)), ("far_nr2", -50, (2.0, -20.0)), ("far_hp", -50, "hp")]
    offs = [json.loads(l)["off"] for l in open(man, encoding="utf-8") if l.strip()]
    for c, _, _ in conds:
        os.makedirs(os.path.join(root, c), exist_ok=True)
    for k, off in enumerate(offs):
        name = f"{stem}_{off}.wav"
        x = rd(os.path.join(src, name))
        rng = np.random.default_rng(off)
        nz = shaped(len(x), psd, rng)  # same noise in every treatment of this clip
        v = max(voiced_rms(x), 1e-6)
        for c, level, t in conds:
            s = 10 ** (level / 20)
            y = x * s / v + nz
            if t == "hp": y = highpass(y)
            elif t: y = wiener(y, N, *t)
            wr(os.path.join(root, c, name), y * TARGET / s)
        if k % 100 == 0: print(k, len(offs), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
