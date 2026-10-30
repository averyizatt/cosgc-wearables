# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Simulated dysarthria: distort typical-speech features toward common dysarthric traits (training augmentation).

Works on the log-mel features inside the trainer (GPU), so it adds no load to the CPU data loaders. Each selected
utterance gets a random subset of:
  - slow, uneven rate: the utterance is cut into 15-50 frame pieces (150-500 ms), each stretched by 1.0-1.8x
    (pitch unchanged, unlike speed perturbation); dysarthric speech is typically slow with irregular timing;
  - slurred transitions: smoothing over time (30-90 ms), so sounds blend into each other (imprecise articulation);
  - blurred spectrum: smoothing across neighbouring mel bands (less distinct formants, imprecise vowels/consonants);
  - tremor: 3-8 Hz loudness modulation of 10-40 % (vocal tremor, unsteady breath support);
  - breathiness: noise that follows the speech envelope, tilted toward high frequencies, at 10-25 dB below the speech.
Not modeled: hypernasality, pitch breaks, monotone pitch, slow syllable repetitions. This is a crude approximation, and
whether it helps on real dysarthric speakers is an experiment, not an assumption (see docs/EXPERIMENTS.md).

Usage (in the training loop, after feature extraction, before SpecAugment):
    f, fl = simulate(f, fl, sel, mean, std)    # sel: bool (B,) which utterances to distort
"""
import math, random
import torch
import torch.nn.functional as F


def _stretch(x, rng):
    """x (T, 80) -> (T', 80), T' >= T: piecewise linear-interpolation time stretch."""
    T = x.shape[0]
    pos, s = [], 0
    while s < T:
        n = min(rng.randint(15, 50), T - s)
        m = max(n, round(n * rng.uniform(1.0, 1.8)))
        pos.append(torch.linspace(s, s + n - 1, m, device=x.device))
        s += n
    p = torch.cat(pos)
    i0 = p.floor().long().clamp(max=T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (p - i0.float())[:, None]
    return x[i0] * (1 - w) + x[i1] * w


def _smooth(x, width, dim):
    """Moving average of odd width along dim 0 (time) or 1 (mel) of x (T, 80), edges replicated."""
    k = torch.ones(1, 1, width, device=x.device) / width
    y = x.T[:, None] if dim == 0 else x[:, None]  # rows to filter, (N, 1, L)
    y = F.conv1d(F.pad(y, (width // 2, width // 2), mode="replicate"), k)[:, 0]
    return y.T if dim == 0 else y


def distort(mel, rng):
    """mel: (T, 80) natural-log mel power of one utterance -> distorted, possibly longer."""
    x = mel
    if rng.random() < 0.7:
        x = _stretch(x, rng)
    if rng.random() < 0.5:
        a = rng.uniform(0.3, 0.8)
        x = (1 - a) * x + a * _smooth(x, rng.choice((3, 5, 7, 9)), 0)
    if rng.random() < 0.5:
        a = rng.uniform(0.3, 0.8)
        x = (1 - a) * x + a * _smooth(x, 3, 1)
    if rng.random() < 0.4:
        t = torch.arange(x.shape[0], device=x.device).float() / 100.0  # 10 ms frames
        d, fm = rng.uniform(0.1, 0.4), rng.uniform(3.0, 8.0)
        x = x + 2 * torch.log1p(d * torch.sin(2 * math.pi * fm * t + rng.uniform(0, 6.283)))[:, None]
    if rng.random() < 0.4:
        p = x.exp()
        env = _smooth(p.mean(1, keepdim=True), 5, 0)  # speech power per frame, smoothed over 50 ms
        tilt = torch.linspace(0.3, 1.7, x.shape[1], device=x.device)[None]  # more noise in high bands
        snr = rng.uniform(10.0, 25.0)
        noise = env * 10 ** (-snr / 10) * tilt * torch.empty_like(p).exponential_()
        x = torch.log(p + noise)
    return x


def simulate(f, fl, sel, mean, std, rng=random):
    """f (B, T, 80) normalized features, fl lengths (frames + 1, as BatchPre returns), sel (B,) bool.
    Returns new (f, fl); the batch gets longer if a stretched utterance is the longest."""
    if not bool(sel.any()):
        return f, fl
    outs, lens = [], []
    for b in range(f.shape[0]):
        n = int(fl[b]) - 1
        x = f[b, :n]
        if sel[b] and n > 1:
            x = (distort(x * std + mean, rng) - mean) / std
        outs.append(x)
        lens.append(x.shape[0])
    T = max(lens) + 1  # keep BatchPre's extra zero frame
    out = f.new_zeros(f.shape[0], T, f.shape[2])
    for b, x in enumerate(outs):
        out[b, : x.shape[0]] = x
    return out, torch.tensor(lens, device=fl.device) + 1
