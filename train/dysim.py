# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Simulated atypical speech at the waveform level: typical speech goes in, speech with one condition's traits comes out, same transcript.

Replaces nothing in simdys.py (mel features, whole clip); this works on a source-filter analysis (WORLD: pitch, spectral envelope, aperiodicity) and
knows roughly which frames are vowels, frication, bursts and silence, so it can weaken consonants, centralize vowels, change the voice source,
insert repetitions and so on. Audio is written to a pack for the normal training pipeline, which then adds room noise and reverberation.

    python dysim.py demo in.wav outdir [--severity 0.7]                 one file per condition, to listen to
    python dysim.py gen --manifest ls100_train.jsonl --out G:/x/sim [--hours 100] [--workers 6] [--seed 1] [--conditions a,b]

Every number here is a starting value set by hand from the clinical descriptions (Darley, Aronson & Brown; Duffy), to be calibrated with dys_stats.py
against real speakers. A trait stays only if training on it lowers the error on real speakers who were held out. A condition with no real test speech
is simulated and untested: that is what may be said about it.

Profiles draw one set of trait values per simulated speaker (typical speaker + condition + severity), so a person's speech stays consistent across clips.
"""
import argparse, hashlib, json, math, multiprocessing as mp, os, sys, time
import numpy as np
import pyworld as pw
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

SR = 16000
FP = 5.0  # analysis frame period, ms
FPS = 1000.0 / FP


# ------------------------------------------------------------------ analysis
class Seg:
    """Frame classes from the analysis: silence, voiced, frication, burst; syllable nuclei from the voiced energy."""

    def __init__(self, f0, sp):
        lin = sp.sum(1) + 1e-12
        self.e = 10 * np.log10(lin)
        floor = np.quantile(self.e, 0.05)
        self.floor = floor
        peak = np.quantile(self.e, 0.95)
        self.sil = self.e < floor + max(6.0, 0.25 * (peak - floor))
        self.voiced = (f0 > 0) & ~self.sil
        bins = sp.shape[1]
        hi = sp[:, int(bins * 0.35):].sum(1) / lin  # share of energy above 2.8 kHz
        self.fric = ~self.voiced & ~self.sil & (hi > 0.2)
        rise = np.diff(self.e, prepend=self.e[0])
        self.burst = ~self.voiced & ~self.sil & (uniform_filter1d(rise, 3) > 3.0)
        self.cons = (~self.voiced & ~self.sil) | self.burst
        self.nuclei = self._nuclei()

    def _nuclei(self):
        v = np.where(self.voiced, self.e, self.floor)
        s = gaussian_filter1d(v, 8)
        p = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]) & self.voiced[1:-1])[0] + 1
        keep, last = [], -999
        for i in p:
            if i - last >= 20:  # at least 100 ms apart
                keep.append(int(i)); last = i
        return np.array(keep, dtype=int)

    def dips(self):
        """Frames between syllables: the lowest point between neighbouring nuclei."""
        n = self.nuclei
        return np.array([a + int(np.argmin(self.e[a:b])) for a, b in zip(n[:-1], n[1:])], dtype=int) if len(n) > 1 else np.array([], dtype=int)


def analyze(x):
    x = x.astype(np.float64)
    f0, t = pw.dio(x, SR, frame_period=FP, f0_floor=60.0, f0_ceil=500.0)
    f0 = pw.stonemask(x, f0, t, SR)
    return f0, pw.cheaptrick(x, f0, t, SR), pw.d4c(x, f0, t, SR)


def synth(f0, sp, ap):
    return pw.synthesize(f0.astype(np.float64), np.maximum(sp, 1e-16).astype(np.float64), np.clip(ap, 0.0, 1.0).astype(np.float64), SR, FP)


# ------------------------------------------------------------------ traits: each takes the frame arrays and returns new ones
def retime(f0, sp, ap, pos):
    """Resample the frames at fractional positions pos: spectra interpolated in log, pitch and aperiodicity follow."""
    pos = np.clip(pos, 0, len(f0) - 1)
    i = np.minimum(pos.astype(int), len(f0) - 2) if len(f0) > 1 else np.zeros(len(pos), int)
    w = (pos - i)[:, None]
    ls = np.log(sp + 1e-16)
    out_sp = np.exp(ls[i] * (1 - w) + ls[i + 1] * w) if len(f0) > 1 else sp[i]
    out_ap = ap[i] * (1 - w) + ap[i + 1] * w if len(f0) > 1 else ap[i]
    near = np.round(pos).astype(int)
    return f0[near], out_sp, out_ap


def rate_map(n, seg, rng, slow, uneven, vowel_long, rushes):
    """Output frame positions: slower overall, unevenly by piece (150-500 ms), vowels longer, some pieces rushed."""
    speed = np.ones(n)
    i = 0
    while i < n:
        k = int(rng.integers(30, 100))
        f = 1.0 / max(0.25, slow * math.exp(rng.normal(0, uneven)))
        if rng.random() < rushes:
            f = rng.uniform(1.15, 1.6)
        speed[i:i + k] = f
        i += k
    speed = uniform_filter1d(speed, 7)
    speed[seg.voiced] /= vowel_long
    return np.cumsum(np.concatenate(([0.0], speed[:-1])))[:: 1], speed


def resample_rate(f0, sp, ap, seg, rng, slow=1.0, uneven=0.0, vowel_long=1.0, rushes=0.0):
    pos, _ = rate_map(len(f0), seg, rng, slow, uneven, vowel_long, rushes)
    q = np.interp(np.arange(0, pos[-1], 1.0), pos, np.arange(len(f0)))
    return q


def insert_pauses(frames, seg_e, cuts, floor_frames, rng):
    """frames: list of arrays indexed the same; cuts: {frame: pause length in frames}."""
    idx, prev = [], 0
    for c in sorted(cuts):
        idx.append(np.arange(prev, c))
        idx.append(rng.choice(floor_frames, cuts[c]))
        prev = c
    idx.append(np.arange(prev, len(seg_e)))
    return np.concatenate(idx).astype(int)


def warp_envelope(sp, factor):
    """Move the formants: frequency axis scaled by factor (voiced frames only by the caller)."""
    n = sp.shape[1]
    src = np.clip(np.arange(n) / factor, 0, n - 1)
    i = np.minimum(src.astype(int), n - 2)
    w = src - i
    return sp[:, i] * (1 - w) + sp[:, i + 1] * w


def centralize(sp, voiced, amount):
    """Vowels drift toward a neutral vowel: each voiced frame's log envelope moves toward this speaker's mean vowel; formants blur."""
    if not voiced.any() or amount <= 0:
        return sp
    ls = np.log(sp + 1e-16)
    mean = ls[voiced].mean(0, keepdims=True)
    out = ls.copy()
    out[voiced] = (1 - amount) * ls[voiced] + amount * mean
    out[voiced] = gaussian_filter1d(out[voiced], 1 + 5 * amount, axis=1)
    return np.exp(out)


def resonance_db(n_bins, f_hz, gain_db, bw_hz):
    fr = np.linspace(0, SR / 2, n_bins)
    return gain_db * np.exp(-0.5 * ((fr - f_hz) / bw_hz) ** 2)


def nasalize(sp, voiced, amount, rng):
    """Hypernasal resonance: low nasal pole, an anti-resonance in the first-formant region, weaker higher formants."""
    if amount <= 0:
        return sp
    n = sp.shape[1]
    zero = rng.uniform(600, 1100)
    db = resonance_db(n, rng.uniform(230, 320), 10 * amount, 90) + resonance_db(n, zero, -16 * amount, 220) + resonance_db(n, 3000, -8 * amount, 1500)
    out = sp.copy()
    out[voiced] = sp[voiced] * 10 ** (db / 10)
    return out


def apply_tilt(sp, db_at_nyquist):
    return sp * 10 ** (np.linspace(0, db_at_nyquist, sp.shape[1]) / 10)[None]


def fluency(f0, sp, ap, seg, rng, rep, block, prolong):
    """Stuttering-like events at syllable onsets: part-syllable repetitions, silent blocks, prolongations. The transcript keeps the intended words."""
    idx, prev = [], 0
    floor = np.where(seg.e <= np.quantile(seg.e, 0.1))[0]
    if not len(floor) or not len(seg.nuclei):
        return np.arange(len(f0))
    starts = np.concatenate(([0], seg.dips()))
    for b in starts:
        idx.append(np.arange(prev, b))
        prev = b
        r = rng.random()
        if r < rep:  # "b-b-boy": the onset, 60-180 ms, 1-4 times, short gaps
            k = int(rng.integers(12, 36))
            for _ in range(int(rng.integers(1, 5))):
                idx.append(np.arange(b, min(b + k, len(f0))))
                idx.append(rng.choice(floor, int(rng.integers(8, 40))))
        elif r < rep + block:
            idx.append(rng.choice(floor, int(rng.integers(40, 300))))  # 0.2-1.5 s
        elif r < rep + block + prolong:
            k = int(rng.integers(8, 24))
            idx.append(np.repeat(np.arange(b, min(b + k, len(f0))), int(rng.integers(3, 8))))
    idx.append(np.arange(prev, len(f0)))
    return np.concatenate(idx).astype(int)


# ------------------------------------------------------------------ one speaker's draw of a condition
def U(rng, lo, hi, s):
    return rng.uniform(lo, hi) * s


# Trait values at severity s. Names: slow (speed multiplier below 1), uneven, vowel_long, rushes, pause (share of syllable gaps with a pause),
# mono (0 keep pitch movement, 1 flat), f0_st (semitones), jitter (cents), tremor (depth, semitones), breaks (per second), stops (voice stoppages per second),
# weak (consonant attenuation, dB), lenite (bursts softened), vowel (centralization), blur (transitions smeared over time), nasal, breathy, strain (dB tilt up),
# rough (period doubling), tilt (dB at Nyquist), decay (dB loudness loss over the clip), am (loudness tremor depth), fast_fade, rep/block/prolong (per syllable).
def profile(cond, s, rng):
    g = lambda lo, hi: U(rng, lo, hi, s)
    p = dict(slow=1.0, uneven=0.0, vowel_long=1.0, rushes=0.0, pause=0.0, mono=0.0, f0_st=0.0, jitter=0.0, tremor=0.0, breaks=0.0, stops=0.0, weak=0.0, lenite=0.0, vowel=0.0, blur=0.0,
             nasal=0.0, breathy=0.0, strain=0.0, rough=0.0, tilt=0.0, decay=0.0, am=0.0, rep=0.0, block=0.0, prolong=0.0, source="")
    slow = lambda lo, hi: 1 - s * rng.uniform(lo, hi)
    if cond == "flaccid":  # weak muscles: breathy, nasal, imprecise (bulbar palsy, myasthenia, part of ALS)
        p.update(slow=slow(.1, .35), breathy=g(.3, .9), nasal=g(.3, 1.), weak=g(4, 14), lenite=g(.3, .9), vowel=g(.1, .4), mono=g(.2, .6), tilt=g(-10, -3), decay=g(1, 6))
    elif cond == "spastic":  # slow, strained, low pitch, reduced stress (bilateral stroke, part of ALS and cerebral palsy)
        p.update(slow=slow(.25, .6), uneven=g(.05, .2), strain=g(3, 10), f0_st=g(-5, -1), mono=g(.4, .85), weak=g(5, 14), lenite=g(.2, .7), vowel=g(.2, .5), blur=g(.2, .7), pause=g(.05, .25), rough=g(0, .4))
    elif cond == "ataxic":  # irregular timing, prolonged sounds, excess equal stress (cerebellar, MS)
        p.update(slow=slow(.2, .5), uneven=g(.2, .55), vowel_long=1 + g(.1, .6), pause=g(.1, .35), vowel=g(.2, .55), weak=g(3, 10), lenite=g(.1, .5), breaks=g(0, .5), jitter=g(5, 25), am=g(.1, .35), blur=g(.1, .4))
    elif cond == "hypokinetic":  # Parkinson's: quiet, flat, short rushes, breathy, blurred consonants
        p.update(slow=1 + s * rng.uniform(-.1, .2), rushes=g(.1, .5), mono=g(.5, 1.), breathy=g(.2, .7), weak=g(5, 16), lenite=g(.3, .9), blur=g(.2, .6), decay=g(3, 12), tilt=g(-8, -2), pause=g(.02, .15), tremor=g(0, .6), rep=g(0, .04))
    elif cond == "hyperkinetic":  # Huntington's, dystonia: unpredictable rate and loudness, sudden stoppages
        p.update(slow=slow(.05, .35), uneven=g(.2, .6), rushes=g(.05, .3), stops=g(.1, .6), am=g(.2, .6), pause=g(.1, .4), weak=g(3, 10), lenite=g(.1, .5), vowel=g(.1, .4), strain=g(0, 5), breaks=g(0, .6))
    elif cond == "uumn":  # one-sided stroke: mild to moderate imprecise articulation, some slowness and harshness
        p.update(slow=slow(.1, .35), weak=g(4, 12), lenite=g(.2, .7), vowel=g(.1, .35), rough=g(0, .3), strain=g(0, 4), blur=g(.1, .4))
    elif cond == "als":  # mixed flaccid and spastic
        p.update(slow=slow(.3, .65), uneven=g(.05, .25), nasal=g(.3, 1.), breathy=g(.1, .5), strain=g(1, 6), mono=g(.3, .8), weak=g(6, 16), lenite=g(.3, .9), vowel=g(.3, .6), blur=g(.2, .6), pause=g(.1, .35), tilt=g(-12, -3))
    elif cond == "cp":  # cerebral palsy: mostly spastic, often with irregular movement
        p.update(slow=slow(.2, .6), uneven=g(.1, .45), strain=g(1, 7), mono=g(.2, .7), weak=g(5, 15), lenite=g(.3, .9), vowel=g(.2, .6), blur=g(.2, .6), pause=g(.1, .35), breaks=g(0, .3), am=g(0, .25), tilt=g(-14, -2))
    elif cond == "stutter":
        p.update(rep=g(.03, .25), block=g(.01, .1), prolong=g(.02, .15))
    elif cond == "apraxia":  # effortful, sounds separated, distorted
        p.update(slow=slow(.2, .5), uneven=g(.1, .35), pause=g(.2, .6), vowel_long=1 + g(.1, .5), weak=g(2, 8), lenite=g(.1, .4), vowel=g(.1, .35), rep=g(0, .06), prolong=g(0, .06))
    elif cond == "dysphonia":  # disordered voice with normal articulation, rough, breathy or strained
        k = rng.integers(3)
        p.update(jitter=g(10, 50), breaks=g(0, .8))
        p.update(rough=g(.3, 1.)) if k == 0 else p.update(breathy=g(.5, 1.)) if k == 1 else p.update(strain=g(4, 12), stops=g(.1, .5), f0_st=g(-4, 0))
    elif cond == "electrolarynx":  # artificial voice source after laryngectomy: flat buzz in every frame
        p.update(source="el", weak=g(2, 10), lenite=g(.2, .7))
    elif cond == "esophageal":  # low, rough, breath-driven voice
        p.update(source="eso", slow=slow(.1, .35), pause=g(.1, .35), rough=g(.3, 1.), weak=g(2, 8), jitter=g(20, 60))
    elif cond == "whisper":  # no voicing at all: aphonia, severe vocal weakness
        p.update(source="whisper", weak=g(0, 4))
    else:
        raise ValueError(cond)
    return p


CONDITIONS = ("flaccid", "spastic", "ataxic", "hypokinetic", "hyperkinetic", "uumn", "als", "cp", "stutter", "apraxia", "dysphonia", "electrolarynx", "esophageal", "whisper")


def simulate(x, cond, s, sp_rng, rng):
    """x int16 or float (at 16 kHz) -> int16. sp_rng draws the speaker's traits (same stream for a speaker's clips), rng this clip's events."""
    x = np.asarray(x, dtype=np.float64)
    level = np.sqrt((x ** 2).mean()) + 1e-9
    if len(x) < SR // 5:
        return np.clip(x, -32768, 32767).astype(np.int16)
    p = profile(cond, s, sp_rng)
    f0, sp, ap = analyze(x / 32768.0)
    seg = Seg(f0, sp)

    # --- articulation, on the original frames so the classes still line up
    if p["vowel"] > 0:
        sp = centralize(sp, seg.voiced, min(0.85, p["vowel"]))
    if p["weak"] > 0:
        sp = sp.copy()
        sp[seg.cons] *= 10 ** (-p["weak"] * rng.uniform(0.6, 1.2) / 10)
    if p["lenite"] > 0 and seg.burst.any():  # bursts become a soft noise, smeared forward: stops turn into fricatives
        sp = sp.copy()
        for t in np.where(seg.burst)[0]:
            if rng.random() < p["lenite"]:
                sp[t] *= 0.25
                ap[t] = 1.0
    if p["blur"] > 0:  # transitions between sounds slow down
        ls = np.log(sp + 1e-16)
        sp = np.exp(gaussian_filter1d(ls, 1 + 6 * p["blur"], axis=0))

    # --- resonance and voice source
    sp = nasalize(sp, seg.voiced, min(1.0, p["nasal"]), sp_rng)
    if p["strain"] > 0:
        sp = apply_tilt(sp, p["strain"])
    if p["tilt"] != 0:
        sp = apply_tilt(sp, p["tilt"])
    if p["breathy"] > 0:
        b = min(0.95, p["breathy"])
        ap = ap ** (1 - b)
        sp = apply_tilt(sp, -4 * b)
    n = len(f0)
    t = np.arange(n) / FPS
    voiced = f0 > 0
    if voiced.any():
        lf = np.log2(np.where(voiced, f0, 1.0)) * 12
        mean = lf[voiced].mean()
        lf[voiced] = mean + (lf[voiced] - mean) * (1 - min(1.0, p["mono"])) + p["f0_st"]
        if p["tremor"] > 0:
            lf[voiced] += p["tremor"] * np.sin(2 * np.pi * sp_rng.uniform(4, 7) * t[voiced] + rng.uniform(0, 6.28))
        if p["jitter"] > 0:
            lf[voiced] += rng.normal(0, p["jitter"] / 100.0, voiced.sum())
        if p["rough"] > 0:  # period doubling: alternate frames low, some noise
            alt = (np.arange(n) // 2) % 2 == 0
            lf[voiced & alt] -= rng.uniform(1, 5) * p["rough"]
            ap = np.maximum(ap, 0.4 * p["rough"])
        f0 = np.where(voiced, 2 ** (lf / 12), 0.0)
    if p["breaks"] > 0:  # pitch breaks, short stretches of noise or an octave jump
        for _ in range(rng.poisson(p["breaks"] * n / FPS)):
            a = int(rng.integers(0, n))
            k = int(rng.integers(10, 60))
            if rng.random() < 0.5:
                f0[a:a + k] = np.where(f0[a:a + k] > 0, f0[a:a + k] * rng.choice((0.5, 2.0)), 0.0)
            else:
                ap[a:a + k] = np.maximum(ap[a:a + k], 0.9)
    if p["source"] == "el":
        f0 = np.where(seg.sil, 0.0, sp_rng.uniform(75, 120))
        ap = np.where(seg.sil[:, None], ap, np.minimum(ap, 0.15))
    elif p["source"] == "eso":
        lo = sp_rng.uniform(45, 85)
        f0 = np.where(f0 > 0, lo * np.exp(rng.normal(0, 0.1, n)), 0.0)
        ap = np.maximum(ap, 0.35)
    elif p["source"] == "whisper":
        f0 = np.zeros(n)
        ap = np.ones_like(ap)

    # --- timing: rate, pauses, stoppages, stuttering events
    if p["slow"] != 1.0 or p["uneven"] or p["vowel_long"] != 1.0 or p["rushes"]:
        q = resample_rate(f0, sp, ap, seg, rng, p["slow"], p["uneven"], p["vowel_long"], p["rushes"])
        f0, sp, ap = retime(f0, sp, ap, q)
        seg = Seg(f0, sp)
    floor = np.where(seg.e <= np.quantile(seg.e, 0.1))[0]
    cuts = {int(c): int(rng.integers(30, 120)) for c in seg.dips() if rng.random() < p["pause"]} if len(floor) and p["pause"] > 0 else {}
    if cuts:
        i = insert_pauses(None, seg.e, cuts, floor, rng)
        f0, sp, ap = f0[i], sp[i], ap[i]
        seg = Seg(f0, sp)
    if p["rep"] or p["block"] or p["prolong"]:
        i = fluency(f0, sp, ap, seg, rng, p["rep"], p["block"], p["prolong"])
        f0, sp, ap = f0[i], sp[i], ap[i]
        seg = Seg(f0, sp)
    if p["stops"] > 0:  # the voice stops suddenly
        n = len(f0)
        for _ in range(rng.poisson(p["stops"] * n / FPS)):
            a, k = int(rng.integers(0, n)), int(rng.integers(20, 80))
            f0[a:a + k] = 0.0
            sp[a:a + k] = sp[a:a + k] * 10 ** (-30 / 10)

    # --- loudness over time
    n = len(f0)
    gain = np.zeros(n)
    if p["decay"] > 0:
        gain -= p["decay"] * np.linspace(0, 1, n)
    if p["am"] > 0:
        slow_noise = gaussian_filter1d(rng.normal(0, 1, n), 40)
        gain += 20 * p["am"] * slow_noise / (slow_noise.std() + 1e-6) / 2
    if gain.any():
        sp = sp * (10 ** (gain / 10))[:, None]
    y = synth(f0, sp, ap)
    y = y * (level / (np.sqrt((y ** 2).mean()) + 1e-9))  # same level as the source, whatever the traits did: level is the training pipeline's business
    return np.clip(y, -32768, 32767).astype(np.int16)


# ------------------------------------------------------------------ pack writer
def _seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _job(a):
    it, seed, conds, sevs, src_cache = a
    if "pack" in it:
        pack = it["pack"]
        if pack not in src_cache:
            src_cache[pack] = np.memmap(pack, dtype="<i2", mode="r")
        x = np.array(src_cache[pack][it["off"]: it["off"] + it["n"]])
    else:
        import soundfile as sf
        x = sf.read(it["wav"], dtype="int16")[0]
        it["off"] = _seed(it["wav"]) % 1000000007  # only names the clip for the random draws below
    # the condition and severity belong to a simulated speaker: typical speaker + one of three draws
    who = _seed(seed, it.get("speaker", ""), _seed(it["off"]) % 3)
    r0 = np.random.default_rng(who)
    cond = conds[int(r0.integers(len(conds)))]
    s = float(r0.uniform(*sevs))
    try:
        return it, cond, s, simulate(x, cond, s, np.random.default_rng(who + 1), np.random.default_rng(_seed(seed, it["off"])))
    except Exception as e:
        print("skip", it["off"], repr(e), flush=True)
        return it, cond, s, None


_cache = {}
def _work(a):
    return _job((*a, _cache))


def gen(a):
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    np.random.default_rng(a.seed).shuffle(items)
    if a.hours:
        keep, tot = [], 0.0
        for it in items:
            if tot >= a.hours * 3600:
                break
            keep.append(it); tot += it.get("n", it.get("duration", 0) * SR) / SR
        items = keep
    conds = tuple(a.conditions.split(",")) if a.conditions else CONDITIONS
    sevs = tuple(float(v) for v in a.severity.split(","))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pack, man = a.out + ".pack", a.out + ".jsonl"
    done = 0
    if os.path.exists(man):  # resume after an interruption
        with open(man, encoding="utf-8") as f:
            rows = [l for l in f if l.strip()]
        done = len(rows)
        pos = (lambda r: r["off"] + r["n"])(json.loads(rows[-1])) if rows else 0
        with open(pack, "r+b") as f:
            f.truncate(pos * 2)
        print("resuming after", done, "clips", flush=True)
    else:
        pos = 0
    t0, audio = time.time(), 0.0
    with open(pack, "ab") as fp, open(man, "a", encoding="utf-8", newline="\n") as fm, mp.Pool(a.workers) as pool:
        for k, (it, cond, s, y) in enumerate(pool.imap(_work, ((it, a.seed, conds, sevs) for it in items[done:]), chunksize=8)):
            if y is None:
                continue
            fp.write(y.astype("<i2").tobytes())
            r = {"wav": it.get("wav", ""), "text": it["text"], "duration": round(len(y) / SR, 2), "source": "sim", "speaker": f"{it.get('speaker', '')}:{cond}", "condition": cond,
                 "severity": round(s, 2), "pack": os.path.abspath(pack), "off": pos, "n": len(y)}
            fm.write(json.dumps(r) + "\n")
            pos += len(y)
            audio += len(y) / SR
            if k % 500 == 0:
                fp.flush(); fm.flush()
                print(k + done, "of", len(items), "clips,", round(audio / 3600, 1), "h out,", round(audio / max(1.0, time.time() - t0), 2), "x real time", flush=True)
    print("finished", flush=True)


def demo(a):
    import soundfile as sf
    x, sr = sf.read(a.wav, dtype="int16")
    assert sr == SR, "16 kHz input"
    os.makedirs(a.outdir, exist_ok=True)
    sf.write(os.path.join(a.outdir, "00_input.wav"), x, SR)
    for k, c in enumerate(CONDITIONS, 1):
        t0 = time.time()
        y = simulate(x, c, a.severity, np.random.default_rng(_seed(c, 1)), np.random.default_rng(_seed(c, 2)))
        sf.write(os.path.join(a.outdir, f"{k:02d}_{c}.wav"), y, SR)
        print(c, round(len(y) / SR, 2), "s", round(time.time() - t0, 2), "s compute")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("wav"); d.add_argument("outdir"); d.add_argument("--severity", type=float, default=0.7)
    g = sub.add_parser("gen"); g.add_argument("--manifest", required=True); g.add_argument("--out", required=True); g.add_argument("--hours", type=float, default=0.0)
    g.add_argument("--workers", type=int, default=6); g.add_argument("--seed", type=int, default=1); g.add_argument("--conditions", default=""); g.add_argument("--severity", default="0.2,1.0")
    a = ap.parse_args()
    demo(a) if a.cmd == "demo" else gen(a)
