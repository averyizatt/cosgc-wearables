# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Simulated atypical speech at the waveform level: typical speech goes in, speech with one condition's traits comes out, same transcript.

Replaces nothing in simdys.py (mel features, whole clip); this works on a source-filter analysis (WORLD: pitch, spectral envelope, aperiodicity) and
knows roughly which frames are vowels, frication, bursts and silence, so it can weaken consonants, centralize vowels, change the voice source,
insert repetitions and so on. Audio is written to a pack for the normal training pipeline, which then adds room noise and reverberation.
The traits are computed on WORLD frames, but by default (--render transparent) the output is the original recording, retimed and repitched by
overlap-add and filtered toward the new envelope; resynthesizing with WORLD (--render world) costs a model 10 points on clean speech (E16).

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phonesim

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


# ------------------------------------------------------------------ transparent rendering (E16)
# Resynthesizing everything with WORLD costs the model 10 points on clean speech and makes every simulated clip detectable (AUC 1.000).
# Here the original recording is kept and only what the traits change is changed: timing and pitch by pitch-synchronous overlap-add of the
# original waveform (Moulines & Charpentier 1990), envelope and voicing by a time-varying filter plus shaped noise. With no traits the output is
# the input. Frames that need voicing the source never had (an unvoiced sound made voiced, an artificial voice source) fall back to WORLD.
HOP = int(SR * FP / 1000)
_HANN = {}
RENDER = "transparent"  # or "world"; gen and demo set it from --render


def pitch_marks(x, f0):
    """Analysis marks: one per pitch period in voiced frames (on the waveform peak near where the period predicts), one per hop elsewhere.
    Returns sample positions, the period at each mark, and whether it is voiced."""
    n = len(f0)
    lp = gaussian_filter1d(x, 2.0)
    pos, per, vo = [], [], []
    t, end = 0, len(x)
    while t < end:
        j = min(n - 1, t // HOP)
        if f0[j] > 0:
            p = int(SR / f0[j])
            if not vo or not vo[-1]:  # entering a voiced stretch: start on the largest peak in the first period
                t = t + int(np.argmax(lp[t: t + p])) if t + p < end else t
            else:
                lo, hi = max(0, t - p // 5), min(end, t + p // 5 + 1)
                t = lo + int(np.argmax(lp[lo:hi])) if hi > lo else t
            pos.append(t); per.append(p); vo.append(True)
            t += p
        else:
            pos.append(t); per.append(HOP); vo.append(False)
            t += HOP
    return np.array(pos), np.array(per), np.array(vo, bool)


def psola(x, marks, src, ratio, voiced_out, trace=None):
    """The original waveform at new times and pitch. src: for each output frame, its (fractional) frame in the original. ratio: the pitch
    wanted in each output frame over the original's there; voiced_out: frames rendered pitch-synchronously."""
    pos, per, vo = marks
    pad = 2 * int(per.max()) + 2 * HOP
    xp = np.concatenate((np.zeros(pad), x, np.zeros(pad)))
    L = len(src) * HOP
    y, w = np.zeros(L + 2 * pad), np.zeros(L + 2 * pad)
    vpos = pos[vo]
    vper = per[vo]
    t, was_voiced, last = 0, False, -16
    while t < L:
        j = min(len(src) - 1, t // HOP)
        s = src[j] * HOP + (t - j * HOP)  # where this output sample comes from in the original
        k = int(np.searchsorted(vpos, s))
        if k == len(vpos) or (k > 0 and s - vpos[k - 1] < vpos[k] - s):
            k -= 1
        # voiced output, once the original's nearest mark is in reach (until then, ordinary hops)
        if voiced_out[j] and len(vpos) and (was_voiced or -vper[k] // 2 <= vpos[k] - s < vper[k]):
            rate = src[j + 1] - src[j] if j + 1 < len(src) else 1.0
            if not was_voiced or (abs(ratio[j] - 1) < 0.01 and abs(rate - 1) < 0.01):
                # entering voicing, or where neither pitch nor timing changes: in phase with the original's mark (no change gives back the input)
                d = int(vpos[k] - s)
                if abs(d) <= vper[k] // 2 or (not was_voiced and 0 <= d < vper[k]):
                    # never back to or before the last grain: where src jumps back (a repetition) snapping could otherwise cycle forever
                    t = max(last + 16, t + d)
                    if t >= L:
                        break
            m, pa = int(vpos[k]), int(vper[k])
            # the next mark: the original's own spacing, scaled for the pitch wanted (unchanged pitch and timing give back the original)
            gap = vpos[k + 1] - vpos[k] if k + 1 < len(vpos) and vpos[k + 1] - vpos[k] < 1.5 * pa else pa
            step = max(16, int(round(gap / ratio[j])))
            was_voiced = True
        else:
            m, pa, step = int(round(s)), HOP, HOP
            if was_voiced:  # back to the hop grid
                step = HOP - t % HOP or HOP
            was_voiced = False
        win = _HANN.get(pa)
        if win is None:
            win = _HANN[pa] = np.hanning(2 * pa + 1)
        if trace is not None:
            trace.append((t, m, pa, step))
        y[pad + t - pa: pad + t + pa + 1] += xp[pad + m - pa: pad + m + pa + 1] * win
        w[pad + t - pa: pad + t + pa + 1] += win
        last = t
        t += step
    return (y / np.maximum(w, 0.5))[pad: pad + L]  # exact where grains line up with the original; lower pitch leaves shallow dips


def filter_frames(y, ref_sp, ref_ap, ref_v, sp, ap, v, rng):
    """Move y (whose frames have envelope ref_sp, aperiodicity ref_ap) to envelope sp and aperiodicity ap. The periodic part is scaled by a
    filter; noise is added where the target is noisier than the source (breathiness, devoicing, lenition, breaths). WORLD's convention:
    aperiodic power = sp * ap**2."""
    from scipy.signal import stft, istft
    nb = sp.shape[1]
    nfft = 2 * (nb - 1)
    _, _, Y = stft(y, SR, window="hann", nperseg=400, noverlap=400 - HOP, nfft=nfft, boundary="zeros", padded=True)
    T = min(Y.shape[1], len(sp))
    Y = Y[:, :T]
    a_ref = np.where(ref_v[:T, None], np.clip(ref_ap[:T], 0, 1), 1.0) ** 2
    a_mod = np.where(v[:T, None], np.clip(ap[:T], 0, 1), 1.0) ** 2
    ratio = (sp[:T] + 1e-16) / (ref_sp[:T] + 1e-16)
    h = np.clip((1 - a_mod + 1e-3) / (1 - a_ref + 1e-3), 0.0, 1.0)
    g2 = np.clip(ratio * h, 1e-4, 100.0)
    g2 = np.exp(gaussian_filter1d(np.log(g2), 1.0, axis=0))  # a little smoothing over time: no musical noise
    # how STFT power relates to WORLD's envelope on this clip (one number), for the added noise
    P = np.abs(Y.T) ** 2
    e = ref_sp[:T].sum(1)
    loud = e > np.quantile(e, 0.3)
    c = np.median(P[loud].sum(1) / (e[loud] + 1e-16)) if loud.any() else 1.0
    extra = np.maximum(0.0, sp[:T] * a_mod - g2 * ref_sp[:T] * a_ref)
    win = np.hanning(400)
    _, _, N = stft(rng.standard_normal(len(y)), SR, window="hann", nperseg=400, noverlap=400 - HOP, nfft=nfft, boundary="zeros", padded=True)
    N = N[:, :T] / np.sqrt((win ** 2).sum() / win.sum() ** 2)  # unit expected power per bin, in scipy's scaling
    Z = Y * np.sqrt(g2.T) + N * np.sqrt(c * extra.T)
    _, z = istft(Z, SR, window="hann", nperseg=400, noverlap=400 - HOP, nfft=nfft, boundary=True)
    return z[: len(y)] if len(z) >= len(y) else np.pad(z, (0, len(y) - len(z)))


def render_transparent(x, orig, src, f0, sp, ap, rng, world_frames=None):
    """x: the original clip (float). orig: its analysis (f0, sp, ap). src: each output frame's position in the original. f0, sp, ap: the
    target frames. world_frames: output frames to take from WORLD (None: those voiced where the source is not)."""
    f00, sp0, ap0 = orig
    n0 = len(f00)
    src = np.clip(src, 0, n0 - 1)
    i = np.minimum(src.astype(int), max(0, n0 - 2))
    w = (src - i)[:, None]
    i1 = np.minimum(i + 1, n0 - 1)
    ref_sp = np.exp(np.log(sp0[i] + 1e-16) * (1 - w) + np.log(sp0[i1] + 1e-16) * w)
    ref_ap = ap0[i] * (1 - w) + ap0[i1] * w
    ref_v = f00[np.round(src).astype(int)] > 0
    v = f0 > 0
    if world_frames is None:
        world_frames = v & ~ref_v
    ref_f0 = f00[np.round(src).astype(int)]
    y = psola(x, pitch_marks(x, f00), src, np.where(v & ref_v, f0 / np.maximum(ref_f0, 1.0), 1.0), v & ref_v)
    y = filter_frames(y, ref_sp, ref_ap, ref_v, sp, ap, v, rng)
    if world_frames.any():
        yw = synth(f0, sp, ap)
        m = np.repeat(gaussian_filter1d(world_frames.astype(float), 1.5), HOP)
        m = np.pad(m, (0, max(0, len(y) - len(m))))[: len(y)]
        yw = np.pad(yw, (0, max(0, len(y) - len(yw))))[: len(y)]
        y = y * (1 - m) + yw * m
    return y


# ------------------------------------------------------------------ traits: each takes the frame arrays and returns new ones
def retime(f0, sp, ap, pos, var=0.8):
    """Resample the frames at fractional positions pos: spectra interpolated in log, pitch and aperiodicity follow."""
    pos = np.clip(pos, 0, len(f0) - 1)
    i = np.minimum(pos.astype(int), len(f0) - 2) if len(f0) > 1 else np.zeros(len(pos), int)
    w = (pos - i)[:, None]
    ls = np.log(sp + 1e-16)
    # interpolated frames are smoother than real slow speech, which keeps its frame-to-frame variation (E5): put it back (WORLD rendering
    # only: the transparent renderer keeps the recording's own variation)
    out_sp = np.exp(ls[i] * (1 - w) + ls[i + 1] * w + var * np.random.default_rng(len(pos)).standard_normal(w.shape) * (4 * w * (1 - w))) if len(f0) > 1 else sp[i]
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


def add_breaths(f0, sp, ap, cuts, amount, rng):
    """An audible in-breath in some of the inserted pauses: broadband noise, 15-30 dB below the speech, rising and falling over 0.2-0.5 s."""
    sp, ap = sp.copy(), ap.copy()
    v = f0 > 0
    ref = np.log(sp[v].mean(0) + 1e-16) if v.any() else np.log(sp.mean(0) + 1e-16)
    shape = gaussian_filter1d(ref, 8)
    shape = 0.5 * shape + 0.5 * shape.mean()  # flatter than speech
    off = 0
    for c in sorted(cuts):
        start, k = c + off, cuts[c]
        off += k
        if k < 30 or rng.random() >= amount:
            continue
        m = int(rng.integers(20, min(100, k - 6) + 1))
        a = start + (k - m) // 2
        env = np.sin(np.linspace(0, np.pi, m)) ** 2
        lvl = -rng.uniform(15, 30) * np.log(10) / 10  # dB below the mean voiced spectrum, in natural log of power
        sp[a:a + m] = np.maximum(sp[a:a + m], np.exp(shape + lvl)[None] * env[:, None])
        ap[a:a + m] = 1.0
    return sp, ap


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
    out[voiced] = gaussian_filter1d(out[voiced], 1 + 2 * amount, axis=1)
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


def runs(mask):
    """(start, end) of each run of True frames."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def sound_errors(f0, sp, ap, seg, rng, omit, swap, devoice):
    """Wrong sounds rather than weak ones (E15, Finding 32): consonants left out, fricatives moved to another place ("s" toward "sh" or "f"),
    voiced consonants said voiceless. Returns new arrays and the frames kept."""
    sp, ap, f0 = sp.copy(), ap.copy(), f0.copy()
    if swap > 0:
        for a, b in runs(seg.fric):
            if rng.random() < swap:
                sp[a:b] = warp_envelope(sp[a:b], rng.choice((0.6, 0.75, 1.35, 1.6)))
    if devoice > 0 and seg.voiced.any():  # weak voiced stretches are mostly voiced consonants and nasals; vowels are the loud ones
        peak = np.quantile(seg.e[seg.voiced], 0.9)
        for a, b in runs(seg.voiced & (seg.e < peak - 12)):
            if b - a <= 30 and rng.random() < devoice:
                f0[a:b] = 0.0
                ap[a:b] = 1.0
                sp[a:b] *= 0.5
    keep = np.ones(len(f0), bool)
    if omit > 0:
        for a, b in runs(seg.cons):
            if b - a <= 40 and rng.random() < omit:
                keep[a:b] = False
    return f0, sp, ap, np.where(keep)[0]


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
DYSARTHRIA = ("flaccid", "spastic", "ataxic", "hypokinetic", "hyperkinetic", "uumn", "als", "cp")


def profile(cond, s, rng):
    p = _profile(cond, s, rng)
    if cond in DYSARTHRIA:  # calibration round 2 (E15): TORGO's dysarthric speakers pause 12-18 % of the time more than controls and have
        # 1-14 dB weaker high frequencies; round 1 gave +1 to +6 % pauses and some conditions brighter, not duller
        p["pause"] = max(p["pause"], s * rng.uniform(0.3, 0.8))
        p["tilt"] = min(p["tilt"], 0.0) - s * rng.uniform(4, 12)
        p["strain"] = min(p["strain"], 4.0)
        p["blur"] = min(p["blur"], 0.35)
        # round 3: sound errors, which no amount of weakening produced; consonant errors are the most frequent kind in dysarthric speech
        p.update(omit=U(rng, .05, .25, s), swap=U(rng, .1, .45, s), devoice=U(rng, .1, .4, s))
        # E16 details: short breath groups with audible breaths, speech that wears down within the utterance, hard voice onsets
        p.update(breath=U(rng, .3, .9, s), fatigue=U(rng, 0, 1, s), hard_onset=U(rng, .2, .7, s) if cond in ("spastic", "cp", "als", "hyperkinetic") else 0.0)
    elif cond == "apraxia":
        p.update(omit=U(rng, .03, .15, s), swap=U(rng, .1, .4, s), devoice=U(rng, .05, .25, s))
    p["slow"] = max(p["slow"], 0.4)  # severities above 1 (round 2: severe sims cost the model 21 %, real severe speakers about 45-50 %) must not stretch speech past 2.5 x
    return p


def _profile(cond, s, rng):
    g = lambda lo, hi: U(rng, lo, hi, s)
    p = dict(slow=1.0, uneven=0.0, vowel_long=1.0, rushes=0.0, pause=0.0, mono=0.0, f0_st=0.0, jitter=0.0, tremor=0.0, breaks=0.0, stops=0.0, weak=0.0, lenite=0.0, vowel=0.0, blur=0.0,
             nasal=0.0, breathy=0.0, strain=0.0, rough=0.0, tilt=0.0, decay=0.0, am=0.0, rep=0.0, block=0.0, prolong=0.0, source="", omit=0.0, swap=0.0, devoice=0.0, breath=0.0, fatigue=0.0, hard_onset=0.0)
    slow = lambda lo, hi: 1 - s * rng.uniform(lo, hi)
    if cond == "flaccid":  # weak muscles: breathy, nasal, imprecise (bulbar palsy, myasthenia, part of ALS)
        p.update(slow=slow(.15, .4), breathy=g(.3, .9), nasal=g(.5, 1.), weak=g(8, 18), lenite=g(.4, 1.), vowel=g(.3, .6), mono=g(.2, .6), tilt=g(-10, -3), decay=g(1, 6))
    elif cond == "spastic":  # slow, strained, low pitch, reduced stress (bilateral stroke, part of ALS and cerebral palsy)
        p.update(slow=slow(.25, .6), uneven=g(.05, .2), strain=g(3, 10), f0_st=g(-5, -1), mono=g(.4, .85), weak=g(5, 14), lenite=g(.2, .7), vowel=g(.2, .5), blur=g(.2, .7), pause=g(.05, .25), rough=g(0, .4))
    elif cond == "ataxic":  # irregular timing, prolonged sounds, excess equal stress (cerebellar, MS)
        p.update(slow=slow(.2, .5), uneven=g(.2, .55), vowel_long=1 + g(.1, .6), pause=g(.1, .35), vowel=g(.2, .55), weak=g(3, 10), lenite=g(.1, .5), breaks=g(0, .5), jitter=g(5, 25), am=g(.1, .35), blur=g(.1, .4))
    elif cond == "hypokinetic":  # Parkinson's: quiet, flat, short rushes, breathy, blurred consonants
        p.update(slow=1 + s * rng.uniform(-.1, .2), rushes=g(.1, .5), mono=g(.5, 1.), breathy=g(.2, .7), weak=g(5, 16), lenite=g(.3, .9), blur=g(.2, .6), decay=g(3, 12), tilt=g(-8, -2), pause=g(.02, .15), tremor=g(0, .6), rep=g(0, .04))
    elif cond == "hyperkinetic":  # Huntington's, dystonia: unpredictable rate and loudness, sudden stoppages
        p.update(slow=slow(.05, .35), uneven=g(.2, .6), rushes=g(.05, .3), stops=g(.1, .6), am=g(.2, .6), pause=g(.1, .4), weak=g(3, 10), lenite=g(.1, .5), vowel=g(.1, .4), strain=g(0, 5), breaks=g(0, .6))
    elif cond == "uumn":  # one-sided stroke: mild to moderate imprecise articulation, some slowness and harshness
        p.update(slow=slow(.15, .4), weak=g(6, 16), lenite=g(.3, .8), vowel=g(.2, .5), rough=g(0, .3), strain=g(0, 4), blur=g(.1, .4))
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


def simulate(x, cond, s, sp_rng, rng, tg=None, log=None):
    """x int16 or float (at 16 kHz) -> int16. sp_rng draws the speaker's traits (same stream for a speaker's clips), rng this clip's events.
    tg: the clip's alignment (phonesim.read_textgrid); with it, sounds are changed one by one (phonesim.py) in place of the frame-class guesses
    of sound_errors. log: a list that receives the sound errors made."""
    x = np.asarray(x, dtype=np.float64)
    level = np.sqrt((x ** 2).mean()) + 1e-9
    if len(x) < SR // 5:
        return np.clip(x, -32768, 32767).astype(np.int16)
    p = profile(cond, s, sp_rng)
    z = phonesim.profile(cond, s, sp_rng) if tg else None
    f0, sp, ap = analyze(x / 32768.0)
    orig = (f0, sp, ap)
    src = np.arange(len(f0), dtype=np.float64)  # each frame's position in the original, through every change of timing
    if tg:  # sound errors first, while the alignment still matches the frames
        f0, sp, ap, errs, k = phonesim.apply(f0, sp, ap, phonesim.phone_frames(tg, len(f0), FPS), z, rng, warp_envelope)
        src = np.interp(k, np.arange(len(src)), src)
        if log is not None:
            log.extend(errs)
        p.update(omit=0.0, swap=0.0, devoice=0.0)
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
    if p["blur"] > 0:  # transitions between sounds slow down; mildly: real dysarthric speech is about as rough frame to frame as typical speech (E5)
        ls = np.log(sp + 1e-16)
        sp = np.exp((1 - p["blur"]) * ls + p["blur"] * gaussian_filter1d(ls, 2, axis=0))
    if p["omit"] or p["swap"] or p["devoice"]:
        f0, sp, ap, k = sound_errors(f0, sp, ap, seg, rng, p["omit"], p["swap"], p["devoice"])
        if len(k) < len(f0):
            f0, sp, ap, src = f0[k], sp[k], ap[k], src[k]
            seg = Seg(f0, sp)

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
        f0, sp, ap = retime(f0, sp, ap, q, 0.8 if RENDER == "world" else 0.0)
        src = np.interp(q, np.arange(len(src)), src)
        seg = Seg(f0, sp)
    floor = np.where(seg.e <= np.quantile(seg.e, 0.1))[0]
    cuts = {int(c): int(rng.integers(30, 120)) for c in seg.dips() if rng.random() < p["pause"]} if len(floor) and p["pause"] > 0 else {}
    if cuts:
        i = insert_pauses(None, seg.e, cuts, floor, rng)
        f0, sp, ap, src = f0[i], sp[i], ap[i], src[i]
        if p["breath"] > 0:
            sp, ap = add_breaths(f0, sp, ap, cuts, p["breath"], rng)
        seg = Seg(f0, sp)
    if p["rep"] or p["block"] or p["prolong"]:
        i = fluency(f0, sp, ap, seg, rng, p["rep"], p["block"], p["prolong"])
        f0, sp, ap, src = f0[i], sp[i], ap[i], src[i]
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
    if p["fatigue"] > 0:  # wearing down: duller and breathier toward the end
        tt = np.linspace(0, 1, n)[:, None]
        sp = sp * 10 ** (-6 * p["fatigue"] * tt * np.linspace(0, 1, sp.shape[1])[None] / 10)
        ap = np.where((f0 > 0)[:, None], np.minimum(1.0, ap + 0.3 * p["fatigue"] * tt), ap)
    if p["hard_onset"] > 0:  # hard glottal attack where voicing starts after a gap
        v = f0 > 0
        for a in np.where(v[1:] & ~v[:-1])[0] + 1:
            if a >= 10 and not v[a - 10:a].any() and rng.random() < p["hard_onset"]:
                ap[a:a + 4] = np.maximum(ap[a:a + 4], 0.6)
                sp[a:a + 4] = apply_tilt(sp[a:a + 4], 6.0) * 2.0
    if RENDER == "world":
        y = synth(f0, sp, ap)
    else:  # an artificial voice source has no counterpart in the recording: WORLD renders it
        y = render_transparent(x / 32768.0, orig, src, f0, sp, ap, rng, np.ones(len(f0), bool) if p["source"] in ("el", "eso") else None)
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
        tg = phonesim.read_textgrid(it["tg"]) if it.get("tg") else None
        errs = []
        y = simulate(x, cond, s, np.random.default_rng(who + 1), np.random.default_rng(_seed(seed, it["off"])), tg, errs)
        it["n_errors"] = len(errs) if tg else None
        return it, cond, s, y
    except Exception as e:
        print("skip", it["off"], repr(e), flush=True)
        return it, cond, s, None


_cache = {}
def _init(real_errors, render="transparent"):
    global RENDER
    RENDER = render
    if real_errors:
        phonesim.load_real(real_errors)


def _work(a):
    return _job((*a, _cache))


def gen(a):
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    np.random.default_rng(a.seed).shuffle(items)
    if a.max_src:  # slowing stretches clips; training drops anything over its --max_dur, so long sources would be made for nothing
        items = [it for it in items if it.get("n", it.get("duration", 0) * SR) / SR <= a.max_src]
    if a.hours:
        keep, tot = [], 0.0
        for it in items:
            if tot >= a.hours * 3600:
                break
            keep.append(it); tot += it.get("n", it.get("duration", 0) * SR) / SR
        items = keep
    if a.align_dir:  # phone alignments, one TextGrid per utterance, found by utterance name wherever they sit under align_dir
        tgs = {os.path.splitext(f)[0]: os.path.join(d, f) for d, _, fs in os.walk(a.align_dir) for f in fs if f.endswith(".TextGrid")}
        for it in items:
            it["tg"] = tgs.get(os.path.splitext(os.path.basename(it.get("wav", "")))[0], "")
        print(sum(1 for it in items if it["tg"]), "of", len(items), "clips have an alignment", flush=True)
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
    with open(pack, "ab") as fp, open(man, "a", encoding="utf-8", newline="\n") as fm, mp.Pool(a.workers, initializer=_init, initargs=(a.real_errors, a.render)) as pool:
        for k, (it, cond, s, y) in enumerate(pool.imap(_work, ((it, a.seed, conds, sevs) for it in items[done:]), chunksize=8)):
            if y is None:
                continue
            fp.write(y.astype("<i2").tobytes())
            r = {"wav": it.get("wav", ""), "text": it["text"], "duration": round(len(y) / SR, 2), "source": "sim", "speaker": f"{it.get('speaker', '')}:{cond}", "condition": cond,
                 "severity": round(s, 2), "pack": os.path.abspath(pack), "off": pos, "n": len(y)}
            if it.get("n_errors") is not None:
                r["sound_errors"] = it["n_errors"]
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
    tg = phonesim.read_textgrid(a.tg) if a.tg else None
    if a.real_errors:
        print(phonesim.load_real(a.real_errors), "measured speakers")
    for k, c in enumerate(CONDITIONS, 1):
        t0 = time.time()
        errs = []
        y = simulate(x, c, a.severity, np.random.default_rng(_seed(c, 1)), np.random.default_rng(_seed(c, 2)), tg, errs)
        sf.write(os.path.join(a.outdir, f"{k:02d}_{c}.wav"), y, SR)
        print(c, round(len(y) / SR, 2), "s", round(time.time() - t0, 2), "s compute", " ".join(f"{e[0][:3]}:{e[1]}>{e[2] or '-'}" for e in errs[:12]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("wav"); d.add_argument("outdir"); d.add_argument("--severity", type=float, default=0.7); d.add_argument("--tg", default="", help="the clip's TextGrid"); d.add_argument("--real_errors", default="")
    g = sub.add_parser("gen"); g.add_argument("--manifest", required=True); g.add_argument("--out", required=True); g.add_argument("--hours", type=float, default=0.0)
    g.add_argument("--workers", type=int, default=6); g.add_argument("--seed", type=int, default=1); g.add_argument("--conditions", default=""); g.add_argument("--severity", default="0.2,1.0")
    g.add_argument("--max_src", type=float, default=0.0, help="skip source clips longer than this many seconds (0: keep all); --hours counts source audio")
    g.add_argument("--real_errors", default="", help="real_errors.py output: sound errors from measured training speakers")
    g.add_argument("--align_dir", default="", help="phone alignments (TextGrid per utterance, e.g. zenodo.org/records/2619474): sound-level errors")
    for q in (d, g):
        q.add_argument("--render", default="transparent", choices=["transparent", "world"], help="transparent: change the recording itself (E16); "
                       "world: resynthesize every frame with the WORLD vocoder (everything before E16)")
    a = ap.parse_args()
    RENDER = a.render
    demo(a) if a.cmd == "demo" else gen(a)
