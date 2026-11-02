# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Sound-level errors for dysim.py: with a phone alignment of the clip, say some sounds wrong the way dysarthric speakers do.

dysim.py's other traits change how the whole voice sounds; this changes which sound is said. Given the clip's phones (ARPAbet, from the
Montreal Forced Aligner alignments of LibriSpeech, McAuliffe et al. 2017; Lugosch et al. 2019, zenodo.org/records/2619474), it works on
the WORLD frames (pitch, spectral envelope, aperiodicity) and:

    substitutes   one sound for another: voicing (b -> p), place (s -> sh), manner (t -> s, s -> t), gliding (r -> w), nasal (b -> m);
                  the new sound takes this speaker's own version of it when the clip has one, else a rule-based change of the old one
    distorts      part of the way toward a substitute (the most frequent error in dysarthric speech)
    deletes       sounds, word-final consonants most often, and one consonant of a cluster
    inserts       a short schwa inside consonant clusters ("blue" -> "buh-lue")
    reduces       vowels toward schwa and diphthongs toward a single vowel ("ay" -> "aa")

The transcript stays the words that were meant: that is what the recognizer must learn to recover. Rates are per sound at severity 1, set by
hand from the clinical literature (Darley, Aronson & Brown 1975; Kent et al. 1989 and later intelligibility-test error studies), to be
calibrated against real speakers.
"""
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d

STOPS = {"P", "B", "T", "D", "K", "G"}
FRICS = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
AFFR = {"CH", "JH"}
NASAL = {"M", "N", "NG"}
LIQ = {"L", "R"}
GLIDE = {"W", "Y"}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
CONS = STOPS | FRICS | AFFR | NASAL | LIQ | GLIDE
VOICED = {"B", "D", "G", "V", "DH", "Z", "ZH", "JH", "M", "N", "NG", "L", "R", "W", "Y"} | VOWELS
VOICE_PAIR = {"P": "B", "T": "D", "K": "G", "F": "V", "TH": "DH", "S": "Z", "SH": "ZH", "CH": "JH"}
VOICE_PAIR.update({v: k for k, v in list(VOICE_PAIR.items())})
PLACE = {"S": ("SH", "TH"), "Z": ("ZH", "DH"), "SH": ("S",), "ZH": ("Z",), "TH": ("F", "S"), "DH": ("D", "V"), "F": ("TH",),
         "K": ("T",), "G": ("D",), "T": ("K",), "D": ("G",), "CH": ("SH",), "JH": ("ZH",)}
SPIRANT = {"P": "F", "B": "V", "T": "S", "D": "Z", "K": "HH", "G": "HH"}  # weak closure: a stop becomes a fricative
STOPPING = {"F": "P", "V": "B", "S": "T", "Z": "D", "TH": "T", "DH": "D", "SH": "CH", "ZH": "JH", "CH": "T", "JH": "D"}
GLIDING = {"R": "W", "L": "W"}
NASALIZE = {"B": "M", "D": "N", "G": "NG", "P": "M", "T": "N", "K": "NG"}  # velopharyngeal weakness (flaccid)
DIPH = {"AY": "AA", "EY": "EH", "OW": "AO", "AW": "AA", "OY": "AO"}
LAX = {"IY": "IH", "UW": "UH", "ER": "AH"}
FRIC_WARP = {("S", "SH"): 0.65, ("Z", "ZH"): 0.65, ("SH", "S"): 1.5, ("ZH", "Z"): 1.5, ("TH", "S"): 1.4, ("S", "TH"): 0.85, ("F", "TH"): 1.1, ("TH", "F"): 0.9}

DYSARTHRIA = ("flaccid", "spastic", "ataxic", "hypokinetic", "hyperkinetic", "uumn", "als", "cp")


def base(p):
    return re.sub(r"\d", "", p).upper()


# ------------------------------------------------------------------ alignments
def read_textgrid(path):
    """{tier name: [(start s, end s, label), ...]} from a Praat TextGrid (long or short text format), empty labels dropped."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    tiers, out = re.split(r'(?:item \[\d+\]:|"IntervalTier")', txt)[1:], {}
    for t in tiers:
        name = re.search(r'name = "([^"]*)"', t) or re.search(r'^\s*"([^"]*)"', t)
        if not name:
            continue
        iv = re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', t)
        if not iv:  # short format: start, end, "label" on consecutive lines
            iv = re.findall(r'\n([\d.]+)\s*\n([\d.]+)\s*\n"([^"]*)"', t)
        out[name.group(1).lower()] = [(float(a), float(b), l.strip()) for a, b, l in iv if l.strip() and l.strip() not in ("sil", "sp", "spn")]
    return out


def phone_frames(tg, n_frames, fps):
    """Phones as frame ranges, each with its word index and position: [(a, b, phone, word, first, last)]."""
    words = tg.get("words", [])
    out = []
    for s, e, l in tg.get("phones", []):
        a, b = int(round(s * fps)), int(round(e * fps))
        a, b = max(0, min(a, n_frames)), max(0, min(b, n_frames))
        if b <= a:
            continue
        mid = (s + e) / 2
        w = next((k for k, (ws, we, _) in enumerate(words) if ws <= mid < we), -1)
        out.append([a, b, base(l), w])
    for k, ph in enumerate(out):
        ph.append(k == 0 or out[k - 1][3] != ph[3])           # first sound of its word
        ph.append(k == len(out) - 1 or out[k + 1][3] != ph[3])  # last sound of its word
    return [tuple(p) for p in out]


# ------------------------------------------------------------------ error profiles
def profile(cond, s, rng):
    """Per-sound error rates at severity s for one simulated speaker. Zero for conditions that are not about articulation."""
    z = dict(subst=0.0, distort=0.0, delete=0.0, final=0.0, cluster=0.0, epen=0.0, vowel=0.0, diph=0.0,
             voicing=1.0, place=1.0, spirant=1.0, stopping=1.0, gliding=1.0, nasal=0.0, devoice_bias=0.5)
    if cond not in DYSARTHRIA and cond != "apraxia":
        return z
    u = lambda lo, hi: rng.uniform(lo, hi) * s
    z.update(subst=u(.05, .16), distort=u(.15, .4), delete=u(.02, .07), final=u(.08, .3), cluster=u(.15, .45), epen=u(.0, .06),
             vowel=u(.1, .35), diph=u(.15, .5))
    if cond == "flaccid":    # weak closure and velum: stops leak into fricatives and nasals
        z.update(spirant=3.0, nasal=2.0, stopping=0.3)
    elif cond == "spastic":  # voicing errors (often devoicing), imprecise consonants
        z.update(voicing=3.0, devoice_bias=0.7, spirant=1.5)
    elif cond == "ataxic":   # vowels most of all, consonants irregularly
        z.update(vowel=z["vowel"] * 1.6, diph=z["diph"] * 1.3, subst=z["subst"] * 0.6, distort=z["distort"] * 1.2)
    elif cond == "hypokinetic":  # undershoot: stops become fricatives, final sounds fade
        z.update(spirant=4.0, stopping=0.2, final=z["final"] * 1.5)
    elif cond == "hyperkinetic":
        z.update(distort=z["distort"] * 1.3, vowel=z["vowel"] * 1.3)
    elif cond == "uumn":     # mild: mostly distortion
        z.update(subst=z["subst"] * 0.5, distort=z["distort"] * 1.2, delete=z["delete"] * 0.5)
    elif cond == "als":
        z.update(spirant=2.5, nasal=1.5, voicing=2.0, gliding=2.0)
    elif cond == "cp":
        z.update(voicing=2.5, gliding=2.0, spirant=1.5)
    elif cond == "apraxia":  # inconsistent substitutions, groping, inserted vowels
        z.update(subst=z["subst"] * 1.5, epen=z["epen"] * 2 + u(.03, .08), voicing=2.0, place=2.0)
    # A dysarthric speaker makes the same errors again and again: each sound gets this speaker's own susceptibility (a few sounds are
    # their trouble sounds, most are fine) and one habitual substitute. Apraxia is the exception: its errors vary from one attempt to the next.
    z["consistency"] = 0.25 if cond == "apraxia" else rng.uniform(0.7, 0.9)
    z["weight"] = {ph: float(rng.lognormal(0.0, 1.0)) for ph in sorted(CONS | VOWELS)}
    m = np.mean(list(z["weight"].values()))
    z["weight"] = {ph: w / m for ph, w in z["weight"].items()}  # mean 1: the overall rate stays the profile's
    z["habit"] = {ph: pick_substitute(ph, z, rng) for ph in sorted(CONS | VOWELS)}
    return z


def pick_substitute(ph, z, rng):
    """A wrong sound for ph, weighted by the condition's error kinds; None if nothing applies."""
    opts = []
    if ph in VOICE_PAIR:
        w = z["voicing"] * (z["devoice_bias"] if ph in VOICED else 1 - z["devoice_bias"]) * 2
        opts.append((VOICE_PAIR[ph], w))
    for q in PLACE.get(ph, ()):
        opts.append((q, z["place"] / len(PLACE[ph])))
    if ph in SPIRANT:
        opts.append((SPIRANT[ph], z["spirant"]))
    if ph in STOPPING:
        opts.append((STOPPING[ph], z["stopping"]))
    if ph in GLIDING:
        opts.append((GLIDING[ph], z["gliding"]))
    if ph in NASALIZE and z["nasal"] > 0:
        opts.append((NASALIZE[ph], z["nasal"]))
    if ph in DIPH:
        opts.append((DIPH[ph], 1.0))
    if ph in LAX:
        opts.append((LAX[ph], 1.0))
    if not opts:
        return None
    w = np.array([o[1] for o in opts], float)
    return opts[int(rng.choice(len(opts), p=w / w.sum()))][0]


# ------------------------------------------------------------------ rendering in the WORLD domain
def _exemplars(ls, f0, ap, phones):
    """This speaker's own version of each sound in the clip: mean log envelope and aperiodicity over its middle frames."""
    acc = {}
    for a, b, ph, *_ in phones:
        m = max(1, (b - a) // 4)
        acc.setdefault(ph, []).append((ls[a + m - 1:b - m + 1] if b - a > 2 else ls[a:b], ap[a + m - 1:b - m + 1] if b - a > 2 else ap[a:b]))
    return {ph: (np.concatenate([x[0] for x in v]).mean(0), np.concatenate([x[1] for x in v]).mean(0)) for ph, v in acc.items()}


def _voiced_f0(f0, a, b):
    """A pitch track for frames a..b that were unvoiced: interpolated between the nearest voiced frames."""
    v = np.where(f0 > 0)[0]
    if not len(v):
        return np.full(b - a, 120.0)
    return np.interp(np.arange(a, b), v, f0[v])


def _render(ls, f0, ap, a, b, src, dst, ex, amount, rng, warp):
    """Move frames a..b from sound src toward sound dst by amount (1 = full substitution). Works in place."""
    n = b - a
    edge = np.minimum(1.0, np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)) / 2.0)[:, None]  # soften the first and last frame
    k = amount * edge
    if dst in ex:  # this speaker's own version of the target sound
        tgt, tap = ex[dst]
        ls[a:b] = (1 - k) * ls[a:b] + k * tgt[None]
        ap[a:b] = (1 - k) * ap[a:b] + k * tap[None]
    elif (src, dst) in FRIC_WARP or (src in FRICS and dst in FRICS):  # place change: move the noise peak
        f = FRIC_WARP.get((src, dst), rng.uniform(0.7, 1.4))
        w = np.log(warp(np.exp(ls[a:b]), f) + 1e-16)
        ls[a:b] = (1 - k) * ls[a:b] + k * w
    elif src in STOPS and dst in FRICS | {"HH"}:  # spirantization: no closure, the burst noise is drawn out
        noise = ls[a:b].max(0) - 3.0
        ls[a:b] = (1 - k) * ls[a:b] + k * noise[None]
        ap[a:b] = (1 - k) * ap[a:b] + k
    elif src in FRICS | AFFR and dst in STOPS | AFFR:  # stopping: silence, then a short burst
        cut = a + max(1, int(n * 0.6))
        ls[a:cut] = (1 - k[: cut - a]) * ls[a:cut] + k[: cut - a] * (ls[a:cut] - 5.0)
    elif dst in NASAL:  # nasal murmur: low energy, low pole, voiced
        mur = ls[a:b].mean(0) - 2.0
        mur[: mur.size // 8] += 1.5
        mur[mur.size // 4:] -= 2.0
        ls[a:b] = (1 - k) * ls[a:b] + k * mur[None]
    elif dst in VOWELS or dst in GLIDE:  # vowel or glide toward the clip's average voiced envelope, smoothed (a weak, neutral sound)
        neutral = gaussian_filter1d(ls[a:b].mean(0), 3)
        ls[a:b] = (1 - k) * ls[a:b] + k * neutral[None]
    # voicing follows the target
    if amount >= 0.5:
        if dst in VOICED and src not in VOICED:
            f0[a:b] = _voiced_f0(f0, a, b)
            ap[a:b] = np.minimum(ap[a:b], 0.5)
        elif dst not in VOICED and src in VOICED:
            f0[a:b] = 0.0
            ap[a:b] = 1.0


def apply(f0, sp, ap, phones, z, rng, warp):
    """f0, sp, ap: WORLD frames; phones from phone_frames. Returns new frames (deletions and insertions change the length) and a list of
    the errors made, as (kind, sound, new sound)."""
    if not phones or not any(z[k] for k in ("subst", "distort", "delete", "final", "cluster", "epen", "vowel", "diph")):
        return f0, sp, ap, []
    f0, ap = f0.copy(), ap.copy()
    ls = np.log(sp + 1e-16)
    ex = _exemplars(ls, f0, ap, phones)
    keep = np.ones(len(f0), bool)
    insert = {}  # frame -> number of schwa frames to insert before it
    log = []
    def pick(ph):  # this speaker's habitual error for ph, or now and then another one
        hab = z.get("habit", {}).get(ph)
        return hab if hab and rng.random() < z.get("consistency", 0.0) else pick_substitute(ph, z, rng)

    for k, (a, b, ph, w, first, last) in enumerate(phones):
        sw = min(4.0, z.get("weight", {}).get(ph, 1.0))  # trouble sounds go wrong far more often than the rest
        nxt = phones[k + 1] if k + 1 < len(phones) else None
        prv = phones[k - 1] if k > 0 else None
        in_cluster = ph in CONS and ((nxt and nxt[2] in CONS and nxt[3] == w) or (prv and prv[2] in CONS and prv[3] == w))
        if ph in CONS:
            p_del = z["delete"] + (z["final"] if last and not first else 0) + (z["cluster"] * 0.5 if in_cluster else 0)
            if rng.random() < p_del * sw:
                keep[a:b] = False
                log.append(("delete", ph, ""))
                continue
            if in_cluster and nxt and nxt[2] in CONS and nxt[3] == w and rng.random() < z["epen"]:
                insert[b] = int(rng.integers(8, 17))  # 40-80 ms
                log.append(("insert", ph, "AH"))
        r = rng.random()
        if ph in VOWELS:
            if ph in DIPH and rng.random() < z["diph"] * sw:
                _render(ls, f0, ap, a + (b - a) // 3, b, ph, DIPH[ph], ex, 1.0, rng, warp)  # the glide is left out
                log.append(("reduce", ph, DIPH[ph]))
            elif ph != "AH" and rng.random() < z["vowel"] * sw:
                _render(ls, f0, ap, a, b, ph, "AH", ex, rng.uniform(0.4, 0.9), rng, warp)
                log.append(("centralize", ph, "AH"))
            continue
        if r < (z["subst"] + z["distort"]) * sw:
            dst = pick(ph)
            if dst is None:
                continue
            full = r < z["subst"] * sw
            _render(ls, f0, ap, a, b, ph, dst, ex, 1.0 if full else rng.uniform(0.3, 0.7), rng, warp)
            log.append(("substitute" if full else "distort", ph, dst))
    sp = np.exp(ls)
    idx = []
    schwa = ex.get("AH")
    for t in range(len(f0)):
        if t in insert:
            idx.extend([-1 - t] * insert[t])  # negative: a schwa frame placed before frame t
        if keep[t]:
            idx.append(t)
    if not idx:
        return f0, sp, ap, log
    idx = np.array(idx)
    real = np.where(idx >= 0, idx, np.minimum(-idx - 1, len(f0) - 1))
    f0n, spn, apn = f0[real].copy(), sp[real].copy(), ap[real].copy()
    ins = idx < 0
    if ins.any():
        if schwa is not None:
            spn[ins] = np.exp(schwa[0])[None]
            apn[ins] = schwa[1][None]
        else:
            spn[ins] = np.exp(gaussian_filter1d(ls[f0 > 0].mean(0) if (f0 > 0).any() else ls.mean(0), 3))[None]
            apn[ins] = 0.2
        vf = f0[f0 > 0]
        f0n[ins] = np.median(vf) if len(vf) else 120.0
    return f0n, spn, apn, log
