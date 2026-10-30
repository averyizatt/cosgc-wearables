# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Acoustic measurements that separate dysarthric from typical speech, to calibrate and check simdys.py.

    python dys_stats.py --nemo <dir> --init init.pt --out stats.json \\
        --set ls=ls_testclean_train.jsonl --set torgo=torgo_test.jsonl [--sim ls:mild,moderate,severe] [--n 300]

For each clip, on the log-mel features (10 ms frames), with leading and trailing silence trimmed:
  rate      letters per second of speaking time (time not in pauses); dysarthric speech is slower
  pause     share of the trimmed clip that is pause (frames more than 30 dB below the clip's loud frames)
  flux      mean change between neighbouring frames during speech (dB per frame, averaged over mel bands);
            lower = smoother, slurred transitions between sounds
  hilo      energy of the top 20 mel bands minus the bottom 20, during speech (dB); lower = duller consonants
  trem      share of the loudness contour's modulation energy (0.5-20 Hz) that lies at 3-8 Hz (tremor band)
Results are grouped by set, then by the manifest's group (dysarthric / control), speaker and kind (word / sentence),
as medians. --sim adds copies of a set distorted by simdys.py at the named severities.
"""
import argparse, collections, json, os, random, re, sys
import numpy as np, soundfile as sf, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_nemo_stream import BatchPre

DB = 10 / np.log(10)  # natural-log power -> dB


def measure(mel, text):
    """mel (T, 80) natural-log power -> dict of measurements, or None if there is too little speech."""
    m = mel.cpu().numpy() * DB
    e = np.log10(np.exp(mel.cpu().numpy()).sum(1) + 1e-10) * 10  # frame energy, dB
    loud = np.percentile(e, 95)
    sp = e > loud - 30
    idx = np.flatnonzero(sp)
    if len(idx) < 20:
        return None
    a, b = idx[0], idx[-1] + 1
    sp, m, e = sp[a:b], m[a:b], e[a:b]
    speak = sp.sum() / 100.0
    letters = len(re.sub(r"[^a-z]", "", text))
    d = np.abs(np.diff(m, axis=0)).mean(1)
    both = sp[1:] & sp[:-1]
    env = e - e.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fr = np.fft.rfftfreq(len(env), 0.01)
    band = spec[(fr >= 0.5) & (fr <= 20)].sum() + 1e-10
    return {"rate": letters / speak, "pause": 1 - sp.mean(), "flux": float(d[both].mean()) if both.any() else np.nan,
            "hilo": float(m[sp, 60:].mean() - m[sp, :20].mean()),
            "trem": float(spec[(fr >= 3) & (fr <= 8)].sum() / band) if len(env) >= 100 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--set", action="append", required=True, help="name=manifest.jsonl")
    ap.add_argument("--sim", default="", help="set:severity,severity  (simulate dysarthria on that set)")
    ap.add_argument("--n", type=int, default=300, help="clips per set (random, fixed seed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(f"{a.nemo}/state_dict_plain.pt", map_location="cpu")
    ini = torch.load(a.init, map_location="cpu")
    pre = BatchPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"]).to(dev)
    sims = {}
    if a.sim:
        s, sev = a.sim.split(":")
        sims[s] = sev.split(",")
    rows = []
    for spec in a.set:
        name, path = spec.split("=", 1)
        items = [json.loads(l) for l in open(path, encoding="utf-8")]
        random.Random(0).shuffle(items)
        variants = [("", None)] + [(f"+sim_{v}", v) for v in sims.get(name, [])]
        for it in items[: a.n]:
            w, _ = sf.read(it["wav"], dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            with torch.no_grad():
                mel = pre.logmel(torch.from_numpy(w)[None].to(dev))[0, :, : len(w) // 160].T  # (T, 80)
            for suffix, sev in variants:
                x = mel
                if sev:
                    from simdys import distort_severity
                    x = distort_severity(mel, sev, random.Random(hash((it["wav"], sev)) & 0xFFFFFFFF))
                r = measure(x, it["text"])
                if r:
                    r.update(set=name + suffix, group=it.get("group", "-"), speaker=it.get("speaker", "-"),
                             kind=it.get("kind", "-"))
                    rows.append(r)
    keys = ("rate", "pause", "flux", "hilo", "trem")
    out = {}
    for by in (("set",), ("set", "group"), ("set", "group", "kind"), ("set", "speaker")):
        g = collections.defaultdict(list)
        for r in rows:
            g[" / ".join(r[k] for k in by)].append(r)
        out[" + ".join(by)] = {k: {m: round(float(np.nanmedian([r[m] for r in v])), 3) for m in keys} | {"n": len(v)}
                               for k, v in sorted(g.items())}
    json.dump(out, open(a.out, "w"), indent=1)
    for by, tab in out.items():
        print(f"\n== by {by}")
        print(f"{'':38s}" + "".join(f"{m:>8s}" for m in keys) + "       n")
        for k, v in tab.items():
            print(f"{k:38s}" + "".join(f"{v[m]:8.3f}" for m in keys) + f"{v['n']:8d}")


if __name__ == "__main__":
    main()
