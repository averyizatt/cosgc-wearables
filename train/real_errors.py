# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Measure which sounds real dysarthric speakers get wrong, and how, to replace phonesim.py's hand-set error rates.

    python real_errors.py --manifest torgo_train.jsonl --cmudict cmudict.dict --out real_errors.json

TORGO's training speakers only. Not UASpeech: its test block is spoken by the same people as its training blocks, so habits learned there
would be the test speakers' own.

For every clip of the TRAINING speakers (never TORGO's test speakers F03 and M05, which only score finished models):
  1. a phoneme recognizer (facebook/wav2vec2-xlsr-53-espeak-cv-ft, IPA output) says which sounds it hears;
  2. CMUdict gives the intended sounds of the prompt;
  3. the two are aligned (edit distance), counting per intended sound: correct, replaced by which sound, deleted; and inserted sounds.
The recognizer itself makes errors, so the same is done for control speakers: their rates are the floor, and a dysarthric speaker's
error rate for a sound is counted only above that floor ("excess"). phonesim.load_real reads the result.

Needs a GPU environment with transformers (G:\\cosgc\\nemovenv). The model's vocabulary is read directly, so no phonemizer is needed.
"""
import argparse, collections, json, os, re, sys
import numpy as np, soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phonesim import base, VOWELS, CONS

TEST_SPEAKERS = {"F03", "M05"}  # TORGO test: never used to fit the simulator
# espeak IPA (as the recognizer writes it) -> ARPAbet; length and stress marks are stripped first
IPA = {"p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G", "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
       "ʃ": "SH", "ʒ": "ZH", "h": "HH", "tʃ": "CH", "dʒ": "JH", "m": "M", "n": "N", "ŋ": "NG", "l": "L", "ɫ": "L", "ɹ": "R", "r": "R", "ɾ": "T",
       "w": "W", "j": "Y", "ʔ": "T", "ɑ": "AA", "a": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH", "ɐ": "AH", "ɔ": "AO", "aʊ": "AW", "aɪ": "AY",
       "ɛ": "EH", "e": "EH", "ɜ": "ER", "ɝ": "ER", "ɚ": "ER", "eɪ": "EY", "ɪ": "IH", "ᵻ": "IH", "i": "IY", "oʊ": "OW", "o": "OW", "əʊ": "OW",
       "ɔɪ": "OY", "ʊ": "UH", "u": "UW", "ɒ": "AA", "ɪə": "IH", "ʊə": "UH", "eə": "EH", "x": "HH", "ç": "HH", "β": "V"}
PHONES = sorted(CONS | VOWELS)


def ipa_to_arpa(tok):
    t = re.sub(r"[ːˑˈˌ̩̯̃]", "", tok)
    if t in IPA:
        return IPA[t]
    if len(t) > 1 and t[0] in IPA:  # an unknown diacritic or offglide: keep the main sound
        return IPA[t[0]]
    return None


def load_cmudict(path):
    d = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith(";;;") or not line.strip():
            continue
        w, *ph = line.split()
        w = re.sub(r"\(\d+\)$", "", w).lower()
        d.setdefault(w, [base(p) for p in ph if not p.startswith("#")])
    return d


def align(ref, hyp):
    """Edit-distance alignment: list of (ref phone or None, hyp phone or None)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def audio(it):
    if "pack" in it:
        return np.array(np.memmap(it["pack"], dtype="<i2", mode="r")[it["off"]: it["off"] + it["n"]], dtype=np.float32) / 32768.0
    w, _ = sf.read(it["wav"], dtype="float32")
    return w.mean(1) if w.ndim > 1 else w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="comma-separated; items need speaker and group (control / dysarthric)")
    ap.add_argument("--cmudict", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    ap.add_argument("--max_per_speaker", type=int, default=1500)
    a = ap.parse_args()
    import torch
    from transformers import Wav2Vec2ForCTC
    from huggingface_hub import hf_hub_download
    vocab = json.load(open(hf_hub_download(a.model, "vocab.json"), encoding="utf-8"))
    inv = {i: t for t, i in vocab.items()}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(a.model).to(dev).eval()
    blank = model.config.pad_token_id
    cmu = load_cmudict(a.cmudict)

    items = [json.loads(l) for p in a.manifest.split(",") for l in open(p, encoding="utf-8") if l.strip()]
    items = [it for it in items if it.get("speaker") not in TEST_SPEAKERS and it.get("group") in ("control", "dysarthric")]
    per = collections.defaultdict(list)
    for it in items:
        per[it["speaker"]].append(it)
    rng = np.random.default_rng(0)
    stats = {}
    for spk, its in sorted(per.items()):
        if len(its) > a.max_per_speaker:
            its = [its[k] for k in sorted(rng.choice(len(its), a.max_per_speaker, replace=False))]
        st = {"group": its[0]["group"], "source": its[0].get("source", ""), "clips": 0, "n": collections.Counter(), "del": collections.Counter(),
              "sub": collections.defaultdict(collections.Counter), "ins": collections.Counter(), "ref_phones": 0}
        for it in its:
            words = re.sub(r"[^a-z' ]", " ", it["text"].lower()).split()
            if not words or any(w not in cmu for w in words):
                continue
            ref = [p for w in words for p in cmu[w]]
            x = audio(it)
            if len(x) < 1600:
                continue
            with torch.inference_mode():
                inp = torch.from_numpy((x - x.mean()) / (x.std() + 1e-7))[None].to(dev)
                ids = model(inp).logits[0].argmax(-1).tolist()
            toks, prev = [], None
            for i in ids:
                if i != prev and i != blank:
                    toks.append(inv.get(i, ""))
                prev = i
            hyp = [p for p in (ipa_to_arpa(t) for t in toks if t.strip() and t not in ("|", "<s>", "</s>", "<unk>")) if p]
            for r, h in align(ref, hyp):
                if r is None:
                    st["ins"][h] += 1
                    continue
                st["n"][r] += 1
                if h is None:
                    st["del"][r] += 1
                elif h != r:
                    st["sub"][r][h] += 1
            st["ref_phones"] += len(ref)
            st["clips"] += 1
        stats[spk] = st
        e = sum(st["del"].values()) + sum(sum(c.values()) for c in st["sub"].values())
        print(f"{spk:8s} {st['group']:10s} {st['clips']:5d} clips  phone error {100 * e / max(1, st['ref_phones']):5.1f} %", flush=True)

    # floor: what the recognizer gets wrong on control speakers
    ctrl = [s for s in stats.values() if s["group"] == "control"]
    def rates(group):
        n = collections.Counter(); d = collections.Counter(); sub = collections.defaultdict(collections.Counter)
        for s in group:
            n.update(s["n"]); d.update(s["del"])
            for r, c in s["sub"].items():
                sub[r].update(c)
        return n, d, sub
    cn, cd, cs = rates(ctrl)
    out = {"model": a.model, "floor_from": sorted(k for k, s in stats.items() if s["group"] == "control"), "speakers": {}}
    for spk, s in stats.items():
        if s["group"] != "dysarthric":
            continue
        ph = {}
        for r in PHONES:
            n = s["n"][r]
            if n < 5:
                continue
            fl_d = cd[r] / max(1, cn[r])
            fl_s = sum(cs[r].values()) / max(1, cn[r])
            sub = {q: c / n - cs[r][q] / max(1, cn[r]) for q, c in s["sub"][r].items()}
            sub = {q: round(v, 4) for q, v in sub.items() if v > 0}
            ph[r] = {"n": n, "del": round(max(0.0, s["del"][r] / n - fl_d), 4), "sub": sub,
                     "raw_del": round(s["del"][r] / n, 4), "raw_sub": round(sum(s["sub"][r].values()) / n, 4), "floor_sub": round(fl_s, 4)}
        tot = sum(s["n"].values())
        err = sum(s["del"].values()) + sum(sum(c.values()) for c in s["sub"].values())
        ctot = sum(cn.values()); cerr = sum(cd.values()) + sum(sum(c.values()) for c in cs.values())
        out["speakers"][spk] = {"source": s["source"], "clips": s["clips"], "phone_error": round(err / max(1, tot), 4),
                                "excess_error": round(err / max(1, tot) - cerr / max(1, ctot), 4),
                                "ins_rate": round(sum(s["ins"].values()) / max(1, tot), 4), "phones": ph}
    json.dump(out, open(a.out, "w"), indent=1)
    print("controls' phone error (the recognizer's floor):", round(100 * (sum(cd.values()) + sum(sum(c.values()) for c in cs.values())) / max(1, sum(cn.values())), 1), "%")
    for spk, v in sorted(out["speakers"].items(), key=lambda kv: kv[1]["excess_error"]):
        top = sorted(((r, q, p) for r, d in v["phones"].items() for q, p in d["sub"].items()), key=lambda t: -t[2])[:6]
        print(f"  {spk:8s} excess {100 * v['excess_error']:5.1f} %  top: " + ", ".join(f"{r}>{q} {100 * p:.0f}%" for r, q, p in top))


if __name__ == "__main__":
    main()
