#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""WER of the C engine (tasr_cli, the device arithmetic) on a packed manifest, with or without the language model.

    python3 eval_cli.py model.tnm torgo_test.jsonl --out res.json [--stream] [--lm ../../models/nemo_lm.tlm] [--beam 4]
                        [--lm_weight w] [--token_bonus b] [--jobs 8] [--wavs DIR]

Standard library only (no numpy), so it runs where the host engine is built. The manifest is one made by
train/pack_audio.py (fields pack, off, n); its .pack file is looked up next to the manifest if the stored path does not
exist (copied from the training PC). Clips are written once as WAV files to --wavs. WER is corpus-level (edit distance
summed over clips / reference words), per speaker, group, microphone and kind like train/eval_manifest.py, with the
same text normalization.
"""
import argparse, collections, json, os, re, subprocess, sys, wave
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(t):  # train/data.py clean_text
    t = t.lower().replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t)).strip()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--lm")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--lm_weight", type=float)
    ap.add_argument("--token_bonus", type=float)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--wavs", default=os.path.expanduser("~/eval_cli_wavs"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0, help="with --stream: encoder frames per chunk (0: stored in the model)")
    ap.add_argument("--left", type=int, default=0, help="with --stream: left context in frames (0: stored in the model)")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[:: max(1, len(items) // a.limit)][: a.limit]
    os.makedirs(a.wavs, exist_ok=True)
    packs = {}
    for k, it in enumerate(items):
        it["_wav"] = os.path.join(a.wavs, f"{os.path.splitext(os.path.basename(a.manifest))[0]}_{it['off']}.wav")
        if os.path.exists(it["_wav"]):
            continue
        p = it["pack"] if os.path.exists(it["pack"]) else os.path.join(os.path.dirname(os.path.abspath(a.manifest)),
                                                                       os.path.basename(it["pack"].replace("\\", "/")))
        if p not in packs:
            packs[p] = open(p, "rb")
        packs[p].seek(2 * it["off"])
        with wave.open(it["_wav"], "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(packs[p].read(2 * it["n"]))
    cli = [os.path.join(HERE, "tasr_cli"), a.model] + (["--stream", "--nchunk", str(a.chunk), "--nleft", str(a.left)] if a.stream else [])
    if a.lm:
        cli += ["--lm", a.lm, "--beam", str(a.beam)]
        cli += ["--lm_weight", str(a.lm_weight)] if a.lm_weight is not None else []
        cli += ["--token_bonus", str(a.token_bonus)] if a.token_bonus is not None else []
    batches = [items[i::a.jobs] for i in range(a.jobs)]

    def run(batch):
        out = subprocess.run(cli + [it["_wav"] for it in batch], capture_output=True, text=True).stdout
        return dict(l.split("\t", 1) if "\t" in l else (l, "") for l in out.splitlines())

    hyp = {}
    with ThreadPoolExecutor(a.jobs) as ex:
        for h in ex.map(run, batches):
            hyp.update(h)
    acc = collections.defaultdict(lambda: [0, 0, 0])  # edits, ref words, clips
    utts = []
    for it in items:
        r, h = norm(it["text"]).split(), norm(hyp.get(it["_wav"], "")).split()
        e = edits(r, h)
        utts.append({"wav": it["wav"], "ref": " ".join(r), "hyp": " ".join(h), "edits": e})
        keys = ["all"] + [f"by_{k}/{it[k]}" for k in ("group", "speaker", "mic", "kind") if k in it]
        for key in keys:
            v = acc[key]; v[0] += e; v[1] += len(r); v[2] += 1
    res = {"model": a.model, "manifest": a.manifest, "clips": len(items), "missing": sum(it["_wav"] not in hyp for it in items),
           "mode": "stream" if a.stream else "full",
           "decoding": f"beam {a.beam} + LM {os.path.basename(a.lm)} (weight {a.lm_weight}, bonus {a.token_bonus}; None = engine default)" if a.lm else "greedy"}
    for key, (e, n, c) in sorted(acc.items()):
        res[key] = {"wer": round(100 * e / max(1, n), 2), "clips": c}
    json.dump(res, open(a.out, "w"), indent=1)
    with open(os.path.splitext(a.out)[0] + "_utts.jsonl", "w") as f:
        for u in utts:
            f.write(json.dumps(u) + "\n")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("by_mic")}, indent=1))


if __name__ == "__main__":
    main()
