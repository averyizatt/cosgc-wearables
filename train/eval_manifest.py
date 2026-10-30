#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Word error rate of a model on a manifest, split by speaker, group, microphone and prompt kind.

    python eval_manifest.py --nemo <dir with state_dict_plain.pt> [--ckpt last.pt] [--init init.pt] \\
        --manifest torgo_test.jsonl --out results.json [--modes full,stream]

--ckpt: weights from train_nemo_stream.py (its "model" entry); without it the --nemo weights are evaluated.
The fixed feature normalization comes from the checkpoint, else from --init (tnm_to_ckpt.py output).
Modes: full = whole utterance at once (the accurate mode); stream = 32-frame chunks with 128 frames of history, what the
device runs. Greedy decoding (no language model), the same arithmetic as the trainer's validation, so numbers here and
in training logs are comparable; the device with its LM does a little better.
Writes per-utterance hypotheses next to the summary so errors can be listened to.
"""
import argparse, collections, json, os, sys
import torch, sentencepiece as spm, soundfile as sf, jiwer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nemo_small import NemoSmall
from train_nemo_stream import BatchPre, SModel, ctc_greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="full,stream")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--left", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only every n-th clip (quick look)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(f"{a.nemo}/state_dict_plain.pt", map_location="cpu")
    m = NemoSmall(sd)
    src = {}
    if a.ckpt:
        src = torch.load(a.ckpt, map_location="cpu")
        m.load_state_dict(src["model"])
    norm = src if src.get("norm_mean") is not None else (torch.load(a.init, map_location="cpu") if a.init else {})
    pre = BatchPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"],
                   norm.get("norm_mean"), norm.get("norm_std")).to(dev)
    net = SModel(m).to(dev).eval()
    sp = spm.SentencePieceProcessor(model_file=f"{a.nemo}/tokenizer.model")
    V = sp.get_piece_size()
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    if a.limit:
        items = items[::a.limit]
    modes = a.modes.split(",")
    rows = []
    with torch.no_grad():
        for n, it in enumerate(items):
            w, _ = sf.read(it["wav"], dtype="float32")
            if w.ndim > 1:
                w = w.mean(1)
            f, fl = pre(torch.from_numpy(w)[None].to(dev), torch.tensor([len(w)], device=dev))
            r = dict(it)
            for mode in modes:
                ch, lf = (0, 0) if mode == "full" else (a.chunk, a.left)
                lg, ol = net(f, fl, ch, lf)
                r["hyp_" + mode] = sp.decode(ctc_greedy(lg[0], int(ol[0]), V))
            rows.append(r)
            if n % 500 == 0:
                print(f"{n}/{len(items)}", flush=True)

    def wer(rs, mode):
        return 100 * jiwer.wer([r["text"] for r in rs], [r["hyp_" + mode] or "<empty>" for r in rs])

    summary = {"model": a.ckpt or a.nemo, "manifest": a.manifest, "clips": len(rows), "decoding": "greedy, no LM"}
    for key in ("group", "speaker", "mic", "kind"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r.get(key, "?")].append(r)
        summary["by_" + key] = {g: {mode: round(wer(rs, mode), 2) for mode in modes} | {"clips": len(rs)}
                               for g, rs in sorted(groups.items())}
    summary["all"] = {mode: round(wer(rows, mode), 2) for mode in modes}
    json.dump(summary, open(a.out, "w"), indent=1)
    with open(os.path.splitext(a.out)[0] + "_utts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_speaker"}, indent=1))
    print("by speaker:")
    for s, v in summary["by_speaker"].items():
        print(f"  {s:5s} " + "  ".join(f"{mode} {v[mode]:6.1f}%" for mode in modes) + f"  ({v['clips']} clips)")


if __name__ == "__main__":
    main()
