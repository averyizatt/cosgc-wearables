"""Score an off-the-shelf NVIDIA model (Parakeet) on our manifests, for a like-for-like comparison with our on-device model.

The Interspeech 2025 Speech Accessibility Project winner fine-tuned nvidia/parakeet-tdt-1.1b; scoring the same model and a smaller one
(nvidia/parakeet-tdt_ctc-110m) untuned on TORGO test, UASpeech and LibriSpeech places our 19 M model on a size curve. Needs NeMo, so it runs
in its own environment (G:\\cosgc\\nemovenv), not the training one. Output has eval_manifest.py's format, mode "full", greedy decoding, no LM.

    python eval_parakeet.py --model nvidia/parakeet-tdt-1.1b --manifest torgo_test.jsonl --out eval/pk11_torgo_test.json
"""
import argparse, collections, json, os, re, time
import numpy as np, soundfile as sf, jiwer, torch


def norm(t):
    t = t.lower().replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", "", t)).strip()


def audio(it):
    if "pack" in it:
        return np.array(np.memmap(it["pack"], dtype="<i2", mode="r")[it["off"]: it["off"] + it["n"]], dtype=np.float32) / 32768.0
    w, _ = sf.read(it["wav"], dtype="float32")
    return w.mean(1) if w.ndim > 1 else w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-1.1b")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="every n-th clip only")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(a.model).eval()
    if torch.cuda.is_available():
        m = m.cuda()
    params = sum(p.numel() for p in m.parameters())
    print(a.model, round(params / 1e6, 1), "M parameters", flush=True)
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.limit:
        items = items[::a.limit]
    rows, t0 = [], time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        for i in range(0, len(items), a.batch):
            part = items[i:i + a.batch]
            hyps = m.transcribe([audio(it) for it in part], batch_size=a.batch, verbose=False)
            hyps = hyps[0] if isinstance(hyps, tuple) else hyps
            for it, h in zip(part, hyps):
                r = dict(it)
                r["text"] = norm(it["text"])
                r["hyp_full"] = norm(h.text if hasattr(h, "text") else h)
                rows.append(r)
            if i % (50 * a.batch) == 0:
                print(f"{i}/{len(items)} {time.time() - t0:.0f}s", flush=True)

    def wer(rs):
        return 100 * jiwer.wer([r["text"] or "<empty>" for r in rs], [r["hyp_full"] or "<empty>" for r in rs])

    summary = {"model": a.model, "params_M": round(params / 1e6, 1), "manifest": a.manifest, "clips": len(rows), "decoding": "model default, no LM"}
    for key in ("group", "speaker", "mic", "kind"):
        g = collections.defaultdict(list)
        for r in rows:
            g[r.get(key, "?")].append(r)
        summary["by_" + key] = {k: {"full": round(wer(rs), 2), "clips": len(rs)} for k, rs in sorted(g.items())}
    summary["all"] = {"full": round(wer(rows), 2)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    with open(os.path.splitext(a.out)[0] + "_utts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_speaker"}, indent=1))


if __name__ == "__main__":
    main()
