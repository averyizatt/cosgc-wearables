# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Voice conversion with kNN-VC (Baas, van Niekerk & Kamper 2023): every frame of the source is rebuilt from the closest frames of a target
speaker's recordings, so the result speaks the source's words with the target's voice and way of making sounds.

    python knnvc_gen.py to_dys   --source ls100_train.jsonl --targets torgo_train.jsonl --out G:/x/vc_todys --hours 50
    python knnvc_gen.py from_dys --source torgo_train.jsonl --targets ls100_train.jsonl --out G:/x/vc_fromdys --per_source 20

to_dys:   LibriSpeech words, a real dysarthric speaker's voice and sound realizations (then dysim.py may add timing on top). Many new
          sentences, but only as many dysarthric voices as there are training speakers.
from_dys: a real dysarthric recording, re-voiced by many LibriSpeech speakers. The timing and articulation errors stay real (the part rules
          miss); only the voice changes. The text is the original prompt.

Targets are TRAINING speakers only. Dysarthric targets come from TORGO's training speakers; UASpeech is excluded because its test block is
spoken by the same people. Output: a .pack and .jsonl as dysim.py writes them, which the trainer reads directly. Needs a GPU
(G:\\cosgc\\nemovenv); the model comes from torch.hub (bshall/knn-vc, WavLM-Large and a prematched HiFi-GAN, about 1.3 GB).
"""
import argparse, collections, json, os, sys, time
import numpy as np

TEST_SPEAKERS = {"F03", "M05"}
SR = 16000


def audio(it):
    if "pack" in it:
        return np.array(np.memmap(it["pack"], dtype="<i2", mode="r")[it["off"]: it["off"] + it["n"]], dtype=np.float32) / 32768.0
    import soundfile as sf
    w, _ = sf.read(it["wav"], dtype="float32")
    return w.mean(1) if w.ndim > 1 else w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["to_dys", "from_dys"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=0.0, help="to_dys: source hours to convert")
    ap.add_argument("--per_source", type=int, default=10, help="from_dys: new voices per dysarthric clip")
    ap.add_argument("--match_min", type=float, default=6.0, help="minutes of target audio in each matching set")
    ap.add_argument("--n_targets", type=int, default=300, help="from_dys: LibriSpeech voices to use")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
    rng = np.random.default_rng(a.seed)
    src = [json.loads(l) for l in open(a.source, encoding="utf-8") if l.strip()]
    tgt = [json.loads(l) for l in open(a.targets, encoding="utf-8") if l.strip()]
    tgt = [t for t in tgt if t.get("speaker") not in TEST_SPEAKERS]
    if a.mode == "to_dys":
        tgt = [t for t in tgt if t.get("group") == "dysarthric"]
        rng.shuffle(src)
        if a.hours:
            keep, tot = [], 0.0
            for it in src:
                if tot >= a.hours * 3600:
                    break
                keep.append(it); tot += it.get("duration", 0)
            src = keep
    else:
        src = [s for s in src if s.get("group") == "dysarthric" and s.get("speaker") not in TEST_SPEAKERS]
    by_t = collections.defaultdict(list)
    for t in tgt:
        by_t[t["speaker"]].append(t)
    speakers = sorted(by_t)
    if a.mode == "from_dys" and len(speakers) > a.n_targets:
        speakers = sorted(rng.choice(speakers, a.n_targets, replace=False).tolist())
    # which source clip goes to which target voice
    jobs = collections.defaultdict(list)
    for it in src:
        for _ in range(1 if a.mode == "to_dys" else a.per_source):
            jobs[speakers[int(rng.integers(len(speakers)))]].append(it)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pack, man = a.out + ".pack", a.out + ".jsonl"
    done = set()
    if os.path.exists(man):  # resume: target speakers already finished are skipped (rows are written per target)
        rows = [json.loads(l) for l in open(man, encoding="utf-8") if l.strip()]
        done = {r["vc_target"] for r in rows}
        pos = rows[-1]["off"] + rows[-1]["n"] if rows else 0
        with open(pack, "r+b") as f:
            f.truncate(pos * 2)
    else:
        pos = 0
    t0, out_s = time.time(), 0.0
    with open(pack, "ab") as fp, open(man, "a", encoding="utf-8", newline="\n") as fm:
        for k, spk in enumerate(speakers):
            if spk in done or not jobs[spk]:
                continue
            pool = list(by_t[spk]); rng.shuffle(pool)
            mats, tot = [], 0.0
            for t in pool:  # this speaker's matching set: several minutes of their own audio
                if tot >= a.match_min * 60:
                    break
                x = audio(t)
                mats.append(torch.from_numpy(x)[None]); tot += len(x) / SR
            with torch.inference_mode():
                ms = vc.get_matching_set(mats)
                rows = []
                for it in jobs[spk]:
                    x = audio(it)
                    if len(x) < SR // 4:
                        continue
                    y = vc.match(vc.get_features(torch.from_numpy(x)[None]), ms, topk=a.topk).cpu().numpy()
                    y = np.clip(y / max(1e-6, np.abs(y).max()) * 0.5 * 32767, -32768, 32767).astype("<i2")
                    fp.write(y.tobytes())
                    r = {"wav": it.get("wav", ""), "text": it["text"], "duration": round(len(y) / SR, 2), "source": "vc_" + a.mode,
                         "speaker": f"{it.get('speaker', '')}>{spk}", "vc_target": spk, "group": it.get("group", ""),
                         "pack": os.path.abspath(pack), "off": pos, "n": len(y)}
                    rows.append(r); pos += len(y); out_s += len(y) / SR
            fm.writelines(json.dumps(r) + "\n" for r in rows)
            fp.flush(); fm.flush()
            del ms
            print(f"{k + 1}/{len(speakers)} {spk}: {len(rows)} clips, {out_s / 3600:.1f} h out, {out_s / max(1, time.time() - t0):.1f} x real time", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
