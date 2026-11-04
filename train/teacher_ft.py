# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""A larger teacher for the on-device model: NVIDIA's Parakeet encoder (CC-BY-4.0) with OUR tokenizer and a CTC head.

The teacher emits the same 1024 pieces (+ blank, last) as the 19 M student, so its per-frame posteriors can teach the student directly
(train_nemo_stream.py --teacher). Its frames are 80 ms (8x subsampling) against the student's 40 ms; the student repeats each teacher frame.
Only the pretrained encoder and NeMo's feature front end are kept; the head is new. parakeet-tdt_ctc-110m fits a 12 GB card for training;
the 1.1 B model does not (about 18 GB of AdamW state alone).

Needs NeMo, so it runs in G:\\cosgc\\nemovenv like eval_parakeet.py.

    python teacher_ft.py train --train a.jsonl,b.jsonl --tokenizer base24x/tokenizer.model --out exp/teacher110 --aug_dir data/aug
    python teacher_ft.py eval --ckpt exp/teacher110/teacher.pt --tokenizer base24x/tokenizer.model --manifest torgo_test.jsonl --out eval/teacher110_torgo_test.json
    python teacher_ft.py offset --ckpt exp/teacher110/teacher.pt --nemo exp/base24x --student exp/big7/last.pt --manifest torgo_train.jsonl

eval writes eval_manifest.py's format (full context, greedy, no LM), so paired_ci.py compares it with the student's evals.
"""
import argparse, collections, json, math, os, random, time
import numpy as np, sentencepiece as spm, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from data import AudioDataset, BucketBatchSampler, collate, load_manifest

FPS = 12.5  # teacher frames per second


class Teacher(nn.Module):
    def __init__(self, name, vocab):
        super().__init__()
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.from_pretrained(name, map_location="cpu")
        self.name = name
        self.pre, self.enc = m.preprocessor, m.encoder
        self.spec = getattr(m, "spec_augmentation", None)
        self.head = nn.Linear(m.cfg.encoder.d_model, vocab + 1)  # blank last, as in the student

    def forward(self, wav, lens, specaug=False):
        """wav (B, N) float in [-1, 1] at 16 kHz -> logits (B, T, V+1) at 12.5 frames/s, lengths."""
        f, fl = self.pre(input_signal=wav, length=lens)
        if specaug and self.spec is not None:
            f = self.spec(input_spec=f, length=fl)
        e, el = self.enc(audio_signal=f, length=fl)
        return self.head(e.transpose(1, 2)), el


def load_teacher(path, dev):
    ck = torch.load(path, map_location="cpu")
    t = Teacher(ck["name"], ck["vocab"])
    t.load_state_dict(ck["state"])
    return t.to(dev).eval()


class AugDataset(AudioDataset):
    def __init__(self, items, aug):
        super().__init__(items)
        self.aug = aug

    def __getitem__(self, i):
        a, ids, j = super().__getitem__(i)
        return (self.aug(a) if self.aug is not None and len(a) > 1600 else a), ids, j


def greedy(lg, n, blank):
    out, prev = [], -1
    for i in lg[:n].argmax(-1).tolist():
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # cpu only for smoke tests
    torch.manual_seed(a.seed); random.seed(a.seed)
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    V = sp.get_piece_size()
    items = load_manifest(a.train.split(","), max_dur=a.max_dur)
    n0 = len(items)
    for it in items:
        it["ids"] = sp.encode(it["text"])
    items = [it for it in items if 0 < len(it["ids"]) < it["dur"] * FPS * 0.8]
    print(f"train utts {len(items)} of {n0} ({sum(it['dur'] for it in items) / 3600:.0f} h)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model = Teacher(a.model, V).to(dev)
    print(a.model, round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M parameters", flush=True)
    head = list(model.head.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr}, {"params": head, "lr": a.lr * a.head_lr_mult}],
                            betas=(0.9, 0.98), weight_decay=1e-3)
    ckpt, step = os.path.join(a.out, "ckpt.pt"), 0
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print("resumed at step", step, flush=True)
    aug = None
    if a.aug_dir:
        from augment import NpAug
        aug = NpAug(a.aug_dir, a.p_noise, a.p_rir, p_speed=a.p_speed)

    def save(final=False):
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)
        if final:
            torch.save({"name": a.model, "vocab": V, "state": model.state_dict(), "step": step}, os.path.join(a.out, "teacher.pt"))

    sampler = BucketBatchSampler(items, a.batch_sec, seed=1 + a.seed)
    ep, t0, ag = step // max(1, len(sampler)), time.time(), 0.0
    model.train()
    while step < a.steps:
        sampler.set_epoch(ep)
        dl = DataLoader(AugDataset(items, aug), batch_sampler=sampler, collate_fn=collate, num_workers=a.workers,
                        pin_memory=dev == "cuda", prefetch_factor=4 if a.workers else None, persistent_workers=False)
        for wav, lens, tokb, tl, _ in dl:
            wav = wav.to(dev).float() / 32768.0
            lens, tokb, tl = lens.to(dev), tokb.to(dev), tl.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lg, ol = model(wav, lens, specaug=True)
            lp = lg.float().log_softmax(-1)
            loss = F.ctc_loss(lp.transpose(0, 1), tokb, ol, tl, blank=V, reduction="sum", zero_infinity=True) / tl.sum()
            lr_f = min(1.0, (step + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
            opt.param_groups[0]["lr"] = a.lr * lr_f
            opt.param_groups[1]["lr"] = a.lr * a.head_lr_mult * lr_f
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            ag += loss.item()
            if step % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
                print(f"step {step} loss {ag / 100:.4f} lr {a.lr * lr_f:.2e} {time.time() - t0:.0f}s mem {mem:.1f} GB", flush=True)
                ag = 0.0
            elif a.steps < 100:  # smoke tests
                print(f"step {step} loss {loss.item():.4f}", flush=True)
            if step % a.save_every == 0:
                save()
            if step >= a.steps:
                break
        ep += 1
    save(final=True)
    print("finished", flush=True)


def evaluate(a):
    import jiwer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_teacher(a.ckpt, dev)
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    V = sp.get_piece_size()
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]  # every clip and its raw text, as in eval_manifest.py
    if a.limit:
        items = items[:: a.limit]
    order = sorted(range(len(items)), key=lambda i: items[i].get("duration", 0))
    hyp = {}
    ds = AudioDataset(items)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for b in range(0, len(order), a.batch):
            idx = order[b: b + a.batch]
            wav, lens, _, _, _ = collate([ds[i] for i in idx])
            lg, ol = model(wav.to(dev).float() / 32768.0, lens.to(dev))
            for k, i in enumerate(idx):
                hyp[i] = sp.decode(greedy(lg[k].float(), int(ol[k]), V))
            if b % (50 * a.batch) == 0:
                print(f"{b}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    rows = [dict(it, hyp_full=hyp[i]) for i, it in enumerate(items)]

    def wer(rs):
        return round(100 * jiwer.wer([r["text"] for r in rs], [r["hyp_full"] or "<empty>" for r in rs]), 2)

    summary = {"model": a.ckpt, "manifest": a.manifest, "clips": len(rows), "decoding": "greedy, no LM, full context"}
    for key in ("group", "speaker", "mic", "kind"):
        by = collections.defaultdict(list)
        for r in rows:
            by[r.get(key, "?")].append(r)
        summary["by_" + key] = {k: {"full": wer(v), "clips": len(v)} for k, v in sorted(by.items())}
    summary["all"] = {"full": wer(rows)}
    print(json.dumps({k: v for k, v in summary.items() if k != "by_speaker"}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "_utts.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def offset(a):
    """By how many student frames (40 ms) do the teacher's token spikes lag the student's? Correlates the two models' per-frame
    probability of emitting a token (1 - blank) on full-context passes, at shifts of -8..8 frames; the best shift goes to
    train_nemo_stream.py --teacher_shift."""
    from nemo_small import NemoSmall
    from train_nemo_stream import BatchPre, SModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = load_teacher(a.ckpt, dev)
    sd = torch.load(os.path.join(a.nemo, "state_dict_plain.pt"), map_location="cpu")
    m = NemoSmall(sd)
    src = torch.load(a.student, map_location="cpu")
    m.load_state_dict(src["model"])
    pre = BatchPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"], src["norm_mean"], src["norm_std"]).to(dev)
    net = SModel(m, 0).to(dev).eval()
    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8")][:: a.limit or 1]
    ds = AudioDataset(items)
    shifts = list(range(-8, 9))
    score = np.zeros(len(shifts))
    with torch.inference_mode():
        for i in range(len(items)):
            w = torch.from_numpy(ds[i][0].astype(np.float32) / 32768.0)[None].to(dev)
            n = torch.tensor([w.shape[1]], device=dev)
            f, fl = pre(w, n)
            lg, ol = net(f, fl, 0, 0)
            s = 1 - lg[0, : int(ol[0])].float().softmax(-1)[:, -1]
            tg, tl = teacher(w, n)
            t = (1 - tg[0, : int(tl[0])].float().softmax(-1)[:, -1]).repeat_interleave(2)
            L = min(len(s), len(t))
            s, t = s[:L] - s[:L].mean(), t[:L] - t[:L].mean()
            for k, d in enumerate(shifts):  # teacher frame j + d against student frame j
                if d >= 0:
                    score[k] += float((s[: L - d] * t[d:]).sum())
                else:
                    score[k] += float((s[-d:] * t[: L + d]).sum())
    for d, v in zip(shifts, score / np.abs(score).max()):
        print(f"shift {d:+d}: {v:.3f}")
    best = shifts[int(score.argmax())]
    if abs(best) <= 4:
        print("best --teacher_shift", best)
    else:  # an untrained or mismatched teacher gives a flat, noisy curve; do not pass that on
        print("offset unclear: best shift", best, "is beyond 4 frames")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True, help="comma-separated jsonl manifests (packed or not)")
    t.add_argument("--tokenizer", required=True, help="the student's SentencePiece model")
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="nvidia/parakeet-tdt_ctc-110m")
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch_sec", type=float, default=100)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--head_lr_mult", type=float, default=10.0, help="the new head learns this much faster")
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--max_dur", type=float, default=20.0)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--save_every", type=int, default=500)
    t.add_argument("--aug_dir", default="")
    t.add_argument("--p_noise", type=float, default=0.6)
    t.add_argument("--p_rir", type=float, default=0.3)
    t.add_argument("--p_speed", type=float, default=0.3)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--limit", type=int, default=0, help="every n-th clip only")
    o = sub.add_parser("offset")
    o.add_argument("--ckpt", required=True, help="teacher.pt")
    o.add_argument("--nemo", required=True, help="the student's base dir (state_dict_plain.pt)")
    o.add_argument("--student", required=True, help="a student checkpoint (last.pt)")
    o.add_argument("--manifest", required=True)
    o.add_argument("--limit", type=int, default=50, help="every n-th clip only")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "offset": offset}[a.cmd](a)


if __name__ == "__main__":
    main()
