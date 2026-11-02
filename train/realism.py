# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""How realistic is the simulated speech? Two measures that do not reward a recording-channel difference.

    python realism.py --real torgo_train.jsonl --sim sim_torgoctl.jsonl[,simval4.jsonl] --out realism.json

Clips are described by a self-supervised speech model (microsoft/wavlm-base-plus, a middle layer, mean and spread over time).

1. Probe: a classifier learns control vs dysarthric from TORGO's TRAINING speakers (same microphones and prompts on both sides; scored
   leaving one speaker out). It then rates each simulated set: the share judged dysarthric, by condition and severity, beside the real speakers'
   and the clean sources'. A realistic simulation moves clean speech to where real dysarthric speakers sit.
2. Discriminator: real dysarthric training speakers against simulated dysarthria whose source is TORGO's own CONTROL training speakers (make it
   with dysim.py gen --manifest torgo_control_train.jsonl), so microphone, room and prompts are the same on both sides. Area under the curve,
   leaving speakers out: 0.5 means the two cannot be told apart; 1.0 means the simulation is obvious. Reported by condition.

Never uses TORGO's test speakers (F03, M05). Needs a GPU environment with transformers and scikit-learn (G:\\cosgc\\nemovenv).
"""
import argparse, collections, json
import numpy as np

TEST_SPEAKERS = {"F03", "M05"}


def audio(it):
    if "pack" in it:
        return np.array(np.memmap(it["pack"], dtype="<i2", mode="r")[it["off"]: it["off"] + it["n"]], dtype=np.float32) / 32768.0
    import soundfile as sf
    w, _ = sf.read(it["wav"], dtype="float32")
    return w.mean(1) if w.ndim > 1 else w


class Embed:
    def __init__(self, name, layer):
        import torch
        from transformers import WavLMModel
        self.torch, self.layer = torch, layer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = WavLMModel.from_pretrained(name).to(self.dev).eval()

    def __call__(self, x):
        t = self.torch
        x = (x - x.mean()) / (x.std() + 1e-7)
        with t.inference_mode():
            h = self.m(t.from_numpy(x[: 16000 * 20])[None].to(self.dev), output_hidden_states=True).hidden_states[self.layer][0]
        return t.cat([h.mean(0), h.std(0)]).float().cpu().numpy()


def grouped_auc(X, y, groups, C=0.1):
    """Leave-one-speaker-out (or 5 speaker folds when there are many) probability of class 1 for every clip, and the area under the curve."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    ug = sorted(set(groups))
    folds = [ug[k::5] for k in range(5)] if len(ug) > 10 else [[g] for g in ug]
    p = np.zeros(len(y))
    for f in folds:
        te = np.isin(groups, f)
        if te.all() or len(set(y[~te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[~te], y[~te])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p, (roc_auc_score(y, p) if len(set(y)) == 2 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="TORGO training manifest (control and dysarthric)")
    ap.add_argument("--sim", required=True, help="comma-separated simulated manifests (dysim.py or knnvc_gen.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--per_speaker", type=int, default=300)
    ap.add_argument("--per_set", type=int, default=3000)
    ap.add_argument("--real_group", default="dysarthric", help="the discriminator's real side: dysarthric (is the simulation like "
                    "real dysarthria?) or control (with a severity-0 set: how detectable is the vocoder alone?)")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(0)
    emb = Embed(a.model, a.layer)

    real = [json.loads(l) for l in open(a.real, encoding="utf-8") if l.strip()]
    real = [r for r in real if r.get("speaker") not in TEST_SPEAKERS and r.get("group") in ("control", "dysarthric")]
    by = collections.defaultdict(list)
    for r in real:
        by[r["speaker"]].append(r)
    real = [r for v in by.values() for r in (v if len(v) <= a.per_speaker else [v[k] for k in rng.choice(len(v), a.per_speaker, replace=False)])]
    Xr = np.stack([emb(audio(r)) for r in real])
    yr = np.array([r["group"] == "dysarthric" for r in real], int)
    gr = np.array([r["speaker"] for r in real])
    p, auc = grouped_auc(Xr, yr, gr)
    res = {"probe_auc_real": round(auc, 3), "real_speakers": {}, "sets": {}}
    for s in sorted(set(gr)):
        res["real_speakers"][s] = {"group": real[list(gr).index(s)]["group"], "p_dys": round(float(p[gr == s].mean()), 3)}
    print(f"probe, control vs dysarthric on held-out training speakers: AUC {auc:.3f}")
    for s, v in res["real_speakers"].items():
        print(f"  {s:6s} {v['group']:10s} judged dysarthric {100 * v['p_dys']:5.1f} %")
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Xr, yr)

    for path in a.sim.split(","):
        sim = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if len(sim) > a.per_set:
            sim = [sim[k] for k in sorted(rng.choice(len(sim), a.per_set, replace=False))]
        Xs = np.stack([emb(audio(r)) for r in sim])
        ps = probe.predict_proba(Xs)[:, 1]
        # the clean sources, for the baseline (their "wav" is the original clip)
        src = [dict(wav=r["wav"]) for r in sim[:500] if r.get("wav")]
        p_src = float(probe.predict_proba(np.stack([emb(audio(r)) for r in src]))[:, 1].mean()) if src else float("nan")
        st = {"clips": len(sim), "p_dys_sim": round(float(ps.mean()), 3), "p_dys_source": round(p_src, 3), "by_condition": {}, "by_severity": {}}
        conds = np.array([r.get("condition", r.get("source", "?")) for r in sim])
        sev = np.array([r.get("severity", -1.0) for r in sim])
        for c in sorted(set(conds)):
            st["by_condition"][c] = {"p_dys": round(float(ps[conds == c].mean()), 3)}
        for lo in np.arange(0, 1.6, 0.2):
            m = (sev >= lo) & (sev < lo + 0.2)
            if m.sum() >= 20:
                st["by_severity"][f"{lo:.1f}"] = round(float(ps[m].mean()), 3)
        # discriminator: only meaningful when the simulation's sources are TORGO control speakers
        from_torgo = np.array(["TORGO" in r.get("wav", "").upper() for r in sim])
        if from_torgo.mean() > 0.9:
            dys = yr == (1 if a.real_group == "dysarthric" else 0)
            X = np.concatenate([Xr[dys], Xs]); y = np.r_[np.zeros(dys.sum(), int), np.ones(len(Xs), int)]
            g = np.r_[gr[dys], np.array(["sim:" + r.get("speaker", "?") for r in sim])]
            pd, dauc = grouped_auc(X, y, g)
            st["discriminator_auc"] = round(dauc, 3)
            from sklearn.metrics import roc_auc_score
            psim = pd[len(Xr[dys]):]
            for c in sorted(set(conds)):
                m = conds == c
                if m.sum() >= 20:
                    st["by_condition"][c]["discriminator_auc"] = round(float(roc_auc_score(np.r_[np.zeros(dys.sum()), np.ones(m.sum())], np.r_[pd[: dys.sum()], psim[m]])), 3)
        res["sets"][path] = st
        print(f"{path}: judged dysarthric {100 * st['p_dys_sim']:.1f} % (its clean sources {100 * p_src:.1f} %)"
              + (f", real vs simulated AUC {st['discriminator_auc']:.3f}" if "discriminator_auc" in st else ""))
        for c, v in sorted(st["by_condition"].items(), key=lambda kv: -kv[1]["p_dys"]):
            print(f"  {c:14s} judged dysarthric {100 * v['p_dys']:5.1f} %" + (f"  discriminator AUC {v['discriminator_auc']:.3f}" if "discriminator_auc" in v else ""))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
