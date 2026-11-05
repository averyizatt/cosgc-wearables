import json, os, sys
# foldsum.py [eval dir]: base vs fold models (E19), float greedy, TORGO fold test by group and speaker, plus LibriSpeech and UASpeech.
# Each fold's models are scored on that fold's held-out speakers only. Paired CIs: train/paired_ci.py --mode full|stream on the *_utts.jsonl.
E = sys.argv[1] if len(sys.argv) > 1 else r"G:\cosgc\exp\eval"
def show(tag, f):
    p = os.path.join(E, f)
    if not os.path.exists(p):
        return
    j = json.load(open(p))
    g = "  ".join(f"{k} {v['full']:.2f}/{v['stream']:.2f}" for k, v in j["by_group"].items())
    s = "  ".join(f"{k} {v['full']:.1f}/{v['stream']:.1f}" for k, v in j.get("by_speaker", {}).items()) if "torgo" in f else ""
    print(f"{tag:22s} {g}\n{'':22s} {s}" if s else f"{tag:22s} {g}")
for fold in "ABC":
    t = "torgo_test" if fold == "A" else f"torgo_fold{fold}_test"
    print(f"== fold {fold} (full/stream)")
    show("base", f"base_{t}.json" if fold != "A" else "base24x_torgo_test.json")
    for r in ("big6", "big7"):
        m = r if fold == "A" else f"fold{fold}_{r}"
        show(m, f"{m}_{t}.json")
        show(m + " LS", f"{m}_ls_testclean_train.json")
        show(m + " UA", f"{m}_uaspeech_test_b2.json")
