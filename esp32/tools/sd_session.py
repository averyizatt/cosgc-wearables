#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Read the board's microSD session logs (firmware option CONFIG_TASR_SDLOG, esp32/firmware/main/sdlog.c).

    python sd_session.py score E:/s0003 [E:/s0004 ...] [-v]      WER of live and final text against the prompts, phrase picks, levels
    python sd_session.py manifest E:/s0003 out.jsonl            a manifest for esp32/host/eval_cli.py (rescore with another model)
    python sd_session.py models models/*.tnm ~/lmtest/*.tnm     each model's fingerprint, to match session.json's model_fnv1a
    python sd_session.py check prompts.txt                      count prompts, flag lines the screen cannot show

A session folder holds session.json, log.jsonl and one WAV per utterance (uNNNN.wav: what the model heard; uNNNNr.wav: the
microphone before gain control). With prompts.txt on the card, each utterance carries the prompt it answered; a tap on the
button goes back one prompt, so when a prompt was said more than once, the last attempt counts. A "text | cue" line in
prompts.txt is scored against text while the screen showed cue (for example a phonetic spelling for an imitation session).
Recordings of people stay on the card or the club's own storage, never in the repository.
"""
import argparse, json, os, re, statistics, sys


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", (t or "").lower().replace("-", " "))).strip()


def edits(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def load(sdir):
    sess = json.load(open(os.path.join(sdir, "session.json"), encoding="utf-8"))
    log = [json.loads(l) for l in open(os.path.join(sdir, "log.jsonl"), encoding="utf-8") if l.strip()]
    return sess, log


def last_attempts(log):
    by = {}
    for u in log:
        if "prompt_idx" in u:
            by[u["prompt_idx"]] = u  # later lines overwrite earlier attempts
    return [by[k] for k in sorted(by)]


def score(a):
    for sdir in a.dirs:
        sess, log = load(sdir)
        notes = "; ".join(sess.get("notes", []))
        print(f"== {sdir}: session {sess.get('session')}, model {sess.get('model_fnv1a')} ({sess.get('model_bytes')} bytes), "
              f"firmware {sess.get('firmware')}, {len(log)} utterances" + (f", notes: {notes}" if notes else ""))
        us = last_attempts(log)
        if not us:
            print("   no prompted utterances (no prompts.txt on the card): nothing to score against")
        else:
            for field, label in (("text", "shown (final)"), ("free", "final, free decoding"), ("stream", "live")):
                e = n = right = 0
                have = [u for u in us if field in u]
                for u in have:
                    de, dn = edits(norm(u["prompt"]), norm(u[field]))
                    e, n, right = e + de, n + dn, right + (de == 0)
                if have:
                    print(f"   {label:22s} WER {100 * e / max(1, n):6.2f} %   fully right {right}/{len(have)}")
            ph = [u for u in us if u.get("conf_level") == 3]
            if any("phrase" in u for u in us):
                ok = sum(norm(u["text"]) == norm(u["prompt"]) for u in ph)
                print(f"   phrase shown on {len(ph)} of {len(us)}: right {ok}, wrong {len(ph) - ok}")
            lv = [u.get("conf_level") for u in us]
            print("   confidence: " + ", ".join(f"{name} {lv.count(k)}" for k, name in ((2, "sure"), (1, "unsure"), (0, "probably wrong"), (-1, "none"))))
            if len(log) > len(us):
                print(f"   {len(log) - len(us)} earlier attempts were said again and are not scored")
        sp = [u["speech_db"] for u in log if "speech_db" in u]
        no = [u["noise_db"] for u in log if "noise_db" in u]
        dl = [u["delay_ms"] for u in log if u.get("full")]
        if sp:
            print(f"   levels: speech {statistics.median(sp):.1f} dBFS, noise {statistics.median(no):.1f} dBFS (medians); "
                  f"final text {statistics.median(dl) if dl else float('nan'):.0f} ms after release (median)")
        if a.v:
            for u in log:
                ref = u.get("prompt", "")
                de, dn = edits(norm(ref), norm(u["text"])) if ref else (0, 0)
                mark = "" if not ref else ("ok" if de == 0 else f"{de}/{dn}")
                print(f"   {u['utt']:4d} {u.get('wav', ''):10s} {mark:6s} said: {ref[:30]:30s} | shown: {u['text'][:30]:30s}"
                      f" | live: {u.get('stream', '')[:30]}")


def manifest(a):
    sess, log = load(a.dir)
    root = os.path.abspath(a.dir)
    n = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for u in log:
            if "wav" not in u:
                continue
            f.write(json.dumps({"wav": os.path.join(root, u["wav"]).replace("\\", "/"), "text": norm(u.get("prompt", "")),
                                "speaker": f"s{sess.get('session', 0):04d}", "utt": u["utt"]}) + "\n")
            n += 1
    print(f"{n} utterances -> {a.out}" + ("" if any("prompt" in u for u in log) else " (no prompts: text is empty)"))


def fnv1a(data):
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def models(a):
    for p in a.tnm:
        data = open(p, "rb").read()
        print(f"{fnv1a(data):08x}  {len(data):9d}  {p}")


def check(a):
    n = 0
    for i, line in enumerate(open(a.prompts, encoding="utf-8"), 1):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        n += 1
        text, _, cue = t.partition("|")
        shown = (cue or text).strip()
        if any(ord(c) < 32 or ord(c) > 126 for c in shown):
            print(f"line {i}: characters the screen cannot show: {shown!r}")
        if not norm(text):
            print(f"line {i}: no words to score against: {t!r}")
    print(f"{n} prompts")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("dirs", nargs="+"); s.add_argument("-v", action="store_true")
    m = sub.add_parser("manifest"); m.add_argument("dir"); m.add_argument("out")
    o = sub.add_parser("models"); o.add_argument("tnm", nargs="+")
    c = sub.add_parser("check"); c.add_argument("prompts")
    a = ap.parse_args()
    {"score": score, "manifest": manifest, "models": models, "check": check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
