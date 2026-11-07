#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Read the board's microSD session logs (firmware option CONFIG_TASR_SDLOG, esp32/firmware/main/sdlog.c).

    python sd_session.py score E:/s0003 [E:/s0004 ...] [-v]      WER of live and final text against the prompts, phrase picks, levels
    python sd_session.py manifest E:/s0003 out.jsonl            a manifest for esp32/host/eval_cli.py (rescore with another model)
    python sd_session.py manifest E:/s0003 b.jsonl --which raw_b   the other microphone's recordings (two-microphone builds): score
                                                                both microphones on the same speech (--which raw for the heard one)
    python sd_session.py models models/*.tnm ~/lmtest/*.tnm     each model's fingerprint, to match session.json's model_fnv1a
    python sd_session.py check prompts.txt                      count prompts, flag lines the screen cannot show
    python sd_session.py table sheet.csv [--md out.md] [--json out.json] [--phrases list.txt]
                                                                every session on the sheet: one row per speaker and condition, a
                                                                pooled row per condition, phrase-mode outcomes (EXPERIMENTS.md tables)

sheet.csv is the paper sheet typed up: one line per session folder, "folder,speaker" (a header line and '#' lines are skipped;
folders relative to the sheet's own folder). The condition is the session's "condition:" note (prompts1-3.txt carry one), else its
prompt file. A condition that mentions "imitat" is labelled imitated in every row: a stress test, never dysarthric accuracy.

A session folder holds session.json, log.jsonl and one WAV per utterance (uNNNN.wav: what the model heard; uNNNNr.wav: the
microphone before gain control). With prompts.txt on the card, each utterance carries the prompt it answered; a tap on the
button goes back one prompt, so when a prompt was said more than once, the last attempt counts. A "text | cue" line in
prompts.txt is scored against text while the screen showed cue (for example a phonetic spelling for an imitation session).
Recordings of people stay on the card or the club's own storage, never in the repository.
"""
import argparse, collections, json, os, re, statistics, sys


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
            if a.which not in u:
                continue
            f.write(json.dumps({"wav": os.path.join(root, u[a.which]).replace("\\", "/"), "text": norm(u.get("prompt", "")),
                                "speaker": f"s{sess.get('session', 0):04d}", "utt": u["utt"],
                                "mic": u.get("mic", 1) if a.which != "raw_b" else 3 - u.get("mic", 1)}) + "\n")
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


def read_sheet(path):
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8-sig"), 1):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        parts = [p.strip() for p in t.split(",")]
        if len(parts) < 2 or not parts[1]:
            sys.exit(f"{path} line {i}: expected folder,speaker, got {t!r}")
        if parts[0].lower() == "folder":
            continue
        folder = parts[0] if os.path.isabs(parts[0]) else os.path.join(base, parts[0])
        rows.append((folder, parts[1]))
    return rows


def condition(sess):
    for n in sess.get("notes", []):
        if n.lower().startswith("condition:"):
            return n.split(":", 1)[1].strip()
    return sess.get("prompt_file") or "unknown"


def tally(us, phrases):
    """Counts over the scored (last) attempts of one group of utterances."""
    t = collections.Counter()
    for u in us:
        t["n"] += 1
        for field in ("text", "free", "stream"):
            if field in u:
                de, dn = edits(norm(u["prompt"]), norm(u[field]))
                t[field + "_e"] += de
                t[field + "_w"] += dn
                t[field + "_n"] += 1
                t[field + "_ok"] += de == 0
        shown = u.get("conf_level") == 3  # the phrase replaced the free text
        if norm(u["prompt"]) in phrases:
            t["list"] += 1
            t["list_right" if shown and norm(u["text"]) == norm(u["prompt"]) else "list_wrong" if shown else "list_free"] += 1
        else:
            t["off"] += 1
            t["off_replaced"] += shown
    return t


def table(a):
    phrases = {norm(l) for l in open(a.phrases, encoding="utf-8") if l.strip() and not l.startswith("#")}
    groups = collections.OrderedDict()  # (condition, speaker) -> scored utterances
    levels = collections.defaultdict(list)
    sessions = []
    for folder, spk in read_sheet(a.sheet):
        if not os.path.exists(os.path.join(folder, "log.jsonl")):
            sys.exit(f"{folder}: no log.jsonl (check the sheet)")
        sess, log = load(folder)
        cond = condition(sess)
        us = [u for u in last_attempts(log) if norm(u.get("prompt"))]
        groups.setdefault((cond, spk), []).extend(us)
        for u in log:
            if "speech_db" in u:
                levels[(cond, spk)].append((u["speech_db"], u["noise_db"], u.get("delay_ms") if u.get("full") else None))
        sessions.append({"folder": folder, "speaker": spk, "condition": cond, "session": sess.get("session"),
                         "model_fnv1a": sess.get("model_fnv1a"), "firmware": sess.get("firmware"),
                         "lm_weight_x100": sess.get("lm_weight_x100"), "token_bonus_x100": sess.get("token_bonus_x100"),
                         "utterances": len(log), "scored": len(us)})
    models = {s["model_fnv1a"] for s in sessions}
    pct = lambda e, n: f"{100 * e / n:.1f}" if n else "–"
    rows, out = [], []
    conds = list(dict.fromkeys(c for c, _ in groups))
    for cond in conds:
        keys = [k for k in groups if k[0] == cond]
        for k in keys + [(cond, "all")]:
            us = groups[k] if k[1] != "all" else [u for kk in keys for u in groups[kk]]
            lv = levels[k] if k[1] != "all" else [x for kk in keys for x in levels[kk]]
            t = tally(us, phrases)
            dl = [d for _, _, d in lv if d is not None]
            r = {"condition": cond, "imitated": "imitat" in cond.lower(), "speaker": k[1], **dict(t),
                 "speech_db": statistics.median([s for s, _, _ in lv]) if lv else None,
                 "noise_db": statistics.median([n for _, n, _ in lv]) if lv else None,
                 "delay_ms": statistics.median(dl) if dl else None}
            rows.append(r)
            label = ("imitated: " if r["imitated"] else "") + cond
            out.append(f"| {label} | {k[1]} | {t['n']} | {pct(t['text_e'], t['text_w'])} | {t['text_ok']}/{t['text_n']} | "
                       f"{pct(t['free_e'], t['free_w'])} | {pct(t['stream_e'], t['stream_w'])} | "
                       f"{t['list_right']} / {t['list_wrong']} / {t['list_free']} of {t['list']} | {t['off_replaced']} of {t['off']} | "
                       + (f"{r['speech_db']:.1f} / {r['noise_db']:.1f}" if lv else "–") + " | "
                       + (f"{r['delay_ms'] / 1000:.2f}" if dl else "–") + " |")
    head = ["| Condition | Speaker | Prompts | Shown WER | Shown fully right | Free WER | Live WER | List phrases: right / wrong / free text "
            "| Off-list replaced | Speech / noise dBFS | Final text after release, s |", "|" + "---|" * 11]
    notes = [f"{len(sessions)} sessions, {len({s['speaker'] for s in sessions})} speakers, model fingerprint(s) {', '.join(sorted(map(str, models)))}.",
             "Last attempt of each prompt; WER is corpus-level over words. 'all' rows pool that condition's speakers."]
    if len(models) > 1:
        notes.append("WARNING: sessions ran different models; do not pool them.")
    if any(r["imitated"] for r in rows):
        notes.append("Imitated rows are a stress test of the device and of phrase mode, never accuracy for dysarthric speakers.")
    text = "\n".join(head + out) + "\n\n" + "\n".join(notes) + "\n"
    print(text)
    if a.md:
        open(a.md, "w", encoding="utf-8", newline="\n").write(text)
    if a.json:
        json.dump({"sessions": sessions, "rows": rows}, open(a.json, "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("dirs", nargs="+"); s.add_argument("-v", action="store_true")
    m = sub.add_parser("manifest"); m.add_argument("dir"); m.add_argument("out")
    m.add_argument("--which", default="wav", choices=["wav", "raw", "raw_b"],
                   help="wav: what the model heard; raw: that microphone before gain control; raw_b: the other microphone (two-microphone builds)")
    o = sub.add_parser("models"); o.add_argument("tnm", nargs="+")
    c = sub.add_parser("check"); c.add_argument("prompts")
    t = sub.add_parser("table"); t.add_argument("sheet"); t.add_argument("--md"); t.add_argument("--json")
    t.add_argument("--phrases", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "phrases_example.txt"),
                   help="the phrase list loaded on the board: decides which prompts are list phrases")
    a = ap.parse_args()
    {"score": score, "manifest": manifest, "models": models, "check": check, "table": table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
