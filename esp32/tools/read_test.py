#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Measure the live device: one person reads known prompts at the board, and the transcripts are scored against them.

    python esp32/tools/read_test.py COM3 --speaker sam --distance 15cm [--out DIR]

Shows one prompt at a time: hold the button, read it, release, wait for the text. Keys: s skip, q quit. Writes, in a new folder under
--out (default ~/lmtest/board_clips; keep recordings of people out of the repository): one WAV per prompt, the microphone audio
before gain control, and clips.jsonl with the prompt, final text, last partial text, delay and level. WER is printed at the end. Needs pyserial.
The sentences are from the IEEE Harvard lists (IEEE Recommended Practice for Speech Quality Measurements, 1969).
"""
import argparse, base64, json, math, os, re, struct, sys, time, wave
import serial

SENTENCES = [
    "the birch canoe slid on the smooth planks", "glue the sheet to the dark blue background",
    "it's easy to tell the depth of a well", "these days a chicken leg is a rare dish",
    "rice is often served in round bowls", "the juice of lemons makes fine punch",
    "the box was thrown beside the parked truck", "four hours of steady work faced us",
    "a large size in stockings is hard to sell", "the boy was there when the sun rose",
    "a rod is used to catch pink salmon", "the source of the huge river is the clear spring",
    "kick the ball straight and follow through", "help the woman get back to her feet",
    "a pot of tea helps to pass the evening", "smoky fires lack flame and heat",
    "the soft cushion broke the man's fall", "the salt breeze came across the sea",
    "the girl at the booth sold fifty bonds", "the small pup gnawed a hole in the sock",
]
PHRASES = ["i need some water", "please call my mother", "can you help me stand up", "turn the lights off", "i am feeling tired today"]
WORDS = ["yes", "no", "help", "stop", "water", "hello", "please", "thanks", "doctor", "bathroom"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("’", "'"))).strip()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[len(h)]


def wer(rows, key):
    e = sum(edits(norm(r["prompt"]).split(), norm(r[key]).split()) for r in rows)
    n = sum(len(norm(r["prompt"]).split()) for r in rows)
    return 100 * e / max(1, n)


def key_pressed():
    try:
        import msvcrt
        return msvcrt.getwch().lower() if msvcrt.kbhit() else ""
    except ImportError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--distance", default="", help="for the record, for example 15cm")
    ap.add_argument("--out", default=os.path.expanduser("~/lmtest/board_clips"))
    a = ap.parse_args()
    out = os.path.join(a.out, f"{a.speaker}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out)
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = a.port, 115200, 0.05
    s.dtr = s.rts = False
    s.open()

    def send(**cmd):
        s.write((json.dumps(cmd) + "\n").encode())

    def events():  # JSON lines from the board
        buf = b""
        while True:
            buf += s.read(8192)
            *lines, buf = buf.split(b"\n")
            for l in lines:
                l = l.strip()
                if l.startswith(b"{"):
                    try:
                        yield json.loads(l)
                    except ValueError:
                        pass
            yield None

    def acked(cmd, wait=1.0, **kw):  # send until the board answers: the first bytes after opening the port can be lost
        for _ in range(3):
            send(cmd=cmd, **kw)
            t, buf = time.time(), b""
            while time.time() - t < wait:
                buf += s.read(4096)
                if f'"type":"ack","cmd":"{cmd}","ok":true'.encode() in buf:
                    return True
        return False

    time.sleep(0.5)
    s.reset_input_buffer()
    if not acked("ping"):  # a run that was interrupted can leave the board at the high speed
        s.baudrate = 921600
        if not acked("ping"):
            sys.exit("the board does not answer on " + a.port + ": is it running, and is anything else using the port?")
    if s.baudrate != 921600:
        if not acked("set", baud=921600):
            sys.exit("the board did not accept the serial speed")
        time.sleep(0.2)
        s.baudrate = 921600
        s.reset_input_buffer()
        if not acked("ping"):
            sys.exit("no answer from the board at 921600 baud")
    if not acked("set", dump=True) or not acked("set", level=True):
        sys.exit("the board did not accept the settings; is the firmware up to date?")
    ev = events()
    rows = []
    prompts = [("sentence", p) for p in SENTENCES] + [("phrase", p) for p in PHRASES] + [("word", p) for p in WORDS]
    print(f"{len(prompts)} prompts. Hold the button, read the line, release, wait for the text. Keys: s skip, q quit. Saving to {out}\n")
    try:
        for k, (kind, prompt) in enumerate(prompts):
            print(f"[{k + 1}/{len(prompts)}]  {prompt.upper()}", flush=True)
            final, partial, pcm, total, peak, skip = None, "", b"", None, -120.0, False
            for e in ev:
                if e is None:
                    c = key_pressed()
                    if c == "q":
                        raise KeyboardInterrupt
                    if c == "s" and final is None:
                        skip = True
                        break
                    continue
                t = e.get("type")
                if t == "level":
                    peak = max(peak, e["db"])
                elif t == "partial":
                    partial = e["text"]
                elif t == "final":
                    final = e
                    print(f"    heard: {e['text']}   ({e['delay_ms'] / 1000:.1f} s after release, peak {peak:.0f} dBFS)", flush=True)
                elif t == "audio":
                    pcm += base64.b64decode(e["b64"])
                    total = e["total"]
                    if len(pcm) >= total * 2:
                        break
                elif t == "warn":
                    print("    warning from the board:", e)
            if skip:
                continue
            name = f"{k:02d}_{kind}.wav"
            with wave.open(os.path.join(out, name), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
            x = struct.unpack(f"<{len(pcm) // 2}h", pcm)
            rms = math.sqrt(sum(v * v for v in x) / max(1, len(x))) / 32768
            rows.append({"wav": name, "prompt": prompt, "kind": kind, "speaker": a.speaker, "distance": a.distance, "final": final["text"], "partial": partial,
                         "delay_ms": final["delay_ms"], "audio_s": final["audio_s"], "peak_dbfs": peak, "rms_dbfs": 20 * math.log10(max(rms, 1e-6))})
            with open(os.path.join(out, "clips.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1]) + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        send(cmd="set", dump=False)
        send(cmd="set", level=False)
        send(cmd="set", baud=115200)
        time.sleep(0.3)
        s.close()
    if rows:
        print(f"\n{len(rows)} clips, final text WER {wer(rows, 'final'):.1f} %, live text WER {wer(rows, 'partial'):.1f} %")
        for kind in ("sentence", "phrase", "word"):
            sub = [r for r in rows if r["kind"] == kind]
            if sub:
                print(f"  {kind}: {len(sub)} clips, final {wer(sub, 'final'):.1f} %, live {wer(sub, 'partial'):.1f} %")
        print(f"Saved in {out}")


if __name__ == "__main__":
    main()
