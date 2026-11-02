#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Load a phrase list onto the device (phrase mode, docs/EXPERIMENTS.md E18). The device keeps it across reboots.

    python esp32/tools/send_phrases.py COM3 my_phrases.txt      # one phrase per line; replaces the list on the device
    python esp32/tools/send_phrases.py COM3 --clear              # phrase mode off
    python esp32/tools/send_phrases.py COM3                      # how many phrases the device holds

Phrases are tokenized here with the model's SentencePiece model (make_phrases.py) and sent in small batches: the device reads
lines of at most 512 bytes. Short lists work best: about 25 phrases gave 84 % fully right utterances for dysarthric speakers on
TORGO test, against 50 % without a list.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oido_client import Oido
from make_phrases import norm, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("phrases", nargs="?")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--sp", default=os.path.join(HERE, "..", "..", "models", "nemo_small", "tokenizer.model"))
    a = ap.parse_args()
    with Oido(a.port) as dev:
        for _ in range(5):  # the first bytes after opening the port can be lost: retry until the device answers
            try:
                dev.request({"cmd": "ping"}, timeout=1.5)
                break
            except TimeoutError:
                pass
        else:
            sys.exit("no answer from the device")
        if a.phrases:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=a.sp)
            seen, rows = set(), []
            for line in open(a.phrases, encoding="utf-8"):
                t = norm(line)
                if t and t not in seen:
                    seen.add(t)
                    rows.append(f"{t}\t{' '.join(map(str, sp.encode(t)))}")
            ack, _ = dev.request({"cmd": "phrases", "clear": True})
            batch = []
            def flush():
                if batch:
                    ack, _ = dev.request({"cmd": "phrases", "add": batch})
                    if not ack.get("ok"):
                        sys.exit(f"device refused: {ack.get('error')}")
                    batch.clear()
            for r in rows:
                if len(json.dumps({"cmd": "phrases", "add": batch + [r], "id": 999999})) > 480:
                    flush()
                batch.append(r)
            flush()
        elif a.clear:
            dev.request({"cmd": "phrases", "clear": True})
        ack, before = dev.request({"cmd": "phrases"})
        info = next((e for e in before if e.get("type") == "phrases"), {})
        print(f"device holds {info.get('count', '?')} phrases ({info.get('bytes', '?')} of {info.get('max_bytes', '?')} bytes)")


if __name__ == "__main__":
    main()
