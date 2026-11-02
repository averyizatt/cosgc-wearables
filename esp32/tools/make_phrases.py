#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Prepare a phrase list for phrase mode: one phrase per line in, "text<TAB>token ids" out, tokenized with the model's own SentencePiece model.

    python3 make_phrases.py my_phrases.txt phrases.tsv [--sp models/nemo_small/tokenizer.model]

Text is lowercased and stripped of punctuation as in training. The device (or tasr_cli --phrases) reads the output.
"""
import argparse, os, re
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--sp", default=os.path.join(HERE, "..", "..", "models", "nemo_small", "tokenizer.model"))
    a = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=a.sp)
    seen, rows = set(), []
    for line in open(a.inp, encoding="utf-8"):
        t = norm(line)
        if t and t not in seen:
            seen.add(t)
            rows.append(f"{t}\t{' '.join(map(str, sp.encode(t)))}\n")
    open(a.out, "w", encoding="utf-8", newline="\n").writelines(rows)
    print(len(rows), "phrases")


if __name__ == "__main__":
    main()
