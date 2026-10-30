#!/usr/bin/env python3
# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Run tasr_cli on the files made by stress_make.py with several models and print word error rate per condition.

    python3 stress_run.py ./tasr_cli stress_dir models/oido_stream.tnm+models/nemo_lm.tlm models/nemo8.tnm+models/nemo_lm.tlm ...

Each model argument is model[+lm]. Needs only Python 3 (no numpy); tasr_cli must be built for the host (esp32/host/Makefile).
"""
import json, os, re, subprocess, sys
from collections import defaultdict


def edits(a, b):
    a, b = a.split(), b.split()
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            c = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
            p = c
    return d[len(b)]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower())).strip()


def main(cli, d, specs):
    refs = json.load(open(os.path.join(d, "refs.json")))
    conds = sorted({f.split("__")[0] for f in refs}, key=lambda c: (c != "clean", c))
    table = defaultdict(dict)
    for spec in specs:
        model, _, lm = spec.partition("+")
        label = os.path.basename(model).replace(".tnm", "") + ("+LM" if lm else "")
        for c in conds:
            files = sorted(f for f in refs if f.startswith(c + "__"))
            cmd = [cli, model] + (["--lm", lm] if lm else []) + [os.path.join(d, f) for f in files]
            out = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
            err = n = 0
            for line in out:
                if "\t" in line:
                    f, hyp = line.split("\t", 1)
                    ref = norm(refs[os.path.basename(f)])
                    err += edits(ref, norm(hyp))
                    n += len(ref.split())
            table[c][label] = 100.0 * err / max(n, 1)
    labels = [os.path.basename(s.partition("+")[0]).replace(".tnm", "") + ("+LM" if "+" in s else "") for s in specs]
    print("%-14s" % "condition" + "".join("%14s" % l for l in labels))
    for c in conds:
        print("%-14s" % c + "".join("%13.1f%%" % table[c][l] for l in labels))
    print("%-14s" % "mean" + "".join("%13.1f%%" % (sum(table[c][l] for c in conds) / len(conds)) for l in labels))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
