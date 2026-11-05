# Copyright (C) 2026 the cosgc-wearables contributors. Licensed under the GNU GPL v3 (see LICENSE).
"""Pool one model's device-engine outputs over several seeds, for paired_ci.py.

    python pool_seeds.py out_utts.jsonl seed0_utts.jsonl seed1_utts.jsonl ...

Each clip's edits and reference words are summed over the seeds, so paired_ci.py on two pooled files compares the
seed-mean models and resamples clips (not seeds).
"""
import json, sys

out, ins = sys.argv[1], sys.argv[2:]
acc = {}
for p in ins:
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        a = acc.setdefault(r["wav"], {"wav": r["wav"], "edits": 0, "ref": ""})
        a["edits"] += r["edits"]
        a["ref"] += " " + r["ref"]
with open(out, "w", encoding="utf-8") as f:
    for r in acc.values():
        f.write(json.dumps(r) + "\n")
print(f"{len(acc)} clips pooled over {len(ins)} files")
