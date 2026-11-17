#!/usr/bin/env python3
"""Split result lines that contain more than one JSON object.

An interrupted append can leave two records on one line, which makes the
one-object-per-line readers (e.g. cgc_ad.analyze.load) fail with
"Extra data". This script rewrites the file with one object per line.

    python scripts/repair_jsonl.py tests/results/e2_e4_results.jsonl          # in place
    python scripts/repair_jsonl.py in.jsonl --out fixed.jsonl                 # keep the original

A backup copy `<file>.bak` is written when repairing in place.
"""
import argparse
import json
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dec, rows, fixed = json.JSONDecoder(), [], 0
    for line in open(a.path, encoding="utf-8"):
        line = line.strip()
        n = 0
        while line:
            obj, i = dec.raw_decode(line)
            rows.append(obj)
            line = line[i:].lstrip()
            n += 1
        fixed += n > 1
    out = a.out or a.path
    if out == a.path:
        shutil.copyfile(a.path, a.path + ".bak")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"{len(rows)} records written to {out}; {fixed} merged line(s) split")


if __name__ == "__main__":
    main()
