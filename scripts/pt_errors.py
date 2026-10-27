"""Break down the agent's wrong answers on the Perception Test cup games.

  python scripts/pt_errors.py "runs/qwen8b/pt*.jsonl" [--data data]

bind/track  the gold option agrees with the bound track under neither viewpoint
read-out    otherwise, the answer is not the option that the stated facts imply
viewpoint   otherwise (the bound track is right, the viewpoint fact is wrong)
unbound     the plan bound no target
other       options that are not positions, unparsable facts, or failed runs"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
POS = r"(first|second|third|fourth|fifth) object from the left"


def classify(r, item):
    if r.get("status") != "OK":
        return "other"
    if r.get("readout") != "resolve" or not r.get("facts"):
        return "unbound"
    m = re.search(POS, item["options"][int(item["answer"])])
    txt = " ".join(map(str, r["facts"]))
    n = re.search(r"all (\d+) are tracked", txt)
    fin = re.findall(r"(first|second|third|fourth|fifth) \w+(?: \w+)? from the (?:person's|camera's)? ?left", txt)
    pov = re.search(r"[Ff]rom the person's point of view[^.]*?(first|second|third|fourth|fifth)", txt)
    k = ORD[pov.group(1)] if pov else (ORD[fin[-1]] if fin else None)
    if not m or k is None or not n:
        return "other"
    g, n = ORD[m.group(1)], int(n.group(1))
    if g not in (k, n + 1 - k):
        return "bind/track"
    pred = r.get("pred")
    pk = None
    if pred and pred in "ABCDEF" and "ABCDEF".index(pred) < len(item["options"]):
        mm = re.search(POS, item["options"]["ABCDEF".index(pred)])
        pk = ORD[mm.group(1)] if mm else None
    return "read-out" if pk != k else "viewpoint"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    items = {it["video_id"]: it for it in json.load(open(Path(a.data) / "items/pt_test.json"))}
    for pattern in a.runs:
        rows = {}
        for f in glob.glob(pattern):
            for line in open(f):
                r = json.loads(line)
                if r.get("id") != "__config__":
                    rows[r["id"]] = r
        wrong = [r for r in rows.values() if not r.get("correct")]
        c = Counter(classify(r, items[str(r["id"])]) for r in wrong)
        print(f"{pattern}: {len(wrong)} of {len(rows)} wrong", dict(c.most_common()))


if __name__ == "__main__":
    main()
