"""Accuracy of runs, and paired comparisons between two runs on the same items.

  python scripts/evaluate.py "runs/qwen8b/pt*.jsonl"
  python scripts/evaluate.py --compare "runs/qwen8b_direct/pt*.jsonl" "runs/qwen8b/pt*.jsonl"
  python scripts/evaluate.py --vote-at 13 "runs/qwen8b_sc/pt*.jsonl"
  python scripts/evaluate.py --first-letter "runs/internvl/pt*.jsonl"

Items that ended in an error count as wrong. For the Perception Test cup games, PT-65 (three cups, at least one
swap) is reported as well. A comparison gives the accuracy difference B - A with an item-level bootstrap 95% CI
and an exact McNemar test."""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
from math import comb
from pathlib import Path


def first_letter(text):
    """The answer letter of a reply without a parsable \\boxed{}: a leading letter or "answer is X"."""
    t = str(text or "")
    m = (re.search(r"\\boxed\{\s*\(?([A-F])", t) or re.match(r"\s*\(?([A-F])[\).:\s]", t)
         or re.search(r"(?:answer|Answer)\s*(?:is|:)?\s*\(?([A-F])\b", t))
    return m.group(1) if m else None


def load(pattern: str, vote_at: int = 0, lenient: bool = False) -> dict:
    """{item id: correct} over all files matching the pattern."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if r.get("id") == "__config__":
                continue
            ok = bool(r.get("correct"))
            if vote_at and r.get("correct_at"):
                ok = bool(r["correct_at"][min(vote_at, len(r["correct_at"])) - 1])
            elif lenient and r.get("status") == "OK":
                pred = r.get("pred") or first_letter(r.get("readout_raw", r.get("raw")))
                ok = pred == r.get("gold")
            out[str(r["id"])] = ok
    if not out:
        raise SystemExit(f"no records match {pattern}")
    return out


def pt65(data: Path) -> set:
    meta = json.load(open(data / "shellgame/perception_test/cup_games_filtered_metadata.json"))
    return {k for k, v in meta.items()
            if v["metadata"]["num_of_cups"] == 3 and (v["metadata"]["num_of_swaps"] or 0) > 0}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(A: dict, B: dict, ids: list, seed: int = 0, n_boot: int = 10000) -> dict:
    a = [int(A[i]) for i in ids]
    b = [int(B[i]) for i in ids]
    n = len(ids)
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return dict(n=n, A=100 * sum(a) / n, B=100 * sum(b) / n, diff=100 * sum(d) / n,
                lo=100 * boots[int(0.025 * n_boot)], hi=100 * boots[int(0.975 * n_boot)],
                b_only=wins, a_only=losses, p=mcnemar(wins, losses))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="file globs, one per run")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--vote-at", type=int, default=0,
                    help="self-consistency runs: majority vote of the first k samples")
    ap.add_argument("--first-letter", action="store_true",
                    help="read replies without a parsable \\boxed{} answer by their first option letter")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    sub = pt65(Path(a.data)) if (Path(a.data) / "shellgame").exists() else set()

    def subsets(ids):
        yield "all", ids
        if sub and any(i in sub for i in ids):
            yield "PT-65", [i for i in ids if i in sub]

    for pattern in a.runs:
        R = load(pattern, a.vote_at, a.first_letter)
        for name, ids in subsets(sorted(R)):
            print(f"{pattern}  {name}: {100 * sum(R[i] for i in ids) / len(ids):.1f}% of {len(ids)}")
    if a.compare:
        A, B = (load(p, a.vote_at, a.first_letter) for p in a.compare)
        for name, ids in subsets(sorted(set(A) & set(B))):
            r = compare(A, B, ids)
            print(f"{name}: n={r['n']}  A={r['A']:.1f}  B={r['B']:.1f}  B-A={r['diff']:+.1f} "
                  f"[{r['lo']:+.1f}, {r['hi']:+.1f}]  B only {r['b_only']}, A only {r['a_only']}, "
                  f"McNemar p={r['p']:.3g}")


if __name__ == "__main__":
    main()
