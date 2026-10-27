"""Direct-answer baselines with the benchmark's prompt, frames and parser.
sc_k = 0: one greedy answer. sc_k = k: k sampled answers and a majority vote (self-consistency); the vote after
the first j samples is recorded for every j <= k."""
from __future__ import annotations

import collections
import time


def vote(preds):
    """Most frequent parsed answer, ties broken by first occurrence; None if no answer parsed."""
    keys = [tuple(p) if isinstance(p, list) else p for p in preds if p is not None]
    if not keys:
        return None
    c = collections.Counter(keys)
    best = max(c.values())
    k = next(x for x in keys if c[x] == best)
    return list(k) if isinstance(k, tuple) else k


def direct(vlm, bench, item, cfg) -> dict:
    t0 = time.time()
    vlm.reset()
    rec = dict(id=item.id, gold=item.gold)
    if item.video is None:
        rec.update(status="NO_VIDEO", pred=None)
        return rec
    if cfg.sc_k > 0:
        vlm.sampling = dict(k=cfg.sc_k, temperature=cfg.sc_temperature, top_p=cfg.sc_top_p, top_k=cfg.sc_top_k,
                            chunk=cfg.sc_chunk)
    try:
        pred, txt = bench.answer(vlm, item, "", cfg)
    finally:
        vlm.sampling = None
    L = vlm.log
    tok_in = sum(x["tok_in"] for x in L)
    if cfg.sc_k > 0:
        preds = [bench.parse(item, t) for t in vlm.samples]
        votes = [vote(preds[: j + 1]) for j in range(len(preds))]
        rec.update(samples=preds, samples_raw=[t[-160:] for t in vlm.samples], votes=votes,
                   correct_at=[bool(bench.correct(item, v)) for v in votes], pred=votes[-1])
        rec["cost"] = dict(vlm_tok_in_per_call=tok_in, vlm_tok_out=vlm.sample_tok_out,
                           vlm_ms=round(sum(x["ms"] for x in L)))
    else:
        rec.update(pred=pred, raw=txt[-400:])
        rec["cost"] = dict(vlm_tok_in=tok_in, vlm_tok_out=sum(x["tok_out"] for x in L),
                           vlm_ms=round(sum(x["ms"] for x in L)))
    rec.update(status="OK", runtime_s=round(time.time() - t0, 1))
    return rec
