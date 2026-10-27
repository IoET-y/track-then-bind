"""Track, Then Bind: plan -> perceive -> evidence moments -> bind -> read out.
One code path and one Config for every benchmark; benchmark-specific prompts and parsers live in benchmarks.py."""
from __future__ import annotations

import glob
import json
import time

from . import evidence, plan as planner
from .bind import bind, clip_indices
from .boxes import draw_numbers
from .evidence import singular
from .perceive import perceive

OVERLAY_HEAD = ("These frames are from a video. Each {carrier_one} has a numbered box ({nums}) drawn on it; a number "
                "always stays with the same physical object.\n\n")


def load_plans(pattern: str) -> dict:
    """Plans recorded by another run (a glob of output files), by item id."""
    plans = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if r.get("id") != "__config__" and isinstance(r.get("plan"), dict):
                plans[str(r["id"])] = dict(r["plan"], raw=r.get("plan_raw", ""))
    return plans


def _overlay(vlm, frames, fps, carriers, carrier_name, item, bench, cfg):
    """Ablation without the bind question: the numbered tracks on the whole video with the benchmark question."""
    n = len(carriers)
    idx = clip_indices(len(frames), fps, cfg)
    imgs = [draw_numbers(frames[t], {i + 1: carriers[i][t][1] for i in range(n)}) for t in idx]
    head = OVERLAY_HEAD.format(carrier_one=singular(carrier_name), nums=", ".join(str(i) for i in range(1, n + 1)))
    txt = vlm.ask_clip(imgs, idx, fps, len(frames), head + item.raw["prompt"], max_new_tokens=1024)
    return bench.extract_mc(txt), txt, len(idx)


def run(vlm, tracker, bench, item, cfg, with_direct=False, dump=None, plans=None) -> dict:
    t0 = time.time()
    vlm.reset()
    tracker.calls, tracker.ms, tracker.frames = 0, 0.0, 0
    rec = dict(id=item.id, gold=item.gold)
    if item.video is None:
        rec.update(status="NO_VIDEO", pred=None)
        return rec
    if with_direct:
        p, txt = bench.answer(vlm, item, "", cfg)
        rec.update(direct=p, direct_raw=txt[-400:])
        n_direct = len(vlm.log)
    else:
        n_direct = 0

    # 1. Plan, or a plan recorded by another run.
    given = (plans or {}).get(str(item.id))
    if given:
        pl = dict(given, raw=given.get("raw", ""))
        rec["plan_given"] = True
    else:
        pl = planner.make_plan(vlm, item.video, item.question_for_plan, cfg)
    rec["plan"] = {k: v for k, v in pl.items() if k != "raw"}
    rec["plan_raw"] = pl["raw"][:300]

    ev_lines, facts, overlay = [], None, None
    if pl["entities"]:
        # 2. Perceive.
        P = perceive(vlm, tracker, item.video, pl, cfg)
        (frames, fps), (bframes, bfps) = P["base"], P["bind"]
        h, w = frames[0].shape[:2]
        rec["perceive"] = dict(fps=round(fps, 2), bind_fps=round(bfps, 2), n_frames=len(frames), dense=P["dense"],
                               reused=P["reused"],
                               entities={k: dict(role=e["role"], start=e["start"], n=len(e["tracks"]))
                                         for k, e in P["entities"].items()})
        ev_lines.append(f"Video length {len(frames) / fps:.1f} s.")
        carriers = [(k, e) for k, e in P["entities"].items() if e["role"] == "carrier" and len(e["tracks_bind"]) > 1]
        targets = [(k, e) for k, e in P["entities"].items() if e["role"] == "target"]
        # 3-4. Evidence moments and binding, when a target must be told apart among several carriers.
        if pl["asks_final_location"] and pl["identity_cue"] in ("visible", "contained") and targets and carriers:
            tname, te = targets[0]
            cname, ce = carriers[0]
            ttrack = te["tracks_bind"][0] if te["tracks_bind"] else None
            scale = bfps / fps
            hits = [(min(len(bframes) - 1, int(round(t * scale))), max(b, key=lambda x: (x[2] - x[0]) * (x[3] - x[1])))
                    for t, b in te["found"].items() if b]
            ms = evidence.moments(ttrack, ce["tracks_bind"], bfps, h, w, pl["identity_cue"], cfg, hits=hits)
            if cfg.readout == "overlay" and bench.supports_overlay:
                overlay = _overlay(vlm, bframes, bfps, ce["tracks_bind"], cname, item, bench, cfg)
                rec["overlay"] = dict(moments=ms, n_frames=overlay[2])
                bound = None
            else:
                bound, brecs, bprompt = bind(vlm, bframes, bfps, ce["tracks_bind"], tname, cname, pl["identity_cue"],
                                             ms, cfg)
                rec["bind"] = dict(moments=ms, calls=brecs, prompt=bprompt, bound=bound)
            if dump is not None:
                json.dump(dict(fps=bfps, carriers=ce["tracks_bind"], target=te["tracks_bind"], found=te["found"],
                               carrier_found=ce["found"], base_fps=fps, video=str(item.video)),
                          open(f"{dump}/{item.id}.json", "w"), default=str)
            if bound is not None:
                last = [r for r in brecs if r["bound"] is not None][-1]
                t_bind = last["window"][0] / bfps if last["window"][0] is not None else None
                fact = evidence.binding_text(tname, cname, bound, t_bind, pl["identity_cue"], ce["tracks_bind"], bfps)
                ev_lines.append(fact)
                t_end, order = evidence.final_order(ce["tracks_bind"])
                rec["geometric_rank"] = order.index(bound) if order else None
                facts = [fact]
                if pl.get("viewpoint") == "person" and t_end is not None:
                    vf, mirrored = evidence.viewpoint_fact(vlm, frames)
                    facts += [vf, evidence.person_rank_fact(tname, cname, pl["identity_cue"], order.index(bound),
                                                            len(order), mirrored)]
                    rec["mirrored"] = mirrored
        # Visibility timelines of targets and subjects, on the first-pass grid.
        for k, e in P["entities"].items():
            if e["role"] in ("target", "subject"):
                if e["tracks"]:
                    ev_lines.append(evidence.timeline_text(k, e["tracks"][0], fps, h, w, cfg))
                else:
                    ev_lines.append(f"'{k}': not found by the detector in {2 * cfg.ground_k - 1} sampled frames.")
    if facts and cfg.readout == "evidence":
        ev_lines = [ev_lines[0]] + facts + [x for x in ev_lines[1:] if x not in facts]
    ev = "\n".join(x for x in ev_lines if x) if len(ev_lines) > 1 else ""
    rec["evidence"] = ev

    # 5. Read out with the benchmark's prompt and parser.
    if overlay is not None:
        pred, txt = overlay[:2]
        rec["readout"] = "overlay"
    elif facts and cfg.readout == "resolve":
        rec["facts"] = facts
        pred, txt = bench.resolve(vlm, item, "\n".join("- " + f for f in facts))
        rec["readout"] = "resolve"
    else:
        pred, txt = bench.answer(vlm, item, ev, cfg)
        rec["readout"] = "evidence"
    rec.update(pred=pred, readout_raw=txt[-400:], status="OK")
    if pred is None and ev:
        # An unparsable read-out falls back to the direct answer.
        pred, txt = bench.answer(vlm, item, "", cfg)
        rec.update(pred=pred, fallback="direct", fallback_raw=txt[-200:])

    L = vlm.log[n_direct:]
    rec["cost"] = dict(vlm_tok_in=sum(x["tok_in"] for x in L), vlm_tok_out=sum(x["tok_out"] for x in L),
                       vlm_ms=round(sum(x["ms"] for x in L)), tracker_ms=round(getattr(tracker, "ms", 0.0)),
                       tracker_obj_frames=getattr(tracker, "frames", 0))
    if with_direct:
        D = vlm.log[:n_direct]
        rec["cost_direct"] = dict(vlm_tok_in=sum(x["tok_in"] for x in D), vlm_tok_out=sum(x["tok_out"] for x in D),
                                  vlm_ms=round(sum(x["ms"] for x in D)))
    rec["calls"] = dict(vlm=len(vlm.log) - n_direct, tracker=tracker.calls,
                        by_kind={k: sum(1 for x in vlm.log[n_direct:] if x["kind"] == k)
                                 for k in sorted({x["kind"] for x in vlm.log[n_direct:]})})
    rec["runtime_s"] = round(time.time() - t0, 1)
    return rec
