"""Step 2, perceive: the VLM boxes every planned entity on a few frames, and SAM2.1 tracks all instances.
A first pass runs at base_fps. Carriers that overlap, move fast between sampled frames or are lost early are
tracked again in a dense pass at a rate close to the native one."""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
from PIL import Image

from . import video
from .boxes import dedupe, iou

GROUND_PROMPT = ('Locate every instance of the following objects in the image, if visible: {names}. Output a JSON '
                 'list of {{"bbox_2d": [x1, y1, x2, y2], "label": "<one of the names above>"}}; output [] if none is '
                 'visible.')


def _label_of(label: str, names: list[str]):
    """Map a returned label to a requested name: exact match, then containment, then shared words."""
    lab = label.strip().lower()
    low = [n.lower() for n in names]
    if lab in low:
        return names[low.index(lab)]
    for n, l in zip(names, low):
        if lab in l or l in lab:
            return n
    words = set(lab.split())
    best = max(names, key=lambda n: len(words & set(n.lower().split())))
    return best if words & set(best.lower().split()) else None


def ground(vlm, frames, names, cfg):
    """Boxes for all entities, one VLM call per frame, on ground_k evenly spaced frames. Entities found on none of
    them get a second round on the frames in between. Returns {name: {frame index: [boxes]}}."""
    h, w = frames[0].shape[:2]
    out = {n: {} for n in names}

    def ground_round(idx, pending):
        for t in idx:
            reply = vlm.ask([Image.fromarray(frames[t])], GROUND_PROMPT.format(names="; ".join(pending)),
                            max_new_tokens=96 + 64 * len(pending))
            per = {n: [] for n in pending}
            m = re.search(r"\[.*\]", reply, re.S)
            items = []
            if m:
                try:
                    items = json.loads(m.group(0))
                except json.JSONDecodeError:
                    items = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                n = _label_of(str(it.get("label", "")), pending) if len(pending) > 1 else pending[0]
                b = it.get("bbox_2d")
                if n and b and len(b) == 4:
                    x0, y0, x1, y1 = (float(v) for v in b)
                    per[n].append([x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h])
            for n in pending:
                out[n][int(t)] = dedupe(per[n])

    first = video.uniform(len(frames), cfg.ground_k)
    ground_round(first, list(names))
    missing = [n for n in names if not any(out[n].values())]
    if missing and len(first) > 1:
        second = sorted({(a + b) // 2 for a, b in zip(first[:-1], first[1:])} - set(first))
        ground_round(second, missing)
    return out


def start(found, single: bool, rule: str):
    """Frame and boxes that prompt the tracker. Several instances: the earliest frame that shows the most common
    instance count. A single target or subject: its largest box."""
    nz = {t: b for t, b in found.items() if b}
    if not nz:
        return None, []
    if single:
        t = max(nz, key=lambda t: max((b[2] - b[0]) * (b[3] - b[1]) for b in nz[t]))
        b = max(nz[t], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return t, [b]
    n = Counter(len(b) for b in nz.values()).most_common(1)[0][0]
    cands = sorted(t for t, b in nz.items() if len(b) == n)
    if rule == "largest_modal":
        t = max(cands, key=lambda t: np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in nz[t]]))
    else:
        t = cands[0]
    return t, nz[t]


def crowded(tracks, cfg) -> bool:
    """True if two instances overlap, or a box moves more than speed_ratio box widths in one step."""
    if not tracks:
        return False
    n_t = len(tracks[0])
    for t in range(n_t):
        bs = [tr[t][1] for tr in tracks if tr[t][1] is not None]
        if any(iou(a, b) > cfg.crowd_iou for i, a in enumerate(bs) for b in bs[i + 1:]):
            return True
    for tr in tracks:
        for t in range(1, n_t):
            a, b = tr[t - 1][1], tr[t][1]
            if a is not None and b is not None:
                if abs((a[0] + a[2]) - (b[0] + b[2])) / 2 > cfg.speed_ratio * max(1.0, a[2] - a[0]):
                    return True
    return False


def lost(tracks, cfg) -> bool:
    """True if the instances are no longer all tracked before keep_until of the video."""
    n_t = len(tracks[0])
    last = max((t for t in range(n_t) if all(tr[t][1] is not None for tr in tracks)), default=-1)
    return last < cfg.keep_until * (n_t - 1)


# Frames, boxes and tracks of the current video, shared by all questions about it.
CACHE: dict = {}


def perceive(vlm, tracker, path, plan, cfg):
    """Ground and track the plan's entities. Returns dict(base=(frames, fps), bind=(frames, fps), native_fps,
    entities, dense, reused); `bind` is the dense grid when the dense pass ran."""
    if CACHE.get("video") != str(path):
        CACHE.clear()
        nat, n_nat = video.probe(path)
        base_rate = min(cfg.base_fps, cfg.max_base_frames / max(n_nat / nat, 1e-6))
        CACHE.update(video=str(path), base=video.read(path, base_rate, cfg.max_side), ents={}, dense=None)
    frames, fps, native = CACHE["base"]
    want = [(e["name"], e["role"]) for e in plan["entities"]]
    pending = [k for k in want if k not in CACHE["ents"]]
    reused = len(want) - len(pending)
    if pending:
        found_all = ground(vlm, frames, [n for n, _ in pending], cfg)
        for name, role in pending:
            found = found_all[name]
            t0, boxes = start(found, single=role != "carrier", rule=cfg.init_rule)
            tracks = tracker.track(frames, t0, boxes) if t0 is not None else []
            CACHE["ents"][(name, role)] = dict(role=role, found={int(k): v for k, v in found.items()}, start=t0,
                                               boxes=boxes, tracks=tracks, tracks_dense=None)
    ents = {name: CACHE["ents"][(name, role)] for name, role in want}
    bind_grid, dense = (frames, fps), False
    multi = [e["tracks"] for e in ents.values() if e["role"] == "carrier" and len(e["tracks"]) > 1]
    if any(crowded(tr, cfg) or lost(tr, cfg) for tr in multi):
        # Target rate: max_track_frames over the video, at most native. video.read takes every k-th native frame,
        # k = round(native / rate), so the dense pass may hold somewhat more than max_track_frames frames.
        hi = min(native, cfg.max_track_frames / max(len(frames) / fps, 1e-6))
        if hi > fps * 1.5:
            if CACHE["dense"] is None:
                CACHE["dense"] = video.read(path, hi, cfg.max_side)
            frames2, fps2, _ = CACHE["dense"]
            for e in ents.values():
                if e["role"] in ("carrier", "target") and e["start"] is not None and e["tracks_dense"] is None:
                    t0 = min(len(frames2) - 1, int(round(e["start"] * fps2 / fps)))
                    e["tracks_dense"] = tracker.track(frames2, t0, e["boxes"])
            bind_grid, dense = (frames2, fps2), True
    for e in ents.values():
        e["tracks_bind"] = e["tracks_dense"] if dense and e["tracks_dense"] is not None else e["tracks"]
    return dict(base=(frames, fps), bind=bind_grid, native_fps=native, entities=ents, dense=dense, reused=reused)
