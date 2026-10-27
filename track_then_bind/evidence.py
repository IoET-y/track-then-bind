"""From tracks to evidence: visibility states and segments, evidence moments, and the sentences given to the
read-out. Apart from viewpoint_fact, nothing here calls a model."""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import video
from .boxes import iou

ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]

FACING_PROMPT = ("Is the person who moves the objects positioned opposite the camera, on the other side of the table "
                 "(so the camera looks at them from the front)? Reply YES or NO.")


def singular(name: str) -> str:
    n = name.strip()
    for a, b in (("ies", "y"), ("ches", "ch"), ("shes", "sh"), ("ses", "s"), ("s", "")):
        if n.endswith(a) and len(n) > len(a) + 1:
            return n[: -len(a)] + b
    return n


def states(track, cfg):
    """Per-frame visibility from the mask area relative to the track's own 90th-percentile area."""
    areas = np.array([a if a else 0 for a, _ in track], float)
    ref = np.percentile(areas[areas > 0], 90) if (areas > 0).any() else 1.0
    return ["fully visible" if a >= cfg.full * ref else "partly hidden" if a >= cfg.partial * ref else "not visible"
            for a in areas]


def segments(st, fps, cfg):
    """Runs of equal state as [state, first, last]; runs shorter than min_seg_s are merged into a neighbour."""
    segs = []
    for i, s in enumerate(st):
        if segs and segs[-1][0] == s:
            segs[-1][2] = i
        else:
            segs.append([s, i, i])
    minf = max(1, int(cfg.min_seg_s * fps))
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for j, sg in enumerate(segs):
            if sg[2] - sg[1] + 1 < minf:
                k = j - 1 if j > 0 else j + 1
                segs[k][1], segs[k][2] = min(segs[k][1], sg[1]), max(segs[k][2], sg[2])
                segs.pop(j)
                merged = []
                for x in segs:
                    if merged and merged[-1][0] == x[0]:
                        merged[-1][2] = x[2]
                    else:
                        merged.append(x)
                segs = merged
                changed = True
                break
    return segs


def at_edge(b, w, h, cfg):
    return bool(b) and (b[0] <= cfg.edge * w or b[1] <= cfg.edge * h or b[2] >= (1 - cfg.edge) * w
                        or b[3] >= (1 - cfg.edge) * h)


def timeline_text(name, track, fps, h, w, cfg) -> str:
    """One entity's visibility timeline: segments, events and overall motion."""
    st = states(track, cfg)
    edges = [at_edge(b, w, h, cfg) for _, b in track]
    segs = segments(st, fps, cfg)
    parts = []
    for j, (s, a, b) in enumerate(segs):
        txt = f"{a / fps:.1f}-{(b + 1) / fps:.1f} s {s}"
        if s == "not visible":
            if j == 0:
                txt += " (has not appeared yet)"
            else:
                prev_edge = edges[segs[j - 1][2]] if segs[j - 1][2] < len(edges) else False
                txt += (" (vanished at the frame edge, so it left the frame)" if prev_edge
                        else " (vanished inside the frame, so it was blocked or hidden)")
            if j == len(segs) - 1 and j > 0:
                txt += " until the end"
        parts.append(txt)
    n_hidden = sum(1 for j, sg in enumerate(segs) if sg[0] == "not visible" and j > 0)
    n_partial = sum(1 for sg in segs if sg[0] == "partly hidden")
    events = []
    for j in range(1, len(segs)):
        prev, cur, t = segs[j - 1][0], segs[j][0], segs[j][1] / fps
        if prev == "not visible" and cur != "not visible":
            events.append(f"{t:.1f} s {'enters the frame' if j == 1 else 'comes back into view'}")
        if cur == "partly hidden" and prev == "fully visible":
            events.append(f"{t:.1f} s gets partly hidden")
        if cur == "not visible":
            pe = edges[segs[j - 1][2]] if segs[j - 1][2] < len(edges) else False
            what = "leaves the frame" if pe else "gets fully hidden"
            if j == len(segs) - 1:
                what += " and disappears for good"
            events.append(f"{t:.1f} s {what}")
    vis = [(i, b) for i, (a_, b) in enumerate(track) if b is not None and a_ > 0]
    motion = ""
    if len(vis) >= 6:
        k3 = max(1, len(vis) // 3)
        c0 = np.mean([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b in vis[:k3]], axis=0)
        c1 = np.mean([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b in vis[-k3:]], axis=0)
        dx, dy = (c1 - c0) / np.array([w, h])
        if max(abs(dx), abs(dy)) < 0.1:
            motion = " Overall motion: stays in roughly the same place."
        elif abs(dx) >= abs(dy):
            motion = f" Overall motion: moves from {'left to right' if dx > 0 else 'right to left'}."
        else:
            motion = f" Overall motion: moves from {'top to bottom' if dy > 0 else 'bottom to top'}."
    return (f"'{name}': " + "; ".join(parts) + f". Times it went fully out of view after being seen: {n_hidden}; "
            f"times partly hidden: {n_partial}. Events in order: " + ("; ".join(events) or "none") + "." + motion)


def moments(target_track, carriers, fps, h, w, cue, cfg, hits=None):
    """Evidence moments, the frames at which the target can be identified.

    visible: the frames on which grounding found the target (`hits`, as bind-grid indices), merged into windows.
        A mask alone cannot tell a face-up card from its back, so these come from the VLM's boxes.
    contained: each time the target's own track vanishes inside the frame.
    Each moment also records the carrier whose box overlaps the target's most. Returns [] if there is no anchor."""
    out = []
    if cue == "visible":
        half = int(cfg.bind_window_s * fps / 2)
        for t, b in sorted(hits or []):
            ov = {i + 1: iou(b, c[t][1]) for i, c in enumerate(carriers) if t < len(c) and c[t][1] is not None}
            best = max(ov, key=ov.get) if ov and max(ov.values()) > 0 else None
            if out and t - half <= out[-1]["end"] + half:
                out[-1]["end"] = t
                out[-1]["carrier_by_overlap"].append(best)
            else:
                out.append(dict(kind="visible", start=t, end=t, carrier_by_overlap=[best]))
        return out
    if target_track is None:
        return []
    st = states(target_track, cfg)
    segs = segments(st, fps, cfg)
    if cue == "contained":
        for j in range(1, len(segs)):
            if segs[j][0] != "not visible" or segs[j - 1][0] == "not visible":
                continue
            last = segs[j - 1][2]
            lb = target_track[last][1]
            if lb is None or at_edge(lb, w, h, cfg):
                continue
            overlap = {i + 1: iou(lb, c[last][1]) for i, c in enumerate(carriers) if c[last][1] is not None}
            best = max(overlap, key=overlap.get) if overlap and max(overlap.values()) > 0 else None
            out.append(dict(kind="contained", start=last, end=segs[j][1], carrier_by_overlap=best))
    return out


def final_order(carriers):
    """The last frame on which every carrier is tracked, and the carriers' ids (1-based) from left to right."""
    n_t = len(carriers[0]) if carriers else 0
    for t in range(n_t - 1, -1, -1):
        if all(c[t][1] is not None for c in carriers):
            return t, sorted(range(1, len(carriers) + 1),
                             key=lambda i: (carriers[i - 1][t][1][0] + carriers[i - 1][t][1][2]) / 2)
    return None, None


def _ord(r):
    return ORD[r] if r < len(ORD) else f"{r + 1}th"


def binding_text(target, carrier_name, bound, t_bind, cue, carriers, fps) -> str:
    """The binding as a fact about the target, followed by the bound carrier's final position."""
    t_end, order = final_order(carriers)
    if order is None or bound is None:
        return ""
    one = singular(carrier_name)
    when = f" (seen at {t_bind:.1f} s)" if t_bind is not None else ""
    rel = f"was hidden under {one} #{bound}{when}" if cue == "contained" else f"is {one} #{bound}{when}"
    rank = order.index(bound)
    return (f"The {target} {rel}. The tracker followed every {one} through the rest of the video. In the last frame "
            f"where all {len(carriers)} are tracked ({t_end / fps:.1f} s), from the camera's left to right they are: "
            + ", ".join(f"#{i}" for i in order) + f". So the {target} is "
            + (f"under the {_ord(rank)} {one}" if cue == "contained" else f"the {_ord(rank)} {one}")
            + " from the camera's left.")


def viewpoint_fact(vlm, frames):
    """Whether the person faces the camera, asked once on 4 evenly spaced frames and stated as a fact.
    Returns (sentence, mirrored)."""
    imgs = [Image.fromarray(frames[i]) for i in video.uniform(len(frames), 4)]
    reply = vlm.ask(imgs, FACING_PROMPT, max_new_tokens=4)
    if reply.strip().upper().startswith("Y"):
        return "The person faces the camera, so the person's left is the camera's right, and vice versa.", True
    return ("The camera shows the scene from the person's own viewpoint, so the person's left is the camera's "
            "left."), False


def person_rank_fact(target, carrier_name, cue, rank, n, mirrored) -> str:
    r = n - 1 - rank if mirrored else rank
    one = singular(carrier_name)
    where = f"under the {_ord(r)} {one}" if cue == "contained" else f"the {_ord(r)} {one}"
    return f"From the person's point of view, the {target} is {where} from the left."
