"""Steps 3-4, bind: on a clip of numbered carrier tracks around each evidence moment, the VLM answers one static
question that binds the target to a track. The prompt depends only on the identity cue chosen by the plan."""
from __future__ import annotations

import re

from . import video
from .boxes import draw_numbers
from .evidence import singular

PROMPTS = {
    "visible": ("These frames are from a video. Each {carrier_one} has a numbered box ({nums}) drawn on it; a number "
                "always stays with the same physical object. Which numbered {carrier_one} is the {target}? "
                "Reply with a single number, or NONE if the {target} cannot be seen."),
    "contained": ("These frames are from a video. Each {carrier_one} has a numbered box ({nums}) drawn on it; a number "
                  "always stays with the same physical object. Which numbered {carrier_one} is the {target} under or "
                  "inside right after it is hidden? Reply with a single number, or NONE if you cannot tell."),
}


def clip_indices(n, fps, cfg, start=None, end=None):
    """Frame indices at bind_fps: a window of bind_window_s around [start, end], or the whole video."""
    step = max(1, int(round(fps / cfg.bind_fps)))
    if start is None:
        lo, hi = 0, n - 1
    else:
        half = int(cfg.bind_window_s * fps / 2)
        lo, hi = max(0, start - half), min(n - 1, end + half)
    idx = list(range(lo, hi + 1, step))
    if len(idx) > cfg.bind_max_frames:
        idx = video.uniform(hi - lo + 1, cfg.bind_max_frames)
        idx = [lo + i for i in idx]
    return idx


def bind(vlm, frames, fps, carriers, target, carrier_name, cue, moments_, cfg):
    """One question per moment, or one on the whole video if no moment was found. The latest valid answer wins.
    Returns (bound track id or None, per-call records, prompt)."""
    n = len(carriers)
    nums = ", ".join(str(i) for i in range(1, n + 1))
    prompt = PROMPTS[cue].format(carrier_one=singular(carrier_name), nums=nums, target=target)
    windows = [(m["start"], m["end"]) for m in moments_] if cfg.bind_window == "moment" else []
    windows = windows or [(None, None)]
    recs, bound = [], None
    for s, e in windows:
        idx = clip_indices(len(frames), fps, cfg, s, e)
        imgs = [draw_numbers(frames[t], {i + 1: carriers[i][t][1] for i in range(n)}) for t in idx]
        reply = vlm.ask_clip(imgs, idx, fps, len(frames), prompt)
        m = re.search(r"\d+", reply)
        num = int(m.group(0)) if m and 1 <= int(m.group(0)) <= n else None
        recs.append(dict(window=[s, e], n_frames=len(idx), reply=reply, bound=num))
        if num is not None:
            bound = num
    return bound, recs, prompt
