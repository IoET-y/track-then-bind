"""An open model behind an OpenAI-compatible chat server (vLLM, or scripts/gemma_server.py) in the VLM role.

Frame conventions follow InternVL3.5's multi-image prompts:
  * the frames of a video or clip are separate images, one tile each, each preceded by "Frame<i> (<t> s): ";
  * a single image keeps the model's default tiling;
  * TOC-Bench frames are sent as TOC-Bench's InternVL runner sends them ("Frame<i>: ", no times).
Requests with more than `max_frames` images are thinned uniformly. Greedy decoding.
Gemma 4 returns boxes as [y1, x1, y2, x2]; with box_order="yx" these are reordered to [x1, y1, x2, y2]."""
from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


def _jpeg_b64(img, quality=90) -> str:
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    b = io.BytesIO()
    im.convert("RGB").save(b, format="JPEG", quality=quality)
    return base64.b64encode(b.getvalue()).decode()


def yx_to_xy(reply: str) -> str:
    num = r"\s*(-?[\d.]+)\s*"
    return re.sub(r'"bbox_2d"\s*:\s*\[' + num + "," + num + "," + num + "," + num + r"\]",
                  lambda m: '"bbox_2d": [%s, %s, %s, %s]' % (m.group(2), m.group(1), m.group(4), m.group(3)), reply)


class ServerVLM:
    def __init__(self, model: str, base_url: str, max_frames: int = 128, box_order: str = "xy"):
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        self.max_frames = max_frames
        self.box_yx = box_order == "yx"
        self.http = requests.Session()
        self.http.trust_env = False
        self.log: list[dict] = []
        self.sampling = None
        self.samples: list[str] = []
        self.sample_tok_out = 0

    def reset(self):
        self.log = []

    def _thin(self, n):
        if n <= self.max_frames:
            return list(range(n))
        return sorted({int(round(x)) for x in np.linspace(0, n - 1, self.max_frames)})

    def _call(self, kind, content, prompt, max_new_tokens, system=None, n_images=0):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = dict(model=self.model, messages=msgs, max_tokens=max_new_tokens, temperature=0.0)
        if n_images > 1:
            body["mm_processor_kwargs"] = {"max_dynamic_patch": 1}
        t0, tries = time.perf_counter(), 0
        while True:
            tries += 1
            try:
                r = self.http.post(self.url, data=json.dumps(body), headers={"Content-Type": "application/json"},
                                   timeout=900)
                j = r.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, ValueError):
                if tries > 20:
                    raise
                time.sleep(min(60, 3 * tries))
                continue
            if "error" in j or "choices" not in j:
                if r.status_code >= 500 and tries <= 5:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"server {r.status_code}: {str(j.get('error', j))[:300]}")
            break
        txt = (j["choices"][0]["message"].get("content") or "").strip()
        u = j.get("usage", {})
        self.log.append(dict(kind=kind, prompt=prompt, reply=txt, n_images=n_images,
                             tok_in=int(u.get("prompt_tokens", 0)), tok_out=int(u.get("completion_tokens", 0)),
                             tok_reason=0, ms=1000 * (time.perf_counter() - t0), tries=tries,
                             finish=j["choices"][0].get("finish_reason"), model=j.get("model")))
        return txt

    @staticmethod
    def _img(b64):
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}

    def _frames_content(self, b64s, stamps, prompt):
        keep = self._thin(len(b64s))
        c = []
        for j, i in enumerate(keep):
            c.append({"type": "text",
                      "text": f"Frame{j + 1} ({stamps[i]:.1f} s): " if stamps is not None else f"Frame{j + 1}: "})
            c.append(self._img(b64s[i]))
        return c + [{"type": "text", "text": prompt}], len(keep)

    def ask(self, images, prompt: str, max_new_tokens: int = 256) -> str:
        if len(images) > 1:
            c, n = self._frames_content([_jpeg_b64(im) for im in images], None, prompt)
        else:
            c, n = [self._img(_jpeg_b64(im)) for im in images] + [{"type": "text", "text": prompt}], len(images)
        txt = self._call("image" if images else "text", c, prompt, max_new_tokens, n_images=n)
        if self.box_yx and len(images) == 1 and "bbox_2d" in prompt:
            self.log[-1]["reply_raw"] = txt
            txt = yx_to_xy(txt)
            self.log[-1]["reply"] = txt
        return txt

    def ask_clip(self, frames, idx, fps: float, n_total: int, prompt: str, max_new_tokens: int = 8) -> str:
        c, n = self._frames_content([_jpeg_b64(f) for f in frames], [i / fps for i in idx], prompt)
        return self._call("clip", c, prompt, max_new_tokens, n_images=n)

    def ask_video(self, video: Path, prompt: str, fps: float, max_new_tokens: int = 1024) -> str:
        cap = cv2.VideoCapture(str(video))
        n, nat = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
        k = max(2, int(round(n / nat * fps)))
        want = set(min(n - 1, int(round(x))) for x in np.linspace(0, n - 1, k))
        b64s, stamps, i = [], [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                b64s.append(_jpeg_b64(f[:, :, ::-1]))
                stamps.append(i / nat)
            i += 1
        cap.release()
        c, m = self._frames_content(b64s, stamps, prompt)
        return self._call("video", c, prompt, max_new_tokens, n_images=m)

    def ask_b64_frames(self, system: str, user: str, frames_b64: list[str], max_new_tokens: int) -> str:
        c, n = self._frames_content(list(frames_b64), None, user)
        return self._call("benchmark_frames", c, user, max_new_tokens, system=system, n_images=n)
