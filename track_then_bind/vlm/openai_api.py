"""An OpenAI model (Chat Completions API) in the VLM role.

Inputs mirror the Qwen3-VL backend call by call:
  ask             the images (JPEG, quality 90), then the prompt;
  ask_clip        the frames, each preceded by its timestamp ("<t seconds>");
  ask_video       frames sampled at `fps` over the video, at most 360x640 pixels each, each preceded by its timestamp;
  ask_b64_frames  TOC-Bench's system prompt, frames and user prompt.
The models return grounding boxes in pixels of the image they were sent; replies to a single-image box prompt are
converted to the 0-1000 grid that the agent parses.

Reads OPENAI_API_KEY and, optionally, OPENAI_BASE_URL."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

VIDEO_MAX_PIXELS = 360 * 640


def _jpeg(img, quality=90, max_pixels=None) -> bytes:
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    im = im.convert("RGB")
    if max_pixels and im.width * im.height > max_pixels:
        s = math.sqrt(max_pixels / (im.width * im.height))
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BICUBIC)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=quality)
    return b.getvalue()


def to_rel1000(reply: str, w: int, h: int) -> str:
    """Rewrite pixel "bbox_2d" values of a reply on the 0-1000 grid; the rest of the reply is kept."""
    def conv(m):
        x0, y0, x1, y1 = (float(v) for v in m.groups())
        r = [round(1000 * x0 / w), round(1000 * y0 / h), round(1000 * x1 / w), round(1000 * y1 / h)]
        return '"bbox_2d": [' + ", ".join(str(v) for v in r) + "]"
    num = r"\s*(-?[\d.]+)\s*"
    return re.sub(r'"bbox_2d"\s*:\s*\[' + num + "," + num + "," + num + "," + num + r"\]", conv, reply)


class OpenAIVLM:
    def __init__(self, model: str, reasoning_effort: str | None = None, reasoning_budget: int = 8192):
        """reasoning_effort: None keeps the API default. reasoning_budget is added to every call's answer budget."""
        self.model = model
        self.effort = reasoning_effort
        self.reasoning_budget = reasoning_budget
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.key = os.environ["OPENAI_API_KEY"]
        self.http = requests.Session()
        self.log: list[dict] = []
        self.sampling = None
        self.samples: list[str] = []
        self.sample_tok_out = 0

    def reset(self):
        self.log = []

    def _call(self, kind, content, prompt, max_new_tokens, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = dict(model=self.model, messages=msgs, max_completion_tokens=max_new_tokens + self.reasoning_budget,
                    seed=42)
        if self.effort:
            body["reasoning_effort"] = self.effort
        data = json.dumps(body).encode()
        hdr = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}
        t0, tries = time.perf_counter(), 0
        while True:
            tries += 1
            try:
                r = self.http.post(self.url, data=data, headers=hdr, timeout=600)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if tries > 12:
                    raise
                time.sleep(min(60, 3 * tries))
                continue
            try:
                j = r.json()
            except ValueError:
                j = {"error": {"message": f"non-JSON reply {r.status_code}"}}
            if "error" not in j:
                break
            if (r.status_code == 429 or r.status_code >= 500) and tries <= 12:
                time.sleep(min(120, 5 * 2 ** min(tries, 5)))
                continue
            raise RuntimeError(f"OpenAI {r.status_code}: {str(j['error'].get('message', ''))[:300]}")
        ch = j["choices"][0]
        txt = (ch["message"].get("content") or "").strip()
        u = j.get("usage", {})
        self.log.append(dict(kind=kind, prompt=prompt, reply=txt,
                             n_images=sum(c.get("type") == "image_url" for c in content),
                             tok_in=int(u.get("prompt_tokens", 0)), tok_out=int(u.get("completion_tokens", 0)),
                             tok_reason=int(u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)),
                             ms=1000 * (time.perf_counter() - t0), tries=tries, finish=ch.get("finish_reason"),
                             model=j.get("model")))
        return txt

    @staticmethod
    def _content(jpegs: list[bytes], prompt: str, stamps=None):
        c = []
        for i, b in enumerate(jpegs):
            if stamps is not None:
                c.append({"type": "text", "text": f"<{stamps[i]:.1f} seconds>"})
            c.append({"type": "image_url",
                      "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
        return c + [{"type": "text", "text": prompt}]

    def ask(self, images, prompt: str, max_new_tokens: int = 256) -> str:
        txt = self._call("image" if images else "text", self._content([_jpeg(im) for im in images], prompt), prompt,
                         max_new_tokens)
        if len(images) == 1 and "bbox_2d" in prompt:
            im = images[0]
            w, h = (im.width, im.height) if isinstance(im, Image.Image) else (im.shape[1], im.shape[0])
            self.log[-1]["reply_raw"] = txt
            txt = to_rel1000(txt, w, h)
            self.log[-1]["reply"] = txt
        return txt

    def ask_clip(self, frames, idx, fps: float, n_total: int, prompt: str, max_new_tokens: int = 8) -> str:
        return self._call("clip", self._content([_jpeg(f) for f in frames], prompt, [i / fps for i in idx]), prompt,
                          max_new_tokens)

    def ask_video(self, video: Path, prompt: str, fps: float, max_new_tokens: int = 1024) -> str:
        cap = cv2.VideoCapture(str(video))
        n, nat = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
        k = max(2, int(round(n / nat * fps)))
        want = set(min(n - 1, int(round(x))) for x in np.linspace(0, n - 1, k))
        frames, stamps, i = [], [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                frames.append(_jpeg(f[:, :, ::-1], max_pixels=VIDEO_MAX_PIXELS))
                stamps.append(i / nat)
            i += 1
        cap.release()
        return self._call("video", self._content(frames, prompt, stamps), prompt, max_new_tokens)

    def ask_b64_frames(self, system: str, user: str, frames_b64: list[str], max_new_tokens: int) -> str:
        return self._call("benchmark_frames", self._content([base64.b64decode(b) for b in frames_b64], user), user,
                          max_new_tokens, system=system)
