"""Qwen3-VL in Hugging Face Transformers, greedy decoding.

Videos go through qwen-vl-utils at the requested frame rate with at most 360x640 pixels per frame. Clips of
pre-selected frames (bind clips, overlays) are passed as one video with their true timestamps."""
from __future__ import annotations

import base64
import io
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

VIDEO_MAX_PIXELS = 360 * 640


def video_metadata(frame_indices, fps: float, total_frames: int, n_processed: int) -> dict[str, Any]:
    """Metadata that places pre-sampled frames on the original video's time axis."""
    if not frame_indices:
        raise ValueError("at least one frame index is required")
    if n_processed < len(frame_indices):
        raise ValueError("preprocessing cannot drop frames")
    idx = list(frame_indices) + [frame_indices[-1]] * (n_processed - len(frame_indices))
    return {"fps": float(fps), "frames_indices": idx, "total_num_frames": int(total_frames)}


class QwenVLM:
    def __init__(self, path: str, device_map: str = "cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map=device_map, attn_implementation="sdpa").eval()
        self.max_new_tokens = 512
        self.log: list[dict[str, Any]] = []
        self.sampling = None      # dict(k, temperature, top_p, top_k, chunk) while drawing k samples per call
        self.samples: list[str] = []
        self.sample_tok_out = 0
        self.tmp = Path(tempfile.mkdtemp(prefix="ttb_clips_"))

    def reset(self):
        self.log = []

    def _rec(self, kind, prompt, reply, n_images=0, tok_in=0, tok_out=0, ms=0.0):
        self.log.append(dict(kind=kind, prompt=prompt, reply=reply, n_images=n_images, tok_in=int(tok_in),
                             tok_out=int(tok_out), ms=float(ms)))

    def _generate(self, **kw):
        """model.generate. While `sampling` is set, k sampled continuations of the same prompt are decoded into
        self.samples and the first one is returned."""
        if not self.sampling:
            return self.model.generate(**kw)
        sp = self.sampling
        kw.update(do_sample=True, temperature=sp["temperature"], top_p=sp["top_p"], top_k=sp["top_k"])
        n_in = kw["input_ids"].shape[1]
        tok = self.processor.tokenizer
        pad = tok.pad_token_id if tok.pad_token_id is not None else -1
        first, self.samples, self.sample_tok_out = None, [], 0
        left = sp["k"]
        # Qwen3-VL in transformers 4.57 cannot expand video inputs for num_return_sequences > 1.
        chunk = 1 if kw.get("pixel_values_videos") is not None else sp["chunk"]
        while left > 0:
            m = min(chunk, left)
            g = self.model.generate(**{**kw, "num_return_sequences": m})
            for row in g[:, n_in:]:
                ids = [int(t) for t in row if int(t) != pad]
                self.sample_tok_out += len(ids)
                self.samples.append(tok.decode(ids, skip_special_tokens=True).strip())
            first = g[:1] if first is None else first
            left -= m
        return first

    def _chat(self, messages, clip_metadata=None):
        """One generation for chat messages. Returns (reply, input tokens, output tokens, ms)."""
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, video_items, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True,
                                                                return_video_metadata=True)
        videos = metadatas = None
        if video_items is not None:
            videos, inferred = zip(*video_items)
            videos = list(videos)
            if clip_metadata is None:
                metadatas = list(inferred)
            else:
                idx, fps, total = clip_metadata
                metadatas = [video_metadata(idx, fps, total, int(v.shape[0])) for v in videos]
        inputs = self.processor(text=[text], images=images, videos=videos, video_metadata=metadatas, padding=True,
                                return_tensors="pt", do_resize=False, **video_kwargs).to(self.model.device)
        tok_in = int(inputs.input_ids.numel())
        self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            out = self._generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        trimmed = out[:, inputs.input_ids.shape[1]:]
        reply = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return reply.strip(), tok_in, int(trimmed.numel()), ms

    def ask(self, images: list[Image.Image], prompt: str, max_new_tokens: int = 256) -> str:
        """Images (possibly none) and a prompt, one turn."""
        self.max_new_tokens = max_new_tokens
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        txt, ti, to, ms = self._chat([{"role": "user", "content": content}])
        self._rec("image" if images else "text", prompt, txt, len(images), ti, to, ms)
        return txt

    def ask_clip(self, frames: list[np.ndarray], idx: list[int], fps: float, n_total: int, prompt: str,
                 max_new_tokens: int = 8) -> str:
        """Pre-selected frames as one video; idx are their indices on a grid of `fps` with n_total frames."""
        d = self.tmp / f"clip{len(self.log)}"
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for j, f in enumerate(frames):
            p = d / f"{j:04d}.jpg"
            Image.fromarray(f).save(p, quality=90)
            paths.append(p)
        self.max_new_tokens = max_new_tokens
        content = [{"type": "video", "video": [p.resolve().as_uri() for p in paths], "fps": 1.0},
                   {"type": "text", "text": prompt}]
        txt, ti, to, ms = self._chat([{"role": "user", "content": content}], (tuple(idx), fps, n_total))
        shutil.rmtree(d, ignore_errors=True)
        self._rec("clip", prompt, txt, len(frames), ti, to, ms)
        return txt

    def ask_video(self, video: Path, prompt: str, fps: float, max_new_tokens: int = 1024) -> str:
        """A video file sampled at `fps`."""
        video = Path(video)
        if not video.is_file():
            raise FileNotFoundError(video)
        self.max_new_tokens = max_new_tokens
        content = [{"type": "video", "video": video.resolve().as_uri(), "fps": fps, "max_pixels": VIDEO_MAX_PIXELS},
                   {"type": "text", "text": prompt}]
        txt, ti, to, ms = self._chat([{"role": "user", "content": content}])
        self._rec("video", prompt, txt, 0, ti, to, ms)
        return txt

    def ask_b64_frames(self, system: str, user: str, frames_b64: list[str], max_new_tokens: int) -> str:
        """TOC-Bench's own Qwen3-VL inference: system prompt, JPEG frames as images, user prompt."""
        proc, model = self.processor, self.model
        imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in frames_b64]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": [*[{"type": "image", "image": im} for im in imgs],
                                             {"type": "text", "text": user}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt", padding=True).to(model.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self._generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, pad_token_id=(proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id))
        txt = proc.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
        self._rec("benchmark_frames", user, txt, len(imgs), inputs["input_ids"].numel(),
                  out.shape[1] - inputs["input_ids"].shape[1], 1000 * (time.perf_counter() - t0))
        return txt

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
