"""OpenAI-compatible chat server for Gemma 4 in Transformers, for use with `--vlm server:<name>@<url>`.

  python scripts/gemma_server.py --model google/gemma-4-31b-it --port 8501

One request at a time, greedy decoding, thinking disabled. Images arrive as base64 data URLs. A single image uses
the processor's default budget. A request with several images (video frames) shares 4,480 visual tokens, the budget
of 64 frames at Gemma 4's video setting of 70 tokens per frame; each frame gets the largest supported budget
(70, 140, 280, 560 or 1,120) that keeps the request within it."""
import argparse
import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--name", default="gemma-4-31b-it")
a = ap.parse_args()
proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
IMAGE_DEFAULT = proc.image_processor.max_soft_tokens
BUDGETS, REQUEST_BUDGET = (70, 140, 280, 560, 1120), 64 * 70
LOCK = threading.Lock()


def convert(messages):
    out, n_img = [], 0
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": [{"type": "text", "text": c}]})
            continue
        parts = []
        for p in c:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image_url":
                url = p["image_url"]["url"]
                img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                parts.append({"type": "image", "image": img})
                n_img += 1
        out.append({"role": m["role"], "content": parts})
    return out, n_img


def answer_text(raw: str) -> str:
    raw = raw.split("<channel|>")[-1]
    raw = re.sub(r"<\|?[A-Za-z_]+\|?>|<eos>|<end_of_turn>", "", raw)
    return raw.strip()


def generate(body):
    msgs, n_img = convert(body["messages"])
    max_tokens = int(body.get("max_tokens") or 512)
    with LOCK:
        proc.image_processor.max_soft_tokens = (max(b for b in BUDGETS if n_img * b <= REQUEST_BUDGET)
                                                if n_img > 1 else IMAGE_DEFAULT)
        inputs = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                          add_generation_prompt=True, enable_thinking=False).to(model.device)
        n_in = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        raw = proc.decode(out[0][n_in:], skip_special_tokens=False)
    n_out = int(out.shape[-1] - n_in)
    return {"id": f"gemma-{time.time_ns()}", "object": "chat.completion", "model": a.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer_text(raw)},
                         "finish_reason": "length" if n_out >= max_tokens else "stop"}],
            "usage": {"prompt_tokens": int(n_in), "completion_tokens": n_out, "total_tokens": int(n_in) + n_out}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": a.name, "object": "model"}]})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, generate(body))
        except Exception as e:
            self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"[:500]}})


if __name__ == "__main__":
    print(f"serving {a.name} on port {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
