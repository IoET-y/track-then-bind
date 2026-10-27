"""Direct answers of a Qwen3-VL model in vLLM, with each benchmark's prompt, frames and parser.

  python scripts/vllm_direct.py --model Qwen/Qwen3-VL-8B-Instruct --benches vet_cup vet_card pt toc --out runs/direct8b
  python scripts/vllm_direct.py ... --n 16 --out runs/sc8b                  # self-consistency, 16 samples
  python scripts/vllm_direct.py ... --fps 16 --max-pixels 288000 --no-resize --toc-frames 96 --out runs/more8b

VET-Bench and the Perception Test: the benchmark prompt on the video sampled by qwen-vl-utils at --fps.
TOC-Bench: the authors' template on --toc-frames uniform frames. Writes <out>/<bench>.jsonl."""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from track_then_bind import benchmarks  # noqa: E402
from track_then_bind.direct import vote  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benches", nargs="+", required=True, choices=benchmarks.NAMES)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=0, help="0: greedy; k: k samples and a majority vote")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--toc-frames", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=0, help="per-frame pixel limit (0: the qwen-vl-utils default)")
    ap.add_argument("--no-resize", action="store_true", help="keep the frame size chosen by qwen-vl-utils")
    ap.add_argument("--max-tokens", type=int, default=2048, help="VET-Bench and Perception Test")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.data:
        benchmarks.DATA = Path(a.data)

    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(a.model)
    # Without the processor's resize a single video can exceed vLLM's default per-step token budget.
    extra = dict(max_num_batched_tokens=a.max_model_len) if a.no_resize else {}
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, seed=0,
              limit_mm_per_prompt={"video": 1, "image": max(a.toc_frames, 1)}, gpu_memory_utilization=a.gpu_util,
              **extra)

    def params(max_tokens):
        if a.n > 0:   # sampling settings recommended for Qwen3-VL-Instruct
            return SamplingParams(n=a.n, temperature=0.7, top_p=0.8, top_k=20, max_tokens=max_tokens, seed=0)
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.benches:
        b = benchmarks.get(name, a.split)
        items = [i for i in b.items() if i.video is not None]
        if a.limit:
            items = items[: a.limit]
        path = out_dir / f"{name}.jsonl"
        done = {json.loads(line)["id"] for line in open(path)} if path.exists() else set()
        items = [i for i in items if i.id not in done]
        with path.open("a") as fh:
            if not done:
                fh.write(json.dumps({"id": "__config__", "args": vars(a), "bench": name}) + "\n")
            for s in range(0, len(items), a.batch):
                chunk, reqs, sps, ok = items[s:s + a.batch], [], [], []
                for it in chunk:
                    try:
                        if name == "toc":
                            system, user = b.er.build_prompt(it.raw)
                            imgs = [Image.open(io.BytesIO(base64.b64decode(x))).convert("RGB")
                                    for x in b.er.sample_frames_uniform(it.video, a.toc_frames)]
                            msgs = [{"role": "system", "content": system},
                                    {"role": "user", "content": [*[{"type": "image", "image": im} for im in imgs],
                                                                 {"type": "text", "text": user}]}]
                            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                            reqs.append({"prompt": prompt, "multi_modal_data": {"image": imgs}})
                            sps.append(params(b.er.get_max_tokens(it.raw["format"])))
                        else:
                            vel = {"type": "video", "video": Path(it.video).resolve().as_uri(), "fps": a.fps}
                            if a.max_pixels:
                                vel["max_pixels"] = a.max_pixels
                            msgs = [{"role": "user", "content": [vel, {"type": "text", "text": it.raw["prompt"]}]}]
                            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                            _, vids, kw = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True,
                                                              return_video_metadata=True)
                            if a.no_resize:
                                kw = dict(kw, do_resize=False)
                            reqs.append({"prompt": prompt, "multi_modal_data": {"video": vids},
                                         "mm_processor_kwargs": kw})
                            sps.append(params(a.max_tokens))
                        ok.append(it)
                    except Exception:
                        fh.write(json.dumps(dict(id=it.id, gold=it.gold, status="ERROR",
                                                 error=traceback.format_exc()[-1500:], correct=False)) + "\n")
                if not reqs:
                    continue
                for it, o in zip(ok, llm.generate(reqs, sps, use_tqdm=False)):
                    texts = [c.text for c in o.outputs]
                    if name == "toc":
                        preds = [b.parse(it, t) for t in texts]
                    else:
                        preds = [benchmarks.extract_answer(t) for t in texts]
                    rec = dict(id=it.id, gold=it.gold, status="OK", tok_in=len(o.prompt_token_ids),
                               tok_out=sum(len(c.token_ids) for c in o.outputs), raw=texts[0][-300:])
                    if a.n > 0:
                        votes = [vote(preds[: j + 1]) for j in range(len(preds))]
                        rec.update(samples=preds, votes=votes, correct_at=[bool(b.correct(it, v)) for v in votes],
                                   pred=votes[-1])
                    else:
                        rec.update(pred=preds[0])
                    rec["correct"] = bool(b.correct(it, rec["pred"]))
                    fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                print(f"[{name}] {s + len(chunk)}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
