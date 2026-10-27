"""Run Track, Then Bind (or a direct-answer baseline) on one benchmark split and write one JSON line per item.

  python -m track_then_bind.run --bench pt --split test --vlm Qwen/Qwen3-VL-8B-Instruct \
      --sam2 facebook/sam2.1-hiera-large --out runs/qwen8b/pt.jsonl

Runs resume: items already in --out are skipped."""
from __future__ import annotations

import argparse
import json
import traceback
import zlib
from pathlib import Path

from . import benchmarks
from .agent import load_plans, run
from .config import Config
from .direct import direct
from .vlm import load_vlm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=benchmarks.NAMES)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--vlm", required=True,
                    help='Qwen3-VL path or id, "openai:<model>", or "server:<served name>@<base url>"')
    ap.add_argument("--sam2", default="facebook/sam2.1-hiera-large", help="SAM2.1 path or id (agent mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="JSON file or string overriding the default configuration")
    ap.add_argument("--data", default=None, help="data root (default: $TTB_DATA or ./data)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default=None, help="comma-separated item ids")
    ap.add_argument("--with-direct", action="store_true", help="also record the direct answer for each item")
    ap.add_argument("--plans-from", default=None, help="glob of another run's outputs whose plans are reused")
    ap.add_argument("--dump", default=None, help="directory for the tracks of bound items")
    ap.add_argument("--device-map", default="cuda:0", help="device map of a Qwen3-VL model, e.g. auto")
    ap.add_argument("--reasoning-effort", default=None, help="OpenAI models (default: the API default)")
    ap.add_argument("--max-frames", type=int, default=128, help="server models: images per request")
    ap.add_argument("--box-order", default="xy", choices=["xy", "yx"], help="server models: box coordinate order")
    a = ap.parse_args()

    if a.data:
        benchmarks.DATA = Path(a.data)
    cfg = Config.load(a.config)
    bench = benchmarks.get(a.bench, a.split)
    items = bench.items()
    if a.ids:
        keep = set(a.ids.split(","))
        items = [i for i in items if i.id in keep]
    # Shard by video and keep a video's questions together, so that they share its boxes and tracks.
    items = sorted((i for i in items if zlib.crc32(str(i.video).encode()) % a.shards == a.shard),
                   key=lambda i: (str(i.video), i.id))
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    if a.dump:
        Path(a.dump).mkdir(parents=True, exist_ok=True)
    plans = load_plans(a.plans_from) if a.plans_from else None

    vlm = load_vlm(a.vlm, device_map=a.device_map, reasoning_effort=a.reasoning_effort, max_frames=a.max_frames,
                   box_order=a.box_order)
    tracker = None
    if cfg.mode == "agent":
        from .tracker import Tracker
        tracker = Tracker(a.sam2)
    with out.open("a") as fh:
        if not done:
            fh.write(json.dumps({"id": "__config__", "config": json.loads(cfg.dump()), "bench": a.bench,
                                 "split": a.split, "vlm": a.vlm}) + "\n")
        for it in items:
            if it.id in done:
                continue
            try:
                if cfg.mode == "direct":
                    rec = direct(vlm, bench, it, cfg)
                else:
                    rec = run(vlm, tracker, bench, it, cfg, with_direct=a.with_direct, dump=a.dump, plans=plans)
                rec["correct"] = bench.correct(it, rec.get("pred"))
                if "direct" in rec:
                    rec["direct_correct"] = bench.correct(it, rec["direct"])
            except Exception:
                rec = dict(id=it.id, gold=it.gold, status="ERROR", error=traceback.format_exc()[-1500:], correct=False)
            if vlm.log and "tok_reason" in vlm.log[0]:
                L = vlm.log
                rec["api"] = dict(model=sorted({x.get("model") for x in L} - {None}), calls=len(L),
                                  tok_in=sum(x["tok_in"] for x in L), tok_out=sum(x["tok_out"] for x in L),
                                  tok_reason=sum(x["tok_reason"] for x in L), images=sum(x["n_images"] for x in L),
                                  finish=[x["finish"] for x in L])
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
    if hasattr(vlm, "cleanup"):
        vlm.cleanup()


if __name__ == "__main__":
    main()
