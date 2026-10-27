"""Write the item files under <data>/items from the official benchmark releases and the lists in splits/.

  python scripts/prepare_data.py --data data

Sources (see README, "Data"):
  pt_test.json                <data>/shellgame/perception_test/cup_games_filtered_metadata.json (VET-Bench)
  pt_dev.json, pt_val_sample  <data>/perception_test/mc_question_val.parquet (Perception Test validation)
  toc_dev.json, toc_test.json <data>/toc_bench/answer_final.json (TOC-Bench)
  videomme_short_medium.json  <data>/videomme/test.parquet (Video-MME)
Missing sources are skipped."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = Path(__file__).resolve().parent.parent / "splits"
CUP_GAME = "The person uses multiple similar objects to play an occlusion game. Where is the hidden object at the end"


def ids(name):
    return [x for x in (SPLITS / name).read_text().split("\n") if x]


def write(path: Path, obj, n):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    print(f"{path.name}: {n} items")


def pt_test(data: Path, out: Path):
    meta = json.load(open(data / "shellgame/perception_test/cup_games_filtered_metadata.json"))
    items = []
    for vid, m in meta.items():
        q = m["mc_question"][0]
        items.append(dict(video_id=vid, question=q["question"], options=q["options"], answer=int(q["answer"])))
    write(out / "pt_test.json", items, len(items))


def perception_test_val(data: Path, out: Path):
    import pyarrow.parquet as pq
    rows = pq.read_table(data / "perception_test/mc_question_val.parquet").to_pylist()

    def item(r):
        return dict(video_id=r["video_name"], question=r["question"], options=list(r["options"]),
                    answer=int(r["answer_id"]))

    cup = {}
    for r in rows:
        if r["question"].startswith(CUP_GAME):
            cup.setdefault(r["video_name"], r)
    dev = [item(cup[v]) for v in ids("pt_dev_videos.txt")]
    write(out / "pt_dev.json", dev, len(dev))

    by_uid = {f"{r['video_name']}#q{r['question_id']}": r for r in rows}
    sample = []
    for uid in ids("pt_val_sample_uids.txt"):
        r = by_uid[uid]
        sample.append(dict(item(r), question_id=r["question_id"], area=r["area"], reasoning=r["reasoning"],
                           tag=list(r["tag"]), uid=uid))
    write(out / "pt_val_sample.json", sample, len(sample))


def toc(data: Path, out: Path):
    src = json.load(open(data / "toc_bench/answer_final.json"))
    by_id = {it["qa_id"]: it for it in src["items"]}
    head = {k: v for k, v in src.items() if k != "items"}
    for split in ("dev", "test"):
        items = [by_id[q] for q in ids(f"toc_{split}_qa_ids.txt")]
        write(out / f"toc_{split}.json", dict(head, items=items), len(items))


def videomme(data: Path, out: Path):
    import pyarrow.parquet as pq
    rows = {r["question_id"]: r for r in pq.read_table(data / "videomme/test.parquet").to_pylist()}
    items = []
    for uid in ids("videomme_short_medium_uids.txt"):
        r = rows[uid]
        items.append(dict(uid=uid, video_id=r["videoID"], vid_num=r["video_id"], duration=r["duration"],
                          domain=r["domain"], sub_category=r["sub_category"], task_type=r["task_type"],
                          question=r["question"], options=list(r["options"]), answer=r["answer"]))
    write(out / "videomme_short_medium.json", items, len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    data = Path(a.data)
    out = data / "items"
    out.mkdir(parents=True, exist_ok=True)
    for build, need in ((pt_test, "shellgame/perception_test/cup_games_filtered_metadata.json"),
                        (perception_test_val, "perception_test/mc_question_val.parquet"),
                        (toc, "toc_bench/answer_final.json"),
                        (videomme, "videomme/test.parquet")):
        if (data / need).exists():
            build(data, out)
        else:
            print(f"skipped {build.__name__}: {data / need} not found")


if __name__ == "__main__":
    main()
