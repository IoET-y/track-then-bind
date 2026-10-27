"""Benchmarks: items, videos, and each benchmark's own prompt, frames and answer parser.
The agent sees only `question_for_plan`. Direct answers and the agent's read-out use the same prompt and parser.

Scorers are imported from the benchmarks' official repositories (see README, "Data"):
  VET-Bench / Perception Test: extract_answer from <data>/shellgame/compute_accuracy.py
  TOC-Bench: prompt template, frame sampler and answer extractor from <data>/toc_bench/eval_runner.py"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import video

DATA = Path(os.environ.get("TTB_DATA", "data"))
LETTERS = "ABCDEF"

EVIDENCE_HEAD = "Automatic tracker evidence (may contain errors; the video frames take precedence):\n"
FACTS_HEAD = ("Facts established by tracking every object through the whole video (identity was fixed when the target "
              "could be seen, and kept by the tracker afterwards):\n")
FACTS_TAIL = "\nAnswer the question using these facts."
RESOLVE_TAIL = "\n\nThe facts above are reliable. Reply with only the final answer in \\boxed{X}."


def _import_from(repo: Path, module: str):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return __import__(module)


def extract_answer(text: str):
    """VET-Bench's multiple-choice parser (\\boxed{X} and common answer phrases)."""
    return _import_from(DATA / "shellgame", "compute_accuracy").extract_answer(text)


@dataclass
class Item:
    id: str
    video: Path | None
    question_for_plan: str
    gold: object
    raw: dict


def pt_prompt(question, options):
    """The Perception Test cup-game prompt of VET-Bench."""
    p = question + "\n\n"
    for i, o in enumerate(options):
        p += f"{LETTERS[i]}. {o}\n"
    return p + ("\nPlease select the correct answer from the options above. "
                "Put your final answer in \\boxed{X} where X is the letter of your choice.")


class MCVideo:
    """VET-Bench and Perception Test: question and options on the video at readout_fps, answer in \\boxed{}."""
    supports_overlay = True

    def __init__(self, name, items_path, video_dir):
        self.name = name
        self.items_path, self.video_dir = Path(items_path), Path(video_dir)

    def items(self):
        out = []
        for it in json.load(open(self.items_path)):
            if "options" in it:   # Perception Test
                q = pt_prompt(it["question"], it["options"])
                out.append(Item(it.get("uid", it["video_id"]), self.video_dir / f"{it['video_id']}.mp4", q,
                                LETTERS[int(it["answer"])], dict(it, prompt=q)))
            else:                 # VET-Bench: the question already lists the options and the output format
                out.append(Item(it["video"][:-4], self.video_dir / it["video"], it["question"], it["answer"],
                                dict(it, prompt=it["question"])))
        return out

    def extract_mc(self, txt):
        return extract_answer(txt)

    def parse(self, item, txt):
        return extract_answer(txt)

    def answer(self, vlm, item, evidence: str, cfg):
        prompt = item.raw["prompt"]
        if evidence:
            prompt = prompt + "\n\n" + EVIDENCE_HEAD + evidence
        txt = vlm.ask_video(item.video, prompt, fps=cfg.readout_fps, max_new_tokens=1024)
        return extract_answer(txt), txt

    def resolve(self, vlm, item, facts: str):
        """Text-only read-out of a bound target: the tracked facts, then the benchmark prompt."""
        txt = vlm.ask([], FACTS_HEAD + facts + FACTS_TAIL + "\n\n" + item.raw["prompt"] + RESOLVE_TAIL,
                      max_new_tokens=512)
        return extract_answer(txt), txt

    def correct(self, item, pred):
        return pred == item.gold


class TOC:
    """TOC-Bench: the authors' prompt template, 32 uniform frames and answer extractor. Tracker evidence is appended
    to the question."""
    supports_overlay = False

    def __init__(self, items_path, videos_root):
        self.er = _import_from(DATA / "toc_bench", "eval_runner")
        self.items_path, self.videos_root = Path(items_path), Path(videos_root)
        self.name = "toc"

    def items(self):
        out = []
        for it in json.load(open(self.items_path))["items"]:
            vp, _ = self.er.resolve_video_path(it["video_id"], self.videos_root)
            text = it["question"]
            for k in ("statement_A", "statement_B"):
                if it.get(k):
                    text += "\n" + it[k]
            for e in it.get("events", []) or []:
                text += "\n" + e["event_text"]
            gold = it.get("correct_order") if it["format"].startswith("ordering") else it.get("correct_answer")
            out.append(Item(it["qa_id"], Path(vp) if vp else None, text, gold, it))
        return out

    def parse(self, item, txt):
        return self.er.extract(re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip(), item.raw["format"])

    def answer(self, vlm, item, evidence: str, cfg):
        it = dict(item.raw)
        if evidence:
            it["question"] = it["question"] + "\n\n" + EVIDENCE_HEAD + evidence
        system, user = self.er.build_prompt(it)
        frames = self.er.sample_frames_uniform(item.video, cfg.toc_frames)
        txt = vlm.ask_b64_frames(system, user, frames, self.er.get_max_tokens(it["format"]))
        return self.er.extract(txt, it["format"]), txt

    def resolve(self, vlm, item, facts: str):
        system, user = self.er.build_prompt(item.raw)
        txt = vlm.ask([], system + "\n\n" + FACTS_HEAD + facts + FACTS_TAIL + "\n\n" + user,
                      max_new_tokens=self.er.get_max_tokens(item.raw["format"]))
        return self.er.extract(txt, item.raw["format"]), txt

    def correct(self, item, pred):
        if pred is None:
            return False
        if item.raw["format"].startswith("ordering"):
            return list(pred) == list(item.gold)
        return str(pred) == str(item.gold)


class VideoMME:
    """Video-MME (short and medium videos): the standard multiple-choice prompt on at most 64 frames (2 fps or
    fewer); the answer is the first option letter of the reply."""
    supports_overlay = True
    PROMPT = ("Select the best answer to the following multiple-choice question based on the video. Respond with only "
              "the letter (A, B, C, or D) of the correct option.\n{question}\n{options}\nThe best answer is:")

    def __init__(self, items_path, video_dir):
        self.name = "videomme"
        self.items_path, self.video_dir = Path(items_path), Path(video_dir)

    def items(self):
        out = []
        for it in json.load(open(self.items_path)):
            q = self.PROMPT.format(question=it["question"], options="\n".join(it["options"]))
            out.append(Item(it["uid"], self.video_dir / f"{it['video_id']}.mp4", q, it["answer"], dict(it, prompt=q)))
        return out

    @staticmethod
    def letter(txt):
        m = re.search(r"\b([ABCD])\b", txt.strip())
        return m.group(1) if m else None

    def extract_mc(self, txt):
        return extract_answer(txt)

    def parse(self, item, txt):
        return self.letter(txt)

    def answer(self, vlm, item, evidence: str, cfg):
        fps, n = video.probe(item.video)
        prompt = item.raw["prompt"]
        if evidence:
            tail = "\nThe best answer is:"
            prompt = prompt.replace(tail, "\n\n" + EVIDENCE_HEAD + evidence + tail)
        txt = vlm.ask_video(item.video, prompt, fps=min(2.0, 64.0 / max(n / fps, 1e-6)), max_new_tokens=16)
        return self.letter(txt), txt

    def resolve(self, vlm, item, facts: str):
        txt = vlm.ask([], FACTS_HEAD + facts + FACTS_TAIL + "\n\n" + item.raw["prompt"], max_new_tokens=16)
        return self.letter(txt), txt

    def correct(self, item, pred):
        return pred == item.gold


NAMES = ["vet_cup", "vet_card", "pt", "toc", "pt_val", "videomme"]


def get(name: str, split: str):
    """Benchmark `name` on split "dev" or "test". Item files under <data>/items are written by
    scripts/prepare_data.py; video locations follow the layout in the README."""
    items = DATA / "items"
    if name == "vet_cup":
        if split != "test":
            raise ValueError("vet_cup has only a test split")
        return MCVideo(name, DATA / "vetbench/cup/QA.json", DATA / "vetbench/cup")
    if name == "vet_card":
        root = DATA / ("vetbench/card" if split == "test" else "card_dev")
        return MCVideo(name, root / "QA.json", root)
    if name == "pt":
        if split == "test":
            return MCVideo(name, items / "pt_test.json", DATA / "shellgame/perception_test/cup_games_filtered")
        return MCVideo(name, items / "pt_dev.json", DATA / "perception_test/videos")
    if name == "toc":
        return TOC(items / f"toc_{split}.json", DATA / "toc_data/videos")
    if name == "pt_val":
        if split != "test":
            raise ValueError("pt_val has only a test split")
        return MCVideo(name, items / "pt_val_sample.json", DATA / "perception_test/videos")
    if name == "videomme":
        if split != "test":
            raise ValueError("videomme has only a test split")
        return VideoMME(items / "videomme_short_medium.json", DATA / "videomme/videos")
    raise KeyError(name)
