"""The single configuration shared by all benchmarks."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # Plan
    plan_fps: float = 2.0              # frame rate at which the planner watches the video

    # Perceive
    ground_k: int = 8                  # frames per grounding round
    max_side: int = 960                # long side of the frames used for tracking (px)
    base_fps: float = 6.0              # rate of the first tracking pass
    max_base_frames: int = 900         # frame limit of the first pass on long videos
    max_track_frames: int = 300        # target frame count of the dense pass
    keep_until: float = 0.9            # dense pass if a carrier track ends before this fraction of the video,
    crowd_iou: float = 0.05            # if carrier boxes overlap by more than this IoU,
    speed_ratio: float = 0.25          # or if a box moves more than this many box widths between sampled frames
    init_rule: str = "earliest_modal"  # prompt the tracker on the earliest frame with the modal instance count

    # Visibility timeline
    full: float = 0.6                  # mask area >= full * 90th-percentile area: fully visible
    partial: float = 0.12              # mask area >= partial * 90th-percentile area: partly hidden
    edge: float = 0.02                 # a last box within this margin of the border: left the frame
    min_seg_s: float = 0.5             # visibility segments shorter than this are merged (s)

    # Bind
    bind_fps: float = 4.0
    bind_window_s: float = 3.0         # length of the clip around a located evidence moment (s)
    bind_max_frames: int = 96
    bind_window: str = "moment"        # "moment" | "whole" (ablation: always bind on the whole video)

    # Read-out
    readout: str = "resolve"           # "resolve": text-only answer from the tracked facts
    #                                    "evidence" (ablation): the facts are added to the prompt with the video
    #                                    "overlay" (ablation): no bind question; numbered tracks + benchmark question
    readout_fps: float = 8.0           # video rate of direct answers on VET-Bench and the Perception Test
    toc_frames: int = 32               # TOC-Bench's uniform frame count

    # Direct-answer baselines
    mode: str = "agent"                # "agent" | "direct"
    sc_k: int = 0                      # 0: one greedy answer; k > 0: k samples and a majority vote
    sc_temperature: float = 0.7        # sampling settings recommended for Qwen3-VL-Instruct
    sc_top_p: float = 0.8
    sc_top_k: int = 20
    sc_chunk: int = 4                  # samples drawn per generate() call

    def dump(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def load(cls, overrides: str | None) -> "Config":
        """Defaults, updated from a JSON file or a JSON string."""
        c = cls()
        if overrides:
            d = json.load(open(overrides)) if overrides.endswith(".json") else json.loads(overrides)
            for k, v in d.items():
                if not hasattr(c, k):
                    raise KeyError(f"unknown config key {k}")
                setattr(c, k, v)
        return c
