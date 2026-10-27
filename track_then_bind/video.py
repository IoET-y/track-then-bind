"""Frame reading: stride sampling to a target rate and resizing to a maximum long side."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def probe(path: Path) -> tuple[float, int]:
    """Native frame rate and frame count."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def read(path: Path, fps_out: float, max_side: int) -> tuple[list[np.ndarray], float, float]:
    """Every k-th frame, k = round(native fps / fps_out), as RGB arrays.
    Returns (frames, effective fps, native fps)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / fps_out)))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            s = min(1.0, max_side / max(h, w))
            if s < 1:
                f = cv2.resize(f, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames, fps / step, fps


def uniform(n_frames: int, k: int) -> list[int]:
    """k evenly spaced indices in [0, n_frames)."""
    return sorted(set(np.linspace(0, n_frames - 1, k).astype(int).tolist()))
