"""Box utilities and the numbered-track overlay shown to the VLM."""
from __future__ import annotations

import cv2
import numpy as np

COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]


def draw_numbers(frame: np.ndarray, boxes: dict[int, list[float] | None]) -> np.ndarray:
    """Draw each track's box and number on a copy of the frame."""
    img = frame.copy()
    scale = max(1.0, img.shape[1] / 640)
    for k, b in boxes.items():
        if b is None:
            continue
        c = COLORS[(k - 1) % len(COLORS)]
        x0, y0, x1, y1 = (int(v) for v in b)
        cv2.rectangle(img, (x0, y0), (x1, y1), c, int(3 * scale))
        cv2.putText(img, str(k), (x0 + 4, max(0, y0) + int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, c,
                    int(3 * scale))
    return img


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def dedupe(boxes, thr=0.5):
    """Drop boxes that overlap an earlier box by IoU >= thr."""
    kept = []
    for b in boxes:
        if all(iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept
