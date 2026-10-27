"""SAM2.1 video predictor (Hugging Face Transformers)."""
from __future__ import annotations

import time

import numpy as np


class Tracker:
    """One call tracks several instances, prompted with boxes on one frame, forward and backward from it."""

    def __init__(self, path: str, device: str = "cuda"):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        self.torch = torch
        self.device = device
        self.proc = Sam2VideoProcessor.from_pretrained(path)
        self.model = Sam2VideoModel.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.calls, self.ms, self.frames = 0, 0.0, 0

    def track(self, frames: list[np.ndarray], t0: int, boxes: list[list[float]]) -> list[list[tuple]]:
        """tracks[i][t] = (mask area, box or None) for the instance prompted with boxes[i] on frame t0."""
        torch = self.torch
        self.calls += 1
        start = time.perf_counter()
        n = len(boxes)
        out = [[(0, None)] * len(frames) for _ in range(n)]
        ids = list(range(1, n + 1))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for reverse in (False, True):
                if reverse and t0 == 0:
                    continue
                s = self.proc.init_video_session(video=frames, inference_device=self.device,
                                                 inference_state_device="cpu", processing_device="cpu",
                                                 video_storage_device="cpu", dtype=torch.bfloat16)
                # The session consumes the id list it is given, so each pass gets a fresh one.
                self.proc.add_inputs_to_inference_session(s, frame_idx=t0, obj_ids=list(ids) if n > 1 else 1,
                                                          input_boxes=[boxes])
                self.model(s, frame_idx=t0)
                for o in self.model.propagate_in_video_iterator(s, start_frame_idx=t0, reverse=reverse):
                    masks = self.proc.post_process_masks([o.pred_masks], binarize=True,
                                                         original_sizes=[[s.video_height, s.video_width]])[0]
                    for slot, oid in enumerate(s.obj_ids):
                        nz = masks[slot, 0].nonzero()
                        if nz.shape[0] < 10:
                            out[int(oid) - 1][o.frame_idx] = (0, None)
                        else:
                            ys, xs = nz[:, 0].float(), nz[:, 1].float()
                            out[int(oid) - 1][o.frame_idx] = (int(nz.shape[0]), [float(xs.min()), float(ys.min()),
                                                                                  float(xs.max()) + 1,
                                                                                  float(ys.max()) + 1])
                del s
        torch.cuda.empty_cache()
        self.ms += 1000 * (time.perf_counter() - start)
        self.frames += len(frames) * len(boxes)
        return out
