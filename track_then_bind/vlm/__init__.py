"""VLM backends. Each exposes ask, ask_clip, ask_video and ask_b64_frames, and logs every call in `log`."""
from __future__ import annotations


def load_vlm(spec: str, device_map: str = "cuda:0", reasoning_effort: str | None = None, max_frames: int = 128,
             box_order: str = "xy"):
    """spec: a Qwen3-VL path or Hugging Face id; "openai:<model>"; or "server:<served name>@<base url>"."""
    if spec.startswith("openai:"):
        from .openai_api import OpenAIVLM
        return OpenAIVLM(spec[len("openai:"):], reasoning_effort=reasoning_effort)
    if spec.startswith("server:"):
        from .openai_compatible import ServerVLM
        name, url = spec[len("server:"):].split("@", 1)
        return ServerVLM(name, url, max_frames=max_frames, box_order=box_order)
    from .qwen import QwenVLM
    return QwenVLM(spec, device_map=device_map)
