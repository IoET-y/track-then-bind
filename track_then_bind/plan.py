"""Step 1, plan: the VLM fills a fixed schema that decides which entities are tracked, how the target can be
recognised, and whether the question asks for the target's final location."""
from __future__ import annotations

import json
import re

PLAN_PROMPT = """You plan how to answer a question about a video.
Question:
{question}

Reply with one JSON object and nothing else:
{{"entities": [{{"name": "<short noun phrase>", "role": "target" | "carrier" | "subject"}}],
  "identity_cue": "visible" | "contained" | "none",
  "category": "<the kind of object the target is, in plural, e.g. playing cards>",
  "asks_final_location": true | false,
  "viewpoint": "camera" | "person"}}

Rules:
- "target": the specific object whose identity or location the question asks about.
- "carrier": objects that move, hide or cover the target, or look the same as it (for example the cups in a shell game). Name them in plural.
- "subject": any other object or person whose visibility, events, state or motion the question asks about.
- "identity_cue": "visible" if the target can be told apart by its look in some frames (for example a face-up card); "contained" if it is hidden inside or under a carrier; "none" if no specific object identity must be followed.
- "asks_final_location": true only if the question asks where the target is at the end of the video (for example which cup, or left / middle / right).
- "viewpoint": "person" if the question asks about left or right from a person's point of view; otherwise "camera".
- Use names as they appear in the question or the video. At most 3 entities."""

FINAL_PROMPT = ("Question about a video:\n{question}\n\nDoes this question ask where a particular object is "
                "located at the end of the video (for example, which container it is in, or whether it is on the left, "
                "in the middle or on the right)? Reply YES or NO.")

NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|several|multiple|many)"


def clean(name: str) -> str:
    """Drop leading articles and counts: 'the three white cups' -> 'white cups'."""
    n = re.sub(r"^(?:the|a|an)\s+", "", name.strip(), flags=re.I)
    n = re.sub(rf"^{NUM}\s+", "", n, flags=re.I)
    return n.strip()


def parse(text: str) -> dict:
    """The plan in a reply; an empty plan if the reply holds no valid JSON object."""
    m = re.search(r"\{.*\}", text, re.S)
    plan = {"entities": [], "identity_cue": "none", "category": "", "asks_final_location": False,
            "viewpoint": "camera", "parse_ok": False}
    if not m:
        return plan
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return plan
    ents = []
    for e in d.get("entities", [])[:3]:
        if isinstance(e, dict) and str(e.get("name", "")).strip():
            role = e.get("role") if e.get("role") in ("target", "carrier", "subject") else "subject"
            name = clean(str(e["name"]))
            if name and all(x["name"] != name for x in ents):
                ents.append({"name": name, "role": role})
    cue = d.get("identity_cue") if d.get("identity_cue") in ("visible", "contained", "none") else "none"
    cat = clean(str(d.get("category") or ""))
    final = d.get("asks_final_location") in (True, "true", "True")
    view = "person" if d.get("viewpoint") == "person" else "camera"
    return {"entities": ents, "identity_cue": cue, "category": cat, "asks_final_location": final, "viewpoint": view,
            "parse_ok": True}


def make_plan(vlm, video_path, question: str, cfg) -> dict:
    reply = vlm.ask_video(video_path, PLAN_PROMPT.format(question=question), fps=cfg.plan_fps, max_new_tokens=200)
    p = parse(reply)
    p["raw"] = reply
    # Whether the question asks for a final location selects the read-out, so it is asked separately, in text only.
    fin = vlm.ask([], FINAL_PROMPT.format(question=question), max_new_tokens=4)
    p["asks_final_location_plan"] = p["asks_final_location"]
    p["asks_final_location"] = fin.strip().upper().startswith("Y")
    # A target recognised by its look must be told apart from objects of its own kind: without a named carrier,
    # all objects of the target's category become the carriers.
    ents = p["entities"]
    if p["asks_final_location"] and p["identity_cue"] == "visible" and p["category"] \
            and any(e["role"] == "target" for e in ents) and not any(e["role"] == "carrier" for e in ents):
        ents.append({"name": p["category"], "role": "carrier"})
    return p
