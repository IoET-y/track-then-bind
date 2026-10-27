"""CPU-only tests of the parts that do not call a model.  Run with: python -m pytest tests"""
from track_then_bind import evidence, plan
from track_then_bind.bind import clip_indices
from track_then_bind.config import Config
from track_then_bind.direct import vote
from track_then_bind.vlm.openai_api import to_rel1000
from track_then_bind.vlm.openai_compatible import yx_to_xy

cfg = Config()
W, H = 640, 480


def ball_track():
    """30 frames at 6 fps: visible and moving right, hidden, visible again, then leaving at the right edge."""
    tr = []
    for t in range(30):
        if t < 8:
            tr.append((900, [100 + 10 * t, 200, 140 + 10 * t, 240]))
        elif t < 14:
            tr.append((0, None))
        elif t < 22:
            tr.append((300 if t < 16 else 900, [300 + 20 * (t - 14), 200, 340 + 20 * (t - 14), 240]))
        elif t < 24:
            tr.append((900, [600, 200, 639, 240]))
        else:
            tr.append((0, None))
    return tr


def cup_tracks():
    return [[(1, [10, 0, 50, 40]) for _ in range(10)],
            [(1, [200, 0, 240, 40]) for _ in range(10)],
            [(1, [100, 0, 140, 40]) for _ in range(9)] + [(0, None)]]


def test_timeline_text():
    assert evidence.timeline_text("red ball", ball_track(), 6.0, H, W, cfg) == (
        "'red ball': 0.0-1.3 s fully visible; 1.3-2.7 s not visible (vanished inside the frame, so it was blocked or "
        "hidden); 2.7-4.0 s fully visible; 4.0-5.0 s not visible (vanished at the frame edge, so it left the frame) "
        "until the end. Times it went fully out of view after being seen: 2; times partly hidden: 0. Events in order: "
        "1.3 s gets fully hidden; 2.7 s comes back into view; 4.0 s leaves the frame and disappears for good. "
        "Overall motion: moves from left to right.")


def test_moments_and_order():
    assert evidence.moments(ball_track(), cup_tracks(), 6.0, H, W, "contained", cfg) == [
        dict(kind="contained", start=7, end=8, carrier_by_overlap=None)]
    assert evidence.final_order(cup_tracks()) == (8, [1, 3, 2])


def test_binding_facts():
    assert evidence.binding_text("ball", "cups", 3, 1.25, "contained", cup_tracks(), 4.0) == (
        "The ball was hidden under cup #3 (seen at 1.2 s). The tracker followed every cup through the rest of the "
        "video. In the last frame where all 3 are tracked (2.0 s), from the camera's left to right they are: "
        "#1, #3, #2. So the ball is under the second cup from the camera's left.")
    assert evidence.binding_text("Queen of Hearts", "playing cards", 2, None, "visible", cup_tracks(), 4.0) == (
        "The Queen of Hearts is playing card #2. The tracker followed every playing card through the rest of the "
        "video. In the last frame where all 3 are tracked (2.0 s), from the camera's left to right they are: "
        "#1, #3, #2. So the Queen of Hearts is the third playing card from the camera's left.")
    assert evidence.person_rank_fact("ball", "cups", "contained", 0, 3, True) == (
        "From the person's point of view, the ball is under the third cup from the left.")


def test_plan_parse():
    p = plan.parse('```json\n{"entities": [{"name": "the three cups", "role": "carrier"}, {"name": "ball", '
                   '"role": "target"}], "identity_cue": "contained", "asks_final_location": true}\n```')
    assert p["parse_ok"] and p["identity_cue"] == "contained"
    assert [e["name"] for e in p["entities"]] == ["cups", "ball"]
    p = plan.parse("I think the cups")
    assert not p["parse_ok"] and p["entities"] == []


def test_clip_indices():
    assert clip_indices(120, 30.0, cfg, 60, 60) == list(range(15, 106, 8))
    assert len(clip_indices(3000, 30.0, cfg)) == cfg.bind_max_frames


def test_box_conversions():
    assert to_rel1000('[{"bbox_2d": [64, 48, 320, 240], "label": "cup"}]', 640, 480) == \
        '[{"bbox_2d": [100, 100, 500, 500], "label": "cup"}]'
    assert yx_to_xy('{"bbox_2d": [1, 2, 3, 4]}') == '{"bbox_2d": [2, 1, 4, 3]}'


def test_vote():
    assert vote(["B", "A", "B", None]) == "B"
    assert vote(["A", "B"]) == "A"
    assert vote([["B", "A"], ["B", "A"], ["A", "B"]]) == ["B", "A"]
    assert vote([None]) is None
