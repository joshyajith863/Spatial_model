import pytest

from app.voice.rule_based_translator import parse


@pytest.mark.parametrize(
    "text,intent",
    [
        ("path from kitchen to hallway", "find_path"),
        ("How do I get from the bedroom to the garage?", "find_path"),
        ("area of the kitchen", "compute_area"),
        ("how big is bedroom 1", "compute_area"),
        ("egress from the bedroom", "check_egress"),
        ("evacuation route from kitchen", "check_egress"),
        ("what is the weather today", "unknown"),
    ],
)
def test_intents(text, intent):
    assert parse(text).intent == intent


def test_slots():
    q = parse("path from the kitchen to the hallway")
    assert (q.room_from, q.room_to) == ("kitchen", "hallway")
