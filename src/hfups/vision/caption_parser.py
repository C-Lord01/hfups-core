"""Parse raw BLIP caption strings into structured disaster-relevant semantic fields."""

from __future__ import annotations

HAZARD_KEYWORDS = [
    "flood", "water", "fire", "smoke", "debris", "damage",
    "collapsed", "submerged", "burning", "destroyed", "wreckage",
]

ACTION_KEYWORDS = [
    "waving", "running", "standing", "sitting", "lying",
    "trapped", "injured", "walking", "climbing", "swimming",
]

ENVIRONMENT_KEYWORDS = {
    "urban": ["street", "road", "building", "city", "bridge"],
    "rural": ["field", "farm", "dirt road", "forest", "hill"],
    "indoor": ["room", "hallway", "floor", "ceiling", "wall"],
}


def parse_caption(caption: str) -> dict:
    """
    Extract disaster-relevant semantic fields from a BLIP caption string.
    Returns a dict with keys:
      - scene_hint: str | None   (e.g. "flooded street", "collapsed building")
      - hazards: list[str]       (e.g. ["flood", "fire", "smoke", "debris"])
      - actions: list[str]       (e.g. ["waving", "running", "trapped"])
      - environment: str | None  (e.g. "urban", "rural", "indoor")
    Use simple keyword matching against these lists.
    No ML — pure string matching. Fast and deterministic.
    """
    lower = caption.lower()

    hazards = [kw for kw in HAZARD_KEYWORDS if kw in lower]
    actions = [kw for kw in ACTION_KEYWORDS if kw in lower]

    environment: str | None = None
    for env_name, keywords in ENVIRONMENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            environment = env_name
            break

    scene_hint: str | None = None
    if hazards:
        hint_parts = []
        if environment:
            hint_parts.append(environment)
        hint_parts.append(hazards[0])
        scene_hint = " ".join(hint_parts) if hint_parts else hazards[0]

    return {
        "scene_hint": scene_hint,
        "hazards": hazards,
        "actions": actions,
        "environment": environment,
    }
