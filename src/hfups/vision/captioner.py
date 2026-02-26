from __future__ import annotations

from dataclasses import dataclass

from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket
from hfups.vision.openimages_dict import OpenImagesDict
from hfups.vision.prompt_utils import grid_to_bucket_phrase, grid_to_phrase

_MODE_LIMITS = {
    "short": 80,
    "medium": 120,
    "long": 160,
}

_HAZARD_TERMS = (
    "fire",
    "smoke",
    "flood",
    "flooded",
    "water",
    "explosion",
    "hazard",
)
_PERSON_TERMS = ("person", "people", "man", "woman", "child")
_VEHICLE_TERMS = ("car", "truck", "bus", "van", "motorcycle", "bicycle", "sedan", "vehicle")
_ANIMAL_TERMS = ("dog", "cat", "bird", "horse", "animal")


@dataclass(frozen=True)
class _CaptionObject:
    obj: KeyframeObject
    name: str
    name_lower: str


def _category_rank(name_lower: str) -> int:
    if any(term in name_lower for term in _HAZARD_TERMS):
        return 0
    if any(term in name_lower for term in _PERSON_TERMS):
        return 1
    if any(term in name_lower for term in _VEHICLE_TERMS):
        return 2
    if any(term in name_lower for term in _ANIMAL_TERMS):
        return 3
    return 4


def _sorted_caption_objects(packet: KeyframePacket, openimages_dict: OpenImagesDict) -> list[_CaptionObject]:
    items: list[_CaptionObject] = []
    for obj in packet.objects:
        cls = openimages_dict.by_id.get(obj.class_id)
        name = cls.name if cls is not None else f"class {obj.class_id}"
        items.append(_CaptionObject(obj=obj, name=name, name_lower=name.lower()))

    return sorted(
        items,
        key=lambda item: (
            -item.obj.confidence,
            -item.obj.size,
            item.obj.class_id,
            item.obj.track_id,
            item.obj.cx,
            item.obj.cy,
        ),
    )


def _ensure_person_included(
    selected: list[_CaptionObject],
    ranked_all: list[_CaptionObject],
) -> list[_CaptionObject]:
    people = [item for item in ranked_all if any(t in item.name_lower for t in _PERSON_TERMS)]
    if not people:
        return selected
    if any(any(t in item.name_lower for t in _PERSON_TERMS) for item in selected):
        return selected

    person = people[0]
    if len(selected) < 3:
        return selected + [person]
    return selected[:2] + [person]


def _build_clause(item: _CaptionObject, mode: str) -> str:
    if mode == "short":
        return f"{item.name_lower} {grid_to_bucket_phrase(item.obj.cx, item.obj.cy)}"
    if mode == "medium":
        return f"{item.name_lower} in the {grid_to_bucket_phrase(item.obj.cx, item.obj.cy)}"
    return f"{item.name_lower} in the {grid_to_phrase(item.obj.cx, item.obj.cy)}"


def _fit_to_limit(candidates: list[str], limit: int, prefix: str = "") -> str:
    parts = candidates[:]
    while parts:
        text = (prefix + "; ".join(parts)).strip()
        if len(text.encode("utf-8")) <= limit:
            return text
        parts = parts[:-1]

    if prefix and len(prefix.encode("utf-8")) <= limit:
        return prefix.strip()

    raw = " ".join(candidates)
    encoded = raw.encode("utf-8")
    if len(encoded) <= limit:
        return raw
    truncated = encoded[:limit]
    while True:
        try:
            return truncated.decode("utf-8").rstrip(" ,;:.")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
            if not truncated:
                return ""


def generate_caption(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    mode: str = "medium",
) -> str:
    """
    Deterministically generate a concise Layer-B caption from a keyframe packet.

    The caption is capped by mode:
    short <= 80 bytes, medium <= 120 bytes, long <= 160 bytes.
    """
    if mode not in _MODE_LIMITS:
        raise ValueError("mode must be one of: short, medium, long")

    if not packet.objects:
        return "no significant objects detected"

    ranked = _sorted_caption_objects(packet, openimages_dict)
    selected = ranked[:3]
    selected = _ensure_person_included(selected, ranked)
    selected = sorted(
        selected,
        key=lambda item: (
            _category_rank(item.name_lower),
            -item.obj.confidence,
            -item.obj.size,
            item.obj.class_id,
            item.obj.track_id,
        ),
    )

    clauses = [_build_clause(item, mode) for item in selected]
    limit = _MODE_LIMITS[mode]

    if mode == "short":
        text = _fit_to_limit(clauses, limit)
    elif mode == "medium":
        text = _fit_to_limit(clauses, limit, prefix="A scene with ")
    else:
        text = _fit_to_limit(clauses, limit, prefix="An incident scene with ")

    text = text.strip().rstrip(";,.")
    if not text:
        text = "no significant objects detected"
    return text
