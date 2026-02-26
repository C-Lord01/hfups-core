from __future__ import annotations

from typing import Literal

from hfups.vision.captioner import generate_caption
from hfups.vision.keyframe_packet import KeyframePacket
from hfups.vision.openimages_dict import OpenImagesDict
from hfups.vision.prompt_utils import conf_to_pct, dxdy_to_direction, grid_to_phrase, size_to_word

TemplateStyle = Literal["concise", "descriptive", "disaster_response", "cinematic"]

_HAZARD_TERMS = (
    "fire",
    "smoke",
    "flood",
    "flooded",
    "water",
    "explosion",
    "landslide",
    "hazard",
)

_VEHICLE_TERMS = ("car", "truck", "bus", "van", "sedan", "vehicle", "motorcycle", "bicycle")
_PERSON_TERMS = ("person", "people", "man", "woman", "child")


def _is_hazard(name: str) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in _HAZARD_TERMS)


def _is_vehicle(name: str) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in _VEHICLE_TERMS)


def _is_person(name: str) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in _PERSON_TERMS)


def _clip_words(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:.") + "."


def _objects_for_prompt(packet: KeyframePacket, openimages_dict: OpenImagesDict) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for obj in packet.objects:
        cls = openimages_dict.by_id.get(obj.class_id)
        name = cls.name if cls is not None else f"class {obj.class_id}"
        items.append(
            {
                "obj": obj,
                "name": name,
                "hazard": _is_hazard(name),
                "vehicle": _is_vehicle(name),
                "person": _is_person(name),
            }
        )
    return sorted(
        items,
        key=lambda item: (
            -int(item["hazard"]),  # type: ignore[arg-type]
            -int(item["person"]),  # type: ignore[arg-type]
            -int(item["vehicle"]),  # type: ignore[arg-type]
            -item["obj"].confidence,  # type: ignore[index]
            -item["obj"].size,  # type: ignore[index]
            item["obj"].class_id,  # type: ignore[index]
            item["obj"].track_id,  # type: ignore[index]
        ),
    )


def _motion_sentence(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    deltas: list[tuple] | None,
) -> str | None:
    if not deltas:
        return None

    by_track = {obj.track_id: obj for obj in packet.objects}
    phrases: list[str] = []
    for raw in deltas:
        try:
            track_id = int(raw[0])
            dx = int(raw[1])
            dy = int(raw[2])
        except (IndexError, TypeError, ValueError):
            continue
        obj = by_track.get(track_id)
        if obj is None:
            continue
        cls = openimages_dict.by_id.get(obj.class_id)
        name = cls.name.lower() if cls is not None else f"class {obj.class_id}"
        phrases.append(f"the {name} {dxdy_to_direction(dx, dy)}")

    if not phrases:
        return None
    if len(phrases) == 1:
        return f"Over the next second {phrases[0]}."
    return "Over the next second " + "; ".join(phrases) + "."


def _concise_template(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    caption: str | None,
    deltas: list[tuple] | None,
) -> str:
    base = caption.strip() if caption else generate_caption(packet, openimages_dict, mode="short")
    base = base.rstrip(".")
    motion = _motion_sentence(packet, openimages_dict, deltas)
    if motion:
        motion = motion.rstrip(".")
        text = f"{base}; {motion}."
    else:
        text = f"{base}."
    return _clip_words(text, max_words=200)


def _descriptive_template(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    caption: str | None,
    deltas: list[tuple] | None,
) -> str:
    opener = caption.strip().rstrip(".") + "." if caption else "A scene showing the following elements."
    object_lines: list[str] = []
    for item in _objects_for_prompt(packet, openimages_dict):
        obj = item["obj"]
        name = str(item["name"])
        object_lines.append(
            f"A {name.lower()} is in the {grid_to_phrase(obj.cx, obj.cy)} with confidence {conf_to_pct(obj.confidence)}."
        )

    motion = _motion_sentence(packet, openimages_dict, deltas)
    parts = [opener, *object_lines]
    if motion:
        parts.append(motion)
    return _clip_words(" ".join(parts), max_words=200)


def _disaster_response_template(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    caption: str | None,
    deltas: list[tuple] | None,
) -> str:
    items = _objects_for_prompt(packet, openimages_dict)
    hazard_items = [item for item in items if bool(item["hazard"])]
    person_items = [item for item in items if bool(item["person"])]
    vehicle_items = [item for item in items if bool(item["vehicle"])]

    opener = "URGENT: possible incident requiring rapid situational awareness."
    if caption:
        opener = f"URGENT: {caption.strip().rstrip('.')}."

    facts: list[str] = []
    if hazard_items:
        item = hazard_items[0]
        obj = item["obj"]
        facts.append(
            f"Hazard observed: {str(item['name']).lower()} at {grid_to_phrase(obj.cx, obj.cy)}."
        )
    if vehicle_items:
        item = vehicle_items[0]
        obj = item["obj"]
        facts.append(
            f"Possible blocked road near a {str(item['name']).lower()} at {grid_to_phrase(obj.cx, obj.cy)}."
        )
    if person_items:
        item = person_items[0]
        obj = item["obj"]
        facts.append(
            f"At least one person visible at {grid_to_phrase(obj.cx, obj.cy)}; possible injuries."
        )

    motion = _motion_sentence(packet, openimages_dict, deltas)
    if motion:
        facts.append(motion)

    return _clip_words(" ".join([opener, *facts]), max_words=200)


def _cinematic_template(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    caption: str | None,
    deltas: list[tuple] | None,
) -> str:
    opener = caption.strip().rstrip(".") + "." if caption else generate_caption(packet, openimages_dict, mode="long").rstrip(".") + "."
    items = _objects_for_prompt(packet, openimages_dict)

    lines: list[str] = [opener]
    for item in items:
        obj = item["obj"]
        name = str(item["name"]).lower()
        lines.append(
            f"Frame the {size_to_word(obj.size)} {name} at {grid_to_phrase(obj.cx, obj.cy)} with confidence {conf_to_pct(obj.confidence)}."
        )

    motion = _motion_sentence(packet, openimages_dict, deltas)
    if motion:
        lines.append(motion)

    lines.append(
        "Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style."
    )
    return _clip_words(" ".join(lines), max_words=200)


def build_nova_prompt(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    caption: str | None = None,
    deltas: list[tuple] | None = None,
    template: TemplateStyle = "descriptive",
) -> str:
    """
    Build a deterministic Nova Canvas-ready prompt from semantic packet content.
    """
    if template == "concise":
        return _concise_template(packet, openimages_dict, caption, deltas)
    if template == "descriptive":
        return _descriptive_template(packet, openimages_dict, caption, deltas)
    if template == "disaster_response":
        return _disaster_response_template(packet, openimages_dict, caption, deltas)
    if template == "cinematic":
        return _cinematic_template(packet, openimages_dict, caption, deltas)
    raise ValueError(
        "template must be one of: concise, descriptive, disaster_response, cinematic"
    )
