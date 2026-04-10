from __future__ import annotations

import warnings
from typing import Literal

from hfups.vision.captioner import generate_caption
from hfups.vision.keyframe_packet import KeyframePacket
from hfups.vision.openimages_dict import OpenImagesDict
from hfups.vision.prompt_utils import dxdy_to_direction, grid_to_bucket_phrase, size_to_word

TemplateStyle = Literal["concise", "descriptive", "disaster_response", "cinematic", "ups"]

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
        position = grid_to_bucket_phrase(obj.cx, obj.cy)
        object_lines.append(f"A {name.lower()} is in the {position}.")

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
    caption_parsed: dict | None = None,
) -> str:
    items = _objects_for_prompt(packet, openimages_dict)
    hazard_items = [item for item in items if bool(item["hazard"])]
    person_items = [item for item in items if bool(item["person"])]
    vehicle_items = [item for item in items if bool(item["vehicle"])]

    opener = "Possible incident requiring rapid situational awareness."

    facts: list[str] = []
    if hazard_items:
        item = hazard_items[0]
        obj = item["obj"]
        facts.append(
            f"Hazard observed: {str(item['name']).lower()} in the {grid_to_bucket_phrase(obj.cx, obj.cy)}."
        )
    if vehicle_items:
        item = vehicle_items[0]
        obj = item["obj"]
        facts.append(
            f"Possible blocked road near a {str(item['name']).lower()} in the {grid_to_bucket_phrase(obj.cx, obj.cy)}."
        )
    if person_items:
        item = person_items[0]
        obj = item["obj"]
        facts.append(
            f"At least one person visible in the {grid_to_bucket_phrase(obj.cx, obj.cy)}; possible injuries."
        )

    motion = _motion_sentence(packet, openimages_dict, deltas)
    if motion:
        facts.append(motion)

    if caption is not None:
        facts.append(f"Scene: {caption.strip().rstrip('.')}.")
    if caption_parsed:
        hazards = caption_parsed.get("hazards") or []
        environment = caption_parsed.get("environment")
        actions = caption_parsed.get("actions") or []
        if hazards:
            facts.append(f"Hazards present: {', '.join(hazards)}.")
        if environment:
            facts.append(f"Environment: {environment}.")
        if actions:
            facts.append(f"Subjects: {', '.join(actions)}.")

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
        position = grid_to_bucket_phrase(obj.cx, obj.cy)
        lines.append(f"Frame the {size_to_word(obj.size)} {name} in the {position}.")

    motion = _motion_sentence(packet, openimages_dict, deltas)
    if motion:
        lines.append(motion)

    lines.append(
        "Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style."
    )
    return _clip_words(" ".join(lines), max_words=200)


def infer_scene_caption(detected_classes: list[str]) -> str:
    """Infer a scene context string from detected object class names.

    Fully deterministic — no external calls. Returns empty string if no
    rule matches (caller should treat empty string as no caption).

    Priority order: first matching rule wins.
    """
    cls = {c.lower() for c in detected_classes}

    if "flood" in cls or "flooded_area" in cls:
        return (
            "urban flooding emergency, vehicles submerged in floodwater, "
            "civilian distress"
        )
    if "fire" in cls and "damaged_building" in cls:
        return "structure fire with building damage, active emergency"
    if "fire" in cls:
        return "active fire emergency"
    if "smoke" in cls and "damaged_building" in cls:
        return "post-incident scene with structural damage and smoke"
    if "smoke" in cls:
        return "smoke present, possible fire or explosion nearby"
    if "accident_vehicle" in cls:
        return "vehicle accident scene, emergency response required"
    if "damaged_building" in cls:
        return "structural damage to buildings, disaster aftermath"
    if "debris" in cls:
        return "debris field, impact or storm damage"
    return ""


def _ups_template(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    caption: str | None,
    deltas: list[tuple] | None,
) -> str:
    items = _objects_for_prompt(packet, openimages_dict)

    # Infer scene context from detected classes if no explicit caption given
    detected_class_names = [str(item["name"]).lower() for item in items]
    effective_caption = caption or infer_scene_caption(detected_class_names) or None

    print(f"[DEBUG _ups_template] detected_class_names: {detected_class_names}")
    print(f"[DEBUG _ups_template] infer_scene_caption result: {repr(infer_scene_caption(detected_class_names))}")
    print(f"[DEBUG _ups_template] effective_caption: {repr(effective_caption)}")

    # Layer A — Realism Booster
    layer_a = (
        "Ultra-realistic documentary photograph, photojournalism style, "
        "natural color science, high dynamic range, no CGI"
    )

    # Layer B — Subject and Scene
    # Build from detected objects using natural spatial language (no grid coords)
    subject_parts: list[str] = []
    # TODO: The encoding layer supports up to 12 objects, but the prompt is capped at 6
    # here to avoid token overflow in FLUX.1-schnell. Revisit when a smarter truncation
    # strategy is implemented (e.g. priority-ranked by confidence or scene relevance).
    for item in items[:6]:
        obj = item["obj"]
        name = str(item["name"]).lower()
        position = grid_to_bucket_phrase(obj.cx, obj.cy)
        size = size_to_word(obj.size)
        subject_parts.append(f"a {size} {name} in the {position}")

    if subject_parts:
        objects_desc = ", ".join(subject_parts)
        if effective_caption:
            layer_b = f"Scene showing: {effective_caption.strip().rstrip('.')}. Additionally: {objects_desc}."
        else:
            layer_b = "Scene showing: " + objects_desc + "."
    else:
        layer_b = ("Scene showing: " + effective_caption.strip().rstrip(".") + ".") if effective_caption else "A disaster scene."

    # Layer C — Camera and Lighting
    layer_c = (
        "Shot on a full-frame DSLR, 24-70mm lens, f/4, natural available light, "
        "wide establishing shot"
    )

    # Layer D — Imperfections
    layer_d = (
        "Subtle motion blur, atmospheric haze, dust particles, "
        "realistic shadows, minor lens distortion"
    )

    parts = [layer_a + ".", layer_b, layer_c + ".", layer_d + "."]
    motion = _motion_sentence(packet, openimages_dict, deltas)
    if motion:
        parts.insert(2, motion)

    return _clip_words(" ".join(parts), max_words=200)


def build_nova_prompt(
    packet: KeyframePacket,
    openimages_dict: OpenImagesDict,
    caption: str | None = None,
    deltas: list[tuple] | None = None,
    template: TemplateStyle = "ups",
    caption_parsed: dict | None = None,
) -> str:
    """
    Build a deterministic image-generation-ready prompt from semantic packet content.

    The default template is "ups" (Universal Prompt Structure), which produces
    photorealistic photography briefs suitable for FLUX.1 and other diffusion models.
    "disaster_response" is retained for backward compatibility but is deprecated.
    """
    if template == "ups":
        return _ups_template(packet, openimages_dict, caption, deltas)
    if template == "concise":
        return _concise_template(packet, openimages_dict, caption, deltas)
    if template == "descriptive":
        return _descriptive_template(packet, openimages_dict, caption, deltas)
    if template == "disaster_response":
        warnings.warn(
            "Template 'disaster_response' is deprecated; use 'ups' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _disaster_response_template(packet, openimages_dict, caption, deltas, caption_parsed)
    if template == "cinematic":
        return _cinematic_template(packet, openimages_dict, caption, deltas)
    raise ValueError(
        "template must be one of: concise, descriptive, disaster_response, cinematic, ups"
    )
