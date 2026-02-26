from __future__ import annotations

from dataclasses import dataclass

from hfups.vision.delta_packet import DeltaPacket
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket
from hfups.vision.openimages_dict import OpenImagesDict


def _bucket_x(cx: int) -> str:
    if cx <= 2:
        return "left"
    if cx <= 4:
        return "center"
    return "right"


def _bucket_y(cy: int) -> str:
    if cy <= 2:
        return "top"
    if cy <= 4:
        return "middle"
    return "bottom"


def _position_phrase(obj: KeyframeObject) -> str:
    return f"{_bucket_y(obj.cy)}-{_bucket_x(obj.cx)} (grid {obj.cx + 1},{obj.cy + 1})"


def _motion_phrase(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return "stays in place"

    words: list[str] = []
    mag = max(abs(dx), abs(dy))
    if mag >= 3:
        words.append("clearly")
    elif mag >= 1:
        words.append("slightly")

    direction_bits: list[str] = []
    if dy < 0:
        direction_bits.append("up")
    elif dy > 0:
        direction_bits.append("down")
    if dx < 0:
        direction_bits.append("left")
    elif dx > 0:
        direction_bits.append("right")

    if direction_bits:
        words.append("toward the " + "-".join(direction_bits))
    else:
        words.append("with minimal movement")
    return "moves " + " ".join(words)


def _clip_words(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def apply_delta_packet(keyframe: KeyframePacket, delta_packet: DeltaPacket) -> KeyframePacket:
    by_track = {obj.track_id: obj for obj in keyframe.objects}
    updates: dict[int, tuple[int, int]] = {}
    for entry in delta_packet.entries:
        src = by_track.get(entry.track_id)
        if src is None:
            continue
        new_cx = max(0, min(7, src.cx + entry.dx))
        new_cy = max(0, min(7, src.cy + entry.dy))
        updates[entry.track_id] = (new_cx, new_cy)

    updated_objects: list[KeyframeObject] = []
    for obj in keyframe.objects:
        if obj.track_id in updates:
            new_cx, new_cy = updates[obj.track_id]
            updated_objects.append(
                KeyframeObject(
                    class_id=obj.class_id,
                    track_id=obj.track_id,
                    cx=new_cx,
                    cy=new_cy,
                    size=obj.size,
                    confidence=obj.confidence,
                )
            )
        else:
            updated_objects.append(obj)
    return KeyframePacket(objects=updated_objects, version=keyframe.version)


def build_nova_prompt(
    keyframe: KeyframePacket,
    openimages_dict: OpenImagesDict,
    *,
    caption: str | None = None,
    delta_packet: DeltaPacket | None = None,
) -> str:
    opener = "A scene showing the following elements."
    if caption:
        opener = f"{caption.strip()}."

    object_sentences: list[str] = []
    for obj in keyframe.objects:
        cls = openimages_dict.by_id.get(obj.class_id)
        name = cls.name if cls is not None else f"class {obj.class_id}"
        object_sentences.append(
            f"A {name.lower()} is in the {_position_phrase(obj)} with confidence {obj.confidence}/15."
        )

    motion_sentences: list[str] = []
    if delta_packet is not None and delta_packet.entries:
        track_to_name: dict[int, str] = {}
        for obj in keyframe.objects:
            cls = openimages_dict.by_id.get(obj.class_id)
            track_to_name[obj.track_id] = cls.name if cls else f"class {obj.class_id}"
        for entry in delta_packet.entries:
            name = track_to_name.get(entry.track_id, f"track {entry.track_id}")
            motion_sentences.append(f"Over the next second, {name.lower()} {_motion_phrase(entry.dx, entry.dy)}.")

    prompt = " ".join([opener, *object_sentences, *motion_sentences]).strip()
    return _clip_words(prompt, max_words=200)


@dataclass(frozen=True)
class StoryboardItem:
    time_s: float
    prompt: str
    keyframe: KeyframePacket


def build_storyboard(
    initial_keyframe: KeyframePacket,
    openimages_dict: OpenImagesDict,
    deltas: list[DeltaPacket],
    *,
    caption: str | None = None,
    start_time_s: float = 0.0,
    delta_step_s: float = 0.5,
) -> list[StoryboardItem]:
    current = initial_keyframe
    timeline: list[StoryboardItem] = [
        StoryboardItem(
            time_s=start_time_s,
            prompt=build_nova_prompt(current, openimages_dict, caption=caption),
            keyframe=current,
        )
    ]

    current_time = start_time_s
    for delta in deltas:
        current_time += delta_step_s
        current = apply_delta_packet(current, delta)
        timeline.append(
            StoryboardItem(
                time_s=current_time,
                prompt=build_nova_prompt(
                    current,
                    openimages_dict,
                    caption=caption,
                    delta_packet=delta,
                ),
                keyframe=current,
            )
        )
    return timeline
