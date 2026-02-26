from __future__ import annotations

import pytest

from hfups.nova.prompt_templates import build_nova_prompt
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket
from hfups.vision.openimages_dict import OpenImagesClass, OpenImagesDict


def _dict_for_templates() -> OpenImagesDict:
    classes = [
        OpenImagesClass(id=0, label="/m/car", name="Car"),
        OpenImagesClass(id=1, label="/m/truck", name="Truck"),
        OpenImagesClass(id=2, label="/m/person", name="Person"),
        OpenImagesClass(id=3, label="/m/smoke", name="Smoke"),
        OpenImagesClass(id=4, label="/m/tree", name="Tree"),
        OpenImagesClass(id=5, label="/m/fire", name="Fire"),
        OpenImagesClass(id=6, label="/m/apc", name="APC"),
        OpenImagesClass(id=7, label="/m/fence", name="Fence"),
        OpenImagesClass(id=8, label="/m/boat", name="Boat"),
        OpenImagesClass(id=9, label="/m/floodwater", name="Floodwater"),
    ]
    return OpenImagesDict(
        classes=classes,
        by_id={c.id: c for c in classes},
        by_label={c.label: c for c in classes},
    )


def _packet(items: list[tuple[int, int, int, int, int]]) -> KeyframePacket:
    return KeyframePacket(
        objects=[
            KeyframeObject(
                class_id=class_id,
                track_id=idx,
                cx=cx,
                cy=cy,
                size=size,
                confidence=conf,
            )
            for idx, (class_id, cx, cy, size, conf) in enumerate(items)
        ]
    )


CANONICAL_CASES = [
    (
        "traffic_head_on",
        _packet([(0, 1, 6, 2, 14), (1, 3, 5, 2, 13), (2, 2, 6, 1, 11), (3, 1, 5, 1, 10)]),
        ["car", "truck", "person", "smoke"],
    ),
    (
        "wildfire_smoke_plume",
        _packet([(3, 4, 1, 3, 14), (4, 3, 4, 2, 12), (4, 5, 4, 2, 11)]),
        ["smoke", "tree"],
    ),
    (
        "weather_tree_block",
        _packet([(4, 3, 5, 3, 13), (0, 5, 6, 2, 12), (5, 5, 6, 2, 11)]),
        ["tree", "car", "fire"],
    ),
    (
        "military_convoy",
        _packet([(6, 1, 4, 1, 13), (6, 2, 4, 1, 12), (6, 3, 4, 1, 12), (6, 4, 4, 1, 11), (6, 5, 4, 1, 11)]),
        ["apc"],
    ),
    (
        "breach_trespass",
        _packet([(2, 2, 5, 1, 13), (2, 3, 5, 1, 12), (2, 4, 5, 1, 12), (2, 5, 5, 1, 11), (7, 3, 4, 2, 10)]),
        ["person", "fence"],
    ),
    (
        "coastal_flooding",
        _packet([(8, 4, 6, 2, 13), (9, 4, 5, 3, 12), (0, 2, 6, 1, 10)]),
        ["boat", "floodwater"],
    ),
]


@pytest.mark.parametrize(("case_name", "packet", "keywords"), CANONICAL_CASES)
def test_prompt_templates_cover_required_phrases_for_canonical_inputs(
    case_name: str,
    packet: KeyframePacket,
    keywords: list[str],
) -> None:
    del case_name
    dictionary = _dict_for_templates()

    concise = build_nova_prompt(packet, dictionary, template="concise")
    descriptive = build_nova_prompt(packet, dictionary, template="descriptive")
    disaster = build_nova_prompt(packet, dictionary, template="disaster_response")
    cinematic = build_nova_prompt(packet, dictionary, template="cinematic")

    assert concise.count(".") <= 1
    assert "A scene showing" in descriptive
    assert ("URGENT" in disaster) or ("possible" in disaster.lower())
    assert cinematic.endswith(
        "Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style."
    )
    assert len(cinematic.split()) <= 200

    lowered_descriptive = descriptive.lower()
    lowered_concise = concise.lower()
    for keyword in keywords:
        assert (keyword in lowered_descriptive) or (keyword in lowered_concise)


def test_concise_template_exact_with_explicit_caption() -> None:
    dictionary = _dict_for_templates()
    packet = _packet([(0, 1, 6, 2, 14), (2, 2, 6, 1, 12), (3, 1, 5, 1, 11)])
    caption = "Overturned sedan lower-left; red pickup middle-left; 1 person nearby; smoke visible"

    concise = build_nova_prompt(
        packet,
        dictionary,
        caption=caption,
        template="concise",
    )

    assert concise == "Overturned sedan lower-left; red pickup middle-left; 1 person nearby; smoke visible."
