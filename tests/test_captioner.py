from hfups.vision.captioner import generate_caption
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket
from hfups.vision.openimages_dict import OpenImagesClass, OpenImagesDict


def _fixture_dict() -> OpenImagesDict:
    classes = [
        OpenImagesClass(id=0, label="/m/fire", name="Fire"),
        OpenImagesClass(id=1, label="/m/person", name="Person"),
        OpenImagesClass(id=2, label="/m/car", name="Car"),
        OpenImagesClass(id=3, label="/m/tree", name="Tree"),
    ]
    return OpenImagesDict(
        classes=classes,
        by_id={c.id: c for c in classes},
        by_label={c.label: c for c in classes},
    )


def _fixture_packet() -> KeyframePacket:
    return KeyframePacket(
        objects=[
            KeyframeObject(class_id=0, track_id=0, cx=1, cy=6, size=2, confidence=15),
            KeyframeObject(class_id=1, track_id=1, cx=3, cy=5, size=1, confidence=12),
            KeyframeObject(class_id=2, track_id=2, cx=4, cy=5, size=2, confidence=11),
            KeyframeObject(class_id=3, track_id=3, cx=6, cy=2, size=1, confidence=9),
        ]
    )


def test_captioner_is_deterministic_and_respects_mode_limits() -> None:
    packet = _fixture_packet()
    dictionary = _fixture_dict()

    short_a = generate_caption(packet, dictionary, mode="short")
    short_b = generate_caption(packet, dictionary, mode="short")
    medium = generate_caption(packet, dictionary, mode="medium")
    long = generate_caption(packet, dictionary, mode="long")

    assert short_a == short_b
    assert len(short_a.encode("utf-8")) <= 80
    assert len(medium.encode("utf-8")) <= 120
    assert len(long.encode("utf-8")) <= 160


def test_captioner_prioritizes_hazard_terms_when_present() -> None:
    caption = generate_caption(_fixture_packet(), _fixture_dict(), mode="medium").lower()
    assert "fire" in caption


def test_captioner_empty_packet_fallback() -> None:
    packet = KeyframePacket(objects=[])
    caption = generate_caption(packet, _fixture_dict(), mode="medium")
    assert caption == "no significant objects detected"
