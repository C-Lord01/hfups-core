import pytest

from hfups.vision.keyframe_packet import (
    KeyframeObject,
    KeyframePacket,
    estimate_airtime_seconds,
)


def test_empty_packet_roundtrip() -> None:
    pkt = KeyframePacket(objects=[])
    encoded = pkt.encode()
    decoded = KeyframePacket.decode(encoded)

    assert decoded == pkt
    assert len(encoded) == 1


def test_single_object_known_pattern_roundtrip() -> None:
    obj = KeyframeObject(
        class_id=512,
        track_id=7,
        cx=3,
        cy=4,
        size=2,
        confidence=10,
    )
    pkt = KeyframePacket(objects=[obj])

    encoded = pkt.encode()
    decoded = KeyframePacket.decode(encoded)

    assert decoded == pkt
    assert len(encoded) == 5


def test_max_objects_roundtrip_and_size() -> None:
    objects = [
        KeyframeObject(
            class_id=i,
            track_id=i % 64,
            cx=i % 8,
            cy=(i + 1) % 8,
            size=i % 4,
            confidence=(i * 3) % 16,
        )
        for i in range(12)
    ]
    pkt = KeyframePacket(objects=objects)

    encoded = pkt.encode()
    decoded = KeyframePacket.decode(encoded)

    assert len(encoded) == 46
    assert decoded == pkt


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="class_id"):
        KeyframeObject(class_id=1024, track_id=0, cx=0, cy=0, size=0, confidence=0)

    with pytest.raises(ValueError, match="track_id"):
        KeyframeObject(class_id=0, track_id=64, cx=0, cy=0, size=0, confidence=0)

    with pytest.raises(ValueError, match="cx"):
        KeyframeObject(class_id=0, track_id=0, cx=8, cy=0, size=0, confidence=0)

    base_obj = KeyframeObject(class_id=0, track_id=0, cx=0, cy=0, size=0, confidence=0)
    with pytest.raises(ValueError, match="object count"):
        KeyframePacket(objects=[base_obj] * 13)


def test_encode_is_deterministic() -> None:
    pkt = KeyframePacket(
        objects=[
            KeyframeObject(
                class_id=42,
                track_id=5,
                cx=2,
                cy=6,
                size=1,
                confidence=12,
            )
        ]
    )
    assert pkt.encode() == pkt.encode()


def test_estimate_airtime_seconds() -> None:
    airtime = estimate_airtime_seconds(46, kbps=10.0)
    assert airtime == pytest.approx(0.0368, abs=1e-9)
