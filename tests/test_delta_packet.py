import pytest

from hfups.vision.delta_packet import DeltaEntry, DeltaPacket


def test_delta_packet_roundtrip_with_edge_values() -> None:
    packet = DeltaPacket(
        entries=[
            DeltaEntry(track_id=0, dx=-4, dy=3),
            DeltaEntry(track_id=63, dx=3, dy=-4),
            DeltaEntry(track_id=7, dx=0, dy=0),
        ]
    )

    encoded = packet.encode()
    decoded = DeltaPacket.decode(encoded)

    assert decoded == packet


def test_delta_packet_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="dx"):
        DeltaEntry(track_id=1, dx=4, dy=0)
    with pytest.raises(ValueError, match="dy"):
        DeltaEntry(track_id=1, dx=0, dy=-5)
    with pytest.raises(ValueError, match="track_id"):
        DeltaEntry(track_id=64, dx=0, dy=0)


def test_delta_packet_rejects_invalid_version() -> None:
    with pytest.raises(ValueError, match="version"):
        DeltaPacket(entries=[], version=1)
