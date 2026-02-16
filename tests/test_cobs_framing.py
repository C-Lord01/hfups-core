import pytest

from hfups.cobs import cobs_decode, cobs_encode
from hfups.framing import decode_frame, encode_frame


def test_cobs_roundtrip_empty() -> None:
    data = b""
    encoded = cobs_encode(data)
    assert b"\x00" not in encoded
    assert cobs_decode(encoded) == data


def test_cobs_roundtrip_with_embedded_zero() -> None:
    data = b"\x11\x22\x00\x33"
    encoded = cobs_encode(data)
    assert b"\x00" not in encoded
    assert cobs_decode(encoded) == data


def test_cobs_roundtrip_large_with_zeros() -> None:
    data = bytes(0 if i % 50 == 0 else (i % 256) for i in range(300))
    encoded = cobs_encode(data)
    assert b"\x00" not in encoded
    assert cobs_decode(encoded) == data


def test_hfups_frame_roundtrip() -> None:
    payload = bytes.fromhex("000040")
    frame = encode_frame(payload)
    assert frame.endswith(b"\x00")
    assert decode_frame(frame) == payload


def test_hfups_frame_corruption_raises_value_error() -> None:
    payload = bytes.fromhex("000040")
    frame = bytearray(encode_frame(payload))
    frame[-2] ^= 0x01

    with pytest.raises(ValueError):
        decode_frame(bytes(frame))


def test_hfups_missing_delimiter_raises_value_error() -> None:
    payload = bytes.fromhex("000040")
    frame = encode_frame(payload)

    with pytest.raises(ValueError):
        decode_frame(frame[:-1])
