import pytest

from hfups.crc import append_crc_be, crc16_ccitt_false, verify_crc_be


def test_crc16_ccitt_false_standard_check_value() -> None:
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_hfups_append_uses_ccitt_false_result() -> None:
    payload = bytes.fromhex("000040")
    expected = payload + crc16_ccitt_false(payload).to_bytes(2, "big")
    assert append_crc_be(payload) == expected


def test_hfups_payload_verify_true() -> None:
    frame = append_crc_be(bytes.fromhex("000040"))
    assert verify_crc_be(frame) is True


def test_hfups_payload_verify_false() -> None:
    frame = append_crc_be(bytes.fromhex("000040"))
    tampered = frame[:-1] + bytes([frame[-1] ^ 0x01])
    assert verify_crc_be(tampered) is False


@pytest.mark.xfail(
    reason="Spec test vector appears inconsistent with CCITT-FALSE parameters; CCITT-FALSE yields 0x8458.",
    strict=True,
)
def test_spec_claimed_vector_for_000040() -> None:
    assert append_crc_be(bytes.fromhex("000040")) == bytes.fromhex("0000400081")
