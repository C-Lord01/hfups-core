"""HFUPS wire framing helpers (COBS + CRC + delimiter)."""

from hfups.cobs import cobs_decode, cobs_encode
from hfups.crc import append_crc_be, verify_crc_be


def encode_frame(payload: bytes) -> bytes:
    """Encode one HFUPS frame as COBS(payload+CRC) followed by 0x00."""
    payload_with_crc = append_crc_be(payload)
    return cobs_encode(payload_with_crc) + b"\x00"


def decode_frame(frame: bytes) -> bytes:
    """Decode one delimited HFUPS frame and verify CRC."""
    if not frame or frame[-1] != 0:
        raise ValueError("Frame delimiter missing")

    encoded = frame[:-1]
    decoded = cobs_decode(encoded)

    if not verify_crc_be(decoded):
        raise ValueError("CRC verification failed")

    return decoded[:-2]
