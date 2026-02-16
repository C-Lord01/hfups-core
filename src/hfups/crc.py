"""CRC-16/CCITT-FALSE helpers for HFUPS."""


def crc16_ccitt_false(data: bytes) -> int:
    """Compute CRC-16/CCITT-FALSE over ``data``."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def append_crc_be(payload: bytes) -> bytes:
    """Append CRC-16/CCITT-FALSE in big-endian byte order."""
    crc = crc16_ccitt_false(payload)
    return payload + crc.to_bytes(2, "big")


def verify_crc_be(payload_plus_crc: bytes) -> bool:
    """Verify trailing big-endian CRC-16/CCITT-FALSE bytes."""
    if len(payload_plus_crc) < 2:
        return False
    payload = payload_plus_crc[:-2]
    expected_crc = int.from_bytes(payload_plus_crc[-2:], "big")
    return crc16_ccitt_false(payload) == expected_crc
