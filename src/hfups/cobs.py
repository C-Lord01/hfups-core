"""COBS (Consistent Overhead Byte Stuffing) helpers for HFUPS."""


def cobs_encode(data: bytes) -> bytes:
    """Encode bytes with standard COBS; output contains no zero bytes."""
    if not data:
        return b"\x01"

    out = bytearray()
    index = 0

    while index < len(data):
        code_index = len(out)
        out.append(0)
        code = 1

        while index < len(data) and data[index] != 0 and code < 0xFF:
            out.append(data[index])
            index += 1
            code += 1

        out[code_index] = code

        if index < len(data) and data[index] == 0:
            index += 1

    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    """Decode standard COBS bytes; raise ValueError on malformed input."""
    if not data:
        raise ValueError("Malformed COBS input: empty")

    out = bytearray()
    index = 0

    while index < len(data):
        code = data[index]
        if code == 0:
            raise ValueError("Malformed COBS input: zero code byte")

        index += 1
        next_index = index + code - 1
        if next_index > len(data):
            raise ValueError("Malformed COBS input: code past end")

        out.extend(data[index:next_index])
        index = next_index

        if code != 0xFF and index < len(data):
            out.append(0)

    return bytes(out)
