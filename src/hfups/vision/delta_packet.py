from __future__ import annotations

from dataclasses import dataclass

# Note: The requested field widths (track_id=6, dx=3, dy=3) total 12 bits/entry.
# The prompt text also mentions "11 bits", which is inconsistent. This
# implementation follows the explicit field widths and uses 12 bits per entry.
DELTA_VERSION = 2
MAX_DELTA_ENTRIES = 31
MAX_DELTA_BYTES = 120


def _validate_range(name: str, value: int, min_value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be in range [{min_value}, {max_value}]")


def _encode_signed_3bit(value: int) -> int:
    _validate_range("delta", value, -4, 3)
    if value < 0:
        return (1 << 3) + value
    return value


def _decode_signed_3bit(value: int) -> int:
    if value >= 4:
        return value - 8
    return value


@dataclass(frozen=True)
class DeltaEntry:
    track_id: int
    dx: int
    dy: int

    def __post_init__(self) -> None:
        _validate_range("track_id", self.track_id, 0, 63)
        _validate_range("dx", self.dx, -4, 3)
        _validate_range("dy", self.dy, -4, 3)


@dataclass(frozen=True)
class DeltaPacket:
    entries: list[DeltaEntry]
    version: int = DELTA_VERSION

    def __post_init__(self) -> None:
        if self.version != DELTA_VERSION:
            raise ValueError(f"version must be {DELTA_VERSION}")
        if len(self.entries) > MAX_DELTA_ENTRIES:
            raise ValueError(f"entry_count must be <= {MAX_DELTA_ENTRIES}")

    def encode(self) -> bytes:
        writer = _BitWriter()
        writer.write(self.version, 3)
        writer.write(len(self.entries), 5)
        for entry in self.entries:
            writer.write(entry.track_id, 6)
            writer.write(_encode_signed_3bit(entry.dx), 3)
            writer.write(_encode_signed_3bit(entry.dy), 3)
        encoded = writer.to_bytes()
        if len(encoded) > MAX_DELTA_BYTES:
            raise ValueError(f"delta packet is {len(encoded)} bytes, exceeds max of {MAX_DELTA_BYTES}")
        return encoded

    @staticmethod
    def decode(data: bytes) -> "DeltaPacket":
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("data must be bytes")
        if not data:
            raise ValueError("data must not be empty")

        reader = _BitReader(bytes(data))
        version = reader.read(3)
        if version != DELTA_VERSION:
            raise ValueError(f"unsupported delta version: {version}")
        entry_count = reader.read(5)
        if entry_count > MAX_DELTA_ENTRIES:
            raise ValueError(f"entry_count must be <= {MAX_DELTA_ENTRIES}, got {entry_count}")

        entries: list[DeltaEntry] = []
        for idx in range(entry_count):
            track_id = reader.read(6)
            dx = _decode_signed_3bit(reader.read(3))
            dy = _decode_signed_3bit(reader.read(3))
            try:
                entries.append(DeltaEntry(track_id=track_id, dx=dx, dy=dy))
            except ValueError as exc:
                raise ValueError(f"invalid delta entry at index {idx}: {exc}") from exc

        if reader.remaining_bits:
            trailing = reader.read(reader.remaining_bits)
            if trailing != 0:
                raise ValueError("trailing bits beyond delta packet structure must be zero")

        return DeltaPacket(entries=entries, version=version)


class _BitWriter:
    def __init__(self) -> None:
        self._value = 0
        self._bit_count = 0

    def write(self, value: int, bits: int) -> None:
        if bits <= 0:
            raise ValueError("bits must be > 0")
        if value < 0 or value >= (1 << bits):
            raise ValueError(f"value {value} does not fit in {bits} bits")
        self._value = (self._value << bits) | value
        self._bit_count += bits

    def to_bytes(self) -> bytes:
        if self._bit_count == 0:
            return b""
        pad_bits = (-self._bit_count) % 8
        final_value = self._value << pad_bits
        byte_count = (self._bit_count + pad_bits) // 8
        return final_value.to_bytes(byte_count, byteorder="big")


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._value = int.from_bytes(data, byteorder="big")
        self._total_bits = len(data) * 8
        self._offset = 0

    @property
    def remaining_bits(self) -> int:
        return self._total_bits - self._offset

    def read(self, bits: int) -> int:
        if bits <= 0:
            raise ValueError("bits must be > 0")
        if bits > self.remaining_bits:
            raise ValueError("not enough bits remaining to read")
        shift = self._total_bits - (self._offset + bits)
        mask = (1 << bits) - 1
        value = (self._value >> shift) & mask
        self._offset += bits
        return value
