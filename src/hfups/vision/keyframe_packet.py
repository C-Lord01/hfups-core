from __future__ import annotations

from dataclasses import dataclass

MAX_OBJECTS = 12
VERSION = 1
MAX_PACKET_BYTES = 220


def _validate_range(name: str, value: int, min_value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be in range [{min_value}, {max_value}]")


@dataclass(frozen=True)
class KeyframeObject:
    class_id: int
    track_id: int
    cx: int
    cy: int
    size: int
    confidence: int

    def __post_init__(self) -> None:
        _validate_range("class_id", self.class_id, 0, 1023)
        _validate_range("track_id", self.track_id, 0, 63)
        _validate_range("cx", self.cx, 0, 7)
        _validate_range("cy", self.cy, 0, 7)
        _validate_range("size", self.size, 0, 3)
        _validate_range("confidence", self.confidence, 0, 15)


@dataclass(frozen=True)
class KeyframePacket:
    objects: list[KeyframeObject]
    version: int = VERSION

    def __post_init__(self) -> None:
        _validate_range("version", self.version, VERSION, VERSION)
        if len(self.objects) > MAX_OBJECTS:
            raise ValueError(f"object count must be <= {MAX_OBJECTS}")
        for idx, obj in enumerate(self.objects):
            if not isinstance(obj, KeyframeObject):
                raise ValueError(f"objects[{idx}] must be a KeyframeObject")

    def encode(self) -> bytes:
        writer = _BitWriter()
        object_count = len(self.objects)
        writer.write(self.version, 3)
        writer.write(object_count, 4)
        writer.write(0, 1)

        for obj in self.objects:
            writer.write(obj.class_id, 10)
            writer.write(obj.track_id, 6)
            writer.write(obj.cx, 3)
            writer.write(obj.cy, 3)
            writer.write(obj.size, 2)
            writer.write(obj.confidence, 4)
            writer.write(0, 2)

        encoded = writer.to_bytes()
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError(
                f"encoded packet is {len(encoded)} bytes, exceeds max of {MAX_PACKET_BYTES}"
            )
        return encoded

    @staticmethod
    def decode(data: bytes) -> "KeyframePacket":
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("data must be bytes")
        if not data:
            raise ValueError("data must not be empty")

        reader = _BitReader(bytes(data))
        version = reader.read(3)
        if version != VERSION:
            raise ValueError(f"unsupported keyframe packet version: {version}")

        object_count = reader.read(4)
        if object_count > MAX_OBJECTS:
            raise ValueError(f"object_count must be <= {MAX_OBJECTS}, got {object_count}")

        header_reserved = reader.read(1)
        if header_reserved != 0:
            raise ValueError("header reserved bit must be 0")

        objects: list[KeyframeObject] = []
        for idx in range(object_count):
            class_id = reader.read(10)
            track_id = reader.read(6)
            cx = reader.read(3)
            cy = reader.read(3)
            size = reader.read(2)
            confidence = reader.read(4)
            reserved = reader.read(2)
            if reserved != 0:
                raise ValueError(f"object {idx} reserved bits must be 0")

            try:
                obj = KeyframeObject(
                    class_id=class_id,
                    track_id=track_id,
                    cx=cx,
                    cy=cy,
                    size=size,
                    confidence=confidence,
                )
            except ValueError as exc:
                raise ValueError(f"invalid object at index {idx}: {exc}") from exc
            objects.append(obj)

        if reader.remaining_bits:
            trailing_bits = reader.read(reader.remaining_bits)
            if trailing_bits != 0:
                raise ValueError("trailing bits beyond packet structure must be zero")

        return KeyframePacket(objects=objects, version=version)


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
        self._data = data
        self._total_bits = len(data) * 8
        self._offset = 0
        self._value = int.from_bytes(data, byteorder="big")

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


def estimate_airtime_seconds(byte_count: int, kbps: float = 10.0) -> float:
    if byte_count < 0:
        raise ValueError("byte_count must be >= 0")
    if kbps <= 0:
        raise ValueError("kbps must be > 0")
    bits = byte_count * 8
    return bits / (kbps * 1000.0)
