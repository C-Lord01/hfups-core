"""HFUPS packet structures and bit packing helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IFrame:
    """HFUPS I-frame fields packed into exactly 40 bits (5 bytes)."""

    q_m: int
    q_u: int
    q_v: int
    flags: int
    timestamp_s: int
    shot_id: int
    psnr_idx: int
    quant_idx: int
    crc4: int


@dataclass(frozen=True)
class MFPacket:
    """HFUPS MF control packet fields packed into 16 bits."""

    cam_move: int
    framing: int
    subject_path: int


@dataclass(frozen=True)
class ClipParams:
    """HFUPS clip parameter fields packed into one payload byte."""

    fps_idx: int
    frame_count: int


def _validate_range(name: str, value: int, bits: int) -> None:
    if not (0 <= value < (1 << bits)):
        raise ValueError(f"{name} out of range for {bits} bits: {value}")


def pack_iframe(msg: IFrame) -> bytes:
    """Pack an IFrame into exactly 5 bytes using MSB-first bit layout."""
    _validate_range("q_m", msg.q_m, 3)
    _validate_range("q_u", msg.q_u, 5)
    _validate_range("q_v", msg.q_v, 5)
    _validate_range("flags", msg.flags, 3)
    _validate_range("timestamp_s", msg.timestamp_s, 12)
    _validate_range("shot_id", msg.shot_id, 4)
    _validate_range("psnr_idx", msg.psnr_idx, 2)
    _validate_range("quant_idx", msg.quant_idx, 2)
    _validate_range("crc4", msg.crc4, 4)

    # Field order from bit 39 down to bit 0:
    # q_m(3), q_u(5), q_v(5), flags(3), timestamp_s(12),
    # shot_id(4), psnr_idx(2), quant_idx(2), crc4(4)
    value = 0
    value |= (msg.q_m & 0x7) << 37
    value |= (msg.q_u & 0x1F) << 32
    value |= (msg.q_v & 0x1F) << 27
    value |= (msg.flags & 0x7) << 24
    value |= (msg.timestamp_s & 0xFFF) << 12
    value |= (msg.shot_id & 0xF) << 8
    value |= (msg.psnr_idx & 0x3) << 6
    value |= (msg.quant_idx & 0x3) << 4
    value |= msg.crc4 & 0xF
    return value.to_bytes(5, "big")


def unpack_iframe(data: bytes) -> IFrame:
    """Unpack exactly 5 I-frame bytes into an IFrame dataclass."""
    if len(data) != 5:
        raise ValueError("I-frame must be exactly 5 bytes")

    value = int.from_bytes(data, "big")
    return IFrame(
        q_m=(value >> 37) & 0x7,
        q_u=(value >> 32) & 0x1F,
        q_v=(value >> 27) & 0x1F,
        flags=(value >> 24) & 0x7,
        timestamp_s=(value >> 12) & 0xFFF,
        shot_id=(value >> 8) & 0xF,
        psnr_idx=(value >> 6) & 0x3,
        quant_idx=(value >> 4) & 0x3,
        crc4=value & 0xF,
    )


def pack_mf(msg: MFPacket) -> bytes:
    """Pack MF packet as marker 0xC0 plus big-endian 16-bit payload."""
    _validate_range("cam_move", msg.cam_move, 6)
    _validate_range("framing", msg.framing, 5)
    _validate_range("subject_path", msg.subject_path, 5)

    payload = ((msg.cam_move & 0x3F) << 10) | ((msg.framing & 0x1F) << 5) | (msg.subject_path & 0x1F)
    return b"\xC0" + payload.to_bytes(2, "big")


def unpack_mf(data: bytes) -> MFPacket:
    """Unpack MF packet bytes (must be 3 bytes and start with 0xC0)."""
    if len(data) != 3:
        raise ValueError("MF packet must be exactly 3 bytes")
    if data[0] != 0xC0:
        raise ValueError("MF packet marker must be 0xC0")

    payload = int.from_bytes(data[1:], "big")
    return MFPacket(
        cam_move=(payload >> 10) & 0x3F,
        framing=(payload >> 5) & 0x1F,
        subject_path=payload & 0x1F,
    )


def pack_clip_params(msg: ClipParams) -> bytes:
    """Pack ClipParams packet as marker 0xC1 plus one packed payload byte."""
    _validate_range("fps_idx", msg.fps_idx, 2)
    _validate_range("frame_count", msg.frame_count, 6)

    payload = ((msg.fps_idx & 0x3) << 6) | (msg.frame_count & 0x3F)
    return b"\xC1" + bytes([payload])


def unpack_clip_params(data: bytes) -> ClipParams:
    """Unpack ClipParams bytes (must be 2 bytes and start with 0xC1)."""
    if len(data) != 2:
        raise ValueError("ClipParams packet must be exactly 2 bytes")
    if data[0] != 0xC1:
        raise ValueError("ClipParams packet marker must be 0xC1")

    payload = data[1]
    return ClipParams(fps_idx=(payload >> 6) & 0x3, frame_count=payload & 0x3F)
