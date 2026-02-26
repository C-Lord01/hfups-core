from __future__ import annotations

from collections.abc import Callable

from hfups.integrations.transport_shim import TransportShim
from hfups.transport.semantic_transport import ReceivedFrame, SemanticTransport

PACKET_TYPE_KEYFRAME = 0x01
PACKET_TYPE_DELTA = 0x02
PACKET_TYPE_CAPTION = 0x03
PACKET_TYPE_ACK = 0x7F
ENVELOPE_MARKER = 0xFE


def wrap_typed_payload(packet_type: int, payload: bytes) -> bytes:
    if not isinstance(packet_type, int) or isinstance(packet_type, bool):
        raise ValueError("packet_type must be an integer")
    if packet_type < 0 or packet_type > 255:
        raise ValueError("packet_type must be in range [0, 255]")
    return bytes([packet_type]) + payload


def unwrap_typed_payload(data: bytes) -> tuple[int, bytes]:
    if not data:
        raise ValueError("typed payload must not be empty")
    return data[0], data[1:]


def wrap_envelope(seq: int, packet_type: int, payload: bytes) -> bytes:
    if not isinstance(seq, int) or isinstance(seq, bool) or not (0 <= seq <= 255):
        raise ValueError("seq must be in range [0, 255]")
    return bytes([ENVELOPE_MARKER, seq]) + wrap_typed_payload(packet_type, payload)


def unwrap_semantic_payload(data: bytes) -> tuple[int, bytes, int | None]:
    """
    Returns (packet_type, body, seq).

    seq is None for legacy non-enveloped payloads.
    """
    if not data:
        raise ValueError("semantic payload must not be empty")
    if data[0] != ENVELOPE_MARKER:
        packet_type, body = unwrap_typed_payload(data)
        return packet_type, body, None
    if len(data) < 3:
        raise ValueError("envelope payload too short")
    seq = data[1]
    packet_type = data[2]
    body = data[3:]
    return packet_type, body, seq


def build_ack_payload(seq: int) -> bytes:
    if not isinstance(seq, int) or isinstance(seq, bool) or not (0 <= seq <= 255):
        raise ValueError("seq must be in range [0, 255]")
    return bytes([PACKET_TYPE_ACK, seq])


def parse_ack_payload(payload: bytes) -> int | None:
    if len(payload) != 2:
        return None
    if payload[0] != PACKET_TYPE_ACK:
        return None
    return payload[1]


def tx_keyframe_packet(
    pkt_bytes: bytes,
    transport: TransportShim | SemanticTransport,
    packet_type: int = PACKET_TYPE_KEYFRAME,
) -> bytes:
    result = transport.send_payload(wrap_typed_payload(packet_type, pkt_bytes))
    if isinstance(result, bytes):
        return result
    return b""


def rx_loop(
    transport: TransportShim | SemanticTransport,
    handler: Callable[[int, bytes], None],
) -> int:
    received = 0

    if hasattr(transport, "recv_payloads"):
        recv_payloads = getattr(transport, "recv_payloads")
        for item in recv_payloads():
            if isinstance(item, ReceivedFrame):
                payload = item.payload
            elif isinstance(item, (bytes, bytearray)):
                payload = bytes(item)
            else:
                payload = item.payload  # type: ignore[attr-defined]
            packet_type, body, _ = unwrap_semantic_payload(payload)
            handler(packet_type, body)
            received += 1
        return received

    while True:
        payload = transport.recv_payload()  # type: ignore[attr-defined]
        if payload is None:
            break
        packet_type, body, _ = unwrap_semantic_payload(payload)
        handler(packet_type, body)
        received += 1
    return received
