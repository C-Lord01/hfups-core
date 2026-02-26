from __future__ import annotations

import random
import re

import pytest

from hfups.framing import encode_frame
from hfups.integrations.tx_rx_keyframe import PACKET_TYPE_KEYFRAME, wrap_typed_payload
from hfups.transport.vara_text_bridge import decode_text_to_stream, encode_stream_to_text
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket


def _make_sample_framed_stream() -> bytes:
    packets = [
        KeyframePacket(
            objects=[
                KeyframeObject(class_id=0, track_id=0, cx=1, cy=6, size=1, confidence=14),
                KeyframeObject(class_id=1, track_id=1, cx=4, cy=5, size=1, confidence=12),
            ]
        ),
        KeyframePacket(
            objects=[
                KeyframeObject(class_id=0, track_id=0, cx=2, cy=6, size=1, confidence=14),
                KeyframeObject(class_id=1, track_id=1, cx=4, cy=4, size=1, confidence=12),
            ]
        ),
        KeyframePacket(
            objects=[
                KeyframeObject(class_id=0, track_id=0, cx=3, cy=6, size=1, confidence=14),
                KeyframeObject(class_id=1, track_id=1, cx=4, cy=3, size=1, confidence=12),
            ]
        ),
    ]
    frames = [
        encode_frame(wrap_typed_payload(PACKET_TYPE_KEYFRAME, packet.encode()))
        for packet in packets
    ]
    return b"".join(frames)


def _flip_first_base64_char(text: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("CHUNK "):
            payload_idx = idx + 1
            payload = lines[payload_idx]
            if not payload:
                continue
            first = payload[0]
            replacement = "A" if first != "A" else "B"
            lines[payload_idx] = replacement + payload[1:]
            return "\n".join(lines) + "\n"
    raise AssertionError("No chunk payload line found")


def _remove_chunk_block(text: str, chunk_index: int) -> str:
    lines = text.splitlines()
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith(f"CHUNK {chunk_index}/"):
            idx += 1
            while idx < len(lines) and lines[idx] != "ENDCHUNK":
                idx += 1
            if idx < len(lines) and lines[idx] == "ENDCHUNK":
                idx += 1
            continue
        out.append(line)
        idx += 1
    return "\n".join(out) + "\n"


def _shuffle_chunk_blocks(text: str) -> str:
    lines = text.splitlines()
    assert lines
    header = lines[0]
    footer = lines[-1]
    blocks: list[list[str]] = []

    idx = 1
    while idx < len(lines) - 1:
        if not lines[idx].startswith("CHUNK "):
            raise AssertionError(f"unexpected line while parsing chunk blocks: {lines[idx]!r}")
        block = [lines[idx]]
        idx += 1
        while idx < len(lines) and lines[idx] != "ENDCHUNK":
            block.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise AssertionError("missing ENDCHUNK while parsing chunk blocks")
        block.append(lines[idx])  # ENDCHUNK
        idx += 1
        blocks.append(block)

    rng = random.Random(1337)
    rng.shuffle(blocks)

    out = [header]
    for block in blocks:
        out.extend(block)
    out.append(footer)
    return "\n".join(out) + "\n"


def test_roundtrip_small_stream() -> None:
    stream_bytes = bytes([value % 256 for value in range(500)])
    text = encode_stream_to_text(stream_bytes, chunk_size=180)
    decoded = decode_text_to_stream(text)
    assert decoded == stream_bytes


def test_roundtrip_real_framed_stream_sample() -> None:
    framed_stream = _make_sample_framed_stream()
    text = encode_stream_to_text(framed_stream, chunk_size=180)
    decoded = decode_text_to_stream(text)
    assert decoded == framed_stream


def test_detect_corruption() -> None:
    stream_bytes = bytes([value % 256 for value in range(500)])
    text = encode_stream_to_text(stream_bytes, chunk_size=180)
    corrupted = _flip_first_base64_char(text)

    with pytest.raises(ValueError, match="CRC mismatch|base64"):
        decode_text_to_stream(corrupted)


def test_detect_missing_chunk() -> None:
    stream_bytes = bytes([value % 256 for value in range(500)])
    text = encode_stream_to_text(stream_bytes, chunk_size=180)
    missing_one = _remove_chunk_block(text, chunk_index=2)

    with pytest.raises(ValueError, match=re.escape("Missing chunk")):
        decode_text_to_stream(missing_one)


def test_out_of_order_chunks_reassembles() -> None:
    stream_bytes = bytes([value % 256 for value in range(500)])
    text = encode_stream_to_text(stream_bytes, chunk_size=180)
    shuffled = _shuffle_chunk_blocks(text)
    decoded = decode_text_to_stream(shuffled)
    assert decoded == stream_bytes
