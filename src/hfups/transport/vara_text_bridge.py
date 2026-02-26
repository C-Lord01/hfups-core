from __future__ import annotations

import base64
import binascii
import hashlib
import re
import zlib
from dataclasses import dataclass


_HEADER_RE = re.compile(r"^HFUPS-VARA-1\s+STREAM\s+([0-9A-Fa-f]{8})\s+TOTAL\s+(\d+)$")
_CHUNK_RE = re.compile(r"^CHUNK\s+(\d+)\s*/\s*(\d+)\s+CRC32\s+([0-9A-Fa-f]{8})$")
_ENDSTREAM_RE = re.compile(r"^ENDSTREAM\s+CRC32\s+([0-9A-Fa-f]{8})\s+BYTES\s+(\d+)$")


@dataclass(frozen=True)
class VaraChunk:
    stream_id: str
    index: int
    total: int
    crc32_hex: str
    b64: str


@dataclass(frozen=True)
class _ParsedStream:
    stream_id: str
    total: int
    stream_crc32_hex: str
    total_bytes: int
    chunks: dict[int, bytes]


def _normalize_lines(text: str) -> list[tuple[int, str]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[tuple[int, str]] = []
    for line_number, line in enumerate(normalized.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        out.append((line_number, stripped))
    return out


def _crc32_hex(data: bytes) -> str:
    return f"{(zlib.crc32(data) & 0xFFFFFFFF):08x}"


def _stream_id_for_bytes(stream_bytes: bytes) -> str:
    return hashlib.sha256(stream_bytes).digest()[:4].hex()


def encode_stream_to_text(
    stream_bytes: bytes,
    chunk_size: int = 180,
) -> str:
    """Return full text document containing header + chunks + footer."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    stream_id = _stream_id_for_bytes(stream_bytes)
    total_chunks = 0 if not stream_bytes else ((len(stream_bytes) + chunk_size - 1) // chunk_size)

    lines: list[str] = [f"HFUPS-VARA-1 STREAM {stream_id} TOTAL {total_chunks}"]
    for index in range(total_chunks):
        start = index * chunk_size
        end = min(len(stream_bytes), start + chunk_size)
        chunk = stream_bytes[start:end]
        chunk_crc = _crc32_hex(chunk)
        b64 = base64.b64encode(chunk).decode("ascii")
        lines.append(f"CHUNK {index + 1}/{total_chunks} CRC32 {chunk_crc}")
        lines.append(b64)
        lines.append("ENDCHUNK")

    stream_crc = _crc32_hex(stream_bytes)
    lines.append(f"ENDSTREAM CRC32 {stream_crc} BYTES {len(stream_bytes)}")
    return "\n".join(lines) + "\n"


def _parse_stream_blocks(text: str) -> list[_ParsedStream]:
    lines = _normalize_lines(text)
    streams: list[_ParsedStream] = []
    idx = 0

    while idx < len(lines):
        line_number, header_line = lines[idx]
        match = _HEADER_RE.match(header_line)
        if match is None:
            raise ValueError(f"Unknown line {line_number}: {header_line!r}")

        stream_id = match.group(1).lower()
        total = int(match.group(2))
        idx += 1

        chunks: dict[int, bytes] = {}
        footer_crc: str | None = None
        footer_bytes: int | None = None

        while idx < len(lines):
            chunk_line_number, current = lines[idx]

            end_match = _ENDSTREAM_RE.match(current)
            if end_match is not None:
                footer_crc = end_match.group(1).lower()
                footer_bytes = int(end_match.group(2))
                idx += 1
                break

            chunk_match = _CHUNK_RE.match(current)
            if chunk_match is None:
                raise ValueError(f"Unknown line {chunk_line_number}: {current!r}")

            chunk_index = int(chunk_match.group(1))
            chunk_total = int(chunk_match.group(2))
            chunk_crc_hex = chunk_match.group(3).lower()
            idx += 1

            payload_lines: list[tuple[int, str]] = []
            found_endchunk = False
            while idx < len(lines):
                payload_line_number, payload_line = lines[idx]
                if payload_line == "ENDCHUNK":
                    found_endchunk = True
                    idx += 1
                    break
                if _CHUNK_RE.match(payload_line) or _ENDSTREAM_RE.match(payload_line) or _HEADER_RE.match(payload_line):
                    raise ValueError(f"Missing ENDCHUNK before line {payload_line_number}")
                payload_lines.append((payload_line_number, payload_line))
                idx += 1

            if not found_endchunk:
                raise ValueError(f"Missing ENDCHUNK for chunk {chunk_index}/{chunk_total}")
            if len(payload_lines) != 1:
                raise ValueError(
                    f"Chunk {chunk_index}/{chunk_total} must contain exactly one base64 line"
                )
            if chunk_total != total:
                raise ValueError(
                    f"Chunk total mismatch in stream {stream_id}: header TOTAL {total} vs chunk TOTAL {chunk_total}"
                )
            if total == 0:
                raise ValueError(f"Stream {stream_id} declares TOTAL 0 but contains chunk records")
            if chunk_index < 1 or chunk_index > total:
                raise ValueError(f"Chunk index out of range in stream {stream_id}: {chunk_index}/{total}")
            if chunk_index in chunks:
                raise ValueError(f"Duplicate chunk index {chunk_index} in stream {stream_id}")

            payload_line_number, payload_b64 = payload_lines[0]
            try:
                chunk_bytes = base64.b64decode(payload_b64.encode("ascii"), validate=True)
            except (binascii.Error, UnicodeEncodeError) as exc:
                raise ValueError(
                    f"Invalid base64 payload at line {payload_line_number} for chunk {chunk_index}/{total}"
                ) from exc

            actual_crc = _crc32_hex(chunk_bytes)
            if actual_crc != chunk_crc_hex:
                raise ValueError(
                    f"Chunk CRC mismatch for {chunk_index}/{total}: expected {chunk_crc_hex}, got {actual_crc}"
                )
            chunks[chunk_index] = chunk_bytes

        if footer_crc is None or footer_bytes is None:
            raise ValueError(f"Missing ENDSTREAM footer for stream {stream_id}")

        streams.append(
            _ParsedStream(
                stream_id=stream_id,
                total=total,
                stream_crc32_hex=footer_crc,
                total_bytes=footer_bytes,
                chunks=chunks,
            )
        )

    return streams


def _assemble_stream(parsed: _ParsedStream) -> bytes:
    if parsed.total == 0:
        stream_bytes = b""
    else:
        missing = [chunk_index for chunk_index in range(1, parsed.total + 1) if chunk_index not in parsed.chunks]
        if missing:
            preview = ",".join(str(value) for value in missing[:5])
            if len(missing) > 5:
                preview += ",..."
            raise ValueError(f"Missing chunk(s) for stream {parsed.stream_id}: {preview}")
        stream_bytes = b"".join(parsed.chunks[index] for index in range(1, parsed.total + 1))

    if len(stream_bytes) != parsed.total_bytes:
        raise ValueError(
            f"Byte count mismatch for stream {parsed.stream_id}: expected {parsed.total_bytes}, got {len(stream_bytes)}"
        )

    actual_stream_crc = _crc32_hex(stream_bytes)
    if actual_stream_crc != parsed.stream_crc32_hex:
        raise ValueError(
            f"Stream CRC mismatch for stream {parsed.stream_id}: expected {parsed.stream_crc32_hex}, got {actual_stream_crc}"
        )

    derived_stream_id = _stream_id_for_bytes(stream_bytes)
    if derived_stream_id != parsed.stream_id:
        raise ValueError(
            f"Stream ID mismatch for stream {parsed.stream_id}: derived {derived_stream_id} from reconstructed bytes"
        )
    return stream_bytes


def extract_stream_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for _, line in _normalize_lines(text):
        match = _HEADER_RE.match(line)
        if match is None:
            continue
        stream_id = match.group(1).lower()
        if stream_id not in seen:
            ids.append(stream_id)
            seen.add(stream_id)
    return ids


def decode_text_to_stream_by_id(text: str, stream_id: str | None = None) -> bytes:
    streams = _parse_stream_blocks(text)
    if not streams:
        raise ValueError("No HFUPS-VARA stream blocks found")

    wanted_id: str | None = None
    if stream_id is not None:
        cleaned = stream_id.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{8}", cleaned):
            raise ValueError("stream_id must be exactly 8 hex characters")
        wanted_id = cleaned

    errors: list[str] = []
    matching_count = 0
    for parsed in streams:
        if wanted_id is not None and parsed.stream_id != wanted_id:
            continue
        matching_count += 1
        try:
            return _assemble_stream(parsed)
        except ValueError as exc:
            errors.append(str(exc))
            if wanted_id is not None:
                break

    if wanted_id is not None and matching_count == 0:
        available = ", ".join(extract_stream_ids(text))
        if available:
            raise ValueError(f"Requested stream_id {wanted_id} not found. Available stream_ids: {available}")
        raise ValueError(f"Requested stream_id {wanted_id} not found")

    if errors:
        raise ValueError(errors[0])
    raise ValueError("No complete stream blocks found")


def decode_text_to_stream(text: str) -> bytes:
    """
    Parse chunks, validate CRCs, validate total, and return reconstructed stream bytes.
    Raise ValueError with clear message if invalid/corrupt/incomplete.
    """
    return decode_text_to_stream_by_id(text, stream_id=None)
