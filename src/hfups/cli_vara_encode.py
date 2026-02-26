from __future__ import annotations

import argparse
import hashlib
import zlib
from pathlib import Path

from hfups.transport.vara_text_bridge import encode_stream_to_text


def _stream_id_for_bytes(stream_bytes: bytes) -> str:
    return hashlib.sha256(stream_bytes).digest()[:4].hex()


def _crc32_hex(data: bytes) -> str:
    return f"{(zlib.crc32(data) & 0xFFFFFFFF):08x}"


def _chunk_count(total_bytes: int, chunk_size: int) -> int:
    if total_bytes == 0:
        return 0
    return (total_bytes + chunk_size - 1) // chunk_size


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode HFUPS framed binary stream into VARA-friendly ASCII text.")
    parser.add_argument("--in-bin", dest="in_bin", type=Path, required=True, help="Input framed binary stream")
    parser.add_argument("--out-txt", dest="out_txt", type=Path, required=True, help="Output text file")
    parser.add_argument("--chunk-bytes", type=int, default=180, help="Raw bytes per chunk (default: 180)")
    parser.add_argument(
        "--wrap",
        type=int,
        default=None,
        help=(
            "Optional max base64 width. Keeps one payload line per chunk by reducing "
            "effective chunk bytes so encoded base64 stays within this width."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.chunk_bytes <= 0:
        parser.error("--chunk-bytes must be > 0")
    if args.wrap is not None and args.wrap <= 0:
        parser.error("--wrap must be > 0")

    try:
        stream_bytes = args.in_bin.read_bytes()
    except Exception as exc:
        print(f"error: failed to read input binary file: {exc}")
        return 1

    chunk_size = int(args.chunk_bytes)
    if args.wrap is not None:
        max_chunk_for_wrap = (args.wrap // 4) * 3
        if max_chunk_for_wrap <= 0:
            print("error: --wrap is too small to represent base64 payload lines")
            return 1
        chunk_size = min(chunk_size, max_chunk_for_wrap)

    try:
        text = encode_stream_to_text(stream_bytes, chunk_size=chunk_size)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    try:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(text, encoding="utf-8", newline="\n")
    except Exception as exc:
        print(f"error: failed to write output text file: {exc}")
        return 1

    stream_id = _stream_id_for_bytes(stream_bytes)
    stream_crc = _crc32_hex(stream_bytes)
    chunks = _chunk_count(len(stream_bytes), chunk_size)
    print(
        f"Encoded stream_id={stream_id} bytes_in={len(stream_bytes)} chunks={chunks} "
        f"chars={len(text)} crc32={stream_crc} out={args.out_txt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
