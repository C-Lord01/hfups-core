from __future__ import annotations

import argparse
import hashlib
import zlib
from pathlib import Path

from hfups.transport.vara_text_bridge import decode_text_to_stream_by_id, extract_stream_ids


def _stream_id_for_bytes(stream_bytes: bytes) -> str:
    return hashlib.sha256(stream_bytes).digest()[:4].hex()


def _crc32_hex(data: bytes) -> str:
    return f"{(zlib.crc32(data) & 0xFFFFFFFF):08x}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode VARA-friendly HFUPS ASCII text back into binary stream.")
    parser.add_argument("--in-txt", dest="in_txt", type=Path, required=True, help="Input text file")
    parser.add_argument("--out-bin", dest="out_bin", type=Path, required=True, help="Output binary file")
    parser.add_argument(
        "--stream-id",
        dest="stream_id",
        default=None,
        help="Optional stream_id (8 hex chars) when input text contains multiple streams",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        text = args.in_txt.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"error: failed to read input text file: {exc}")
        return 1

    try:
        stream_bytes = decode_text_to_stream_by_id(text, stream_id=args.stream_id)
    except ValueError as exc:
        print(f"error: {exc}")
        available = extract_stream_ids(text)
        if available:
            print(f"available stream_ids: {', '.join(available)}")
        return 1

    try:
        args.out_bin.parent.mkdir(parents=True, exist_ok=True)
        args.out_bin.write_bytes(stream_bytes)
    except Exception as exc:
        print(f"error: failed to write output binary file: {exc}")
        return 1

    stream_id = _stream_id_for_bytes(stream_bytes)
    stream_crc = _crc32_hex(stream_bytes)
    print(
        f"Decoded stream_id={stream_id} bytes_out={len(stream_bytes)} "
        f"crc32={stream_crc} verified=ok out={args.out_bin}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
