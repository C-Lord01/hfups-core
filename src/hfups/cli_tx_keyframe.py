from __future__ import annotations

import argparse
import time
from pathlib import Path

from hfups.integrations.transport_shim import TransportShim, write_bin
from hfups.integrations.tx_rx_keyframe import (
    PACKET_TYPE_ACK,
    PACKET_TYPE_CAPTION,
    PACKET_TYPE_DELTA,
    PACKET_TYPE_KEYFRAME,
    build_ack_payload,
    parse_ack_payload,
    wrap_envelope,
    wrap_typed_payload,
)
from hfups.transport.tcp_transport import TcpClientTransport, pack_frame
from hfups.vision.class_mapping import ClassMapper, resolve_mapping_path
from hfups.vision.delta_packet import DeltaEntry, DeltaPacket
from hfups.vision.keyframe_builder import KeyframeBuilder, make_tracker_assigner
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket, estimate_airtime_seconds
from hfups.vision.openimages_dict import (
    default_openimages_v7_dict_path,
    load_openimages_v7_boxable_dict,
)
from hfups.vision.tracker import SimpleIoUTracker
from hfups.vision.yolo_adapter import UltralyticsNotInstalledError, YoloRunner


def _parse_host_port(value: str) -> tuple[str, int]:
    try:
        host, port_text = value.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("TCP endpoint must be HOST:PORT") from exc
    host = host.strip()
    if not host:
        raise ValueError("TCP host must not be empty")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("TCP port must be an integer") from exc
    if not (1 <= port <= 65535):
        raise ValueError("TCP port must be in range [1, 65535]")
    return host, port


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transmit semantic keyframe packets using HFUPS framing.")
    parser.add_argument("--image", type=Path, help="Input image path")
    parser.add_argument("--images-dir", type=Path, help="Directory containing input images")
    parser.add_argument("--model", type=Path, help="YOLO model path")
    parser.add_argument("--dict", dest="dict_path", type=Path, default=default_openimages_v7_dict_path())
    parser.add_argument(
        "--mapping",
        dest="mapping_arg",
        default="preset:openimages",
        help="YOLO-to-OpenImages mapping path or preset:{openimages,coco}",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-objects", type=int, default=12)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--loop", type=int, default=1, help="Number of packets to transmit")
    parser.add_argument("--fps", type=float, default=1.0, help="Semantic frame rate for timing/airtime display")
    parser.add_argument(
        "--pack-type",
        choices=["keyframe", "delta", "caption"],
        default="keyframe",
        help="Semantic payload type to transmit",
    )
    parser.add_argument("--caption", default="A concise incident scene caption.")
    parser.add_argument("--out-bin", type=Path, help="Optional output path for framed bytestream")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock packets (no YOLO)")
    parser.add_argument(
        "--transport",
        choices=["shim", "file", "tcp"],
        default="shim",
        help="Transport backend",
    )
    parser.add_argument(
        "--tcp",
        dest="tcp_endpoint",
        help="TCP endpoint CONNECT_HOST:PORT (required for --transport tcp)",
    )
    parser.add_argument("--ack", action="store_true", help="Enable envelope ACK mode")
    parser.add_argument("--ack-timeout", type=float, default=0.8, help="ACK wait timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries after ACK timeout")
    parser.add_argument("--strict-ack", action="store_true", help="Exit non-zero if ACK fails after retries")
    return parser


def _iter_images(single: Path | None, folder: Path | None) -> list[Path]:
    if single and folder:
        raise ValueError("Use either --image or --images-dir, not both")
    if single:
        return [single]
    if folder:
        files = sorted([p for p in folder.iterdir() if p.is_file()])
        if not files:
            raise ValueError(f"No files found in --images-dir: {folder}")
        return files
    return []


def _mock_keyframe(step: int) -> KeyframePacket:
    return KeyframePacket(
        objects=[
            KeyframeObject(class_id=0, track_id=0, cx=min(7, 1 + step), cy=6, size=1, confidence=14),
            KeyframeObject(class_id=1, track_id=1, cx=4, cy=max(0, 5 - step), size=1, confidence=12),
            KeyframeObject(class_id=2, track_id=2, cx=6, cy=2, size=2, confidence=11),
        ]
    )


def _mock_delta(step: int) -> DeltaPacket:
    dx = 1 if step % 2 == 0 else 0
    dy = -1 if step % 2 == 1 else 0
    return DeltaPacket(entries=[DeltaEntry(track_id=0, dx=dx, dy=0), DeltaEntry(track_id=1, dx=0, dy=dy)])


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Install pillow: python -m pip install pillow") from exc

    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be > 0")
    return int(width), int(height)


def _build_real_keyframes(args: argparse.Namespace, count: int) -> list[bytes]:
    image_paths = _iter_images(args.image, args.images_dir)
    if not image_paths:
        raise ValueError("Provide --image or --images-dir unless --mock is used")
    if args.model is None:
        raise ValueError("--model is required when not using --mock")

    openimages = load_openimages_v7_boxable_dict(args.dict_path)
    mapper = ClassMapper(openimages, mapping_path=resolve_mapping_path(args.mapping_arg))
    tracker = SimpleIoUTracker()
    builder = KeyframeBuilder(
        class_mapper=mapper,
        grid=args.grid,
        max_objects=args.max_objects,
        confidence_threshold=args.conf,
        assign_track_ids=make_tracker_assigner(tracker),
    )
    runner = YoloRunner(args.model)

    payloads: list[bytes] = []
    for i in range(count):
        image_path = image_paths[i % len(image_paths)]
        detections = runner.detect(image_path, conf=args.conf)
        width, height = _image_size(image_path)
        packet = builder.build(detections, width, height)
        payloads.append(packet.encode())
    return payloads


def _build_payloads(args: argparse.Namespace) -> tuple[int, list[bytes]]:
    count = args.loop
    if count <= 0:
        raise ValueError("--loop must be > 0")

    if args.pack_type == "caption":
        body = args.caption.encode("utf-8")
        if len(body) > 160:
            raise ValueError("Caption UTF-8 payload must be <= 160 bytes")
        return PACKET_TYPE_CAPTION, [body for _ in range(count)]

    if args.pack_type == "delta":
        if args.mock:
            deltas = [_mock_delta(i).encode() for i in range(count)]
            return PACKET_TYPE_DELTA, deltas
        raise ValueError("Delta transmission currently requires --mock mode")

    if args.mock:
        keyframes = [_mock_keyframe(i).encode() for i in range(count)]
        return PACKET_TYPE_KEYFRAME, keyframes

    return PACKET_TYPE_KEYFRAME, _build_real_keyframes(args, count)


def _print_tx_line(
    index: int,
    packet_type: int,
    payload: bytes,
    framed: bytes,
    *,
    seq: int | None,
    acked: bool | None,
) -> None:
    preview = payload.hex()[:48]
    airtime = estimate_airtime_seconds(len(framed), kbps=10.0)
    ack_text = ""
    if seq is not None:
        if acked is True:
            ack_text = f" seq={seq} ack=ok"
        elif acked is False:
            ack_text = f" seq={seq} ack=timeout"
        else:
            ack_text = f" seq={seq}"
    print(
        f"TX[{index}] type=0x{packet_type:02x} payload={len(payload)}B "
        f"frame={len(framed)}B airtime={airtime:.4f}s{ack_text} hex={preview}"
    )


def _wait_for_ack(client: TcpClientTransport, seq: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        frame = client.recv_one(remaining)
        if frame is None:
            return False
        ack_seq = parse_ack_payload(frame.payload)
        if ack_seq is None:
            continue
        if ack_seq == seq:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.grid != 8:
        parser.error("--grid currently only supports 8")
    if args.max_objects < 1 or args.max_objects > 12:
        parser.error("--max-objects must be in range [1, 12]")
    if args.conf < 0.0 or args.conf > 1.0:
        parser.error("--conf must be in range [0.0, 1.0]")
    if args.fps <= 0.0:
        parser.error("--fps must be > 0")
    if args.ack_timeout <= 0:
        parser.error("--ack-timeout must be > 0")
    if args.retries < 0:
        parser.error("--retries must be >= 0")
    if args.transport == "tcp" and not args.tcp_endpoint:
        parser.error("--tcp is required for --transport tcp")
    if args.transport == "file" and not args.out_bin:
        parser.error("--out-bin is required for --transport file")

    try:
        packet_type, payloads = _build_payloads(args)
    except UltralyticsNotInstalledError as exc:
        print(str(exc))
        return 2
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    shim: TransportShim | None = None
    client: TcpClientTransport | None = None
    file_frames: list[bytes] = []
    seq = 0
    ack_failures = 0

    try:
        if args.transport == "shim":
            shim = TransportShim()
        elif args.transport == "tcp":
            host, port = _parse_host_port(args.tcp_endpoint)
            client = TcpClientTransport(host=host, port=port)

        for i, payload in enumerate(payloads):
            if args.ack:
                semantic_payload = wrap_envelope(seq, packet_type, payload)
                seq_for_line: int | None = seq
            else:
                semantic_payload = wrap_typed_payload(packet_type, payload)
                seq_for_line = None

            framed = pack_frame(semantic_payload)

            attempts = args.retries + 1
            acked: bool | None = None
            for attempt in range(attempts):
                if args.transport == "file":
                    file_frames.append(framed)
                elif args.transport == "shim":
                    assert shim is not None
                    shim.send_payload(semantic_payload)
                else:
                    assert client is not None
                    client.send_payload(semantic_payload)

                if not args.ack or args.transport != "tcp":
                    acked = None if not args.ack else True
                    break

                assert client is not None
                if _wait_for_ack(client, seq, args.ack_timeout):
                    acked = True
                    break

                acked = False
                if attempt < attempts - 1:
                    print(f"WARN: ACK timeout for seq={seq}, retry {attempt + 1}/{args.retries}")

            if args.ack and args.transport == "tcp" and acked is False:
                ack_failures += 1
                if args.strict_ack:
                    print(f"error: ACK failed for seq={seq} after {attempts} attempts")
                    return 1
                print(f"WARN: ACK failed for seq={seq} after {attempts} attempts; continuing")

            _print_tx_line(
                i,
                packet_type,
                payload,
                framed,
                seq=seq_for_line,
                acked=acked,
            )

            if args.ack:
                seq = (seq + 1) % 256

        if args.out_bin:
            if args.transport == "file":
                stream = b"".join(file_frames)
            elif args.transport == "shim":
                assert shim is not None
                stream = shim.sent_stream_bytes()
            else:
                stream = b""
            if stream:
                write_bin(args.out_bin, stream)
                print(f"Wrote framed stream: {args.out_bin}")

        if ack_failures and not args.strict_ack:
            print(f"ACK summary: {ack_failures} packets were not acknowledged.")
    finally:
        if client is not None:
            client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
