from __future__ import annotations

import argparse
from pathlib import Path

from hfups.framing import decode_frame
from hfups.integrations.transport_shim import TransportShim, read_bin
from hfups.integrations.tx_rx_keyframe import (
    PACKET_TYPE_ACK,
    PACKET_TYPE_CAPTION,
    PACKET_TYPE_DELTA,
    PACKET_TYPE_KEYFRAME,
    build_ack_payload,
    unwrap_semantic_payload,
)
from hfups.nova.prompt_builder import apply_delta_packet
from hfups.nova.prompt_templates import build_nova_prompt
from hfups.transport.semantic_transport import ReceivedFrame
from hfups.transport.tcp_transport import TcpServerTransport
from hfups.vision.delta_packet import DeltaPacket
from hfups.vision.keyframe_packet import KeyframePacket
from hfups.vision.openimages_dict import (
    default_openimages_v7_dict_path,
    load_openimages_v7_boxable_dict,
)


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
    parser = argparse.ArgumentParser(description="Receive framed semantic packets and render Nova prompts.")
    parser.add_argument(
        "--transport",
        choices=["shim", "file", "tcp"],
        default="shim",
        help="Transport backend",
    )
    parser.add_argument("--tcp", dest="tcp_endpoint", help="TCP endpoint LISTEN_HOST:PORT for --transport tcp")
    parser.add_argument("--in-bin", dest="in_bin", type=Path, help="Framed bytestream input file for file/shim mode")
    parser.add_argument("--dict", dest="dict_path", type=Path, default=default_openimages_v7_dict_path())
    parser.add_argument("--out-prompt", type=Path, help="Optional output text file for prompts")
    parser.add_argument("--playback", action="store_true", help="Print timestamped semantic playback prompts")
    parser.add_argument("--fps", type=float, default=1.0, help="Playback keyframe rate")
    parser.add_argument("--ack", action="store_true", help="Enable ACK responses for enveloped TCP payloads")
    return parser


def _render_keyframe_objects(packet: KeyframePacket, names_by_id: dict[int, str]) -> list[str]:
    lines = ["#  track  class_id  name        cx cy size conf"]
    for idx, obj in enumerate(packet.objects):
        name = names_by_id.get(obj.class_id, f"class_{obj.class_id}")
        lines.append(
            f"{idx:<2} {obj.track_id:<6} {obj.class_id:<8} {name[:10]:<10} "
            f"{obj.cx:<2} {obj.cy:<2} {obj.size:<4} {obj.confidence:<4}"
        )
    return lines


def _iter_file_payloads(stream: bytes) -> list[ReceivedFrame]:
    shim = TransportShim()
    shim.load_framed_stream(stream)

    received: list[ReceivedFrame] = []
    while True:
        raw_frame = shim.recv_framed()
        if raw_frame is None:
            break
        try:
            payload = decode_frame(raw_frame)
        except ValueError:
            continue
        received.append(ReceivedFrame(payload=payload, raw_frame=raw_frame))
    return received


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.transport == "tcp" and not args.tcp_endpoint:
        parser.error("--tcp is required for --transport tcp")
    if args.transport in {"file", "shim"} and args.in_bin is None:
        parser.error("--in-bin is required for --transport file and --transport shim")
    if args.ack and args.transport != "tcp":
        parser.error("--ack is only supported with --transport tcp")

    try:
        openimages = load_openimages_v7_boxable_dict(args.dict_path)
    except Exception as exc:
        print(f"error: failed to load dict: {exc}")
        return 1

    names_by_id = {cls.id: cls.name for cls in openimages.classes}

    prompts_out: list[str] = []
    playback_lines: list[str] = []
    pending_caption: str | None = None
    current_keyframe: KeyframePacket | None = None
    next_keyframe_time = 0.0
    keyframe_dt = 1.0 / args.fps
    delta_dt = keyframe_dt / 2.0
    last_keyframe_time = 0.0

    tcp_transport: TcpServerTransport | None = None
    received_any = False
    if args.transport == "tcp":
        try:
            host, port = _parse_host_port(args.tcp_endpoint)
        except ValueError as exc:
            print(f"error: {exc}")
            return 1
        tcp_transport = TcpServerTransport(host=host, port=port)
        print(f"Listening on {host}:{port}")
        event_source = tcp_transport.recv_payloads()
    else:
        try:
            stream = read_bin(args.in_bin)
        except Exception as exc:
            print(f"error: failed to read stream file: {exc}")
            return 1
        event_source = _iter_file_payloads(stream)

    try:
        for idx, frame in enumerate(event_source):
            received_any = True
            try:
                packet_type, body, seq = unwrap_semantic_payload(frame.payload)
            except ValueError as exc:
                print(f"RX[{idx}] payload decode error: {exc}")
                continue

            if seq is not None and args.ack and packet_type != PACKET_TYPE_ACK and tcp_transport is not None:
                tcp_transport.send_payload(build_ack_payload(seq))

            if packet_type == PACKET_TYPE_ACK:
                if body:
                    print(f"RX[{idx}] ACK payload ignored ({len(body)}B body)")
                else:
                    print(f"RX[{idx}] ACK received")
                continue

            if packet_type == PACKET_TYPE_CAPTION:
                pending_caption = body.decode("utf-8", errors="replace")
                print(f"RX[{idx}] Caption ({len(body)}B): {pending_caption}")
                continue

            if packet_type == PACKET_TYPE_KEYFRAME:
                try:
                    keyframe = KeyframePacket.decode(body)
                except Exception as exc:
                    print(f"RX[{idx}] Keyframe decode error: {exc}")
                    continue

                caption = pending_caption
                pending_caption = None

                prompt = build_nova_prompt(keyframe, openimages, caption=caption)
                current_keyframe = keyframe

                print(f"RX[{idx}] Keyframe ({len(body)}B) objects={len(keyframe.objects)}")
                for line in _render_keyframe_objects(keyframe, names_by_id):
                    print(line)
                print(f"Prompt: {prompt}")

                prompts_out.append(prompt)
                last_keyframe_time = next_keyframe_time
                if args.playback:
                    playback_lines.append(f"t={last_keyframe_time:.1f}s {prompt}")
                next_keyframe_time += keyframe_dt
                continue

            if packet_type == PACKET_TYPE_DELTA:
                try:
                    delta = DeltaPacket.decode(body)
                except Exception as exc:
                    print(f"RX[{idx}] Delta decode error: {exc}")
                    continue

                if current_keyframe is None:
                    print(f"RX[{idx}] Delta received before keyframe; ignored.")
                    continue

                current_keyframe = apply_delta_packet(current_keyframe, delta)
                prompt = build_nova_prompt(
                    current_keyframe,
                    openimages,
                    deltas=[(e.track_id, e.dx, e.dy) for e in delta.entries],
                )
                print(f"RX[{idx}] Delta ({len(body)}B) entries={len(delta.entries)}")
                print(f"Prompt: {prompt}")
                prompts_out.append(prompt)
                if args.playback:
                    playback_lines.append(f"t={last_keyframe_time + delta_dt:.1f}s {prompt}")
                continue

            print(f"RX[{idx}] Unknown packet type 0x{packet_type:02x} ({len(body)}B)")
    finally:
        if tcp_transport is not None:
            tcp_transport.close()

    if not received_any:
        print("No packets received.")
        return 0

    if args.playback and playback_lines:
        print("Playback:")
        for line in playback_lines:
            print(line)

    if args.out_prompt:
        args.out_prompt.parent.mkdir(parents=True, exist_ok=True)
        args.out_prompt.write_text("\n".join(prompts_out) + "\n", encoding="utf-8")
        print(f"Wrote prompts: {args.out_prompt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
