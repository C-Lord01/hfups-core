import json
import socket
import threading
import time
from pathlib import Path

from hfups.integrations.transport_shim import TransportShim
from hfups.integrations.tx_rx_keyframe import (
    PACKET_TYPE_DELTA,
    PACKET_TYPE_KEYFRAME,
    rx_loop,
    unwrap_semantic_payload,
    tx_keyframe_packet,
)
from hfups.transport.tcp_transport import TcpClientTransport, TcpServerTransport
from hfups.vision.class_mapping import ClassMapper
from hfups.vision.delta_packet import DeltaEntry, DeltaPacket
from hfups.vision.keyframe_builder import KeyframeBuilder
from hfups.vision.keyframe_packet import KeyframePacket
from hfups.vision.openimages_dict import OpenImagesClass, OpenImagesDict
from hfups.vision.yolo_adapter import Detection


def _openimages_dict() -> OpenImagesDict:
    classes = [
        OpenImagesClass(id=0, label="/m/person", name="Person"),
        OpenImagesClass(id=1, label="/m/car", name="Car"),
        OpenImagesClass(id=2, label="/m/dog", name="Dog"),
    ]
    return OpenImagesDict(
        classes=classes,
        by_id={c.id: c for c in classes},
        by_label={c.label: c for c in classes},
    )


def _mapping_file(tmp_path: Path) -> Path:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"person": "/m/person", "car": "/m/car", "dog": "/m/dog"}) + "\n",
        encoding="utf-8",
    )
    return mapping_path


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _connect_client_with_retry(host: str, port: int, timeout_s: float = 2.0) -> TcpClientTransport:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return TcpClientTransport(host=host, port=port, timeout_s=0.05)
        except OSError:
            time.sleep(0.02)
    raise RuntimeError("timed out waiting for test TCP server")


def test_tx_rx_integration_for_keyframe_and_delta(tmp_path: Path) -> None:
    mapper = ClassMapper(_openimages_dict(), mapping_path=_mapping_file(tmp_path))
    builder = KeyframeBuilder(mapper)

    detections = [
        Detection("person", 0.9, 20, 20, 140, 260),
        Detection("car", 0.85, 280, 300, 520, 500),
    ]
    keyframe = builder.build(detections, image_width=640, image_height=480)
    delta = DeltaPacket(entries=[DeltaEntry(track_id=0, dx=1, dy=0)])

    transport = TransportShim()
    tx_keyframe_packet(keyframe.encode(), transport, packet_type=PACKET_TYPE_KEYFRAME)
    tx_keyframe_packet(delta.encode(), transport, packet_type=PACKET_TYPE_DELTA)

    received: list[tuple[int, bytes]] = []
    count = rx_loop(transport, lambda packet_type, payload: received.append((packet_type, payload)))

    assert count == 2
    assert received[0][0] == PACKET_TYPE_KEYFRAME
    assert received[1][0] == PACKET_TYPE_DELTA

    decoded_keyframe = KeyframePacket.decode(received[0][1])
    decoded_delta = DeltaPacket.decode(received[1][1])

    assert decoded_keyframe == keyframe
    assert decoded_delta == delta


def test_tx_rx_tcp_integration_keyframe(tmp_path: Path) -> None:
    mapper = ClassMapper(_openimages_dict(), mapping_path=_mapping_file(tmp_path))
    builder = KeyframeBuilder(mapper)

    detections = [
        Detection("person", 0.9, 20, 20, 140, 260),
        Detection("car", 0.85, 280, 300, 520, 500),
    ]
    keyframe = builder.build(detections, image_width=640, image_height=480)

    host = "127.0.0.1"
    port = _free_port()
    received_payloads: list[bytes] = []

    def server_worker() -> None:
        transport = TcpServerTransport(host=host, port=port, timeout_s=0.05)
        try:
            for frame in transport.recv_payloads():
                received_payloads.append(frame.payload)
                if received_payloads:
                    break
        finally:
            transport.close()

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    client = _connect_client_with_retry(host, port)
    try:
        tx_keyframe_packet(keyframe.encode(), client, packet_type=PACKET_TYPE_KEYFRAME)
    finally:
        client.close()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(received_payloads) == 1
    packet_type, body, seq = unwrap_semantic_payload(received_payloads[0])
    assert packet_type == PACKET_TYPE_KEYFRAME
    assert seq is None
    assert body == keyframe.encode()

    decoded_keyframe = KeyframePacket.decode(body)
    assert decoded_keyframe == keyframe
