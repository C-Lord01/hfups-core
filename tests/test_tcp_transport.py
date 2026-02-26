from __future__ import annotations

import socket
import threading
import time

from hfups.integrations.tx_rx_keyframe import (
    PACKET_TYPE_KEYFRAME,
    build_ack_payload,
    parse_ack_payload,
    unwrap_semantic_payload,
    wrap_envelope,
    wrap_typed_payload,
)
from hfups.transport.tcp_transport import TcpClientTransport, TcpServerTransport


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


def test_tcp_transport_roundtrip_single_payload() -> None:
    host = "127.0.0.1"
    port = _free_port()
    expected = b"\x01semantic-keyframe"
    received: list[bytes] = []

    def server_worker() -> None:
        transport = TcpServerTransport(host=host, port=port, timeout_s=0.05)
        try:
            for frame in transport.recv_payloads():
                received.append(frame.payload)
                if len(received) >= 1:
                    break
        finally:
            transport.close()

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    client = _connect_client_with_retry(host, port)
    try:
        client.send_payload(expected)
    finally:
        client.close()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert received == [expected]


def test_tcp_transport_roundtrip_multiple_payloads() -> None:
    host = "127.0.0.1"
    port = _free_port()
    payloads = [wrap_typed_payload(PACKET_TYPE_KEYFRAME, bytes([i, i + 1])) for i in range(10)]
    received: list[bytes] = []

    def server_worker() -> None:
        transport = TcpServerTransport(host=host, port=port, timeout_s=0.05)
        try:
            for frame in transport.recv_payloads():
                received.append(frame.payload)
                if len(received) >= len(payloads):
                    break
        finally:
            transport.close()

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    client = _connect_client_with_retry(host, port)
    try:
        for payload in payloads:
            client.send_payload(payload)
    finally:
        client.close()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert received == payloads


def test_tcp_transport_ack_mode_success() -> None:
    host = "127.0.0.1"
    port = _free_port()
    seq = 37
    payload = wrap_envelope(seq, PACKET_TYPE_KEYFRAME, b"\x10\x20")

    def server_worker() -> None:
        transport = TcpServerTransport(host=host, port=port, timeout_s=0.05)
        try:
            for frame in transport.recv_payloads():
                packet_type, body, rx_seq = unwrap_semantic_payload(frame.payload)
                assert packet_type == PACKET_TYPE_KEYFRAME
                assert body == b"\x10\x20"
                assert rx_seq == seq
                transport.send_payload(build_ack_payload(seq))
                break
        finally:
            transport.close()

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    client = _connect_client_with_retry(host, port)
    try:
        client.send_payload(payload)
        ack_frame = client.recv_one(1.0)
    finally:
        client.close()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert ack_frame is not None
    assert parse_ack_payload(ack_frame.payload) == seq


def test_tcp_transport_ack_timeout_no_raise() -> None:
    host = "127.0.0.1"
    port = _free_port()
    attempts = 3
    received = 0

    def server_worker() -> None:
        nonlocal received
        transport = TcpServerTransport(host=host, port=port, timeout_s=0.05)
        try:
            for frame in transport.recv_payloads():
                packet_type, body, seq = unwrap_semantic_payload(frame.payload)
                assert packet_type == PACKET_TYPE_KEYFRAME
                assert body == b"\x99"
                assert seq is not None
                received += 1
                if received >= attempts:
                    break
        finally:
            transport.close()

    thread = threading.Thread(target=server_worker, daemon=True)
    thread.start()

    client = _connect_client_with_retry(host, port)
    try:
        for seq in range(attempts):
            client.send_payload(wrap_envelope(seq, PACKET_TYPE_KEYFRAME, b"\x99"))
            # No ACK is sent by server in this test. Timeout path should be clean.
            assert client.recv_one(0.1) is None
    finally:
        client.close()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert received == attempts
