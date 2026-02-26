from __future__ import annotations

import socket
import time
from collections import deque
from collections.abc import Iterable

from hfups.framing import decode_frame, encode_frame
from hfups.streaming import FrameStreamDecoder
from hfups.transport.semantic_transport import ReceivedFrame, SemanticTransport


def pack_frame(payload: bytes) -> bytes:
    return encode_frame(payload)


def unpack_frames_from_stream(buffer: bytearray) -> list[tuple[bytes, bytes]]:
    """
    Decode complete framed packets from buffer.

    Returns list of (payload, raw_frame) and leaves incomplete tail bytes in buffer.
    """
    decoder = FrameStreamDecoder()
    frames = decoder.feed(bytes(buffer))
    if not frames:
        return []

    consumed = sum(len(frame) for frame in frames)
    del buffer[:consumed]

    out: list[tuple[bytes, bytes]] = []
    for raw in frames:
        try:
            payload = decode_frame(raw)
        except ValueError:
            continue
        out.append((payload, raw))
    return out


class _BaseTcpTransport(SemanticTransport):
    def __init__(self, timeout_s: float = 0.2) -> None:
        self._timeout_s = timeout_s
        self._buffer = bytearray()
        self._closed = False
        self._pending: deque[ReceivedFrame] = deque()

    def _recv_bytes(self, max_bytes: int = 4096) -> bytes:
        raise NotImplementedError

    def _send_bytes(self, data: bytes) -> None:
        raise NotImplementedError

    def _set_recv_timeout(self, timeout_s: float) -> None:
        raise NotImplementedError

    def send_payload(self, payload: bytes) -> None:
        if self._closed:
            raise RuntimeError("transport is closed")
        self._send_bytes(pack_frame(payload))

    def _drain_pending(self) -> Iterable[ReceivedFrame]:
        while self._pending:
            yield self._pending.popleft()

    def recv_payloads(self) -> Iterable[ReceivedFrame]:
        while not self._closed:
            yielded_any = False
            for item in self._drain_pending():
                yielded_any = True
                yield item

            try:
                data = self._recv_bytes()
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                break

            self._buffer.extend(data)
            decoded = unpack_frames_from_stream(self._buffer)
            for payload, raw in decoded:
                yielded_any = True
                yield ReceivedFrame(payload=payload, raw_frame=raw)

            if yielded_any:
                continue

        self.close()

    def recv_one(self, timeout_s: float) -> ReceivedFrame | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._closed:
            if self._pending:
                return self._pending.popleft()

            remaining = max(0.01, min(self._timeout_s, deadline - time.monotonic()))
            self._set_recv_timeout(remaining)
            try:
                data = self._recv_bytes()
            except socket.timeout:
                continue
            except OSError:
                self.close()
                return None

            if not data:
                self.close()
                return None

            self._buffer.extend(data)
            for payload, raw in unpack_frames_from_stream(self._buffer):
                self._pending.append(ReceivedFrame(payload=payload, raw_frame=raw))

            if self._pending:
                return self._pending.popleft()
        return None


class TcpServerTransport(_BaseTcpTransport):
    def __init__(self, host: str, port: int, timeout_s: float = 0.2) -> None:
        super().__init__(timeout_s=timeout_s)
        self._host = host
        self._port = port
        self._listen_sock: socket.socket | None = None
        self._conn: socket.socket | None = None

    def _ensure_connection(self) -> None:
        if self._conn is not None:
            return
        if self._listen_sock is None:
            listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_sock.bind((self._host, self._port))
            listen_sock.listen(1)
            listen_sock.settimeout(self._timeout_s)
            self._listen_sock = listen_sock

        while not self._closed and self._conn is None:
            try:
                conn, _ = self._listen_sock.accept()
                conn.settimeout(self._timeout_s)
                self._conn = conn
            except socket.timeout:
                continue

    def _recv_bytes(self, max_bytes: int = 4096) -> bytes:
        self._ensure_connection()
        if self._conn is None:
            return b""
        return self._conn.recv(max_bytes)

    def _send_bytes(self, data: bytes) -> None:
        self._ensure_connection()
        if self._conn is None:
            raise RuntimeError("no connected client")
        self._conn.sendall(data)

    def _set_recv_timeout(self, timeout_s: float) -> None:
        if self._conn is not None:
            self._conn.settimeout(timeout_s)
        elif self._listen_sock is not None:
            self._listen_sock.settimeout(timeout_s)

    def close(self) -> None:
        self._closed = True
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            finally:
                self._listen_sock = None


class TcpClientTransport(_BaseTcpTransport):
    def __init__(self, host: str, port: int, timeout_s: float = 0.2) -> None:
        super().__init__(timeout_s=timeout_s)
        self._host = host
        self._port = port
        self._sock = socket.create_connection((host, port), timeout=timeout_s)
        self._sock.settimeout(timeout_s)

    def _recv_bytes(self, max_bytes: int = 4096) -> bytes:
        return self._sock.recv(max_bytes)

    def _send_bytes(self, data: bytes) -> None:
        self._sock.sendall(data)

    def _set_recv_timeout(self, timeout_s: float) -> None:
        self._sock.settimeout(timeout_s)

    def close(self) -> None:
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass

