"""TCP link adapters for HFUPS."""

import socket


class TCPClientLink:
    """TCP client adapter that connects to a remote host/port."""

    def __init__(self, host: str, port: int, timeout_s: float = 0.5) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def open(self) -> None:
        """Open the TCP client connection."""
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
        self._sock.settimeout(self._timeout_s)

    def connect(self) -> None:
        """Compatibility alias for open()."""
        self.open()

    def close(self) -> None:
        """Close the active client socket."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send(self, data: bytes) -> None:
        """Send bytes over the client connection."""
        if self._sock is None:
            raise RuntimeError("TCP client is not connected")
        self._sock.sendall(data)

    def recv(self, max_bytes: int = 4096) -> bytes:
        """Receive bytes; returns b'' on timeout or peer disconnect."""
        if self._sock is None:
            raise RuntimeError("TCP client is not connected")
        try:
            return self._sock.recv(max_bytes)
        except (socket.timeout, ConnectionResetError, OSError):
            return b""


class TCPServerLink:
    """TCP server adapter that listens and accepts a single client."""

    def __init__(self, host: str, port: int, timeout_s: float = 0.5) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._listen_sock: socket.socket | None = None
        self._conn: socket.socket | None = None

    def open(self) -> None:
        """Bind/listen and accept exactly one client connection."""
        if self._conn is not None:
            return
        if self._listen_sock is None:
            listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_sock.bind((self._host, self._port))
            listen_sock.listen(1)
            listen_sock.settimeout(self._timeout_s)
            self._listen_sock = listen_sock

        while self._conn is None:
            try:
                conn, _ = self._listen_sock.accept()
                conn.settimeout(self._timeout_s)
                self._conn = conn
            except socket.timeout:
                continue

    def connect(self) -> None:
        """Compatibility alias for open()."""
        self.open()

    def close(self) -> None:
        """Close accepted connection and listening socket."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._listen_sock is not None:
            self._listen_sock.close()
            self._listen_sock = None

    def send(self, data: bytes) -> None:
        """Send bytes to the accepted client."""
        if self._conn is None:
            raise RuntimeError("TCP server has no accepted client")
        self._conn.sendall(data)

    def recv(self, max_bytes: int = 4096) -> bytes:
        """Receive bytes; returns b'' on timeout or peer disconnect."""
        if self._conn is None:
            raise RuntimeError("TCP server has no accepted client")
        try:
            return self._conn.recv(max_bytes)
        except (socket.timeout, ConnectionResetError, OSError):
            return b""
