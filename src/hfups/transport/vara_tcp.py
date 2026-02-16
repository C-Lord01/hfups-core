"""VARA TCP adapter using separate command and data sockets."""

import socket


class VARATCPLink:
    """Connect to VARA command/data ports and expose HFUPS link methods."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        command_port: int = 8300,
        data_port: int = 8301,
        timeout_s: float = 0.5,
    ) -> None:
        self._host = host
        self._command_port = command_port
        self._data_port = data_port
        self._timeout_s = timeout_s
        self._cmd_sock: socket.socket | None = None
        self._data_sock: socket.socket | None = None

    def open(self) -> None:
        """Open command and data TCP sockets."""
        if self._cmd_sock is not None or self._data_sock is not None:
            return

        cmd_sock = socket.create_connection((self._host, self._command_port), timeout=self._timeout_s)
        data_sock = socket.create_connection((self._host, self._data_port), timeout=self._timeout_s)
        cmd_sock.settimeout(self._timeout_s)
        data_sock.settimeout(self._timeout_s)
        self._cmd_sock = cmd_sock
        self._data_sock = data_sock

    def close(self) -> None:
        """Close command and data sockets."""
        if self._data_sock is not None:
            self._data_sock.close()
            self._data_sock = None
        if self._cmd_sock is not None:
            self._cmd_sock.close()
            self._cmd_sock = None

    def send(self, data: bytes) -> None:
        """Send bytes over the VARA DATA socket."""
        if self._data_sock is None:
            raise RuntimeError("VARA data socket is not open")
        self._data_sock.sendall(data)

    def recv(self, max_bytes: int = 4096) -> bytes:
        """Receive bytes from VARA DATA socket; return b'' on timeout."""
        if self._data_sock is None:
            raise RuntimeError("VARA data socket is not open")
        try:
            return self._data_sock.recv(max_bytes)
        except (socket.timeout, ConnectionResetError, OSError):
            return b""

    def send_cmd(self, line: str) -> None:
        """Send one command line to VARA COMMAND socket, terminated by CR."""
        if self._cmd_sock is None:
            raise RuntimeError("VARA command socket is not open")
        payload = line.rstrip("\r\n").encode("ascii", errors="replace") + b"\r"
        self._cmd_sock.sendall(payload)

    def recv_cmd(self, max_bytes: int = 4096) -> str:
        """Receive command response text from COMMAND socket; empty on timeout."""
        if self._cmd_sock is None:
            raise RuntimeError("VARA command socket is not open")
        try:
            data = self._cmd_sock.recv(max_bytes)
        except (socket.timeout, ConnectionResetError, OSError):
            return ""
        return data.decode("latin-1", errors="replace")
