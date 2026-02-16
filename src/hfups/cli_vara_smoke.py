"""Simple VARA command-port smoke test."""

import socket
import time


def _send_cmd(sock: socket.socket, cmd: str) -> None:
    sock.sendall(cmd.encode("ascii", errors="replace") + b"\r")


def _drain_responses(sock: socket.socket, window_s: float) -> None:
    deadline = time.monotonic() + window_s
    while time.monotonic() < deadline:
        try:
            data = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            return
        if not data:
            continue
        print(data.decode("latin-1", errors="replace"))


def main() -> None:
    """Run a small VARA command-port probe sequence."""
    host = "127.0.0.1"
    port = 8300
    timeout_s = 0.5

    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.settimeout(timeout_s)

    try:
        _send_cmd(sock, "VERSION")
        _drain_responses(sock, 1.0)

        _send_cmd(sock, "MYCALL DEMO DEMO-T")
        _drain_responses(sock, 1.0)

        _send_cmd(sock, "LISTEN ON")
        _drain_responses(sock, 1.0)

        time.sleep(1.0)

        _send_cmd(sock, "LISTEN OFF")
        _drain_responses(sock, 1.0)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
