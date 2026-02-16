"""Serial link adapter for HFUPS with lazy pyserial import."""


class SerialLink:
    """Small serial transport wrapper that imports pyserial on open()."""

    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 0.2) -> None:
        self._port = port
        self._baud = baud
        self._timeout_s = timeout_s
        self._serial = None

    def open(self) -> None:
        """Open the serial port."""
        if self._serial is not None:
            return
        import serial

        self._serial = serial.Serial(self._port, self._baud, timeout=self._timeout_s)

    def close(self) -> None:
        """Close the serial port if open."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def send(self, data: bytes) -> None:
        """Write bytes to the serial link."""
        if self._serial is None:
            raise RuntimeError("Serial link is not open")
        self._serial.write(data)

    def recv(self, max_bytes: int = 4096) -> bytes:
        """Read up to max_bytes from serial."""
        if self._serial is None:
            raise RuntimeError("Serial link is not open")
        return self._serial.read(max_bytes)
