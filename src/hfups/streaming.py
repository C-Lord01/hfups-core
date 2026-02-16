"""Streaming helpers for splitting HFUPS delimited frames from byte streams."""


class FrameStreamDecoder:
    """Incrementally collect bytes and emit complete 0x00-delimited frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        """Add bytes and return any complete frames (each includes delimiter)."""
        if data:
            self._buffer.extend(data)

        frames: list[bytes] = []
        start = 0

        while True:
            try:
                end = self._buffer.index(0, start)
            except ValueError:
                break

            frames.append(bytes(self._buffer[start : end + 1]))
            start = end + 1

        if start:
            del self._buffer[:start]

        return frames


def iter_frames_from_stream(chunks: list[bytes]) -> list[bytes]:
    """Split arbitrary chunks into complete 0x00-delimited frames."""
    decoder = FrameStreamDecoder()
    frames: list[bytes] = []

    for chunk in chunks:
        frames.extend(decoder.feed(chunk))

    return frames
