from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue

from hfups.framing import decode_frame, encode_frame
from hfups.streaming import iter_frames_from_stream


class TransportShim:
    """
    In-process transport adapter that reuses HFUPS framing (COBS + CRC + 0x00).

    This shim is intentionally simple and synchronous for deterministic tests:
    payload bytes -> framed bytes -> queue -> framed bytes -> payload bytes.
    """

    def __init__(self) -> None:
        self._frames: Queue[bytes] = Queue()
        self._sent_frames: list[bytes] = []

    def send_payload(self, payload: bytes) -> bytes:
        frame = encode_frame(payload)
        self._frames.put(frame)
        self._sent_frames.append(frame)
        return frame

    def send_framed(self, frame: bytes) -> None:
        self._frames.put(frame)
        self._sent_frames.append(frame)

    def recv_payload(self) -> bytes | None:
        try:
            frame = self._frames.get_nowait()
        except Empty:
            return None
        return decode_frame(frame)

    def recv_framed(self) -> bytes | None:
        try:
            return self._frames.get_nowait()
        except Empty:
            return None

    def load_framed_stream(self, stream: bytes) -> None:
        for frame in iter_frames_from_stream([stream]):
            self._frames.put(frame)

    def sent_stream_bytes(self) -> bytes:
        return b"".join(self._sent_frames)


def write_bin(path: str | Path, data: bytes) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def read_bin(path: str | Path) -> bytes:
    return Path(path).read_bytes()
