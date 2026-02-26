from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class ReceivedFrame:
    payload: bytes
    raw_frame: bytes


class SemanticTransport(Protocol):
    def send_payload(self, payload: bytes) -> None:
        ...

    def recv_payloads(self) -> Iterable[ReceivedFrame]:
        """
        Yield ReceivedFrame items as they arrive.

        Implementations should block until at least one frame is available,
        then continue yielding frames until the stream ends or close() is called.
        """

    def close(self) -> None:
        ...
