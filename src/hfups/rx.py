"""HFUPS receiver loop utilities."""

import time

from hfups.framing import decode_frame
from hfups.state import HFUPSState, apply_payload, state_to_dict
from hfups.streaming import FrameStreamDecoder


def run_rx(link, *, max_seconds: float = 10.0) -> dict:
    """Read from a link, decode frames, and return final state summary."""
    deadline = time.monotonic() + max_seconds
    decoder = FrameStreamDecoder()
    state = HFUPSState()
    rejected_frames = 0

    while time.monotonic() < deadline:
        chunk = link.recv()
        if not chunk:
            continue

        for frame in decoder.feed(chunk):
            try:
                payload = decode_frame(frame)
                apply_payload(state, payload)
            except ValueError:
                rejected_frames += 1

    result = state_to_dict(state)
    result["rejected_frames"] = rejected_frames
    return result
