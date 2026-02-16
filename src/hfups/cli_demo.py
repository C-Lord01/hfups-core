"""Runnable end-to-end HFUPS demo using the current protocol stack."""

import json

from hfups.framing import decode_frame, encode_frame
from hfups.state import HFUPSState, apply_payload, state_to_dict
from hfups.streaming import FrameStreamDecoder


def _chunk_stream(data: bytes, sizes: list[int]) -> list[bytes]:
    chunks: list[bytes] = []
    index = 0
    size_index = 0

    while index < len(data):
        size = sizes[size_index % len(sizes)]
        chunks.append(data[index : index + size])
        index += size
        size_index += 1

    return chunks


def run_demo() -> dict:
    """Run a deterministic end-to-end framing/decoding/state-update demo."""
    iframe_payload = bytes.fromhex("0000400081")
    mf_payload = bytes.fromhex("C0A824")
    clip_payload = bytes.fromhex("C191")

    frames = [encode_frame(iframe_payload), encode_frame(mf_payload), encode_frame(clip_payload)]
    stream = b"".join(frames)

    chunk_sizes = [1, 2, 5, 3, 8, 13, 21, 34]
    chunks = _chunk_stream(stream, chunk_sizes)

    decoder = FrameStreamDecoder()
    state = HFUPSState()
    rejected_frames = 0

    for chunk in chunks:
        for frame in decoder.feed(chunk):
            try:
                payload = decode_frame(frame)
                apply_payload(state, payload)
            except ValueError:
                rejected_frames += 1

    result = state_to_dict(state)
    result["rejected_frames"] = rejected_frames
    return result


def main() -> None:
    """Print demo result as JSON."""
    print(json.dumps(run_demo(), sort_keys=True))


if __name__ == "__main__":
    main()
