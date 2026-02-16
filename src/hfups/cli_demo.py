"""Runnable end-to-end HFUPS demo using the current protocol stack."""

import argparse
import json

from hfups.framing import decode_frame, encode_frame
from hfups.link_sim import LinkSimConfig, simulate_link
from hfups.state import HFUPSState, apply_payload, state_to_dict
from hfups.streaming import FrameStreamDecoder


def run_demo_sim(cfg: LinkSimConfig) -> dict:
    """Run deterministic demo flow through an impaired simulated link."""
    iframe_payload = bytes.fromhex("0000400081")
    mf_payload = bytes.fromhex("C0A824")
    clip_payload = bytes.fromhex("C191")

    frames = [encode_frame(iframe_payload), encode_frame(mf_payload), encode_frame(clip_payload)]
    stream = b"".join(frames)
    chunks = simulate_link(stream, cfg)

    decoder = FrameStreamDecoder()
    state = HFUPSState()
    rejected_frames = 0
    frames_received = 0

    for chunk in chunks:
        out_frames = decoder.feed(chunk)
        frames_received += len(out_frames)
        for frame in out_frames:
            try:
                payload = decode_frame(frame)
                apply_payload(state, payload)
            except ValueError:
                rejected_frames += 1

    result = state_to_dict(state)
    result["bytes_tx"] = len(stream)
    result["bytes_rx"] = sum(len(c) for c in chunks)
    result["frames_sent"] = len(frames)
    result["frames_received"] = frames_received
    result["rejected_frames"] = rejected_frames
    return result


def run_demo() -> dict:
    """Run a deterministic end-to-end demo with no link impairment."""
    return run_demo_sim(LinkSimConfig())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HFUPS end-to-end demo")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Chunk drop probability")
    parser.add_argument("--flip-rate", type=float, default=0.0, help="Bit-flip probability per chunk")
    parser.add_argument("--max-chunk", type=int, default=64, help="Maximum chunk size")
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic RNG seed")
    return parser


def main() -> None:
    """Print demo result as JSON."""
    args = _build_parser().parse_args()
    cfg = LinkSimConfig(
        drop_rate=args.drop_rate,
        flip_rate=args.flip_rate,
        max_chunk=args.max_chunk,
        seed=args.seed,
    )
    print(json.dumps(run_demo_sim(cfg), sort_keys=True))


if __name__ == "__main__":
    main()
