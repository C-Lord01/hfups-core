"""Runnable end-to-end HFUPS demo using the current protocol stack."""

import argparse
import json
import os

from hfups.nova.prompt_templates import build_nova_prompt
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket, estimate_airtime_seconds
from hfups.vision.openimages_dict import default_openimages_v7_dict_path, load_openimages_v7_boxable_dict


def run_demo_sim() -> dict:
    """Run a deterministic demo using the KeyframePacket pipeline."""
    # Build a deterministic 3-object packet using class_ids 0 (Person), 1 (Car), 2 (Dog)
    objects = [
        KeyframeObject(class_id=0, track_id=0, cx=3, cy=4, size=2, confidence=12),
        KeyframeObject(class_id=1, track_id=1, cx=6, cy=2, size=1, confidence=10),
        KeyframeObject(class_id=2, track_id=2, cx=1, cy=6, size=0, confidence=8),
    ]
    packet = KeyframePacket(objects=objects)

    # Roundtrip encode/decode
    encoded = packet.encode()
    KeyframePacket.decode(encoded)

    # Load dictionary and build prompt
    dict_path = default_openimages_v7_dict_path()
    openimages_dict = load_openimages_v7_boxable_dict(dict_path)
    prompt = build_nova_prompt(packet, openimages_dict, template="disaster_response")

    airtime = estimate_airtime_seconds(len(encoded), kbps=10.0)

    return {
        "encoded_bytes": len(encoded),
        "airtime_10kbps_seconds": airtime,
        "prompt": prompt,
        "object_count": len(objects),
        "template": "disaster_response",
    }


def run_demo() -> dict:
    """Run a deterministic end-to-end demo."""
    return run_demo_sim()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HFUPS end-to-end demo")
    parser.add_argument("--out", help="Optional output JSON file path (UTF-8)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON with indentation")
    return parser


def _format_json(result: dict, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(result, sort_keys=True, indent=2)
    return json.dumps(result, sort_keys=True)


def _write_json_atomic(path: str, result: dict, *, pretty: bool) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        if pretty:
            json.dump(result, f, sort_keys=True, indent=2)
        else:
            json.dump(result, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main(argv: list[str] | None = None) -> None:
    """Print demo result as JSON and optionally write to a UTF-8 file."""
    args = _build_parser().parse_args(argv)

    result = run_demo_sim()

    if args.out:
        _write_json_atomic(args.out, result, pretty=args.pretty)

    print(_format_json(result, pretty=args.pretty))


if __name__ == "__main__":
    main()
