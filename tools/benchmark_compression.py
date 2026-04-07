"""Benchmark HFUPS compression ratio vs JPEG and raw image size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Install pillow: python -m pip install pillow") from exc

    with Image.open(image_path) as img:
        return int(img.width), int(img.height)


def _run_pipeline(
    image_path: Path,
    model_path: Path,
    conf: float,
    max_objects: int,
    mapping_arg: str,
    dict_path: Path | None,
) -> tuple[bytes, int]:
    """Run YOLO → KeyframePacket encode, return (encoded_bytes, object_count)."""
    from hfups.vision.class_mapping import ClassMapper, resolve_mapping_path
    from hfups.vision.keyframe_builder import KeyframeBuilder, make_tracker_assigner
    from hfups.vision.openimages_dict import (
        default_openimages_v7_dict_path,
        load_openimages_v7_boxable_dict,
    )
    from hfups.vision.tracker import SimpleIoUTracker
    from hfups.vision.yolo_adapter import UltralyticsNotInstalledError, YoloRunner

    resolved_dict_path = dict_path or default_openimages_v7_dict_path()
    openimages_dict = load_openimages_v7_boxable_dict(resolved_dict_path)
    mapping_path = resolve_mapping_path(mapping_arg)
    mapper = ClassMapper(openimages_dict, mapping_path=mapping_path)

    runner = YoloRunner(model_path)
    try:
        detections = runner.detect(image_path, conf=conf)
    except UltralyticsNotInstalledError as exc:
        raise SystemExit(str(exc)) from exc

    tracker = SimpleIoUTracker()
    builder = KeyframeBuilder(
        class_mapper=mapper,
        grid=8,
        max_objects=max_objects,
        confidence_threshold=conf,
        assign_track_ids=make_tracker_assigner(tracker),
    )

    width, height = _get_image_dimensions(image_path)
    packet, _ = builder.build_with_stats(detections, image_width=width, image_height=height)
    encoded = packet.encode()
    return encoded, len(packet.objects)


def _print_table(rows: list[tuple[str, str]]) -> None:
    label_w = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"  {label:<{label_w}}  {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark HFUPS compression ratio for a single image."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--model", required=True, type=Path, help="YOLO model path (.pt)")
    parser.add_argument(
        "--dict",
        dest="dict_path",
        type=Path,
        default=None,
        help="Open Images dict JSON path (default: bundled)",
    )
    parser.add_argument(
        "--mapping",
        dest="mapping_arg",
        default="preset:openimages",
        help="YOLO-to-OpenImages mapping (default: preset:openimages)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=12,
        help="Maximum objects in packet (default: 12)",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 1

    input_size_bytes = args.image.stat().st_size
    width, height = _get_image_dimensions(args.image)
    raw_size_bytes = width * height * 3

    print(f"\nRunning pipeline on {args.image.name} ...")
    encoded, object_count = _run_pipeline(
        image_path=args.image,
        model_path=args.model,
        conf=args.conf,
        max_objects=args.max_objects,
        mapping_arg=args.mapping_arg,
        dict_path=args.dict_path,
    )
    packet_size = len(encoded)

    ratio_vs_jpeg = input_size_bytes / packet_size if packet_size > 0 else float("inf")
    ratio_vs_raw = raw_size_bytes / packet_size if packet_size > 0 else float("inf")

    results = {
        "image": str(args.image),
        "model": str(args.model),
        "input_file_size_bytes": input_size_bytes,
        "input_file_size_kb": round(input_size_bytes / 1024, 2),
        "dimensions": {"width": width, "height": height},
        "raw_size_bytes": raw_size_bytes,
        "packet_size_bytes": packet_size,
        "object_count": object_count,
        "compression_ratio_vs_jpeg": round(ratio_vs_jpeg, 2),
        "compression_ratio_vs_raw": round(ratio_vs_raw, 2),
    }

    # Save JSON
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    stem = args.image.stem
    out_path = outputs_dir / f"benchmark_{stem}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print summary table
    print("\n--- Benchmark Results ---")
    _print_table([
        ("Input file",           args.image.name),
        ("Input size",           f"{input_size_bytes:,} bytes  ({results['input_file_size_kb']} KB)"),
        ("Dimensions",           f"{width} x {height} px"),
        ("Estimated raw size",   f"{raw_size_bytes:,} bytes  ({raw_size_bytes / 1024:.1f} KB)"),
        ("Packet size",          f"{packet_size} bytes"),
        ("Object count",         str(object_count)),
        ("Ratio vs JPEG",        f"{ratio_vs_jpeg:.1f}x"),
        ("Ratio vs raw",         f"{ratio_vs_raw:.1f}x"),
        ("Results saved to",     str(out_path)),
    ])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
