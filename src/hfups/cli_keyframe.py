from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hfups.nova.prompt_templates import build_nova_prompt
from hfups.vision.class_mapping import ClassMapper, resolve_mapping_path
from hfups.vision.keyframe_builder import KeyframeBuilder, make_tracker_assigner
from hfups.vision.keyframe_packet import KeyframePacket, estimate_airtime_seconds
from hfups.vision.openimages_dict import (
    default_openimages_v7_dict_path,
    load_openimages_v7_boxable_dict,
)
from hfups.vision.tracker import SimpleIoUTracker
from hfups.vision.yolo_adapter import Detection, UltralyticsNotInstalledError, YoloRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO detections through HFUPS keyframe packet encoding."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--model", required=True, type=Path, help="YOLO model path")
    parser.add_argument(
        "--dict",
        dest="dict_path",
        type=Path,
        default=default_openimages_v7_dict_path(),
        help="Open Images dict JSON path",
    )
    parser.add_argument(
        "--mapping",
        dest="mapping_arg",
        default="preset:openimages",
        help="YOLO-to-OpenImages mapping path or preset:{openimages,coco}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (try --conf 0.10 for demos)",
    )
    parser.add_argument("--max-objects", type=int, default=12, help="Maximum selected objects")
    parser.add_argument("--grid", type=int, default=8, help="Grid size (only 8 supported)")
    parser.add_argument("--out-bin", type=Path, help="Optional output path for raw packet bytes")
    parser.add_argument("--out-hex", type=Path, help="Optional output path for lowercase hex string")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print compact JSON summary to stdout",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Decode encoded bytes and assert equality",
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Print top raw detections (mapped or unmapped)",
    )
    parser.add_argument(
        "--debug-mapping",
        action="store_true",
        help="Print unique unmapped class names",
    )
    parser.add_argument(
        "--debug-top",
        type=int,
        default=5,
        help="Number of raw detections to show in debug output",
    )
    parser.add_argument(
        "--nova-template",
        choices=["concise", "descriptive", "disaster_response", "cinematic"],
        default="descriptive",
        help="Nova prompt template style",
    )
    parser.add_argument("--out-prompt", type=Path, help="Optional output path for Nova prompt text")
    return parser


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Install pillow: python -m pip install pillow") from exc

    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be > 0")
    return int(width), int(height)


def _sorted_top_raw_detections(detections: list[Detection], top_n: int) -> list[dict]:
    if top_n <= 0:
        return []
    ordered = sorted(
        detections,
        key=lambda det: (
            -det.confidence,
            det.class_name.lower(),
            det.x1,
            det.y1,
            det.x2,
            det.y2,
        ),
    )[:top_n]
    return [
        {
            "name": det.class_name,
            "conf": float(det.confidence),
            "bbox": [float(det.x1), float(det.y1), float(det.x2), float(det.y2)],
        }
        for det in ordered
    ]


def _sorted_unmapped_class_names(detections: list[Detection], mapper: ClassMapper) -> list[str]:
    names = {
        det.class_name.strip()
        for det in detections
        if mapper.map_name_to_class_id(det.class_name) is None
    }
    return sorted(names, key=lambda value: (value.lower(), value))


def _format_human_output(
    *,
    image_path: Path,
    model_path: Path,
    image_width: int,
    image_height: int,
    total_detections: int,
    mapped_detections: int,
    selected_objects: int,
    encoded_len: int,
    airtime_s: float,
    model_info: dict | None,
    object_rows: list[dict[str, int | str]],
    hex_string: str,
    nova_prompt: str | None,
    top_raw_detections: list[dict] | None,
    unmapped_class_names: list[str] | None,
) -> str:
    lines = [
        f"Image: {image_path}",
        f"Model: {model_path}",
        f"Image size: {image_width}x{image_height}",
        f"Detections: {total_detections}",
        f"Mapped: {mapped_detections}",
        f"Selected: {selected_objects}",
        f"Encoded bytes: {encoded_len}",
        f"Airtime @10kbps: {airtime_s:.4f} sec",
    ]
    if model_info is not None:
        class_count = model_info.get("class_count")
        sample = model_info.get("names_sample", [])
        lines.append(f"Model classes: {class_count}")
        if sample:
            lines.append(f"Model names sample: {', '.join(str(name) for name in sample[:5])}")

    if total_detections > 0 and mapped_detections == 0:
        lines.append("No detections mapped. Check mapping file or use --mapping preset:coco")

    lines.extend([
        "",
        "Objects:",
        "#  track  class_id  name        cx cy size conf",
    ])
    for idx, row in enumerate(object_rows):
        lines.append(
            f"{idx:<2} {row['track_id']:<6} {row['class_id']:<8} "
            f"{str(row['name'])[:10]:<10} {row['cx']:<2} {row['cy']:<2} {row['size']:<4} {row['confidence']:<4}"
        )

    if top_raw_detections is not None:
        lines.extend(["", "Top Raw Detections:"])
        if top_raw_detections:
            for idx, item in enumerate(top_raw_detections):
                bbox = item["bbox"]
                lines.append(
                    f"{idx}. {item['name']} conf={item['conf']:.3f} "
                    f"bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]"
                )
        else:
            lines.append("(none)")

    if unmapped_class_names is not None:
        lines.extend(["", "Unmapped Class Names:"])
        if unmapped_class_names:
            lines.extend(unmapped_class_names)
        else:
            lines.append("(none)")

    if nova_prompt:
        lines.extend(["", "Nova Prompt:", nova_prompt])
    lines.extend(["Hex:", hex_string])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.grid != 8:
        parser.error("--grid currently only supports value 8")
    if args.max_objects < 1 or args.max_objects > 12:
        parser.error("--max-objects must be in range [1, 12]")
    if args.conf < 0.0 or args.conf > 1.0:
        parser.error("--conf must be in range [0.0, 1.0]")
    if args.debug_top < 0:
        parser.error("--debug-top must be >= 0")

    try:
        mapping_path = resolve_mapping_path(args.mapping_arg)
        openimages_dict = load_openimages_v7_boxable_dict(args.dict_path)
        mapper = ClassMapper(openimages_dict, mapping_path=mapping_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    runner = YoloRunner(args.model)
    model_info: dict | None = None
    try:
        model_info = runner.model_info()
        detections = runner.detect(args.image, conf=args.conf)
    except UltralyticsNotInstalledError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: YOLO inference failed: {exc}", file=sys.stderr)
        return 1

    try:
        image_width, image_height = _image_size(args.image)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: failed to read image: {exc}", file=sys.stderr)
        return 1

    tracker = SimpleIoUTracker()
    builder = KeyframeBuilder(
        class_mapper=mapper,
        grid=args.grid,
        max_objects=args.max_objects,
        confidence_threshold=args.conf,
        assign_track_ids=make_tracker_assigner(tracker),
    )

    try:
        packet, stats = builder.build_with_stats(detections, image_width=image_width, image_height=image_height)
        encoded = packet.encode()
    except Exception as exc:
        print(f"error: failed to build packet: {exc}", file=sys.stderr)
        return 1

    hex_string = encoded.hex()
    if args.out_bin:
        args.out_bin.parent.mkdir(parents=True, exist_ok=True)
        args.out_bin.write_bytes(encoded)
    if args.out_hex:
        args.out_hex.parent.mkdir(parents=True, exist_ok=True)
        args.out_hex.write_text(hex_string, encoding="utf-8")

    if args.self_check:
        try:
            decoded = KeyframePacket.decode(encoded)
        except Exception as exc:
            print(f"error: self-check decode failed: {exc}", file=sys.stderr)
            return 1
        if decoded != packet:
            print("error: self-check failed: decoded packet does not match encoded packet", file=sys.stderr)
            return 1

    object_rows: list[dict[str, int | str]] = []
    for obj in packet.objects:
        cls = openimages_dict.by_id.get(obj.class_id)
        object_rows.append(
            {
                "track_id": obj.track_id,
                "class_id": obj.class_id,
                "name": cls.name if cls else f"class_{obj.class_id}",
                "cx": obj.cx,
                "cy": obj.cy,
                "size": obj.size,
                "confidence": obj.confidence,
            }
        )

    summary = {
        "image": str(args.image),
        "model": str(args.model),
        "image_size": {"width": image_width, "height": image_height},
        "detections": len(detections),
        "mapped": stats["mapped_detections"],
        "selected": stats["selected_objects"],
        "encoded_bytes": len(encoded),
        "airtime_10kbps_seconds": estimate_airtime_seconds(len(encoded), kbps=10.0),
        "objects": object_rows,
        "hex": hex_string,
    }
    if model_info is not None:
        summary["model_info"] = model_info

    top_raw_detections: list[dict] | None = None
    if args.debug_detections:
        top_raw_detections = _sorted_top_raw_detections(detections, args.debug_top)
        summary["top_raw_detections"] = top_raw_detections

    unmapped_class_names: list[str] | None = None
    if args.debug_mapping:
        unmapped_class_names = _sorted_unmapped_class_names(detections, mapper)
        summary["unmapped_class_names"] = unmapped_class_names

    nova_prompt = build_nova_prompt(
        packet,
        openimages_dict,
        template=args.nova_template,
    )
    summary["nova_template"] = args.nova_template
    summary["nova_prompt"] = nova_prompt

    if args.out_prompt:
        args.out_prompt.parent.mkdir(parents=True, exist_ok=True)
        args.out_prompt.write_text(nova_prompt + "\n", encoding="utf-8")

    if args.print_json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    else:
        print(
            _format_human_output(
                image_path=args.image,
                model_path=args.model,
                image_width=image_width,
                image_height=image_height,
                total_detections=len(detections),
                mapped_detections=stats["mapped_detections"],
                selected_objects=stats["selected_objects"],
                encoded_len=len(encoded),
                airtime_s=summary["airtime_10kbps_seconds"],
                model_info=model_info,
                object_rows=object_rows,
                hex_string=hex_string,
                nova_prompt=nova_prompt,
                top_raw_detections=top_raw_detections,
                unmapped_class_names=unmapped_class_names,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
