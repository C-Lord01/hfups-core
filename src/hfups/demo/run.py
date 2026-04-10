"""HFUPS demo runner: detect objects, encode packet, optionally invoke Nova Canvas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hfups.nova.bedrock_client import invoke_nova_canvas
from hfups.nova.prompt_templates import build_nova_prompt
from hfups.transport.vara_text_bridge import encode_stream_to_text
from hfups.vision.class_mapping import ClassMapper
from hfups.vision.keyframe_builder import KeyframeBuilder
from hfups.vision.keyframe_packet import estimate_airtime_seconds
from hfups.vision.openimages_dict import default_openimages_v7_dict_path, load_openimages_v7_boxable_dict
from hfups.vision.yolo_adapter import YoloRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HFUPS demo runner: detect, encode, optionally generate image via Nova Canvas"
    )
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument(
        "--model",
        default="models/yolov8n.pt",
        help="Path to YOLO weights file (default: models/yolov8n.pt)",
    )
    parser.add_argument("--nova", action="store_true", help="Invoke Nova Canvas to generate image")
    parser.add_argument(
        "--template",
        choices=["concise", "descriptive", "disaster_response", "cinematic", "ups"],
        default="ups",
        help="Prompt template style (default: ups)",
    )
    parser.add_argument(
        "--hf-model",
        default="black-forest-labs/FLUX.1-schnell",
        help="HuggingFace model ID for image generation (default: black-forest-labs/FLUX.1-schnell)",
    )
    parser.add_argument("--out", default="outputs", help="Output folder path (default: outputs)")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        # YOLOv8n nano at 0.15 misses partially occluded objects, a common condition
        # in disaster scenes (debris, smoke, crowd occlusion). 0.10 recovers these.
        help="YOLO confidence threshold (default: 0.10)",
    )
    parser.add_argument(
        "--backend",
        choices=["huggingface", "bedrock"],
        default="huggingface",
        help="Image generation backend (default: huggingface)",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional scene description to include in the prompt (e.g. 'flooded street with debris'). "
             "If omitted, the prompt is built from detected objects only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Step a: Validate image
    image_path = Path(args.image)
    if not image_path.exists() or not image_path.is_file():
        print(f"ERROR: Image file not found or not readable: {args.image}", file=sys.stderr)
        sys.exit(1)

    image_size_kb = image_path.stat().st_size / 1024.0

    # Step b: YOLO detection
    runner = YoloRunner(args.model)
    detections = runner.detect(image_path, conf=args.conf)

    # Step c: Build KeyframePacket
    dict_path = default_openimages_v7_dict_path()
    openimages_dict = load_openimages_v7_boxable_dict(dict_path)
    class_mapper = ClassMapper(openimages_dict)
    builder = KeyframeBuilder(class_mapper=class_mapper, max_objects=12, confidence_threshold=args.conf)

    # Attempt to read real image dimensions
    try:
        from PIL import Image as _PIL
        with _PIL.open(image_path) as _img:
            img_w, img_h = _img.size
    except Exception:
        img_w, img_h = 640, 640

    packet = builder.build(detections, img_w, img_h)

    # Step d: Encode packet, calculate airtime
    encoded = packet.encode()
    encoded_bytes = len(encoded)
    airtime_ms = estimate_airtime_seconds(encoded_bytes, kbps=10.0) * 1000.0

    # Step e: VARA text bridge
    vara_out = encode_stream_to_text(encoded)

    # Step f: Use operator-supplied caption if provided (no auto-captioning)
    caption_str: str | None = args.caption or None

    # Step g: Build Nova prompt
    prompt = build_nova_prompt(
        packet,
        openimages_dict,
        caption=caption_str,
        template=args.template,
    )

    # Step h: Print summary
    print(f"Input image: {args.image} ({image_size_kb:.1f} KB)")
    print(f"Objects detected: {len(packet.objects)}")
    print(f"Encoded packet size: {encoded_bytes} bytes")
    print(f"Estimated airtime at 10kbps: {airtime_ms:.1f} ms")
    if caption_str is not None:
        print(f"Caption: {caption_str}")
    print(f"Prompt:\n{prompt}")

    # Step i/j: Nova invocation or clean exit
    if not args.nova:
        sys.exit(0)

    if args.backend == "huggingface":
        from hfups.nova.hf_client import invoke_hf_image
        image_bytes = invoke_hf_image(prompt, model=args.hf_model, width=1024, height=1024)
    else:
        # Check boto3
        try:
            import boto3  # noqa: F401
        except ImportError:
            print(
                "ERROR: boto3 is not installed. Install it with: pip install boto3>=1.34",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            image_bytes = invoke_nova_canvas(
                prompt,
                region="us-east-1",
                profile=args.profile,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    # Save outputs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    recon_path = out_dir / "recon.png"
    recon_path.write_bytes(image_bytes)

    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    vara_path = out_dir / "vara_out.txt"
    vara_path.write_text(vara_out, encoding="utf-8")

    summary = {
        "input_image": str(args.image),
        "image_size_kb": image_size_kb,
        "object_count": len(packet.objects),
        "encoded_bytes": encoded_bytes,
        "airtime_10kbps_ms": airtime_ms,
        "template": args.template,
        "prompt": prompt,
    }
    summary["caption"] = caption_str
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved recon.png to {recon_path}")
    print(f"Saved prompt.txt to {prompt_path}")
    print(f"Saved vara_out.txt to {vara_path}")
    print(f"Saved summary.json to {summary_path}")


if __name__ == "__main__":
    main()
