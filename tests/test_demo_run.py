"""Tests for hfups.demo.run CLI entry point."""

from __future__ import annotations

import base64
import io
import json
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hfups.demo.run as demo_run
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket
from hfups.vision.yolo_adapter import Detection


def _make_minimal_png() -> bytes:
    """Return a valid 1x1 white PNG as bytes."""
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + ctype + data
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return c + struct.pack(">I", crc)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # 1x1 white RGB pixel
    raw = b"\x00\xff\xff\xff"
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _sample_detections() -> list[Detection]:
    return [
        Detection(class_name="person", confidence=0.9, x1=10, y1=10, x2=100, y2=200),
        Detection(class_name="car", confidence=0.8, x1=200, y1=50, x2=400, y2=250),
    ]


def _make_sample_packet() -> KeyframePacket:
    return KeyframePacket(objects=[
        KeyframeObject(class_id=0, track_id=0, cx=3, cy=4, size=2, confidence=12),
        KeyframeObject(class_id=1, track_id=1, cx=6, cy=2, size=1, confidence=10),
    ])


def test_demo_run_missing_image_exits_1(tmp_path) -> None:
    """Passing a nonexistent image path should exit with code 1."""
    from hfups.demo import run as demo_run
    with pytest.raises(SystemExit) as exc_info:
        demo_run.main(["--image", str(tmp_path / "nonexistent.jpg")])
    assert exc_info.value.code == 1


def test_demo_run_no_nova_exits_clean(tmp_path) -> None:
    """With valid image and no --nova, should exit cleanly with code 0."""
    # Create a dummy image file
    image_path = tmp_path / "some_image.jpg"
    image_path.write_bytes(_make_minimal_png())

    sample_packet = _make_sample_packet()

    mock_runner = MagicMock()
    mock_runner.detect.return_value = _sample_detections()

    mock_builder = MagicMock()
    mock_builder.build.return_value = sample_packet

    with (
        patch("hfups.demo.run.YoloRunner", return_value=mock_runner),
        patch("hfups.demo.run.KeyframeBuilder", return_value=mock_builder),
        patch("hfups.demo.run.load_openimages_v7_boxable_dict"),
        patch("hfups.demo.run.default_openimages_v7_dict_path"),
        patch("hfups.demo.run.build_nova_prompt", return_value="test prompt"),
        patch("hfups.demo.run.ClassMapper"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            demo_run.main(["--image", str(image_path)])
        assert exc_info.value.code == 0


def test_demo_run_nova_saves_files(tmp_path) -> None:
    """With --nova, recon.png and summary.json should be created in the output folder."""
    image_path = tmp_path / "some_image.jpg"
    image_path.write_bytes(_make_minimal_png())
    out_dir = tmp_path / "out"

    sample_packet = _make_sample_packet()
    png_bytes = _make_minimal_png()

    mock_runner = MagicMock()
    mock_runner.detect.return_value = _sample_detections()

    mock_builder = MagicMock()
    mock_builder.build.return_value = sample_packet

    with (
        patch("hfups.demo.run.YoloRunner", return_value=mock_runner),
        patch("hfups.demo.run.KeyframeBuilder", return_value=mock_builder),
        patch("hfups.demo.run.load_openimages_v7_boxable_dict"),
        patch("hfups.demo.run.default_openimages_v7_dict_path"),
        patch("hfups.demo.run.build_nova_prompt", return_value="test prompt"),
        patch("hfups.demo.run.ClassMapper"),
        patch("hfups.demo.run.invoke_nova_canvas", return_value=png_bytes),
        # boto3 available
        patch.dict("sys.modules", {"boto3": MagicMock()}),
    ):
        demo_run.main([
            "--image", str(image_path),
            "--nova",
            "--out", str(out_dir),
        ])

    assert (out_dir / "recon.png").exists()
    assert (out_dir / "summary.json").exists()

    with (out_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "object_count" in summary
    assert "encoded_bytes" in summary
    assert "prompt" in summary
