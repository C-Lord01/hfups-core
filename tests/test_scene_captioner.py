"""Tests for caption_parser (pure keyword matching — no model needed)."""

from hfups.vision.caption_parser import parse_caption


def test_caption_parser_hazards():
    result = parse_caption("a flooded street with debris and smoke")
    assert "flood" in result["hazards"]
    assert "debris" in result["hazards"]
    assert "smoke" in result["hazards"]


def test_caption_parser_environment():
    result = parse_caption("a collapsed building on a city street")
    assert result["environment"] == "urban"
