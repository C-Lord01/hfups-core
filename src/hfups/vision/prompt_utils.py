from __future__ import annotations


def _bucket_x(cx: int) -> str:
    if cx <= 2:
        return "left"
    if cx <= 4:
        return "center"
    return "right"


def _bucket_y(cy: int) -> str:
    if cy <= 2:
        return "top"
    if cy <= 4:
        return "middle"
    return "bottom"


def grid_to_phrase(cx: int, cy: int) -> str:
    """
    Convert 8x8 grid coordinates to a human-readable position phrase.

    The phrase uses coarse buckets (left/center/right, top/middle/bottom)
    plus 1-based grid coordinates for readability.
    """
    x_bucket = _bucket_x(cx)
    y_bucket = _bucket_y(cy)
    return f"{y_bucket}-{x_bucket} (grid {cx + 1},{cy + 1})"


def dxdy_to_direction(dx: int, dy: int) -> str:
    """
    Convert grid-space motion deltas into deterministic motion wording.

    Magnitude tiers are computed from abs(dx)+abs(dy):
    1 -> slightly, 2 -> moderately, >=3 -> strongly.
    """
    if dx == 0 and dy == 0:
        return "stays in place"

    magnitude = abs(dx) + abs(dy)
    if magnitude <= 1:
        strength = "slightly"
    elif magnitude == 2:
        strength = "moderately"
    else:
        strength = "strongly"

    direction_parts: list[str] = []
    if dy < 0:
        direction_parts.append("up")
    elif dy > 0:
        direction_parts.append("down")
    if dx < 0:
        direction_parts.append("left")
    elif dx > 0:
        direction_parts.append("right")

    if not direction_parts:
        return f"moves {strength}"
    return f"moves {strength} " + "-".join(direction_parts)


def size_to_word(size: int) -> str:
    """Map quantized size bucket 0..3 to a descriptive size word."""
    mapping = {
        0: "small",
        1: "medium",
        2: "large",
        3: "very large",
    }
    return mapping.get(size, "unknown-size")


def conf_to_pct(conf: int) -> str:
    """
    Convert quantized confidence (0..15) to a 5%-rounded percentage string.
    """
    pct = int(round((conf / 15.0) * 100.0 / 5.0)) * 5
    pct = max(0, min(100, pct))
    return f"{pct}%"


def grid_to_bucket_phrase(cx: int, cy: int) -> str:
    """
    Return coarse position phrase without explicit grid coordinates.
    """
    return f"{_bucket_y(cy)}-{_bucket_x(cx)}"
