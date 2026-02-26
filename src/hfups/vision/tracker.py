from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _TrackedBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class SimpleIoUTracker:
    def __init__(self, max_tracks: int = 64, iou_threshold: float = 0.3):
        if max_tracks <= 0 or max_tracks > 64:
            raise ValueError("max_tracks must be in range [1, 64]")
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError("iou_threshold must be in range [0.0, 1.0]")

        self._max_tracks = max_tracks
        self._iou_threshold = iou_threshold
        self._previous: list[_TrackedBox] = []
        self._next_track_id = 0

    def _allocate_track_id(self, used_ids: set[int]) -> int:
        for offset in range(self._max_tracks):
            candidate = (self._next_track_id + offset) % self._max_tracks
            if candidate not in used_ids:
                self._next_track_id = (candidate + 1) % self._max_tracks
                return candidate

        candidate = self._next_track_id
        self._next_track_id = (candidate + 1) % self._max_tracks
        return candidate

    def assign(self, detections: list[tuple[int, float, float, float, float]]) -> list[int]:
        current_boxes: list[tuple[int, float, float, float, float]] = []
        for idx, item in enumerate(detections):
            if len(item) != 5:
                raise ValueError(f"detection {idx} must contain (class_id, x1, y1, x2, y2)")
            class_id, x1, y1, x2, y2 = item
            if not isinstance(class_id, int) or isinstance(class_id, bool):
                raise ValueError(f"detection {idx} class_id must be an integer")
            current_boxes.append((class_id, float(x1), float(y1), float(x2), float(y2)))

        assigned = [-1] * len(current_boxes)
        used_prev_indices: set[int] = set()
        used_ids: set[int] = set()

        for i, (class_id, x1, y1, x2, y2) in enumerate(current_boxes):
            best_prev_idx = -1
            best_iou = -1.0
            best_track_id = 10_000
            for prev_idx, prev in enumerate(self._previous):
                if prev_idx in used_prev_indices:
                    continue
                if prev.class_id != class_id:
                    continue

                score = _iou((x1, y1, x2, y2), (prev.x1, prev.y1, prev.x2, prev.y2))
                if score > best_iou or (score == best_iou and prev.track_id < best_track_id):
                    best_iou = score
                    best_prev_idx = prev_idx
                    best_track_id = prev.track_id

            if best_prev_idx != -1 and best_iou >= self._iou_threshold:
                reused_id = self._previous[best_prev_idx].track_id
                assigned[i] = reused_id
                used_prev_indices.add(best_prev_idx)
                used_ids.add(reused_id)

        for i, track_id in enumerate(assigned):
            if track_id != -1:
                continue
            new_id = self._allocate_track_id(used_ids)
            assigned[i] = new_id
            used_ids.add(new_id)

        self._previous = [
            _TrackedBox(
                class_id=class_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                track_id=assigned[idx],
            )
            for idx, (class_id, x1, y1, x2, y2) in enumerate(current_boxes)
        ]
        return assigned
