from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hfups.vision.class_mapping import ClassMapper
from hfups.vision.keyframe_packet import (
    KeyframeObject,
    KeyframePacket,
    estimate_airtime_seconds,
)
from hfups.vision.openimages_dict import (
    default_openimages_v7_dict_path,
    load_openimages_v7_boxable_dict,
)
from hfups.vision.tracker import SimpleIoUTracker
from hfups.vision.yolo_adapter import Detection, YoloRunner

TrackAssigner = Callable[[list[KeyframeObject], list[Detection]], list[int]]


@dataclass(frozen=True)
class _MappedDetection:
    detection: Detection
    class_id: int
    area: float


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _iou(a: Detection, b: Detection) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def make_tracker_assigner(tracker: SimpleIoUTracker) -> TrackAssigner:
    def _assigner(
        objects: list[KeyframeObject],
        detections: list[Detection],
    ) -> list[int]:
        tracked = [
            (obj.class_id, det.x1, det.y1, det.x2, det.y2)
            for obj, det in zip(objects, detections)
        ]
        return tracker.assign(tracked)

    return _assigner


class KeyframeBuilder:
    def __init__(
        self,
        class_mapper: ClassMapper,
        grid: int = 8,
        max_objects: int = 12,
        confidence_threshold: float = 0.25,
        assign_track_ids: TrackAssigner | None = None,
    ):
        if grid != 8:
            raise ValueError("grid must be 8 for keyframe packet encoding")
        if max_objects <= 0 or max_objects > 12:
            raise ValueError("max_objects must be in range [1, 12]")
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be in range [0.0, 1.0]")

        self._class_mapper = class_mapper
        self._grid = grid
        self._max_objects = max_objects
        self._confidence_threshold = confidence_threshold
        self._assign_track_ids = assign_track_ids

    def _filter_mapped_detections(self, detections: list[Detection]) -> list[_MappedDetection]:
        filtered: list[_MappedDetection] = []
        for det in detections:
            if det.confidence < self._confidence_threshold:
                continue

            class_id = self._class_mapper.map_name_to_class_id(det.class_name)
            if class_id is None:
                continue

            filtered.append(
                _MappedDetection(
                    detection=det,
                    class_id=class_id,
                    area=det.w * det.h,
                )
            )
        return filtered

    def _dedupe_by_class(self, detections: list[_MappedDetection]) -> list[_MappedDetection]:
        grouped: dict[int, list[_MappedDetection]] = {}
        for item in detections:
            grouped.setdefault(item.class_id, []).append(item)

        deduped: list[_MappedDetection] = []
        for class_id in sorted(grouped):
            candidates = sorted(
                grouped[class_id],
                key=lambda item: (
                    -item.detection.confidence,
                    -item.area,
                    item.detection.x1,
                    item.detection.y1,
                    item.detection.x2,
                    item.detection.y2,
                    item.detection.class_name.lower(),
                ),
            )
            chosen: list[_MappedDetection] = []
            for candidate in candidates:
                if not chosen:
                    chosen.append(candidate)
                    continue
                if len(chosen) >= 2:
                    break
                if all(_iou(candidate.detection, existing.detection) < 0.2 for existing in chosen):
                    chosen.append(candidate)
            deduped.extend(chosen)

        return deduped

    def _select_candidates(self, detections: list[_MappedDetection]) -> list[_MappedDetection]:
        deduped = self._dedupe_by_class(detections)
        ranked = sorted(
            deduped,
            key=lambda item: (
                -item.detection.confidence,
                -item.area,
                item.class_id,
                item.detection.x1,
                item.detection.y1,
                item.detection.x2,
                item.detection.y2,
                item.detection.class_name.lower(),
            ),
        )
        return ranked[: self._max_objects]

    def _quantize_coord(self, center: float, span: int) -> int:
        q = math.floor((center / span) * self._grid)
        return _clamp(q, 0, self._grid - 1)

    @staticmethod
    def _quantize_size(area_ratio: float) -> int:
        if area_ratio < 0.02:
            return 0
        if area_ratio < 0.08:
            return 1
        if area_ratio < 0.20:
            return 2
        return 3

    @staticmethod
    def _quantize_confidence(confidence: float) -> int:
        return _clamp(int(round(confidence * 15)), 0, 15)

    def _build_objects(
        self,
        selected: list[_MappedDetection],
        image_width: int,
        image_height: int,
    ) -> list[KeyframeObject]:
        pixel_area = image_width * image_height
        base_objects = [
            KeyframeObject(
                class_id=item.class_id,
                track_id=0,
                cx=self._quantize_coord(item.detection.cx, image_width),
                cy=self._quantize_coord(item.detection.cy, image_height),
                size=self._quantize_size(item.area / pixel_area),
                confidence=self._quantize_confidence(item.detection.confidence),
            )
            for item in selected
        ]

        if self._assign_track_ids is None:
            return base_objects

        track_ids = self._assign_track_ids(
            base_objects,
            [item.detection for item in selected],
        )
        if len(track_ids) != len(base_objects):
            raise ValueError("assign_track_ids must return one track_id per object")

        tracked_objects: list[KeyframeObject] = []
        for idx, (obj, track_id) in enumerate(zip(base_objects, track_ids)):
            if not isinstance(track_id, int) or isinstance(track_id, bool):
                raise ValueError(f"track_id at index {idx} must be an integer")
            if track_id < 0 or track_id > 63:
                raise ValueError(f"track_id at index {idx} must be in range [0, 63]")
            tracked_objects.append(
                KeyframeObject(
                    class_id=obj.class_id,
                    track_id=track_id,
                    cx=obj.cx,
                    cy=obj.cy,
                    size=obj.size,
                    confidence=obj.confidence,
                )
            )
        return tracked_objects

    def build(self, detections: list[Detection], image_width: int, image_height: int) -> KeyframePacket:
        packet, _ = self.build_with_stats(detections, image_width, image_height)
        return packet

    def build_with_stats(
        self,
        detections: list[Detection],
        image_width: int,
        image_height: int,
    ) -> tuple[KeyframePacket, dict[str, int]]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be > 0")

        mapped = self._filter_mapped_detections(detections)
        selected = self._select_candidates(mapped)
        objects = self._build_objects(selected, image_width, image_height)

        packet = KeyframePacket(objects=objects)
        return packet, {
            "mapped_detections": len(mapped),
            "selected_objects": len(objects),
        }


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ValueError(
            "Pillow is required for build_keyframe_from_image image-size lookup. "
            "Install it with `pip install pillow`."
        ) from exc

    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be > 0")
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


def _sorted_unmapped_class_names(
    detections: list[Detection],
    class_mapper: ClassMapper,
) -> list[str]:
    names = {
        det.class_name.strip()
        for det in detections
        if class_mapper.map_name_to_class_id(det.class_name) is None
    }
    return sorted(names, key=lambda value: (value.lower(), value))


def build_keyframe_from_image(
    image_path: Path,
    model_path: Path,
    dict_path: Path | None = None,
    mapping_path: str | Path | None = None,
    *,
    conf: float = 0.25,
    max_objects: int = 12,
    grid: int = 8,
    debug_top_n: int = 5,
    include_raw: bool = False,
    include_unmapped: bool = False,
) -> tuple[KeyframePacket, dict]:
    dictionary = load_openimages_v7_boxable_dict(dict_path or default_openimages_v7_dict_path())
    class_mapper = ClassMapper(dictionary, mapping_path=mapping_path)
    runner = YoloRunner(model_path)

    detections = runner.detect(image_path, conf=conf)
    image_width, image_height = _read_image_size(image_path)

    builder = KeyframeBuilder(
        class_mapper=class_mapper,
        grid=grid,
        max_objects=max_objects,
        confidence_threshold=conf,
    )
    packet, build_stats = builder.build_with_stats(detections, image_width, image_height)
    encoded = packet.encode()

    debug = {
        "total_detections": len(detections),
        "mapped_detections": build_stats["mapped_detections"],
        "selected_objects": build_stats["selected_objects"],
        "encoded_bytes_length": len(encoded),
        "airtime_10kbps_seconds": estimate_airtime_seconds(len(encoded), kbps=10.0),
    }
    if include_raw:
        debug["top_raw_detections"] = _sorted_top_raw_detections(detections, debug_top_n)
    if include_unmapped:
        debug["unmapped_class_names"] = _sorted_unmapped_class_names(detections, class_mapper)
    return packet, debug
