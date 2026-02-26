from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UltralyticsNotInstalledError(RuntimeError):
    """Raised when Ultralytics is not installed but YOLO inference is requested."""


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)


class YoloRunner:
    def __init__(self, model_path: str | Path):
        self._model_path = Path(model_path)
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - depends on optional runtime dep
                raise UltralyticsNotInstalledError(
                    "Ultralytics not installed. Install with: python -m pip install ultralytics"
                ) from exc
            self._model = YOLO(str(self._model_path))
        return self._model

    def detect(self, image_path: str | Path, conf: float = 0.25) -> list[Detection]:
        model = self._get_model()
        results = model.predict(source=str(image_path), conf=conf, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()
        cls_list = boxes.cls.tolist()
        names = getattr(model, "names", {})

        detections: list[Detection] = []
        for xyxy, score, class_idx_raw in zip(xyxy_list, conf_list, cls_list):
            class_idx = int(class_idx_raw)
            if isinstance(names, dict):
                class_name_raw = names.get(class_idx, str(class_idx))
            elif isinstance(names, list) and 0 <= class_idx < len(names):
                class_name_raw = names[class_idx]
            else:
                class_name_raw = str(class_idx)

            detections.append(
                Detection(
                    class_name=str(class_name_raw),
                    confidence=float(score),
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                )
            )
        return detections

    def model_info(self) -> dict:
        model = self._get_model()
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            ordered = [str(names[key]) for key in sorted(names)]
        elif isinstance(names, list):
            ordered = [str(name) for name in names]
        else:
            ordered = []
        return {
            "class_count": len(ordered),
            "names_sample": ordered[:10],
        }
