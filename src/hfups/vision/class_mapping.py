from __future__ import annotations

import json
from pathlib import Path

from hfups.vision.openimages_dict import OpenImagesDict, find_repo_root


def default_yolo_to_openimages_mapping_path() -> Path:
    return find_repo_root() / "data" / "mappings" / "yolo_to_openimages.json"


def default_yolo_to_openimages_coco_baseline_mapping_path() -> Path:
    return find_repo_root() / "data" / "mappings" / "yolo_to_openimages.coco_baseline.json"


def resolve_mapping_path(mapping_arg: str | Path | None) -> Path:
    """
    Resolve mapping path from preset or filesystem input.

    Supports:
      - None => default mapping
      - "preset:coco" => COCO baseline mapping
      - "preset:openimages" => default Open Images mapping
      - else => filesystem path
    """
    if mapping_arg is None:
        return default_yolo_to_openimages_mapping_path()

    if isinstance(mapping_arg, Path):
        return mapping_arg

    arg = mapping_arg.strip()
    if not arg:
        return default_yolo_to_openimages_mapping_path()

    preset = arg.lower()
    if preset == "preset:coco":
        return default_yolo_to_openimages_coco_baseline_mapping_path()
    if preset == "preset:openimages":
        return default_yolo_to_openimages_mapping_path()
    return Path(arg)


class ClassMapper:
    def __init__(
        self,
        openimages_dict: OpenImagesDict,
        mapping_path: str | Path | None = None,
    ) -> None:
        self._dict = openimages_dict
        self._mapping_path = resolve_mapping_path(mapping_path)
        self._name_to_class_id = self._load_mapping(self._mapping_path)

    def _load_mapping(self, mapping_path: Path) -> dict[str, int]:
        try:
            raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Unable to read mapping file {mapping_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON mapping file {mapping_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError("Mapping file must be a JSON object")

        name_to_class_id: dict[str, int] = {}
        for raw_name, raw_label in raw.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("Mapping names must be non-empty strings")
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError(f"Mapping label for {raw_name!r} must be a non-empty string")

            mapped_name = raw_name.strip().lower()
            mapped_label = raw_label.strip()

            cls = self._dict.by_label.get(mapped_label)
            if cls is None:
                raise ValueError(
                    f"Mapping for {raw_name!r} references unknown Open Images label {mapped_label!r}"
                )
            name_to_class_id[mapped_name] = cls.id

        return name_to_class_id

    def map_name_to_class_id(self, name: str) -> int | None:
        return self._name_to_class_id.get(name.strip().lower())
