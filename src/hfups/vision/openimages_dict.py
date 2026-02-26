from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EXPECTED_SCHEMA = "hfups.dict.openimages.boxable.v1"


@dataclass(frozen=True)
class OpenImagesClass:
    id: int
    label: str
    name: str


@dataclass(frozen=True)
class OpenImagesDict:
    classes: list[OpenImagesClass]
    by_id: dict[int, OpenImagesClass]
    by_label: dict[str, OpenImagesClass]


def find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    current = start_path if start_path.is_dir() else start_path.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise ValueError(f"Could not find repository root from {start_path}")


def default_openimages_v7_dict_path() -> Path:
    return find_repo_root() / "dict" / "openimages_v7_boxable.json"


def load_openimages_v7_boxable_dict(path: str | Path) -> OpenImagesDict:
    json_path = Path(path)
    try:
        raw_data = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read dictionary file at {json_path}: {exc}") from exc

    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in dictionary file {json_path}: {exc}") from exc

    schema = data.get("schema")
    if schema != EXPECTED_SCHEMA:
        raise ValueError(
            f"Unsupported schema {schema!r}; expected {EXPECTED_SCHEMA!r}"
        )

    source = data.get("source")
    if not isinstance(source, dict):
        raise ValueError("Dictionary source must be an object")
    if source.get("version") != "v7":
        raise ValueError(
            f"Unsupported source version {source.get('version')!r}; expected 'v7'"
        )
    if source.get("subset") != "boxable":
        raise ValueError(
            f"Unsupported source subset {source.get('subset')!r}; expected 'boxable'"
        )

    raw_classes = data.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("Dictionary classes must be a non-empty list")

    classes: list[OpenImagesClass] = []
    by_id: dict[int, OpenImagesClass] = {}
    by_label: dict[str, OpenImagesClass] = {}

    for idx, raw_class in enumerate(raw_classes):
        if not isinstance(raw_class, dict):
            raise ValueError(f"Class at index {idx} must be an object")

        raw_id = raw_class.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            raise ValueError(f"Class at index {idx} has non-integer id {raw_id!r}")
        if raw_id != idx:
            raise ValueError(
                f"Class id at index {idx} must be contiguous and equal to {idx}; got {raw_id}"
            )

        label = raw_class.get("label")
        if not isinstance(label, str) or not label.startswith("/"):
            raise ValueError(
                f"Class id {raw_id} has invalid label {label!r}; expected a string starting with '/'"
            )
        if label in by_label:
            raise ValueError(f"Duplicate label found: {label!r}")

        name = raw_class.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Class id {raw_id} has invalid name {name!r}")

        entry = OpenImagesClass(id=raw_id, label=label, name=name)
        classes.append(entry)
        by_id[raw_id] = entry
        by_label[label] = entry

    return OpenImagesDict(classes=classes, by_id=by_id, by_label=by_label)
