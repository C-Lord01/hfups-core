import json
from pathlib import Path

import pytest

from hfups.vision.class_mapping import (
    ClassMapper,
    default_yolo_to_openimages_mapping_path,
    resolve_mapping_path,
)
from hfups.vision.openimages_dict import (
    OpenImagesClass,
    OpenImagesDict,
    default_openimages_v7_dict_path,
    load_openimages_v7_boxable_dict,
)


def _make_openimages_dict() -> OpenImagesDict:
    classes = [
        OpenImagesClass(id=0, label="/m/person", name="Person"),
        OpenImagesClass(id=1, label="/m/car", name="Car"),
        OpenImagesClass(id=2, label="/m/bus", name="Bus"),
    ]
    return OpenImagesDict(
        classes=classes,
        by_id={entry.id: entry for entry in classes},
        by_label={entry.label: entry for entry in classes},
    )


def test_map_name_to_class_id_returns_expected_id(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "person": "/m/person",
                "car": "/m/car",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    mapper = ClassMapper(_make_openimages_dict(), mapping_path=mapping_path)

    assert mapper.map_name_to_class_id("person") == 0
    assert mapper.map_name_to_class_id("Person") == 0
    assert mapper.map_name_to_class_id("car") == 1
    assert mapper.map_name_to_class_id("unknown") is None


def test_class_mapping_rejects_unknown_openimages_label(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"person": "/m/missing"}, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown Open Images label"):
        ClassMapper(_make_openimages_dict(), mapping_path=mapping_path)


def test_default_mapping_file_exists() -> None:
    assert default_yolo_to_openimages_mapping_path().is_file()


def test_repo_mapping_file_is_valid_for_repo_openimages_dict() -> None:
    openimages_dict = load_openimages_v7_boxable_dict(default_openimages_v7_dict_path())
    mapper = ClassMapper(openimages_dict)
    assert mapper.map_name_to_class_id("person") is not None


def test_resolve_mapping_path_presets_and_custom_path() -> None:
    openimages_path = resolve_mapping_path("preset:openimages")
    coco_path = resolve_mapping_path("preset:coco")
    custom_path = resolve_mapping_path("C:/tmp/custom_mapping.json")

    assert openimages_path == default_yolo_to_openimages_mapping_path()
    assert openimages_path.is_file()
    assert coco_path.name == "yolo_to_openimages.coco_baseline.json"
    assert coco_path.is_file()
    assert custom_path == Path("C:/tmp/custom_mapping.json")
