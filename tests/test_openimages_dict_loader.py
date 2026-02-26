import json
from pathlib import Path

import pytest

from hfups.vision.openimages_dict import load_openimages_v7_boxable_dict


def _valid_payload() -> dict:
    return {
        "schema": "hfups.dict.openimages.boxable.v1",
        "source": {
            "dataset": "openimages",
            "version": "v7",
            "subset": "boxable",
            "input_file": "data/openimages/class-descriptions-boxable.csv",
        },
        "created_utc": "2026-02-26T00:00:00Z",
        "classes": [
            {"id": 0, "label": "/m/01g317", "name": "Person"},
            {"id": 1, "label": "/m/0199g", "name": "Car"},
            {"id": 2, "label": "/m/015qff", "name": "Dog"},
        ],
    }


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_load_openimages_v7_boxable_dict_builds_indexes(tmp_path: Path) -> None:
    input_json = tmp_path / "openimages_v7_boxable.json"
    _write_payload(input_json, _valid_payload())

    loaded = load_openimages_v7_boxable_dict(input_json)

    assert loaded.by_id[0].name == "Person"
    assert loaded.by_label["/m/0199g"].id == 1
    assert len(loaded.classes) == 3


def test_load_openimages_v7_boxable_dict_rejects_duplicate_labels(tmp_path: Path) -> None:
    input_json = tmp_path / "openimages_v7_boxable.json"
    payload = _valid_payload()
    payload["classes"][2]["label"] = "/m/0199g"
    _write_payload(input_json, payload)

    with pytest.raises(ValueError, match="Duplicate label"):
        load_openimages_v7_boxable_dict(input_json)


def test_load_openimages_v7_boxable_dict_rejects_non_contiguous_ids(tmp_path: Path) -> None:
    input_json = tmp_path / "openimages_v7_boxable.json"
    payload = _valid_payload()
    payload["classes"][1]["id"] = 4
    _write_payload(input_json, payload)

    with pytest.raises(ValueError, match="contiguous"):
        load_openimages_v7_boxable_dict(input_json)


def test_load_openimages_v7_boxable_dict_rejects_wrong_schema(tmp_path: Path) -> None:
    input_json = tmp_path / "openimages_v7_boxable.json"
    payload = _valid_payload()
    payload["schema"] = "wrong.schema"
    _write_payload(input_json, payload)

    with pytest.raises(ValueError, match="Unsupported schema"):
        load_openimages_v7_boxable_dict(input_json)
