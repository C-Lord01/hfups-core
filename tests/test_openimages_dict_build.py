import json
import re
from pathlib import Path

from tools.build_openimages_dict import build_openimages_dict


def _write_mini_csv(path: Path) -> None:
    path.write_text(
        "\ufeffLabelName,DisplayName\r\n"
        "/m/zz03,Third\r\n"
        "/m/aa01,First\r\n"
        "/m/mm02,Second\r\n",
        encoding="utf-8",
    )


def test_build_openimages_dict_preserves_order_and_ids(tmp_path: Path) -> None:
    input_csv = tmp_path / "class-descriptions-boxable.csv"
    output_json = tmp_path / "openimages_v7_boxable.json"
    _write_mini_csv(input_csv)

    build_openimages_dict(
        input_csv,
        output_json,
        created_utc="2026-02-26T00:00:00Z",
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["schema"] == "hfups.dict.openimages.boxable.v1"
    assert [entry["label"] for entry in result["classes"]] == [
        "/m/zz03",
        "/m/aa01",
        "/m/mm02",
    ]
    assert [entry["id"] for entry in result["classes"]] == [0, 1, 2]
    assert result["created_utc"] == "2026-02-26T00:00:00Z"


def test_build_openimages_dict_sets_created_utc_in_iso8601_z(tmp_path: Path) -> None:
    input_csv = tmp_path / "class-descriptions-boxable.csv"
    output_json = tmp_path / "openimages_v7_boxable.json"
    _write_mini_csv(input_csv)

    build_openimages_dict(input_csv, output_json)

    result = json.loads(output_json.read_text(encoding="utf-8"))
    created_utc = result["created_utc"]
    assert isinstance(created_utc, str)
    assert created_utc.endswith("Z")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created_utc)


def test_build_openimages_dict_is_deterministic_with_fixed_timestamp(tmp_path: Path) -> None:
    input_csv = tmp_path / "class-descriptions-boxable.csv"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    _write_mini_csv(input_csv)

    created_utc = "2026-02-26T00:00:00Z"
    build_openimages_dict(input_csv, first_output, created_utc=created_utc)
    build_openimages_dict(input_csv, second_output, created_utc=created_utc)

    assert first_output.read_text(encoding="utf-8") == second_output.read_text(encoding="utf-8")
