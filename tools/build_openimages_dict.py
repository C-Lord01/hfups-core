from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "hfups.dict.openimages.boxable.v1"
SOURCE_INPUT_FILE = "data/openimages/class-descriptions-boxable.csv"


def _utc_now_iso8601_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_classes(csv_path: Path) -> list[tuple[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    classes: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue

            if (
                row_idx == 1
                and len(row) >= 2
                and row[0].strip() == "LabelName"
                and row[1].strip() == "DisplayName"
            ):
                continue

            if len(row) < 2:
                raise ValueError(f"CSV row {row_idx} must have at least 2 columns")

            label = row[0].strip()
            name = row[1].strip()
            if not label:
                raise ValueError(f"CSV row {row_idx} has empty label")
            if not name:
                raise ValueError(f"CSV row {row_idx} has empty display name")
            classes.append((label, name))

    if not classes:
        raise ValueError("Input CSV has no classes")

    return classes


def build_openimages_dict(
    input_path: str | Path,
    output_path: str | Path,
    *,
    created_utc: str | None = None,
) -> dict[str, Any]:
    input_csv = Path(input_path)
    output_json = Path(output_path)

    class_rows = _read_classes(input_csv)
    created_utc_value = created_utc or _utc_now_iso8601_z()

    payload = {
        "schema": SCHEMA,
        "source": {
            "dataset": "openimages",
            "version": "v7",
            "subset": "boxable",
            "input_file": SOURCE_INPUT_FILE,
        },
        "created_utc": created_utc_value,
        "classes": [
            {"id": class_id, "label": label, "name": name}
            for class_id, (label, name) in enumerate(class_rows)
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic Open Images V7 boxable classes dictionary JSON."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        type=Path,
        help="Path to class-descriptions-boxable.csv",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        type=Path,
        help="Path to write openimages_v7_boxable.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        payload = build_openimages_dict(args.input_path, args.output_path)
    except Exception as exc:  # pragma: no cover - exercised through CLI use
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"classes: {len(payload['classes'])}")
    print(f"output: {Path(args.output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
