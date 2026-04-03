"""One-off script: rebuild dict/openimages_v7_boxable.json from the disaster vocab CSV."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make both src/ and repo root importable (mirrors pytest's rootdir behaviour)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.build_openimages_dict import build_openimages_dict

INPUT_CSV = REPO_ROOT / "data" / "openimages" / "class-descriptions-boxable.csv"
OUTPUT_JSON = REPO_ROOT / "dict" / "openimages_v7_boxable.json"


def main() -> None:
    payload = build_openimages_dict(INPUT_CSV, OUTPUT_JSON)
    n = len(payload["classes"])
    print(f"Written {n} classes to {OUTPUT_JSON}")
    assert n == 29, f"Expected 29 classes, got {n}"
    print("OK — 29 entries confirmed.")


if __name__ == "__main__":
    main()
