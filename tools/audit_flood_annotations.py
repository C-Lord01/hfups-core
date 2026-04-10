"""Audit flood and flooded_area annotations in the merged dataset.

Reports per-split counts of flood-positive images and annotation totals
so we know the current data distribution before sourcing more.
"""

from __future__ import annotations

from pathlib import Path

YAML_PATH = Path(
    "C:/Users/Chris/OneDrive/Documents/Hackerthons/HFUPS/Datasets/merged/data.yaml"
)

FLOOD_CLS = 0        # flood
FLOODED_AREA_CLS = 1  # flooded_area

SPLITS = {
    "train": None,
    "valid": None,
    "test":  None,
}


def parse_yaml_paths(yaml_path: Path) -> dict[str, Path]:
    """Extract train/val/test image directory paths from a YOLO data.yaml."""
    paths: dict[str, Path] = {}
    mapping = {"train": "train", "val": "valid", "test": "test"}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for yaml_key, split_key in mapping.items():
            if line.startswith(f"{yaml_key}:"):
                value = line.split(":", 1)[1].strip()
                paths[split_key] = Path(value)
    return paths


def audit_split(img_dir: Path) -> dict:
    lbl_dir = img_dir.parent / "labels"

    total_images = 0
    flood_positive_images = 0
    flood_annotations = 0
    flooded_area_annotations = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        total_images += 1

        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            continue

        has_flood = False
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(line.split()[0])
            if cls == FLOOD_CLS:
                flood_annotations += 1
                has_flood = True
            elif cls == FLOODED_AREA_CLS:
                flooded_area_annotations += 1
                has_flood = True

        if has_flood:
            flood_positive_images += 1

    return {
        "total_images": total_images,
        "flood_positive_images": flood_positive_images,
        "flood_annotations": flood_annotations,
        "flooded_area_annotations": flooded_area_annotations,
    }


def main() -> None:
    split_paths = parse_yaml_paths(YAML_PATH)

    col_w = 22
    print(f"\n{'Flood Annotation Audit':^80}")
    print(f"Dataset: {YAML_PATH}\n")
    header = (
        f"{'Split':<8} {'Total imgs':>{col_w}} {'Flood+ imgs':>{col_w}}"
        f" {'flood (cls 0)':>{col_w}} {'flooded_area (cls 1)':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    totals = {"total_images": 0, "flood_positive_images": 0,
              "flood_annotations": 0, "flooded_area_annotations": 0}

    for split in ("train", "valid", "test"):
        img_dir = split_paths.get(split)
        if img_dir is None or not img_dir.exists():
            print(f"{split:<8} {'(not found)':>{col_w}}")
            continue

        r = audit_split(img_dir)
        pct = 100 * r["flood_positive_images"] / r["total_images"] if r["total_images"] else 0
        print(
            f"{split:<8}"
            f" {r['total_images']:>{col_w},}"
            f" {r['flood_positive_images']:>{col_w},}  ({pct:.1f}%)"
            f" {r['flood_annotations']:>{col_w},}"
            f" {r['flooded_area_annotations']:>{col_w},}"
        )
        for k in totals:
            totals[k] += r[k]

    print("-" * len(header))
    total_pct = 100 * totals["flood_positive_images"] / totals["total_images"] if totals["total_images"] else 0
    print(
        f"{'TOTAL':<8}"
        f" {totals['total_images']:>{col_w},}"
        f" {totals['flood_positive_images']:>{col_w},}  ({total_pct:.1f}%)"
        f" {totals['flood_annotations']:>{col_w},}"
        f" {totals['flooded_area_annotations']:>{col_w},}"
    )
    print()


if __name__ == "__main__":
    main()
