"""Merge multiple YOLOv8 disaster datasets into a single unified dataset.

Output: C:/Users/Chris/OneDrive/Documents/Hackerthons/HFUPS/Datasets/merged

Final class mapping (9 classes):
    0: flood
    1: flooded_area
    2: car
    3: damaged_building
    4: debris
    5: fire
    6: smoke
    7: accident_vehicle
    8: person
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("C:/Users/Chris/OneDrive/Documents/Hackerthons/HFUPS/Datasets")
MERGED_DIR = DATASETS_DIR / "merged"

FINAL_CLASSES = [
    "flood",
    "flooded_area",
    "car",
    "damaged_building",
    "debris",
    "fire",
    "smoke",
    "accident_vehicle",
    "person",
]

# Each entry: (dataset_folder_name, {src_class_id: dst_class_id | None})
# None means DROP that class.
SOURCES: list[tuple[str, dict[int, int | None]]] = [
    (
        "flood detection.v2i.yolov8",
        {0: 0},  # flood → flood
    ),
    (
        "Flood.v1i.yolov8",
        {
            0: 2,   # car → car
            1: 3,   # house → damaged_building
            2: 8,   # person → person
        },
    ),
    (
        "Hurricane Damage.v6i.yolov8",
        {
            0: 2,    # Car → car
            1: 3,    # Damaged_Roof → damaged_building
            2: 4,    # Debris → debris
            3: 1,    # Flooded_Area → flooded_area
            4: None, # Undamaged_Roof → DROP
            5: None, # Vegetation → DROP
        },
    ),
    (
        "Object Detection Dataset in Post Flood Scenarios.v4i.yolov8",
        {
            0: None, # bike → DROP
            1: 2,    # car → car
            2: 2,    # heavy vehicle → car
            3: 8,    # person → person
        },
    ),
    (
        "Vehicle Accident.v4i.yolov8",
        {
            0: 7,   # accident → accident_vehicle
            1: 2,   # vehicle → car
        },
    ),
    (
        "wildFire.v1i.yolov8",
        {
            0: 5,   # fire → fire
            1: 6,   # smoke → smoke
        },
    ),
]

SPLITS = ["train", "valid", "test"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def remap_label_file(
    src_label: Path,
    dst_label: Path,
    class_map: dict[int, int | None],
) -> int:
    """Read a YOLO label file, remap class IDs, write to dst_label.

    Returns the number of annotations written (0 means file was skipped).
    Skips rows whose class maps to None (dropped).
    """
    if not src_label.exists():
        return 0

    lines = src_label.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        src_cls = int(parts[0])
        dst_cls = class_map.get(src_cls)
        if dst_cls is None:
            continue
        out_lines.append(f"{dst_cls} {' '.join(parts[1:])}")

    if not out_lines:
        return 0

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines)


def find_label_path(img_path: Path) -> Path:
    """Derive label path from image path (images/ → labels/, strip image ext)."""
    parts = list(img_path.parts)
    # Replace the 'images' directory component with 'labels'
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def unique_dst_name(dst_dir: Path, stem: str, suffix: str, prefix: str) -> Path:
    """Return a collision-free path: prefix_stem[_N].suffix."""
    candidate = dst_dir / f"{prefix}_{stem}{suffix}"
    if not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = dst_dir / f"{prefix}_{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Clean and recreate merged directory
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    # Counters
    images_per_split: dict[str, int] = defaultdict(int)
    annotations_per_class: dict[int, int] = defaultdict(int)
    skipped_images = 0

    for ds_name, class_map in SOURCES:
        ds_dir = DATASETS_DIR / ds_name
        # Short prefix for collision avoidance (first word, lowercased)
        prefix = ds_name.split(".")[0].split()[0].lower()

        for split in SPLITS:
            img_dir = ds_dir / split / "images"
            if not img_dir.exists():
                continue

            dst_img_dir = MERGED_DIR / split / "images"
            dst_lbl_dir = MERGED_DIR / split / "labels"
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                src_label = find_label_path(img_path)

                # Determine destination label path (collision-safe)
                dst_label = unique_dst_name(dst_lbl_dir, img_path.stem, ".txt", prefix)
                n_written = remap_label_file(src_label, dst_label, class_map)

                if n_written == 0:
                    # All annotations dropped or label missing — skip image
                    skipped_images += 1
                    continue

                # Copy image with same collision-safe stem
                dst_img = dst_img_dir / dst_label.with_suffix(img_path.suffix).name
                shutil.copy2(img_path, dst_img)
                images_per_split[split] += 1

                # Count annotations
                for line in dst_label.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        annotations_per_class[int(line.split()[0])] += 1

    # Write data.yaml
    merged_abs = MERGED_DIR.resolve()
    yaml_lines = [
        f"train: {(merged_abs / 'train' / 'images').as_posix()}",
        f"val: {(merged_abs / 'valid' / 'images').as_posix()}",
        f"test: {(merged_abs / 'test' / 'images').as_posix()}",
        "",
        f"nc: {len(FINAL_CLASSES)}",
        "names:",
    ]
    for cls in FINAL_CLASSES:
        yaml_lines.append(f"  - {cls}")
    yaml_lines.append("")
    (MERGED_DIR / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    # Summary
    print("\n=== Merge complete ===")
    print(f"Output: {MERGED_DIR}")
    print(f"\nImages per split:")
    for split in SPLITS:
        print(f"  {split:<8}: {images_per_split.get(split, 0)}")
    total = sum(images_per_split.values())
    print(f"  {'total':<8}: {total}")
    print(f"\n  skipped (all-drop): {skipped_images}")

    print(f"\nAnnotations per class:")
    for cls_id, cls_name in enumerate(FINAL_CLASSES):
        count = annotations_per_class.get(cls_id, 0)
        print(f"  {cls_id}  {cls_name:<20}: {count}")
    total_ann = sum(annotations_per_class.values())
    print(f"  {'total':<22}: {total_ann}")


if __name__ == "__main__":
    main()
