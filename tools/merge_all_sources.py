"""Merge three flood dataset sources into flood_v2 with 17-class remapping.

Idempotent: re-running will skip already-copied images.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLOOD_V2_DIR = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\flood_v2"
)

SOURCES = [
    {
        "name": "flood_detection_v2",
        "base": Path(r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\flood detection.v2i.yolov8"),
        "splits": ["train", "valid", "test"],
        "remap": {0: 0},
        "skip": [],
    },
    {
        "name": "flood_v1",
        "base": Path(r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\Flood.v1i.yolov8"),
        "splits": ["train"],
        "remap": {0: 6, 1: 10, 2: 7},
        "skip": [],
    },
    {
        "name": "post_flood_scenarios",
        "base": Path(r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\Object Detection Dataset in Post Flood Scenarios.v4i.yolov8"),
        "splits": ["train", "valid", "test"],
        "remap": {1: 6, 2: 6, 3: 7},
        "skip": [0],
    },
]

CLASS_NAMES = [
    "flood",              # 0
    "flooded_area",       # 1
    "flooded_road",       # 2
    "flooded_bridge",     # 3
    "flooded_carpark",    # 4
    "submerged_car",      # 5
    "car",                # 6
    "person",             # 7
    "person_on_vehicle",  # 8
    "person_in_water",    # 9
    "residential_building",  # 10
    "utility_pole",       # 11
    "carpark",            # 12
    "ocean",              # 13
    "waves",              # 14
    "debris_floating",    # 15
    "rescue_boat",        # 16
]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
OK_THRESHOLD = 300
SPARSE_THRESHOLD = 1


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def determine_target_split(source: dict, src_split: str, stem: str) -> str:
    """Honor existing split if source has multiple; hash-assign if single-split source."""
    if len(source["splits"]) > 1:
        return src_split  # train/valid/test already defined by source

    # Single-split source (flood_v1 train-only): deterministic hash assignment
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "valid"
    else:
        return "test"


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

def process_source(source: dict) -> dict[str, int]:
    """Process one source. Returns {images_added, annotations_added}."""
    name = source["name"]
    base = source["base"]
    remap: dict[int, int] = source["remap"]
    skip: list[int] = source["skip"]

    images_added = 0
    annotations_added = 0

    all_images: list[tuple[str, Path, Path]] = []  # (src_split, img_path, lbl_path)
    for src_split in source["splits"]:
        img_dir = base / src_split / "images"
        lbl_dir = base / src_split / "labels"
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping.")
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            all_images.append((src_split, img_path, lbl_path))

    for src_split, img_path, lbl_path in tqdm(all_images, desc=name, unit="img"):
        stem = img_path.stem
        target_split = determine_target_split(source, src_split, stem)

        dest_img_dir = FLOOD_V2_DIR / target_split / "images"
        dest_lbl_dir = FLOOD_V2_DIR / target_split / "labels"
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        dest_image = dest_img_dir / img_path.name
        dest_label = dest_lbl_dir / f"{stem}.txt"

        # Idempotent: skip if already copied
        if dest_image.exists():
            continue

        # Copy image
        shutil.copy2(img_path, dest_image)
        images_added += 1

        # Process label
        remapped_lines: list[str] = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls in skip:
                    continue
                if cls not in remap:
                    continue
                new_cls = remap[cls]
                remapped_lines.append(f"{new_cls} {' '.join(parts[1:])}")

        dest_label.write_text(
            "\n".join(remapped_lines) + ("\n" if remapped_lines else ""),
            encoding="utf-8",
        )
        annotations_added += len(remapped_lines)

    return {"images_added": images_added, "annotations_added": annotations_added}


# ---------------------------------------------------------------------------
# Post-merge report
# ---------------------------------------------------------------------------

def post_merge_report(source_stats: list[tuple[str, dict]]) -> None:
    print("\n=== Post-Merge Dataset Report ===\n")

    # Split image counts
    split_counts: dict[str, int] = {}
    total_images = 0
    for split in ("train", "valid", "test"):
        img_dir = FLOOD_V2_DIR / split / "images"
        n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS) if img_dir.exists() else 0
        split_counts[split] = n
        total_images += n

    print(f"{'Split':<12} {'Images':>8}")
    print("-" * 22)
    for split in ("train", "valid", "test"):
        print(f"{split:<12} {split_counts[split]:>8,}")
    print(f"{'total':<12} {total_images:>8,}")

    # Annotation counts per class
    class_counts: dict[int, int] = defaultdict(int)
    for split in ("train", "valid", "test"):
        lbl_dir = FLOOD_V2_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.iterdir():
            if lbl_file.suffix != ".txt":
                continue
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    class_counts[int(line.split()[0])] += 1

    print(f"\n{'ID':<5} {'Class':<24} {'Annotations':>12}  Status")
    print("-" * 56)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = class_counts.get(cls_id, 0)
        if count == 0:
            status = "MISSING"
        elif count < OK_THRESHOLD:
            status = "SPARSE"
        else:
            status = "ok"
        print(f"{cls_id:<5} {cls_name:<24} {count:>12,}  {status}")

    # Per-source contribution
    print(f"\n{'Source':<24} {'Images added':>14} {'Annotations added':>18}")
    print("-" * 58)
    for src_name, stats in source_stats:
        print(
            f"{src_name:<24} {stats['images_added']:>14,} {stats['annotations_added']:>18,}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not FLOOD_V2_DIR.exists():
        print(f"ERROR: FLOOD_V2_DIR not found: {FLOOD_V2_DIR}")
        return

    source_stats: list[tuple[str, dict]] = []

    for source in SOURCES:
        print(f"\n--- Processing: {source['name']} ---")
        stats = process_source(source)
        source_stats.append((source["name"], stats))
        print(f"  Added: {stats['images_added']:,} images, {stats['annotations_added']:,} annotations")

    post_merge_report(source_stats)


if __name__ == "__main__":
    main()
