"""Prepare Flood.v1i.yolov8 dataset against the 17-class disaster vocabulary.

Steps:
    1. Remap existing labels to new class IDs
    2. Auto-annotate with existing disaster model (flood-specific classes only)
    3. Merge remapped + auto labels
    4. Split 80/10/10 into train/valid/test
    5. Write data.yaml and print class distribution report
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_IMAGES = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\Flood.v1i.yolov8\train\images"
)
SOURCE_LABELS = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\Flood.v1i.yolov8\train\labels"
)
EXISTING_MODEL = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\HFUPS Repo\models\yolov8s_disaster.pt"
)
OUTPUT_DIR = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\flood_v2"
)
CONF_THRESHOLD = 0.25
SPLIT = (0.80, 0.10, 0.10)  # train / valid / test
BATCH_SIZE = 32
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 17-class vocabulary
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "flood",             # 0
    "flooded_area",      # 1
    "flooded_road",      # 2
    "flooded_bridge",    # 3
    "flooded_carpark",   # 4
    "submerged_car",     # 5
    "car",               # 6
    "person",            # 7
    "person_on_vehicle", # 8
    "person_in_water",   # 9
    "residential_building",  # 10
    "utility_pole",      # 11
    "carpark",           # 12
    "ocean",             # 13
    "waves",             # 14
    "debris_floating",   # 15
    "rescue_boat",       # 16
]

# Source dataset class map: old_id → new_id
REMAP = {
    0: 6,   # car → car
    1: 10,  # house → residential_building
    2: 7,   # person → person
}

# Old disaster model (9-class) → new 17-class IDs
# Only keep flood-specific classes; skip car(2)/person(8)/damaged_building(3)
# Old: 0=flood,1=flooded_area,2=car,3=damaged_building,4=debris,5=fire,6=smoke,7=accident_vehicle,8=person
OLD_TO_NEW = {
    0: 0,   # flood → flood
    1: 1,   # flooded_area → flooded_area
    4: 15,  # debris → debris_floating
}
# Classes to retain from model predictions (flood-specific only, no overlap with remapped)
AUTO_KEEP_OLD = set(OLD_TO_NEW.keys())

SPARSE_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Step 1: Remap existing labels
# ---------------------------------------------------------------------------

def remap_labels(source_labels: Path) -> dict[str, list[str]]:
    """Read all label files, apply REMAP, return {stem: [lines]}."""
    remapped: dict[str, list[str]] = {}
    for lbl_path in source_labels.iterdir():
        if lbl_path.suffix != ".txt":
            continue
        out_lines: list[str] = []
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = REMAP.get(old_cls)
            if new_cls is None:
                continue
            out_lines.append(f"{new_cls} {' '.join(parts[1:])}")
        remapped[lbl_path.stem] = out_lines
    return remapped


# ---------------------------------------------------------------------------
# Step 2: Auto-annotate with existing model
# ---------------------------------------------------------------------------

def auto_annotate(
    source_images: Path,
    model_path: Path,
    image_paths: list[Path],
) -> dict[str, list[str]]:
    """Run inference, keep flood-specific predictions, return {stem: [lines]}."""
    from ultralytics import YOLO
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):  # type: ignore[misc]
            return it

    model = YOLO(str(model_path))
    auto: dict[str, list[str]] = defaultdict(list)

    batches = [image_paths[i:i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Auto-annotating", unit="batch"):
        results = model.predict(
            [str(p) for p in batch],
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        for img_path, result in zip(batch, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            stem = img_path.stem
            boxes = result.boxes
            for cls_tensor, xywhn in zip(boxes.cls, boxes.xywhn):
                old_cls = int(cls_tensor.item())
                if old_cls not in AUTO_KEEP_OLD:
                    continue
                new_cls = OLD_TO_NEW[old_cls]
                cx, cy, w, h = xywhn.tolist()
                auto[stem].append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return dict(auto)


# ---------------------------------------------------------------------------
# Step 3: Merge labels
# ---------------------------------------------------------------------------

def merge_labels(
    remapped: dict[str, list[str]],
    auto: dict[str, list[str]],
    all_stems: list[str],
    temp_dir: Path,
) -> None:
    """Write merged label files to temp_dir/{stem}.txt for all stems."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    for stem in all_stems:
        lines = remapped.get(stem, []) + auto.get(stem, [])
        (temp_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Step 4: Split and copy
# ---------------------------------------------------------------------------

def split_and_copy(
    all_stems: list[str],
    source_images: Path,
    temp_labels: Path,
    output_dir: Path,
    split: tuple[float, float, float],
) -> dict[str, list[str]]:
    """Shuffle, split, copy images + labels. Returns {split_name: [stems]}."""
    rng = random.Random(RANDOM_SEED)
    shuffled = list(all_stems)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * split[0])
    n_valid = int(n * split[1])

    splits = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test":  shuffled[n_train + n_valid:],
    }

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for split_name, stems in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            # Find image file (any extension)
            img_src = None
            for ext in img_extensions:
                candidate = source_images / f"{stem}{ext}"
                if candidate.exists():
                    img_src = candidate
                    break
            if img_src:
                shutil.copy2(img_src, img_out / img_src.name)

            # Copy label
            lbl_src = temp_labels / f"{stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_out / f"{stem}.txt")
            else:
                # Write empty file — YOLO expects label files to exist
                (lbl_out / f"{stem}.txt").write_text("", encoding="utf-8")

    return splits


# ---------------------------------------------------------------------------
# Step 5: Write data.yaml and report
# ---------------------------------------------------------------------------

def write_yaml(output_dir: Path) -> None:
    abs_path = output_dir.resolve().as_posix()
    lines = [
        f"path: {abs_path}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for name in CLASS_NAMES:
        lines.append(f"  - {name}")
    lines.append("")
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def report(
    output_dir: Path,
    splits: dict[str, list[str]],
) -> None:
    print("\n=== Dataset Report ===")
    print(f"Output: {output_dir}\n")

    print(f"{'Split':<8} {'Images':>8}")
    print("-" * 18)
    for split_name in ("train", "valid", "test"):
        n = len(splits.get(split_name, []))
        print(f"{split_name:<8} {n:>8,}")
    print(f"{'total':<8} {sum(len(v) for v in splits.values()):>8,}")

    # Count annotations per class across entire dataset
    class_counts: dict[int, int] = defaultdict(int)
    for split_name in ("train", "valid", "test"):
        lbl_dir = output_dir / split_name / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.iterdir():
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    class_counts[int(line.split()[0])] += 1

    print(f"\n{'ID':<4} {'Class':<22} {'Annotations':>12}  {'Status'}")
    print("-" * 56)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = class_counts.get(cls_id, 0)
        if count == 0:
            status = "MISSING"
        elif count < SPARSE_THRESHOLD:
            status = "SPARSE"
        else:
            status = "ok"
        print(f"{cls_id:<4} {cls_name:<22} {count:>12,}  {status}")

    print(
        "\nAuto-annotated classes are unreviewed predictions. "
        "Classes flagged SPARSE or MISSING require SageMaker Ground Truth review."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Validate inputs
    for path, label in [
        (SOURCE_IMAGES, "SOURCE_IMAGES"),
        (SOURCE_LABELS, "SOURCE_LABELS"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    if not EXISTING_MODEL.exists():
        print(f"ERROR: EXISTING_MODEL not found: {EXISTING_MODEL}", file=sys.stderr)
        print("Run: python tools/train_disaster_model.py", file=sys.stderr)
        sys.exit(1)

    # Collect image stems
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in SOURCE_IMAGES.iterdir() if p.suffix.lower() in img_extensions
    )
    all_stems = [p.stem for p in image_paths]
    print(f"Found {len(all_stems)} source images.")

    # Step 1
    print("\nStep 1: Remapping existing labels...")
    remapped = remap_labels(SOURCE_LABELS)
    print(f"  Remapped {len(remapped)} label files.")

    # Step 2
    print("\nStep 2: Auto-annotating with existing model...")
    auto = auto_annotate(SOURCE_IMAGES, EXISTING_MODEL, image_paths)
    auto_count = sum(len(v) for v in auto.values())
    print(f"  Generated {auto_count} auto-annotations across {len(auto)} images.")

    # Step 3
    print("\nStep 3: Merging labels...")
    with tempfile.TemporaryDirectory() as tmp:
        temp_labels = Path(tmp) / "labels"
        merge_labels(remapped, auto, all_stems, temp_labels)
        print(f"  Merged labels written to temp dir.")

        # Step 4
        print("\nStep 4: Splitting and copying files...")
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        splits = split_and_copy(all_stems, SOURCE_IMAGES, temp_labels, OUTPUT_DIR, SPLIT)
        print(f"  Split: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")

    # Step 5
    print("\nStep 5: Writing data.yaml...")
    write_yaml(OUTPUT_DIR)

    report(OUTPUT_DIR, splits)


if __name__ == "__main__":
    main()
