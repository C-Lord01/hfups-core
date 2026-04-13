"""Extract frames from MP4 videos, auto-annotate with disaster model, write to flood_v2."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_DIR = Path(r"C:\Users\Chris\OneDrive\Documents\Hackerthons\Videos to split")
FLOOD_V2_DIR = Path(r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\flood_v2")
MODEL_PATH = Path(r"models/yolov8s_disaster.pt")
CONF_THRESHOLD = 0.20
FRAME_INTERVAL = 3
MIN_ANNOTATIONS = 0
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
BATCH_SIZE = 32

CVAT_DIR = Path(r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\cvat_unlabeled")

SCENE_HINTS = {
    "Beach Carpark_Waves": [4, 6, 12, 13, 14],
    "Flood Bridge":        [0, 2, 3, 14],
    "Person Submerge Car": [0, 1, 5, 6, 7, 8, 9],
}

CLASS_NAMES = {
    0:  "flood",
    1:  "flooded_area",
    2:  "flooded_road",
    3:  "flooded_bridge",
    4:  "flooded_carpark",
    5:  "submerged_car",
    6:  "car",
    7:  "person",
    8:  "person_on_vehicle",
    9:  "person_in_water",
    10: "residential_building",
    12: "carpark",
    13: "ocean",
    14: "waves",
    15: "debris_floating",
}

# Annotation counts before this run (from merge_all_sources.py output)
BEFORE_COUNTS = {
    0: 1040, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    6: 9757, 7: 24480, 8: 0, 9: 0, 10: 14038,
    12: 0, 13: 0, 14: 0, 15: 0,
}
BEFORE_SPLIT_COUNTS = {"train": 9579, "valid": 1956, "test": 1847}

OK_THRESHOLD = 300
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def determine_split(stem: str) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "valid"
    else:
        return "test"


def scene_prefix(video_stem: str) -> str:
    """Extract scene prefix: everything before the trailing space+number."""
    parts = video_stem.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return video_stem


# ---------------------------------------------------------------------------
# Step 1: Discover videos
# ---------------------------------------------------------------------------

def discover_videos() -> dict[str, list[Path]]:
    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not mp4_files:
        print(f"ERROR: No .mp4 files found in {VIDEO_DIR}", file=sys.stderr)
        sys.exit(1)

    grouped: dict[str, list[Path]] = defaultdict(list)
    for p in mp4_files:
        grouped[scene_prefix(p.stem)].append(p)

    print(f"\n--- Step 1: Discovered {len(mp4_files)} videos ---")
    for scene, paths in grouped.items():
        print(f"  [{scene}]")
        for p in paths:
            print(f"    {p.name}")

    return dict(grouped)


# ---------------------------------------------------------------------------
# Step 2: Extract frames
# ---------------------------------------------------------------------------

def extract_frames(
    grouped: dict[str, list[Path]],
    temp_dir: Path,
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """Extract every FRAME_INTERVAL-th frame.
    Returns ({scene: [frame_paths]}, {scene: frame_count}).
    """
    try:
        import cv2
    except ImportError:
        print("Run: pip install opencv-python")
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    temp_dir.mkdir(parents=True, exist_ok=True)
    scene_frames: dict[str, list[Path]] = defaultdict(list)
    scene_frame_counts: dict[str, int] = defaultdict(int)
    total_extracted = 0

    print(f"\n--- Step 2: Extracting frames (every {FRAME_INTERVAL}th) ---")

    for scene, videos in grouped.items():
        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            expected = total_frames // FRAME_INTERVAL
            print(f"  {video_path.name}: {total_frames} frames @ {fps:.1f} fps -> ~{expected} extractions")

            frame_idx = 0
            extracted = 0
            pbar = tqdm(total=expected, desc=f"  {video_path.stem[:40]}", unit="frame", leave=False)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % FRAME_INTERVAL == 0:
                    out_name = f"{video_path.stem}_frame_{frame_idx:05d}.jpg"
                    out_path = temp_dir / out_name
                    cv2.imwrite(str(out_path), frame)
                    scene_frames[scene].append(out_path)
                    extracted += 1
                    pbar.update(1)
                frame_idx += 1

            pbar.close()
            cap.release()
            scene_frame_counts[scene] += extracted
            total_extracted += extracted
            print(f"    Extracted: {extracted:,} frames")

    print(f"\nTotal frames extracted: {total_extracted:,}")
    return dict(scene_frames), dict(scene_frame_counts)


# ---------------------------------------------------------------------------
# Step 2b: Copy frames to CVAT import directories
# ---------------------------------------------------------------------------

# Map scene names to safe directory names
SCENE_DIR_NAMES = {
    "Beach Carpark_Waves": "Beach_Carpark_Waves",
    "Flood Bridge":        "Flood_Bridge",
    "Person Submerge Car": "Person_Submerge_Car",
}


def copy_to_cvat(scene_frames: dict[str, list[Path]]) -> None:
    """Copy all extracted frames to CVAT_DIR organized by scene."""
    print(f"\n--- Step 2b: Copying frames to CVAT import dirs ---")

    if CVAT_DIR.exists():
        shutil.rmtree(CVAT_DIR)

    for scene, frames in scene_frames.items():
        dir_name = SCENE_DIR_NAMES.get(scene, scene.replace(" ", "_"))
        dest_dir = CVAT_DIR / dir_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in frames:
            shutil.copy2(frame_path, dest_dir / frame_path.name)

        print(f"  {dir_name}: {len(frames):,} frames")

    print(f"CVAT import folders written to: {CVAT_DIR}")


# ---------------------------------------------------------------------------
# Step 3: Auto-annotate
# ---------------------------------------------------------------------------

def auto_annotate(
    scene_frames: dict[str, list[Path]],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Run YOLO inference on all frames.
    Returns:
        labels: {frame_stem: [yolo_lines]}
        discard_count: {"total": N}
    """
    if not MODEL_PATH.exists():
        print(f"ERROR: MODEL_PATH not found: {MODEL_PATH}", file=sys.stderr)
        print("Run: python tools/train_disaster_model.py", file=sys.stderr)
        sys.exit(1)

    from ultralytics import YOLO
    from tqdm import tqdm

    model = YOLO(str(MODEL_PATH))

    all_frames: list[Path] = []
    for frames in scene_frames.values():
        all_frames.extend(frames)

    labels: dict[str, list[str]] = {}

    batches = [all_frames[i:i + BATCH_SIZE] for i in range(0, len(all_frames), BATCH_SIZE)]

    print(f"\n--- Step 3: Auto-annotating {len(all_frames):,} frames ---")

    for batch in tqdm(batches, desc="Annotating", unit="batch"):
        results = model.predict(
            [str(p) for p in batch],
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )
        for frame_path, result in zip(batch, results):
            stem = frame_path.stem
            lines: list[str] = []

            if result.boxes is not None and len(result.boxes) > 0:
                for cls_t, xywhn_t, conf_t in zip(
                    result.boxes.cls,
                    result.boxes.xywhn,
                    result.boxes.conf,
                ):
                    cls_id = int(cls_t.item())
                    if float(conf_t.item()) < CONF_THRESHOLD:
                        continue
                    cx, cy, w, h = xywhn_t.tolist()
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if MIN_ANNOTATIONS > 0 and len(lines) < MIN_ANNOTATIONS:
                pass  # discard: frame has fewer detections than required minimum
            else:
                labels[stem] = lines

    annotated = sum(1 for v in labels.values() if v)
    unannotated = len(all_frames) - annotated
    print(f"Total frames with detections: {annotated:,}")
    print(f"Frames with no detections (will get empty labels): {unannotated:,}")

    return labels, {"total": unannotated}


# ---------------------------------------------------------------------------
# Step 4: Split and write
# ---------------------------------------------------------------------------

def split_and_write(
    scene_frames: dict[str, list[Path]],
    labels: dict[str, list[str]],
) -> tuple[dict[str, dict], dict[str, int], int, int]:
    """
    Copy images + write labels to flood_v2.
    Returns (scene_stats, split_added, frames_with_annotations, frames_empty_labels).
    """
    split_added: dict[str, int] = defaultdict(int)
    scene_stats: dict[str, dict] = {}
    frames_with_annotations = 0
    frames_empty_labels = 0

    for scene, frames in scene_frames.items():
        added = 0
        class_counts: dict[int, int] = defaultdict(int)

        for frame_path in frames:
            stem = frame_path.stem

            split = determine_split(stem)
            dest_img_dir = FLOOD_V2_DIR / split / "images"
            dest_lbl_dir = FLOOD_V2_DIR / split / "labels"
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir.mkdir(parents=True, exist_ok=True)

            dest_img = dest_img_dir / frame_path.name
            dest_lbl = dest_lbl_dir / f"{stem}.txt"

            if dest_img.exists():
                continue  # idempotent

            shutil.copy2(frame_path, dest_img)
            lines = labels.get(stem, [])
            dest_lbl.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

            if lines:
                frames_with_annotations += 1
                for line in lines:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
            else:
                frames_empty_labels += 1

            added += 1
            split_added[split] += 1

        scene_stats[scene] = {"added": added, "class_counts": dict(class_counts)}

    return scene_stats, dict(split_added), frames_with_annotations, frames_empty_labels


# ---------------------------------------------------------------------------
# Step 5: Report
# ---------------------------------------------------------------------------

def full_report(
    scene_frame_counts: dict[str, int],
    total_extracted: int,
    total_unannotated: int,
    scene_stats: dict[str, dict],
    split_added: dict[str, int],
    frames_with_annotations: int = 0,
    frames_empty_labels: int = 0,
) -> None:
    frames_added = sum(s["added"] for s in scene_stats.values())

    print("\n=== Post-Extraction Dataset Report ===\n")
    print(f"Frames extracted: {total_extracted:,}")
    print(f"Frames with model detections: {total_extracted - total_unannotated:,}")
    print(f"Frames with no model detections (empty labels): {total_unannotated:,}")
    print(f"Frames added to flood_v2: {frames_added:,}")
    if frames_with_annotations or frames_empty_labels:
        print(f"  Frames written with annotations: {frames_with_annotations:,}")
        print(f"  Frames written with empty label files: {frames_empty_labels:,}")
    print(f"CVAT import folders written to: {CVAT_DIR}")
    for scene, count in scene_frame_counts.items():
        dir_name = SCENE_DIR_NAMES.get(scene, scene.replace(" ", "_"))
        print(f"  {dir_name}: {count:,} frames")

    # Split counts
    print(f"\n{'Split':<10} {'Images (before)':>16} {'Images (after)':>15}")
    print("-" * 45)
    total_before = sum(BEFORE_SPLIT_COUNTS.values())
    total_after = 0
    for split in ("train", "valid", "test"):
        before = BEFORE_SPLIT_COUNTS[split]
        after = before + split_added.get(split, 0)
        total_after += after
        print(f"{split:<10} {before:>16,} {after:>15,}")
    print(f"{'total':<10} {total_before:>16,} {total_after:>15,}")

    # Scan current annotation counts
    after_counts: dict[int, int] = defaultdict(int)
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
                    after_counts[int(line.split()[0])] += 1

    print(f"\n{'ID':<5} {'Class':<24} {'Ann. before':>12} {'Ann. after':>11} {'Delta':>7}  Status")
    print("-" * 70)
    for cls_id in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[cls_id]
        before = BEFORE_COUNTS.get(cls_id, 0)
        after = after_counts.get(cls_id, 0)
        delta = after - before
        if after == 0:
            status = "MISSING"
        elif after < OK_THRESHOLD:
            status = "SPARSE"
        else:
            status = "ok"
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        print(f"{cls_id:<5} {name:<24} {before:>12,} {after:>11,} {delta_str:>7}  {status}")

    # Per-scene contribution
    print(f"\n{'Scene':<25} {'Frames added':>13}  Top classes detected")
    print("-" * 70)
    for scene, stats in scene_stats.items():
        added = stats["added"]
        cc = stats["class_counts"]
        top = sorted(cc.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(CLASS_NAMES.get(c, str(c)) for c, _ in top) if top else "(none)"
        print(f"{scene:<25} {added:>13,}  {top_str}")


# ---------------------------------------------------------------------------
# Report-only mode: scan existing flood_v2 and CVAT dirs without re-running
# ---------------------------------------------------------------------------

def report_only() -> None:
    """Scan flood_v2 and CVAT dirs, print Step 5 report without re-extraction."""
    print("\n--- Report-only mode: scanning existing data ---")

    # Count images per split
    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        img_dir = FLOOD_V2_DIR / split / "images"
        split_counts[split] = sum(
            1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
        ) if img_dir.exists() else 0

    # Count CVAT frames per scene
    scene_frame_counts: dict[str, int] = {}
    for scene, dir_name in SCENE_DIR_NAMES.items():
        cvat_scene_dir = CVAT_DIR / dir_name
        if cvat_scene_dir.exists():
            scene_frame_counts[scene] = sum(
                1 for p in cvat_scene_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
            )
        else:
            scene_frame_counts[scene] = 0
    total_cvat = sum(scene_frame_counts.values())

    # Count empty vs annotated label files added from video frames
    # Identify them by stem pattern: contains "_frame_"
    empty_count = 0
    annotated_count = 0
    for split in ("train", "valid", "test"):
        lbl_dir = FLOOD_V2_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.iterdir():
            if lbl_file.suffix != ".txt" or "_frame_" not in lbl_file.stem:
                continue
            content = lbl_file.read_text(encoding="utf-8").strip()
            if content:
                annotated_count += 1
            else:
                empty_count += 1
    frames_added = empty_count + annotated_count

    # Annotation counts across all labels
    after_counts: dict[int, int] = defaultdict(int)
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
                    after_counts[int(line.split()[0])] += 1

    # Build scene_stats shell from CVAT counts (no per-class breakdown available)
    scene_stats = {scene: {"added": 0, "class_counts": {}} for scene in SCENE_DIR_NAMES}

    # Print
    print("\n=== Post-Extraction Dataset Report ===\n")
    print(f"Frames added to flood_v2 (video frames): {frames_added:,}")
    print(f"  Frames written with annotations: {annotated_count:,}")
    print(f"  Frames written with empty label files: {empty_count:,}")
    print(f"CVAT import folders: {CVAT_DIR}")
    for scene, count in scene_frame_counts.items():
        dir_name = SCENE_DIR_NAMES.get(scene, scene.replace(" ", "_"))
        print(f"  {dir_name}: {count:,} frames")
    print(f"  Total: {total_cvat:,} frames")

    print(f"\n{'Split':<10} {'Images (before)':>16} {'Images (after)':>15}")
    print("-" * 45)
    total_before = sum(BEFORE_SPLIT_COUNTS.values())
    total_after = sum(split_counts.values())
    for split in ("train", "valid", "test"):
        print(f"{split:<10} {BEFORE_SPLIT_COUNTS[split]:>16,} {split_counts[split]:>15,}")
    print(f"{'total':<10} {total_before:>16,} {total_after:>15,}")

    print(f"\n{'ID':<5} {'Class':<24} {'Ann. before':>12} {'Ann. after':>11} {'Delta':>7}  Status")
    print("-" * 70)
    for cls_id in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[cls_id]
        before = BEFORE_COUNTS.get(cls_id, 0)
        after = after_counts.get(cls_id, 0)
        delta = after - before
        status = "MISSING" if after == 0 else ("SPARSE" if after < OK_THRESHOLD else "ok")
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        print(f"{cls_id:<5} {name:<24} {before:>12,} {after:>11,} {delta_str:>7}  {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="Scan existing flood_v2 and CVAT dirs, print report only")
    args = parser.parse_args()

    if args.report_only:
        report_only()
        return

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("Run: pip install opencv-python")
        sys.exit(1)

    if not VIDEO_DIR.exists():
        print(f"ERROR: VIDEO_DIR not found: {VIDEO_DIR}", file=sys.stderr)
        sys.exit(1)

    # Step 1
    grouped = discover_videos()

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp) / "frames"

        # Step 2
        scene_frames, scene_frame_counts = extract_frames(grouped, temp_dir)
        total_extracted = sum(len(v) for v in scene_frames.values())

        # Step 2b: CVAT copy (before annotation)
        copy_to_cvat(scene_frames)

        # Step 3
        labels, discard_info = auto_annotate(scene_frames)
        unannotated = discard_info["total"]

        # Step 4
        print(f"\n--- Step 4: Writing to flood_v2 ---")
        scene_stats, split_added, frames_with_ann, frames_empty = split_and_write(
            scene_frames, labels
        )
        print(f"  Written: {sum(s['added'] for s in scene_stats.values()):,} frames")
        print(f"  With annotations: {frames_with_ann:,}")
        print(f"  Empty label files: {frames_empty:,}")

    # Step 5
    full_report(
        scene_frame_counts, total_extracted, unannotated,
        scene_stats, split_added, frames_with_ann, frames_empty,
    )


if __name__ == "__main__":
    main()
