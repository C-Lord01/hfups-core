"""Score and rank flood.v1i images by flood likelihood, copy top candidates.

Two signals:
  1. Color heuristic (HSV brown/grey water detection) — fast, CPU parallel
  2. Model confidence (yolov8s_disaster class-0 flood detections)

Combined: flood_score = color_score * 0.6 + model_score * 0.4
"""

from __future__ import annotations

import csv
import os
import shutil
import statistics
from multiprocessing import Pool, cpu_count
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
OUTPUT_DIR = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\sagemaker_candidates"
)
TOP_N = 1500
MODEL_PATH = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\HFUPS Repo\models\yolov8s_disaster.pt"
)
CONF_THRESHOLD = 0.15
BATCH_SIZE = 32

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Signal 1 — color heuristic (multiprocessing-safe top-level function)
# ---------------------------------------------------------------------------

def _color_score_one(img_path_str: str) -> tuple[str, float]:
    """Compute color_score for a single image. Must be top-level for pickling."""
    import cv2
    import numpy as np

    img = cv2.imread(img_path_str)
    if img is None:
        return img_path_str, 0.0

    img_resized = cv2.resize(img, (320, 320))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Brown / muddy water
    brown = (
        (h >= 10) & (h <= 30) &
        (s >= 50) & (s <= 255) &
        (v >= 50) & (v <= 200)
    )
    # Grey / blue water
    grey = (
        (h >= 90) & (h <= 130) &
        (s >= 30) & (s <= 200) &
        (v >= 50) & (v <= 220)
    )

    total = 320 * 320
    ratio = (brown.sum() + grey.sum()) / total
    score = min(ratio * 10.0, 1.0)
    return img_path_str, float(score)


def compute_color_scores(image_paths: list[Path]) -> dict[str, float]:
    """Run color scoring in parallel across all images."""
    from tqdm import tqdm

    path_strs = [str(p) for p in image_paths]
    workers = min(cpu_count(), 8)
    scores: dict[str, float] = {}

    with Pool(workers) as pool:
        for path_str, score in tqdm(
            pool.imap_unordered(_color_score_one, path_strs, chunksize=50),
            total=len(path_strs),
            desc="Color scoring",
            unit="img",
        ):
            scores[Path(path_str).stem] = score

    return scores


# ---------------------------------------------------------------------------
# Signal 2 — model inference
# ---------------------------------------------------------------------------

def compute_model_scores(image_paths: list[Path]) -> dict[str, float]:
    """Run batch inference, return max class-0 confidence per image."""
    from tqdm import tqdm
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    scores: dict[str, float] = {}

    batches = [image_paths[i:i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Model scoring", unit="batch"):
        results = model.predict(
            [str(p) for p in batch],
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        for img_path, result in zip(batch, results):
            max_flood_conf = 0.0
            if result.boxes is not None and len(result.boxes) > 0:
                for cls_tensor, conf_tensor in zip(result.boxes.cls, result.boxes.conf):
                    if int(cls_tensor.item()) == 0:
                        max_flood_conf = max(max_flood_conf, float(conf_tensor.item()))
            scores[img_path.stem] = max_flood_conf

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    if not SOURCE_IMAGES.exists():
        print(f"ERROR: SOURCE_IMAGES not found: {SOURCE_IMAGES}", file=sys.stderr)
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"ERROR: MODEL_PATH not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(
        p for p in SOURCE_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
    )
    print(f"Found {len(image_paths):,} images to score.")

    # Signal 1
    print("\n--- Signal 1: Color heuristic ---")
    color_scores = compute_color_scores(image_paths)

    # Signal 2
    print("\n--- Signal 2: Model inference ---")
    model_scores = compute_model_scores(image_paths)

    # Combine
    rows: list[dict] = []
    for img_path in image_paths:
        stem = img_path.stem
        cs = color_scores.get(stem, 0.0)
        ms = model_scores.get(stem, 0.0)
        fs = cs * 0.6 + ms * 0.4
        rows.append({
            "filename": img_path.name,
            "flood_score": fs,
            "color_score": cs,
            "model_score": ms,
        })

    rows.sort(key=lambda r: r["flood_score"], reverse=True)

    # Copy top N
    out_img_dir = OUTPUT_DIR / "images"
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for row in rows[:TOP_N]:
        src = SOURCE_IMAGES / row["filename"]
        shutil.copy2(src, out_img_dir / row["filename"])

    # Write CSV
    csv_path = OUTPUT_DIR / "candidates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "flood_score", "color_score", "model_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "filename": row["filename"],
                "flood_score": f"{row['flood_score']:.6f}",
                "color_score": f"{row['color_score']:.6f}",
                "model_score": f"{row['model_score']:.6f}",
            })

    # Summary
    all_scores = [r["flood_score"] for r in rows]
    print("\n=== Summary ===")
    print(f"Total images scored: {len(rows):,}")
    print(f"Score distribution:")
    print(f"  min:    {min(all_scores):.4f}")
    print(f"  max:    {max(all_scores):.4f}")
    print(f"  mean:   {statistics.mean(all_scores):.4f}")
    print(f"  median: {statistics.median(all_scores):.4f}")
    print(f"Images with flood_score > 0.5: {sum(1 for s in all_scores if s > 0.5):,}")
    print(f"Images with flood_score > 0.3: {sum(1 for s in all_scores if s > 0.3):,}")
    print(f"\nTop {TOP_N} candidates copied to: {out_img_dir}")
    print(f"candidates.csv written to:       {csv_path}")

    print("\nTop 10 candidates:")
    print(f"  {'filename':<55} {'flood':>7} {'color':>7} {'model':>7}")
    print("  " + "-" * 80)
    for row in rows[:10]:
        print(
            f"  {row['filename']:<55}"
            f" {row['flood_score']:>7.4f}"
            f" {row['color_score']:>7.4f}"
            f" {row['model_score']:>7.4f}"
        )


if __name__ == "__main__":
    main()
