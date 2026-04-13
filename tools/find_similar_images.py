"""Find flood pool images most visually similar to HFUPS demo images via CLIP ViT-B/32."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("Run: pip install torch torchvision")
    sys.exit(1)

try:
    import clip
except ImportError:
    print("Run: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_IMAGES = [
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\HFUPS Repo\data\demo\1.jpg",
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\HFUPS Repo\data\demo\2.jpg",
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\HFUPS Repo\data\demo\4.jpg",
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\HFUPS Repo\data\demo\6.jpg",
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\HFUPS Repo\data\demo\9.jpg",
]

DEMO_LABELS = ["demo_1", "demo_2", "demo_4", "demo_6", "demo_9"]

DEMO_DESCRIPTIONS = {
    "demo_1": "urban flood",
    "demo_2": "urban flood",
    "demo_4": "coastal surge",
    "demo_6": "coastal surge",
    "demo_9": "flooded bridge",
}

POOL_DIR = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\Flood.v1i.yolov8\train\images"
)
POOL_LABELS_DIR = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\Flood.v1i.yolov8\train\labels"
)
OUTPUT_DIR = Path(
    r"C:\Users\Chris\OneDrive\Documents\Hackerthons\HFUPS\Datasets\similarity_matches"
)

TOP_K = 300
BATCH_SIZE = 64
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Source dataset original class names (0=car, 1=house, 2=person)
SOURCE_CLASS_NAMES = {0: "car", 1: "house", 2: "person"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def parse_existing_classes(label_path: Path) -> str:
    """Return comma-separated class names present in the label file."""
    if not label_path.exists():
        return ""
    present: set[str] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cls_id = int(line.split()[0])
        name = SOURCE_CLASS_NAMES.get(cls_id)
        if name:
            present.add(name)
    return ",".join(sorted(present))


# ---------------------------------------------------------------------------
# Step 1: Load CLIP
# ---------------------------------------------------------------------------

def load_clip():
    print(f"Loading CLIP ViT-B/32 on {DEVICE}...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    print(f"CLIP loaded. Device: {DEVICE}")
    return model, preprocess


# ---------------------------------------------------------------------------
# Step 2: Embed pool images
# ---------------------------------------------------------------------------

def embed_pool(model, preprocess) -> tuple[np.ndarray, list[str]]:
    pool_paths = sorted(
        p for p in POOL_DIR.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not pool_paths:
        print(f"ERROR: POOL_DIR is empty: {POOL_DIR}", file=sys.stderr)
        sys.exit(1)

    filenames: list[str] = []
    embeddings: list[np.ndarray] = []

    batches = [pool_paths[i:i + BATCH_SIZE] for i in range(0, len(pool_paths), BATCH_SIZE)]

    with torch.no_grad():
        for batch_paths in tqdm(batches, desc="Embedding pool", unit="batch"):
            imgs = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = preprocess(Image.open(p).convert("RGB"))
                    imgs.append(img)
                    valid_paths.append(p)
                except Exception:
                    pass  # skip unreadable images

            if not imgs:
                continue

            batch_tensor = torch.stack(imgs).to(DEVICE)
            feats = model.encode_image(batch_tensor).float()
            feats = feats.cpu().numpy()

            for path, vec in zip(valid_paths, feats):
                filenames.append(path.name)
                embeddings.append(vec)

    pool_emb = l2_normalize(np.stack(embeddings))
    print(f"Pool embedded: {len(filenames)} images")
    return pool_emb, filenames


# ---------------------------------------------------------------------------
# Step 3: Embed demo images
# ---------------------------------------------------------------------------

def embed_demos(model, preprocess) -> list[tuple[str, np.ndarray]]:
    results: list[tuple[str, np.ndarray]] = []
    with torch.no_grad():
        for path_str, label in zip(DEMO_IMAGES, DEMO_LABELS):
            p = Path(path_str)
            if not p.exists():
                print(f"ERROR: Demo image not found: {p}", file=sys.stderr)
                sys.exit(1)
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(img).float().cpu().numpy()[0]
            feat = feat / max(np.linalg.norm(feat), 1e-8)
            results.append((label, feat))
    print(f"Demo images embedded: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Step 4 & 5: Similarity search + deduplicate
# ---------------------------------------------------------------------------

def search_and_dedup(
    demo_embeddings: list[tuple[str, np.ndarray]],
    pool_emb: np.ndarray,
    pool_filenames: list[str],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Returns:
        deduped: list of {pool_filename, similarity_score, closest_demo_image}
        per_demo: {label: top-K list of dicts}
    """
    per_demo: dict[str, list[dict]] = {}
    # best_score[pool_filename] = {score, demo_label}
    best: dict[str, dict] = {}

    for label, demo_vec in demo_embeddings:
        sims = pool_emb @ demo_vec  # shape (N,)
        top_indices = np.argsort(sims)[::-1][:TOP_K]

        hits = []
        for idx in top_indices:
            fname = pool_filenames[idx]
            score = float(sims[idx])
            hits.append({"pool_filename": fname, "similarity_score": score, "closest_demo_image": label})
            if fname not in best or score > best[fname]["similarity_score"]:
                best[fname] = {"pool_filename": fname, "similarity_score": score, "closest_demo_image": label}

        per_demo[label] = hits

    deduped = sorted(best.values(), key=lambda x: x["similarity_score"], reverse=True)
    print(f"Total unique after dedup: {len(deduped)}")
    return deduped, per_demo


# ---------------------------------------------------------------------------
# Step 6: Copy output
# ---------------------------------------------------------------------------

def copy_output(deduped: list[dict]) -> None:
    out_img = OUTPUT_DIR / "images"
    out_lbl = OUTPUT_DIR / "labels"
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    csv_path = OUTPUT_DIR / "matches.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pool_filename", "similarity_score", "closest_demo_image", "existing_classes_present"],
        )
        writer.writeheader()

        for row in deduped:
            fname = row["pool_filename"]
            src_img = POOL_DIR / fname
            stem = Path(fname).stem
            lbl_src = POOL_LABELS_DIR / f"{stem}.txt"

            # Copy image
            if src_img.exists():
                shutil.copy2(src_img, out_img / fname)

            # Copy or create label
            if lbl_src.exists():
                shutil.copy2(lbl_src, out_lbl / f"{stem}.txt")
            else:
                (out_lbl / f"{stem}.txt").write_text("", encoding="utf-8")

            existing_classes = parse_existing_classes(lbl_src)
            writer.writerow({
                "pool_filename": fname,
                "similarity_score": f"{row['similarity_score']:.6f}",
                "closest_demo_image": row["closest_demo_image"],
                "existing_classes_present": existing_classes,
            })

    print(f"Copied to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Step 7: Report
# ---------------------------------------------------------------------------

def report(deduped: list[dict], per_demo: dict[str, list[dict]]) -> None:
    col = 30
    header = (
        f"{'Demo image':<{col}} {'Matches':>8}   {'Top sim':>8}   {'Bot sim':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for label in DEMO_LABELS:
        desc = DEMO_DESCRIPTIONS.get(label, "")
        display = f"{label} ({desc})"
        hits = per_demo.get(label, [])
        n = len(hits)
        top_sim = hits[0]["similarity_score"] if hits else 0.0
        bot_sim = hits[-1]["similarity_score"] if hits else 0.0
        print(f"{display:<{col}} {n:>8,}   {top_sim:>8.4f}   {bot_sim:>8.4f}")

    print("-" * len(header))
    print(f"Total unique after dedup: {len(deduped):,}")
    print(f"Copied to: {OUTPUT_DIR}")

    print()
    for label in DEMO_LABELS:
        desc = DEMO_DESCRIPTIONS.get(label, "")
        print(f"\nTop 5 — {label} ({desc}):")
        for hit in per_demo.get(label, [])[:5]:
            print(f"  {hit['pool_filename']:<60}  sim={hit['similarity_score']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not POOL_DIR.exists():
        print(f"ERROR: POOL_DIR not found: {POOL_DIR}", file=sys.stderr)
        sys.exit(1)

    model, preprocess = load_clip()

    print("\n--- Step 2: Embedding pool images ---")
    pool_emb, pool_filenames = embed_pool(model, preprocess)

    print("\n--- Step 3: Embedding demo images ---")
    demo_embeddings = embed_demos(model, preprocess)

    print("\n--- Step 4 & 5: Similarity search + dedup ---")
    deduped, per_demo = search_and_dedup(demo_embeddings, pool_emb, pool_filenames)

    print("\n--- Step 6: Copying output ---")
    copy_output(deduped)

    print("\n--- Step 7: Report ---")
    report(deduped, per_demo)


if __name__ == "__main__":
    main()
