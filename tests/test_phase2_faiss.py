"""Phase 2 -- FAISS vocabulary index + delta magnitude tests.

Validates that:
1. VocabIndex builds correctly from CLIP embeddings of all 10 demo images,
   and k=1 self-query returns each image's own label.
2. Intra-group embedding deltas are smaller on average than inter-group
   deltas -- confirming delta compression is viable.
3. Empirical delta magnitude distribution (min/max/mean) for the white
   paper and patent prosecution record.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("open_clip")
pytest.importorskip("faiss")

from hfups_embed_lab.vocab import VocabIndex

# ---------------------------------------------------------------------------
# Shared fixtures (copied from test_embedding_coherence.py to avoid
# cross-test imports that break pytest collection)
# ---------------------------------------------------------------------------

IMAGE_DIR = Path(__file__).parent.parent / "data" / "demo"

GROUPS: dict[str, list[int]] = {
    "coastal_surge": [1, 2],
    "urban_flood":   [5, 9],
    "fire_smoke":    [6, 7],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    return model, preprocess


def _embed_all(model, preprocess) -> dict[str, np.ndarray]:
    """Embed all 10 demo images. Returns {str(img_id): unit_vector}."""
    import torch
    from PIL import Image

    embeddings: dict[str, np.ndarray] = {}
    for img_id in range(1, 11):
        path = IMAGE_DIR / f"{img_id}.jpg"
        image = Image.open(path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(tensor)
        vec = features[0].numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        embeddings[str(img_id)] = vec / norm if norm > 0 else vec
    return embeddings


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vocab_index_builds():
    """k=1 self-query for each image should return its own label."""
    model, preprocess = _load_clip()
    embeddings = _embed_all(model, preprocess)

    index = VocabIndex.build(embeddings)
    assert len(index) == 10

    misses = []
    for label, vec in embeddings.items():
        results = index.query(vec, k=1)
        returned_label = results[0]["label"]
        if returned_label != label:
            misses.append((label, returned_label, results[0]["distance"]))

    if misses:
        for orig, got, dist in misses:
            print(f"  image {orig} -> nearest={got} (dist={dist:.6f})")

    assert len(misses) == 0, (
        f"{len(misses)}/10 images did not self-identify at k=1: {misses}"
    )


def test_delta_magnitudes():
    """Intra-group L2 delta norms must be smaller on average than inter-group."""
    model, preprocess = _load_clip()
    embeddings = _embed_all(model, preprocess)

    group_vecs: dict[str, list[np.ndarray]] = {
        name: [embeddings[str(img_id)] for img_id in ids]
        for name, ids in GROUPS.items()
    }

    intra_norms: list[float] = []
    print("\nIntra-group deltas:")
    for name, vecs in group_vecs.items():
        for a, b in itertools.combinations(vecs, 2):
            norm = _l2(a, b)
            intra_norms.append(norm)
            print(f"  {name}: ||delta|| = {norm:.4f}")

    inter_norms: list[float] = []
    print("\nInter-group deltas:")
    group_names = list(group_vecs.keys())
    for gi, gj in itertools.combinations(range(len(group_names)), 2):
        gi_name, gj_name = group_names[gi], group_names[gj]
        pair_norms = [
            _l2(a, b)
            for a in group_vecs[gi_name]
            for b in group_vecs[gj_name]
        ]
        inter_norms.extend(pair_norms)
        print(f"  {gi_name} vs {gj_name}: mean ||delta|| = {np.mean(pair_norms):.4f}")

    intra_mean = float(np.mean(intra_norms))
    inter_mean = float(np.mean(inter_norms))
    print(f"\nIntra-group mean ||delta||: {intra_mean:.4f}")
    print(f"Inter-group mean ||delta||: {inter_mean:.4f}")
    print(f"Compression advantage ratio: {inter_mean / intra_mean:.2f}x")

    assert intra_mean < inter_mean, (
        f"Expected intra-group deltas smaller than inter-group; "
        f"got intra={intra_mean:.4f} >= inter={inter_mean:.4f}"
    )


def test_delta_magnitude_distribution():
    """Print empirical delta magnitude stats across all 45 image pairs.

    Results are printed for the white paper and patent prosecution record.
    """
    model, preprocess = _load_clip()
    embeddings = _embed_all(model, preprocess)

    labels = list(embeddings.keys())
    all_norms: list[float] = []

    print(f"\nDelta magnitudes for all {len(labels)*(len(labels)-1)//2} pairs:")
    for i, j in itertools.combinations(range(len(labels)), 2):
        la, lb = labels[i], labels[j]
        norm = _l2(embeddings[la], embeddings[lb])
        all_norms.append(norm)
        print(f"  img {la} vs img {lb}: {norm:.4f}")

    print(f"\nMin  ||delta||: {min(all_norms):.4f}")
    print(f"Max  ||delta||: {max(all_norms):.4f}")
    print(f"Mean ||delta||: {float(np.mean(all_norms)):.4f}")
    print(f"Std  ||delta||: {float(np.std(all_norms)):.4f}")

    assert all(n >= 0 for n in all_norms)
