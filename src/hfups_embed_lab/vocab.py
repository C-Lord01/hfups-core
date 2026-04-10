"""FAISS vocabulary index for HFUPS embedding lab — Phase 2.

Builds and queries a FAISS flat-L2 index over L2-normalised CLIP ViT-B/32
embeddings. Normalising before indexing gives cosine similarity via L2
distance: for unit vectors, ||a - b||^2 = 2(1 - cos(a, b)).

Usage:
    index = VocabIndex.build({"flood": emb_a, "debris": emb_b, ...})
    results = index.query(query_vec, k=3)
    # [{"label": "flood", "distance": 0.12}, ...]

    index.save("outputs/vocab")
    index = VocabIndex.load("outputs/vocab")
"""

from __future__ import annotations

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise each row. Zero vectors are left as-is."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class VocabIndex:
    """Cosine-similarity vocabulary index backed by FAISS IndexFlatL2.

    Vectors are L2-normalised before insertion so that L2 distance on the
    index is monotonically equivalent to cosine distance.
    """

    def __init__(self, labels: list[str], embeddings: np.ndarray, _index: object) -> None:
        if len(labels) != len(embeddings):
            raise ValueError(
                f"labels length {len(labels)} != embeddings rows {len(embeddings)}"
            )
        self._labels = labels
        self._embeddings = embeddings
        self._index = _index

    @classmethod
    def build(cls, vocab: dict[str, np.ndarray]) -> "VocabIndex":
        """Build the index from a {label: embedding} dict."""
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required: pip install faiss-cpu") from exc

        if not vocab:
            raise ValueError("vocab dict must not be empty")

        labels = list(vocab.keys())
        raw = np.stack([np.asarray(v, dtype=np.float32) for v in vocab.values()])

        embeddings = _normalize(raw)
        d = embeddings.shape[1]

        index = faiss.IndexFlatL2(d)
        index.add(embeddings)

        return cls(labels=labels, embeddings=embeddings, _index=index)

    def query(self, vector: np.ndarray, k: int = 1) -> list[dict]:
        """Return the k nearest vocabulary entries for a query vector."""
        q = np.asarray(vector, dtype=np.float32)
        if q.ndim == 1:
            q = q[np.newaxis, :]
        q = _normalize(q)

        k = min(k, len(self._labels))
        distances, indices = self._index.search(q, k)

        return [
            {"label": self._labels[int(idx)], "distance": float(dist)}
            for dist, idx in zip(distances[0], indices[0])
        ]

    def save(self, path: str) -> None:
        """Serialise index and labels to disk (<path>.faiss + <path>.labels.npy)."""
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required: pip install faiss-cpu") from exc
        faiss.write_index(self._index, f"{path}.faiss")
        np.save(f"{path}.labels.npy", np.array(self._labels))

    @classmethod
    def load(cls, path: str) -> "VocabIndex":
        """Load a VocabIndex previously saved with save()."""
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required: pip install faiss-cpu") from exc

        index = faiss.read_index(f"{path}.faiss")
        labels = np.load(f"{path}.labels.npy", allow_pickle=False).tolist()

        n, d = index.ntotal, index.d
        embeddings = faiss.rev_swig_ptr(index.get_xb(), n * d).reshape(n, d).copy()

        return cls(labels=labels, embeddings=embeddings, _index=index)

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        n = len(self._labels)
        d = self._embeddings.shape[1] if n else 0
        return f"VocabIndex(n={n}, dim={d})"
