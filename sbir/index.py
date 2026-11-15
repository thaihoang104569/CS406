"""Vector index for retrieval (FAISS IndexFlatIP = exact cosine on L2-normalised vectors).

Same interface as before (VectorIndex(dim, use_gpu), add, search), plus:
  * search() runs in query batches, so 12k x 10k queries never materialise a full
    similarity matrix in the NumPy fallback, and top_k > gallery size is clamped
    (FAISS would otherwise return -1 ids);
  * reset(), ntotal, and save()/load() of the stored vectors (.npy) so the demo can
    reload a gallery without re-encoding images;
  * the NumPy fallback uses argpartition (O(N) per query) instead of a full argsort.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class VectorIndex:
    """FAISS wrapper for cosine-similarity retrieval, with a NumPy fallback."""

    def __init__(self, dim: int = 512, use_gpu: bool = False, batch_size: int = 1024):
        self.dim = dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._vectors: Optional[np.ndarray] = None  # kept for save() and the fallback
        self._init_index()

    # ------------------------------------------------------------------ setup
    def _init_index(self):
        try:
            import faiss
            index = faiss.IndexFlatIP(self.dim)
            if self.use_gpu and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
                self._gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
            elif self.use_gpu:
                print("[VectorIndex] FAISS GPU not available, using CPU index.")
                self.use_gpu = False
            self.index = index
        except ImportError:
            self.index = None  # NumPy fallback

    @property
    def backend(self) -> str:
        if self.index is None:
            return "numpy"
        return "faiss-gpu" if self.use_gpu else "faiss-cpu"

    @property
    def ntotal(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def reset(self):
        self._vectors = None
        if self.index is not None:
            self.index.reset()

    # ------------------------------------------------------------------ data
    def add(self, vectors: np.ndarray):
        """Add vectors of shape (N, dim); they are L2-normalised here."""
        vectors = np.atleast_2d(vectors)
        assert vectors.shape[1] == self.dim, f"Expected dim {self.dim}, got {vectors.shape[1]}"
        vectors = l2_normalize(vectors)
        if self.index is not None:
            self.index.add(vectors)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query_vectors: np.ndarray, top_k: int = 200,
               batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k neighbours of each query, in query batches.

        Returns (scores, indices), both (num_queries, min(top_k, ntotal)), sorted by
        descending cosine similarity.
        """
        if self.ntotal == 0:
            raise RuntimeError("VectorIndex is empty: call add() first.")
        q = l2_normalize(np.atleast_2d(query_vectors))
        assert q.shape[1] == self.dim, f"Expected dim {self.dim}, got {q.shape[1]}"
        k = max(1, min(int(top_k), self.ntotal))
        bs = batch_size or self.batch_size

        scores = np.empty((len(q), k), dtype=np.float32)
        ids = np.empty((len(q), k), dtype=np.int64)
        for s in range(0, len(q), bs):
            chunk = q[s:s + bs]
            if self.index is not None:
                sc, ix = self.index.search(chunk, k)
            else:
                sims = chunk @ self._vectors.T
                part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                part_s = np.take_along_axis(sims, part, axis=1)
                order = np.argsort(-part_s, axis=1)
                ix = np.take_along_axis(part, order, axis=1)
                sc = np.take_along_axis(part_s, order, axis=1)
            scores[s:s + len(chunk)], ids[s:s + len(chunk)] = sc, ix
        return scores, ids

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        """Save the (normalised) gallery vectors as float16 .npy (~10 MB for 10k x 512)."""
        if self._vectors is None:
            raise RuntimeError("Nothing to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._vectors.astype(np.float16))

    @classmethod
    def load(cls, path: str, use_gpu: bool = False, batch_size: int = 1024) -> "VectorIndex":
        vectors = np.load(path).astype(np.float32)
        idx = cls(dim=vectors.shape[1], use_gpu=use_gpu, batch_size=batch_size)
        idx.add(vectors)
        return idx
