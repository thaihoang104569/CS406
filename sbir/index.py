"""Vector Indexing and Retrieval module using FAISS.
Uses IndexFlatIP for exact cosine similarity search on normalized vectors.
"""

from typing import Tuple
import numpy as np


class VectorIndex:
    """FAISS wrapper for vector indexing and cosine similarity retrieval.
    """
    def __init__(self, dim: int = 512, use_gpu: bool = False):
        self.dim = dim
        self.use_gpu = use_gpu
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dim)
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        except ImportError:
            # Fallback if FAISS is not installed
            self.index = None
            self.stored_vectors = None

    def add(self, vectors: np.ndarray):
        """Add normalized float32 vectors to index.
        Shape: (N, dim)
        """
        assert vectors.shape[1] == self.dim, f"Expected dim {self.dim}, got {vectors.shape[1]}"
        vectors = vectors.astype(np.float32)
        
        # Ensure vectors are L2-normalized for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

        if self.index is not None:
            self.index.add(vectors)
        else:
            if self.stored_vectors is None:
                self.stored_vectors = vectors
            else:
                self.stored_vectors = np.vstack([self.stored_vectors, vectors])

    def search(self, query_vectors: np.ndarray, top_k: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Search top_k nearest neighbors for queries.
        
        Returns:
            scores: shape (num_queries, top_k)
            indices: shape (num_queries, top_k)
        """
        query_vectors = query_vectors.astype(np.float32)
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors / np.clip(norms, 1e-12, None)

        if self.index is not None:
            scores, indices = self.index.search(query_vectors, top_k)
            return scores, indices
        else:
            # Fallback to NumPy matrix multiplication
            sims = np.dot(query_vectors, self.stored_vectors.T)
            top_indices = np.argsort(-sims, axis=1)[:, :top_k]
            top_scores = np.take_along_axis(sims, top_indices, axis=1)
            return top_scores, top_indices
