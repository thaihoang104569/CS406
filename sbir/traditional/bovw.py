"""Bag-of-Visual-Words (BoVW) and Spatial Pyramid Matching module.
Includes:
- MiniBatchKMeans codebook training
- TF-IDF weighting (smooth IDF)
- Spatial Pyramid representation (1x1 + 2x2 levels -> 5000-D for K=1000)
- Power normalization (signed square root) and L2 normalization
- Serialization (save / load codebook)
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class VisualCodebook:
    """Trains and applies a visual vocabulary codebook using MiniBatchKMeans with TF-IDF and SPM."""

    def __init__(
        self,
        n_clusters: int = 1000,
        batch_size: int = 4096,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            n_init=3,
        )
        self.idf_weights: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, descriptors: np.ndarray):
        """Fit visual vocabulary codebook on sample descriptors."""
        if len(descriptors) == 0:
            raise ValueError("Cannot fit codebook on empty descriptors array.")
        self.kmeans.fit(descriptors.astype(np.float32))
        self.is_fitted = True

    def assign_words(self, descriptors: np.ndarray) -> np.ndarray:
        """Assign descriptors to nearest cluster centers."""
        if len(descriptors) == 0 or not self.is_fitted:
            return np.empty((0,), dtype=np.int32)
        return self.kmeans.predict(descriptors.astype(np.float32))

    def fit_idf(self, word_histograms: np.ndarray):
        """Computes smooth Inverse Document Frequency (IDF) weights across training image histograms.

        Args:
            word_histograms: 2D array of shape (N_images, n_clusters) containing word frequency counts.
        """
        N = len(word_histograms)
        if N == 0:
            return

        # Document frequency DF(w): count of images where word w appears (count > 0)
        df = np.sum(word_histograms > 0, axis=0)

        # Smooth IDF: log((N + 1) / (DF + 1)) + 1
        self.idf_weights = (np.log((N + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)

    def compute_spatial_pyramid(
        self,
        descriptors: np.ndarray,
        positions: np.ndarray,
        img_size: int = 256,
        levels: List[int] = [1, 2],
        apply_idf: bool = True,
    ) -> np.ndarray:
        """Computes Spatial Pyramid Histogram (Level 1: 1x1, Level 2: 2x2 -> total 5 spatial bins).
        Vector dimension = 5 * n_clusters (5000-D when K=1000).

        Normalizations applied:
        1. TF-IDF weighting (optional, if fitted)
        2. Power normalization: sign(v) * sqrt(|v|)
        3. L2 normalization: v / ||v||_2
        """
        total_cells = sum(lvl * lvl for lvl in levels)
        total_dim = total_cells * self.n_clusters

        if len(descriptors) == 0 or not self.is_fitted:
            return np.zeros(total_dim, dtype=np.float32)

        words = self.assign_words(descriptors)
        histograms = []

        for lvl in levels:
            bin_size = img_size / float(lvl)
            for row in range(lvl):
                for col in range(lvl):
                    # Mask positions falling into this grid cell
                    mask = (
                        (positions[:, 0] >= col * bin_size) & (positions[:, 0] < (col + 1) * bin_size) &
                        (positions[:, 1] >= row * bin_size) & (positions[:, 1] < (row + 1) * bin_size)
                    )
                    cell_words = words[mask]
                    hist, _ = np.histogram(cell_words, bins=self.n_clusters, range=(0, self.n_clusters))
                    hist = hist.astype(np.float32)

                    # Apply TF-IDF if available
                    if apply_idf and self.idf_weights is not None:
                        hist = hist * self.idf_weights

                    histograms.append(hist)

        pyramid_vector = np.concatenate(histograms)

        # Power normalization (signed square root)
        pyramid_vector = np.sign(pyramid_vector) * np.sqrt(np.abs(pyramid_vector))

        # L2 normalization
        norm = np.linalg.norm(pyramid_vector)
        if norm > 1e-12:
            pyramid_vector = pyramid_vector / norm

        return pyramid_vector.astype(np.float32)

    def save(self, filepath: Union[str, Path]):
        """Serializes the codebook (KMeans estimator + IDF weights) to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "n_clusters": self.n_clusters,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "kmeans": self.kmeans,
            "idf_weights": self.idf_weights,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "VisualCodebook":
        """Loads a serialized VisualCodebook from file."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        codebook = cls(
            n_clusters=state["n_clusters"],
            batch_size=state["batch_size"],
            random_state=state["random_state"],
        )
        codebook.is_fitted = state["is_fitted"]
        codebook.idf_weights = state["idf_weights"]
        if state["kmeans"] is not None:
            codebook.kmeans = state["kmeans"]
        return codebook
