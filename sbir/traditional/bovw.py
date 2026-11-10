"""Bag-of-Visual-Words (BoVW) and Spatial Pyramid Matching module.
Includes:
- MiniBatchKMeans codebook training
- TF-IDF weighting
- Spatial Pyramid representation (1x1 + 2x2 levels)
- Power and L2 normalization
"""

from typing import List, Optional
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class VisualCodebook:
    """Trains and applies a visual vocabulary codebook using MiniBatchKMeans."""
    def __init__(self, n_clusters: int = 1000, batch_size: int = 4096, random_state: int = 42):
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            n_init=3
        )
        self.idf_weights = None

    def fit(self, descriptors: np.ndarray):
        """Fit codebook on sample descriptors."""
        self.kmeans.fit(descriptors)

    def assign_words(self, descriptors: np.ndarray) -> np.ndarray:
        """Assign descriptors to nearest cluster centers."""
        if len(descriptors) == 0:
            return np.empty((0,), dtype=np.int32)
        return self.kmeans.predict(descriptors)

    def compute_spatial_pyramid(
        self,
        descriptors: np.ndarray,
        positions: np.ndarray,
        img_size: int = 256,
        levels: List[int] = [1, 2]
    ) -> np.ndarray:
        """Computes Spatial Pyramid Histogram (Level 1: 1x1, Level 2: 2x2 -> total 5 spatial bins).
        Vector dimension = 5 * n_clusters (5000-D when K=1000).
        """
        if len(descriptors) == 0:
            total_bins = sum(lvl * lvl for lvl in levels)
            return np.zeros(total_bins * self.n_clusters, dtype=np.float32)

        words = self.assign_words(descriptors)
        histograms = []

        for lvl in levels:
            bin_size = img_size / lvl
            for row in range(lvl):
                for col in range(lvl):
                    # Mask positions falling into this grid cell
                    mask = (
                        (positions[:, 0] >= col * bin_size) & (positions[:, 0] < (col + 1) * bin_size) &
                        (positions[:, 1] >= row * bin_size) & (positions[:, 1] < (row + 1) * bin_size)
                    )
                    cell_words = words[mask]
                    hist, _ = np.histogram(cell_words, bins=self.n_clusters, range=(0, self.n_clusters))
                    histograms.append(hist.astype(np.float32))

        pyramid_vector = np.concatenate(histograms)

        # Power normalization (signed square root)
        pyramid_vector = np.sign(pyramid_vector) * np.sqrt(np.abs(pyramid_vector))

        # L2 normalization
        norm = np.linalg.norm(pyramid_vector)
        if norm > 1e-12:
            pyramid_vector /= norm

        return pyramid_vector
