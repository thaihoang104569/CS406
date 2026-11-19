"""Category-level triplet sampling for E3/E4 (no torch: unit-testable anywhere).

Each epoch visits every training sketch once (shuffled). For sketch i of class y:
  positive = a random photo of class y   (any photo of the class, incl. extended photos:
                                          category-level SBIR, the sketch's own source photo
                                          is not required)
  negative = a random photo of another class (uniform over all photos not of class y)
Triplets are drawn in the main process from (seed, epoch), so runs are reproducible and
DataLoader workers only load images.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


class TripletSampler:
    def __init__(self, sketch_labels: Sequence[int], photo_labels: Sequence[int], seed: int = 42):
        self.sketch_labels = np.asarray(sketch_labels, dtype=np.int64)
        self.photo_labels = np.asarray(photo_labels, dtype=np.int64)
        self.seed = seed
        self.by_class: Dict[int, np.ndarray] = {
            int(c): np.flatnonzero(self.photo_labels == c) for c in np.unique(self.photo_labels)
        }
        missing = set(np.unique(self.sketch_labels).tolist()) - set(self.by_class)
        if missing:
            raise ValueError(f"Classes with sketches but no photos: {sorted(missing)[:5]}")
        if len(self.by_class) < 2:
            raise ValueError("Need at least two classes to draw negatives.")

    def __len__(self) -> int:
        return len(self.sketch_labels)

    def epoch_triplets(self, epoch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(sketch_idx, pos_photo_idx, neg_photo_idx) for one epoch, sketch order shuffled."""
        rng = np.random.default_rng([self.seed, epoch])
        s_idx = rng.permutation(len(self.sketch_labels))
        y = self.sketch_labels[s_idx]
        pos = np.empty(len(s_idx), dtype=np.int64)
        for c in np.unique(y):
            m = y == c
            pos[m] = rng.choice(self.by_class[int(c)], size=int(m.sum()))
        # negatives: uniform over all photos, resample the ones that hit the anchor's class
        neg = rng.integers(0, len(self.photo_labels), size=len(s_idx))
        bad = self.photo_labels[neg] == y
        while bad.any():
            neg[bad] = rng.integers(0, len(self.photo_labels), size=int(bad.sum()))
            bad = self.photo_labels[neg] == y
        return s_idx, pos, neg


def debug_subset(categories: Sequence[str], n_classes: int, seed: int) -> List[str]:
    """Deterministic subset of class names for --debug runs."""
    cats = sorted(set(categories))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(cats, size=min(n_classes, len(cats)), replace=False).tolist())
