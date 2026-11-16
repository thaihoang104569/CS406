"""Unit tests for Dense HOG feature extraction (sbir/traditional/features.py).
"""

import numpy as np
import pytest
from sbir.traditional.features import extract_dense_hog


def test_extract_dense_hog_empty_image():
    # An all-black image (no edges) should filter out all patches
    blank = np.zeros((256, 256), dtype=np.uint8)
    desc, pos = extract_dense_hog(blank, patch_size=32, step_size=8, min_edge_ratio=0.02)
    assert desc.shape == (0, 324)
    assert pos.shape == (0, 2)


def test_extract_dense_hog_with_edges():
    img = np.zeros((256, 256), dtype=np.uint8)
    # Draw a 100x100 square with white border
    img[50:150, 50] = 255
    img[50:150, 150] = 255
    img[50, 50:150] = 255
    img[150, 50:150] = 255

    desc, pos = extract_dense_hog(
        img,
        patch_size=32,
        step_size=8,
        orientations=9,
        cell_size=8,
        min_edge_ratio=0.01,
    )

    assert len(desc) > 0
    assert desc.shape[1] == 324
    assert pos.shape == (len(desc), 2)
    assert np.all(desc >= 0.0)
    # Check that positions are within image bounds
    assert np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] <= 256)
    assert np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] <= 256)


def test_extract_dense_hog_small_image():
    # Image smaller than patch_size should safely return empty
    small = np.ones((16, 16), dtype=np.uint8) * 255
    desc, pos = extract_dense_hog(small, patch_size=32, step_size=8)
    assert desc.shape == (0, 324)
    assert pos.shape == (0, 2)
