"""Unit tests for VisualCodebook (sbir/traditional/bovw.py).
Tests include KMeans fitting, word assignment, TF-IDF calculation,
Spatial Pyramid Matching (SPM 5000-D), normalization, and save/load roundtrip.
"""

import numpy as np
import pytest
from sbir.traditional.bovw import VisualCodebook


def test_visual_codebook_fit_and_assign():
    np.random.seed(42)
    # 200 random descriptors of dimension 324 (HOG descriptor dim)
    descriptors = np.random.randn(200, 324).astype(np.float32)
    
    cb = VisualCodebook(n_clusters=10, batch_size=50, random_state=42)
    cb.fit(descriptors)
    assert cb.is_fitted
    assert cb.kmeans.cluster_centers_.shape == (10, 324)

    words = cb.assign_words(descriptors[:20])
    assert len(words) == 20
    assert np.all(words >= 0) and np.all(words < 10)

    # Empty descriptors
    empty_words = cb.assign_words(np.empty((0, 324)))
    assert len(empty_words) == 0


def test_visual_codebook_idf_and_spatial_pyramid():
    np.random.seed(42)
    descriptors = np.random.randn(300, 324).astype(np.float32)
    positions = np.random.randint(0, 256, size=(300, 2)).astype(np.int32)

    cb = VisualCodebook(n_clusters=20, batch_size=100, random_state=42)
    cb.fit(descriptors)

    # Mock word histograms for 5 images
    dummy_histograms = np.random.randint(0, 5, size=(5, 20)).astype(np.float32)
    cb.fit_idf(dummy_histograms)
    assert cb.idf_weights is not None
    assert len(cb.idf_weights) == 20
    assert np.all(cb.idf_weights > 0)

    # Compute Spatial Pyramid: 1x1 (1) + 2x2 (4) = 5 cells -> 5 * 20 = 100-D
    spm_vec = cb.compute_spatial_pyramid(descriptors[:50], positions[:50], img_size=256, levels=[1, 2], apply_idf=True)
    assert spm_vec.shape == (100,)
    assert spm_vec.dtype == np.float32
    # Check L2 normalization
    norm = np.linalg.norm(spm_vec)
    assert norm == pytest.approx(1.0, abs=1e-5)

    # Empty input returns zero vector
    zero_vec = cb.compute_spatial_pyramid(np.empty((0, 324)), np.empty((0, 2)), levels=[1, 2])
    assert zero_vec.shape == (100,)
    assert np.all(zero_vec == 0)


def test_visual_codebook_save_load(tmp_path):
    np.random.seed(42)
    descriptors = np.random.randn(100, 324).astype(np.float32)
    positions = np.random.randint(0, 256, size=(100, 2)).astype(np.int32)

    cb = VisualCodebook(n_clusters=15, batch_size=50, random_state=42)
    cb.fit(descriptors)
    cb.fit_idf(np.ones((4, 15), dtype=np.float32))

    orig_vec = cb.compute_spatial_pyramid(descriptors, positions, levels=[1, 2])

    save_path = tmp_path / "codebook.pkl"
    cb.save(save_path)
    assert save_path.exists()

    loaded_cb = VisualCodebook.load(save_path)
    assert loaded_cb.is_fitted
    assert loaded_cb.n_clusters == 15
    assert loaded_cb.idf_weights is not None

    loaded_vec = loaded_cb.compute_spatial_pyramid(descriptors, positions, levels=[1, 2])
    np.testing.assert_allclose(orig_vec, loaded_vec, rtol=1e-5, atol=1e-5)
