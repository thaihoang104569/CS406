"""Tests for sbir/index.py: FAISS and NumPy fallback must agree with brute force."""

import numpy as np
import pytest

from sbir.index import VectorIndex, l2_normalize


def brute_force(q, g, k):
    sims = l2_normalize(q) @ l2_normalize(g).T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(sims, order, axis=1), order


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((37, 16)).astype(np.float32), rng.standard_normal((500, 16)).astype(np.float32)


@pytest.mark.parametrize("force_numpy", [False, True])
def test_search_matches_brute_force_in_batches(data, force_numpy):
    q, g = data
    idx = VectorIndex(dim=16, batch_size=8)  # several batches
    if force_numpy:
        idx.index = None
    idx.add(g[:300])
    idx.add(g[300:])  # add twice: ids continue
    scores, ids = idx.search(q, top_k=20)
    ref_s, ref_i = brute_force(q, g, 20)
    assert ids.shape == (37, 20)
    np.testing.assert_allclose(scores, ref_s, atol=1e-5)
    np.testing.assert_array_equal(ids, ref_i)


@pytest.mark.parametrize("force_numpy", [False, True])
def test_top_k_larger_than_gallery_is_clamped(data, force_numpy):
    q, g = data
    idx = VectorIndex(dim=16)
    if force_numpy:
        idx.index = None
    idx.add(g[:5])
    scores, ids = idx.search(q, top_k=200)
    assert ids.shape == (37, 5)
    assert (ids >= 0).all()
    assert np.all(np.diff(scores, axis=1) <= 1e-6)  # sorted descending


def test_save_load_roundtrip(tmp_path, data):
    q, g = data
    idx = VectorIndex(dim=16)
    idx.add(g)
    path = tmp_path / "gallery.npy"
    idx.save(str(path))
    loaded = VectorIndex.load(str(path))
    assert loaded.ntotal == 500
    _, a = idx.search(q, top_k=10)
    _, b = loaded.search(q, top_k=10)
    assert (a == b).mean() > 0.95  # float16 storage may swap near-ties


def test_empty_and_wrong_dim():
    idx = VectorIndex(dim=8)
    with pytest.raises(RuntimeError):
        idx.search(np.zeros((1, 8)), top_k=5)
    with pytest.raises(AssertionError):
        idx.add(np.zeros((2, 4)))
    idx.add(np.ones((3, 8)))
    idx.reset()
    assert idx.ntotal == 0
