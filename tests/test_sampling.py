"""Tests for the category-level triplet sampler used by E3/E4 training."""

import numpy as np
import pytest

from sbir.deep.sampling import TripletSampler, debug_subset


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return rng.integers(0, 5, size=300), np.repeat(np.arange(5), [40, 10, 25, 5, 60])


def test_triplets_are_valid(labels):
    sk, ph = labels
    s, p, n = TripletSampler(sk, ph, seed=1).epoch_triplets(0)
    assert sorted(s.tolist()) == list(range(len(sk)))          # every sketch exactly once
    assert (ph[p] == sk[s]).all()                               # positives: same class
    assert (ph[n] != sk[s]).all()                               # negatives: other class


def test_reproducible_and_epochs_differ(labels):
    sk, ph = labels
    a = TripletSampler(sk, ph, seed=7)
    b = TripletSampler(sk, ph, seed=7)
    for x, y in zip(a.epoch_triplets(3), b.epoch_triplets(3)):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a.epoch_triplets(0)[1], a.epoch_triplets(1)[1])


def test_positives_cover_many_photos(labels):
    sk, ph = labels
    smp = TripletSampler(sk, ph, seed=3)
    seen = set()
    for e in range(20):
        seen.update(smp.epoch_triplets(e)[1].tolist())
    assert len(seen) > 0.8 * len(ph)                            # not stuck on a few photos


def test_errors():
    with pytest.raises(ValueError):
        TripletSampler([0, 1], [0, 0])                          # class 1 has no photo
    with pytest.raises(ValueError):
        TripletSampler([0, 0], [0, 0])                          # one class only


def test_debug_subset():
    cats = [f"c{i}" for i in range(30)]
    a, b = debug_subset(cats, 20, 42), debug_subset(cats, 20, 42)
    assert a == b and len(a) == 20 and set(a) <= set(cats)
