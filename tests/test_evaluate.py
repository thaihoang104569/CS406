"""Unit tests for SBIR evaluation metrics (sbir/evaluate.py).
Tests include exact hand-calculated assertions, cutoff modes, and edge cases.
"""

import numpy as np
import pytest
from sbir.evaluate import (
    compute_precision_at_k,
    compute_average_precision,
    evaluate_retrieval,
)


def test_compute_precision_at_k():
    rel_mask = np.array([True, False, True, True, False])
    
    assert compute_precision_at_k(rel_mask, k=1) == pytest.approx(1.0)
    assert compute_precision_at_k(rel_mask, k=2) == pytest.approx(0.5)
    assert compute_precision_at_k(rel_mask, k=3) == pytest.approx(2.0 / 3.0)
    assert compute_precision_at_k(rel_mask, k=4) == pytest.approx(0.75)
    assert compute_precision_at_k(rel_mask, k=5) == pytest.approx(0.6)
    assert compute_precision_at_k(rel_mask, k=10) == pytest.approx(3.0 / 10.0)
    assert compute_precision_at_k(rel_mask, k=0) == 0.0
    assert compute_precision_at_k(np.array([], dtype=bool), k=5) == 0.0


def test_compute_average_precision_hand_calculated():
    # Ranked relevance: [1, 0, 1, 1, 0]
    # R = 3 total relevant items in gallery
    # Precisions at ranks:
    # rank 1: 1/1 = 1.0 (rel=1)
    # rank 2: 1/2 = 0.5 (rel=0)
    # rank 3: 2/3 (rel=1)
    # rank 4: 3/4 = 0.75 (rel=1)
    # rank 5: 3/5 = 0.6 (rel=0)
    rel_mask = np.array([1, 0, 1, 1, 0], dtype=bool)
    total_r = 3

    # 1. Full AP (across whole list)
    # AP = (1.0 + 2/3 + 3/4) / 3 = (29/12) / 3 = 29/36 ≈ 0.80555556
    expected_full_ap = (1.0 + (2.0 / 3.0) + 0.75) / 3.0
    assert compute_average_precision(rel_mask, total_relevant=total_r) == pytest.approx(expected_full_ap)

    # 2. Cutoff at k=3 -> [1, 0, 1]
    # sum of precision * rel in top 3 = 1.0 + 2/3 = 5/3 ≈ 1.66666667
    
    # Mode: min_r_k (standard benchmark: denom = min(3, 3) = 3)
    # AP@3 = (5/3) / 3 = 5/9 ≈ 0.55555556
    expected_min_r_k = (5.0 / 3.0) / 3.0
    assert compute_average_precision(rel_mask, total_relevant=total_r, k_cutoff=3, mode="min_r_k") == pytest.approx(expected_min_r_k)

    # Mode: rel_in_k (Phụ lục B variant: denom = hits = 2)
    # AP@3 = (5/3) / 2 = 5/6 ≈ 0.83333333
    expected_rel_in_k = (5.0 / 3.0) / 2.0
    assert compute_average_precision(rel_mask, total_relevant=total_r, k_cutoff=3, mode="rel_in_k") == pytest.approx(expected_rel_in_k)

    # Mode: total_r with large gallery R=10
    # Shows why total_r is bad when R > K: (5/3) / 10 = 1/6 ≈ 0.16666667
    assert compute_average_precision(rel_mask, total_relevant=10, k_cutoff=3, mode="total_r") == pytest.approx((5.0 / 3.0) / 10.0)
    # While min_r_k caps denom at k_cutoff=3:
    assert compute_average_precision(rel_mask, total_relevant=10, k_cutoff=3, mode="min_r_k") == pytest.approx((5.0 / 3.0) / 3.0)


def test_compute_average_precision_edge_cases():
    # All irrelevant
    zeros = np.array([0, 0, 0, 0], dtype=bool)
    assert compute_average_precision(zeros, total_relevant=5) == 0.0
    assert compute_average_precision(zeros, total_relevant=5, k_cutoff=2) == 0.0

    # Total relevant is 0
    assert compute_average_precision(zeros, total_relevant=0) == 0.0

    # Perfect retrieval: [1, 1, 1], R=3
    perfect = np.array([1, 1, 1], dtype=bool)
    assert compute_average_precision(perfect, total_relevant=3) == pytest.approx(1.0)
    assert compute_average_precision(perfect, total_relevant=3, k_cutoff=3, mode="min_r_k") == pytest.approx(1.0)
    assert compute_average_precision(perfect, total_relevant=3, k_cutoff=3, mode="rel_in_k") == pytest.approx(1.0)

    # Rare class where R < K: R=2, K=5, retrieved [1, 1, 0, 0, 0]
    # min_r_k should yield 1.0 because model retrieved all 2 existing relevant items perfectly!
    rare_perfect = np.array([1, 1, 0, 0, 0], dtype=bool)
    assert compute_average_precision(rare_perfect, total_relevant=2, k_cutoff=5, mode="min_r_k") == pytest.approx(1.0)


def test_evaluate_retrieval_synthetic():
    # 2 queries, 4 gallery items
    # Query 0 has label 0
    # Query 1 has label 1
    # Gallery has labels [0, 1, 0, 1]
    query_labels = np.array([0, 1])
    gallery_labels = np.array([0, 1, 0, 1])

    # Construct similarity matrix such that:
    # Query 0 ranks gallery as: [0, 2, 1, 3] -> labels: [0, 0, 1, 1] -> perfect for class 0!
    # Query 1 ranks gallery as: [0, 1, 3, 2] -> labels: [0, 1, 1, 0] -> rank 1 is wrong (0), rank 2 & 3 are 1.
    sim_matrix = np.array([
        [0.9, 0.4, 0.8, 0.3],  # Q0: indices 0, 2, 1, 3 -> labels 0, 0, 1, 1
        [0.9, 0.8, 0.3, 0.7],  # Q1: indices 0, 1, 3, 2 -> labels 0, 1, 1, 0
    ])

    results = evaluate_retrieval(
        similarity_matrix=sim_matrix,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        top_k_list=[1, 2],
        map_cutoff=2,
        map_mode="min_r_k",
    )

    # For Q0:
    # Ranked labels: [0, 0, 1, 1], rel_mask: [1, 1, 0, 0], R=2
    # AP = (1/1 + 2/2) / 2 = 1.0
    # AP@2 (min_r_k) = (1/1 + 2/2) / 2 = 1.0
    # P@1 = 1.0, P@2 = 1.0

    # For Q1:
    # Ranked labels: [0, 1, 1, 0], rel_mask: [0, 1, 1, 0], R=2
    # Precisions: rank 1: 0, rank 2: 1/2 = 0.5, rank 3: 2/3
    # AP = (0.5 + 2/3) / 2 = (7/6) / 2 = 7/12 ≈ 0.58333333
    # AP@2 (min_r_k): top 2 ranks -> [0, 1]. hits=1, denom=min(2, 2)=2.
    # AP@2 = (0.5) / 2 = 0.25
    # AP@2 (rel_in_k): hits=1, denom=1. AP@2 = 0.5 / 1 = 0.5
    # P@1 = 0.0, P@2 = 1/2 = 0.5

    # Mean values:
    # mAP = (1.0 + 7/12) / 2 = 19/24 ≈ 0.79166667
    # mAP@2 = (1.0 + 0.25) / 2 = 1.25 / 2 = 0.625
    # mAP@2_topk_hit = (1.0 + 0.5) / 2 = 0.75
    # P@1 = (1.0 + 0.0) / 2 = 0.5
    # P@2 = (1.0 + 0.5) / 2 = 0.75

    assert results["mAP"] == pytest.approx(19.0 / 24.0)
    assert results["mAP@2"] == pytest.approx(0.625)
    assert results["mAP@2_topk_hit"] == pytest.approx(0.75)
    assert results["P@1"] == pytest.approx(0.5)
    assert results["P@2"] == pytest.approx(0.75)
