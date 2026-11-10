"""Module for standard SBIR evaluation metrics:
- Precision@K (P@K)
- Average Precision (AP)
- mean Average Precision (mAP)
- mAP@K (e.g. mAP@200)
"""

from typing import Dict, List, Optional, Union
import numpy as np


def compute_precision_at_k(relevant_mask: np.ndarray, k: int) -> float:
    """Calculate Precision@K for a single query.
    
    Args:
        relevant_mask: 1D boolean array where True indicates the item is relevant (same class).
        k: Cutoff rank (e.g., 10, 100, 200).
        
    Returns:
        Precision at K (float between 0.0 and 1.0).
    """
    if k <= 0:
        return 0.0
    cutoff = min(k, len(relevant_mask))
    if cutoff == 0:
        return 0.0
    return float(np.sum(relevant_mask[:cutoff]) / k)


def compute_average_precision(
    relevant_mask: np.ndarray,
    total_relevant: Optional[int] = None,
    k_cutoff: Optional[int] = None,
) -> float:
    """Calculate Average Precision (AP) for a single query.
    
    Args:
        relevant_mask: 1D boolean array where True indicates the item is relevant (same class).
        total_relevant: Total number of relevant items in the gallery for this query.
                        If None, defaults to sum(relevant_mask).
        k_cutoff: Optional cutoff rank (e.g. 200 for mAP@200).
        
    Returns:
        Average Precision score (float between 0.0 and 1.0).
    """
    if total_relevant is None:
        total_relevant = int(np.sum(relevant_mask))
        
    if total_relevant == 0:
        return 0.0

    if k_cutoff is not None:
        relevant_mask = relevant_mask[:k_cutoff]

    ranks = np.arange(1, len(relevant_mask) + 1)
    cum_correct = np.cumsum(relevant_mask)
    precisions = cum_correct / ranks

    # AP = sum(Precision@k * rel(k)) / total_relevant
    ap = np.sum(precisions * relevant_mask) / total_relevant
    return float(ap)


def evaluate_retrieval(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k_list: List[int] = [10, 100, 200],
    map_cutoff: int = 200,
) -> Dict[str, float]:
    """Evaluates full SBIR retrieval performance across all queries.
    
    Args:
        similarity_matrix: 2D array of shape (num_queries, num_gallery).
                           Higher values mean higher similarity.
        query_labels: 1D array of class labels for queries.
        gallery_labels: 1D array of class labels for gallery images.
        top_k_list: List of K values for Precision@K calculation.
        map_cutoff: Cutoff rank for mAP@K (default: 200).
        
    Returns:
        Dictionary containing aggregate metrics:
        'mAP', 'mAP@200', 'P@10', 'P@100', 'P@200', etc.
    """
    num_queries = len(query_labels)
    assert similarity_matrix.shape == (num_queries, len(gallery_labels)), (
        f"Shape mismatch: sim_matrix {similarity_matrix.shape} vs "
        f"queries ({num_queries}) and gallery ({len(gallery_labels)})"
    )

    # Sort each query's gallery items by similarity descending
    ranked_indices = np.argsort(-similarity_matrix, axis=1)

    aps = []
    aps_at_cutoff = []
    precisions_at_k = {k: [] for k in top_k_list}

    # Count total relevant per class in gallery
    unique_labels, counts = np.unique(gallery_labels, return_counts=True)
    gallery_class_counts = dict(zip(unique_labels, counts))

    for q_idx in range(num_queries):
        q_label = query_labels[q_idx]
        total_relevant = gallery_class_counts.get(q_label, 0)
        
        ranked_labels = gallery_labels[ranked_indices[q_idx]]
        rel_mask = (ranked_labels == q_label)

        # Full AP
        ap = compute_average_precision(rel_mask, total_relevant=total_relevant)
        aps.append(ap)

        # AP@cutoff (e.g. AP@200)
        ap_cutoff = compute_average_precision(
            rel_mask, total_relevant=total_relevant, k_cutoff=map_cutoff
        )
        aps_at_cutoff.append(ap_cutoff)

        # P@K
        for k in top_k_list:
            p_k = compute_precision_at_k(rel_mask, k=k)
            precisions_at_k[k].append(p_k)

    results = {
        "mAP": float(np.mean(aps)),
        f"mAP@{map_cutoff}": float(np.mean(aps_at_cutoff)),
    }
    for k in top_k_list:
        results[f"P@{k}"] = float(np.mean(precisions_at_k[k]))

    return results
