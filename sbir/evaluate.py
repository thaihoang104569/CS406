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
    mode: str = "min_r_k",
) -> float:
    """Calculate Average Precision (AP) for a single query.
    
    Args:
        relevant_mask: 1D boolean array where True indicates the item is relevant (same class).
        total_relevant: Total number of relevant items in the gallery for this query.
                        If None, defaults to sum(relevant_mask).
        k_cutoff: Optional cutoff rank (e.g. 200 for mAP@200).
        mode: Normalization mode when k_cutoff is provided:
            - 'min_r_k' (default, standard benchmark): denom = min(total_relevant, k_cutoff).
              Ensures score is 1.0 if all possible relevant items within top-K are retrieved.
            - 'rel_in_k' (variant from Phụ lục B / top-K hit): denom = sum(relevant_mask[:k_cutoff]).
              Measures ranking quality among retrieved relevant items.
            - 'total_r': denom = total_relevant (gallery-wide, can artificially suppress AP@K).
            
    Returns:
        Average Precision score (float between 0.0 and 1.0).
    """
    if k_cutoff is not None:
        relevant_mask = relevant_mask[:k_cutoff]

    hits = int(np.sum(relevant_mask))
    if hits == 0:
        return 0.0

    if total_relevant is None:
        total_relevant = hits

    if k_cutoff is not None:
        if mode == "min_r_k":
            denom = min(total_relevant, k_cutoff)
        elif mode == "rel_in_k":
            denom = hits
        elif mode == "total_r":
            denom = total_relevant
        else:
            raise ValueError(f"Unknown AP normalization mode: {mode}")
    else:
        denom = total_relevant

    if denom <= 0:
        return 0.0

    ranks = np.arange(1, len(relevant_mask) + 1)
    cum_correct = np.cumsum(relevant_mask)
    precisions = cum_correct / ranks

    # AP = sum(Precision@k * rel(k)) / denom
    ap = np.sum(precisions * relevant_mask) / denom
    return float(ap)


def evaluate_retrieval(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k_list: List[int] = [10, 100, 200],
    map_cutoff: int = 200,
    map_mode: str = "min_r_k",
) -> Dict[str, float]:
    """Evaluates full SBIR retrieval performance across all queries.
    
    Args:
        similarity_matrix: 2D array of shape (num_queries, num_gallery).
                           Higher values mean higher similarity.
        query_labels: 1D array of class labels for queries.
        gallery_labels: 1D array of class labels for gallery images.
        top_k_list: List of K values for Precision@K calculation.
        map_cutoff: Cutoff rank for mAP@K (default: 200).
        map_mode: Normalization mode for mAP@cutoff ('min_r_k' or 'rel_in_k').
        
    Returns:
        Dictionary containing aggregate metrics:
        'mAP', 'mAP@200', 'mAP@200_topk_hit', 'P@10', 'P@100', 'P@200', etc.
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
    aps_at_cutoff_hit = []
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

        # AP@cutoff (default standard min_r_k)
        ap_cutoff = compute_average_precision(
            rel_mask, total_relevant=total_relevant, k_cutoff=map_cutoff, mode=map_mode
        )
        aps_at_cutoff.append(ap_cutoff)

        # AP@cutoff topk_hit variant for reference
        ap_cutoff_hit = compute_average_precision(
            rel_mask, total_relevant=total_relevant, k_cutoff=map_cutoff, mode="rel_in_k"
        )
        aps_at_cutoff_hit.append(ap_cutoff_hit)

        # P@K
        for k in top_k_list:
            p_k = compute_precision_at_k(rel_mask, k=k)
            precisions_at_k[k].append(p_k)

    results = {
        "mAP": float(np.mean(aps)),
        f"mAP@{map_cutoff}": float(np.mean(aps_at_cutoff)),
        f"mAP@{map_cutoff}_topk_hit": float(np.mean(aps_at_cutoff_hit)),
    }
    for k in top_k_list:
        results[f"P@{k}"] = float(np.mean(precisions_at_k[k]))

    return results
