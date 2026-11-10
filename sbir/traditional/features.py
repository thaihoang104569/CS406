"""Dense HOG feature extraction module for binary edges / sketches.
"""

from typing import List, Tuple
import cv2
import numpy as np


def extract_dense_hog(
    binary_img: np.ndarray,
    patch_size: int = 32,
    step_size: int = 8,
    orientations: int = 9,
    cell_size: int = 8,
    min_edge_ratio: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts dense HOG descriptors across sliding windows.
    
    Args:
        binary_img: 2D uint8 edge map / sketch image (e.g. 256x256).
        patch_size: Size of sliding window (32x32).
        step_size: Stride of sliding window (8px).
        orientations: Number of gradient bins (9).
        cell_size: Pixels per cell (8x8).
        min_edge_ratio: Discard empty patches with edge ratio < min_edge_ratio.
        
    Returns:
        descriptors: ndarray of shape (N, 324) where 324 = 9 * 4 * 9 (block 2x2 cells, 4 cells per block).
        positions: ndarray of shape (N, 2) storing (x, y) coordinates of each patch.
    """
    hog = cv2.HOGDescriptor(
        _winSize=(patch_size, patch_size),
        _blockSize=(cell_size * 2, cell_size * 2),
        _blockStride=(cell_size, cell_size),
        _cellSize=(cell_size, cell_size),
        _nbins=orientations
    )
    
    h, w = binary_img.shape[:2]
    descriptors = []
    positions = []
    
    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            patch = binary_img[y:y+patch_size, x:x+patch_size]
            edge_ratio = np.count_nonzero(patch) / (patch_size * patch_size)
            
            # Skip uniform / empty patches
            if edge_ratio < min_edge_ratio:
                continue
                
            desc = hog.compute(patch)
            descriptors.append(desc.flatten())
            positions.append((x + patch_size // 2, y + patch_size // 2))
            
    if not descriptors:
        return np.empty((0, 324), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
        
    return np.array(descriptors, dtype=np.float32), np.array(positions, dtype=np.int32)
