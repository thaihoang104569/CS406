"""Image Preprocessing Module for SBIR Baseline (E1).
Includes:
- Photo preprocessing: Bilateral filter, Auto-Canny edge detection, connected component filtering, bounding box crop & pad.
- Sketch preprocessing: Otsu thresholding, Zhang-Suen skeletonization, dilation.
- Real-world sketch photo enhancement: Morphological closing for background illumination removal and adaptive thresholding.
"""

import cv2
import numpy as np


def resize_pad(image: np.ndarray, target_size: int = 256, pad_color: int = 255) -> np.ndarray:
    """Resize maintaining aspect ratio with long side = target_size, then pad to square."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    interpolation = cv2.INTER_NEAREST if len(np.unique(image)) <= 2 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_color)
    else:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[pad_color]*3)
    return padded


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Computes Canny edges with automatic thresholds based on image median."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def filter_small_components(binary_edge_map: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Filters out disconnected noisy edge segments with pixel count < min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_edge_map, connectivity=8)
    filtered = np.zeros_like(binary_edge_map)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered


def preprocess_photo(
    image: np.ndarray,
    target_size: int = 256,
    bilateral_d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    auto_canny_sigma: float = 0.33,
    min_comp_area: int = 20,
) -> np.ndarray:
    """Full pipeline for photo preprocessing:
    Resize/Pad -> Gray -> Bilateral -> Auto-Canny -> ConnectedComponents Filter -> Dilate.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    gray_square = resize_pad(gray, target_size=target_size, pad_color=255)
    filtered = cv2.bilateralFilter(gray_square, d=bilateral_d,
                                  sigmaColor=sigma_color, sigmaSpace=sigma_space)
    edges = auto_canny(filtered, sigma=auto_canny_sigma)
    clean_edges = filter_small_components(edges, min_area=min_comp_area)
    
    # 3x3 dilation for spatial tolerance
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(clean_edges, kernel, iterations=1)
    return dilated_edges


def preprocess_sketch(
    sketch: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    """Full pipeline for digital sketch:
    Binarization -> Bounding Box Crop & Pad -> Skeletonize -> 3x3 Dilation.
    """
    if len(sketch.shape) == 3:
        gray = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    else:
        gray = sketch.copy()

    # Otsu thresholding (strokes = 255, background = 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find bounding box of strokes
    pts = cv2.findNonZero(binary)
    if pts is not None:
        x, y, w, h = cv2.boundingRect(pts)
        pad = int(0.10 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(binary.shape[1], x + w + pad), min(binary.shape[0], y + h + pad)
        cropped = binary[y0:y1, x0:x1]
    else:
        cropped = binary

    resized = resize_pad(cropped, target_size=target_size, pad_color=0)
    
    # Thinning / Skeletonization
    try:
        from skimage.morphology import skeletonize
        skeleton = (skeletonize(resized > 0) * 255).astype(np.uint8)
    except ImportError:
        skeleton = resized

    # 3x3 dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    return dilated
