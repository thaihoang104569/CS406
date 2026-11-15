"""Unit tests for image preprocessing pipeline (sbir/traditional/preprocess.py).
Tests include resize/padding, auto Canny, connected component filtering,
bounding box standardizing, and full photo/sketch pipelines.
"""

import numpy as np
import pytest
import cv2
from sbir.traditional.preprocess import (
    resize_pad,
    auto_canny,
    filter_small_components,
    crop_bbox_pad,
    preprocess_photo,
    preprocess_sketch,
)


def test_resize_pad():
    # Test non-square image (height 100, width 200)
    img = np.ones((100, 200), dtype=np.uint8) * 128
    padded = resize_pad(img, target_size=256, pad_color=0)
    assert padded.shape == (256, 256)
    assert padded.dtype == np.uint8

    # Check color image
    color_img = np.ones((150, 300, 3), dtype=np.uint8) * 100
    color_padded = resize_pad(color_img, target_size=256, pad_color=255)
    assert color_padded.shape == (256, 256, 3)
    # Check padding border is 255
    assert color_padded[0, 0, 0] == 255


def test_auto_canny():
    # Create image with a clear bright circle on dark background
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, 255, -1)
    
    edges = auto_canny(img, sigma=0.33)
    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8
    # Should contain detected edge pixels
    assert np.count_nonzero(edges) > 0
    # Values must be binary 0 or 255
    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})


def test_filter_small_components():
    # Create an image with one large connected component (area > 50) and several single pixel noises
    binary = np.zeros((100, 100), dtype=np.uint8)
    # Large component: 10x10 block = 100 px
    binary[20:30, 20:30] = 255
    # Noise: isolated 2x2 blocks = 4 px
    binary[80:82, 80:82] = 255
    binary[10, 10] = 255

    filtered = filter_small_components(binary, min_area=20)
    assert filtered.shape == (100, 100)
    # Large component must be preserved
    assert np.all(filtered[20:30, 20:30] == 255)
    # Small noise must be removed
    assert np.all(filtered[80:82, 80:82] == 0)
    assert filtered[10, 10] == 0


def test_crop_bbox_pad():
    # Create a small stroke at the corner of a 200x200 canvas
    canvas = np.zeros((200, 200), dtype=np.uint8)
    canvas[10:30, 10:30] = 255  # 20x20 square at top-left
    
    standardized = crop_bbox_pad(canvas, target_size=256, pad_ratio=0.10, pad_color=0)
    assert standardized.shape == (256, 256)
    # After cropping and centering, non-zero pixels should now be approximately centered
    pts = cv2.findNonZero(standardized)
    assert pts is not None
    x, y, w, h = cv2.boundingRect(pts)
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    # Must be reasonably close to center (128, 128)
    assert abs(center_x - 128) < 15
    assert abs(center_y - 128) < 15


def test_preprocess_photo():
    # Create a synthetic 300x400 RGB photo containing a shape with some gradient and noise
    photo = np.full((300, 400, 3), 200, dtype=np.uint8)
    # Dark object in center
    cv2.rectangle(photo, (100, 80), (300, 220), (30, 40, 50), -1)

    result = preprocess_photo(photo, target_size=256)
    assert result.shape == (256, 256)
    assert result.dtype == np.uint8
    # Must be binary edge map
    unique_vals = set(np.unique(result))
    assert unique_vals.issubset({0, 255})
    # Edges should be extracted
    assert np.count_nonzero(result) > 0


def test_preprocess_sketch():
    # Create a digital sketch: white background (255) with black strokes (0)
    sketch = np.full((300, 300), 255, dtype=np.uint8)
    # Draw a black star / cross
    cv2.line(sketch, (50, 50), (250, 250), 0, thickness=5)
    cv2.line(sketch, (50, 250), (250, 50), 0, thickness=5)

    result = preprocess_sketch(sketch, target_size=256)
    assert result.shape == (256, 256)
    assert result.dtype == np.uint8
    unique_vals = set(np.unique(result))
    assert unique_vals.issubset({0, 255})
    # Strokes should be preserved as 255 on 0 background
    assert np.count_nonzero(result) > 0
