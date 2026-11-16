"""Unit tests for photo caching pipeline (scripts/cache_images.py).
Tests include shortest-side resizing, no-upscale rule, aspect ratio preservation,
manifest generation, and cache_info output.
"""

import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.cache_images import process_and_save_photo, run_caching


def test_process_and_save_photo_shortest_side_and_no_upscale(tmp_path):
    # 1. Landscape image: 800 x 600 (min side = 600)
    # Expected: min side scaled to 256 -> new size = (round(800 * 256 / 600), 256) = (341, 256)
    src_landscape = tmp_path / "landscape.png"
    Image.new("RGB", (800, 600), color=(100, 150, 200)).save(src_landscape)
    dst_landscape = tmp_path / "dst_landscape.jpg"

    info = process_and_save_photo(src_landscape, dst_landscape, min_size=256, quality=95)
    assert info is not None
    assert info["orig_w"] == 800 and info["orig_h"] == 600
    assert info["new_w"] == 341 and info["new_h"] == 256
    with Image.open(dst_landscape) as img:
        assert img.size == (341, 256)
        assert img.format == "JPEG"

    # 2. Portrait image: 400 x 800 (min side = 400)
    # Expected: min side scaled to 256 -> new size = (256, round(800 * 256 / 400)) = (256, 512)
    src_portrait = tmp_path / "portrait.png"
    Image.new("RGB", (400, 800), color=(50, 120, 180)).save(src_portrait)
    dst_portrait = tmp_path / "dst_portrait.jpg"

    info_p = process_and_save_photo(src_portrait, dst_portrait, min_size=256, quality=95)
    assert info_p is not None
    assert info_p["orig_w"] == 400 and info_p["orig_h"] == 800
    assert info_p["new_w"] == 256 and info_p["new_h"] == 512
    with Image.open(dst_portrait) as img:
        assert img.size == (256, 512)

    # 3. Small image: 180 x 220 (min side = 180 <= 256)
    # Expected: NO UPSCALE, dimensions stay (180, 220)
    src_small = tmp_path / "small.png"
    Image.new("RGB", (180, 220), color=(20, 40, 60)).save(src_small)
    dst_small = tmp_path / "dst_small.jpg"

    info_s = process_and_save_photo(src_small, dst_small, min_size=256, quality=95)
    assert info_s is not None
    assert info_s["new_w"] == 180 and info_s["new_h"] == 220
    with Image.open(dst_small) as img:
        assert img.size == (180, 220)


def test_run_caching_dummy_mode(tmp_path):
    out_dir = tmp_path / "test_cache_output"
    success = run_caching(
        sketchy_root="nonexistent",
        flickr_root="nonexistent",
        out_dir=str(out_dir),
        min_size=256,
        quality=95,
        workers=2,
        debug=True,
        dummy_test=True,
    )
    assert success is True

    # Verify cache_info.json
    info_path = out_dir / "cache_info.json"
    assert info_path.exists()
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    assert info["rule"] == "shortest_side_256_no_upscale"
    assert info["target_shortest_side"] == 256
    assert info["total_photos"] == 20

    # Verify cache_manifest.csv
    manifest_path = out_dir / "cache_manifest.csv"
    assert manifest_path.exists()
    df_manifest = pd.read_csv(manifest_path)
    assert len(df_manifest) == 20
    expected_cols = {"filepath", "dataset", "orig_w", "orig_h", "new_w", "new_h", "bytes"}
    assert expected_cols.issubset(set(df_manifest.columns))
    assert all(df_manifest["bytes"] > 0)
