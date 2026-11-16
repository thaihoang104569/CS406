"""One-shot image caching script for real photos (Sketchy Extended & Flickr15K).
Spec (Section 2.4 of 4-Week Plan, updated 15/11):
- Caches ONLY real photos (Sketchy: 73,002 photos; Flickr15K: 14,660 photos).
  Sketches are NOT cached (always read directly from raw data).
- Resize rule: SHORTEST side = 256 px, maintaining aspect ratio.
  If shortest side <= 256 px, keep original size (DO NOT UPSCALE).
- Image.draft('RGB', size) before decoding large JPEGs for speed.
- Filter: Image.Resampling.LANCZOS, format: JPEG quality=95, optimize=True.
- Preserves relative paths:
    <out_dir>/sketchy/<relpath>
    <out_dir>/flickr15k/<relpath>
- Outputs metadata:
    <out_dir>/cache_info.json
    <out_dir>/cache_manifest.csv (filepath,orig_w,orig_h,new_w,new_h,bytes)

Usage on Kaggle:
    python scripts/cache_images.py \
        --sketchy_root /kaggle/input/datasets/hongonthi/sketchyextended/sketchy \
        --flickr_root  /kaggle/input/datasets/hongonthi/flickr15k/Flickr15K \
        --out_dir      /kaggle/working/sbir-cache-256-photos \
        --workers      4

Local quick debug verification:
    python scripts/cache_images.py --debug
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import PIL
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def scan_sketchy_photos(root_dir: Path) -> List[Tuple[Path, Path]]:
    """Scan only photos in train/photo/ and val/photo/ under sketchy_root."""
    if not root_dir.exists():
        return []
    photo_pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        dp = Path(dirpath)
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in IMG_EXTENSIONS:
                full_path = dp / fname
                rel_path = full_path.relative_to(root_dir)
                parts = rel_path.as_posix().split("/")
                # Sketchy layout: <split>/photo/<category>/<filename>
                if len(parts) >= 2 and parts[1] == "photo":
                    photo_pairs.append((full_path, rel_path))
    return sorted(photo_pairs, key=lambda x: str(x[1]))


def scan_flickr15k_photos(root_dir: Path) -> List[Tuple[Path, Path]]:
    """Scan only photos under images/ (sketches/ are excluded)."""
    if not root_dir.exists():
        return []
    photo_pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        dp = Path(dirpath)
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in IMG_EXTENSIONS:
                full_path = dp / fname
                rel_path = full_path.relative_to(root_dir)
                parts = rel_path.as_posix().split("/")
                # Flickr layout: images/<folder>/<filename>
                if len(parts) >= 1 and parts[0] == "images":
                    photo_pairs.append((full_path, rel_path))
    return sorted(photo_pairs, key=lambda x: str(x[1]))


def process_and_save_photo(
    src_path: Path,
    dst_path: Path,
    min_size: int = 256,
    quality: int = 95,
) -> Optional[Dict]:
    """Loads photo, resizes SHORTEST side to min_size (no upscale if shortest <= min_size),
    saves as JPEG quality 95, and returns manifest record dict.
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            orig_w, orig_h = img.size
            min_dim = min(orig_w, orig_h)

            if min_dim > min_size:
                scale = min_size / float(min_dim)
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))

                # Call draft() to hint JPEG decoder if applicable
                try:
                    img.draft("RGB", (new_w, new_h))
                except Exception:
                    pass
            else:
                new_w, new_h = orig_w, orig_h

            # Convert to RGB (handles grayscale, palette, RGBA) without EXIF rotation
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Perform resize with LANCZOS if dimensions differ
            if img.size != (new_w, new_h):
                img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

            # Save as JPEG with quality 95
            img.save(dst_path, format="JPEG", quality=quality, optimize=True)

        file_bytes = dst_path.stat().st_size
        return {
            "orig_w": orig_w,
            "orig_h": orig_h,
            "new_w": new_w,
            "new_h": new_h,
            "bytes": file_bytes,
        }
    except Exception as e:
        print(f"[WARNING] Failed to process {src_path}: {e}", file=sys.stderr)
        return None


def run_caching(
    sketchy_root: str,
    flickr_root: str,
    out_dir: str,
    min_size: int = 256,
    quality: int = 95,
    workers: int = 8,
    debug: bool = False,
    dummy_test: bool = False,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # List of (full_src_path, dst_path, dataset_name, rel_filepath_for_manifest)
    tasks: List[Tuple[Path, Path, str, str]] = []

    if dummy_test:
        print("[INFO] Running in dummy_test mode. Generating synthetic photos...")
        dummy_src = out_path / "_dummy_src"
        dummy_src.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            # Non-square photos: 800x600 (landscape) and 500x700 (portrait)
            w, h = (800, 600) if i % 2 == 0 else (500, 700)
            p = dummy_src / f"photo_{i:03d}.jpg"
            Image.new("RGB", (w, h), color=(i * 12, 80 + i * 5, 120)).save(p, format="JPEG", quality=90)
            rel_manifest = f"dummy/photo_{i:03d}.jpg"
            dst = out_path / "dummy" / f"photo_{i:03d}.jpg"
            tasks.append((p, dst, "dummy", rel_manifest))
    else:
        # 1. Sketchy photos only
        sketchy_p = Path(sketchy_root)
        if sketchy_p.exists():
            print(f"[INFO] Scanning Sketchy photos at: {sketchy_p}")
            sk_pairs = scan_sketchy_photos(sketchy_p)
            print(f"       Found {len(sk_pairs):,} Sketchy photos (sketches skipped).")
            for full_p, rel_p in sk_pairs:
                rel_manifest = f"sketchy/{rel_p.as_posix()}"
                dst = out_path / "sketchy" / rel_p.with_suffix(".jpg")
                tasks.append((full_p, dst, "sketchy", rel_manifest))
        else:
            print(f"[WARNING] Sketchy root does not exist: {sketchy_root}")

        # 2. Flickr15k photos only
        flickr_p = Path(flickr_root)
        if flickr_p.exists():
            print(f"[INFO] Scanning Flickr15k photos at: {flickr_p}")
            fl_pairs = scan_flickr15k_photos(flickr_p)
            print(f"       Found {len(fl_pairs):,} Flickr15k photos (sketches skipped).")
            for full_p, rel_p in fl_pairs:
                rel_manifest = f"flickr15k/{rel_p.as_posix()}"
                dst = out_path / "flickr15k" / rel_p.with_suffix(".jpg")
                tasks.append((full_p, dst, "flickr15k", rel_manifest))
        else:
            print(f"[WARNING] Flickr15k root does not exist: {flickr_root}")

    if not tasks:
        print("[ERROR] No photos found to process. Please check dataset paths or use --dummy_test.")
        return False

    if debug:
        print(f"[DEBUG MODE] Limiting to first 200 photos for fast verification.")
        tasks = tasks[:200]

    print(f"\n--- Starting Photo Caching (Shortest side = {min_size}px) ---")
    print(f"Total photos:   {len(tasks):,}")
    print(f"Output folder:  {out_path.resolve()}")
    print(f"Min dimension:  {min_size} px (shortest side, no upscale)")
    print(f"JPEG Quality:   {quality}")
    print(f"Worker threads: {workers}")

    start_time = time.time()
    manifest_rows = []
    success_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(process_and_save_photo, src, dst, min_size, quality): (src, dst, ds, rel_m)
            for src, dst, ds, rel_m in tasks
        }

        for fut in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Caching photos"):
            src, dst, ds, rel_m = future_to_task[fut]
            info = fut.result()
            if info is not None:
                success_count += 1
                manifest_rows.append({
                    "filepath": rel_m,
                    "dataset": ds,
                    "orig_w": info["orig_w"],
                    "orig_h": info["orig_h"],
                    "new_w": info["new_w"],
                    "new_h": info["new_h"],
                    "bytes": info["bytes"],
                })

    elapsed = time.time() - start_time
    rate = success_count / max(elapsed, 0.001)

    # Sort manifest for deterministic output
    manifest_rows.sort(key=lambda r: r["filepath"])

    # 1. Write cache_manifest.csv
    manifest_path = out_path / "cache_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filepath", "dataset", "orig_w", "orig_h", "new_w", "new_h", "bytes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 2. Write cache_info.json
    total_bytes = sum(r["bytes"] for r in manifest_rows)
    info_dict = {
        "rule": "shortest_side_256_no_upscale",
        "target_shortest_side": min_size,
        "jpeg_quality": quality,
        "resampling_filter": "LANCZOS",
        "draft_optimization": True,
        "pillow_version": PIL.__version__,
        "git_commit": get_git_commit(),
        "created_at": datetime.datetime.now().isoformat(),
        "total_photos": len(manifest_rows),
        "sketchy_photos": len([r for r in manifest_rows if r["dataset"] == "sketchy"]),
        "flickr15k_photos": len([r for r in manifest_rows if r["dataset"] == "flickr15k"]),
        "total_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "elapsed_seconds": round(elapsed, 2),
    }

    info_path = out_path / "cache_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)

    print(f"\n--- Photo Caching Summary ---")
    print(f"Successfully cached: {success_count:,} / {len(tasks):,} photos")
    print(f"Total size:          {info_dict['total_size_mb']} MB")
    print(f"Time elapsed:        {elapsed:.2f} s ({rate:.1f} photos/sec)")
    print(f"Manifest written to: {manifest_path.resolve()}")
    print(f"Info written to:     {info_path.resolve()}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache real photos resized to shortest side = 256px (no upscale).")
    parser.add_argument("--sketchy_root", type=str, default="data/raw/sketchy",
                        help="Path to raw Sketchy dataset directory")
    parser.add_argument("--flickr_root", type=str, default="data/raw/flickr15k",
                        help="Path to raw Flickr15k dataset directory")
    parser.add_argument("--out_dir", type=str, default="data/cache_256_photos",
                        help="Destination directory for cached photos (e.g. sbir-cache-256-photos)")
    parser.add_argument("--min_size", type=int, default=256,
                        help="Shortest side dimension to resize photos to (default: 256)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG compression quality (1-100, default: 95)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent worker threads")
    parser.add_argument("--debug", action="store_true",
                        help="Only process the first 200 photos for fast verification")
    parser.add_argument("--dummy_test", action="store_true",
                        help="Generate and cache synthetic photos for end-to-end local test")

    args = parser.parse_args()

    # If roots don't exist and debug is passed without explicit roots, fallback to dummy test
    if args.debug and not Path(args.sketchy_root).exists() and not Path(args.flickr_root).exists():
        print("[INFO] Dataset directories not found on local machine. Running with --dummy_test for debug verification.")
        args.dummy_test = True

    run_caching(
        sketchy_root=args.sketchy_root,
        flickr_root=args.flickr_root,
        out_dir=args.out_dir,
        min_size=args.min_size,
        quality=args.quality,
        workers=args.workers,
        debug=args.debug,
        dummy_test=args.dummy_test,
    )
