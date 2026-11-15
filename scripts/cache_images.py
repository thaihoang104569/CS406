"""One-shot image caching script for Sketchy Extended & Flickr15K datasets.
Resizes images maintaining aspect ratio with longest side <= 256px, saved as JPEG (quality=95).
Preserves relative folder structure so code can simply switch data_root to cache folder.

Usage on Kaggle:
    python scripts/cache_images.py \
        --sketchy_root /kaggle/input/datasets/hongonthi/sketchyextended/sketchy \
        --flickr_root  /kaggle/input/datasets/hongonthi/flickr15k/Flickr15K \
        --out_dir      /kaggle/working/sbir-cache-256 \
        --workers      8

Usage for local testing (--debug):
    python scripts/cache_images.py --debug
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def scan_images(root_dir: Path) -> List[Tuple[Path, Path]]:
    """Recursively scans directory for images, returning list of (full_path, relative_path)."""
    if not root_dir.exists():
        return []
    
    image_pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        dp = Path(dirpath)
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in IMG_EXTENSIONS:
                full_path = dp / fname
                rel_path = full_path.relative_to(root_dir)
                image_pairs.append((full_path, rel_path))
    return image_pairs


def process_and_save_image(src_path: Path, dst_path: Path, max_size: int = 256, quality: int = 95) -> bool:
    """Loads image, resizes longest side to max_size, and saves to dst_path."""
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            # Convert to RGB if needed (handles grayscale, palette, RGBA)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / float(max(w, h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            
            # Save as JPEG with quality 95
            img.save(dst_path, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"[WARNING] Failed to process {src_path}: {e}", file=sys.stderr)
        return False


def run_caching(
    sketchy_root: str,
    flickr_root: str,
    out_dir: str,
    max_size: int = 256,
    quality: int = 95,
    workers: int = 8,
    debug: bool = False,
    dummy_test: bool = False,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    tasks: List[Tuple[Path, Path]] = []

    if dummy_test:
        print("[INFO] Running in dummy_test mode. Generating dummy images...")
        dummy_src = out_path / "_dummy_src"
        dummy_src.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            p = dummy_src / f"img_{i:03d}.png"
            Image.new("RGB", (600, 400), color=(i * 10, 100, 150)).save(p)
            tasks.append((p, Path("dummy") / f"img_{i:03d}.jpg"))
    else:
        # 1. Sketchy tasks
        sketchy_p = Path(sketchy_root)
        if sketchy_p.exists():
            print(f"[INFO] Scanning Sketchy dataset at: {sketchy_p}")
            sk_pairs = scan_images(sketchy_p)
            print(f"       Found {len(sk_pairs):,} Sketchy images.")
            for full_p, rel_p in sk_pairs:
                # Target path: out_dir/sketchy/<rel_p>.jpg
                dst = out_path / "sketchy" / rel_p.with_suffix(".jpg")
                tasks.append((full_p, dst))
        else:
            print(f"[WARNING] Sketchy root does not exist: {sketchy_root}")

        # 2. Flickr15k tasks
        flickr_p = Path(flickr_root)
        if flickr_p.exists():
            print(f"[INFO] Scanning Flickr15k dataset at: {flickr_p}")
            fl_pairs = scan_images(flickr_p)
            print(f"       Found {len(fl_pairs):,} Flickr15k images.")
            for full_p, rel_p in fl_pairs:
                # Target path: out_dir/flickr15k/<rel_p>.jpg
                dst = out_path / "flickr15k" / rel_p.with_suffix(".jpg")
                tasks.append((full_p, dst))
        else:
            print(f"[WARNING] Flickr15k root does not exist: {flickr_root}")

    if not tasks:
        print("[ERROR] No images found to process. Please check dataset paths or use --dummy_test.")
        return False

    if debug:
        print(f"[DEBUG MODE] Limiting to first 200 images for fast verification.")
        tasks = tasks[:200]

    print(f"\n--- Starting 256px Image Caching ---")
    print(f"Total images:   {len(tasks):,}")
    print(f"Output folder:  {out_path.resolve()}")
    print(f"Max dimension:  {max_size} px")
    print(f"JPEG Quality:   {quality}")
    print(f"Worker threads: {workers}")

    start_time = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_and_save_image, src, dst, max_size, quality): (src, dst)
            for src, dst in tasks
        }
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Caching images"):
            if fut.result():
                success_count += 1

    elapsed = time.time() - start_time
    rate = success_count / max(elapsed, 0.001)

    print(f"\n--- Caching Summary ---")
    print(f"Successfully cached: {success_count:,} / {len(tasks):,} images")
    print(f"Time elapsed:        {elapsed:.2f} s ({rate:.1f} images/sec)")
    print(f"Output directory:    {out_path.resolve()}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache dataset images resized to max 256px for fast training.")
    parser.add_argument("--sketchy_root", type=str, default="data/raw/sketchy",
                        help="Path to raw Sketchy dataset directory")
    parser.add_argument("--flickr_root", type=str, default="data/raw/flickr15k",
                        help="Path to raw Flickr15k dataset directory")
    parser.add_argument("--out_dir", type=str, default="data/cache_256",
                        help="Destination directory for cached 256px images")
    parser.add_argument("--max_size", type=int, default=256,
                        help="Longest dimension to resize images to")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG compression quality (1-100)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent worker threads")
    parser.add_argument("--debug", action="store_true",
                        help="Only process the first 200 images for fast verification")
    parser.add_argument("--dummy_test", action="store_true",
                        help="Generate and cache synthetic images for end-to-end local test")

    args = parser.parse_args()

    # If roots don't exist and debug is passed without explicit roots, fallback to dummy test
    if args.debug and not Path(args.sketchy_root).exists() and not Path(args.flickr_root).exists():
        print("[INFO] Dataset directories not found on local machine. Running with --dummy_test for debug verification.")
        args.dummy_test = True

    run_caching(
        sketchy_root=args.sketchy_root,
        flickr_root=args.flickr_root,
        out_dir=args.out_dir,
        max_size=args.max_size,
        quality=args.quality,
        workers=args.workers,
        debug=args.debug,
        dummy_test=args.dummy_test,
    )
