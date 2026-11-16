"""E1 – Traditional Baseline (Canny + Dense HOG + Bag-of-Visual-Words + Spatial Pyramid).

Pipeline:
  1. Train Seen Codebook:
     - Preprocess sample train photos & sketches (94 seen classes).
     - Extract Dense HOG descriptors (32x32 patch, stride 8, cell 8x8 -> 324-D).
     - Sample 300,000 descriptors to train MiniBatchKMeans (K=1000).
     - Compute document frequency to fit smooth TF-IDF weights.
     - Save codebook to <weights_dir>/codebook_k1000.pkl.
  2. Feature Extraction (Spatial Pyramid Matching 1x1 + 2x2 -> 5000-D):
     - Photo: bilateral filter -> Auto-Canny -> CC filter -> crop bbox 10% pad -> Dense HOG.
     - Sketch: Otsu -> crop bbox 10% pad -> Zhang-Suen -> 3x3 dilate -> Dense HOG.
     - SPM 5000-D vector + Power Normalization + L2 Normalization.
     - Saves embeddings in standard hand-off format (Section 2.3 of Plan):
         <feat_dir>/<dataset>_<split>_<sketch|photo>.npy  (N x 5000 float16, L2-normalised)
         <feat_dir>/<dataset>_<split>_<sketch|photo>.csv  (item_id, category)
  3. Evaluation:
     - Cosine similarity retrieval on Sketchy Test (21 unseen classes) & Flickr15K (33 classes).
     - Computes mAP, mAP@200, P@10, P@100, P@200, and per-class AP.
     - Writes results to <out_dir>/metrics.json.

Usage on Kaggle:
    python scripts/run_e1_baseline.py \
        --sketchy_csv /kaggle/working/data_prep/splits/sketchy_zero_shot_splits.csv \
        --sketchy_photo_root /kaggle/input/sbir-cache-256-photos/sketchy \
        --sketchy_sketch_root /kaggle/input/datasets/hongonthi/sketchyextended/sketchy \
        --flickr_csv /kaggle/working/data_prep/splits/flickr15k_splits.csv \
        --flickr_photo_root /kaggle/input/sbir-cache-256-photos/flickr15k \
        --flickr_sketch_root /kaggle/input/datasets/hongonthi/flickr15k/Flickr15K \
        --out_dir /kaggle/working/results/E1 \
        --feat_dir /kaggle/working/features/E1 \
        --weights_dir /kaggle/working/weights/E1 \
        --workers 4

Usage for quick local test (--debug / --mock):
    python scripts/run_e1_baseline.py --mock --debug 20
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbir.evaluate import evaluate_retrieval  # noqa: E402
from sbir.traditional.bovw import VisualCodebook  # noqa: E402
from sbir.traditional.features import extract_dense_hog  # noqa: E402
from sbir.traditional.preprocess import preprocess_photo, preprocess_sketch  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def resolve_image_path(photo_root: str, sketch_root: str, relpath: str, item_type: str) -> str:
    """Resolve full path: photo reads from photo_root, sketch reads from sketch_root.
    
    Robustly handles subfolder prefixes and extension changes (.png <-> .jpg).
    """
    root = photo_root if item_type == "photo" else sketch_root
    
    # Direct candidate
    direct = os.path.join(root, relpath)
    candidates = [
        direct,
        os.path.splitext(direct)[0] + ".jpg",
        os.path.splitext(direct)[0] + ".png",
    ]
    # In case root has dataset subfolder or vice versa
    for dataset_prefix in ["sketchy", "flickr15k"]:
        if not root.rstrip("/\\").endswith(dataset_prefix):
            sub = os.path.join(root, dataset_prefix, relpath)
            candidates.extend([
                sub,
                os.path.splitext(sub)[0] + ".jpg",
                os.path.splitext(sub)[0] + ".png",
            ])
            
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return direct


def load_dataset_split(csv_path: str, split: str, debug: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df = df[df.split == split]
    if df.empty:
        raise ValueError(f"No rows with split='{split}' in {csv_path}")

    if debug > 0:
        per = max(2, debug // max(1, df.category.nunique()))
        df = df.groupby(["item_type", "category"], group_keys=False).head(per)

    sketches = df[df.item_type == "sketch"].reset_index(drop=True)
    photos = df[df.item_type == "photo"].reset_index(drop=True)
    return sketches, photos


def extract_single_image_features(
    image_path: str,
    item_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads image, applies preprocessing (photo or sketch), extracts dense HOG."""
    try:
        # Load image via OpenCV
        img = cv2.imread(image_path)
        if img is None:
            # Fallback to PIL
            with Image.open(image_path) as pil_img:
                img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        if item_type == "photo":
            processed = preprocess_photo(img, target_size=256)
        else:
            processed = preprocess_sketch(img, target_size=256)

        desc, pos = extract_dense_hog(
            processed,
            patch_size=32,
            step_size=8,
            orientations=9,
            cell_size=8,
            min_edge_ratio=0.02,
        )
        return desc, pos
    except Exception as e:
        print(f"[WARNING] Feature extraction failed for {image_path}: {e}", file=sys.stderr)
        return np.empty((0, 324), dtype=np.float32), np.empty((0, 2), dtype=np.int32)


# ----------------------------------------------------------------------------- Codebook Training
def train_visual_codebook(
    train_df: pd.DataFrame,
    photo_root: str,
    sketch_root: str,
    n_clusters: int = 1000,
    max_descriptors: int = 300000,
    weights_path: Optional[Path] = None,
    workers: int = 4,
) -> VisualCodebook:
    """Samples descriptors from train set and fits MiniBatchKMeans + TF-IDF."""
    if weights_path and weights_path.exists():
        print(f"[INFO] Found pre-trained codebook at: {weights_path}. Loading...")
        return VisualCodebook.load(weights_path)

    print(f"\n--- [1/3] Training Visual Codebook (K={n_clusters}, sample={max_descriptors:,}) ---")
    
    # Shuffle train images
    shuffled = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    collected_descriptors: List[np.ndarray] = []
    total_desc_count = 0
    sample_histograms: List[np.ndarray] = []

    codebook = VisualCodebook(n_clusters=n_clusters, batch_size=4096, random_state=42)

    print("Collecting Dense HOG descriptors from seen training set...")
    for idx in range(len(shuffled)):
        row = shuffled.iloc[idx]
        img_path = resolve_image_path(photo_root, sketch_root, row["filepath"], row["item_type"])
        if not os.path.exists(img_path):
            continue

        desc, pos = extract_single_image_features(img_path, row["item_type"])
        if len(desc) > 0:
            collected_descriptors.append(desc)
            total_desc_count += len(desc)

        if total_desc_count >= max_descriptors:
            break

    if not collected_descriptors:
        raise RuntimeError("No HOG descriptors could be extracted from training set.")

    all_desc = np.vstack(collected_descriptors)
    if len(all_desc) > max_descriptors:
        indices = np.random.RandomState(42).choice(len(all_desc), size=max_descriptors, replace=False)
        all_desc = all_desc[indices]

    print(f"Fitting MiniBatchKMeans on {len(all_desc):,} descriptors...")
    t0 = time.time()
    codebook.fit(all_desc)
    print(f"KMeans fitted in {time.time() - t0:.1f} s.")

    # Compute IDF from sample image word histograms
    print("Fitting TF-IDF weights across training images...")
    for desc in collected_descriptors[:1000]:  # sample 1000 images for IDF
        words = codebook.assign_words(desc)
        h, _ = np.histogram(words, bins=n_clusters, range=(0, n_clusters))
        sample_histograms.append(h)

    if sample_histograms:
        codebook.fit_idf(np.array(sample_histograms, dtype=np.float32))

    if weights_path:
        codebook.save(weights_path)
        print(f"Saved VisualCodebook to {weights_path}")

    return codebook


# ----------------------------------------------------------------------------- SPM Encoding
def encode_dataset_split(
    df: pd.DataFrame,
    codebook: VisualCodebook,
    photo_root: str,
    sketch_root: str,
    item_type: str,
    workers: int = 4,
    mock: bool = False,
) -> np.ndarray:
    """Encodes all items in df into L2-normalized 5000-D SPM vectors."""
    n_items = len(df)
    dim = 5 * codebook.n_clusters

    if mock:
        print(f"[MOCK] Generating random 5000-D embeddings for {n_items} {item_type} items...")
        np.random.seed(42)
        vecs = np.random.randn(n_items, dim).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def process_item(i: int) -> np.ndarray:
        row = df.iloc[i]
        img_p = resolve_image_path(photo_root, sketch_root, row["filepath"], item_type)
        if not os.path.exists(img_p):
            return np.zeros(dim, dtype=np.float32)

        desc, pos = extract_single_image_features(img_p, item_type)
        return codebook.compute_spatial_pyramid(desc, pos, img_size=256, levels=[1, 2], apply_idf=True)

    vectors = np.empty((n_items, dim), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, vec in tqdm(enumerate(executor.map(process_item, range(n_items))),
                             total=n_items, desc=f"SPM 5000-D ({item_type})"):
            vectors[idx] = vec

    return vectors


# ----------------------------------------------------------------------------- Main Pipeline
def run_e1_evaluation(
    dataset_name: str,
    sketches_df: pd.DataFrame,
    photos_df: pd.DataFrame,
    sketch_vecs: np.ndarray,
    photo_vecs: np.ndarray,
    feat_dir: Path,
    map_cutoff: int = 200,
) -> Dict:
    feat_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save embeddings and CSVs in hand-off format
    base_sk = f"{dataset_name}_test_sketch"
    base_ph = f"{dataset_name}_test_photo"

    np.save(feat_dir / f"{base_sk}.npy", sketch_vecs.astype(np.float16))
    sketches_df[["item_id", "category"]].to_csv(feat_dir / f"{base_sk}.csv", index=False)

    np.save(feat_dir / f"{base_ph}.npy", photo_vecs.astype(np.float16))
    photos_df[["item_id", "category"]].to_csv(feat_dir / f"{base_ph}.csv", index=False)

    # 2. Compute similarity matrix in batches (queries @ gallery.T)
    t0 = time.time()
    n_queries = len(sketch_vecs)
    n_gallery = len(photo_vecs)
    sim_matrix = np.empty((n_queries, n_gallery), dtype=np.float32)

    bs = 1024
    for s in range(0, n_queries, bs):
        chunk = sketch_vecs[s:s + bs]
        sim_matrix[s:s + len(chunk)] = chunk @ photo_vecs.T

    sim_time = time.time() - t0

    # 3. Evaluate retrieval
    query_labels = sketches_df["category"].values
    gallery_labels = photos_df["category"].values

    metrics = evaluate_retrieval(
        similarity_matrix=sim_matrix,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        top_k_list=[10, 100, 200],
        map_cutoff=map_cutoff,
        map_mode="min_r_k",
    )

    # 4. Per-class AP
    unique_classes = sorted(np.unique(query_labels))
    per_class = {}
    for cls_name in unique_classes:
        q_mask = (query_labels == cls_name)
        if not np.any(q_mask):
            continue
        cls_sim = sim_matrix[q_mask]
        cls_q_labels = query_labels[q_mask]
        cls_res = evaluate_retrieval(
            similarity_matrix=cls_sim,
            query_labels=cls_q_labels,
            gallery_labels=gallery_labels,
            top_k_list=[10, 100, 200],
            map_cutoff=map_cutoff,
            map_mode="min_r_k",
        )
        per_class[cls_name] = {
            "n_queries": int(np.sum(q_mask)),
            "mAP": round(cls_res["mAP"], 4),
            f"mAP@{map_cutoff}": round(cls_res[f"mAP@{map_cutoff}"], 4),
            "P@10": round(cls_res["P@10"], 4),
        }

    return {
        "n_sketches": n_queries,
        "n_photos": n_gallery,
        "n_classes": len(unique_classes),
        "visual": metrics,
        "per_class": per_class,
        "sim_time_sec": round(sim_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="E1 Baseline: Canny + Dense HOG + BoVW + Spatial Pyramid (5000-D)")
    parser.add_argument("--sketchy_csv", type=str, default="data/splits/sketchy_zero_shot_splits.csv")
    parser.add_argument("--sketchy_photo_root", type=str, default="data/cache_256_photos/sketchy")
    parser.add_argument("--sketchy_sketch_root", type=str, default="data/raw/sketchy")
    parser.add_argument("--flickr_csv", type=str, default="data/splits/flickr15k_splits.csv")
    parser.add_argument("--flickr_photo_root", type=str, default="data/cache_256_photos/flickr15k")
    parser.add_argument("--flickr_sketch_root", type=str, default="data/raw/flickr15k")
    parser.add_argument("--out_dir", type=str, default="results/E1")
    parser.add_argument("--feat_dir", type=str, default="features/E1")
    parser.add_argument("--weights_dir", type=str, default="weights/E1")
    parser.add_argument("--k_words", type=int, default=1000)
    parser.add_argument("--sample_descriptors", type=int, default=300000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--debug", type=int, default=0, help="Debug mode: limit items per class")
    parser.add_argument("--mock", action="store_true", help="Mock mode: test logic with random vectors")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    feat_dir = Path(args.feat_dir)
    weights_dir = Path(args.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== E1 Baseline: Canny + Dense HOG + BoVW (K={args.k_words}, 5000-D) ===")
    print(f"Git commit: {get_git_commit()}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Feat dir:   {feat_dir.resolve()}")

    codebook_path = weights_dir / f"codebook_k{args.k_words}.pkl"

    # 1. Codebook
    if not args.mock:
        if codebook_path.exists():
            codebook = VisualCodebook.load(codebook_path)
        else:
            # Load train set to train codebook
            train_df = pd.read_csv(args.sketchy_csv)
            train_df = train_df[train_df.split == "train"]
            codebook = train_visual_codebook(
                train_df,
                photo_root=args.sketchy_photo_root,
                sketch_root=args.sketchy_sketch_root,
                n_clusters=args.k_words,
                max_descriptors=args.sample_descriptors,
                weights_path=codebook_path,
                workers=args.workers,
            )
    else:
        codebook = VisualCodebook(n_clusters=args.k_words)
        codebook.is_fitted = True

    results = {}

    # 2. Sketchy Test (21 unseen classes)
    if os.path.exists(args.sketchy_csv) or args.mock:
        print("\n--- [2/3] Evaluating Sketchy Test (21 unseen classes) ---")
        if not args.mock:
            sk_sketches, sk_photos = load_dataset_split(args.sketchy_csv, split="test", debug=args.debug)
        else:
            classes = [f"cls_{i}" for i in range(5)]
            sk_sketches = pd.DataFrame({
                "item_id": [f"sk_{i}" for i in range(25)],
                "category": np.random.choice(classes, 25),
                "filepath": [f"sk_{i}.png" for i in range(25)],
            })
            sk_photos = pd.DataFrame({
                "item_id": [f"ph_{i}" for i in range(50)],
                "category": np.random.choice(classes, 50),
                "filepath": [f"ph_{i}.jpg" for i in range(50)],
            })

        print(f"Sketchy Test items: {len(sk_sketches):,} sketches, {len(sk_photos):,} photos.")
        sk_vecs = encode_dataset_split(sk_sketches, codebook, args.sketchy_photo_root, args.sketchy_sketch_root, "sketch", args.workers, args.mock)
        ph_vecs = encode_dataset_split(sk_photos, codebook, args.sketchy_photo_root, args.sketchy_sketch_root, "photo", args.workers, args.mock)

        results["sketchy_test"] = run_e1_evaluation(
            dataset_name="sketchy",
            sketches_df=sk_sketches,
            photos_df=sk_photos,
            sketch_vecs=sk_vecs,
            photo_vecs=ph_vecs,
            feat_dir=feat_dir,
            map_cutoff=200,
        )

    # 3. Flickr15K Test (33 classes)
    if os.path.exists(args.flickr_csv) or args.mock:
        print("\n--- [3/3] Evaluating Flickr15K Test (33 classes) ---")
        if not args.mock:
            fl_sketches, fl_photos = load_dataset_split(args.flickr_csv, split="test", debug=args.debug)
        else:
            classes = [f"fl_cls_{i}" for i in range(5)]
            fl_sketches = pd.DataFrame({
                "item_id": [f"fl_sk_{i}" for i in range(15)],
                "category": np.random.choice(classes, 15),
                "filepath": [f"fl_sk_{i}.png" for i in range(15)],
            })
            fl_photos = pd.DataFrame({
                "item_id": [f"fl_ph_{i}" for i in range(40)],
                "category": np.random.choice(classes, 40),
                "filepath": [f"fl_ph_{i}.jpg" for i in range(40)],
            })

        print(f"Flickr15K Test items: {len(fl_sketches):,} sketches, {len(fl_photos):,} photos.")
        fl_sk_vecs = encode_dataset_split(fl_sketches, codebook, args.flickr_photo_root, args.flickr_sketch_root, "sketch", args.workers, args.mock)
        fl_ph_vecs = encode_dataset_split(fl_photos, codebook, args.flickr_photo_root, args.flickr_sketch_root, "photo", args.workers, args.mock)

        results["flickr15k_test"] = run_e1_evaluation(
            dataset_name="flickr15k",
            sketches_df=fl_sketches,
            photos_df=fl_photos,
            sketch_vecs=fl_sk_vecs,
            photo_vecs=fl_ph_vecs,
            feat_dir=feat_dir,
            map_cutoff=200,
        )

    # Save metrics.json
    final_output = {
        "experiment": "E1_baseline_traditional",
        "git_commit": get_git_commit(),
        "config": vars(args),
        "results": results,
    }

    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"✅ E1 Baseline completed successfully!")
    print(f"Metrics saved to: {metrics_file.resolve()}")
    if "sketchy_test" in results:
        v = results["sketchy_test"]["visual"]
        print(f"Sketchy Test: mAP@200 = {v['mAP@200']:.4f} | P@10 = {v['P@10']:.4f} | P@200 = {v['P@200']:.4f}")
    if "flickr15k_test" in results:
        v = results["flickr15k_test"]["visual"]
        print(f"Flickr15K:   mAP = {v['mAP']:.4f} | mAP@200 = {v['mAP@200']:.4f} | P@10 = {v['P@10']:.4f}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
