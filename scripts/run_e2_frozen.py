"""E2 – frozen CLIP ViT-B/32 baseline (+ the text-prompt variant of the proposal, section 6).

For every evaluated set (Sketchy test/val, Flickr15K) it
  1. encodes sketches and photos with the frozen CLIP image encoder (no training),
  2. saves the embeddings in the hand-off format of the plan (section 2.3):
       <feat_dir>/<dataset>_<split>_<sketch|photo>.npy  (N x 512 float16, L2-normalised)
       <feat_dir>/<dataset>_<split>_<sketch|photo>.csv  (item_id, category, same row order)
  3. evaluates sketch -> photo retrieval by cosine (sbir/evaluate.py: mAP, mAP@200, P@K),
  4. text variant: classify each sketch zero-shot against "a photo of a <class>" for the
     classes of that set, then rank photos by similarity to the PREDICTED class's text
     embedding (semantic retrieval instead of visual retrieval),
  5. records per-class AP, the top-1 sketch classification accuracy, encoding time and
     FAISS query time (inputs for E5),
and writes everything to <out_dir>/metrics.json.

Kaggle (GPU):
    python scripts/run_e2_frozen.py \
        --sketchy_csv /kaggle/working/data_prep/splits/sketchy_zero_shot_splits.csv \
        --sketchy_root <raw>/sketchy   --sketchy_photo_root <cache>/sketchy \
        --flickr_csv /kaggle/working/data_prep/splits/flickr15k_splits.csv \
        --flickr_root <raw>/Flickr15K  --flickr_photo_root <cache>/flickr15k \
        --out_dir /kaggle/working/results/E2 --feat_dir /kaggle/working/features/E2
    Sketches are always read from the raw roots; photos from *_photo_root when given (the
    photo-only 256-px cache, plan section 2.4), otherwise from the raw roots too.
Local check without torch/CLIP (synthetic embeddings, real CSVs + evaluation + saving):
    python scripts/run_e2_frozen.py --sketchy_csv ... --mock --debug
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbir.evaluate import evaluate_retrieval  # noqa: E402
from sbir.index import VectorIndex  # noqa: E402

DISTRACTOR = "__distractor__"


# ----------------------------------------------------------------------------- data
def resolve(root: str, rel: str) -> str:
    """Path of `rel` under `root`; falls back to the .jpg the 256-px cache writes."""
    p = os.path.join(root, rel)
    if os.path.exists(p):
        return p
    alt = os.path.splitext(p)[0] + ".jpg"
    return alt if os.path.exists(alt) else p


def load_set(csv_path: str, split: str, debug: int):
    df = pd.read_csv(csv_path)
    df = df[df.split == split]
    if df.empty:
        raise ValueError(f"No rows with split={split} in {csv_path}")
    if debug:  # a few items per class, deterministic
        per = max(2, debug // max(1, df.category.nunique()))
        df = df.groupby(["item_type", "category"], group_keys=False).head(per)
    sk = df[df.item_type == "sketch"].reset_index(drop=True)
    ph = df[df.item_type == "photo"].reset_index(drop=True)
    return sk, ph


def class_prompt(name: str, template: str) -> str:
    return template.format(name.replace("_", " ").replace("-", " ").strip())


# ----------------------------------------------------------------------------- encoders
class ClipEncoder:
    def __init__(self, args):
        import torch
        from PIL import Image, ImageFilter
        from torch.utils.data import DataLoader, Dataset

        from sbir.deep.frozen import FrozenCLIPExtractor

        self.torch, self.args = torch, args
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained = None if args.pretrained.lower() == "none" else args.pretrained
        self.clip = FrozenCLIPExtractor(args.model, pretrained, device=device)
        if self.clip.model is None:
            raise SystemExit("open_clip_torch is not installed (pip install open_clip_torch)")
        self.device = device
        preprocess = self.clip.preprocess

        class Files(Dataset):
            def __init__(self, paths, dilate):
                self.paths, self.dilate = paths, dilate

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                if self.dilate > 1:  # dark strokes on white: a min filter thickens them
                    img = img.filter(ImageFilter.MinFilter(self.dilate))
                return preprocess(img)

        self.Files, self.DataLoader = Files, DataLoader

    def images(self, paths, is_sketch):
        dilate = self.args.sketch_dilate if is_sketch else 0
        loader = self.DataLoader(self.Files(paths, dilate), batch_size=self.args.batch_size,
                                 num_workers=self.args.workers, pin_memory=self.device == "cuda")
        out = [self.clip.encode_image_batch(b, amp=True) for b in loader]
        return self.torch.cat(out).numpy()

    def texts(self, prompts):
        return self.clip.encode_text(prompts).float().numpy()


class MockEncoder:
    """Synthetic class-structured embeddings: tests the whole pipeline without torch/CLIP."""

    def __init__(self, args, dim=512):
        self.dim = dim

    def _class_vec(self, name):
        return np.random.default_rng(zlib.crc32(name.encode())).standard_normal(self.dim)

    def _noisy(self, labels, seed, strength):
        rng = np.random.default_rng(seed)
        x = np.stack([self._class_vec(c) for c in labels]) + strength * rng.standard_normal((len(labels), self.dim))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    def images_from_labels(self, labels, is_sketch):
        return self._noisy(labels, 1 if is_sketch else 2, 9.0 if is_sketch else 7.0)

    def texts_from_names(self, names):
        return self._noisy(names, 3, 2.0)


# ----------------------------------------------------------------------------- evaluation
def per_class(sim, q_lab, g_lab):
    out = {}
    for c in sorted(set(q_lab)):
        m = q_lab == c
        r = evaluate_retrieval(sim[m], q_lab[m], g_lab, top_k_list=[10])
        out[c] = {"n_queries": int(m.sum()), "mAP": round(r["mAP"], 4), "mAP@200": round(r["mAP@200"], 4),
                  "P@10": round(r["P@10"], 4)}
    return out


def rounded(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def check_paths(name, kind, root, rels):
    if not root:
        raise SystemExit(f"[{name}] no root folder given for {kind}s")
    p = resolve(root, rels.iloc[0])
    if not os.path.exists(p):
        raise SystemExit(f"[{name}] {kind} not found: {p}  -> check the {kind} root ({root})")


def run_set(name, csv_path, root, photo_root, split, enc, args, feat_dir):
    sk, ph = load_set(csv_path, split, args.debug)
    photo_root = photo_root or root
    if not args.mock:
        check_paths(name, "sketch", root, sk.filepath)
        check_paths(name, "photo", photo_root, ph.filepath)
        print(f"[{name}] sketches from {root} | photos from {photo_root}", flush=True)
    print(f"\n[{name}] {len(sk)} sketches x {len(ph)} photos, {sk.category.nunique()} query classes", flush=True)

    t0 = time.time()
    if args.mock:
        fs = enc.images_from_labels(sk.category.tolist(), True)
    else:
        fs = enc.images([resolve(root, p) for p in sk.filepath], True)
    t_sk = time.time() - t0
    t0 = time.time()
    if args.mock:
        fp = enc.images_from_labels(ph.category.tolist(), False)
    else:
        fp = enc.images([resolve(photo_root, p) for p in ph.filepath], False)
    t_ph = time.time() - t0

    for kind, feats, rows in (("sketch", fs, sk), ("photo", fp, ph)):
        stem = feat_dir / f"{name}_{kind}"
        np.save(f"{stem}.npy", feats.astype(np.float16))
        rows[["item_id", "category"]].to_csv(f"{stem}.csv", index=False)

    q_lab, g_lab = sk.category.to_numpy(), ph.category.to_numpy()
    sim = fs @ fp.T
    visual = evaluate_retrieval(sim, q_lab, g_lab, top_k_list=[10, 100, 200], map_cutoff=200)

    # text variant: zero-shot classify the sketch, retrieve by the predicted class's text embedding
    classes = sorted(set(g_lab) - {DISTRACTOR})
    prompts = [class_prompt(c, args.template) for c in classes]
    t = enc.texts_from_names(classes) if args.mock else enc.texts(prompts)
    pred = (fs @ t.T).argmax(1)
    acc = float((np.array(classes)[pred] == q_lab).mean())
    text = evaluate_retrieval(t[pred] @ fp.T, q_lab, g_lab, top_k_list=[10, 100, 200], map_cutoff=200)

    # FAISS query time over the gallery (E5)
    idx = VectorIndex(dim=fp.shape[1], use_gpu=args.faiss_gpu)
    idx.add(fp)
    t0 = time.time()
    idx.search(fs, top_k=args.top_k)
    t_q = time.time() - t0

    res = {
        "n_sketches": int(len(sk)), "n_photos": int(len(ph)), "n_classes": len(classes),
        "visual": rounded(visual),
        "text_variant": {**rounded(text), "sketch_top1_accuracy": round(acc, 4)},
        "per_class_visual": per_class(sim, q_lab, g_lab),
        "timing": {
            "encode_ms_per_sketch": round(1000 * t_sk / max(1, len(sk)), 3),
            "encode_ms_per_photo": round(1000 * t_ph / max(1, len(ph)), 3),
            "query_ms_per_sketch": round(1000 * t_q / max(1, len(sk)), 4),
            "index_backend": idx.backend,
            "index_mb_float16": round(fp.shape[0] * fp.shape[1] * 2 / 2**20, 2),
        },
        "prompts_example": prompts[:3],
    }
    v, tv = res["visual"], res["text_variant"]
    print(f"  visual : mAP {v['mAP']:.4f} | mAP@200 {v['mAP@200']:.4f} | P@10 {v['P@10']:.4f} | P@200 {v['P@200']:.4f}")
    print(f"  text   : mAP {tv['mAP']:.4f} | mAP@200 {tv['mAP@200']:.4f} | P@10 {tv['P@10']:.4f} | top-1 acc {acc:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sketchy_csv", required=True)
    ap.add_argument("--sketchy_root", default="", help="raw Sketchy folder (train/, val/); sketches always read here")
    ap.add_argument("--sketchy_photo_root", default="", help="<cache>/sketchy for photos; empty = use --sketchy_root")
    ap.add_argument("--splits", nargs="+", default=["test", "val"], help="Sketchy splits to evaluate")
    ap.add_argument("--flickr_csv")
    ap.add_argument("--flickr_root", default="", help="raw Flickr15K folder (images/, sketches/)")
    ap.add_argument("--flickr_photo_root", default="", help="<cache>/flickr15k for photos; empty = use --flickr_root")
    ap.add_argument("--out_dir", default="results/E2")
    ap.add_argument("--feat_dir", default="features/E2")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--template", default="a photo of a {}")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sketch_dilate", type=int, default=0,
                    help="min-filter size (odd, e.g. 3) to thicken sketch strokes before CLIP; 0 = off")
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--faiss_gpu", action="store_true")
    ap.add_argument("--debug", type=int, default=0, metavar="N", help="use only ~N items per split/type")
    ap.add_argument("--mock", action="store_true", help="synthetic embeddings, no torch/CLIP needed")
    args = ap.parse_args()

    out_dir, feat_dir = Path(args.out_dir), Path(args.feat_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    enc = MockEncoder(args) if args.mock else ClipEncoder(args)

    sets = [(f"sketchy_{s}", args.sketchy_csv, args.sketchy_root, args.sketchy_photo_root, s) for s in args.splits]
    if args.flickr_csv:
        sets.append(("flickr15k_test", args.flickr_csv, args.flickr_root, args.flickr_photo_root, "test"))

    t0 = time.time()
    results = {name: run_set(name, csv, root, proot, split, enc, args, feat_dir) for name, csv, root, proot, split in sets}
    metrics = {
        "experiment": "E2_frozen_clip" + ("_MOCK" if args.mock else "") + ("_DEBUG" if args.debug else ""),
        "git_commit": git_commit(),
        "config": vars(args),
        "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== E2 summary (visual retrieval / text variant) ===")
    for name, r in results.items():
        v, tv = r["visual"], r["text_variant"]
        print(f"{name:16s} mAP@200 {v['mAP@200']:.4f} / {tv['mAP@200']:.4f}   mAP {v['mAP']:.4f} / {tv['mAP']:.4f}   "
              f"P@200 {v['P@200']:.4f} / {tv['P@200']:.4f}")
    print(f"\nmetrics -> {out_dir / 'metrics.json'}\nfeatures -> {feat_dir}")


if __name__ == "__main__":
    main()
