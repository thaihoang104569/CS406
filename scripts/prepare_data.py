"""One-shot data preparation for Sketchy Extended + Flickr15K: EDA, splits, leakage check.

Run on Kaggle (reads the raw dataset folders, writes everything to --out_dir):

    python scripts/prepare_data.py \
        --sketchy_root /kaggle/input/datasets/hongonthi/sketchyextended/sketchy \
        --flickr_root  /kaggle/input/datasets/hongonthi/flickr15k/Flickr15K \
        --out_dir      /kaggle/working/data_prep

Output (copy splits/ to data/splits/ and the rest to results/data/ in the repo):

    data_prep/
    ├── report.json                     every number for the report (counts, splits, image stats, checks)
    ├── splits/
    │   ├── sketchy_zero_shot_splits.csv    item level: item_id, filepath, category, item_type, split, photo_id, dataset
    │   ├── sketchy_class_splits.csv        class level: 94 train / 10 val / 21 test
    │   ├── flickr15k_splits.csv            item level, everything split = test
    │   └── flickr15k_classes.csv           33 classes: class_id, name, image folders
    └── figures/                        5 PNGs (class counts, samples, image statistics)

Split rules
-----------
Sketchy: dataset folder train/ (104 classes) = seen, folder val/ (21 classes) = unseen TEST
(Yelamarthi et al. 2018). n_val seen classes (default 10, seed 42) become the validation split
used for tuning / early stopping, the rest is train. Split by CLASS, before any processing.
Flickr15K: no training data -> all "test". Photo label = class_id from list_images.txt (the 60
image folders are NOT the 33 classes); photos missing from the list are distractors. Sketch
label = number in its file name (sketches/<user>/<class_id>.png).
Paths in the CSVs are relative to the dataset root (use SBIRDataset(..., data_root=...)).
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
DISTRACTOR = "__distractor__"
SPLIT_ORDER = ["train", "val", "test"]


# ----------------------------------------------------------------------------- helpers
def scan(root):
    out, stack = [], [root]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif os.path.splitext(e.name)[1].lower() in IMG_EXT:
                    out.append(os.path.relpath(e.path, root).replace(os.sep, "/"))
    return pd.Series(sorted(out), dtype=str)


def stem_of(relpath):
    return relpath.str.split("/").str[-1].str.rsplit(".", n=1).str[0]


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- Sketchy
def build_sketchy(root, n_val, seed):
    rel = scan(root)
    parts = rel.str.split("/")
    df = pd.DataFrame({"filepath": rel, "folder": parts.str[0], "item_type": parts.str[1],
                       "category": parts.str[-2]})
    bad = ~df.folder.isin(["train", "val"]) | ~df.item_type.isin(["photo", "sketch"]) | (parts.str.len() < 4)
    if bad.any():
        raise ValueError(f"Unexpected Sketchy layout, e.g. {df.loc[bad, 'filepath'].head(3).tolist()}")

    stem = stem_of(df.filepath)
    # sketch "n02691156_10151-3" was drawn from photo "n02691156_10151"
    df["photo_id"] = stem.where(df.item_type == "photo", stem.str.replace(r"-\d+$", "", regex=True))
    # extended photos are named ext_1.jpg, ext_2.jpg ... in every class -> id must include the class
    df["item_id"] = "sketchy/" + df.item_type + "/" + df.category + "/" + stem
    df["dataset"] = "sketchy"
    if not df.item_id.is_unique:
        raise ValueError(f"Duplicate item_id: {df.loc[df.item_id.duplicated(), 'filepath'].head(3).tolist()}")

    seen = sorted(df.loc[df.folder == "train", "category"].unique())
    unseen = sorted(df.loc[df.folder == "val", "category"].unique())
    if set(seen) & set(unseen):
        raise ValueError("Sketchy train/ and val/ folders share classes: not a zero-shot split.")
    val_classes = sorted(random.Random(seed).sample(seen, n_val))
    cls = pd.DataFrame({"category": seen + unseen,
                        "source_folder": ["train"] * len(seen) + ["val"] * len(unseen)})
    cls["split"] = np.select([cls.source_folder == "val", cls.category.isin(val_classes)], ["test", "val"], "train")
    df["split"] = df.category.map(dict(zip(cls.category, cls.split)))

    # --- checks / statistics
    photos, sketches = df[df.item_type == "photo"], df[df.item_type == "sketch"]
    photo_split = photos.groupby(["category", "photo_id"]).split.first()
    sk_photo_split = pd.Series(list(zip(sketches.category, sketches.photo_id))).map(photo_split)
    spp = sketches.groupby(["category", "photo_id"]).size()
    counts = df.pivot_table(index="category", columns="item_type", values="item_id", aggfunc="count", fill_value=0)
    stats = {
        "files": int(len(df)),
        "classes": {s: int((cls.split == s).sum()) for s in SPLIT_ORDER},
        "items": {f"{s}/{t}": int(n) for (s, t), n in df.groupby(["split", "item_type"]).size().items()},
        "val_classes": val_classes,
        "test_classes": unseen,
        "original_photos_with_sketches": int(len(spp)),
        "extended_photos": int(len(photos) - photos.set_index(["category", "photo_id"]).index.isin(spp.index).sum()),
        "sketches_per_photo": {"mean": round(float(spp.mean()), 3), "min": int(spp.min()), "max": int(spp.max())},
        "sketches_without_photo": sketches.loc[sk_photo_split.isna().values, "filepath"].tolist(),
        "sketch_photo_pairs_across_splits": int((sk_photo_split.notna() & (sk_photo_split.values != sketches.split.values)).sum()),
        "per_class": {"photo_min": int(counts.photo.min()), "photo_max": int(counts.photo.max()),
                      "sketch_min": int(counts.sketch.min()), "sketch_max": int(counts.sketch.max())},
        "extensions": {f"{t}{e}": int(n) for (t, e), n in
                       df.groupby([df.item_type, df.filepath.str.extract(r"(\.\w+)$")[0].str.lower()]).size().items()},
    }
    cols = ["item_id", "filepath", "category", "item_type", "split", "photo_id", "dataset"]
    return df[cols].sort_values(["split", "item_type", "item_id"], ignore_index=True), cls, stats


# ----------------------------------------------------------------------------- Flickr15K
def build_flickr(root):
    rel = scan(root)
    parts = rel.str.split("/")
    df = pd.DataFrame({"filepath": rel, "top": parts.str[0], "group": parts.str[1]})
    df = df[df.top.isin(["images", "sketches"]) & (parts.str.len() == 3)].copy()
    df["item_type"] = df.top.map({"images": "photo", "sketches": "sketch"})
    stem = stem_of(df.filepath)

    li = []
    with open(os.path.join(root, "list_images.txt"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            toks = line.strip().replace("\\", "/").split(maxsplit=1)
            if len(toks) == 2 and toks[0].isdigit():
                li.append((int(toks[0]), re.sub(r"^(\./)?images/", "", toks[1].strip())))
    li = pd.DataFrame(li, columns=["class_id", "key"])
    if li.empty:
        raise ValueError("list_images.txt: no '<class_id> <folder>/<file>' lines")

    is_sk = df.item_type == "sketch"
    df["class_id"] = df.filepath.str.replace(r"^images/", "", regex=True).map(dict(zip(li.key, li.class_id)))
    df.loc[is_sk, "class_id"] = pd.to_numeric(stem[is_sk], errors="coerce")
    if df.loc[is_sk, "class_id"].isna().any():
        raise ValueError(f"Sketch names are not class ids: {df.loc[is_sk & df.class_id.isna(), 'filepath'].head().tolist()}")

    lab = df[(df.item_type == "photo") & df.class_id.notna()]
    g = lab.groupby("class_id")["group"]
    cls = pd.DataFrame({"class_id": g.size().index.astype(int),
                        "category": g.agg(lambda x: x.value_counts().index[0]).values,  # majority folder
                        "folders": g.agg(lambda x: ";".join(sorted(x.unique()))).values,
                        "n_photos": g.size().values})
    cls["n_sketches"] = cls.class_id.map(df[is_sk].groupby("class_id").size()).fillna(0).astype(int)
    dup = cls.category.duplicated(keep=False)
    cls.loc[dup, "category"] = cls.loc[dup, "category"] + "_" + cls.loc[dup, "class_id"].astype(str)

    df["category"] = df.class_id.map(dict(zip(cls.class_id, cls.category))).fillna(DISTRACTOR)
    df["item_id"] = "flickr15k/" + df.item_type + "/" + df.group + "/" + stem
    df["photo_id"] = stem.where(df.item_type == "photo", "")
    df["split"], df["dataset"] = "test", "flickr15k"
    df["class_id"] = df.class_id.astype("Int64")
    unlab = df[df.category == DISTRACTOR]
    stats = {
        "files": int(len(df)),
        "photos": int((~is_sk).sum()), "sketches": int(is_sk.sum()),
        "image_folders": int(df.loc[~is_sk, "group"].nunique()),
        "classes": int(len(cls)),
        "sketch_users": int(df.loc[is_sk, "group"].nunique()),
        "list_images_lines": int(len(li)),
        "list_lines_without_file": int((~li.key.isin(set(df.filepath.str[len("images/"):]))).sum()),
        "distractor_photos": int(len(unlab)),
        "distractor_folders": {k: int(v) for k, v in unlab.group.value_counts().items()},
    }
    cols = ["item_id", "filepath", "category", "item_type", "split", "photo_id", "dataset", "class_id"]
    return df[cols].sort_values(["item_type", "item_id"], ignore_index=True), cls, stats


# ----------------------------------------------------------------------------- image statistics
def img_stats(path):
    try:
        with Image.open(path) as im:
            w, h, mode = im.width, im.height, im.mode
            im = im.convert("L")
            im.thumbnail((512, 512))  # keep native resolution of small sketches: Flickr strokes are 1px grey
            a = np.asarray(im, dtype=np.uint8)
        return w, h, mode, float(a.mean()), float((a < 200).mean())
    except Exception as e:  # noqa: BLE001
        return None, None, f"ERROR:{type(e).__name__}", None, None


def sample_image_stats(df, root, per_group, seed, workers):
    s = (df.sample(frac=1, random_state=seed).groupby(["split", "item_type"]).head(per_group)
           .reset_index(drop=True))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda p: img_stats(os.path.join(root, p)), s.filepath))
    s[["width", "height", "mode", "mean_gray", "ink_ratio"]] = pd.DataFrame(res)
    return s


def summarize_stats(s):
    out = {}
    for (sp, t), g in s.groupby(["split", "item_type"]):
        ok = g[~g["mode"].astype(str).str.startswith("ERROR")]
        out[f"{sp}/{t}"] = {
            "n_sampled": int(len(g)), "unreadable": g.loc[g["mode"].astype(str).str.startswith("ERROR"), "filepath"].tolist(),
            "width": [int(ok.width.min()), int(ok.width.median()), int(ok.width.max())],
            "height": [int(ok.height.min()), int(ok.height.median()), int(ok.height.max())],
            "modes": dict(Counter(ok["mode"])),
            "mean_gray": round(float(ok.mean_gray.mean()), 2),
            "ink_ratio": round(float(ok.ink_ratio.mean()), 4),
        }
    return out


# ----------------------------------------------------------------------------- figures
def make_figures(sk, sk_stats, fl, fl_cls, fl_stats, sk_root, fl_root, fig_dir, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}
    rng = random.Random(seed)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    def show(ax, path, title):
        ax.axis("off")
        ax.set_title(title, fontsize=7)
        if path:
            try:
                ax.imshow(Image.open(path).convert("RGB"))
            except Exception:  # noqa: BLE001
                pass

    # 1. Sketchy: files per class, coloured by split
    c = sk.pivot_table(index=["split", "category"], columns="item_type", values="item_id", aggfunc="count", fill_value=0)
    c = c.reset_index().sort_values(["split", "photo"], key=lambda x: x.map(SPLIT_ORDER.index) if x.name == "split" else -x)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(c) * 0.13), 7), sharex=True)
    for ax, t in zip(axes, ["photo", "sketch"]):
        ax.bar(range(len(c)), c[t], color=c.split.map(colors), width=0.85)
        ax.set_ylabel(f"# {t}")
    axes[1].set_xticks(range(len(c)), c.category, rotation=90, fontsize=5)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()],
                   labels=[f"{k} ({sk_stats['classes'][k]} lớp)" for k in colors], ncol=3)
    axes[0].set_title("Sketchy Extended: số file mỗi lớp theo split")
    save(fig, "1_sketchy_class_counts")

    # 2. Sketchy: photo + 4 sketches drawn from it, 2 classes per split
    ph, sks = sk[sk.item_type == "photo"], sk[sk.item_type == "sketch"]
    paired = ph.merge(sks[["category", "photo_id"]].drop_duplicates(), on=["category", "photo_id"])
    rows = []
    for split in SPLIT_ORDER:
        cats = sorted(paired.loc[paired.split == split, "category"].unique())
        for cat in rng.sample(cats, min(2, len(cats))):
            p = paired[(paired.split == split) & (paired.category == cat)].sample(1, random_state=seed).iloc[0]
            s = sks[(sks.category == cat) & (sks.photo_id == p.photo_id)].filepath.head(4).tolist()
            rows.append((split, cat, p.filepath, s))
    if rows:
        fig, axes = plt.subplots(len(rows), 5, figsize=(9, 1.9 * len(rows)), squeeze=False)
        for r, (split, cat, p, s) in enumerate(rows):
            show(axes[r, 0], os.path.join(sk_root, p), f"[{split}] {cat}")
            for k in range(4):
                show(axes[r, k + 1], os.path.join(sk_root, s[k]) if k < len(s) else None, "sketch" if k < len(s) else "")
        fig.suptitle("Sketchy: ảnh và các sketch vẽ từ chính ảnh đó")
        save(fig, "2_sketchy_samples")

    if fl is not None:
        # 3. Flickr15K: photos per class (label from list_images.txt) + distractors
        f = fl_cls.sort_values("n_photos", ascending=False)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(f.category, f.n_photos, color="tab:green")
        ax.set_xticks(range(len(f)), f.category, rotation=90, fontsize=7)
        ax.set_title(f"Flickr15K: {len(f)} lớp, {int(f.n_photos.sum())} ảnh có nhãn, "
                     f"{fl_stats['distractor_photos']} ảnh nhiễu, {fl_stats['sketches']} sketch")
        save(fig, "3_flickr15k_class_counts")

        # 4. Flickr15K: one photo + sketches of 3 users for 6 classes
        fph, fsk = fl[(fl.item_type == "photo") & (fl.category != DISTRACTOR)], fl[fl.item_type == "sketch"]
        cats = rng.sample(sorted(fph.category.unique()), min(6, fph.category.nunique()))
        fig, axes = plt.subplots(len(cats), 4, figsize=(7.5, 1.9 * len(cats)), squeeze=False)
        for r, cat in enumerate(cats):
            show(axes[r, 0], os.path.join(fl_root, fph[fph.category == cat].sample(1, random_state=seed).filepath.iloc[0]), cat)
            s = fsk[fsk.category == cat].filepath.head(3).tolist()
            for k in range(3):
                show(axes[r, k + 1], os.path.join(fl_root, s[k]) if k < len(s) else None,
                     f"user {s[k].split('/')[1]}" if k < len(s) else "")
        fig.suptitle("Flickr15K: ảnh và sketch của cùng lớp")
        save(fig, "4_flickr15k_samples")

    return plt


def figure_image_stats(plt, stats_df, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (ds, t), g in stats_df.groupby(["dataset", "item_type"]):
        axes[0].scatter(g.width, g.height, s=4, alpha=0.4, label=f"{ds}/{t}")
    axes[0].set(title="Độ phân giải (mẫu)", xlabel="width", ylabel="height", xscale="log", yscale="log")
    axes[0].legend(markerscale=4)
    for ds, g in stats_df[stats_df.item_type == "sketch"].groupby("dataset"):
        axes[1].hist(g.ink_ratio.dropna(), bins=60, alpha=0.6, density=True, label=ds)
    axes[1].set(title="Mật độ nét sketch (tỉ lệ pixel < 200)", xlabel="ink ratio")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "5_image_stats.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sketchy_root", required=True, help="folder containing train/ and val/")
    ap.add_argument("--flickr_root", help="folder containing images/, sketches/, list_images.txt")
    ap.add_argument("--out_dir", default="/kaggle/working/data_prep")
    ap.add_argument("--n_val", type=int, default=10, help="seen classes held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_per_group", type=int, default=500, help="images read per split/type for pixel stats")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    split_dir, fig_dir = out / "splits", out / "figures"
    split_dir.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args)}

    log("[1/5] Sketchy: scan + zero-shot split")
    sk, sk_cls, sk_stats = build_sketchy(args.sketchy_root, args.n_val, args.seed)
    sk.to_csv(split_dir / "sketchy_zero_shot_splits.csv", index=False)
    sk_cls.to_csv(split_dir / "sketchy_class_splits.csv", index=False)
    report["sketchy"] = sk_stats
    log(f"      {sk_stats['classes']}  {sk_stats['items']}")

    fl = fl_cls = fl_stats = None
    if args.flickr_root:
        log("[2/5] Flickr15K: scan + labels from list_images.txt")
        fl, fl_cls, fl_stats = build_flickr(args.flickr_root)
        fl.to_csv(split_dir / "flickr15k_splits.csv", index=False)
        fl_cls.to_csv(split_dir / "flickr15k_classes.csv", index=False)
        report["flickr15k"] = fl_stats
        log(f"      {fl_stats['classes']} classes, {fl_stats['photos']} photos "
            f"({fl_stats['distractor_photos']} distractors), {fl_stats['sketches']} sketches")
    else:
        log("[2/5] Flickr15K: skipped (no --flickr_root)")

    log("[3/5] Leakage check")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from verify_splits import verify_split_csv  # noqa: E402
    ok = verify_split_csv(str(split_dir / "sketchy_zero_shot_splits.csv"))
    tv = set(sk_cls.category[sk_cls.split == "train"]) & set(sk_cls.category[sk_cls.split == "val"])
    report["checks"] = {
        "verify_splits_passed": bool(ok),
        "train_val_shared_classes": sorted(tv),
        "sketch_photo_pairs_across_splits": sk_stats["sketch_photo_pairs_across_splits"],
    }
    if not ok or tv or sk_stats["sketch_photo_pairs_across_splits"]:
        raise SystemExit("LEAKAGE DETECTED - see output above")

    log(f"[4/5] Image statistics ({args.sample_per_group} files per split/type)")
    parts = [sample_image_stats(sk, args.sketchy_root, args.sample_per_group, args.seed, args.workers).assign(dataset="sketchy")]
    if fl is not None:
        parts.append(sample_image_stats(fl, args.flickr_root, args.sample_per_group, args.seed, args.workers)
                     .assign(dataset="flickr15k"))
    stats_df = pd.concat(parts, ignore_index=True)
    report["image_stats"] = {ds: summarize_stats(g) for ds, g in stats_df.groupby("dataset")}

    if not args.no_figures:
        log("[5/5] Figures")
        plt = make_figures(sk, sk_stats, fl, fl_cls, fl_stats, args.sketchy_root, args.flickr_root, fig_dir, args.seed)
        figure_image_stats(plt, stats_df, fig_dir)

    report["runtime_sec"] = round(time.time() - t0, 1)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"\nDone in {report['runtime_sec']}s -> {out.resolve()}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            log(f"  {p.stat().st_size / 1024:9.1f} KB  {p.relative_to(out)}")


if __name__ == "__main__":
    main()
