"""E3 / E4 – train CLIP + visual prompts (+ per-branch LayerNorm), Sain et al. CVPR 2023.

One script for the whole protocol:
  1. train on the `--train_splits` classes (default: train = 94 classes) with
         L = L_tri + lambda * (L_cls^sketch + L_cls^photo)        (E3: --lambda_cls 0)
  2. after every epoch: mAP@200 on the `--val_split` classes (10 pseudo-unseen classes);
     keep prompts_best.pt (highest val mAP@200) and prompts_last.pt (for --resume);
  3. --eval_test: load the best checkpoint and evaluate ONCE on the 21 unseen test classes
     (+ Flickr15K if given), write results/<exp>/metrics.json and the hand-off embeddings
     features/<exp>/<dataset>_test_<sketch|photo>.npy|csv.
Optional final retrain on all 104 seen classes: --train_splits train val --val_split none
--epochs <best epoch> --eval_test.

Sketches are always read from the raw roots, photos from *_photo_root (256-px cache).
Defaults come from configs/deep_e2_e4.yaml (paper: Adam lr 1e-5, 60 epochs, batch 64,
mu 0.3, lambda 0.5, prompt 3x768 in layer 1, LayerNorm tuning; tau 0.01 is ours).

Kaggle:
    python -m sbir.deep.train --exp E4 --sketchy_csv .../sketchy_zero_shot_splits.csv \
        --sketchy_root <raw>/sketchy --sketchy_photo_root <cache>/sketchy --eval_test \
        --flickr_csv .../flickr15k_splits.csv --flickr_root <raw>/Flickr15K --flickr_photo_root <cache>/flickr15k
    python -m sbir.deep.train --exp E3 --lambda_cls 0 ...          # ablation without L_cls
    add --debug for a ~3-minute end-to-end check; --resume <ckpt_dir>/<exp>/prompts_last.pt
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sbir.deep.losses import SBIRTripletLoss, SemanticClassificationLoss  # noqa: E402
from sbir.deep.prompt_clip import build_prompted_clip  # noqa: E402
from sbir.deep.sampling import TripletSampler, debug_subset  # noqa: E402
from sbir.evaluate import evaluate_retrieval  # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DISTRACTOR = "__distractor__"


# ----------------------------------------------------------------------------- utils
def resolve(root: str, rel: str) -> str:
    p = os.path.join(root, rel)
    if os.path.exists(p):
        return p
    alt = os.path.splitext(p)[0] + ".jpg"
    return alt if os.path.exists(alt) else p


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def log(msg):
    print(msg, flush=True)


def rounded(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def check_path(kind, root, rel):
    if not root:
        raise SystemExit(f"no root folder given for {kind}s")
    p = resolve(root, rel)
    if not os.path.exists(p):
        raise SystemExit(f"{kind} not found: {p}  -> check the {kind} root ({root})")


def train_transform(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class Images(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


class Triplets(Dataset):
    def __init__(self, sk_paths, ph_paths, sk_labels, s_idx, pos, neg, tf):
        self.sk, self.ph, self.y = sk_paths, ph_paths, sk_labels
        self.s, self.p, self.n, self.tf = s_idx, pos, neg, tf

    def __len__(self):
        return len(self.s)

    def _load(self, path):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, i):
        s = self.s[i]
        return (self._load(self.sk[s]), self._load(self.ph[self.p[i]]), self._load(self.ph[self.n[i]]),
                int(self.y[s]))


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def encode(model, paths, branch, tf, args, device):
    loader = DataLoader(Images(paths, tf), batch_size=args.eval_batch_size, num_workers=args.workers,
                        pin_memory=device == "cuda")
    out = []
    for x in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device == "cuda"):
            f = model.encode_image(x.to(device, non_blocking=True), branch)
        out.append(f.float().cpu())
    return torch.cat(out).numpy()


def per_class(sim, q, g):
    res = {}
    for c in sorted(set(q)):
        m = q == c
        r = evaluate_retrieval(sim[m], q[m], g, top_k_list=[10])
        res[c] = {"n_queries": int(m.sum()), "mAP": round(r["mAP"], 4), "mAP@200": round(r["mAP@200"], 4),
                  "P@10": round(r["P@10"], 4)}
    return res


def evaluate_set(model, sk, ph, sketch_root, photo_root, tf, args, device, detailed=False):
    model.eval()
    t0 = time.time()
    fs = encode(model, [resolve(sketch_root, p) for p in sk.filepath], "sketch", tf, args, device)
    fp = encode(model, [resolve(photo_root, p) for p in ph.filepath], "photo", tf, args, device)
    t_enc = time.time() - t0
    q, g = sk.category.to_numpy(), ph.category.to_numpy()
    sim = fs @ fp.T
    res = {"visual": rounded(evaluate_retrieval(sim, q, g, top_k_list=[10, 100, 200], map_cutoff=200))}
    if detailed:
        res["per_class_visual"] = per_class(sim, q, g)
        res["n_sketches"], res["n_photos"] = int(len(sk)), int(len(ph))
        res["timing"] = {"encode_ms_per_image": round(1000 * t_enc / max(1, len(sk) + len(ph)), 3)}
    return res, fs, fp


# ----------------------------------------------------------------------------- data
def rows(df, splits, kind):
    return df[df.split.isin(splits) & (df.item_type == kind)].reset_index(drop=True)


def subset(df, classes, per_class_cap):
    df = df[df.category.isin(classes)]
    if per_class_cap:
        df = df.groupby("category", group_keys=False).head(per_class_cap)
    return df.reset_index(drop=True)


def eval_frames(csv, split, debug, seed):
    df = pd.read_csv(csv)
    sk, ph = rows(df, [split], "sketch"), rows(df, [split], "photo")
    if debug:
        cls = debug_subset(sk.category, 5, seed)
        sk, ph = subset(sk, cls, 30), subset(ph, cls + [DISTRACTOR], 30)
    return sk, ph


# ----------------------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="E4", help="experiment name: E3, E4, E4_deep, E4_final ...")
    ap.add_argument("--config", default=str(REPO / "configs" / "deep_e2_e4.yaml"))
    ap.add_argument("--sketchy_csv", required=True)
    ap.add_argument("--sketchy_root", required=True, help="raw Sketchy (train/, val/): sketches")
    ap.add_argument("--sketchy_photo_root", default="", help="<cache>/sketchy for photos; empty = raw")
    ap.add_argument("--flickr_csv", default="")
    ap.add_argument("--flickr_root", default="")
    ap.add_argument("--flickr_photo_root", default="")
    ap.add_argument("--train_splits", nargs="+", default=["train"])
    ap.add_argument("--val_split", default="val", help="'none' = no validation (final retrain)")
    # hyper-parameters (None = take from the yaml config)
    for name, typ in [("epochs", int), ("lr", float), ("batch_size", int), ("lambda_cls", float),
                      ("margin", float), ("tau", float), ("num_prompts", int)]:
        ap.add_argument(f"--{name}", type=typ, default=None)
    ap.add_argument("--deep", action="store_true", default=None, help="prompts at all 12 layers (B-Deep)")
    ap.add_argument("--no_tune_ln", action="store_true", help="freeze LayerNorms too (w/o LayerNorm ablation)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--feat_dir", default="features")
    ap.add_argument("--resume", default="", help="path to prompts_last.pt")
    ap.add_argument("--eval_test", action="store_true", help="evaluate the best checkpoint on the test split")
    ap.add_argument("--eval_only", default="", help="checkpoint to evaluate; skips training")
    ap.add_argument("--max_steps", type=int, default=0, help="cap steps per epoch (quick checks)")
    ap.add_argument("--debug", action="store_true", help="20 classes x 40 items, 2 epochs, ~3 min")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) if os.path.exists(args.config) else {}
    tr, ls, md, bb = cfg.get("training", {}), cfg.get("loss", {}), cfg.get("model", {}), cfg.get("backbone", {})
    defaults = {"epochs": tr.get("epochs", 60), "lr": tr.get("lr", 1e-5), "batch_size": tr.get("batch_size", 64),
                "lambda_cls": ls.get("lambda_cls", 0.5), "margin": ls.get("triplet_margin", 0.3),
                "tau": ls.get("classification_temperature", 0.01), "num_prompts": md.get("prompt_tokens", 3),
                "deep": md.get("deep_prompting", False), "model": bb.get("name", "ViT-B-32")}
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.tune_layernorm = (not args.no_tune_ln) and bb.get("tune_layernorm", True)
    args.amp = not args.no_amp
    args.sketchy_photo_root = args.sketchy_photo_root or args.sketchy_root
    args.flickr_photo_root = args.flickr_photo_root or args.flickr_root
    if args.debug:
        args.epochs = min(args.epochs, 2)
        args.exp += "_debug"
    return args


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir) / args.exp
    res_dir = Path(args.results_dir) / args.exp
    feat_dir = Path(args.feat_dir) / args.exp
    for d in (ckpt_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)
    log(f"[{args.exp}] device={device} lambda={args.lambda_cls} lr={args.lr} epochs={args.epochs} "
        f"bs={args.batch_size} M={args.num_prompts} deep={args.deep} tune_ln={args.tune_layernorm} amp={args.amp}")

    model, _, eval_tf = build_prompted_clip(args.model, None if args.pretrained == "none" else args.pretrained,
                                            args.num_prompts, deep=args.deep,
                                            tune_layernorm=args.tune_layernorm, device=device)
    log(f"trainable parameters: {model.num_trainable():,}")
    best_path, last_path = ckpt_dir / "prompts_best.pt", ckpt_dir / "prompts_last.pt"
    history, best, best_epoch = [], -1.0, -1

    if not args.eval_only:
        # ------------------------------------------------------------ training data
        df = pd.read_csv(args.sketchy_csv)
        sk, ph = rows(df, args.train_splits, "sketch"), rows(df, args.train_splits, "photo")
        if args.debug:
            cls = debug_subset(sk.category, 20, args.seed)
            sk, ph = subset(sk, cls, 40), subset(ph, cls, 40)
        classes = sorted(sk.category.unique())
        cidx = {c: i for i, c in enumerate(classes)}
        check_path("sketch", args.sketchy_root, sk.filepath.iloc[0])
        check_path("photo", args.sketchy_photo_root, ph.filepath.iloc[0])
        sk_paths = [resolve(args.sketchy_root, p) for p in sk.filepath]
        ph_paths = [resolve(args.sketchy_photo_root, p) for p in ph.filepath]
        sk_y = sk.category.map(cidx).to_numpy()
        sampler = TripletSampler(sk_y, ph.category.map(cidx).to_numpy(), seed=args.seed)
        log(f"train: {len(classes)} classes, {len(sk):,} sketches, {len(ph):,} photos "
            f"({len(sk) // args.batch_size} steps/epoch)")

        val = None
        if args.val_split != "none":
            val = eval_frames(args.sketchy_csv, args.val_split, args.debug, args.seed)
            log(f"val: {val[0].category.nunique()} classes, {len(val[0]):,} sketches, {len(val[1]):,} photos")

        text = model.encode_text(classes, device=device)  # (C, 512) frozen anchors, seen classes only
        tri, cls_loss = SBIRTripletLoss(args.margin), SemanticClassificationLoss(args.tau)
        opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
        train_tf = train_transform()
        start = 0

        if args.resume:
            ck = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_prompt_state_dict(ck["prompts"])
            opt.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            start, best, best_epoch, history = ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck["history"]
            log(f"resumed from {args.resume}: epoch {start}, best val mAP@200 {best:.4f} @ {best_epoch}")

        for epoch in range(start, args.epochs):
            s_idx, pos, neg = sampler.epoch_triplets(epoch)
            if args.max_steps:
                n = args.max_steps * args.batch_size
                s_idx, pos, neg = s_idx[:n], pos[:n], neg[:n]
            loader = DataLoader(Triplets(sk_paths, ph_paths, sk_y, s_idx, pos, neg, train_tf),
                                batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=device == "cuda", drop_last=True)
            model.train()
            t0, sums, steps = time.time(), np.zeros(3), 0
            for xs, xp, xn, y in loader:
                xs, xp, xn, y = (t.to(device, non_blocking=True) for t in (xs, xp, xn, y))
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device == "cuda"):
                    fs = model.encode_sketch(xs)
                    fp = model.encode_photo(torch.cat([xp, xn]))
                fs, fp = fs.float(), fp.float()
                fpp, fpn = fp[: len(y)], fp[len(y):]
                l_tri = tri(fs, fpp, fpn)
                l_cls = cls_loss(fs, y, text) + cls_loss(fpp, y, text) if args.lambda_cls > 0 else torch.zeros((), device=device)
                loss = l_tri + args.lambda_cls * l_cls
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sums += [loss.item(), l_tri.item(), l_cls.item()]
                steps += 1
                if steps % 100 == 0:
                    log(f"  epoch {epoch + 1} step {steps}: loss {sums[0] / steps:.4f} "
                        f"(tri {sums[1] / steps:.4f}, cls {sums[2] / steps:.4f})")
            rec = {"epoch": epoch + 1, "loss": round(sums[0] / max(1, steps), 5), "l_tri": round(sums[1] / max(1, steps), 5),
                   "l_cls": round(sums[2] / max(1, steps), 5), "train_sec": round(time.time() - t0, 1),
                   "sec_per_step": round((time.time() - t0) / max(1, steps), 4)}

            if val is not None:
                vres, _, _ = evaluate_set(model, *val, args.sketchy_root, args.sketchy_photo_root, eval_tf, args, device)
                rec["val"] = vres["visual"]
                if rec["val"]["mAP@200"] > best:
                    best, best_epoch = rec["val"]["mAP@200"], epoch + 1
                    torch.save({"prompts": model.prompt_state_dict(), "epoch": epoch, "config": vars(args)}, best_path)
            history.append(rec)
            log(f"epoch {epoch + 1}/{args.epochs}: loss {rec['loss']:.4f} | {rec['sec_per_step']}s/step | "
                + (f"val mAP@200 {rec['val']['mAP@200']:.4f} (best {best:.4f} @ {best_epoch})" if val is not None else ""))
            torch.save({"prompts": model.prompt_state_dict(), "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                        "history": history, "config": vars(args)}, last_path)
            json.dump({"history": history, "best_val_mAP@200": best, "best_epoch": best_epoch},
                      open(res_dir / "history.json", "w"), indent=2)

        if val is None:  # final retrain: the last epoch is the model
            torch.save({"prompts": model.prompt_state_dict(), "epoch": args.epochs - 1, "config": vars(args)}, best_path)
            best_epoch = args.epochs

    # ---------------------------------------------------------------- test evaluation
    if args.eval_test or args.eval_only:
        ck_path = args.eval_only or best_path
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_prompt_state_dict(ck["prompts"])
        log(f"\n[test] checkpoint {ck_path} (epoch {ck.get('epoch', -1) + 1})")
        feat_dir.mkdir(parents=True, exist_ok=True)
        sets = [("sketchy_test", args.sketchy_csv, args.sketchy_root, args.sketchy_photo_root)]
        if args.flickr_csv:
            sets.append(("flickr15k_test", args.flickr_csv, args.flickr_root, args.flickr_photo_root))
        results = {}
        for name, csv, sroot, proot in sets:
            sk, ph = eval_frames(csv, "test", args.debug, args.seed)
            check_path("sketch", sroot, sk.filepath.iloc[0])
            check_path("photo", proot, ph.filepath.iloc[0])
            res, fs, fp = evaluate_set(model, sk, ph, sroot, proot, eval_tf, args, device, detailed=True)
            for kind, feats, rws in (("sketch", fs, sk), ("photo", fp, ph)):
                np.save(feat_dir / f"{name}_{kind}.npy", feats.astype(np.float16))
                rws[["item_id", "category"]].to_csv(feat_dir / f"{name}_{kind}.csv", index=False)
            results[name] = res
            v = res["visual"]
            log(f"  {name}: mAP@200 {v['mAP@200']:.4f} | mAP {v['mAP']:.4f} | P@10 {v['P@10']:.4f} | P@200 {v['P@200']:.4f}")
        metrics = {"experiment": args.exp, "git_commit": git_commit(), "config": vars(args),
                   "checkpoint": str(ck_path), "best_epoch": best_epoch if best_epoch > 0 else ck.get("epoch", -1) + 1,
                   "best_val_mAP@200": best if best >= 0 else None, "trainable_params": model.num_trainable(),
                   "history": history, "results": results}
        json.dump(metrics, open(res_dir / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log(f"metrics -> {res_dir / 'metrics.json'} | features -> {feat_dir}")


if __name__ == "__main__":
    main()
