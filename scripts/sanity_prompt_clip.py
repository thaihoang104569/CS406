"""Sanity check for PromptedCLIP using dummy tensors (no dataset needed).

Checks:
  1. With M = 0 prompts the re-implemented forward == clip_model.encode_image.
  2. Output shapes / L2 norm, and only the prompts are trainable.
  3. A few optimisation steps of L = L_tri + lambda * L_cls on a tiny dummy set:
     loss goes down, backbone weights stay unchanged, prompts change.
  4. Prompt-only checkpoint save / load round trip.
  5. Peak VRAM + time per step (GPU) to size batch_size before real training.

Usage (Kaggle, GPU, real CLIP weights):
    python scripts/sanity_prompt_clip.py --pretrained openai --batch_size 64 --amp
Usage (offline / CPU, random weights, no download):
    python scripts/sanity_prompt_clip.py --steps 15 --batch_size 8
"""

import argparse
import io
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sbir.deep.losses import SBIRTripletLoss, SemanticClassificationLoss  # noqa: E402
from sbir.deep.prompt_clip import build_prompted_clip  # noqa: E402


def make_dummy_data(num_classes: int, per_class: int, img_size: int, seed: int):
    """Photos = class prototype + noise; sketches = binarised grey version of the photo."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(num_classes, 3, img_size, img_size, generator=g)
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    photos = protos[labels] + 0.3 * torch.randn(len(labels), 3, img_size, img_size, generator=g)
    grey = photos.mean(1, keepdim=True)
    sketches = (grey > grey.mean()).float().expand(-1, 3, -1, -1) * 2 - 1
    return sketches.contiguous(), photos, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="none", help="'openai' on Kaggle, 'none' = random weights")
    ap.add_argument("--num_prompts", type=int, default=3)
    ap.add_argument("--shallow", action="store_true", help="prompts only at the first layer")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_cls", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = None if args.pretrained.lower() == "none" else args.pretrained
    use_amp = args.amp and device == "cuda"
    print(f"device={device}  pretrained={pretrained}  M={args.num_prompts}  "
          f"deep={not args.shallow}  amp={use_amp}")

    model, _, _ = build_prompted_clip(args.model, pretrained, args.num_prompts,
                                      deep=not args.shallow, device=device)
    clip_model = model.clip_model
    img = clip_model.visual.image_size
    img_size = img[0] if isinstance(img, (tuple, list)) else img

    # ---- 1. equivalence with M = 0 -------------------------------------
    from sbir.deep.prompt_clip import PromptedCLIP
    zero = PromptedCLIP(clip_model, num_prompts=0, deep=True).to(device)
    x = torch.randn(2, 3, img_size, img_size, device=device)
    with torch.no_grad():
        ref = torch.nn.functional.normalize(clip_model.encode_image(x), dim=-1)
        ours = zero.encode_photo(x)
    diff = (ref - ours).abs().max().item()
    print(f"[1] M=0 vs encode_image max|diff| = {diff:.2e}")
    assert diff < 1e-4, "re-implemented forward does not match CLIP"

    # ---- 2. shapes + trainable params ----------------------------------
    with torch.no_grad():
        fs, fp = model(x, x)
    assert fs.shape == (2, 512) and fp.shape == (2, 512), fs.shape
    assert torch.allclose(fs.norm(dim=-1), torch.ones(2, device=device), atol=1e-4)
    expected = 2 * model.num_prompt_layers * args.num_prompts * model.width
    assert model.num_trainable() == expected, (model.num_trainable(), expected)
    total = sum(p.numel() for p in model.parameters())
    print(f"[2] out={tuple(fs.shape)}  trainable={model.num_trainable():,} / {total:,} "
          f"({100 * model.num_trainable() / total:.3f}%)")

    # ---- 3. dummy training ----------------------------------------------
    class_names = [f"dummy_class_{i}" for i in range(args.num_classes)]
    per_class = max(2, args.batch_size // args.num_classes)
    sketches, photos, labels = make_dummy_data(args.num_classes, per_class, img_size, args.seed)
    text_anchors = model.encode_text(class_names, device=device)  # (C, 512), frozen

    tri_loss = SBIRTripletLoss(margin=0.3)
    cls_loss = SemanticClassificationLoss(temperature=0.01)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    backbone_before = sum(p.double().sum().item() for p in clip_model.parameters())
    prompts_before = model.sketch_prompts.detach().clone()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    losses, t0 = [], time.time()
    n = len(labels)
    for step in range(args.steps):
        idx = torch.randperm(n)[: args.batch_size]
        s, p, y = sketches[idx].to(device), photos[idx].to(device), labels[idx].to(device)
        # Negative = a photo from a different class (hardest-free, random).
        neg_idx = torch.stack([
            torch.nonzero(labels != yi.item()).flatten()[torch.randint(0, n - per_class, (1,))]
            for yi in y.cpu()
        ]).flatten()
        pn = photos[neg_idx].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            fs = model.encode_sketch(s)
            fp = model.encode_photo(torch.cat([p, pn]))
            fpp, fpn = fp[: len(y)], fp[len(y):]
            l_tri = tri_loss(fs.float(), fpp.float(), fpn.float())
            l_cls = (cls_loss(fs.float(), y, text_anchors) + cls_loss(fpp.float(), y, text_anchors))
            loss = l_tri + args.lambda_cls * l_cls

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if step == 0 or (step + 1) % 5 == 0:
            print(f"    step {step + 1:3d}  loss={loss.item():.4f}  tri={l_tri.item():.4f}  cls={l_cls.item():.4f}")
    dt = (time.time() - t0) / args.steps

    assert all(p.grad is None for p in clip_model.parameters()), "backbone received gradients"
    backbone_after = sum(p.double().sum().item() for p in clip_model.parameters())
    assert backbone_after == backbone_before, "backbone weights changed"
    assert not torch.equal(prompts_before, model.sketch_prompts.detach()), "prompts did not update"
    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    print(f"[3] loss {first:.4f} -> {last:.4f}  ({dt:.2f} s/step, batch={args.batch_size})")
    assert last < first, "loss did not decrease on the dummy set"

    # ---- 4. prompt checkpoint round trip --------------------------------
    buf = io.BytesIO()
    torch.save(model.prompt_state_dict(), buf)
    size_kb = buf.tell() / 1024
    buf.seek(0)
    fresh = PromptedCLIP(clip_model, num_prompts=args.num_prompts, deep=not args.shallow).to(device)
    fresh.load_prompt_state_dict(torch.load(buf))
    model.eval()
    with torch.no_grad():
        a, b = model.encode_sketch(x), fresh.encode_sketch(x)
    assert torch.allclose(a, b, atol=1e-5)
    print(f"[4] prompt checkpoint {size_kb:.0f} KB, reload OK")

    # ---- 5. resources ---------------------------------------------------
    if device == "cuda":
        print(f"[5] peak VRAM = {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
