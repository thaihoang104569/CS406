"""CLIP + visual prompt learning for ZS-SBIR (Sain et al., CVPR 2023, arXiv 2303.13440).

Default configuration = the paper's ZS-SBIR model ("Ours" in its Table 1):
  * CLIP ViT-B/32; sketch encoder F_s and photo encoder F_p are BOTH initialised from
    CLIP's image encoder (two copies), and stay frozen EXCEPT their LayerNorm layers,
    which are trained separately per branch (paper sec. 4: "we tune ... the trainable
    parameters of every layer normalisation (LN) layers");
  * one learnable prompt per branch, v_s and v_p, each of shape (3 x 768), injected
    in the FIRST transformer layer only (paper sec. 6: "Our prompts are injected in
    the first layer of transformer").
Trainable parameters: prompts 2 x 3 x 768 + LayerNorms 2 x 26 x 1536 = 84,480.

Ablations (paper Table 1 baselines):
  * deep=True               -> prompts at every layer (VPT-deep, baseline "B-Deep");
  * tune_layernorm=False    -> backbone fully frozen, one shared encoder
                               (shallow prompts only = baseline "B-IP").

Token layout (VPT, Jia et al. ECCV 2022):
    layer 0     : [CLS, P[0], patch_1 .. patch_49]          (prompts get no pos-embedding)
    layer i > 0 : deep only - the prompt outputs of layer i-1 are replaced by P[i].
The embedding is ln_post(CLS) @ proj, L2-normalised (512-D), as in CLIP. With M = 0 and
untouched LayerNorms the forward pass equals ``clip_model.encode_image``
(checked in scripts/sanity_prompt_clip.py).

Re-implemented from the paper; the official Sketch_LVM code is only a reference.
"""

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BRANCHES = ("sketch", "photo")


def to_open_clip_name(model_name: str) -> str:
    """Map OpenAI-style names ('ViT-B/32') to open_clip names ('ViT-B-32')."""
    return model_name.replace("/", "-")


class PromptedCLIP(nn.Module):
    """open_clip CLIP + learnable visual prompts (+ per-branch LayerNorms) for sketch and photo.

    Args:
        clip_model: an ``open_clip`` CLIP model (e.g. from ``open_clip.create_model``).
        tokenizer: matching ``open_clip`` tokenizer (only needed for ``encode_text``).
        num_prompts: M, prompt tokens per prompted layer (paper: 3).
        deep: False (paper) -> prompts in the first layer only; True -> every layer.
        tune_layernorm: True (paper) -> separate sketch/photo copies of the image encoder
            whose LayerNorm weights are trainable; False -> one fully frozen shared encoder.
        prompt_init_std: std of the normal initialisation of the prompts.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer=None,
        num_prompts: int = 3,
        deep: bool = False,
        tune_layernorm: bool = True,
        prompt_init_std: float = 0.02,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.num_prompts = num_prompts
        self.deep = deep
        self.tune_layernorm = tune_layernorm

        visual = clip_model.visual
        if getattr(visual, "attn_pool", None) is not None:
            raise NotImplementedError("Attentional-pooling ViTs are not supported.")
        self.width = visual.conv1.out_channels
        self.depth = len(visual.transformer.resblocks)
        self.num_prompt_layers = self.depth if deep else 1
        # Newer open_clip transformers are batch-first; older ones expect (L, N, D).
        self._batch_first = getattr(visual.transformer, "batch_first", False)

        # Freeze the whole CLIP model (image + text encoders).
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # Paper: F_s and F_p initialised from CLIP's image encoder, LayerNorms trainable per branch.
        if tune_layernorm:
            self.visual_sketch = copy.deepcopy(visual)
            self.visual_photo = copy.deepcopy(visual)
            for vis in (self.visual_sketch, self.visual_photo):
                for m in vis.modules():
                    if isinstance(m, nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = True

        shape = (self.num_prompt_layers, num_prompts, self.width)
        self.sketch_prompts = nn.Parameter(torch.randn(shape) * prompt_init_std)
        self.photo_prompts = nn.Parameter(torch.randn(shape) * prompt_init_std)

    # ------------------------------------------------------------------ utils
    def train(self, mode: bool = True):
        """Frozen parts stay in eval mode while prompts / LayerNorms train."""
        super().train(mode)
        self.clip_model.eval()
        if self.tune_layernorm:
            self.visual_sketch.eval()
            self.visual_photo.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _check(self, branch: str):
        if branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")

    def _prompts(self, branch: str) -> torch.Tensor:
        self._check(branch)
        return self.sketch_prompts if branch == "sketch" else self.photo_prompts

    def _visual(self, branch: str) -> nn.Module:
        self._check(branch)
        if not self.tune_layernorm:
            return self.clip_model.visual
        return self.visual_sketch if branch == "sketch" else self.visual_photo

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._batch_first:
            return block(x)
        return block(x.permute(1, 0, 2)).permute(1, 0, 2)

    # ---------------------------------------------------------------- forward
    def encode_image(self, x: torch.Tensor, branch: str, normalize: bool = True) -> torch.Tensor:
        """Encode a batch of preprocessed images (B, 3, 224, 224) -> (B, 512)."""
        v = self._visual(branch)
        prompts = self._prompts(branch)
        M = self.num_prompts

        x = v.conv1(x)                                   # (B, D, g, g)
        B, D = x.shape[0], x.shape[1]
        x = x.reshape(B, D, -1).permute(0, 2, 1)         # (B, g*g, D)
        cls = v.class_embedding.to(x.dtype).expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)                   # (B, 1+g*g, D)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)

        # Layer 0: insert prompts right after CLS (prompts get no positional embedding).
        p0 = prompts[0].to(x.dtype).expand(B, M, D)
        x = torch.cat([x[:, :1], p0, x[:, 1:]], dim=1)   # (B, 1+M+g*g, D)

        for i, block in enumerate(v.transformer.resblocks):
            if 0 < i < self.num_prompt_layers:           # deep prompting only
                pi = prompts[i].to(x.dtype).expand(B, M, D)
                x = torch.cat([x[:, :1], pi, x[:, 1 + M:]], dim=1)
            x = self._run_block(block, x)

        feat = v.ln_post(x[:, 0])                        # CLS token
        if v.proj is not None:
            feat = feat @ v.proj
        return F.normalize(feat, dim=-1) if normalize else feat

    def encode_sketch(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_image(x, "sketch")

    def encode_photo(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_image(x, "photo")

    def forward(self, sketch: torch.Tensor, photo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_sketch(sketch), self.encode_photo(photo)

    @torch.no_grad()
    def encode_text(
        self,
        class_names: Sequence[str],
        template: str = "a photo of a {}",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Frozen text anchors t_c for each class name -> (C, 512), L2-normalised."""
        if self.tokenizer is None:
            raise RuntimeError("PromptedCLIP was built without a tokenizer.")
        device = device or self.sketch_prompts.device
        texts = [template.format(c.replace("_", " ")) for c in class_names]
        tokens = self.tokenizer(texts).to(device)
        feat = self.clip_model.encode_text(tokens)
        return F.normalize(feat.float(), dim=-1)

    # ------------------------------------------------------------ checkpoints
    def _layernorm_state(self) -> Dict[str, torch.Tensor]:
        if not self.tune_layernorm:
            return {}
        return {f"{b}.{n}": p.detach().cpu()
                for b, vis in (("sketch", self.visual_sketch), ("photo", self.visual_photo))
                for n, p in vis.named_parameters() if p.requires_grad}

    def prompt_state_dict(self) -> dict:
        """Only what is learned (prompts + per-branch LayerNorms, ~340 KB); CLIP is reloaded from open_clip."""
        return {
            "sketch_prompts": self.sketch_prompts.detach().cpu(),
            "photo_prompts": self.photo_prompts.detach().cpu(),
            "layernorms": self._layernorm_state(),
            "num_prompts": self.num_prompts,
            "deep": self.deep,
            "tune_layernorm": self.tune_layernorm,
        }

    def load_prompt_state_dict(self, state: dict) -> None:
        for key in ("num_prompts", "deep"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Checkpoint {key}={state[key]} does not match the model.")
        if state.get("tune_layernorm", False) != self.tune_layernorm:
            raise ValueError("Checkpoint tune_layernorm does not match the model.")
        with torch.no_grad():
            self.sketch_prompts.copy_(state["sketch_prompts"])
            self.photo_prompts.copy_(state["photo_prompts"])
            if self.tune_layernorm:
                params = {f"{b}.{n}": p
                          for b, vis in (("sketch", self.visual_sketch), ("photo", self.visual_photo))
                          for n, p in vis.named_parameters() if p.requires_grad}
                missing = set(params) ^ set(state["layernorms"])
                if missing:
                    raise ValueError(f"LayerNorm keys mismatch, e.g. {sorted(missing)[:3]}")
                for k, p in params.items():
                    p.copy_(state["layernorms"][k])


# Backwards-compatible alias for the earlier stub name.
PromptedVisionTransformer = PromptedCLIP


def build_prompted_clip(
    model_name: str = "ViT-B-32",
    pretrained: Optional[str] = "openai",
    num_prompts: int = 3,
    deep: bool = False,
    tune_layernorm: bool = True,
    device: str = "cpu",
):
    """Create a PromptedCLIP plus the open_clip train/eval preprocess transforms.

    Defaults reproduce the paper (shallow prompts, M=3, per-branch LayerNorm tuning).
    ``pretrained=None`` builds random weights without downloading anything
    (handy for dummy-tensor tests before the dataset/weights are available).
    """
    import open_clip

    name = to_open_clip_name(model_name)
    clip_model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
        name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(name)
    model = PromptedCLIP(clip_model, tokenizer, num_prompts=num_prompts, deep=deep,
                         tune_layernorm=tune_layernorm).to(device)
    return model, preprocess_train, preprocess_eval
