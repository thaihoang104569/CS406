"""CLIP with Deep Visual Prompt Learning (Sain et al., CVPR 2023).

The whole CLIP ViT-B/32 backbone (image + text encoder) is frozen. Only two sets
of visual prompts are learned:

    P_S  (sketch branch)  shape (L, M, D)
    P_P  (photo branch)   shape (L, M, D)

with L = number of ViT layers that receive prompts (12 for deep prompting,
1 for shallow), M = prompt tokens per layer, D = ViT width (768 for ViT-B/32).

Token layout fed to the transformer (VPT-deep, Jia et al. ECCV 2022):

    layer 0     : [CLS, P[0], patch_1 .. patch_49]
    layer i > 0 : the M prompt outputs of layer i-1 are discarded and replaced
                  by the fresh learnable P[i]; CLS and patch tokens flow through.

The final embedding is ln_post(CLS) @ proj, L2-normalised (512-D), exactly as in
the original CLIP image encoder. With M = 0 the forward pass reduces to
``clip_model.encode_image`` (checked in scripts/sanity_prompt_clip.py).

Re-implemented from the paper description; the official Sketch_LVM repository
was only used as a reference for the overall design.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BRANCHES = ("sketch", "photo")


def to_open_clip_name(model_name: str) -> str:
    """Map OpenAI-style names ('ViT-B/32') to open_clip names ('ViT-B-32')."""
    return model_name.replace("/", "-")


class PromptedCLIP(nn.Module):
    """Frozen open_clip model + learnable deep visual prompts for sketch and photo.

    Args:
        clip_model: an ``open_clip`` CLIP model (e.g. from ``open_clip.create_model``).
        tokenizer: matching ``open_clip`` tokenizer (only needed for ``encode_text``).
        num_prompts: M, prompt tokens per layer (paper / config: M in {1, 3, 5}).
        deep: True -> prompts at every transformer layer; False -> first layer only.
        prompt_init_std: std of the normal initialisation of the prompts.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer=None,
        num_prompts: int = 3,
        deep: bool = True,
        prompt_init_std: float = 0.02,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.num_prompts = num_prompts
        self.deep = deep

        visual = clip_model.visual
        if getattr(visual, "attn_pool", None) is not None:
            raise NotImplementedError("Attentional-pooling ViTs are not supported.")
        self.width = visual.conv1.out_channels
        self.depth = len(visual.transformer.resblocks)
        self.num_prompt_layers = self.depth if deep else 1
        # Newer open_clip transformers are batch-first; older ones expect (L, N, D).
        self._batch_first = getattr(visual.transformer, "batch_first", False)

        # Freeze the entire CLIP backbone (image and text encoders).
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        shape = (self.num_prompt_layers, num_prompts, self.width)
        self.sketch_prompts = nn.Parameter(torch.randn(shape) * prompt_init_std)
        self.photo_prompts = nn.Parameter(torch.randn(shape) * prompt_init_std)

    # ------------------------------------------------------------------ utils
    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode even while the prompts train."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def _prompts(self, branch: str) -> torch.Tensor:
        if branch == "sketch":
            return self.sketch_prompts
        if branch == "photo":
            return self.photo_prompts
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._batch_first:
            return block(x)
        return block(x.permute(1, 0, 2)).permute(1, 0, 2)

    # ---------------------------------------------------------------- forward
    def encode_image(self, x: torch.Tensor, branch: str, normalize: bool = True) -> torch.Tensor:
        """Encode a batch of preprocessed images (B, 3, 224, 224) -> (B, 512)."""
        v = self.clip_model.visual
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
            if 0 < i < self.num_prompt_layers:
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
    def prompt_state_dict(self) -> dict:
        """Only the learned prompts (~a few hundred KB) - backbone is reloaded from open_clip."""
        return {
            "sketch_prompts": self.sketch_prompts.detach().cpu(),
            "photo_prompts": self.photo_prompts.detach().cpu(),
            "num_prompts": self.num_prompts,
            "deep": self.deep,
        }

    def load_prompt_state_dict(self, state: dict) -> None:
        if state["num_prompts"] != self.num_prompts or state["deep"] != self.deep:
            raise ValueError("Checkpoint prompt config does not match the model.")
        with torch.no_grad():
            self.sketch_prompts.copy_(state["sketch_prompts"])
            self.photo_prompts.copy_(state["photo_prompts"])


# Backwards-compatible alias for the earlier stub name.
PromptedVisionTransformer = PromptedCLIP


def build_prompted_clip(
    model_name: str = "ViT-B-32",
    pretrained: Optional[str] = "openai",
    num_prompts: int = 3,
    deep: bool = True,
    device: str = "cpu",
):
    """Create a PromptedCLIP plus the open_clip train/eval preprocess transforms.

    ``pretrained=None`` builds random weights without downloading anything
    (handy for dummy-tensor tests before the dataset/weights are available).
    """
    import open_clip

    name = to_open_clip_name(model_name)
    clip_model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
        name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(name)
    model = PromptedCLIP(clip_model, tokenizer, num_prompts=num_prompts, deep=deep).to(device)
    return model, preprocess_train, preprocess_eval
