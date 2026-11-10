"""CLIP with Deep Visual Prompt Learning (Sain et al., CVPR 2023).
Freezes ViT-B/32 backbone and introduces learnable visual prompt tokens
for Sketch branch (P_S) and Photo branch (P_P).
"""

import torch
import torch.nn as nn


class PromptedVisionTransformer(nn.Module):
    """Adds deep learnable visual prompt tokens to a Vision Transformer backbone."""
    def __init__(
        self,
        clip_model,
        num_prompts: int = 3,
        prompt_dim: int = 768,
        num_layers: int = 12,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.num_layers = num_layers

        # Freeze the entire CLIP backbone
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Learnable prompts for Sketch branch
        self.sketch_prompts = nn.Parameter(
            torch.randn(num_layers, num_prompts, prompt_dim) * 0.02
        )
        # Learnable prompts for Photo branch
        self.photo_prompts = nn.Parameter(
            torch.randn(num_layers, num_prompts, prompt_dim) * 0.02
        )

    def forward_sketch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for sketch input with sketch prompts."""
        # Stub: will inject self.sketch_prompts into ViT transformer blocks
        # Returns normalized 512-D embedding
        feat = self.clip_model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)

    def forward_photo(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for photo input with photo prompts."""
        feat = self.clip_model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)
