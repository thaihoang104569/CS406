"""Frozen CLIP Baseline (E2).
Extracts 512-D embeddings directly from OpenAI CLIP ViT-B/32 without fine-tuning.
"""

from typing import List
import torch
from PIL import Image


class FrozenCLIPExtractor:
    """Wrapper around pretrained CLIP ViT-B/32 for zero-shot embedding extraction."""
    def __init__(self, model_name: str = "ViT-B/32", pretrained: str = "openai", device: str = "cpu"):
        self.device = device
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        except ImportError:
            print("[WARNING] open_clip_torch not installed. Mocking FrozenCLIPExtractor.")
            self.model = None
            self.preprocess = None
            self.tokenizer = None

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a single PIL image or batch to normalized 512-D feature vector."""
        if self.model is None:
            # Fallback mock embedding
            vec = torch.randn(1, 512)
            return vec / vec.norm(dim=-1, keepdim=True)
            
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu()

    @torch.no_grad()
    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """Encode category prompt texts (e.g. 'a photo of a [class]')."""
        if self.model is None:
            vec = torch.randn(len(text_list), 512)
            return vec / vec.norm(dim=-1, keepdim=True)
            
        tokens = self.tokenizer(text_list).to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu()
