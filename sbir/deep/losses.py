"""Loss functions for CLIP + Prompt Learning:
1. Triplet Loss with Cosine Distance:
   L_tri = max(0, margin + dist(s, p+) - dist(s, p-))
2. Semantic Classification Loss via Pretrained Text Encoder:
   L_cls = CrossEntropy(sim(feat, text_anchors) / tau, label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SBIRTripletLoss(nn.Module):
    """Triplet ranking loss on cosine distance."""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_sketch: torch.Tensor,
        positive_photo: torch.Tensor,
        negative_photo: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            anchor_sketch: (B, D) normalized sketch embeddings
            positive_photo: (B, D) normalized positive photo embeddings
            negative_photo: (B, D) normalized negative photo embeddings
        """
        # Cosine distance = 1 - cosine_similarity
        pos_dist = 1.0 - F.cosine_similarity(anchor_sketch, positive_photo, dim=-1)
        neg_dist = 1.0 - F.cosine_similarity(anchor_sketch, negative_photo, dim=-1)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class SemanticClassificationLoss(nn.Module):
    """Classification loss forcing visual embeddings to align with frozen text anchor embeddings."""
    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        target_labels: torch.Tensor,
        text_anchors: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            visual_embeddings: (B, D) normalized image/sketch embeddings
            target_labels: (B,) class indices in C_seen
            text_anchors: (C_seen, D) normalized text embeddings for seen classes
        """
        # Logits = (visual @ text.T) / tau
        logits = torch.matmul(visual_embeddings, text_anchors.T) / self.temperature
        return F.cross_entropy(logits, target_labels)
