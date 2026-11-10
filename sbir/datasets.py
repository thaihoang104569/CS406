"""Dataset loaders for Flickr15k and Sketchy Extended datasets.
Supports loading pairs, gallery photo sets, and query sketch sets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class SBIRDataset(Dataset):
    """General SBIR Dataset loaded from a standardized CSV split file.
    
    Columns expected:
        - item_id: unique identifier
        - filepath: relative or absolute path to image
        - category: class name
        - item_type: 'sketch' or 'photo'
        - split: 'train', 'val', or 'test'
    """
    def __init__(
        self,
        split_csv: str,
        split: str = "train",
        item_type: Optional[str] = None,
        transform = None,
    ):
        self.df = pd.read_csv(split_csv)
        self.df = self.df[self.df["split"] == split]
        if item_type is not None:
            self.df = self.df[self.df["item_type"] == item_type]
            
        self.df = self.df.reset_index(drop=True)
        self.transform = transform
        
        # Build class-to-index mapping
        self.classes = sorted(self.df["category"].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        category = row["category"]
        label = self.class_to_idx[category]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label, category
