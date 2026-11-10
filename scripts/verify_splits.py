"""Verification script to prevent Data Leakage between Train, Validation, and Test splits.
Checks:
1. Category-level disjointness between Seen (Train/Val) and Unseen (Test) sets.
2. Disjointness of Image IDs and Sketch IDs across all splits.
3. Class balance and distribution report.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def verify_split_csv(split_csv_path: str) -> bool:
    """Verifies that the CSV split file has zero data leakage.
    
    Expected CSV columns:
        - item_id: unique identifier of image/sketch
        - split: 'train', 'val', or 'test'
        - category: class name
        - item_type: 'sketch' or 'photo'
    """
    path = Path(split_csv_path)
    if not path.exists():
        print(f"[ERROR] Split file not found: {split_csv_path}")
        return False

    df = pd.read_csv(path)
    required_cols = {"item_id", "split", "category", "item_type"}
    if not required_cols.issubset(set(df.columns)):
        print(f"[ERROR] Missing required columns in {split_csv_path}. Required: {required_cols}")
        return False

    splits = set(df["split"].unique())
    print(f"Loaded {len(df)} records across splits: {splits}")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_classes = set(train_df["category"].unique())
    val_classes = set(val_df["category"].unique())
    test_classes = set(test_df["category"].unique())

    print(f"\n--- Class Distribution ---")
    print(f"Train classes count: {len(train_classes)}")
    print(f"Val classes count:   {len(val_classes)}")
    print(f"Test classes count:  {len(test_classes)}")

    has_error = False

    # 1. Zero-shot check: Train and Test must have disjoint classes
    train_test_overlap = train_classes.intersection(test_classes)
    if train_test_overlap:
        print(f"[LEAKAGE ERROR] Train and Test share classes: {train_test_overlap}")
        has_error = True
    else:
        print("[PASS] Train and Test have completely disjoint classes (0% overlap).")

    # Val and Test must also be disjoint
    val_test_overlap = val_classes.intersection(test_classes)
    if val_test_overlap:
        print(f"[LEAKAGE ERROR] Val and Test share classes: {val_test_overlap}")
        has_error = True
    else:
        print("[PASS] Val and Test have completely disjoint classes (0% overlap).")

    # 2. Item ID check: No single item_id should appear in multiple splits
    train_ids = set(train_df["item_id"])
    val_ids = set(val_df["item_id"])
    test_ids = set(test_df["item_id"])

    id_overlap_train_val = train_ids.intersection(val_ids)
    id_overlap_train_test = train_ids.intersection(test_ids)
    id_overlap_val_test = val_ids.intersection(test_ids)

    if id_overlap_train_val:
        print(f"[LEAKAGE ERROR] Item ID overlap between Train and Val ({len(id_overlap_train_val)} items).")
        has_error = True
    if id_overlap_train_test:
        print(f"[LEAKAGE ERROR] Item ID overlap between Train and Test ({len(id_overlap_train_test)} items).")
        has_error = True
    if id_overlap_val_test:
        print(f"[LEAKAGE ERROR] Item ID overlap between Val and Test ({len(id_overlap_val_test)} items).")
        has_error = True

    if not (id_overlap_train_val or id_overlap_train_test or id_overlap_val_test):
        print("[PASS] No item_id overlap across any splits.")

    if has_error:
        print("\n❌ DATA LEAKAGE DETECTED! Do NOT use this split for experiments.")
        return False
    else:
        print("\n✅ SPLIT VERIFICATION SUCCESSFUL! Zero data leakage confirmed.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Zero-shot splits to prevent data leakage.")
    parser.add_argument("--split_file", type=str, default="data/splits/sketchy_zero_shot_splits.csv",
                        help="Path to CSV containing split metadata.")
    args = parser.parse_args()
    success = verify_split_csv(args.split_file)
    sys.exit(0 if success else 1)
