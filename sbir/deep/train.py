"""Training script skeleton for CLIP + Prompt Learning (E3 and E4).
"""

import argparse
import yaml
import torch


def train(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config['experiment']}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Training loop setup completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/deep_e2_e4.yaml")
    args = parser.parse_args()
    train(args.config)
