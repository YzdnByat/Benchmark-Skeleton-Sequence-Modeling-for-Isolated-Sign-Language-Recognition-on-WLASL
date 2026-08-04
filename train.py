# ==========================================
# TRAIN.PY - MAIN ENTRY POINT
# ==========================================
import argparse
import os
import torch

from src.utils import load_config, set_seed, setup_experiment_dir, setup_logger
from src.dataset import build_dataloaders
from src.model import build_model
from src.trainer import train_paper_strategy, train_progressive_strategy
from src.evaluator import evaluate_and_plot_cm


def main():
    parser = argparse.ArgumentParser(description="Sign Language Video Swin Transformer Training")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the active YAML config file")
    args = parser.parse_args()

    # 1. Load active YAML configuration
    config = load_config(args.config)

    # 2. Set global random seed for reproducibility
    set_seed(config.get("seed", 42))

    # 3. Setup experiment output directory & dual terminal logging
    exp_dir = setup_experiment_dir(config)
    setup_logger(exp_dir)

    print("\n" + "=" * 50)
    print(f"🎬 STARTING EXPERIMENT: {config.get('experiment_name', 'default')}")
    print(f"📄 Config File: {args.config}")
    print("=" * 50)

    # 4. Select Hardware Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Running on compute device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")

    # 5. Build DataLoaders (using official WLASL splits)
    train_loader, val_loader, test_loader, config = build_dataloaders(config)

    # 6. Build Model Architecture
    model = build_model(config)
    model = model.to(device)

    # 7. Execute Chosen Training Strategy
    strategy = config.get("training_strategy", "paper_end2end").lower()

    if strategy == "paper_end2end":
        best_weights_path = train_paper_strategy(model, train_loader, val_loader, config, exp_dir, device)
    elif strategy == "progressive":
        best_weights_path = train_progressive_strategy(model, train_loader, val_loader, config, exp_dir, device)
    else:
        raise ValueError(f"❌ Unknown training strategy specified: {strategy}")

    # 8. Load Best Checkpoint Weights & Run Full Test Evaluation with Analytics
    print(f"\n📂 Loading best model weights from: {best_weights_path}")
    model.load_state_dict(torch.load(best_weights_path))

    evaluate_and_plot_cm(model, test_loader, exp_dir, device)

    print("\n" + "=" * 50)
    print(f"🎉 EXPERIMENT COMPLETE! All logs, charts, and model checkpoints are saved in:\n👉 {exp_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()