# ==========================================
# SRC/UTILS.PY - HELPER, LOGGING & METRICS
# ==========================================
import sys
import os
import random
import numpy as np
import torch
import yaml


class Logger(object):
    """
    Duplicates stdout to a log file so all console logs, epoch losses,
    and accuracy prints are saved automatically to training.log.
    """

    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log = open(log_filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logger(exp_dir: str):
    """Directs sys.stdout to log to training.log inside the experiment output directory."""
    log_file = os.path.join(exp_dir, "training.log")
    sys.stdout = Logger(log_file)
    print(f"📝 Logging active. All terminal outputs are saving to: {log_file}")


def load_config(config_path: str) -> dict:
    """Reads and parses a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Configuration file not found at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Sets random seed across all libraries for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Global seed set to: {seed}")


def setup_experiment_dir(config: dict) -> str:
    """
    Creates a dedicated directory for the experiment under output_dir/experiment_name/
    and saves an archival copy of the active YAML config file there.
    """
    base_output_dir = config.get("output_dir", "./outputs")
    exp_name = config.get("experiment_name", "default_exp")
    exp_dir = os.path.join(base_output_dir, exp_name)

    os.makedirs(exp_dir, exist_ok=True)

    # Save a copy of the active YAML config inside the output directory
    saved_config_path = os.path.join(exp_dir, "config_used.yaml")
    with open(saved_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"📂 Experiment directory ready: {exp_dir}")
    return exp_dir


def accuracy_top_k(outputs: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    """Computes the Top-K accuracy, dynamically capping K to available classes."""
    with torch.no_grad():
        num_classes = outputs.size(1)
        # Cap K to the number of output classes so it doesn't crash on small dummy runs
        valid_topk = tuple(min(k, num_classes) for k in topk)
        maxk = max(valid_topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in valid_topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k / batch_size).item())
        return res