"""Small shared helpers: seeding, config loading, paths, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the baselines
        pass


def load_config(path: str | os.PathLike = "configs/default.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def processed_path(dataset: str, seed: int) -> Path:
    return ROOT / "data" / "processed" / dataset / f"seed{seed}.npz"


def scores_path(dataset: str, model: str, seed: int) -> Path:
    return ROOT / "outputs" / dataset / "scores" / f"{model}_seed{seed}.npz"


def ensure_dir(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
