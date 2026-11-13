"""Train baselines for every seed and write score files (contract in evaluate.py)."""
from __future__ import annotations

import argparse

import numpy as np

from .baselines import run_iforest, run_ocsvm
from .data import load_split
from .utils import ensure_dir, get_logger, load_config, scores_path, set_seed

log = get_logger("baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--models", nargs="+", default=["iforest", "ocsvm"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = args.dataset or cfg["dataset"]

    for seed in cfg["seeds"]:
        set_seed(seed)
        sp = load_split(ds, seed)
        for name in args.models:
            fn = run_iforest if name == "iforest" else run_ocsvm
            s_val, s_test, sec = fn(sp["X_train"], sp["X_val"], sp["X_test"], seed, cfg["baselines"][name])
            np.savez(ensure_dir(scores_path(ds, name, seed)), s_val=s_val, s_test=s_test)
            log.info("%s seed %d done in %.1fs", name, seed, sec)


if __name__ == "__main__":
    main()
