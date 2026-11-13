"""Data loading, preprocessing and splitting for the ULB creditcardfraud dataset.

Protocol (important for an *unsupervised* study):
  * Split first, preprocess second. The scaler is fit on the training NORMALS only.
  * Models train on normals only (``X_train``) or, for the robust AE, on the
    unlabeled/contaminated training set (``X_train_all``). Labels are never used for
    training; ``y_val`` is used only for threshold / hyper-parameter selection.
  * ``y_test`` is touched only for the final report.

Run:  python -m src.data --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from .utils import ROOT, ensure_dir, get_logger, load_config, processed_path

log = get_logger("data")


def load_ulb(path: str) -> pd.DataFrame:
    p = ROOT / path if not str(path).startswith("/") else path
    df = pd.read_csv(p)
    expected = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def feature_frame(df: pd.DataFrame, drop_time: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (features, labels). Amount is log1p-transformed (heavy right tail)."""
    X = df.drop(columns=["Class"]).copy()
    X["Amount"] = np.log1p(X["Amount"].clip(lower=0))
    if drop_time:
        X = X.drop(columns=["Time"])
    return X, df["Class"].to_numpy().astype(int)


def split_indices(
    y: np.ndarray, time: np.ndarray | None, seed: int, fractions=(0.6, 0.2, 0.2), mode="stratified"
):
    n = len(y)
    f_tr, f_va, f_te = fractions
    assert abs(f_tr + f_va + f_te - 1) < 1e-8
    idx = np.arange(n)
    if mode == "time":
        if time is None:
            raise ValueError("time split requires a Time column")
        order = np.argsort(time, kind="stable")
        a, b = int(n * f_tr), int(n * (f_tr + f_va))
        return order[:a], order[a:b], order[b:]
    # stratified random split
    tr, rest = train_test_split(idx, train_size=f_tr, stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=f_va / (f_va + f_te), stratify=y[rest], random_state=seed)
    return tr, va, te


def build_split(df: pd.DataFrame, seed: int, cfg: dict) -> dict:
    d = cfg["data"]
    X_df, y = feature_frame(df, drop_time=d["drop_time"])
    time = df["Time"].to_numpy() if "Time" in df else None
    tr, va, te = split_indices(y, time, seed, tuple(d["fractions"]), d["split"])

    X = X_df.to_numpy(dtype=np.float32)
    scaler = RobustScaler() if d["scaler"] == "robust" else StandardScaler()
    normal_tr = tr[y[tr] == 0]
    scaler.fit(X[normal_tr])  # <- fit on training normals only (no leakage)

    def T(i):
        return scaler.transform(X[i]).astype(np.float32)

    return dict(
        X_train=T(normal_tr),            # normals only: used by AE / IF / OCSVM
        X_train_all=T(tr),               # unlabeled mix: used by RDA (and "AE on contaminated")
        y_train_all=y[tr],               # kept ONLY for diagnostics (e.g. contamination rate)
        X_val=T(va), y_val=y[va],
        X_test=T(te), y_test=y[te],
        feature_names=np.array(list(X_df.columns), dtype="U"),
    )


def save_split(dataset: str, seed: int, split: dict) -> None:
    out = ensure_dir(processed_path(dataset, seed))
    np.savez_compressed(out, **split)


def load_split(dataset: str, seed: int) -> dict:
    p = processed_path(dataset, seed)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `python -m src.data` first")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def summarize(split: dict) -> dict:
    s = {}
    for part in ("train_all", "val", "test"):
        y = split[f"y_{part}"]
        s[part] = dict(n=int(len(y)), frauds=int(y.sum()), fraud_rate=float(y.mean()))
    s["train_normal_n"] = int(len(split["X_train"]))
    s["n_features"] = int(split["X_train"].shape[1])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw", default=None, help="override raw_path")
    ap.add_argument("--dataset", default=None, help="override dataset name (output folder)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = args.raw or cfg["raw_path"]
    name = args.dataset or cfg["dataset"]
    df = load_ulb(raw)
    log.info("loaded %s: %d rows, %d frauds (%.4f%%)", raw, len(df), df.Class.sum(), 100 * df.Class.mean())

    for seed in cfg["seeds"]:
        split = build_split(df, seed, cfg)
        save_split(name, seed, split)
        s = summarize(split)
        log.info("seed %d | train_normal=%d | val frauds=%d | test frauds=%d",
                 seed, s["train_normal_n"], s["val"]["frauds"], s["test"]["frauds"])
        if seed == cfg["seeds"][0]:
            meta = ensure_dir(ROOT / "outputs" / name / "split_summary.json")
            meta.write_text(json.dumps({"seed": seed, **s}, indent=2))


if __name__ == "__main__":
    main()
