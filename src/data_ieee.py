"""IEEE-CIS preprocessing (Day 3). Produces the SAME npz contract as src/data.py, so
src.run_baselines / src.train / src.report work unchanged:  --config configs/ieee.yaml

Needs data/raw/ieee/train_transaction.csv and train_identity.csv (accept the competition rules on
Kaggle first). The competition test set has no labels, so everything is split from the train files,
by time (TransactionDT).

Design choices (state them in the report):
  * time-based split train/val/test = first 60% / next 20% / last 20% of TransactionDT
  * columns with >max_missing missing rate (measured on the training part) are dropped
  * categoricals -> frequency encoding (train frequencies); numerics -> median impute (train normals)
  * missing-indicators for a handful of partially-missing columns (missingness is informative here)
  * features capped to the `max_features` least-missing columns so the AE stays small and fast
  * RobustScaler fit on train normals, then clipped to [-clip, clip] (heavy tails)

Status: written and unit-tested on a mock frame ONLY - run it on the real files and inspect.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .data import save_split, summarize
from .utils import ROOT, ensure_dir, get_logger, load_config

log = get_logger("data_ieee")
DROP = ["TransactionID", "isFraud", "TransactionDT"]


def preprocess_ieee(df: pd.DataFrame, seed: int, cfg: dict) -> dict:
    d = cfg["data"]
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    y = df["isFraud"].to_numpy().astype(int)
    n = len(df)
    a, b = int(n * d["fractions"][0]), int(n * (d["fractions"][0] + d["fractions"][1]))
    tr, va, te = np.arange(a), np.arange(a, b), np.arange(b, n)

    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    if "TransactionAmt" in X:
        X["TransactionAmt"] = np.log1p(X["TransactionAmt"].clip(lower=0))
    miss = X.iloc[tr].isna().mean()
    keep = miss[miss <= d["max_missing"]].index.tolist()
    X = X[keep]

    # frequency-encode categoricals using TRAIN frequencies
    for c in X.select_dtypes(exclude="number").columns:  # object (pandas<3) or str dtype (pandas>=3)
        freq = X.iloc[tr][c].value_counts(normalize=True)
        X[c] = X[c].astype(object).map(freq).astype(float)  # unseen / NaN -> NaN -> imputed below
    X = X.astype(float)

    # keep least-missing, non-constant columns
    miss = X.iloc[tr].isna().mean()
    nonconst = X.iloc[tr].nunique() > 1
    cols = miss[nonconst].sort_values(kind="stable").index[: d["max_features"]].tolist()
    X = X[cols]

    # missing indicators for partially-missing columns (cap), before imputation
    part = [c for c in cols if 0.05 < miss[c] <= d["max_missing"]][: d.get("n_missing_indicators", 20)]
    ind = X[part].isna().astype(float).add_suffix("_isna")

    normal_tr = tr[y[tr] == 0]
    med = X.iloc[normal_tr].median()
    X = pd.concat([X.fillna(med).fillna(0.0), ind], axis=1)

    scaler = RobustScaler().fit(X.iloc[normal_tr])

    def T(i):
        Z = scaler.transform(X.iloc[i])
        return np.clip(Z, -d["clip"], d["clip"]).astype(np.float32)

    rng = np.random.default_rng(seed)
    cap = d.get("train_normal_max")
    sub = normal_tr if not cap or len(normal_tr) <= cap else rng.choice(normal_tr, cap, replace=False)
    return dict(X_train=T(np.sort(sub)), X_train_all=T(tr), y_train_all=y[tr],
                X_val=T(va), y_val=y[va], X_test=T(te), y_test=y[te],
                feature_names=np.array(list(X.columns), dtype="U"))


def load_ieee(raw_dir: str) -> pd.DataFrame:
    p = ROOT / raw_dir
    tr = pd.read_csv(p / "train_transaction.csv")
    idn = pd.read_csv(p / "train_identity.csv")
    return tr.merge(idn, on="TransactionID", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ieee.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = load_ieee(cfg["raw_dir"])
    log.info("loaded %d rows, fraud rate %.4f", len(df), df.isFraud.mean())
    from .utils import processed_path  # noqa: F401
    for seed in cfg["seeds"]:
        sp = preprocess_ieee(df, seed, cfg)
        save_split(cfg["dataset"], seed, sp)
        log.info("seed %d: %s", seed, summarize(sp))


if __name__ == "__main__":
    main()
