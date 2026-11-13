"""Generate a SYNTHETIC dataset with the same schema as Kaggle's creditcard.csv.

Purpose: smoke-test the pipeline / CI before the real data is downloaded.
Numbers produced on this data are NOT results - never put them in the report.

  python scripts/make_synthetic_ulb.py --n 60000 --fraud-rate 0.005 --out data/raw/synthetic_creditcard.csv
"""
import argparse

import numpy as np
import pandas as pd


def make(n=60000, fraud_rate=0.005, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = 28
    scales = np.linspace(2.0, 0.4, d)                       # PCA-like decaying spread
    A = rng.normal(size=(d, d)) * 0.15 + np.eye(d)          # mild correlations
    n_f = int(round(n * fraud_rate))
    n_n = n - n_f
    normal = (rng.normal(size=(n_n, d)) * scales) @ A
    shift = np.zeros(d)
    idx = [0, 1, 2, 3, 9, 11, 13]                           # a few "informative" components
    shift[idx] = rng.choice([-1, 1], len(idx)) * rng.uniform(1.5, 3.5, len(idx)) * scales[idx]
    fraud = (rng.normal(size=(n_f, d)) * scales * 1.4 + shift) @ A
    V = np.vstack([normal, fraud])
    y = np.r_[np.zeros(n_n, int), np.ones(n_f, int)]
    amount = np.r_[rng.lognormal(3.3, 1.3, n_n), rng.lognormal(3.9, 1.6, n_f)]
    t = np.sort(rng.uniform(0, 172792, n))
    perm = rng.permutation(n)                               # shuffle then attach sorted Time
    df = pd.DataFrame(V[perm], columns=[f"V{i}" for i in range(1, d + 1)])
    df.insert(0, "Time", t)
    df["Amount"], df["Class"] = amount[perm].round(2), y[perm]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--fraud-rate", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/synthetic_creditcard.csv")
    a = ap.parse_args()
    df = make(a.n, a.fraud_rate, a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows, {df.Class.sum()} frauds  (SYNTHETIC)")
