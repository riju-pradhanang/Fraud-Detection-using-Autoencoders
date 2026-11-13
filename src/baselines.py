"""Isolation Forest and One-Class SVM baselines (trained on normal transactions only).

Run:  python -m src.run_baselines --config configs/default.yaml
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def run_iforest(X_train, X_val, X_test, seed: int, p: dict):
    t = time.time()
    m = IsolationForest(
        n_estimators=p.get("n_estimators", 300),
        max_samples=p.get("max_samples", 256),
        contamination=p.get("contamination", "auto"),
        random_state=seed, n_jobs=-1,
    ).fit(X_train)
    # score_samples: higher == more normal  ->  negate
    return -m.score_samples(X_val), -m.score_samples(X_test), time.time() - t


def run_ocsvm(X_train, X_val, X_test, seed: int, p: dict):
    t = time.time()
    rng = np.random.default_rng(seed)
    n = min(len(X_train), p.get("max_train", 30000))  # OCSVM is O(n^2): subsample
    sub = X_train[rng.choice(len(X_train), n, replace=False)]
    m = OneClassSVM(kernel=p.get("kernel", "rbf"), nu=p.get("nu", 0.02),
                    gamma=p.get("gamma", "scale")).fit(sub)
    # decision_function: higher == more normal  ->  negate
    return -m.decision_function(X_val), -m.decision_function(X_test), time.time() - t
