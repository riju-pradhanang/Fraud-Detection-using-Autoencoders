"""Fast unit tests (≈30 s on CPU). Run:  pytest -q"""
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from make_synthetic_ulb import make
from src.data import build_split, split_indices
from src.data_ieee import preprocess_ieee
from src.evaluate import choose_threshold, evaluate_scores
from src.models_ae import AE, fit_ae, fit_rda, rda_scores, recon_error, shrink_l1, shrink_l21
from src.utils import load_config


@pytest.fixture(scope="module")
def df():
    return make(n=12000, fraud_rate=0.01, seed=1)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/smoke_test.yaml")


# ---------------------------------------------------------------- shrinkage operators
def test_shrink_l1_values():
    R = np.array([[3.0, -0.5, 0.2], [-2.0, 1.0, 0.0]])
    out = shrink_l1(R, 1.0)
    np.testing.assert_allclose(out, [[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def test_shrink_l21_rowwise():
    R = np.array([[3.0, 4.0], [0.3, 0.4]])  # norms 5 and 0.5
    out = shrink_l21(R, 1.0)
    np.testing.assert_allclose(out[0], [3.0 * 0.8, 4.0 * 0.8])  # scaled by 1 - 1/5
    np.testing.assert_allclose(out[1], [0.0, 0.0])              # below threshold -> zeroed


# ---------------------------------------------------------------- data protocol
def test_split_no_leakage_and_normal_only_train(df, cfg):
    sp = build_split(df, seed=0, cfg=cfg)
    assert sp["y_train_all"].sum() > 0                       # contaminated set really has frauds
    assert len(sp["X_train"]) == int((sp["y_train_all"] == 0).sum())  # X_train == normals only
    n = len(df)
    assert len(sp["y_train_all"]) + len(sp["y_val"]) + len(sp["y_test"]) == n
    assert abs(np.median(sp["X_train"], axis=0)).max() < 1e-4  # RobustScaler fit on train normals
    assert sp["y_val"].sum() > 0 and sp["y_test"].sum() > 0    # stratified: both have frauds


def test_time_split_is_chronological(df):
    tr, va, te = split_indices(df.Class.to_numpy(), df.Time.to_numpy(), 0, (0.6, 0.2, 0.2), "time")
    t = df.Time.to_numpy()
    assert t[tr].max() <= t[va].min() and t[va].max() <= t[te].min()


# ---------------------------------------------------------------- evaluation
def test_evaluate_perfect_and_random():
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    perfect = evaluate_scores(y, y.astype(float), y, y.astype(float))
    assert perfect["pr_auc"] == pytest.approx(1.0) and perfect["f1"] == pytest.approx(1.0)
    rnd = evaluate_scores(y, np.random.default_rng(0).random(1000), y, np.random.default_rng(1).random(1000))
    assert rnd["pr_auc"] < 0.2


def test_inverted_scores_warn(caplog):
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    with caplog.at_level(logging.WARNING):
        evaluate_scores(y, -y.astype(float), y, -y.astype(float), name="flipped")
    assert any("sign flipped" in r.message for r in caplog.records)


def test_threshold_from_validation_only():
    y = np.r_[np.zeros(95, int), np.ones(5, int)]
    s = np.r_[np.linspace(0, 0.5, 95), np.linspace(0.6, 1.0, 5)]
    assert 0.5 < choose_threshold(y, s, "f1") <= 1.0


# ---------------------------------------------------------------- models
def test_ae_trains_and_detects(df, cfg):
    sp = build_split(df, seed=0, cfg=cfg)
    model, info = fit_ae(sp["X_train"], {**cfg["ae"], "epochs": 8}, seed=0)
    assert isinstance(model, AE) and info["epochs_run"] >= 1
    s = recon_error(model, sp["X_test"])
    assert s.shape == (len(sp["X_test"]),) and np.isfinite(s).all()
    assert roc_auc_score(sp["y_test"], s) > 0.8   # synthetic frauds are clearly shifted


def test_rda_absorbs_outliers_and_scores(df, cfg):
    sp = build_split(df, seed=0, cfg=cfg)
    c = {**cfg["rda"], "outer_iters": 4, "inner_epochs": 2, "lam_quantile": 0.98}
    model, info = fit_rda(sp["X_train_all"], c, seed=0)
    assert 0.0 < info["S_row_frac"] <= 0.03          # ~ (1 - quantile) of rows land in S
    sc = rda_scores(model, sp["X_test"], c, info["lam"])
    assert roc_auc_score(sp["y_test"], sc["recon"]) > 0.8


# ---------------------------------------------------------------- IEEE preprocessing (mock frame)
def test_ieee_preprocess_mock():
    rng = np.random.default_rng(0)
    n = 3000
    mock = pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": rng.permutation(n) * 60,
        "isFraud": (rng.random(n) < 0.035).astype(int), "TransactionAmt": rng.lognormal(4, 1, n),
        "ProductCD": rng.choice(list("WCRHS"), n), "card1": rng.integers(1000, 20000, n).astype(float),
        "D1": np.where(rng.random(n) < 0.3, np.nan, rng.random(n)),
        "V1": np.where(rng.random(n) < 0.95, np.nan, rng.random(n)),   # >90% missing -> dropped
        "id_30": np.where(rng.random(n) < 0.6, None, "Windows 10"),
    })
    cfg = {"data": {"fractions": [0.6, 0.2, 0.2], "max_missing": 0.9, "max_features": 50,
                    "n_missing_indicators": 5, "clip": 10.0, "train_normal_max": 1000}}
    sp = preprocess_ieee(mock, seed=0, cfg=cfg)
    assert sp["X_train"].shape[1] == sp["X_test"].shape[1] == len(sp["feature_names"])
    assert "V1" not in sp["feature_names"] and np.isfinite(sp["X_test"]).all()
    assert sp["X_train"].shape[0] <= 1000 and abs(sp["X_train"]).max() <= 10.0
