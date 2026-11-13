"""Vanilla autoencoder and Robust Deep Autoencoder (RDA).

RDA (Zhou & Paffenroth, KDD 2017) splits the data as  X = L_D + S  where L_D is what a deep
autoencoder can reconstruct and S is a sparse matrix that soaks up anomalies/noise:

        min_{theta, S}  || L_D - D(E(L_D)) ||_2  +  lambda * ||S||_{1 or 2,1}      s.t.  X - L_D - S = 0

We solve it by alternating optimisation (the paper uses an ADMM / Bregman-style splitting;
this is the simplified proximal-alternating variant, state this in the report):

    repeat:  L_D <- X - S ;  train AE on L_D (a few epochs) ;  S <- shrink(X - AE(L_D), lambda)

Differences from the paper that you should mention in the report:
  * mini-batch Adam instead of full-batch training,
  * single proximal step per round (no dual variable),
  * for unseen data (val/test) S is unknown, so we score with the residual of the trained AE.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .utils import get_logger

log = get_logger("models")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AE(nn.Module):
    """Symmetric MLP autoencoder. hidden=[20,10,5] -> d-20-10-5-10-20-d (linear bottleneck/output)."""

    def __init__(self, d: int, hidden=(20, 10, 5)):
        super().__init__()
        dims = [d, *hidden]
        enc = []
        for i in range(len(dims) - 1):
            enc.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc.append(nn.ReLU())
        rdims = dims[::-1]
        dec = []
        for i in range(len(rdims) - 1):
            dec.append(nn.Linear(rdims[i], rdims[i + 1]))
            if i < len(rdims) - 2:
                dec.append(nn.ReLU())
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ------------------------------------------------------------------ helpers
def _to_t(X, device):
    return torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)


def _train_epochs(model, X: torch.Tensor, opt, epochs: int, bs: int) -> float:
    model.train()
    n, last = len(X), 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=X.device)
        tot = 0.0
        for i in range(0, n, bs):
            xb = X[perm[i:i + bs]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        last = tot / n
    return last


@torch.no_grad()
def residuals(model, X, device=None, bs: int = 16384) -> np.ndarray:
    """R = X - AE(X) as a numpy array (n, d)."""
    device = device or next(model.parameters()).device
    model.eval()
    Xt = _to_t(X, device)
    out = [(Xt[i:i + bs] - model(Xt[i:i + bs])).cpu() for i in range(0, len(Xt), bs)]
    return torch.cat(out).numpy()


def recon_error(model, X, device=None) -> np.ndarray:
    """Per-sample mean squared reconstruction error (the standard AE anomaly score)."""
    return (residuals(model, X, device) ** 2).mean(1)


# ------------------------------------------------------------------ vanilla AE
def fit_ae(X_train, cfg: dict, seed: int, device=None):
    """Train on X_train with early stopping on a held-out slice of the SAME (unlabeled) data."""
    device = device or get_device()
    rng = np.random.default_rng(seed)
    n = len(X_train)
    perm = rng.permutation(n)
    n_hold = max(int(cfg.get("holdout_frac", 0.1) * n), 1)
    hold, tr = X_train[perm[:n_hold]], X_train[perm[n_hold:]]

    model = AE(X_train.shape[1], cfg["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    Xt, best, best_state, bad, hist = _to_t(tr, device), np.inf, None, 0, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        tr_loss = _train_epochs(model, Xt, opt, 1, cfg["batch_size"])
        val_loss = float(recon_error(model, hold, device).mean())
        hist.append((ep, tr_loss, val_loss))
        if val_loss < best - 1e-6:
            best, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get("patience", 8):
                break
    model.load_state_dict(best_state)
    return model, dict(history=hist, best_val_loss=best, epochs_run=len(hist), seconds=time.time() - t0)


# ------------------------------------------------------------------ RDA
def shrink_l1(R: np.ndarray, lam: float) -> np.ndarray:
    """Element-wise soft threshold: prox of lam*||.||_1."""
    return np.sign(R) * np.maximum(np.abs(R) - lam, 0.0)


def shrink_l21(R: np.ndarray, lam: float) -> np.ndarray:
    """Row-wise group soft threshold: prox of lam*||.||_{2,1}. Zeroes rows with ||r_i|| <= lam."""
    nrm = np.linalg.norm(R, axis=1, keepdims=True)
    return R * np.maximum(1.0 - lam / np.maximum(nrm, 1e-12), 0.0)


def shrink(R, lam, norm="l21"):
    return shrink_l21(R, lam) if norm == "l21" else shrink_l1(R, lam)


def _lambda_for_round(R, cfg) -> float:
    q = cfg.get("lam_quantile")
    if q is None:
        return float(cfg["lam"])
    vals = np.linalg.norm(R, axis=1) if cfg.get("norm", "l21") == "l21" else np.abs(R).ravel()
    return float(np.quantile(vals, q))


def fit_rda(X_all, cfg: dict, seed: int, device=None):
    """Alternating optimisation on UNLABELED (possibly contaminated) data X_all.

    Returns (model, info). info['lam'] is the lambda from the last round (used to score new data).
    """
    device = device or get_device()
    X_all = np.asarray(X_all, dtype=np.float32)
    n, d = X_all.shape
    model = AE(d, cfg["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    S = np.zeros_like(X_all)
    hist, t0, lam = [], time.time(), float(cfg.get("lam") or 0.5)
    for it in range(cfg["outer_iters"]):
        L = _to_t(X_all - S, device)                                   # L_D <- X - S
        loss = _train_epochs(model, L, opt, cfg["inner_epochs"], cfg["batch_size"])
        R = S + residuals(model, X_all - S, device)                    # X - AE(L_D) = S + (L_D - AE(L_D))
        lam = _lambda_for_round(R, cfg)
        S = shrink(R, lam, cfg.get("norm", "l21")).astype(np.float32)  # S <- shrink(X - AE(L_D))
        nz = float((np.linalg.norm(S, axis=1) > 0).mean())
        hist.append((it, loss, lam, nz))
        log.info("RDA round %2d/%d | ae_loss=%.4f | lambda=%.4f | rows in S=%.2f%%",
                 it + 1, cfg["outer_iters"], loss, lam, 100 * nz)
    return model, dict(history=hist, lam=lam, seconds=time.time() - t0, S_row_frac=hist[-1][3])


def rda_scores(model, X, cfg: dict, lam: float, device=None) -> dict:
    """Two scores for unseen data:
       'recon'  - squared residual of the trained AE (same scoring as vanilla AE),
       'sparse' - norm of the shrunk residual ||shrink(x - AE(x))|| (tiny recon term breaks ties)."""
    R = residuals(model, X, device)
    recon = (R ** 2).mean(1)
    sparse = np.linalg.norm(shrink(R, lam, cfg.get("norm", "l21")), axis=1) + 1e-9 * recon
    return dict(recon=recon, sparse=sparse)
