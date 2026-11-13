"""Metrics, threshold selection and plots.

Score contract (shared by EVERY model, including baselines):
    outputs/<dataset>/scores/<model>_seed<k>.npz  with arrays  s_val, s_test
    Higher score == more anomalous.  Labels come from data/processed/<dataset>/seed<k>.npz.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from .utils import get_logger

log = get_logger("evaluate")


def sanity_check_direction(y: np.ndarray, s: np.ndarray, name: str = "") -> None:
    """Catch the classic bug: scores where HIGH means NORMAL."""
    if y.sum() == 0:
        return
    if np.mean(s[y == 1]) < np.mean(s[y == 0]):
        log.warning("[%s] mean score of frauds < mean score of normals - is the sign flipped?", name)


def choose_threshold(y_val, s_val, mode: str = "f1", target_recall: float = 0.8) -> float:
    """Pick the decision threshold on VALIDATION data only."""
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    if mode == "recall":
        ok = np.where(rec >= target_recall)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])  # highest threshold still reaching recall
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def metrics_at(y, s, thr) -> dict:
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(thr),
    )


def evaluate_scores(y_val, s_val, y_test, s_test, mode="f1", target_recall=0.8, name="") -> dict:
    sanity_check_direction(y_val, s_val, name)
    thr = choose_threshold(y_val, s_val, mode, target_recall)
    out = dict(
        pr_auc=float(average_precision_score(y_test, s_test)),
        roc_auc=float(roc_auc_score(y_test, s_test)),
        pr_auc_val=float(average_precision_score(y_val, s_val)),
        random_pr_auc=float(np.mean(y_test)),  # PR-AUC of a random scorer == fraud rate
    )
    out.update(metrics_at(y_test, s_test, thr))
    return out


# ----------------------------------------------------------------------------- plots
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pr_curves(curves: dict, path, title="Precision-Recall (test)") -> None:
    """curves: {model: (y_true, scores)}"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    base = None
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, lw=1.8, label=f"{name} (AP={average_precision_score(y, s):.3f})")
        base = float(np.mean(y))
    if base is not None:
        ax.axhline(base, ls="--", c="gray", lw=1, label=f"random ({base:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_score_hist(y, s, path, title="Anomaly score distribution") -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    lo, hi = np.percentile(s, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(np.clip(s[y == 0], lo, hi), bins=bins, alpha=0.6, density=True, label="normal")
    ax.hist(np.clip(s[y == 1], lo, hi), bins=bins, alpha=0.6, density=True, label="fraud")
    ax.set(xlabel="score", ylabel="density", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(m: dict, path, title="Confusion matrix (test)") -> None:
    plt = _plt()
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred normal", "pred fraud"],
           yticklabels=["true normal", "true fraud"], title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
