"""Aggregate all score files -> metrics table (mean +- std over seeds) + figures.

Run:  python -m src.report --dataset ulb
Outputs (outputs/<dataset>/):  results_per_seed.csv, results_table.md, results_table.csv,
                               figures/pr_curves_seed<k>.png, score_hist_<model>.png, confusion_<model>.png
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from .data import load_split
from .evaluate import (evaluate_scores, plot_confusion, plot_pr_curves, plot_score_hist)
from .utils import ROOT, ensure_dir, get_logger, load_config

log = get_logger("report")

ORDER = ["iforest", "ocsvm", "ae", "ae_all", "rda", "rda_sparse"]
PRETTY = {"iforest": "Isolation Forest", "ocsvm": "One-Class SVM", "ae": "AE (normals)",
          "ae_all": "AE (contaminated)", "rda": "RDA (recon score)", "rda_sparse": "RDA (sparse score)"}


def _models(ds: str):
    d = ROOT / "outputs" / ds / "scores"
    names = sorted({re.sub(r"_seed\d+\.npz$", "", f.name) for f in d.glob("*.npz")})
    return sorted(names, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--plot-seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = args.dataset or cfg["dataset"]
    out = ROOT / "outputs" / ds
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ev = cfg["eval"]
    plot_seed = cfg["seeds"][0] if args.plot_seed is None else args.plot_seed

    rows, curves = [], {}
    for name in _models(ds):
        for seed in cfg["seeds"]:
            f = ROOT / "outputs" / ds / "scores" / f"{name}_seed{seed}.npz"
            if not f.exists():
                continue
            sp, z = load_split(ds, seed), np.load(f)
            m = evaluate_scores(sp["y_val"], z["s_val"], sp["y_test"], z["s_test"],
                                ev["threshold"], ev["target_recall"], name)
            rows.append(dict(model=name, seed=seed, **m))
            if seed == plot_seed:
                curves[PRETTY.get(name, name)] = (sp["y_test"], z["s_test"])
                plot_score_hist(sp["y_test"], z["s_test"], fig_dir / f"score_hist_{name}.png",
                                f"{PRETTY.get(name, name)}: test score distribution")
                plot_confusion(m, fig_dir / f"confusion_{name}.png",
                               f"{PRETTY.get(name, name)} (thr from val)")
    if not rows:
        raise SystemExit("no score files found - run the baselines / train scripts first")

    df = pd.DataFrame(rows)
    df.to_csv(ensure_dir(out / "results_per_seed.csv"), index=False)
    plot_pr_curves(curves, fig_dir / f"pr_curves_seed{plot_seed}.png",
                   f"{ds}: precision-recall on test (seed {plot_seed})")

    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    g = df.groupby("model", sort=False)
    mean, std = g[cols].mean(), g[cols].std(ddof=1).fillna(0.0)
    table = pd.DataFrame({c: [f"{mean.loc[m, c]:.3f} ± {std.loc[m, c]:.3f}" for m in mean.index] for c in cols},
                         index=[PRETTY.get(m, m) for m in mean.index])
    table.insert(0, "n_seeds", [int(g.size()[m]) for m in mean.index])
    table.index.name = "model"
    table.to_csv(out / "results_table.csv")
    rnd = df["random_pr_auc"].mean()
    md = table.reset_index().to_markdown(index=False) if hasattr(pd.DataFrame, "to_markdown") else table.to_string()
    (out / "results_table.md").write_text(f"{md}\n\nRandom-scorer PR-AUC (= test fraud rate): {rnd:.4f}\n")
    print(f"\n{ds}: test metrics, mean ± std over seeds (threshold chosen on validation, mode={ev['threshold']})\n")
    print(table.to_string())
    print(f"\nrandom PR-AUC = {rnd:.4f}\nfigures -> {fig_dir}")


if __name__ == "__main__":
    main()
