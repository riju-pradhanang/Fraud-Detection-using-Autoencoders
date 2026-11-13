"""Train AE / RDA for every seed and write score files.

Examples
  python -m src.train --model ae                       # AE trained on normals only
  python -m src.train --model ae_all                   # AE trained on the contaminated training set
  python -m src.train --model rda                      # robust AE (+ rda_sparse scores when norm: l1)
  python -m src.train --model rda --sweep-lam          # pick lambda / quantile on validation PR-AUC (first seed only)
  python -m src.train --model ae --hidden 20 10 2 --tag b2   # bottleneck ablation -> model name ae_b2
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import load_split
from .models_ae import fit_ae, fit_rda, get_device, rda_scores, recon_error
from .utils import ROOT, ensure_dir, get_logger, load_config, scores_path, set_seed

log = get_logger("train")


def _save(ds, name, seed, s_val, s_test):
    np.savez(ensure_dir(scores_path(ds, name, seed)), s_val=s_val, s_test=s_test)


def _log_json(ds, name, seed, info):
    p = ensure_dir(ROOT / "outputs" / ds / "logs" / f"{name}_seed{seed}.json")
    p.write_text(json.dumps(info, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", choices=["ae", "ae_all", "rda"], required=True)
    ap.add_argument("--hidden", nargs="+", type=int, default=None, help="override encoder widths")
    ap.add_argument("--lam", type=float, default=None, help="override RDA lambda")
    ap.add_argument("--lam-quantile", type=float, default=None, help="override RDA lambda-quantile")
    ap.add_argument("--tag", default="", help="suffix for the model name (ablations)")
    ap.add_argument("--sweep-lam", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = args.dataset or cfg["dataset"]
    seeds = args.seeds or cfg["seeds"]
    device = get_device()
    suffix = f"_{args.tag}" if args.tag else ""
    log.info("device=%s dataset=%s model=%s seeds=%s", device, ds, args.model, seeds)

    if args.model in ("ae", "ae_all"):
        c = dict(cfg["ae"])
        if args.hidden:
            c["hidden"] = args.hidden
        for seed in seeds:
            set_seed(seed)
            sp = load_split(ds, seed)
            X = sp["X_train"] if args.model == "ae" else sp["X_train_all"]
            model, info = fit_ae(X, c, seed, device)
            _save(ds, args.model + suffix, seed, recon_error(model, sp["X_val"], device),
                  recon_error(model, sp["X_test"], device))
            _log_json(ds, args.model + suffix, seed, info)
            log.info("%s seed %d | epochs=%d best_holdout_mse=%.4f | %.0fs", args.model, seed,
                     info["epochs_run"], info["best_val_loss"], info["seconds"])
        return

    # ---- RDA
    c = dict(cfg["rda"])
    if args.hidden:
        c["hidden"] = args.hidden
    if args.lam is not None:
        c["lam"], c["lam_quantile"] = args.lam, None
    if args.lam_quantile is not None:
        c["lam_quantile"] = args.lam_quantile

    if args.sweep_lam:
        rows = []
        seed = seeds[0]
        sp = load_split(ds, seed)
        grid = [("lam_quantile", q) for q in c["lam_quantile_grid"]] if c.get("lam_quantile_grid") \
            else [("lam", v) for v in c["lam_grid"]]
        for key, val in grid:
            set_seed(seed)
            cc = {**c, "lam_quantile": None, "lam": None}
            cc[key] = val
            model, info = fit_rda(sp["X_train_all"], cc, seed, device)
            sc = rda_scores(model, sp["X_val"], cc, info["lam"], device)
            row = dict(setting=key, value=val, lambda_used=info["lam"], S_row_frac=info["S_row_frac"],
                       val_pr_auc_recon=average_precision_score(sp["y_val"], sc["recon"]),
                       val_pr_auc_sparse=average_precision_score(sp["y_val"], sc["sparse"]))
            rows.append(row)
            log.info("sweep: %s", row)
        df = pd.DataFrame(rows)
        out = ensure_dir(ROOT / "outputs" / ds / f"rda_lambda_sweep{suffix}.csv")
        df.to_csv(out, index=False)
        log.info("saved %s\n%s", out, df.to_string(index=False))
        return

    for seed in seeds:
        set_seed(seed)
        sp = load_split(ds, seed)
        model, info = fit_rda(sp["X_train_all"], c, seed, device)
        sv = rda_scores(model, sp["X_val"], c, info["lam"], device)
        st = rda_scores(model, sp["X_test"], c, info["lam"], device)
        _save(ds, "rda" + suffix, seed, sv["recon"], st["recon"])
        if c.get("norm", "l21") == "l1":  # for l21 the sparse score is a monotone transform of recon
            _save(ds, "rda_sparse" + suffix, seed, sv["sparse"], st["sparse"])
        _log_json(ds, "rda" + suffix, seed, info)
        log.info("rda seed %d done in %.0fs (S rows=%.2f%%)", seed, info["seconds"], 100 * info["S_row_frac"])


if __name__ == "__main__":
    main()
