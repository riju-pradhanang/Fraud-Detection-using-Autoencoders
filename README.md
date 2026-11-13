# Unsupervised Fraud Detection with (Robust) Autoencoders

Group project (3 people, 4 days). Unsupervised anomaly detection on card-transaction data.

* **Datasets:** [ULB creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (main), [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection) (second dataset, Day 3)
* **Base paper:** Zhou & Paffenroth, *Anomaly Detection with Robust Deep Autoencoders*, KDD 2017
* **Models:** vanilla AE, Robust Deep Autoencoder (RDA) vs. Isolation Forest, One-Class SVM
* **Metrics:** PR-AUC (primary), ROC-AUC, precision, recall, F1, PR curves, confusion matrices

> **Status:** Phase 1 (Day 1) is implemented and tested on *synthetic* data. The real datasets still have to be
> downloaded on your machines, see [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md). No real results exist yet.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test                    # 10 unit tests, ~10 s
make smoke                   # full pipeline on SYNTHETIC data, ~1 min (numbers are meaningless)

# real data (needs ~/.kaggle/kaggle.json; accept IEEE-CIS rules on Kaggle first)
bash scripts/download_data.sh          # -> data/raw/creditcard.csv, data/raw/ieee/*.csv
make all                               # splits -> baselines -> AE -> AE(contaminated) -> RDA -> report
```

No Kaggle CLI? Download `creditcard.csv` manually from the dataset page into `data/raw/`.

## Layout

```
configs/default.yaml     ULB experiment settings (seeds, models, thresholds)    configs/ieee.yaml  IEEE-CIS
configs/smoke_test.yaml  tiny settings for the synthetic smoke test
src/data.py              load, log1p(Amount), split, scale (fit on train normals), save .npz
src/data_ieee.py         IEEE-CIS preprocessing (same .npz contract)  [Day 3, tested on a mock frame only]
src/baselines.py         Isolation Forest, One-Class SVM          src/run_baselines.py  runner
src/models_ae.py         AE + RDA (alternating optimisation, L1 / L2,1 shrinkage)
src/train.py             runner: --model ae | ae_all | rda, --sweep-lam, --hidden, --tag
src/evaluate.py          metrics, validation-threshold, plots     src/report.py  tables + figures
tests/                   unit tests          scripts/  data download + synthetic generator
docs/                    ROADMAP.md, NEXT_STEPS.md (task board)     report/  report outline
```

## The protocol (what makes this "unsupervised" and leakage-free)

1. **Split first** (stratified 60/20/20 per seed, or `split: time`), preprocess second; scaler fit on **training normals only**.
2. Models never see labels: AE / IF / OCSVM train on normals only (`X_train`); RDA trains on the **unlabeled mixed**
   training set (`X_train_all`), which is the setting it was designed for. `ae_all` is the matching control
   (a plain AE on the same contaminated data).
3. Validation labels are used **only** to pick the threshold (and λ). **Test** is touched only for reporting.
4. **Score contract:** every model writes `outputs/<dataset>/scores/<model>_seed<k>.npz` with `s_val`, `s_test`
   (higher = more anomalous). `src.report` turns those into the final table, so people can work in parallel.
5. 5 seeds → mean ± std. Random-scorer PR-AUC equals the fraud rate (~0.0017 on ULB), shown for reference.

## Useful commands

```bash
python -m src.train --model rda --sweep-lam                 # λ (quantile) sweep on validation PR-AUC, seed 0
python -m src.train --model rda --lam-quantile 0.99         # fix the chosen setting, all seeds
python -m src.train --model ae --hidden 20 10 2 --tag b2    # bottleneck ablation -> model "ae_b2"
python -m src.report --dataset ulb                          # table + figures in outputs/ulb/
```

## Deviations from the paper (state these in the report)

* Mini-batch Adam instead of full-batch training; one proximal (shrinkage) step per round, no dual variable / ADMM multiplier.
* λ is set as a **quantile** of per-row residual norms each round (scale-free; ≈(1-q) of rows are moved into S).
  Set `lam_quantile: null` to use an absolute `lam` instead.
* The paper's S is defined for the training data only. For unseen val/test rows we score with the trained AE's residual.
* With the row-wise (L2,1) norm the "sparse" score is a monotone transform of reconstruction error, so it has the same
  ranking (same PR-AUC); `rda_sparse` is therefore only written for `norm: l1`.
