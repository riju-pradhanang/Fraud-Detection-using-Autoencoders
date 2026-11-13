PY ?= python
CFG ?= configs/default.yaml

.PHONY: setup test smoke data baselines ae ae_all rda sweep report all clean-smoke ieee-data
setup:      ; pip install -r requirements.txt
test:       ; $(PY) -m pytest -q

# --- real ULB experiments (needs data/raw/creditcard.csv) ---
data:       ; $(PY) -m src.data --config $(CFG)
baselines:  ; $(PY) -m src.run_baselines --config $(CFG)
ae:         ; $(PY) -m src.train --config $(CFG) --model ae
ae_all:     ; $(PY) -m src.train --config $(CFG) --model ae_all
rda:        ; $(PY) -m src.train --config $(CFG) --model rda
sweep:      ; $(PY) -m src.train --config $(CFG) --model rda --sweep-lam
report:     ; $(PY) -m src.report --config $(CFG)
all: data baselines ae ae_all rda report

# --- IEEE-CIS (Day 3) ---
ieee-data:  ; $(PY) -m src.data_ieee --config configs/ieee.yaml

# --- 1-minute pipeline check on SYNTHETIC data (numbers are meaningless) ---
smoke:
	$(PY) scripts/make_synthetic_ulb.py
	$(MAKE) CFG=configs/smoke_test.yaml data baselines ae ae_all rda report
clean-smoke: ; rm -rf outputs/ulb_synth data/processed/ulb_synth data/raw/synthetic_creditcard.csv
