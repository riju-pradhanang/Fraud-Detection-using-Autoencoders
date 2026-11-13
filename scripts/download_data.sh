#!/usr/bin/env bash
# Download both datasets with the Kaggle CLI.
# Prereqs: pip install kaggle ; put your API token at ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create Token)
#          For IEEE-CIS you must first click "Join Competition / accept rules" on the competition page.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> ULB creditcardfraud"
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip      # -> data/raw/creditcard.csv

echo ">> IEEE-CIS (needs accepted competition rules)"
mkdir -p data/raw/ieee
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee
(cd data/raw/ieee && unzip -o ieee-fraud-detection.zip train_transaction.csv train_identity.csv)

ls -lh data/raw data/raw/ieee
