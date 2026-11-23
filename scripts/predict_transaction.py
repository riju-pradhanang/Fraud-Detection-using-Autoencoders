"""
Interactive CLI for Transaction Analysis using the trained 14-Bottleneck Autoencoder.
Usage:
    python scripts/predict_transaction.py --demo normal
    python scripts/predict_transaction.py --demo fraud
    python scripts/predict_transaction.py --amount 1500 --demo unusual
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

MODEL_DIR = "outputs/saved_model"

class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dim=14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(29, 20),
            nn.ReLU(),
            nn.Linear(20, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 29)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_system():
    with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    model = Autoencoder(14)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ae14_model.pt"), weights_only=True))
    model.eval()
    return model, scaler, meta["threshold"], meta["feats"]

def analyze_transaction(features_dict, txn_id="TXN_10025"):
    model, scaler, threshold, feats = load_system()
    
    # Build dataframe row
    row = pd.DataFrame([features_dict])
    
    # Log1p on Amount
    amt = float(row.get("Amount", [100.0])[0])
    row["Amount"] = np.log1p(max(0, amt))
    
    # Scale features
    scaled = scaler.transform(row[feats]).astype(np.float32)
    
    # Forward pass
    with torch.no_grad():
        x_t = torch.tensor(scaled)
        recon_t = model(x_t)
        recon_error = float(((x_t - recon_t) ** 2).mean().item())
        
    is_fraud = recon_error >= threshold
    prediction = "Fraud" if is_fraud else "Normal"
    status_icon = "🚨 FRAUD DETECTED" if is_fraud else "✅ NORMAL TRANSACTION"
    
    print("\n" + "="*50)
    print("      Credit Card Fraud Detection System      ")
    print("           Transaction Analysis Result         ")
    print("="*50)
    print(f"Transaction ID        : {txn_id}")
    print(f"Transaction Amount (₹): {amt:,.2f}")
    print(f"Reconstruction Error  : {recon_error:.4f}")
    print(f"Decision Threshold    : {threshold:.4f}")
    print("-" * 50)
    print(f"Prediction            : {prediction} ({status_icon})")
    print("="*50)
    if is_fraud:
        print(">> Action: Transaction exceeds safety threshold. Flagged for review.")
    else:
        print(">> Action: Transaction pattern matches normal profile. Approved.")
    print("="*50 + "\n")
    
    return {
        "txn_id": txn_id,
        "amount": amt,
        "recon_error": recon_error,
        "threshold": threshold,
        "prediction": prediction
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["normal", "fraud", "unusual"], default="normal")
    parser.add_argument("--id", default="TXN_10025")
    parser.add_argument("--amount", type=float, default=None)
    args = parser.parse_args()
    
    # Load dataset sample as baseline features
    df = pd.read_csv("data/raw/creditcard.csv")
    feats = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    if args.demo == "normal":
        sample = df[df["Class"] == 0].iloc[10][feats].to_dict()
        if args.amount is not None:
            sample["Amount"] = args.amount
        analyze_transaction(sample, txn_id=args.id)
    elif args.demo == "fraud":
        sample = df[df["Class"] == 1].iloc[2][feats].to_dict()
        if args.amount is not None:
            sample["Amount"] = args.amount
        analyze_transaction(sample, txn_id=args.id)
    elif args.demo == "unusual":
        # Create an artificial outlier pattern
        sample = df[df["Class"] == 0].iloc[0][feats].to_dict()
        sample["V1"] = -15.5
        sample["V2"] = 12.3
        sample["V3"] = -18.0
        sample["Amount"] = args.amount if args.amount is not None else 85000.0
        analyze_transaction(sample, txn_id=args.id)
