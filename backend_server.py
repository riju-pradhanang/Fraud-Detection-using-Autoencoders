"""
FastAPI Server for Real-Time Credit Card Fraud Detection.
Uses the trained 14-Bottleneck Autoencoder and RobustScaler.
Zero dummy or hardcoded predictions.
"""

import os
import json
from io import BytesIO
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- Paths & Globals ---
MODEL_DIR = "outputs/saved_model"
DATA_PATH = "data/raw/creditcard.csv"

# Load Model Architecture (14-Bottleneck Autoencoder)
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

app = FastAPI(title="Credit Card Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
print("Loading model, scaler, and metadata...")
with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
    META = json.load(f)

THRESHOLD = float(META["threshold"])
FEATS = META["feats"] # 29 features: V1-V28, Amount

SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

MODEL = Autoencoder(14)
MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ae14_model.pt"), weights_only=True))
MODEL.eval()

# Preload some sample transactions from dataset for easy testing / verification
print("Loading test samples from dataset...")
DF_RAW = pd.read_csv(DATA_PATH)
DATASET_STATS = {
    "total_transactions": len(DF_RAW),
    "normal_transactions": int((DF_RAW.Class == 0).sum()),
    "fraud_transactions": int((DF_RAW.Class == 1).sum()),
    "fraud_rate": round(float((DF_RAW.Class == 1).mean() * 100), 4),
    "threshold": THRESHOLD,
    "bottleneck_dim": 14,
    "architecture": "29 → 20 → 14 → 20 → 29"
}

# Real samples from test set with their true label for verification
NORMAL_SAMPLES = DF_RAW[DF_RAW.Class == 0].iloc[100:110].to_dict(orient="records")
FRAUD_SAMPLES = DF_RAW[DF_RAW.Class == 1].iloc[10:20].to_dict(orient="records")

# In-memory history for dashboard
HISTORY: List[Dict[str, Any]] = []
MAX_BATCH_BYTES = 300 * 1024 * 1024
MAX_BATCH_ROWS = 500000
MAX_BATCH_INFERENCE_ROWS = 8192

# --- Schemas ---
class TransactionInput(BaseModel):
    transaction_id: str = Field(default="TXN_10025")
    amount: float = Field(..., ge=0, description="Transaction Amount in currency")
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dictionary containing V1 to V28. If omitted, average normal baseline is used."
    )

class QuickTestRequest(BaseModel):
    sample_type: str = Field(..., description="'normal' or 'fraud' to test real transaction from dataset")
    index: int = Field(default=0, ge=0, le=9)
    custom_amount: Optional[float] = None

# Baseline normal vector for missing PCA features
MEDIAN_NORMALS = DF_RAW[DF_RAW.Class == 0][[f"V{i}" for i in range(1, 29)]].median().to_dict()

def predict_single(txn_id: str, amount: float, feat_dict: Dict[str, float]) -> Dict[str, Any]:
    # Ensure all 29 features exist
    row_data = {}
    for i in range(1, 29):
        col = f"V{i}"
        row_data[col] = float(feat_dict.get(col, MEDIAN_NORMALS[col]))
    
    # Log1p on Amount (matching training preprocessing exactly)
    amt_processed = float(np.log1p(max(0.0, amount)))
    row_data["Amount"] = amt_processed

    # Construct DataFrame to maintain column order
    df_row = pd.DataFrame([row_data])[FEATS]

    # Transform using saved RobustScaler
    scaled_array = SCALER.transform(df_row).astype(np.float32)

    # Autoencoder Reconstruction
    with torch.no_grad():
        x_tensor = torch.tensor(scaled_array)
        recon_tensor = MODEL(x_tensor)
        # Mean Squared Error across all 29 dimensions
        mse = float(((x_tensor - recon_tensor) ** 2).mean().item())

    # Decision rule: compares against validation-tuned threshold
    is_fraud = mse >= THRESHOLD
    prediction = "FRAUD" if is_fraud else "NORMAL"
    
    # Simple, clear explanation of why the model reached this decision
    if is_fraud:
        explanation = (
            f"Reconstruction error ({mse:.4f}) exceeded the security threshold ({THRESHOLD:.4f}). "
            f"The combination of transaction amount and PCA behavioral vectors deviated significantly "
            f"from the learned normal distribution."
        )
    else:
        explanation = (
            f"Reconstruction error ({mse:.4f}) is below the threshold ({THRESHOLD:.4f}). "
            f"The transaction pattern was successfully compressed and reconstructed by the Autoencoder, "
            f"matching legitimate purchasing behavior."
        )

    res = {
        "transaction_id": txn_id,
        "amount": round(float(amount), 2),
        "reconstruction_error": round(mse, 4),
        "threshold": round(THRESHOLD, 4),
        "prediction": prediction,
        "is_fraud": is_fraud,
        "error_to_threshold_ratio": round(mse / THRESHOLD, 2),
        "explanation": explanation
    }

    # Save to history
    HISTORY.insert(0, res)
    if len(HISTORY) > 50:
        HISTORY.pop()

    return res

@app.get("/api/stats")
def get_stats():
    return DATASET_STATS

@app.get("/api/samples")
def get_samples():
    """Returns sample transactions from the dataset with real V1-V28 and Amount."""
    def clean(records):
        cleaned = []
        for r in records:
            item = {k: round(float(v), 4) for k, v in r.items() if k != "Class"}
            item["TrueClass"] = int(r["Class"])
            cleaned.append(item)
        return cleaned
    return {
        "normal_samples": clean(NORMAL_SAMPLES),
        "fraud_samples": clean(FRAUD_SAMPLES)
    }

@app.get("/api/history")
def get_history():
    return HISTORY

@app.post("/api/predict")
def predict_transaction(tx: TransactionInput):
    try:
        feats = tx.features if tx.features else {}
        return predict_single(tx.transaction_id, tx.amount, feats)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Score CSV transactions with the same preprocessing and model as single predictions."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    content = await file.read(MAX_BATCH_BYTES + 1)
    if len(content) > MAX_BATCH_BYTES:
        raise HTTPException(status_code=413, detail="CSV file exceeds the 300 MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    try:
        df = pd.read_csv(BytesIO(content))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}") from e

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV must contain at least one transaction.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=400, detail=f"CSV exceeds the {MAX_BATCH_ROWS:,}-row limit.")

    required = ["Amount", *[f"V{i}" for i in range(1, 29)]]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {', '.join(missing)}.",
        )

    try:
        model_frame = df[required].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail="Amount and V1-V28 must contain numeric values.") from e

    values = model_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise HTTPException(status_code=400, detail="Amount and V1-V28 cannot contain blank or non-finite values.")
    if (model_frame["Amount"] < 0).any():
        raise HTTPException(status_code=400, detail="Amount values must be non-negative.")

    model_frame["Amount"] = np.log1p(model_frame["Amount"])
    errors = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), MAX_BATCH_INFERENCE_ROWS):
        end = min(start + MAX_BATCH_INFERENCE_ROWS, len(df))
        scaled = SCALER.transform(model_frame.iloc[start:end][FEATS]).astype(np.float32)
        with torch.no_grad():
            x_tensor = torch.from_numpy(scaled)
            errors[start:end] = ((x_tensor - MODEL(x_tensor)) ** 2).mean(dim=1).numpy()

    predictions = errors >= THRESHOLD

    def make_result(position: int) -> Dict[str, Any]:
        row = df.iloc[position]
        txn_id = row.get("TransactionID", row.get("transaction_id", f"TXN_{position + 1}"))
        item = {
            key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
            for key, value in row.to_dict().items()
        }
        item.update({
            "transaction_id": str(txn_id),
            "amount": round(float(row["Amount"]), 2),
            "reconstruction_error": round(float(errors[position]), 4),
            "threshold": round(THRESHOLD, 4),
            "prediction": "FRAUD" if predictions[position] else "NORMAL",
            "is_fraud": bool(predictions[position]),
        })
        return item

    flagged = [make_result(int(position)) for position in np.flatnonzero(predictions)]
    for position in range(len(df) - 1, max(-1, len(df) - 51), -1):
        item = make_result(position)
        HISTORY.insert(0, {
            "transaction_id": item["transaction_id"],
            "amount": item["amount"],
            "reconstruction_error": item["reconstruction_error"],
            "threshold": item["threshold"],
            "prediction": item["prediction"],
            "is_fraud": item["is_fraud"],
        })
    del HISTORY[50:]

    return {
        "total": len(df),
        "fraud_count": len(flagged),
        "normal_count": len(df) - len(flagged),
        "flagged_rows": flagged,
    }

@app.post("/api/predict-sample")
def predict_sample(req: QuickTestRequest):
    """Loads an actual row from creditcard.csv and sends it through the trained Autoencoder."""
    if req.sample_type == "normal":
        row = NORMAL_SAMPLES[req.index % len(NORMAL_SAMPLES)]
        prefix = "NORM"
    else:
        row = FRAUD_SAMPLES[req.index % len(FRAUD_SAMPLES)]
        prefix = "FRD"

    txn_id = f"TXN_{prefix}_{1000 + req.index}"
    amt = req.custom_amount if req.custom_amount is not None else float(row["Amount"])
    feats = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    
    result = predict_single(txn_id, amt, feats)
    result["actual_dataset_label"] = "NORMAL" if row["Class"] == 0 else "FRAUD"
    return result

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_server:app", host="127.0.0.1", port=8000, reload=False)
