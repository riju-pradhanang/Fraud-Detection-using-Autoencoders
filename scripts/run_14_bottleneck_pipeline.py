"""
Script implementing the exact step-by-step workflow with:
1. 14-bottleneck Autoencoder (29 -> 20 -> 14 -> 20 -> 29)
2. Comparison with bottlenecks 8 and 20 (as requested in Step 4)
3. Time-based chronological split (Step 2)
4. RobustScaler fit on train only (Step 3)
5. Decision threshold at target recall = 0.80 on validation (Step 7)
6. Baselines: Isolation Forest, One-Class SVM, and supervised XGBoost (Step 9)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

out_dir = "outputs/bottleneck_14_experiment"
fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

print("="*70)
print("STEP 1: Load and inspect data")
print("="*70)
df = pd.read_csv("data/raw/creditcard.csv")
print("Class counts:")
print(df.Class.value_counts())
print("Missing values:", df.isna().sum().sum())

print("\n" + "="*70)
print("STEP 2: Time-based chronological split (60% Train, 20% Val, 20% Test)")
print("="*70)
df = df.sort_values("Time").reset_index(drop=True)
n = len(df)
train = df.iloc[:int(0.6 * n)].copy()
val   = df.iloc[int(0.6 * n):int(0.8 * n)].copy()
test  = df.iloc[int(0.8 * n):].copy()

for name, d in [("Train", train), ("Validation", val), ("Test", test)]:
    print(f"{name:10s} rows: {len(d):7d} | Frauds: {d.Class.sum():3d} ({d.Class.mean()*100:.4f}%)")

print("\n" + "="*70)
print("STEP 3: RobustScaler (fit on training data ONLY)")
print("="*70)
feats = [f"V{i}" for i in range(1, 29)] + ["Amount"]
# Log1p on Amount to handle heavy skew
train["Amount"] = np.log1p(train["Amount"].clip(lower=0))
val["Amount"]   = np.log1p(val["Amount"].clip(lower=0))
test["Amount"]  = np.log1p(test["Amount"].clip(lower=0))

scaler = RobustScaler().fit(train[feats])
X_train_full = scaler.transform(train[feats]).astype(np.float32)
y_train_full = train.Class.values

X_train = X_train_full[y_train_full == 0] # Normal only for unsupervised models
X_val   = scaler.transform(val[feats]).astype(np.float32)
y_val   = val.Class.values
X_test  = scaler.transform(test[feats]).astype(np.float32)
y_test  = test.Class.values

print(f"X_train (normals only): {X_train.shape}")
print(f"X_train_full (for XGBoost): {X_train_full.shape} with {y_train_full.sum()} frauds")
print(f"X_val:                  {X_val.shape} with {y_val.sum()} frauds")
print(f"X_test:                 {X_test.shape} with {y_test.sum()} frauds")

# Evaluation helper
def evaluate_model(y_true_val, s_val, y_true_test, s_test, target_recall=0.80):
    prec, rec, thr = precision_recall_curve(y_true_val, s_val)
    ok = np.where(rec[:-1] >= target_recall)[0]
    best_idx = ok[np.argmax(prec[:-1][ok])] if len(ok) > 0 else 0
    threshold = thr[best_idx]
    
    pred_test = (s_test >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true_test, pred_test, average="binary", zero_division=0)
    cm = confusion_matrix(y_true_test, pred_test)
    pr_auc = average_precision_score(y_true_test, s_test)
    roc_auc = roc_auc_score(y_true_test, s_test)
    
    return {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Precision": p,
        "Recall": r,
        "F1": f,
        "Threshold": threshold,
        "CM": cm
    }

# PyTorch Autoencoder definition matching 29 -> 20 -> bottleneck -> 20 -> 29
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
            nn.Linear(20, 29) # Linear output
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(bottleneck_dim, epochs=100, batch_size=256, patience=5):
    print(f"\n--- Training Autoencoder with Bottleneck = {bottleneck_dim} (29 -> 20 -> {bottleneck_dim} -> 20 -> 29) ---")
    model = Autoencoder(bottleneck_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 10% validation split on training normals for early stopping
    val_size = int(len(X_train) * 0.1)
    train_data = torch.tensor(X_train[:-val_size], dtype=torch.float32)
    val_data   = torch.tensor(X_train[-val_size:], dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(train_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0
    train_losses, val_losses = [], []
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(x)
        ep_loss /= len(train_data)
        
        # Eval holdout
        model.eval()
        with torch.no_grad():
            v_recon = model(val_data)
            v_loss = criterion(v_recon, val_data).item()
            
        train_losses.append(ep_loss)
        val_losses.append(v_loss)
        
        if v_loss < best_val_loss - 1e-5:
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep+1} | Train Loss: {ep_loss:.4f} | Val Loss: {v_loss:.4f}")
                break
                
    model.load_state_dict(best_weights)
    print(f"Trained in {time.time()-t0:.1f}s | Best Holdout Val MSE: {best_val_loss:.4f}")
    
    # Score val and test by reconstruction error
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        s_val = ((X_val_t - model(X_val_t))**2).mean(dim=1).numpy()
        s_test = ((X_test_t - model(X_test_t))**2).mean(dim=1).numpy()
        
    return s_val, s_test, train_losses, val_losses

results = {}

# 1. Main model: 14-bottleneck Autoencoder
s_val_14, s_test_14, tr_loss_14, val_loss_14 = train_autoencoder(14)
results["Autoencoder (Bottleneck 14) ⭐"] = {
    "uses_labels": "No",
    **evaluate_model(y_val, s_val_14, y_test, s_test_14, target_recall=0.80),
    "s_test": s_test_14
}

# 2. Ablations: Bottleneck 8 and Bottleneck 20 (Step 4 request)
s_val_8, s_test_8, _, _ = train_autoencoder(8)
results["Autoencoder (Bottleneck 8)"] = {
    "uses_labels": "No",
    **evaluate_model(y_val, s_val_8, y_test, s_test_8, target_recall=0.80),
    "s_test": s_test_8
}

s_val_20, s_test_20, _, _ = train_autoencoder(20)
results["Autoencoder (Bottleneck 20)"] = {
    "uses_labels": "No",
    **evaluate_model(y_val, s_val_20, y_test, s_test_20, target_recall=0.80),
    "s_test": s_test_20
}

# 3. Baselines: Isolation Forest
print("\n--- Running Isolation Forest ---")
t0 = time.time()
iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train)
iso_s_val = -iso.score_samples(X_val) # Negate so higher = fraud
iso_s_test = -iso.score_samples(X_test)
print(f"Isolation Forest done in {time.time()-t0:.1f}s")
results["Isolation Forest"] = {
    "uses_labels": "No",
    **evaluate_model(y_val, iso_s_val, y_test, iso_s_test, target_recall=0.80),
    "s_test": iso_s_test
}

# 4. Baselines: One-Class SVM (subsample 20k rows as in step 9)
print("\n--- Running One-Class SVM ---")
t0 = time.time()
idx = np.random.RandomState(42).choice(len(X_train), 20000, replace=False)
ocsvm = OneClassSVM(nu=0.01, gamma="scale").fit(X_train[idx])
oc_s_val = -ocsvm.decision_function(X_val)
oc_s_test = -ocsvm.decision_function(X_test)
print(f"One-Class SVM done in {time.time()-t0:.1f}s")
results["One-Class SVM"] = {
    "uses_labels": "No",
    **evaluate_model(y_val, oc_s_val, y_test, oc_s_test, target_recall=0.80),
    "s_test": oc_s_test
}

# 5. Baselines: Supervised XGBoost (uses labels)
print("\n--- Running XGBoost (Supervised Benchmark) ---")
t0 = time.time()
scale_pos_weight = float((y_train_full == 0).sum() / (y_train_full == 1).sum())
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
).fit(X_train_full, y_train_full)
xgb_s_val = xgb.predict_proba(X_val)[:, 1]
xgb_s_test = xgb.predict_proba(X_test)[:, 1]
print(f"XGBoost done in {time.time()-t0:.1f}s")
results["XGBoost (Supervised)"] = {
    "uses_labels": "Yes",
    **evaluate_model(y_val, xgb_s_val, y_test, xgb_s_test, target_recall=0.80),
    "s_test": xgb_s_test
}

# Build summary table
print("\n" + "="*85)
print("FINAL RESULTS TABLE (Time-split, Target Recall = 80% on Validation)")
print("="*85)
table_data = []
for name, m in results.items():
    table_data.append({
        "Model": name,
        "Uses fraud labels?": m["uses_labels"],
        "PR-AUC": f"{m['PR-AUC']:.4f}",
        "ROC-AUC": f"{m['ROC-AUC']:.4f}",
        "Precision": f"{m['Precision']:.4f}",
        "Recall": f"{m['Recall']:.4f}",
        "F1": f"{m['F1']:.4f}"
    })
res_df = pd.DataFrame(table_data)
print(res_df.to_string(index=False))
res_df.to_csv(os.path.join(out_dir, "benchmark_table.csv"), index=False)

# Plot Loss Curve for 14-bottleneck AE
plt.figure(figsize=(6, 4))
plt.plot(tr_loss_14, label="Train Loss (MSE)")
plt.plot(val_loss_14, label="Holdout Val Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("14-Bottleneck Autoencoder: Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "ae14_loss_curve.png"), dpi=160)
plt.close()

# Plot PR Curves
plt.figure(figsize=(7, 5))
for name, m in results.items():
    p, r, _ = precision_recall_curve(y_test, m["s_test"])
    plt.plot(r, p, label=f"{name} (PR-AUC={m['PR-AUC']:.3f})")
random_floor = float(np.mean(y_test))
plt.axhline(random_floor, ls="--", color="gray", label=f"Random floor ({random_floor:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves on Test Set (Time Split)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "pr_curves_comparison.png"), dpi=160)
plt.close()

# Plot Anomaly Score Histogram for 14-bottleneck
plt.figure(figsize=(6.5, 4))
s_norm = s_test_14[y_test == 0]
s_fraud = s_test_14[y_test == 1]
# Log10 scale for reconstruction error
plt.hist(np.log10(s_norm + 1e-6), bins=60, density=True, alpha=0.6, label="Normal")
plt.hist(np.log10(s_fraud + 1e-6), bins=60, density=True, alpha=0.6, label="Fraud")
plt.xlabel("log10(Reconstruction Error)")
plt.ylabel("Density")
plt.title("14-Bottleneck AE: Reconstruction Error Distribution (Test)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "ae14_score_histogram.png"), dpi=160)
plt.close()

print(f"\nAll plots and results saved to {out_dir}")
