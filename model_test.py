"""
Irish Home Retrofit — LightGBM Model Testing
=============================================
Comprehensive evaluation on the held-out 20% test set (264,913 rows).
Covers: regression metrics, BER band accuracy, residual analysis,
        error distribution, and sample predictions.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, confusion_matrix, classification_report
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DATA_PATH   = "46_Col_final_with_county.parquet"
OUTPUT_DIR  = Path("model_analysis_outputs")
TARGET      = "BerRating"
TEST_SIZE   = 0.20
OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)

# ── BER Band mapping (Irish SEAI standard) ────────────────────────────────────
# BerRating is in kWh/m²/yr; bands follow SEAI thresholds
def to_ber_band(rating):
    if   rating <= 25:   return "A1"
    elif rating <= 50:   return "A2"
    elif rating <= 75:   return "A3"
    elif rating <= 100:  return "B1"
    elif rating <= 125:  return "B2"
    elif rating <= 150:  return "B3"
    elif rating <= 175:  return "C1"
    elif rating <= 200:  return "C2"
    elif rating <= 225:  return "C3"
    elif rating <= 260:  return "D1"
    elif rating <= 300:  return "D2"
    elif rating <= 340:  return "E1"
    elif rating <= 380:  return "E2"
    elif rating <= 450:  return "F"
    else:                return "G"

BER_ORDER = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","E1","E2","F","G"]

# ── Load & prepare ────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
DROP_COLS = ["SHRenewableResources", "WHRenewableResources", "HSEffAdjFactor", "WHEffAdjFactor"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

q_lo = df[TARGET].quantile(0.01)
q_hi = df[TARGET].quantile(0.99)
df = df[(df[TARGET] >= q_lo) & (df[TARGET] <= q_hi)]
print(f"  Dataset after trim: {df.shape}")

X = df.drop(columns=[TARGET])
y = df[TARGET].values

cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Train LightGBM (best params) ──────────────────────────────────────────────
print("\nTraining LightGBM...")
tree_pre = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", "passthrough", num_cols),
], remainder="passthrough")

model = Pipeline([
    ("pre", tree_pre),
    ("model", lgb.LGBMRegressor(
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        n_estimators=800, num_leaves=31, learning_rate=0.08,
        max_depth=6, subsample=0.7, colsample_bytree=0.9,
    )),
])
model.fit(X_train, y_train)

# ── Predictions ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, q_lo, q_hi)  # clip to valid range

# ── Regression Metrics ────────────────────────────────────────────────────────
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
bias = np.mean(y_pred - y_test)  # mean prediction bias

print("\n" + "="*55)
print("  REGRESSION METRICS  (Test set: 264,913 rows)")
print("="*55)
print(f"  R²                    : {r2:.4f}")
print(f"  RMSE                  : {rmse:.2f} kWh/m²/yr")
print(f"  MAE                   : {mae:.2f} kWh/m²/yr")
print(f"  MAPE                  : {mape:.2f}%")
print(f"  Mean Bias (pred-actual): {bias:+.2f} kWh/m²/yr")

# Percentile error breakdown
residuals = y_pred - y_test
print(f"\n  Residual percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    {p:3d}th : {np.percentile(np.abs(residuals), p):.1f} kWh/m²/yr")

# ── BER Band accuracy ─────────────────────────────────────────────────────────
y_test_band = pd.Series(y_test).map(to_ber_band)
y_pred_band = pd.Series(y_pred).map(to_ber_band)

exact_acc    = (y_test_band == y_pred_band).mean() * 100
within_1_acc = (
    (y_test_band.map(BER_ORDER.index) - y_pred_band.map(BER_ORDER.index)).abs() <= 1
).mean() * 100
within_2_acc = (
    (y_test_band.map(BER_ORDER.index) - y_pred_band.map(BER_ORDER.index)).abs() <= 2
).mean() * 100

print("\n" + "="*55)
print("  BER BAND ACCURACY  (15 bands: A1 → G)")
print("="*55)
print(f"  Exact band match     : {exact_acc:.1f}%")
print(f"  Within ±1 band       : {within_1_acc:.1f}%")
print(f"  Within ±2 bands      : {within_2_acc:.1f}%")

# Per-band MAE
band_df = pd.DataFrame({"actual": y_test, "predicted": y_pred, "band": y_test_band})
band_mae = band_df.groupby("band").apply(
    lambda g: pd.Series({
        "count": len(g),
        "MAE": mean_absolute_error(g["actual"], g["predicted"]),
        "R²":  r2_score(g["actual"], g["predicted"]) if len(g) > 1 else np.nan,
    })
).reindex([b for b in BER_ORDER if b in band_df["band"].unique()])

print("\n  Per-Band Performance:")
print(f"  {'Band':>4}  {'Count':>8}  {'MAE':>8}  {'R²':>6}")
print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}")
for band, row in band_mae.iterrows():
    print(f"  {band:>4}  {int(row['count']):>8,}  {row['MAE']:>8.2f}  {row['R²']:>6.4f}")

# ── Save metrics to CSV ───────────────────────────────────────────────────────
metrics_df = pd.DataFrame([{
    "R²": round(r2, 4),
    "RMSE": round(rmse, 2),
    "MAE": round(mae, 2),
    "MAPE (%)": round(mape, 2),
    "Mean Bias": round(bias, 2),
    "Exact Band Acc (%)": round(exact_acc, 1),
    "Within-1 Band Acc (%)": round(within_1_acc, 1),
    "Within-2 Band Acc (%)": round(within_2_acc, 1),
}])
metrics_df.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
band_mae.to_csv(OUTPUT_DIR / "test_per_band_metrics.csv")

# ── Sample predictions ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SAMPLE PREDICTIONS  (20 random test rows)")
print("="*55)
rng = np.random.default_rng(0)
idx = rng.choice(len(y_test), size=20, replace=False)
sample_df = pd.DataFrame({
    "Actual":    np.round(y_test[idx], 1),
    "Predicted": np.round(y_pred[idx], 1),
    "Error":     np.round(y_pred[idx] - y_test[idx], 1),
    "ActualBand":    pd.Series(y_test[idx]).map(to_ber_band).values,
    "PredictedBand": pd.Series(y_pred[idx]).map(to_ber_band).values,
})
print(sample_df.to_string(index=False))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("LightGBM — Full Test Set Evaluation (264,913 rows)", fontsize=14, fontweight="bold")

# 1. Predicted vs Actual
ax = axes[0, 0]
idx_plot = np.random.choice(len(y_test), 30_000, replace=False)
ax.scatter(y_test[idx_plot], y_pred[idx_plot], alpha=0.1, s=3, color="#2196F3")
lims = [y_test.min(), y_test.max()]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect")
ax.set_xlabel("Actual BerRating"); ax.set_ylabel("Predicted BerRating")
ax.set_title(f"Predicted vs Actual  (R²={r2:.4f})")
ax.legend()

# 2. Residual distribution
ax = axes[0, 1]
ax.hist(residuals, bins=100, color="#2196F3", edgecolor="none", alpha=0.8)
ax.axvline(0, color="red", lw=1.5, linestyle="--")
ax.axvline(np.mean(residuals), color="orange", lw=1.5, linestyle="--", label=f"Mean={bias:+.1f}")
ax.set_xlabel("Residual (Predicted − Actual)")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
ax.legend()

# 3. Residuals vs Predicted
ax = axes[0, 2]
ax.scatter(y_pred[idx_plot], residuals[idx_plot], alpha=0.1, s=3, color="#2196F3")
ax.axhline(0, color="red", lw=1.5, linestyle="--")
ax.set_xlabel("Predicted BerRating"); ax.set_ylabel("Residual")
ax.set_title("Residuals vs Predicted")

# 4. Absolute error CDF
ax = axes[1, 0]
abs_err = np.abs(residuals)
ax.plot(np.sort(abs_err), np.linspace(0, 1, len(abs_err)), color="#2196F3", lw=2)
for thresh, ls in [(25, "--"), (50, ":"), (75, "-.")]:
    pct = (abs_err <= thresh).mean() * 100
    ax.axvline(thresh, color="gray", lw=1, linestyle=ls, label=f"≤{thresh}: {pct:.1f}%")
ax.set_xlabel("Absolute Error (kWh/m²/yr)"); ax.set_ylabel("Cumulative Fraction")
ax.set_title("Absolute Error CDF")
ax.legend(fontsize=9); ax.set_xlim(0, 200)

# 5. Per-band MAE bar chart
ax = axes[1, 1]
bands_present = [b for b in BER_ORDER if b in band_mae.index]
maes = [band_mae.loc[b, "MAE"] for b in bands_present]
ax.bar(bands_present, maes, color="#2196F3", edgecolor="white")
ax.set_xlabel("BER Band"); ax.set_ylabel("MAE (kWh/m²/yr)")
ax.set_title("Per-Band MAE")
ax.tick_params(axis="x", rotation=45)

# 6. Per-band R²
ax = axes[1, 2]
r2s = [band_mae.loc[b, "R²"] for b in bands_present]
ax.bar(bands_present, r2s, color="#4CAF50", edgecolor="white")
ax.set_xlabel("BER Band"); ax.set_ylabel("R²")
ax.set_title("Per-Band R²")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_test_evaluation.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"\n  Saved → {OUTPUT_DIR}/model_test_evaluation.png")

# ── Save full report ──────────────────────────────────────────────────────────
report = f"""
================================================================
LightGBM — Full Dataset Test Evaluation
================================================================
Test rows : 264,913  (20% held-out split)
Dataset   : {DATA_PATH}

REGRESSION METRICS
------------------
R²                     : {r2:.4f}
RMSE                   : {rmse:.2f} kWh/m²/yr
MAE                    : {mae:.2f} kWh/m²/yr
MAPE                   : {mape:.2f}%
Mean Bias (pred-actual): {bias:+.2f} kWh/m²/yr

BER BAND ACCURACY
-----------------
Exact band match  : {exact_acc:.1f}%
Within ±1 band    : {within_1_acc:.1f}%
Within ±2 bands   : {within_2_acc:.1f}%

PER-BAND PERFORMANCE
--------------------
{band_mae.to_string()}
================================================================
""".strip()

print("\n" + report)
(OUTPUT_DIR / "model_test_report.txt").write_text(report)
print(f"\n  Saved → {OUTPUT_DIR}/model_test_report.txt")
print("\nDone.")
