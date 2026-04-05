"""
Irish Home Retrofit BER Prediction — Full Dataset Model Evaluation
==================================================================
Uses best hyperparameters from 50K-sample tuning run.
Trains on the complete dataset (no sampling) with 80/20 split.
No CV — params are already tuned.

Target: BerRating (kWh/m²/yr)
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
DATA_PATH   = "46_Col_final_with_county.parquet"
OUTPUT_DIR  = Path("model_analysis_outputs")
TARGET      = "BerRating"
TEST_SIZE   = 0.20

OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)

# ── Load full dataset ─────────────────────────────────────────────────────────
print("Loading full dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"  Full dataset shape: {df.shape}")

DROP_COLS = ["SHRenewableResources", "WHRenewableResources", "HSEffAdjFactor", "WHEffAdjFactor"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Remove extreme outliers in target (keep 1st–99th percentile)
q_lo = df[TARGET].quantile(0.01)
q_hi = df[TARGET].quantile(0.99)
df = df[(df[TARGET] >= q_lo) & (df[TARGET] <= q_hi)]
print(f"  After outlier trim: {df.shape}")

X = df.drop(columns=[TARGET])
y = df[TARGET].values

# ── Feature types ─────────────────────────────────────────────────────────────
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"\n  Categorical features : {len(cat_cols)}")
print(f"  Numerical features   : {len(num_cols)}")

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
cat_ridge_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
num_ridge_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
ridge_preprocessor = ColumnTransformer([
    ("cat", cat_ridge_pipe, cat_cols),
    ("num", num_ridge_pipe, num_cols),
], remainder="passthrough")

tree_preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", "passthrough", num_cols),
], remainder="passthrough")

# ── Models with best params from 50K tuning run ───────────────────────────────
models_cfg = {
    "LightGBM": Pipeline([
        ("pre", tree_preprocessor),
        ("model", lgb.LGBMRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
            n_estimators=800,
            num_leaves=31,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.9,
        )),
    ]),
    "Random Forest": Pipeline([
        ("pre", tree_preprocessor),
        ("model", RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            n_estimators=100,     # reduced from 300 for full-data feasibility
            max_depth=35,
            max_features=0.5,
        )),
    ]),
    "Ridge": Pipeline([
        ("pre", ridge_preprocessor),
        ("model", Ridge(alpha=32.9)),
    ]),
}

# ── Train & evaluate ──────────────────────────────────────────────────────────
results = {}
best_pipelines = {}

for name, pipeline in models_cfg.items():
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")
    t0 = time.time()

    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_pipelines[name] = pipeline

    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    train_r2  = r2_score(y_train, y_pred_train)
    test_r2   = r2_score(y_test,  y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae  = mean_absolute_error(y_test, y_pred_test)

    results[name] = {
        "Train R²":  round(train_r2, 4),
        "Test R²":   round(test_r2,  4),
        "Test RMSE": round(test_rmse, 2),
        "Test MAE":  round(test_mae,  2),
        "Time (s)":  round(elapsed, 1),
    }

    print(f"  Train R²   : {train_r2:.4f}")
    print(f"  Test R²    : {test_r2:.4f}")
    print(f"  Test RMSE  : {test_rmse:.2f} kWh/m²/yr")
    print(f"  Test MAE   : {test_mae:.2f} kWh/m²/yr")
    print(f"  Time       : {elapsed:.1f}s")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\n" + "="*65)
print("  FULL DATASET — MODEL COMPARISON SUMMARY")
print("="*65)
summary_df = pd.DataFrame(results).T
summary_df.index.name = "Model"
print(summary_df.to_string())
summary_df.to_csv(OUTPUT_DIR / "model_comparison_full_dataset.csv")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison — BER Rating Prediction (Full Dataset)", fontsize=14, fontweight="bold")

colors = {"LightGBM": "#2196F3", "Random Forest": "#4CAF50", "Ridge": "#FF9800"}
names  = list(results.keys())

ax = axes[0]
vals = [results[n]["Test R²"] for n in names]
bars = ax.bar(names, vals, color=[colors[n] for n in names], edgecolor="white", linewidth=1.5)
ax.set_ylim(0, 1.05)
ax.set_title("Test R²  (higher = better)")
ax.set_ylabel("R²")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax = axes[1]
vals = [results[n]["Test RMSE"] for n in names]
bars = ax.bar(names, vals, color=[colors[n] for n in names], edgecolor="white", linewidth=1.5)
ax.set_title("Test RMSE  (lower = better)")
ax.set_ylabel("RMSE (kWh/m²/yr)")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax = axes[2]
vals = [results[n]["Test MAE"] for n in names]
bars = ax.bar(names, vals, color=[colors[n] for n in names], edgecolor="white", linewidth=1.5)
ax.set_title("Test MAE  (lower = better)")
ax.set_ylabel("MAE (kWh/m²/yr)")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_full_dataset.png", bbox_inches="tight")
plt.close()
print(f"\n  Saved → {OUTPUT_DIR}/model_comparison_full_dataset.png")

# ── Predicted vs Actual for best model ───────────────────────────────────────
best_model_name = summary_df["Test R²"].astype(float).idxmax()
print(f"\n  BEST MODEL: {best_model_name}")

best_pipe   = best_pipelines[best_model_name]
y_pred_best = best_pipe.predict(X_test)

# Sample 20K for scatter plot readability
idx = np.random.choice(len(y_test), size=min(20_000, len(y_test)), replace=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Best Model: {best_model_name} — Full Dataset", fontsize=13, fontweight="bold")

ax = axes[0]
ax.scatter(y_test[idx], y_pred_best[idx], alpha=0.15, s=5, color=colors[best_model_name])
lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual BerRating")
ax.set_ylabel("Predicted BerRating")
ax.set_title("Predicted vs Actual")
ax.legend()

ax = axes[1]
residuals = y_test[idx] - y_pred_best[idx]
ax.scatter(y_pred_best[idx], residuals, alpha=0.15, s=5, color=colors[best_model_name])
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Predicted BerRating")
ax.set_ylabel("Residual")
ax.set_title("Residual Plot")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_model_diagnostics_full.png", bbox_inches="tight")
plt.close()
print(f"  Saved → {OUTPUT_DIR}/best_model_diagnostics_full.png")

# ── Feature importance for LightGBM ──────────────────────────────────────────
if "LightGBM" in best_pipelines:
    lgb_pipe     = best_pipelines["LightGBM"]
    lgb_model    = lgb_pipe.named_steps["model"]
    pre          = lgb_pipe.named_steps["pre"]
    feature_names = [f.split("__")[-1] for f in pre.get_feature_names_out()]
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": lgb_model.feature_importances_})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
    ax.set_title("LightGBM — Top 20 Feature Importances (Full Dataset)", fontweight="bold")
    ax.set_xlabel("Importance (split count)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lgb_feature_importance_full.png", bbox_inches="tight")
    plt.close()
    fi_df.to_csv(OUTPUT_DIR / "lgb_feature_importance_full.csv", index=False)
    print(f"  Saved → {OUTPUT_DIR}/lgb_feature_importance_full.png")

# ── Save text report ──────────────────────────────────────────────────────────
report_lines = [
    "="*65,
    "Irish Home Retrofit — BER Rating Prediction (FULL DATASET)",
    "="*65,
    f"Dataset  : {DATA_PATH}",
    f"Rows     : {len(df):,} (after outlier trim)",
    f"Split    : 80% train / 20% test  ({len(X_train):,} train | {len(X_test):,} test)",
    f"Target   : {TARGET} (numerical regression, kWh/m²/yr)",
    f"Note     : Hyperparameters fixed from 50K-sample tuning run",
    "",
    "-"*65,
    "RESULTS",
    "-"*65,
]
for name, res in results.items():
    report_lines += [
        f"\n{name}",
        f"  Train R²     : {res['Train R²']}",
        f"  Test R²      : {res['Test R²']}",
        f"  Test RMSE    : {res['Test RMSE']} kWh/m²/yr",
        f"  Test MAE     : {res['Test MAE']} kWh/m²/yr",
        f"  Training Time: {res['Time (s)']}s",
    ]

report_lines += [
    "",
    "="*65,
    f"BEST MODEL: {best_model_name}",
    f"  Test R²   = {results[best_model_name]['Test R²']}",
    f"  Test RMSE = {results[best_model_name]['Test RMSE']}",
    f"  Test MAE  = {results[best_model_name]['Test MAE']}",
    "="*65,
]

report_text = "\n".join(report_lines)
print("\n" + report_text)
(OUTPUT_DIR / "model_comparison_full_dataset_report.txt").write_text(report_text)
print(f"\n  Saved → {OUTPUT_DIR}/model_comparison_full_dataset_report.txt")
print("\nDone.")
