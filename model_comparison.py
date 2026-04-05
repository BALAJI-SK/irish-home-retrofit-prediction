"""
Irish Home Retrofit BER Prediction — Model Comparison
======================================================
Compares 3 regression models on a 50K stratified sample of the dataset.
Target: BerRating (numerical, kWh/m²/yr equivalent)

Models:
  1. LightGBM       — gradient boosting, handles categoricals natively
  2. Random Forest  — bagging ensemble, robust to outliers
  3. Ridge          — linear baseline (OrdinalEncoded + StandardScaler)

Pipeline:
  - 50K stratified sample from 1.35M rows
  - Train/Test split 80/20
  - RandomizedSearchCV (5-fold CV) for hyperparameter tuning
  - Metrics: R², RMSE, MAE on held-out test set
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

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
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
RANDOM_SEED   = 42
DATA_PATH     = "46_Col_final_with_county.parquet"
OUTPUT_DIR    = Path("model_analysis_outputs")
TARGET        = "BerRating"
SAMPLE_SIZE   = 50_000   # rows used for training + evaluation
TEST_SIZE     = 0.20
CV_FOLDS      = 5
N_ITER_SEARCH = 20       # RandomizedSearchCV iterations per model

OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)

# ── Load & sample ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"  Full dataset shape: {df.shape}")

# Drop known collinear columns (from prior VIF analysis)
DROP_COLS = ["SHRenewableResources", "WHRenewableResources", "HSEffAdjFactor", "WHEffAdjFactor"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Remove extreme outliers in target (keep 1st–99th percentile)
q_lo = df[TARGET].quantile(0.01)
q_hi = df[TARGET].quantile(0.99)
df = df[(df[TARGET] >= q_lo) & (df[TARGET] <= q_hi)]
print(f"  After outlier trim: {df.shape}")

# Stratified sample: bin BerRating into 10 strata
df["_stratum"] = pd.qcut(df[TARGET], q=10, labels=False, duplicates="drop")
sample = df.groupby("_stratum", group_keys=False).apply(
    lambda g: g.sample(min(len(g), SAMPLE_SIZE // 10), random_state=RANDOM_SEED)
)
if "_stratum" in sample.columns:
    sample = sample.drop(columns=["_stratum"])
sample = sample.reset_index(drop=True)
print(f"  Stratified sample size: {len(sample)}")

X = sample.drop(columns=[TARGET])
y = sample[TARGET].values

# ── Feature types ─────────────────────────────────────────────────────────────
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"\n  Categorical features: {len(cat_cols)}")
print(f"  Numerical features:   {len(num_cols)}")

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Ridge needs encoded + scaled features; LightGBM/RF handle categoricals via OrdinalEncoder
ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# For Ridge: impute → encode/scale (handles NaN)
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

# For tree models: just ordinal-encode categoricals
tree_preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", "passthrough", num_cols),
], remainder="passthrough")

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# ── Model definitions & search spaces ────────────────────────────────────────
models_cfg = {
    "LightGBM": {
        "pipeline": Pipeline([
            ("pre", tree_preprocessor),
            ("model", lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)),
        ]),
        "param_dist": {
            "model__n_estimators":      [200, 400, 600, 800],
            "model__num_leaves":        [31, 63, 127],
            "model__learning_rate":     [0.05, 0.08, 0.1, 0.15],
            "model__max_depth":         [-1, 6, 10],
            "model__min_child_samples": [10, 20, 50],
            "model__subsample":         [0.7, 0.8, 1.0],
            "model__colsample_bytree":  [0.7, 0.9, 1.0],
            "model__reg_alpha":         [0, 0.1, 0.5],
            "model__reg_lambda":        [0, 0.1, 1.0],
        },
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ("pre", tree_preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "param_dist": {
            "model__n_estimators":      [100, 200, 300],
            "model__max_depth":         [None, 15, 25, 35],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf":  [1, 2, 4],
            "model__max_features":      ["sqrt", "log2", 0.5],
        },
    },
    "Ridge": {
        "pipeline": Pipeline([
            ("pre", ridge_preprocessor),
            ("model", Ridge()),
        ]),
        "param_dist": {
            "model__alpha": np.logspace(-2, 4, 30),
        },
    },
}

# ── Train & evaluate ──────────────────────────────────────────────────────────
results = {}
best_pipelines = {}

for name, cfg in models_cfg.items():
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")
    t0 = time.time()

    search = RandomizedSearchCV(
        cfg["pipeline"],
        cfg["param_dist"],
        n_iter=N_ITER_SEARCH,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best = search.best_estimator_
    best_pipelines[name] = best

    y_pred_train = best.predict(X_train)
    y_pred_test  = best.predict(X_test)

    train_r2   = r2_score(y_train, y_pred_train)
    test_r2    = r2_score(y_test, y_pred_test)
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae   = mean_absolute_error(y_test, y_pred_test)
    cv_r2_mean = search.best_score_
    cv_r2_std  = search.cv_results_["std_test_score"][search.best_index_]

    results[name] = {
        "Train R²":  round(train_r2, 4),
        "Test R²":   round(test_r2, 4),
        "CV R² (mean)": round(cv_r2_mean, 4),
        "CV R² (std)":  round(cv_r2_std, 4),
        "Test RMSE": round(test_rmse, 2),
        "Test MAE":  round(test_mae, 2),
        "Time (s)":  round(elapsed, 1),
        "Best Params": search.best_params_,
    }

    print(f"\n  Best Params : {search.best_params_}")
    print(f"  Train R²    : {train_r2:.4f}")
    print(f"  Test R²     : {test_r2:.4f}")
    print(f"  CV R²       : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"  Test RMSE   : {test_rmse:.2f}")
    print(f"  Test MAE    : {test_mae:.2f}")
    print(f"  Time        : {elapsed:.1f}s")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\n" + "="*65)
print("  MODEL COMPARISON SUMMARY")
print("="*65)
summary_df = pd.DataFrame({
    k: {m: v for m, v in res.items() if m != "Best Params"}
    for k, res in results.items()
}).T
summary_df.index.name = "Model"
print(summary_df.to_string())
summary_df.to_csv(OUTPUT_DIR / "model_comparison.csv")
print(f"\n  Saved → {OUTPUT_DIR}/model_comparison.csv")

# ── Determine winner ──────────────────────────────────────────────────────────
best_model_name = summary_df["Test R²"].astype(float).idxmax()
print(f"\n  BEST MODEL: {best_model_name}  (Test R² = {summary_df.loc[best_model_name, 'Test R²']})")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison — BER Rating Prediction (50K sample)", fontsize=14, fontweight="bold")

colors = {"LightGBM": "#2196F3", "Random Forest": "#4CAF50", "Ridge": "#FF9800"}

# Bar chart: Test R²
ax = axes[0]
names = list(results.keys())
test_r2s = [results[n]["Test R²"] for n in names]
bars = ax.bar(names, test_r2s, color=[colors[n] for n in names], edgecolor="white", linewidth=1.5)
ax.set_ylim(0, 1.05)
ax.set_title("Test R²  (higher = better)")
ax.set_ylabel("R²")
for bar, val in zip(bars, test_r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Bar chart: Test RMSE
ax = axes[1]
rmses = [results[n]["Test RMSE"] for n in names]
bars = ax.bar(names, rmses, color=[colors[n] for n in names], edgecolor="white", linewidth=1.5)
ax.set_title("Test RMSE  (lower = better)")
ax.set_ylabel("RMSE (kWh/m²/yr)")
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Bar chart: Test MAE
ax = axes[2]
maes = [results[n]["Test MAE"] for n in names]
bars = ax.bar(names, maes, color=[colors[n] for n in names], edgecolor="white", linewidth=1.5)
ax.set_title("Test MAE  (lower = better)")
ax.set_ylabel("MAE (kWh/m²/yr)")
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison_metrics.png", bbox_inches="tight")
print(f"  Saved → {OUTPUT_DIR}/model_comparison_metrics.png")
plt.close()

# ── Predicted vs Actual for best model ───────────────────────────────────────
best_pipe = best_pipelines[best_model_name]
y_pred_best = best_pipe.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Best Model: {best_model_name} — Predicted vs Actual", fontsize=13, fontweight="bold")

ax = axes[0]
ax.scatter(y_test, y_pred_best, alpha=0.25, s=8, color=colors[best_model_name])
lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual BerRating")
ax.set_ylabel("Predicted BerRating")
ax.set_title("Predicted vs Actual")
ax.legend()

ax = axes[1]
residuals = y_test - y_pred_best
ax.scatter(y_pred_best, residuals, alpha=0.25, s=8, color=colors[best_model_name])
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Predicted BerRating")
ax.set_ylabel("Residual")
ax.set_title("Residual Plot")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_model_diagnostics.png", bbox_inches="tight")
print(f"  Saved → {OUTPUT_DIR}/best_model_diagnostics.png")
plt.close()

# ── Feature importance for LightGBM ──────────────────────────────────────────
if "LightGBM" in best_pipelines:
    lgb_pipe = best_pipelines["LightGBM"]
    lgb_model = lgb_pipe.named_steps["model"]
    pre = lgb_pipe.named_steps["pre"]
    feature_names = pre.get_feature_names_out()
    # Strip prefixes added by ColumnTransformer
    feature_names = [f.split("__")[-1] for f in feature_names]
    importances = lgb_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
    ax.set_title("LightGBM — Top 20 Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance (split count)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lgb_feature_importance.png", bbox_inches="tight")
    print(f"  Saved → {OUTPUT_DIR}/lgb_feature_importance.png")
    plt.close()
    fi_df.to_csv(OUTPUT_DIR / "lgb_feature_importance.csv", index=False)

# ── Save text report ──────────────────────────────────────────────────────────
report_lines = [
    "="*65,
    "Irish Home Retrofit — BER Rating Prediction: Model Comparison",
    "="*65,
    f"Dataset  : {DATA_PATH}",
    f"Sample   : {SAMPLE_SIZE:,} rows (stratified by BerRating deciles)",
    f"Split    : {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test",
    f"CV Folds : {CV_FOLDS}-fold KFold",
    f"Target   : {TARGET} (numerical regression)",
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
        f"  CV R² (mean) : {res['CV R² (mean)']} ± {res['CV R² (std)']}",
        f"  Test RMSE    : {res['Test RMSE']}",
        f"  Test MAE     : {res['Test MAE']}",
        f"  Training Time: {res['Time (s)']}s",
        f"  Best Params  : {res['Best Params']}",
    ]

report_lines += [
    "",
    "="*65,
    f"BEST MODEL: {best_model_name}",
    f"  Test R²  = {results[best_model_name]['Test R²']}",
    f"  Test RMSE = {results[best_model_name]['Test RMSE']}",
    f"  Test MAE  = {results[best_model_name]['Test MAE']}",
    "="*65,
]

report_text = "\n".join(report_lines)
print("\n" + report_text)
(OUTPUT_DIR / "model_comparison_report.txt").write_text(report_text)
print(f"\n  Saved → {OUTPUT_DIR}/model_comparison_report.txt")
print("\nDone.")
