"""
Irish Home Retrofit — BER Rating Prediction
============================================
Proper ML Pipeline:
  1. Load full dataset
  2. Train / Test split  (80 / 20, stratified by BER decile)
  3. Hyperparameter tuning via RandomizedSearchCV + 5-fold CV
     (tuning on 200K stratified subsample of train set for feasibility;
      final model retrained on full train set with best params)
  4. Evaluation on held-out test set: R², RMSE, MAE

Models:
  - Ridge          (linear baseline)
  - Random Forest  (bagging ensemble)
  - LightGBM       (gradient boosting)
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

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
DATA_PATH      = "46_Col_final_with_county.parquet"
OUTPUT_DIR     = Path("model_analysis_outputs")
TARGET         = "BerRating"
TEST_SIZE      = 0.20
TUNE_SAMPLE    = 200_000   # rows used for CV + hyperparameter tuning
CV_FOLDS       = 5
N_ITER         = 20        # RandomizedSearchCV iterations

OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & clean
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading dataset")
print("=" * 65)

df = pd.read_parquet(DATA_PATH)
print(f"  Raw shape            : {df.shape}")

DROP_COLS = ["SHRenewableResources", "WHRenewableResources",
             "HSEffAdjFactor", "WHEffAdjFactor"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Remove extreme outliers in target (1st–99th percentile)
q_lo = df[TARGET].quantile(0.01)
q_hi = df[TARGET].quantile(0.99)
df   = df[(df[TARGET] >= q_lo) & (df[TARGET] <= q_hi)]
print(f"  After outlier trim   : {df.shape}")
print(f"  Target range         : [{q_lo:.1f}, {q_hi:.1f}] kWh/m²/yr")

X = df.drop(columns=[TARGET])
y = df[TARGET].values

cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"  Categorical features : {len(cat_cols)}")
print(f"  Numerical features   : {len(num_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train / Test split
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Train / Test split  (80 / 20)")
print("=" * 65)

# Stratify on BER decile bins to preserve distribution
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = strata,
)
print(f"  Full train set  : {len(X_train):,} rows")
print(f"  Test set        : {len(X_test):,}  rows  (held-out, not touched until final eval)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Preprocessing pipelines
# ─────────────────────────────────────────────────────────────────────────────
# Ridge: impute → ordinal-encode cats → scale all
cat_ridge = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
num_ridge = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
ridge_pre = ColumnTransformer([
    ("cat", cat_ridge, cat_cols),
    ("num", num_ridge, num_cols),
], remainder="passthrough")

# Tree models: ordinal-encode cats, pass nums through
tree_pre = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", "passthrough", num_cols),
], remainder="passthrough")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Hyperparameter search spaces
# ─────────────────────────────────────────────────────────────────────────────
models_cfg = {
    "Ridge": {
        "pipeline": Pipeline([
            ("pre",   ridge_pre),
            ("model", Ridge()),
        ]),
        "param_dist": {
            "model__alpha": np.logspace(-2, 4, 50),
        },
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ("pre",   tree_pre),
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
    "LightGBM": {
        "pipeline": Pipeline([
            ("pre",   tree_pre),
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
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Tuning subsample (200K stratified rows from train set)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 5 — Hyperparameter tuning on {TUNE_SAMPLE:,}-row stratified subsample")
print(f"         ({CV_FOLDS}-fold CV, {N_ITER} random search iterations per model)")
print("=" * 65)

tune_strata = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train,
    train_size   = TUNE_SAMPLE,
    random_state = RANDOM_SEED,
    stratify     = tune_strata,
)
print(f"  Tuning subset size : {len(X_tune):,} rows")

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Train, tune, evaluate
# ─────────────────────────────────────────────────────────────────────────────
results      = {}
best_params  = {}
cv_scores    = {}

for name, cfg in models_cfg.items():
    print(f"\n{'─'*65}")
    print(f"  Model : {name}")
    print(f"{'─'*65}")

    # --- Hyperparameter search on tuning subsample ---
    print(f"  [1/3] RandomizedSearchCV ({CV_FOLDS}-fold) on {TUNE_SAMPLE:,} rows ...")
    t0 = time.time()
    search = RandomizedSearchCV(
        estimator  = cfg["pipeline"],
        param_distributions = cfg["param_dist"],
        n_iter     = N_ITER,
        scoring    = "r2",
        cv         = cv,
        random_state = RANDOM_SEED,
        n_jobs     = -1,
        refit      = False,
        verbose    = 0,
    )
    search.fit(X_tune, y_tune)
    tune_time = time.time() - t0

    best_p = search.best_params_
    cv_r2  = search.best_score_
    cv_std = search.cv_results_["std_test_score"][search.best_index_]
    best_params[name] = best_p
    cv_scores[name]   = {"cv_r2": cv_r2, "cv_std": cv_std}

    print(f"  Best CV R²  : {cv_r2:.4f} ± {cv_std:.4f}  (tuning time: {tune_time:.0f}s)")
    print(f"  Best params : {best_p}")

    # --- Retrain on full train set with best params ---
    print(f"  [2/3] Retraining on full train set ({len(X_train):,} rows) ...")
    final_pipe = cfg["pipeline"]
    final_pipe.set_params(**best_p)
    t0 = time.time()
    final_pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Retrain time: {train_time:.0f}s")

    # --- Evaluate on held-out test set ---
    print(f"  [3/3] Evaluating on held-out test set ({len(X_test):,} rows) ...")
    y_pred_train = final_pipe.predict(X_train)
    y_pred_test  = final_pipe.predict(X_test)

    train_r2  = r2_score(y_train,   y_pred_train)
    test_r2   = r2_score(y_test,    y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae  = mean_absolute_error(y_test, y_pred_test)

    results[name] = {
        "CV R² (mean)":  round(cv_r2,     4),
        "CV R² (±std)":  round(cv_std,    4),
        "Train R²":      round(train_r2,  4),
        "Test R²":       round(test_r2,   4),
        "Test RMSE":     round(test_rmse, 2),
        "Test MAE":      round(test_mae,  2),
        "Tune Time (s)": round(tune_time, 1),
        "Train Time (s)":round(train_time,1),
        "Best Params":   best_p,
    }

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  CV R²       : {cv_r2:>7.4f} ± {cv_std:.4f}  │")
    print(f"  │  Train R²    : {train_r2:>7.4f}           │")
    print(f"  │  Test R²     : {test_r2:>7.4f}           │")
    print(f"  │  Test RMSE   : {test_rmse:>7.2f} kWh/m²/yr  │")
    print(f"  │  Test MAE    : {test_mae:>7.2f} kWh/m²/yr  │")
    print(f"  └─────────────────────────────────┘")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("STEP 7 — FINAL MODEL COMPARISON SUMMARY")
print("=" * 65)

display_cols = ["CV R² (mean)", "CV R² (±std)", "Train R²", "Test R²", "Test RMSE", "Test MAE"]
summary_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k in display_cols}
    for name, res in results.items()
}).T
summary_df.index.name = "Model"
print(summary_df.to_string())

best_model = summary_df["Test R²"].astype(float).idxmax()
print(f"\n  BEST MODEL: {best_model}  (Test R² = {summary_df.loc[best_model, 'Test R²']})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Plots
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"ML Pipeline — BER Rating Prediction  |  Full Dataset ({len(df):,} rows)",
    fontsize=13, fontweight="bold"
)

colors = {"LightGBM": "#2196F3", "Random Forest": "#4CAF50", "Ridge": "#FF9800"}
names  = list(results.keys())

metric_specs = [
    ("CV R² (mean)", "CV R²  (mean ± std, 5-fold)",  "R²",              True,  0),
    ("Test R²",      "Test R²  (higher = better)",    "R²",              True,  0),
    ("Test RMSE",    "Test RMSE  (lower = better)",   "RMSE (kWh/m²/yr)",False, 0.5),
]

for ax, (metric, title, ylabel, is_r2, offset) in zip(axes, metric_specs):
    vals = [results[n][metric] for n in names]
    errs = [results[n]["CV R² (±std)"] for n in names] if metric == "CV R² (mean)" else None
    bars = ax.bar(names, vals, color=[colors[n] for n in names],
                  edgecolor="white", linewidth=1.5,
                  yerr=errs, capsize=5 if errs else 0)
    if is_r2:
        ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + offset + (0.01 if is_r2 else 0),
                f"{val:.4f}" if is_r2 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ml_pipeline_comparison.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"\n  Saved → {OUTPUT_DIR}/ml_pipeline_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save report
# ─────────────────────────────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "Irish Home Retrofit — BER Rating Prediction: ML Pipeline",
    "=" * 65,
    f"Dataset       : {DATA_PATH}",
    f"Total rows    : {len(df):,}  (after outlier trim)",
    f"Train set     : {len(X_train):,} rows  (80%)",
    f"Test set      : {len(X_test):,}  rows  (20%, held-out)",
    f"Tuning subset : {TUNE_SAMPLE:,} rows  (stratified from train set)",
    f"CV            : {CV_FOLDS}-fold KFold",
    f"Tuning        : RandomizedSearchCV  ({N_ITER} iterations)",
    f"Target        : {TARGET}  (kWh/m²/yr)",
    "",
    "-" * 65,
    "RESULTS",
    "-" * 65,
]
for name, res in results.items():
    report_lines += [
        f"\n{name}",
        f"  CV R²        : {res['CV R² (mean)']} ± {res['CV R² (±std)']}",
        f"  Train R²     : {res['Train R²']}",
        f"  Test R²      : {res['Test R²']}",
        f"  Test RMSE    : {res['Test RMSE']} kWh/m²/yr",
        f"  Test MAE     : {res['Test MAE']} kWh/m²/yr",
        f"  Tune time    : {res['Tune Time (s)']}s",
        f"  Train time   : {res['Train Time (s)']}s",
        f"  Best Params  : {res['Best Params']}",
    ]

report_lines += [
    "",
    "=" * 65,
    f"BEST MODEL: {best_model}",
    f"  Test R²   = {results[best_model]['Test R²']}",
    f"  Test RMSE = {results[best_model]['Test RMSE']}",
    f"  Test MAE  = {results[best_model]['Test MAE']}",
    "=" * 65,
]

report_text = "\n".join(report_lines)
print("\n" + report_text)
(OUTPUT_DIR / "ml_pipeline_report.txt").write_text(report_text)
print(f"\n  Saved → {OUTPUT_DIR}/ml_pipeline_report.txt")
print("\nDone.")
