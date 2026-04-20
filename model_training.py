"""
Irish Home Retrofit BER Prediction - ML Pipeline
=================================================
Regression pipeline for predicting BER Rating on 1.35M Irish dwellings.

Models  : LightGBM  |  Random Forest  |  Ridge Regression
Strategy: Tune on stratified subsample → refit best params on full train set.
          This cuts search time by ~5-10x without sacrificing final quality.

Outputs (model_analysis_outputs/):
  - model_comparison.csv          : metrics table
  - evaluation_report.txt         : full text report
  - feature_importance_*.csv      : per-model feature importances
  - model_*.pkl                   : serialised best estimators
  - 01_ … 18_*.png                : diagnostic plots
"""

import os
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, KFold,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
RANDOM_SEED   = 42
DATA_PATH     = "46_Col_final_with_county.parquet"
OUTPUT_DIR    = Path("model_analysis_outputs")
TARGET        = "BerRating"
CV_FOLDS      = 5

# Rows used during RandomizedSearchCV.
# Full 1M+ rows makes search prohibitively slow; a stratified 200K sample
# gives stable CV estimates in a fraction of the time.
TUNE_SAMPLE   = 200_000

OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "training.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11,
})
sns.set_theme(style="whitegrid", palette="muted")


# ══════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════
log.info("=" * 60)
log.info("STEP 1: Loading and inspecting data")
log.info("=" * 60)

df = pd.read_parquet(DATA_PATH)
log.info(f"Shape  : {df.shape}")
log.info(f"Memory : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Drop columns flagged for collinearity / leakage
DROP_COLS = ["SHRenewableResources", "WHRenewableResources",
             "HSEffAdjFactor", "WHEffAdjFactor"]
DROP_COLS = [c for c in DROP_COLS if c in df.columns]
df.drop(columns=DROP_COLS, inplace=True)
log.info(f"Dropped collinear columns: {DROP_COLS}")

CAT_COLS = df.select_dtypes(include=["object", "category"]).columns.tolist()
NUM_COLS = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]

log.info(f"Categorical features : {len(CAT_COLS)}")
log.info(f"Numerical features   : {len(NUM_COLS)}")
log.info(f"Target               : {TARGET}")
log.info(f"\nTarget statistics:\n{df[TARGET].describe().to_string()}")

# Save descriptive stats & correlations
df.describe(include="all").to_csv(OUTPUT_DIR / "data_summary_statistics.csv")
log.info("Saved: data_summary_statistics.csv")

corr_with_target = (
    df[NUM_COLS + [TARGET]].corr()[TARGET]
    .drop(TARGET).sort_values(key=abs, ascending=False)
)
corr_with_target.to_csv(OUTPUT_DIR / "feature_correlation_with_target.csv",
                        header=["pearson_r"])
log.info(f"\nTop 10 features correlated with {TARGET}:\n"
         f"{corr_with_target.head(10).to_string()}")


# ══════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 2: Preprocessing & splits")
log.info("=" * 60)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(np.float64)

# Stratify by decile to preserve the target distribution
y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y_bins
)
log.info(f"Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}")

# ── Tuning subsample (stratified) ──────────────
tune_size = min(TUNE_SAMPLE, len(X_train))
y_train_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, train_size=tune_size,
    random_state=RANDOM_SEED, stratify=y_train_bins,
)
log.info(f"Tuning subsample : {len(X_tune):,} rows ({tune_size/len(X_train)*100:.0f}% of train)")

# ── Preprocessing pipeline (for Ridge) ─────────
num_pipe = Pipeline([("scaler", StandardScaler())])
cat_pipe = Pipeline([("ord_enc", OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
))])
preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM_COLS),
    ("cat", cat_pipe, CAT_COLS),
], remainder="drop")


# ══════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 3: Model definitions")
log.info("=" * 60)

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


# ── Helper: encode categoricals as 'category' dtype (LightGBM native) ──
def encode_lgbm(X_df):
    out = X_df.copy()
    for c in CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


# ── Helper: ordinal-encode categoricals for sklearn models ──────────────
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
ord_enc.fit(X_train[CAT_COLS])          # fit once on full train

def encode_sklearn(X_df):
    out = X_df.copy()
    out[CAT_COLS] = ord_enc.transform(X_df[CAT_COLS])
    return out.astype(np.float32)


X_train_lgbm = encode_lgbm(X_train)
X_test_lgbm  = encode_lgbm(X_test)
X_tune_lgbm  = encode_lgbm(X_tune)

X_train_sk   = encode_sklearn(X_train)
X_test_sk    = encode_sklearn(X_test)
X_tune_sk    = encode_sklearn(X_tune)


# ── 3a. LightGBM param grid ────────────────────
lgbm_base = lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
lgbm_param_dist = {
    "n_estimators":      [300, 500, 800, 1000],
    "learning_rate":     [0.03, 0.05, 0.08, 0.1],
    "max_depth":         [-1, 8, 12],
    "num_leaves":        [63, 127, 255],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.8, 1.0],
    "reg_alpha":         [0.0, 0.1, 0.5],
    "reg_lambda":        [0.0, 0.1, 1.0],
    "min_child_samples": [20, 50],
}

# ── 3b. Random Forest param grid ───────────────
rf_base = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
rf_param_dist = {
    "n_estimators":      [200, 300, 400],
    "max_depth":         [None, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.4],
    "max_samples":       [0.7, 0.85, 1.0],
}

# ── 3c. Ridge param grid ───────────────────────
ridge_pipeline = Pipeline([
    ("prep", preprocessor),
    ("ridge", Ridge(random_state=RANDOM_SEED)),
])
ridge_param_dist = {"ridge__alpha": np.logspace(-2, 5, 60)}


# ══════════════════════════════════════════════
# 4. TUNE → REFIT PIPELINE
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 4: Hyperparameter tuning & training")
log.info(f"        (search on {len(X_tune):,}-row subsample, final fit on {len(X_train):,} rows)")
log.info("=" * 60)

results = {}


def evaluate(model, X_tr, y_tr, X_te, y_te, name):
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    return {
        "model_name":  name,
        "train_r2":    r2_score(y_tr, y_pred_tr),
        "test_r2":     r2_score(y_te, y_pred_te),
        "train_rmse":  np.sqrt(mean_squared_error(y_tr, y_pred_tr)),
        "test_rmse":   np.sqrt(mean_squared_error(y_te, y_pred_te)),
        "train_mae":   mean_absolute_error(y_tr, y_pred_tr),
        "test_mae":    mean_absolute_error(y_te, y_pred_te),
    }


# ── 4a. LightGBM ──────────────────────────────
log.info("\n--- LightGBM ---")
t0 = time.time()

lgbm_search = RandomizedSearchCV(
    lgbm_base, lgbm_param_dist,
    n_iter=30, cv=cv, scoring="r2",
    refit=False, random_state=RANDOM_SEED, n_jobs=1, verbose=0,
)
lgbm_search.fit(X_tune_lgbm, y_tune)
lgbm_best_params = lgbm_search.best_params_
log.info(f"Best params (tuning): {lgbm_best_params}")
log.info(f"Best CV R² on subsample: {lgbm_search.best_score_:.4f}")

# Refit on full training set with best params
best_lgbm = lgb.LGBMRegressor(
    **lgbm_best_params, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
)
best_lgbm.fit(X_train_lgbm, y_train)
lgbm_time = time.time() - t0

# CV on full training set with best model
lgbm_cv = cross_val_score(best_lgbm, X_train_lgbm, y_train,
                           cv=cv, scoring="r2", n_jobs=1)

lgbm_metrics = evaluate(best_lgbm, X_train_lgbm, y_train, X_test_lgbm, y_test, "LightGBM")
lgbm_metrics.update({
    "cv_mean_r2": lgbm_cv.mean(),
    "cv_std_r2":  lgbm_cv.std(),
    "training_time_s": lgbm_time,
})
results["LightGBM"] = lgbm_metrics
log.info(
    f"LightGBM  Train R²={lgbm_metrics['train_r2']:.4f}  "
    f"Test R²={lgbm_metrics['test_r2']:.4f}  "
    f"CV={lgbm_cv.mean():.4f}±{lgbm_cv.std():.4f}  "
    f"Time={lgbm_time:.1f}s"
)


# ── 4b. Random Forest ─────────────────────────
log.info("\n--- Random Forest ---")
t0 = time.time()

rf_search = RandomizedSearchCV(
    rf_base, rf_param_dist,
    n_iter=20, cv=cv, scoring="r2",
    refit=False, random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
)
rf_search.fit(X_tune_sk, y_tune)
rf_best_params = rf_search.best_params_
log.info(f"Best params (tuning): {rf_best_params}")
log.info(f"Best CV R² on subsample: {rf_search.best_score_:.4f}")

# Refit on full training set
best_rf = RandomForestRegressor(
    **rf_best_params, random_state=RANDOM_SEED, n_jobs=-1
)
best_rf.fit(X_train_sk, y_train)
rf_time = time.time() - t0

rf_cv = cross_val_score(best_rf, X_train_sk, y_train,
                        cv=cv, scoring="r2", n_jobs=-1)

rf_metrics = evaluate(best_rf, X_train_sk, y_train, X_test_sk, y_test, "Random Forest")
rf_metrics.update({
    "cv_mean_r2": rf_cv.mean(),
    "cv_std_r2":  rf_cv.std(),
    "training_time_s": rf_time,
})
results["Random Forest"] = rf_metrics
log.info(
    f"Random Forest  Train R²={rf_metrics['train_r2']:.4f}  "
    f"Test R²={rf_metrics['test_r2']:.4f}  "
    f"CV={rf_cv.mean():.4f}±{rf_cv.std():.4f}  "
    f"Time={rf_time:.1f}s"
)


# ── 4c. Ridge ─────────────────────────────────
log.info("\n--- Ridge Regression ---")
t0 = time.time()

ridge_search = RandomizedSearchCV(
    ridge_pipeline, ridge_param_dist,
    n_iter=30, cv=cv, scoring="r2",
    refit=True, random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
)
# Ridge is fast enough to tune on the full training set
ridge_search.fit(X_train, y_train)
best_ridge = ridge_search.best_estimator_
ridge_time = time.time() - t0
log.info(f"Best alpha: {ridge_search.best_params_}")
log.info(f"Best CV R² on full train: {ridge_search.best_score_:.4f}")

ridge_cv = cross_val_score(best_ridge, X_train, y_train,
                            cv=cv, scoring="r2", n_jobs=-1)

ridge_metrics = evaluate(best_ridge, X_train, y_train, X_test, y_test, "Ridge")
ridge_metrics.update({
    "cv_mean_r2": ridge_cv.mean(),
    "cv_std_r2":  ridge_cv.std(),
    "training_time_s": ridge_time,
})
results["Ridge"] = ridge_metrics
log.info(
    f"Ridge  Train R²={ridge_metrics['train_r2']:.4f}  "
    f"Test R²={ridge_metrics['test_r2']:.4f}  "
    f"CV={ridge_cv.mean():.4f}±{ridge_cv.std():.4f}  "
    f"Time={ridge_time:.1f}s"
)


# ══════════════════════════════════════════════
# 5. MODEL COMPARISON TABLE
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 5: Model comparison")
log.info("=" * 60)


def verdict(train_r2, test_r2):
    gap = train_r2 - test_r2
    if test_r2 < 0.70:
        return "Underfitting"
    if gap > 0.08:
        return "Overfitting"
    return "Well-fitted"


comparison_rows = []
for name, m in results.items():
    comparison_rows.append({
        "Model":            name,
        "Train R²":         round(m["train_r2"],    4),
        "Validation R²":    round(m["cv_mean_r2"],  4),
        "Test R²":          round(m["test_r2"],     4),
        "Train RMSE":       round(m["train_rmse"],  4),
        "Test RMSE":        round(m["test_rmse"],   4),
        "Train MAE":        round(m["train_mae"],   4),
        "Test MAE":         round(m["test_mae"],    4),
        "CV Std R²":        round(m["cv_std_r2"],   4),
        "Training Time(s)": round(m["training_time_s"], 1),
        "Verdict":          verdict(m["train_r2"], m["test_r2"]),
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
log.info("\nModel comparison table:")
log.info("\n" + comparison_df.to_string(index=False))


# ══════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 6: Feature importance")
log.info("=" * 60)

lgbm_fi = pd.DataFrame({
    "feature":    best_lgbm.feature_name_,
    "importance": best_lgbm.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
lgbm_fi.to_csv(OUTPUT_DIR / "feature_importance_lightgbm.csv", index=False)
log.info(f"Top 10 LightGBM features:\n{lgbm_fi.head(10).to_string(index=False)}")

rf_fi = pd.DataFrame({
    "feature":    X_train_sk.columns.tolist(),
    "importance": best_rf.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)
rf_fi.to_csv(OUTPUT_DIR / "feature_importance_random_forest.csv", index=False)
log.info(f"Top 10 RF features:\n{rf_fi.head(10).to_string(index=False)}")


# ══════════════════════════════════════════════
# 7. SAVE MODELS
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 7: Saving models")
log.info("=" * 60)

joblib.dump(best_lgbm,  OUTPUT_DIR / "model_lightgbm.pkl")
joblib.dump(best_rf,    OUTPUT_DIR / "model_random_forest.pkl")
joblib.dump(best_ridge, OUTPUT_DIR / "model_ridge.pkl")
log.info("Saved: model_lightgbm.pkl, model_random_forest.pkl, model_ridge.pkl")


# ══════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 8: Generating plots")
log.info("=" * 60)

# Collect CV scores dicts for later plots
cv_scores = {
    "LightGBM":     lgbm_cv,
    "Random Forest": rf_cv,
    "Ridge":         ridge_cv,
}

# ── Plot 1: Correlation heatmap ───────────────
log.info("Plot 1: Correlation heatmap")
sample_n = min(50_000, len(df))
df_s = df[NUM_COLS + [TARGET]].sample(sample_n, random_state=RANDOM_SEED)
corr_matrix = df_s.corr()

fig, ax = plt.subplots(figsize=(20, 18))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm",
            center=0, ax=ax, linewidths=0.3)
ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_correlation_heatmap.png")
plt.close(fig)
log.info("Saved: 01_correlation_heatmap.png")

# ── Plot 2: Feature importance – LightGBM ────
log.info("Plot 2: Feature importance – LightGBM")
TOP_N = 25
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(data=lgbm_fi.head(TOP_N), y="feature", x="importance",
            ax=ax, palette="viridis")
ax.set_title(f"LightGBM Feature Importance (Top {TOP_N})",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Importance (Split count)")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_feature_importance_lightgbm.png")
plt.close(fig)
log.info("Saved: 02_feature_importance_lightgbm.png")

# ── Plot 3: Feature importance – Random Forest ─
log.info("Plot 3: Feature importance – Random Forest")
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(data=rf_fi.head(TOP_N), y="feature", x="importance",
            ax=ax, palette="magma")
ax.set_title(f"Random Forest Feature Importance (Top {TOP_N})",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Importance (Mean Decrease Impurity)")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_feature_importance_random_forest.png")
plt.close(fig)
log.info("Saved: 03_feature_importance_random_forest.png")

# ── Plots 4-6: Residuals ──────────────────────
def residual_plot(model, X_tr, y_tr, X_te, y_te, name, fname):
    log.info(f"Plot: Residual – {name}")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, Xd, yd, split in zip(axes,
                                  [X_tr, X_te], [y_tr, y_te],
                                  ["Train", "Test"]):
        y_pred = model.predict(Xd)
        res = np.array(yd) - y_pred
        ax.scatter(y_pred, res, alpha=0.15, s=4, color="steelblue")
        ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Predicted BerRating")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{name} – {split} Residuals")
        ax.text(0.97, 0.95, f"R²={r2_score(yd, y_pred):.4f}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    plt.suptitle(f"Residual Plot – {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    log.info(f"Saved: {fname}")

residual_plot(best_lgbm,  X_train_lgbm, y_train, X_test_lgbm, y_test,
              "LightGBM",      "04_residual_lightgbm.png")
residual_plot(best_rf,    X_train_sk,   y_train, X_test_sk,   y_test,
              "Random Forest", "05_residual_random_forest.png")
residual_plot(best_ridge, X_train,      y_train, X_test,      y_test,
              "Ridge",         "06_residual_ridge.png")

# ── Plots 7-9: Predicted vs Actual ───────────
def pred_vs_actual_plot(model, X_te, y_te, name, fname):
    log.info(f"Plot: Pred vs Actual – {name}")
    y_pred = model.predict(X_te)
    idx = np.random.choice(len(y_te), size=min(20_000, len(y_te)), replace=False)
    y_s, yp_s = np.array(y_te)[idx], y_pred[idx]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_s, yp_s, alpha=0.15, s=4, color="cornflowerblue")
    lo, hi = min(y_s.min(), yp_s.min()), max(y_s.max(), yp_s.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    ax.set_xlabel("Actual BerRating")
    ax.set_ylabel("Predicted BerRating")
    ax.set_title(f"{name} – Predicted vs Actual\nR²={r2:.4f}  RMSE={rmse:.2f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)
    log.info(f"Saved: {fname}")

pred_vs_actual_plot(best_lgbm,  X_test_lgbm, y_test, "LightGBM",      "07_pred_vs_actual_lightgbm.png")
pred_vs_actual_plot(best_rf,    X_test_sk,   y_test, "Random Forest", "08_pred_vs_actual_rf.png")
pred_vs_actual_plot(best_ridge, X_test,      y_test, "Ridge",         "09_pred_vs_actual_ridge.png")

# ── Plot 10: Error distribution ───────────────
log.info("Plot 10: Error distribution")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_cfgs = [
    (best_lgbm,  X_test_lgbm, "LightGBM",      "dodgerblue"),
    (best_rf,    X_test_sk,   "Random Forest",  "darkorange"),
    (best_ridge, X_test,      "Ridge",          "mediumseagreen"),
]
for ax, (model, Xte, name, color) in zip(axes, model_cfgs):
    errors = np.array(y_test) - model.predict(Xte)
    ax.hist(errors, bins=80, color=color, edgecolor="none", alpha=0.8)
    ax.axvline(0,             color="red",  linewidth=1.5, linestyle="--")
    ax.axvline(errors.mean(), color="navy", linewidth=1.5, linestyle=":",
               label=f"Mean={errors.mean():.2f}")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}\nError Distribution")
    ax.legend(fontsize=9)
plt.suptitle("Error Distribution Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "10_error_distribution.png")
plt.close(fig)
log.info("Saved: 10_error_distribution.png")

# ── Plot 11: CV score box-plot ────────────────
log.info("Plot 11: Cross-validation score comparison")
fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(cv_scores))
colors_cv = ["dodgerblue", "darkorange", "mediumseagreen"]
for i, (name, scores) in enumerate(cv_scores.items()):
    ax.boxplot(scores, positions=[i], widths=0.4,
               patch_artist=True,
               boxprops=dict(facecolor=colors_cv[i], alpha=0.6),
               medianprops=dict(color="black", linewidth=2))
    ax.scatter([i] * len(scores), scores, color=colors_cv[i], zorder=5, s=60)
    ax.text(i, scores.mean() + 0.001, f"μ={scores.mean():.4f}",
            ha="center", fontsize=9)
ax.set_xticks(positions)
ax.set_xticklabels(cv_scores.keys(), fontsize=11)
ax.set_ylabel("Cross-validation R²")
ax.set_title("Cross-validation Score Comparison (5-Fold)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "11_cross_validation_scores.png")
plt.close(fig)
log.info("Saved: 11_cross_validation_scores.png")

# ── Plot 12: Model R² bar chart ───────────────
log.info("Plot 12: Model comparison bar chart")
metrics_to_plot = ["Train R²", "Validation R²", "Test R²"]
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df))
width = 0.25
bar_colors = ["#2196F3", "#4CAF50", "#FF9800"]
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, comparison_df[metric], width,
           label=metric, color=bar_colors[i], alpha=0.85, edgecolor="white")
ax.set_xticks(x + width)
ax.set_xticklabels(comparison_df["Model"], fontsize=12)
ax.set_ylabel("R² Score")
ax.set_ylim(0, 1.05)
ax.set_title("Model Comparison – R² Scores", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", fontsize=8, padding=2)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "12_model_comparison_bar.png")
plt.close(fig)
log.info("Saved: 12_model_comparison_bar.png")

# ── Plot 13: RMSE comparison ──────────────────
log.info("Plot 13: RMSE comparison")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison_df))
width = 0.35
ax.bar(x - width/2, comparison_df["Train RMSE"], width, label="Train RMSE",
       color="#E91E63", alpha=0.8)
ax.bar(x + width/2, comparison_df["Test RMSE"], width, label="Test RMSE",
       color="#9C27B0", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"], fontsize=12)
ax.set_ylabel("RMSE")
ax.set_title("Model Comparison – RMSE", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "13_model_comparison_rmse.png")
plt.close(fig)
log.info("Saved: 13_model_comparison_rmse.png")

# ── Plot 14: MAE comparison ───────────────────
log.info("Plot 14: MAE comparison")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, comparison_df["Train MAE"], width, label="Train MAE",
       color="#00BCD4", alpha=0.8)
ax.bar(x + width/2, comparison_df["Test MAE"], width, label="Test MAE",
       color="#FF5722", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"], fontsize=12)
ax.set_ylabel("MAE")
ax.set_title("Model Comparison – MAE", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "14_model_comparison_mae.png")
plt.close(fig)
log.info("Saved: 14_model_comparison_mae.png")

# ── Plot 15: LightGBM training loss curve ────
log.info("Plot 15: LightGBM train/validation loss curve")
best_params_lgbm = lgbm_search.best_params_.copy()
lgbm_loss_model = lgb.LGBMRegressor(
    **best_params_lgbm, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
)
eval_result = {}
X_tl, X_vl, y_tl, y_vl = train_test_split(
    X_train_lgbm, y_train, test_size=0.15, random_state=RANDOM_SEED
)
callbacks = [lgb.record_evaluation(eval_result), lgb.early_stopping(50, verbose=False)]
lgbm_loss_model.fit(
    X_tl, y_tl,
    eval_set=[(X_tl, y_tl), (X_vl, y_vl)],
    eval_metric="rmse",
    callbacks=callbacks,
)
train_losses = eval_result["training"]["rmse"]
val_losses   = eval_result["valid_1"]["rmse"]

fig, ax = plt.subplots(figsize=(12, 6))
iters = np.arange(1, len(train_losses) + 1)
ax.plot(iters, train_losses, color="#2196F3", label="Train RMSE", linewidth=1.5)
ax.plot(iters, val_losses,   color="#FF5722", label="Val RMSE",   linewidth=1.5)
best_iter = int(np.argmin(val_losses)) + 1
ax.axvline(best_iter, color="red", linestyle="--", linewidth=1.2,
           label=f"Best iter={best_iter}")
ax.set_xlabel("Boosting Iteration")
ax.set_ylabel("RMSE")
ax.set_title("LightGBM – Train vs Validation Loss Curve",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.4)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "15_lgbm_train_val_loss.png")
plt.close(fig)
log.info("Saved: 15_lgbm_train_val_loss.png")

# ── Plot 16: Target distribution ─────────────
log.info("Plot 16: Target distribution")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y.values, bins=100, color="#5C6BC0", edgecolor="none", alpha=0.85)
ax.set_xlabel("BerRating")
ax.set_ylabel("Count")
ax.set_title(f"BerRating Distribution (n={len(y):,})",
             fontsize=14, fontweight="bold")
ax.axvline(y.mean(),   color="red",    linestyle="--", linewidth=1.5,
           label=f"Mean={y.mean():.1f}")
ax.axvline(y.median(), color="orange", linestyle=":",  linewidth=1.5,
           label=f"Median={y.median():.1f}")
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "16_target_distribution.png")
plt.close(fig)
log.info("Saved: 16_target_distribution.png")


# ══════════════════════════════════════════════
# 9. EVALUATION REPORT
# ══════════════════════════════════════════════
log.info("\n" + "=" * 60)
log.info("STEP 9: Writing evaluation report")
log.info("=" * 60)

best_model_name = comparison_df.loc[comparison_df["Test R²"].idxmax(), "Model"]
best_row = comparison_df[comparison_df["Model"] == best_model_name].iloc[0]

report_lines = [
    "=" * 70,
    "IRISH HOME RETROFIT – BER RATING PREDICTION",
    "EVALUATION METRICS REPORT",
    f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 70,
    "",
    "DATASET SUMMARY",
    "-" * 40,
    f"  Total samples     : {len(df):,}",
    f"  Training samples  : {len(X_train):,}",
    f"  Test samples      : {len(X_test):,}",
    f"  Tuning subsample  : {len(X_tune):,}",
    f"  Features (total)  : {X.shape[1]}",
    f"  Categorical       : {len(CAT_COLS)}",
    f"  Numerical         : {len(NUM_COLS)}",
    f"  Target            : {TARGET}",
    f"  Target mean       : {y.mean():.2f}",
    f"  Target std        : {y.std():.2f}",
    f"  Target range      : [{y.min():.0f}, {y.max():.0f}]",
    "",
    "MODEL SELECTION RATIONALE",
    "-" * 40,
    "  1. LightGBM     – GBDT, handles mixed types natively, fastest among boosters",
    "  2. Random Forest – Bagging ensemble, robust to outliers, strong baseline",
    "  3. Ridge         – Regularised linear model, interpretable, captures linear trends",
    "",
    "TRAINING STRATEGY",
    "-" * 40,
    "  • 80/20 stratified train-test split (stratified by BER decile)",
    f"  • Hyperparameter search on {len(X_tune):,}-row stratified subsample",
    "  • Best params refitted on full training set",
    "  • 5-fold cross-validation on full training set for final CV scores",
    "",
    "MODEL COMPARISON",
    "-" * 40,
    comparison_df.to_string(index=False),
    "",
    "CROSS-VALIDATION DETAIL (5-Fold, full training set)",
    "-" * 40,
]
for name, scores in cv_scores.items():
    report_lines.append(
        f"  {name}: {[f'{s:.4f}' for s in scores]}"
        f"  mean={scores.mean():.4f}  std={scores.std():.4f}"
    )

report_lines += [
    "",
    "BEST MODEL",
    "-" * 40,
    f"  Name          : {best_model_name}",
    f"  Train R²      : {best_row['Train R²']:.4f}",
    f"  Validation R² : {best_row['Validation R²']:.4f}",
    f"  Test R²       : {best_row['Test R²']:.4f}",
    f"  Test RMSE     : {best_row['Test RMSE']:.4f}",
    f"  Test MAE      : {best_row['Test MAE']:.4f}",
    f"  Verdict       : {best_row['Verdict']}",
    "",
    "TOP 10 FEATURES (LightGBM importance)",
    "-" * 40,
    lgbm_fi.head(10).to_string(index=False),
    "",
    "TOP 10 FEATURES (Random Forest importance)",
    "-" * 40,
    rf_fi.head(10).to_string(index=False),
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
report_path = OUTPUT_DIR / "evaluation_report.txt"
report_path.write_text(report_text)
log.info("Saved: evaluation_report.txt")
log.info("\n" + report_text)


# ══════════════════════════════════════════════
# 10. REQUIREMENTS
# ══════════════════════════════════════════════
req_path = Path("requirements.txt")
if not req_path.exists():
    req_path.write_text(
        "lightgbm>=4.0\n"
        "scikit-learn>=1.3\n"
        "pandas>=2.0\n"
        "numpy>=1.24\n"
        "matplotlib>=3.7\n"
        "seaborn>=0.12\n"
        "joblib>=1.3\n"
        "pyarrow>=12.0\n"
        "scipy>=1.11\n"
    )
    log.info("Created: requirements.txt")
else:
    log.info("requirements.txt already exists – skipping")


log.info("\n" + "=" * 60)
log.info("ALL STEPS COMPLETE")
log.info(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
log.info("=" * 60)
