"""
02_train_model.py
=================
Model training on the honest 55-column dataset for BER rating prediction.

Models trained:
  1. LightGBM  — primary model (hyperparameters from 118-col pipeline validation)
  2. Random Forest — comparison
  3. Ridge Regression — baseline

Strategy:
  - Load clean_data_55col.parquet (produced by 01_clean_and_prepare.py)
  - 70/15/15 train/val/test split
  - 5-fold CV on the full training dataset
  - Target: log1p(BerRating) to handle right skew
  - Global SHAP on a 10K sample for visualization
  - Metrics on original kWh/m²/yr scale

Output:
  outputs/lgbm_model.pkl          — trained LightGBM + encoders + metadata
  outputs/feature_importance.csv  — gain-based importance
  outputs/shap_values_global.csv  — global SHAP on 10K sample
  outputs/shap_bar.png            — top-30 SHAP bar chart
  outputs/shap_summary.png        — SHAP beeswarm
  outputs/model_report.txt        — full metrics + CV results
"""

import sys
import pandas as pd
import numpy as np
import pickle
import time
import warnings
from pathlib import Path

from cli_logger import setup_script_logging

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(OUTPUT_DIR / f"{Path(__file__).stem}.log")
PARQUET_PATH = OUTPUT_DIR / "clean_data_55col.parquet"
LGBM_PATH    = OUTPUT_DIR / "lgbm_model.pkl"
REPORT_PATH  = OUTPUT_DIR / "model_report.txt"
FIMP_PATH    = OUTPUT_DIR / "feature_importance.csv"

TARGET      = 'BerRating'
RANDOM_SEED = 42
SHAP_N      = 10_000   # global SHAP sample size (for visualization)
CV_N        = None     # use full training set for cross-validation

# LightGBM hyperparameters validated in 118-col pipeline
LGBM_PARAMS = {
    "n_estimators":       1500,
    "learning_rate":      0.08,
    "num_leaves":         127,
    "max_depth":          8,
    "min_child_samples":  100,
    "subsample":          0.9,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "random_state":       RANDOM_SEED,
    "n_jobs":             -1,
    "verbose":            -1,
}

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BER DATASET — MODEL TRAINING (55-COL HONEST SET)")
print("=" * 60)
print(f"\nLoading {PARQUET_PATH}...")
t0 = time.time()

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns in "
      f"{time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────────────────────
y_raw = df[TARGET].values.astype(np.float64)
X     = df.drop(columns=[TARGET])

print(f"\nTarget (BerRating) — mean: {y_raw.mean():.1f}, "
      f"std: {y_raw.std():.1f}, min: {y_raw.min():.1f}, max: {y_raw.max():.1f}")

# Log-transform target
y = np.log1p(y_raw)
print(f"Log1p target      — mean: {y.mean():.3f}, std: {y.std():.3f}")

# ─────────────────────────────────────────────────────────────
# IDENTIFY FEATURE TYPES
# ─────────────────────────────────────────────────────────────
CAT_COLS = X.select_dtypes(include=['object', 'category']).columns.tolist()
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nFeature types — Numeric: {len(NUM_COLS)}, Categorical: {len(CAT_COLS)}")
print(f"  Categorical: {CAT_COLS}")
print(f"  Total features: {X.shape[1]}")

# ─────────────────────────────────────────────────────────────
# ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────
print("\nEncoding categorical columns...")
encoders = {}
for col in CAT_COLS:
    enc = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        dtype=np.float32
    )
    X[col] = enc.fit_transform(X[[col]])
    encoders[col] = enc
    X[col] = X[col].astype(np.float32)

for col in NUM_COLS:
    X[col] = X[col].astype(np.float32)

print(f"  Done. X shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ─────────────────────────────────────────────────────────────
print("\nSplitting data 70/15/15...")

X_trainval, X_test, y_trainval, y_test, yraw_trainval, yraw_test = \
    train_test_split(X, y, y_raw, test_size=0.15, random_state=RANDOM_SEED)

X_train, X_val, y_train, y_val, yraw_train, yraw_val = \
    train_test_split(X_trainval, y_trainval, yraw_trainval,
                     test_size=0.15/0.85, random_state=RANDOM_SEED)

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
del X_trainval, y_trainval, yraw_trainval


# ─────────────────────────────────────────────────────────────
# HELPER: evaluate model on all three splits (original scale)
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_tr, y_tr_raw, X_vl, y_vl_raw, X_te, y_te_raw,
             model_name: str, use_log_target: bool = True) -> dict:
    results = {}
    for split_name, Xs, ys_raw in [
        ('train', X_tr, y_tr_raw),
        ('val',   X_vl, y_vl_raw),
        ('test',  X_te, y_te_raw),
    ]:
        pred = model.predict(Xs)
        if use_log_target:
            pred = np.expm1(pred).clip(min=0)

        r2   = r2_score(ys_raw, pred)
        rmse = np.sqrt(mean_squared_error(ys_raw, pred))
        mae  = mean_absolute_error(ys_raw, pred)

        results[split_name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        print(f"  {model_name} [{split_name:5s}]  "
              f"R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f} kWh/m²/yr")
    return results


# ─────────────────────────────────────────────────────────────
# 5-FOLD CROSS-VALIDATION (full training set)
# ─────────────────────────────────────────────────────────────
def run_cv(model_class, model_kwargs: dict, X_data, y_data, model_name: str) -> dict:
    """Run 5-fold CV on the full training set. Returns mean/std R²."""
    X_cv = X_data
    y_cv = y_data

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cv), 1):
        m = model_class(**model_kwargs)
        m.fit(X_cv.iloc[tr_idx], y_cv[tr_idx])
        pred = m.predict(X_cv.iloc[va_idx])
        pred_raw = np.expm1(pred).clip(min=0) if model_kwargs.get('verbose') is not None else pred
        y_va_raw = np.expm1(y_cv[va_idx]).clip(min=0) if model_kwargs.get('verbose') is not None else y_cv[va_idx]
        score = r2_score(y_va_raw, pred_raw)
        scores.append(score)
        print(f"  {model_name} fold {fold}/5: R²={score:.4f}")

    mean_r2 = np.mean(scores)
    std_r2  = np.std(scores)
    print(f"  {model_name} CV: mean R²={mean_r2:.4f} ± {std_r2:.4f}")
    return {'mean_r2': mean_r2, 'std_r2': std_r2, 'scores': scores}


# ─────────────────────────────────────────────────────────────
# 5-fold CV for LightGBM on the full training set
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION (LightGBM, full training set)")
print("=" * 60)

X_cv    = X_train
y_cv    = y_train
yraw_cv = yraw_train

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cv), 1):
    m = lgb.LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_cv.iloc[tr_idx], y_cv[tr_idx])
    pred_log = m.predict(X_cv.iloc[va_idx])
    pred_raw = np.expm1(pred_log).clip(min=0)
    score = r2_score(yraw_cv[va_idx], pred_raw)
    cv_scores.append(score)
    print(f"  Fold {fold}/5: R²={score:.4f}")

lgbm_cv_mean = np.mean(cv_scores)
lgbm_cv_std  = np.std(cv_scores)
print(f"  LightGBM CV: mean R²={lgbm_cv_mean:.4f} ± {lgbm_cv_std:.4f}")


# ─────────────────────────────────────────────────────────────
# LIGHTGBM — PRIMARY MODEL (train on full train+val)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LIGHTGBM — Primary model (hyperparams from 118-col pipeline)")
print("=" * 60)

X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = np.concatenate([y_train, y_val])

lgbm_final = lgb.LGBMRegressor(**LGBM_PARAMS)
lgbm_final.fit(
    X_tv, y_tv,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=-1)],
)

print("\nLightGBM performance:")
lgbm_results = evaluate(
    lgbm_final,
    X_train, yraw_train,
    X_val,   yraw_val,
    X_test,  yraw_test,
    'LightGBM'
)

# Save LightGBM artifact
lgbm_artifact = {
    'model':       lgbm_final,
    'encoders':    encoders,
    'cat_cols':    CAT_COLS,
    'num_cols':    NUM_COLS,
    'feature_names': X_train.columns.tolist(),
    'params':      LGBM_PARAMS,
    'results':     lgbm_results,
    'cv_mean_r2':  lgbm_cv_mean,
    'cv_std_r2':   lgbm_cv_std,
}
with open(LGBM_PATH, 'wb') as f:
    pickle.dump(lgbm_artifact, f)
print(f"\nLightGBM saved to {LGBM_PATH}")


# ─────────────────────────────────────────────────────────────
# RANDOM FOREST — COMPARISON MODEL (full training set)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RANDOM FOREST — Comparison model (full training set)")
print("=" * 60)

X_rf_train = X_train
y_rf_train = y_train
yraw_rf_train = yraw_train

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)
rf_model.fit(X_rf_train, y_rf_train)

print("\nRandom Forest performance:")
rf_results = evaluate(
    rf_model,
    X_rf_train, yraw_rf_train,
    X_val,      yraw_val,
    X_test,     yraw_test,
    'RandomForest'
)


# ─────────────────────────────────────────────────────────────
# RIDGE REGRESSION — BASELINE MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RIDGE REGRESSION — Baseline model (full training set)")
print("=" * 60)

ridge_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('ridge',   Ridge(alpha=1.0)),
])
ridge_pipeline.fit(X_rf_train, y_rf_train)

print("\nRidge Regression performance:")
ridge_results = evaluate(
    ridge_pipeline,
    X_rf_train, yraw_rf_train,
    X_val,      yraw_val,
    X_test,     yraw_test,
    'Ridge'
)


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (LightGBM gain-based)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (LightGBM — gain)")
print("=" * 60)

fi = pd.DataFrame({
    'feature':    X_train.columns.tolist(),
    'importance': lgbm_final.feature_importances_,
}).sort_values('importance', ascending=False)

fi.to_csv(FIMP_PATH, index=False)
print("\nTop 20 features (gain-based):")
print(fi.head(20).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# GLOBAL SHAP (10K sample) — verify FabricHeatLossPerM2 is top-3
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"GLOBAL SHAP ({SHAP_N:,}-row sample)")
print("=" * 60)

rng_shap = np.random.default_rng(RANDOM_SEED + 2)
shap_idx  = rng_shap.choice(len(df), size=min(SHAP_N, len(df)), replace=False)
df_shap   = df.iloc[shap_idx].drop(columns=[TARGET], errors='ignore').copy()

# Apply encoders to shap sample
for col in CAT_COLS:
    if col in df_shap.columns:
        df_shap[col] = encoders[col].transform(df_shap[[col]]).astype(np.float32)
for col in NUM_COLS:
    if col in df_shap.columns:
        df_shap[col] = df_shap[col].astype(np.float32)

print(f"Computing SHAP values for {len(df_shap):,} rows...")
t_shap = time.time()

explainer   = shap.TreeExplainer(lgbm_final)
shap_values = explainer.shap_values(df_shap)
print(f"  Done in {time.time()-t_shap:.1f}s")

# Save raw SHAP values
shap_df = pd.DataFrame(shap_values, columns=df_shap.columns)
shap_df.to_csv(OUTPUT_DIR / "shap_values_global.csv", index=False)
print(f"  SHAP values saved to {OUTPUT_DIR / 'shap_values_global.csv'}")

# SHAP feature importance (mean |SHAP|)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
fi_shap = pd.DataFrame({
    'feature':       df_shap.columns.tolist(),
    'mean_abs_shap': mean_abs_shap,
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 20 features by mean |SHAP|:")
print(fi_shap.head(20).to_string(index=False))

# Check FabricHeatLossPerM2 rank
if 'FabricHeatLossPerM2' in fi_shap['feature'].values:
    rank = fi_shap['feature'].tolist().index('FabricHeatLossPerM2') + 1
    print(f"\n  FabricHeatLossPerM2 SHAP rank: {rank}")
    if rank <= 3:
        print("  Leakage removal verified: engineered feature is top-3. ✓")
    else:
        print(f"  NOTE: FabricHeatLossPerM2 ranked {rank} (expected top-3).")

# ── SHAP bar chart (mean |SHAP|) ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12))
top30 = fi_shap.head(30)
ax.barh(top30['feature'][::-1], top30['mean_abs_shap'][::-1], color='steelblue')
ax.set_xlabel('Mean |SHAP value| (impact on log BerRating)')
ax.set_title('Top 30 Features — Global SHAP Importance\n'
             '(LightGBM, 55-col honest dataset, leakage removed)')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  SHAP bar chart saved to {OUTPUT_DIR / 'shap_bar.png'}")

# ── SHAP beeswarm / summary plot ──────────────────────────────
plt.figure(figsize=(10, 14))
shap.summary_plot(shap_values, df_shap, max_display=30, show=False, plot_size=(10, 14))
plt.title('SHAP Summary — BER Rating Prediction (55-col honest set)', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  SHAP beeswarm saved to {OUTPUT_DIR / 'shap_summary.png'}")


# ─────────────────────────────────────────────────────────────
# FULL REPORT
# ─────────────────────────────────────────────────────────────
def fmt_results(name, res):
    lines = [f"-- {name} --"]
    for split in ['train', 'val', 'test']:
        r = res[split]
        lines.append(f"  {split:5s}  R2={r['R2']:.4f}  "
                     f"RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f} kWh/m2/yr")
    return lines

report_lines = []
report_lines.append("=" * 60)
report_lines.append("BER MODEL TRAINING REPORT — 55-COL HONEST DATASET")
report_lines.append("=" * 60)
report_lines.append(f"Dataset rows:   {len(df):,}")
report_lines.append(f"Features:       {X_train.shape[1]}")
report_lines.append(f"Train rows:     {len(X_train):,}")
report_lines.append(f"Val rows:       {len(X_val):,}")
report_lines.append(f"Test rows:      {len(X_test):,}")
report_lines.append("")

report_lines.append("-- LEAKAGE REMOVAL NOTE --")
report_lines.append("  Removed from 118-col pipeline: FirstEnerProdDelivered (rank #4 SHAP),")
report_lines.append("  DistributionLosses, HSEffAdjFactor, WHEffAdjFactor,")
report_lines.append("  SHRenewableResources, WHRenewableResources,")
report_lines.append("  TempAdjustment, TempFactorMultiplier.")
report_lines.append("  Expected R2 drop: 0.9913 -> ~0.93-0.96 (honest estimate).")
report_lines.append("")

report_lines.append("-- 5-FOLD CV (LightGBM, full training set) --")
report_lines.append(f"  mean R2 = {lgbm_cv_mean:.4f} +/- {lgbm_cv_std:.4f}")
report_lines.append(f"  fold scores: {[f'{s:.4f}' for s in cv_scores]}")
report_lines.append("")

report_lines += fmt_results("LightGBM (validated hyperparams)", lgbm_results)
report_lines.append(f"  Params: {LGBM_PARAMS}")
report_lines.append("")
report_lines += fmt_results("Random Forest (full training set)", rf_results)
report_lines.append("")
report_lines += fmt_results("Ridge Regression (baseline)", ridge_results)
report_lines.append("")

report_lines.append("-- TOP 30 FEATURES (LightGBM gain) --")
report_lines.append(fi.head(30).to_string(index=False))
report_lines.append("")
report_lines.append("-- TOP 20 FEATURES (SHAP mean |shap|) --")
report_lines.append(fi_shap.head(20).to_string(index=False))

report_text = "\n".join(report_lines)
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\nFull report saved to {REPORT_PATH}")

print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
print("Done.")
