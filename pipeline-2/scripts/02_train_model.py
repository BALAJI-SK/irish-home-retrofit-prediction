"""
02_train_model.py
=================
Model training for Irish SEAI BER energy rating prediction.

Strategy (backed by Papers 2, 4, 5, 7):
  - Primary model : LightGBM  (fastest, best accuracy on tabular BER data)
  - Comparison    : XGBoost   (cross-validation benchmark)
  - Split         : 70% train / 15% val / 15% test  (Paper 7)
  - Tuning        : RandomizedSearchCV on validation set
  - Target        : log1p(BerRating) to handle right skew (Paper 1)
  - Metrics       : R², RMSE, MAE (on original kWh/m²/yr scale)

Output:
  outputs/lgbm_model.pkl       — trained LightGBM pipeline
  outputs/xgb_model.pkl        — trained XGBoost pipeline
  outputs/model_report.txt     — full metrics + feature importance
  outputs/feature_importance.csv
"""

import pandas as pd
import numpy as np
import pickle
import time
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path("outputs")
PARQUET_PATH  = OUTPUT_DIR / "clean_data.parquet"
LGBM_PATH     = OUTPUT_DIR / "lgbm_model.pkl"
XGB_PATH      = OUTPUT_DIR / "xgb_model.pkl"
REPORT_PATH   = OUTPUT_DIR / "model_report.txt"
FIMP_PATH     = OUTPUT_DIR / "feature_importance.csv"

TARGET      = 'BerRating'
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BER DATASET — MODEL TRAINING")
print("=" * 60)
print(f"\nLoading clean parquet from {PARQUET_PATH}...")
t0 = time.time()

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns in "
      f"{time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────────────────────
# Separate features and target
y_raw = df[TARGET].values.astype(np.float64)
X     = df.drop(columns=[TARGET])

print(f"\nTarget (BerRating) — mean: {y_raw.mean():.1f}, "
      f"std: {y_raw.std():.1f}, min: {y_raw.min():.1f}, max: {y_raw.max():.1f}")

# Log-transform target (Paper 1: right-skewed distribution)
# Using log1p(BerRating) — safe because BER_MIN >= 0 after cleaning
y = np.log1p(y_raw)
print(f"Log1p target    — mean: {y.mean():.3f}, std: {y.std():.3f}")

# ─────────────────────────────────────────────────────────────
# IDENTIFY FEATURE TYPES
# ─────────────────────────────────────────────────────────────
CAT_COLS = X.select_dtypes(include=['object', 'category']).columns.tolist()
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nFeature types — Numeric: {len(NUM_COLS)}, Categorical: {len(CAT_COLS)}")
print(f"  Categorical: {CAT_COLS}")

# ─────────────────────────────────────────────────────────────
# ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────
# LightGBM and XGBoost both work with integer-encoded categoricals.
# We use OrdinalEncoder with handle_unknown='use_encoded_value'.
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

# Ensure all numeric columns are float32
for col in NUM_COLS:
    X[col] = X[col].astype(np.float32)

print(f"  Done. X shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)  — Paper 7
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
# HELPER: evaluate model and print metrics
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_tr, y_tr_log, y_tr_raw,
                       X_vl, y_vl_log, y_vl_raw,
                       X_te, y_te_log, y_te_raw,
                       model_name: str) -> dict:
    """Evaluate on all three splits, report on original scale."""
    results = {}
    for split_name, Xs, ys_log, ys_raw in [
        ('train', X_tr, y_tr_log, y_tr_raw),
        ('val',   X_vl, y_vl_log, y_vl_raw),
        ('test',  X_te, y_te_log, y_te_raw),
    ]:
        pred_log = model.predict(Xs)
        pred_raw = np.expm1(pred_log)            # inverse of log1p
        pred_raw = np.clip(pred_raw, 0, None)    # no negative BER

        r2   = r2_score(ys_raw, pred_raw)
        rmse = np.sqrt(mean_squared_error(ys_raw, pred_raw))
        mae  = mean_absolute_error(ys_raw, pred_raw)

        results[split_name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        print(f"  {model_name} [{split_name:5s}]  "
              f"R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f} kWh/m²/yr")
    return results


# ─────────────────────────────────────────────────────────────
# ① LIGHTGBM — PRIMARY MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("① LIGHTGBM — Primary model")
print("=" * 60)

# Step A: Quick baseline with sensible defaults
print("\nStep A: Baseline LightGBM...")
lgbm_base = lgb.LGBMRegressor(
    n_estimators    = 1000,
    learning_rate   = 0.05,
    max_depth       = -1,
    num_leaves      = 127,
    min_child_samples = 50,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 0.1,
    random_state    = RANDOM_SEED,
    n_jobs          = -1,
    verbose         = -1,
)
lgbm_base.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=-1)],
)
lgbm_base_results = evaluate(
    lgbm_base, X_train, y_train, yraw_train,
    X_val, y_val, yraw_val, X_test, y_test, yraw_test,
    'LGBM_base'
)

# Step B: Hyperparameter search on a 200K subsample
# Running CV on 945K rows × 30 candidates × 3 folds = 90 full fits → too slow.
# Best-practice: find optimal hyperparameters on a representative subsample,
# then retrain the winner on the full training set.
print("\nStep B: Randomized hyperparameter search (LightGBM, 200K subsample)...")
SEARCH_N = 200_000
rng_search = np.random.default_rng(RANDOM_SEED)
search_idx = rng_search.choice(len(X_train), size=min(SEARCH_N, len(X_train)), replace=False)
X_search = X_train.iloc[search_idx]
y_search = y_train[search_idx]
print(f"  Search subsample: {len(X_search):,} rows  "
      f"(from {len(X_train):,} train rows)")

lgbm_param_dist = {
    'n_estimators'      : [500, 800, 1000, 1500],
    'learning_rate'     : [0.01, 0.03, 0.05, 0.08],
    'num_leaves'        : [63, 127, 255, 511],
    'max_depth'         : [-1, 6, 8, 10],
    'min_child_samples' : [20, 50, 100, 200],
    'subsample'         : [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree'  : [0.6, 0.7, 0.8, 0.9],
    'reg_alpha'         : [0.0, 0.05, 0.1, 0.5],
    'reg_lambda'        : [0.0, 0.1, 0.5, 1.0],
}

lgbm_search = RandomizedSearchCV(
    lgb.LGBMRegressor(
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    ),
    param_distributions=lgbm_param_dist,
    n_iter=20,          # 20 configs × 2 folds × 200K rows — ~15 min total
    scoring='neg_root_mean_squared_error',
    cv=2,
    random_state=RANDOM_SEED,
    n_jobs=1,
    verbose=1,
    refit=True,
)
lgbm_search.fit(X_search, y_search)

print(f"\nBest LGBM params: {lgbm_search.best_params_}")
lgbm_best = lgbm_search.best_estimator_

# Retrain best on train+val with early stopping
print("\nRetraining best LGBM on train+val combined...")
X_tv = pd.concat([X_train, X_val], ignore_index=True)
y_tv = np.concatenate([y_train, y_val])

lgbm_final = lgb.LGBMRegressor(
    **lgbm_search.best_params_,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1,
)
lgbm_final.fit(X_tv, y_tv)

print("\nFinal LightGBM performance:")
lgbm_final_results = evaluate(
    lgbm_final,
    X_train, y_train, yraw_train,
    X_val,   y_val,   yraw_val,
    X_test,  y_test,  yraw_test,
    'LGBM_final'
)

# Save LightGBM
lgbm_artifact = {
    'model':       lgbm_final,
    'encoders':    encoders,
    'cat_cols':    CAT_COLS,
    'num_cols':    NUM_COLS,
    'best_params': lgbm_search.best_params_,
    'results':     lgbm_final_results,
}
with open(LGBM_PATH, 'wb') as f:
    pickle.dump(lgbm_artifact, f)
print(f"\nLightGBM saved to {LGBM_PATH}")


# ─────────────────────────────────────────────────────────────
# ② XGBOOST — COMPARISON MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("② XGBOOST — Comparison model")
print("=" * 60)

print("\nStep A: Baseline XGBoost...")
xgb_base = xgb.XGBRegressor(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 10,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
    tree_method      = 'hist',   # memory-efficient histogram method
    early_stopping_rounds = 50,
    eval_metric      = 'rmse',
    verbosity        = 0,
)
xgb_base.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
xgb_base_results = evaluate(
    xgb_base, X_train, y_train, yraw_train,
    X_val, y_val, yraw_val, X_test, y_test, yraw_test,
    'XGB_base'
)

print("\nStep B: Randomized hyperparameter search (XGBoost, 200K subsample)...")
# Reuse same subsample indices as LightGBM search
xgb_param_dist = {
    'n_estimators'      : [500, 800, 1000],
    'learning_rate'     : [0.01, 0.03, 0.05],
    'max_depth'         : [4, 6, 8],
    'min_child_weight'  : [5, 10, 20],
    'subsample'         : [0.7, 0.8, 0.9],
    'colsample_bytree'  : [0.7, 0.8, 0.9],
    'reg_alpha'         : [0.0, 0.1, 0.5],
    'reg_lambda'        : [0.5, 1.0, 2.0],
    'gamma'             : [0.0, 0.1, 0.5],
}

xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(
        tree_method='hist', random_state=RANDOM_SEED,
        n_jobs=-1, verbosity=0
    ),
    param_distributions=xgb_param_dist,
    n_iter=15,
    scoring='neg_root_mean_squared_error',
    cv=2,
    random_state=RANDOM_SEED,
    n_jobs=1,
    verbose=1,
    refit=True,
)
xgb_search.fit(X_search, y_search)

print(f"\nBest XGB params: {xgb_search.best_params_}")

# Retrain on train+val
print("\nRetraining best XGBoost on train+val combined...")
xgb_final = xgb.XGBRegressor(
    **xgb_search.best_params_,
    tree_method='hist',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0,
)
xgb_final.fit(X_tv, y_tv)

print("\nFinal XGBoost performance:")
xgb_final_results = evaluate(
    xgb_final,
    X_train, y_train, yraw_train,
    X_val,   y_val,   yraw_val,
    X_test,  y_test,  yraw_test,
    'XGB_final'
)

# Save XGBoost
xgb_artifact = {
    'model':       xgb_final,
    'encoders':    encoders,
    'cat_cols':    CAT_COLS,
    'num_cols':    NUM_COLS,
    'best_params': xgb_search.best_params_,
    'results':     xgb_final_results,
}
with open(XGB_PATH, 'wb') as f:
    pickle.dump(xgb_artifact, f)
print(f"\nXGBoost saved to {XGB_PATH}")


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

print("\nTop 30 features:")
print(fi.head(30).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# FULL REPORT
# ─────────────────────────────────────────────────────────────
report_lines = []
report_lines.append("=" * 60)
report_lines.append("BER MODEL TRAINING REPORT")
report_lines.append("=" * 60)
report_lines.append(f"Dataset rows:   {len(df):,}")
report_lines.append(f"Features:       {X_train.shape[1]}")
report_lines.append(f"Train rows:     {len(X_train):,}")
report_lines.append(f"Val rows:       {len(X_val):,}")
report_lines.append(f"Test rows:      {len(X_test):,}")
report_lines.append("")

def fmt_results(name, res):
    lines = [f"── {name} ──"]
    for split in ['train', 'val', 'test']:
        r = res[split]
        lines.append(f"  {split:5s}  R²={r['R2']:.4f}  "
                     f"RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f} kWh/m²/yr")
    return lines

report_lines += fmt_results("LightGBM baseline", lgbm_base_results)
report_lines.append("")
report_lines += fmt_results("LightGBM FINAL (tuned)", lgbm_final_results)
report_lines.append(f"  Best params: {lgbm_search.best_params_}")
report_lines.append("")
report_lines += fmt_results("XGBoost baseline", xgb_base_results)
report_lines.append("")
report_lines += fmt_results("XGBoost FINAL (tuned)", xgb_final_results)
report_lines.append(f"  Best params: {xgb_search.best_params_}")
report_lines.append("")
report_lines.append("── TOP 30 FEATURES (LightGBM gain) ─────────────────────")
report_lines.append(fi.head(30).to_string(index=False))

report_text = "\n".join(report_lines)
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\nFull report saved to {REPORT_PATH}")

print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
print("Done.")
