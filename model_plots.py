"""
Comprehensive Model Diagnostic Plots
=====================================
Generates all major industry-standard plots for the BER Rating regression task:

  1. Feature Importance (LightGBM — gain & split)
  2. Correlation Heatmap (top numerical features vs target)
  3. Prediction vs Actual (LightGBM on held-out test set)
  4. Residual Plot (residuals vs predicted)
  5. Error Distribution (histogram + Q-Q plot)
  6. Train vs Validation Loss Curve (LightGBM per-iteration RMSE)
  7. Learning Curve (LightGBM R² vs training-set size)
  8. Cross-Validation Score Plot (all 3 models)
  9. Model Comparison Bar Chart (Train / Val / Test R² + RMSE)

LightGBM is retrained here (≈20s) to capture eval_set curves and fresh predictions.
RF and Ridge metrics come from the saved ml_pipeline run.
"""

import warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "46_Col_final_with_county.parquet"
OUTPUT_DIR  = Path("model_analysis_outputs")
TARGET      = "BerRating"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Best params from ml_pipeline.py RandomizedSearchCV run
LGB_BEST_PARAMS = dict(
    subsample        = 1.0,
    reg_lambda       = 1.0,
    reg_alpha        = 0.1,
    num_leaves       = 63,
    n_estimators     = 800,
    min_child_samples= 50,
    max_depth        = -1,
    learning_rate    = 0.1,
    colsample_bytree = 0.9,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
    verbose          = -1,
)

# Model comparison metrics from ml_pipeline run (used for summary charts)
PIPELINE_RESULTS = {
    "Ridge": dict(
        train_r2=0.8368, val_r2=0.8360, test_r2=0.8369,
        train_rmse=50.68, test_rmse=50.70,
        cv_mean=0.8360, cv_std=0.001, train_time=7.4,
    ),
    "Random Forest": dict(
        train_r2=0.9900, val_r2=0.9327, test_r2=0.9470,
        train_rmse=12.54, test_rmse=28.91,
        cv_mean=0.9327, cv_std=0.001, train_time=424.7,
    ),
    "LightGBM": dict(
        train_r2=0.9712, val_r2=0.9603, test_r2=0.9655,
        train_rmse=21.29, test_rmse=23.34,
        cv_mean=0.9603, cv_std=0.001, train_time=19.4,
    ),
}

OUTPUT_DIR.mkdir(exist_ok=True)
PALETTE = {"LightGBM": "#2196F3", "Random Forest": "#4CAF50", "Ridge": "#FF9800"}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & prepare data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Loading dataset ...")
print("=" * 65)

df = pd.read_parquet(DATA_PATH)
DROP_COLS = ["SHRenewableResources", "WHRenewableResources",
             "HSEffAdjFactor", "WHEffAdjFactor"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

q_lo, q_hi = df[TARGET].quantile(0.01), df[TARGET].quantile(0.99)
df = df[(df[TARGET] >= q_lo) & (df[TARGET] <= q_hi)]
print(f"  Rows after outlier trim : {len(df):,}")

X = df.drop(columns=[TARGET])
y = df[TARGET].values

cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"  Features: {len(num_cols)} numerical, {len(cat_cols)} categorical")

# Train / Test split
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strata
)

# Validation split from train (for loss curves)
strata_train = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=strata_train
)
print(f"  Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Preprocessing (ordinal-encode cats)
# ─────────────────────────────────────────────────────────────────────────────
enc = ColumnTransformer(
    [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
    remainder="passthrough"
)
X_tr_enc   = enc.fit_transform(X_tr)
X_val_enc  = enc.transform(X_val)
X_test_enc = enc.transform(X_test)
X_train_enc = enc.transform(X_train)

# Feature names after encoding
feat_names = cat_cols + num_cols

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train LightGBM with eval_set callbacks (for loss curves)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Training LightGBM (with eval_set for loss curve) ...")
print("=" * 65)

train_losses, val_losses = [], []

class LossCallback:
    def __init__(self):
        self.train_rmse = []
        self.val_rmse   = []
    def __call__(self, env):
        for item in env.evaluation_result_list:
            name, metric, val, _ = item
            if name == "training":
                self.train_rmse.append(val)
            elif name == "valid_1":
                self.val_rmse.append(val)

lgb_dtrain = lgb.Dataset(X_tr_enc,   label=y_tr,   feature_name=feat_names)
lgb_dval   = lgb.Dataset(X_val_enc,  label=y_val,  reference=lgb_dtrain,
                          feature_name=feat_names)
lgb_params = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":        LGB_BEST_PARAMS["num_leaves"],
    "learning_rate":     LGB_BEST_PARAMS["learning_rate"],
    "max_depth":         LGB_BEST_PARAMS["max_depth"],
    "min_child_samples": LGB_BEST_PARAMS["min_child_samples"],
    "subsample":         LGB_BEST_PARAMS["subsample"],
    "colsample_bytree":  LGB_BEST_PARAMS["colsample_bytree"],
    "reg_alpha":         LGB_BEST_PARAMS["reg_alpha"],
    "reg_lambda":        LGB_BEST_PARAMS["reg_lambda"],
    "n_jobs":            -1,
    "verbosity":         -1,
    "seed":              RANDOM_SEED,
}

evals_result = {}
cb = LossCallback()
t0 = time.time()
booster = lgb.train(
    lgb_params,
    lgb_dtrain,
    num_boost_round   = LGB_BEST_PARAMS["n_estimators"],
    valid_sets        = [lgb_dtrain, lgb_dval],
    valid_names       = ["training", "valid_1"],
    callbacks         = [lgb.record_evaluation(evals_result), lgb.log_evaluation(period=100)],
)
lgb_time = time.time() - t0
print(f"  Training time: {lgb_time:.1f}s")

train_rmse_curve = evals_result["training"]["rmse"]
val_rmse_curve   = evals_result["valid_1"]["rmse"]

# Final predictions on full test set (retrain on full X_train for fairness)
lgb_dtrain_full = lgb.Dataset(X_train_enc, label=y_train, feature_name=feat_names)
booster_full = lgb.train(
    lgb_params, lgb_dtrain_full,
    num_boost_round = LGB_BEST_PARAMS["n_estimators"],
    callbacks       = [lgb.log_evaluation(period=200)],
)
y_pred_train = booster_full.predict(X_train_enc)
y_pred_test  = booster_full.predict(X_test_enc)

train_r2   = r2_score(y_train, y_pred_train)
test_r2    = r2_score(y_test,  y_pred_test)
test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
residuals  = y_test - y_pred_test
print(f"  Full-train → Test  R²={test_r2:.4f}  RMSE={test_rmse:.2f}")

# Feature importances
fi_gain  = pd.Series(booster_full.feature_importance(importance_type="gain"),
                     index=feat_names).sort_values(ascending=False)
fi_split = pd.Series(booster_full.feature_importance(importance_type="split"),
                     index=feat_names).sort_values(ascending=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Learning curve (R² vs training set size, LightGBM only)
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing learning curve (LightGBM, 5 sizes) ...")
lc_fracs  = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
lc_train, lc_val = [], []

for frac in lc_fracs:
    n = max(500, int(frac * len(X_train)))
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X_train), n, replace=False)
    Xf, yf = X_train_enc[idx], y_train[idx]
    strata_f = pd.qcut(yf, q=5, labels=False, duplicates="drop")
    Xf_tr, Xf_val, yf_tr, yf_val = train_test_split(
        Xf, yf, test_size=0.2, random_state=RANDOM_SEED, stratify=strata_f
    )
    ds_tr  = lgb.Dataset(Xf_tr,  label=yf_tr,  feature_name=feat_names)
    ds_val = lgb.Dataset(Xf_val, label=yf_val, reference=ds_tr)
    er = {}
    lgb.train(lgb_params, ds_tr, num_boost_round=400,
              valid_sets=[ds_tr, ds_val], valid_names=["train", "val"],
              callbacks=[lgb.record_evaluation(er), lgb.log_evaluation(period=9999)])
    t_rmse = er["train"]["rmse"][-1]
    v_rmse = er["val"]["rmse"][-1]
    lc_train.append(t_rmse)
    lc_val.append(v_rmse)
    print(f"  {frac*100:5.0f}%  n={n:>8,}  train_rmse={t_rmse:.2f}  val_rmse={v_rmse:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Correlation data
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing correlations ...")
sample_n = min(100_000, len(df))
df_sample = df.sample(sample_n, random_state=RANDOM_SEED)
num_df    = df_sample[num_cols + [TARGET]].copy()
# Drop cols with zero variance
num_df    = num_df.loc[:, num_df.std() > 0]
corr_with_target = (
    num_df.corr()[TARGET]
    .drop(TARGET)
    .abs()
    .sort_values(ascending=False)
    .head(18)
)
top_feats = corr_with_target.index.tolist()
corr_matrix = num_df[top_feats + [TARGET]].corr()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ALL PLOTS
# ─────────────────────────────────────────────────────────────────────────────
PLOT_SAMPLE = 20_000          # points in scatter / residual plots
rng = np.random.RandomState(RANDOM_SEED)
idx_s = rng.choice(len(y_test), min(PLOT_SAMPLE, len(y_test)), replace=False)
y_test_s  = y_test[idx_s]
y_pred_s  = y_pred_test[idx_s]
resid_s   = y_test_s - y_pred_s

TITLE_STYLE = dict(fontsize=13, fontweight="bold")
LABEL_STYLE = dict(fontsize=10)

# ── helper ──────────────────────────────────────────────────────────────────
def save(name):
    p = OUTPUT_DIR / name
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {p}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Feature Importance (Gain + Split side by side)
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/9] Feature Importance ...")
TOP_N = 20
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("LightGBM — Feature Importance\n(Best Model: Test R²=0.9655)",
             **TITLE_STYLE)

for ax, (fi, imp_type, color) in zip(axes, [
    (fi_gain.head(TOP_N),  "Gain  (predictive power)",  "#1565C0"),
    (fi_split.head(TOP_N), "Split (usage frequency)",   "#00796B"),
]):
    fi_sorted = fi.sort_values(ascending=True)
    bars = ax.barh(fi_sorted.index, fi_sorted.values, color=color, alpha=0.85,
                   edgecolor="white")
    for bar, val in zip(bars, fi_sorted.values):
        ax.text(bar.get_width() + fi_sorted.max() * 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:,.0f}", va="center", fontsize=7.5)
    ax.set_title(f"Top {TOP_N} Features by {imp_type}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Importance Score", **LABEL_STYLE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
save("plot1_feature_importance.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Correlation Heatmap
# ════════════════════════════════════════════════════════════════════════════
print("[2/9] Correlation Heatmap ...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Correlation Analysis — Numerical Features vs BER Rating",
             **TITLE_STYLE)

# Left: heatmap of top features
import matplotlib.colors as mcolors
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
cmap = plt.cm.RdBu_r
im = axes[0].imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
axes[0].set_xticks(range(len(corr_matrix.columns)))
axes[0].set_yticks(range(len(corr_matrix.columns)))
axes[0].set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=7.5)
axes[0].set_yticklabels(corr_matrix.columns, fontsize=7.5)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        v = corr_matrix.values[i, j]
        axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if abs(v) > 0.6 else "black")
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
axes[0].set_title("Pearson Correlation Matrix\n(Top 18 features + target)", fontsize=11, fontweight="bold")

# Right: bar chart of |correlation with target|
corr_full = (
    num_df.corr()[TARGET].drop(TARGET)
    .reindex(corr_with_target.index)
)
colors_bar = ["#D32F2F" if v < 0 else "#1565C0" for v in corr_full.values]
corr_sorted = corr_full.sort_values()
colors_sorted = ["#D32F2F" if v < 0 else "#1565C0" for v in corr_sorted.values]
bars = axes[1].barh(corr_sorted.index, corr_sorted.values,
                    color=colors_sorted, alpha=0.85, edgecolor="white")
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Pearson r with BER Rating", **LABEL_STYLE)
axes[1].set_title("Feature Correlation with Target\n(+ positive, − negative)",
                  fontsize=11, fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
save("plot2_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Prediction vs Actual
# ════════════════════════════════════════════════════════════════════════════
print("[3/9] Prediction vs Actual ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("LightGBM — Prediction vs Actual  (BER Rating kWh/m²/yr)",
             **TITLE_STYLE)

mn, mx = min(y_test_s.min(), y_pred_s.min()), max(y_test_s.max(), y_pred_s.max())

# Scatter
ax = axes[0]
sc = ax.scatter(y_test_s, y_pred_s, alpha=0.15, s=3, c="#1565C0", rasterized=True)
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual BER Rating", **LABEL_STYLE)
ax.set_ylabel("Predicted BER Rating", **LABEL_STYLE)
ax.set_title(f"Scatter Plot  (n={PLOT_SAMPLE:,})\nTest R²={test_r2:.4f}  RMSE={test_rmse:.2f}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# 2D Hexbin density
ax = axes[1]
hb = ax.hexbin(y_test_s, y_pred_s, gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
ax.plot([mn, mx], [mn, mx], "b--", linewidth=1.5, label="Perfect fit")
cb = plt.colorbar(hb, ax=ax)
cb.set_label("log(count)", fontsize=9)
ax.set_xlabel("Actual BER Rating", **LABEL_STYLE)
ax.set_ylabel("Predicted BER Rating", **LABEL_STYLE)
ax.set_title("Density Plot (log scale)\nShows prediction concentration",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save("plot3_prediction_vs_actual.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Residual Plots
# ════════════════════════════════════════════════════════════════════════════
print("[4/9] Residual Plots ...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("LightGBM — Residual Analysis  (Regression Diagnostics)",
             **TITLE_STYLE)

# Residuals vs Predicted
ax = axes[0]
ax.scatter(y_pred_s, resid_s, alpha=0.15, s=3, c="#00796B", rasterized=True)
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Predicted Values", **LABEL_STYLE)
ax.set_ylabel("Residuals (Actual − Predicted)", **LABEL_STYLE)
ax.set_title("Residuals vs Predicted\n(Random scatter = good fit)",
             fontsize=11, fontweight="bold")
# Add ±1 RMSE bands
rmse_s = np.sqrt(mean_squared_error(y_test_s, y_pred_s))
ax.axhline(+rmse_s, color="orange", linewidth=1, linestyle=":", label=f"+RMSE ({rmse_s:.1f})")
ax.axhline(-rmse_s, color="orange", linewidth=1, linestyle=":", label=f"−RMSE ({rmse_s:.1f})")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Scale-Location plot (sqrt(|residuals|) vs predicted)
ax = axes[1]
sqrt_abs_resid = np.sqrt(np.abs(resid_s))
ax.scatter(y_pred_s, sqrt_abs_resid, alpha=0.15, s=3, c="#7B1FA2", rasterized=True)
# Smoothed trend line
order = np.argsort(y_pred_s)
from scipy.ndimage import uniform_filter1d
smooth = uniform_filter1d(sqrt_abs_resid[order], size=len(order)//20)
ax.plot(y_pred_s[order], smooth, "r-", linewidth=2, label="Trend")
ax.set_xlabel("Predicted Values", **LABEL_STYLE)
ax.set_ylabel("√|Residuals|", **LABEL_STYLE)
ax.set_title("Scale-Location Plot\n(Flat trend = homoscedasticity)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Residuals vs Actual
ax = axes[2]
ax.scatter(y_test_s, resid_s, alpha=0.15, s=3, c="#E65100", rasterized=True)
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Actual Values", **LABEL_STYLE)
ax.set_ylabel("Residuals", **LABEL_STYLE)
ax.set_title("Residuals vs Actual\n(Reveals systematic bias)",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save("plot4_residual_plots.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Error Distribution
# ════════════════════════════════════════════════════════════════════════════
print("[5/9] Error Distribution ...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("LightGBM — Error Distribution Analysis", **TITLE_STYLE)

# Histogram
ax = axes[0]
ax.hist(resid_s, bins=80, color="#1565C0", alpha=0.8, edgecolor="white", density=True)
mu, sigma = resid_s.mean(), resid_s.std()
xr = np.linspace(resid_s.min(), resid_s.max(), 300)
ax.plot(xr, stats.norm.pdf(xr, mu, sigma), "r-", linewidth=2.5, label=f"N(μ={mu:.2f}, σ={sigma:.2f})")
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Residuals (kWh/m²/yr)", **LABEL_STYLE)
ax.set_ylabel("Density", **LABEL_STYLE)
ax.set_title(f"Residual Histogram\nμ={mu:.2f}  σ={sigma:.2f}  MAE={np.abs(resid_s).mean():.2f}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Q-Q plot
ax = axes[1]
(osm, osr), (slope, intercept, r) = stats.probplot(resid_s, dist="norm")
ax.scatter(osm, osr, alpha=0.3, s=4, c="#00796B", rasterized=True, label="Quantiles")
ql = np.array([osm[0], osm[-1]])
ax.plot(ql, slope * ql + intercept, "r-", linewidth=2, label=f"Normal line  R={r:.4f}")
ax.set_xlabel("Theoretical Quantiles", **LABEL_STYLE)
ax.set_ylabel("Sample Quantiles", **LABEL_STYLE)
ax.set_title("Q-Q Plot\n(Straight line = normally distributed residuals)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Absolute percentage error CDF
ax = axes[2]
ape = np.abs(resid_s) / np.abs(y_test_s) * 100
ape_sorted = np.sort(ape)
cdf = np.arange(1, len(ape_sorted)+1) / len(ape_sorted)
ax.plot(ape_sorted, cdf * 100, color="#1565C0", linewidth=2)
for pct in [10, 20]:
    threshold = np.percentile(ape_sorted, pct*5)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90th percentile")
ax.fill_between(ape_sorted[ape_sorted <= 10], 0, cdf[ape_sorted <= 10]*100,
                alpha=0.2, color="green", label=f"APE ≤ 10%: {(ape<=10).mean()*100:.1f}%")
ax.set_xlabel("Absolute Percentage Error (%)", **LABEL_STYLE)
ax.set_ylabel("Cumulative % of Predictions", **LABEL_STYLE)
ax.set_xlim(0, 60)
mape = ape.mean()
ax.set_title(f"APE Cumulative Distribution\nMAPE={mape:.2f}%",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save("plot5_error_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Train vs Validation Loss Curve (LightGBM)
# ════════════════════════════════════════════════════════════════════════════
print("[6/9] Train vs Validation Loss Curve ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("LightGBM — Train vs Validation Loss Curve (RMSE per Boosting Round)",
             **TITLE_STYLE)

iters = np.arange(1, len(train_rmse_curve) + 1)

# Full curve
ax = axes[0]
ax.plot(iters, train_rmse_curve, color="#1565C0", linewidth=1.5, label="Train RMSE", alpha=0.9)
ax.plot(iters, val_rmse_curve,   color="#D32F2F",  linewidth=1.5, label="Val RMSE",   alpha=0.9)
ax.set_xlabel("Boosting Round", **LABEL_STYLE)
ax.set_ylabel("RMSE (kWh/m²/yr)", **LABEL_STYLE)
ax.set_title("Full Training Curve\n(All 800 rounds)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Zoomed last 400 rounds
ax = axes[1]
zoom_start = 400
ax.plot(iters[zoom_start:], train_rmse_curve[zoom_start:], color="#1565C0",
        linewidth=1.5, label=f"Train RMSE (final={train_rmse_curve[-1]:.2f})")
ax.plot(iters[zoom_start:], val_rmse_curve[zoom_start:],   color="#D32F2F",
        linewidth=1.5, label=f"Val RMSE   (final={val_rmse_curve[-1]:.2f})")
gap = val_rmse_curve[-1] - train_rmse_curve[-1]
ax.set_xlabel("Boosting Round", **LABEL_STYLE)
ax.set_ylabel("RMSE (kWh/m²/yr)", **LABEL_STYLE)
ax.set_title(f"Zoomed — Last {LGB_BEST_PARAMS['n_estimators']-zoom_start} Rounds\n"
             f"Final gap (Val−Train): {gap:.2f} kWh/m²/yr",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save("plot6_train_val_loss_curve.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Learning Curve (R² vs training set size)
# ════════════════════════════════════════════════════════════════════════════
print("[7/9] Learning Curve ...")
lc_sizes = [int(f * len(X_train)) for f in lc_fracs]

# Convert RMSE to R² using std(y_train)
y_std = y_train.std()
def rmse_to_r2(rmse, y_std):
    return max(0, 1 - (rmse**2 / y_std**2))

lc_train_r2 = [rmse_to_r2(r, y_std) for r in lc_train]
lc_val_r2   = [rmse_to_r2(r, y_std) for r in lc_val]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("LightGBM — Learning Curve\n(Performance vs Training Set Size)",
             **TITLE_STYLE)

# RMSE
ax = axes[0]
ax.plot(lc_sizes, lc_train, "o-", color="#1565C0", linewidth=2, markersize=7, label="Train RMSE")
ax.plot(lc_sizes, lc_val,   "s-", color="#D32F2F",  linewidth=2, markersize=7, label="Val RMSE")
ax.fill_between(lc_sizes, lc_train, lc_val, alpha=0.12, color="gray",
                label="Train-Val gap")
ax.set_xscale("log")
ax.set_xlabel("Training Set Size (log scale)", **LABEL_STYLE)
ax.set_ylabel("RMSE (kWh/m²/yr)", **LABEL_STYLE)
ax.set_title("RMSE vs Training Size", fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# R²
ax = axes[1]
ax.plot(lc_sizes, lc_train_r2, "o-", color="#1565C0", linewidth=2, markersize=7, label="Train R²")
ax.plot(lc_sizes, lc_val_r2,   "s-", color="#D32F2F",  linewidth=2, markersize=7, label="Val R²")
ax.fill_between(lc_sizes, lc_train_r2, lc_val_r2, alpha=0.12, color="gray",
                label="Train-Val gap")
ax.set_xscale("log")
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Training Set Size (log scale)", **LABEL_STYLE)
ax.set_ylabel("R²", **LABEL_STYLE)
ax.set_title("R² vs Training Size", fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save("plot7_learning_curve.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Cross-Validation Score Plot (all 3 models)
# ════════════════════════════════════════════════════════════════════════════
print("[8/9] Cross-Validation Score Plot ...")
models_cv = list(PIPELINE_RESULTS.keys())
cv_means  = [PIPELINE_RESULTS[m]["cv_mean"] for m in models_cv]
cv_stds   = [PIPELINE_RESULTS[m]["cv_std"]  for m in models_cv]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Cross-Validation Score Comparison (5-fold RandomizedSearchCV)",
             **TITLE_STYLE)

# Bar chart with error bars
ax = axes[0]
x = np.arange(len(models_cv))
bars = ax.bar(x, cv_means, yerr=cv_stds, capsize=8,
              color=[PALETTE[m] for m in models_cv],
              edgecolor="white", linewidth=1.5, alpha=0.85,
              error_kw=dict(elinewidth=2, ecolor="black"))
for bar, val, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.008,
            f"{val:.4f}\n±{std:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(models_cv, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("CV R² Score", **LABEL_STYLE)
ax.set_title("CV Mean R² ± Std\n(Higher = better, lower std = more stable)",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Train / Val / Test R² grouped
ax = axes[1]
width = 0.25
x2 = np.arange(len(models_cv))
train_r2s = [PIPELINE_RESULTS[m]["train_r2"] for m in models_cv]
val_r2s   = [PIPELINE_RESULTS[m]["val_r2"]   for m in models_cv]
test_r2s  = [PIPELINE_RESULTS[m]["test_r2"]  for m in models_cv]

b1 = ax.bar(x2 - width, train_r2s, width, label="Train R²",    color="#1565C0", alpha=0.85)
b2 = ax.bar(x2,          val_r2s,  width, label="Val R² (CV)", color="#42A5F5", alpha=0.85)
b3 = ax.bar(x2 + width,  test_r2s, width, label="Test R²",     color="#0D47A1", alpha=0.85)
for bars_group, vals in zip([b1, b2, b3], [train_r2s, val_r2s, test_r2s]):
    for bar, val in zip(bars_group, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_xticks(x2); ax.set_xticklabels(models_cv, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("R²", **LABEL_STYLE)
ax.set_title("Train / Validation / Test R²\nAll 3 Models",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
save("plot8_cv_score_plot.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT 9 — Full Model Comparison Dashboard
# ════════════════════════════════════════════════════════════════════════════
print("[9/9] Model Comparison Bar Chart ...")
fig = plt.figure(figsize=(22, 10))
fig.suptitle(
    "Full Model Comparison Dashboard — BER Rating Prediction\n"
    "Dataset: 46_Col_final_with_county.parquet  |  1,324,561 rows  |  80/20 split",
    fontsize=14, fontweight="bold"
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

models_all  = list(PIPELINE_RESULTS.keys())
bar_clrs    = [PALETTE[m] for m in models_all]
x_all       = np.arange(len(models_all))
width_all   = 0.35

VERDICT = {"Ridge": "Well-fitted", "Random Forest": "Overfitting", "LightGBM": "Well-fitted"}
VERDICT_COLOR = {"Well-fitted": "#388E3C", "Overfitting": "#D32F2F", "Underfitting": "#F57C00"}

def metric_bars(ax, vals, title, ylabel, color="#1565C0", fmt=".4f", ylim=None, annot_offset=0.005):
    bars = ax.bar(models_all, vals, color=bar_clrs, edgecolor="white", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + annot_offset,
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    if ylim: ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# 1. Test R²
ax1 = fig.add_subplot(gs[0, 0])
metric_bars(ax1, [PIPELINE_RESULTS[m]["test_r2"]  for m in models_all],
            "Test R²",  "R²", ylim=(0, 1.1))

# 2. Train R²
ax2 = fig.add_subplot(gs[0, 1])
metric_bars(ax2, [PIPELINE_RESULTS[m]["train_r2"] for m in models_all],
            "Train R²", "R²", ylim=(0, 1.1))

# 3. Test RMSE
ax3 = fig.add_subplot(gs[0, 2])
metric_bars(ax3, [PIPELINE_RESULTS[m]["test_rmse"] for m in models_all],
            "Test RMSE", "kWh/m²/yr", fmt=".1f", annot_offset=0.3)

# 4. Train RMSE
ax4 = fig.add_subplot(gs[0, 3])
metric_bars(ax4, [PIPELINE_RESULTS[m]["train_rmse"] for m in models_all],
            "Train RMSE", "kWh/m²/yr", fmt=".1f", annot_offset=0.3)

# 5. CV Mean ± Std
ax5 = fig.add_subplot(gs[1, 0])
cv_m = [PIPELINE_RESULTS[m]["cv_mean"] for m in models_all]
cv_s = [PIPELINE_RESULTS[m]["cv_std"]  for m in models_all]
bars5 = ax5.bar(models_all, cv_m, yerr=cv_s, capsize=6,
                color=bar_clrs, edgecolor="white", linewidth=1.2, alpha=0.85,
                error_kw=dict(elinewidth=2, ecolor="black"))
for bar, v, s in zip(bars5, cv_m, cv_s):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.008,
             f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax5.set_title("CV Mean Score ± Std", fontsize=10, fontweight="bold")
ax5.set_ylabel("R²", fontsize=9); ax5.set_ylim(0, 1.1)
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
ax5.tick_params(axis="x", labelsize=9)

# 6. Training time
ax6 = fig.add_subplot(gs[1, 1])
metric_bars(ax6, [PIPELINE_RESULTS[m]["train_time"] for m in models_all],
            "Training Time", "Seconds", fmt=".1f", annot_offset=2)

# 7. Train R² - Test R² gap
ax7 = fig.add_subplot(gs[1, 2])
gaps = [PIPELINE_RESULTS[m]["train_r2"] - PIPELINE_RESULTS[m]["test_r2"] for m in models_all]
gap_colors = [VERDICT_COLOR[VERDICT[m]] for m in models_all]
bars7 = ax7.bar(models_all, gaps, color=gap_colors, edgecolor="white", linewidth=1.2, alpha=0.85)
for bar, val in zip(bars7, gaps):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax7.axhline(0.02, color="red", linestyle="--", linewidth=1.2, label="Overfit threshold (0.02)")
ax7.set_title("Overfit Gap\n(Train R² − Test R²)", fontsize=10, fontweight="bold")
ax7.set_ylabel("Gap", fontsize=9)
ax7.legend(fontsize=8)
ax7.spines["top"].set_visible(False); ax7.spines["right"].set_visible(False)
ax7.tick_params(axis="x", labelsize=9)

# 8. Verdict table
ax8 = fig.add_subplot(gs[1, 3])
ax8.axis("off")
table_data = [
    ["Model", "Verdict", "Train R²", "Test R²"],
    *[
        [m, VERDICT[m], f"{PIPELINE_RESULTS[m]['train_r2']:.4f}", f"{PIPELINE_RESULTS[m]['test_r2']:.4f}"]
        for m in models_all
    ]
]
tbl = ax8.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc="center", loc="center",
                bbox=[0, 0.1, 1, 0.85])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if row == 0:
        cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
    else:
        m = models_all[row - 1]
        v = VERDICT[m]
        if col == 1:
            cell.set_facecolor(VERDICT_COLOR[v] + "33")
            cell.set_text_props(color=VERDICT_COLOR[v], fontweight="bold")
        else:
            cell.set_facecolor("#F5F5F5" if row % 2 == 0 else "white")
ax8.set_title("Final Verdicts", fontsize=10, fontweight="bold", y=0.98)

save("plot9_model_comparison_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ALL PLOTS SAVED TO:", OUTPUT_DIR)
print("=" * 65)
plots = [
    "plot1_feature_importance.png       — Feature Importance (Gain + Split)",
    "plot2_correlation_heatmap.png      — Correlation Heatmap",
    "plot3_prediction_vs_actual.png     — Prediction vs Actual (scatter + density)",
    "plot4_residual_plots.png           — Residual Diagnostics (3 panels)",
    "plot5_error_distribution.png       — Error Distribution (hist + Q-Q + APE CDF)",
    "plot6_train_val_loss_curve.png     — Train vs Val Loss Curve (LightGBM)",
    "plot7_learning_curve.png           — Learning Curve (R² vs training size)",
    "plot8_cv_score_plot.png            — Cross-Validation Score Plot",
    "plot9_model_comparison_dashboard.png — Full Model Comparison Dashboard",
]
for p in plots:
    print(f"  {p}")
print("=" * 65)
