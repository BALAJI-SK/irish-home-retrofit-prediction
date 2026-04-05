"""
Model Comparison Table
======================
Produces a full comparison table with:
  - Train R²,  Validation R² (CV mean),  Test R²
  - Train RMSE, Test RMSE
  - CV mean score, CV std
  - Training time
  - Verdict: Overfitting / Underfitting / Well-fitted

Metrics are taken from the existing ml_pipeline run.
Train RMSE is derived as:  RMSE_train = std(y_train) * sqrt(1 - Train_R²)
which is mathematically equivalent to computing it directly.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("model_analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset to get y_train std (needed for Train RMSE)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "46_Col_final_with_county.parquet"
TARGET     = "BerRating"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

from sklearn.model_selection import train_test_split

print("Loading dataset to compute y_train statistics ...")
df = pd.read_parquet(DATA_PATH, columns=[TARGET])

# Same outlier trim as ml_pipeline.py
q_lo = df[TARGET].quantile(0.01)
q_hi = df[TARGET].quantile(0.99)
df   = df[(df[TARGET] >= q_lo) & (df[TARGET] <= q_hi)]

y = df[TARGET].values
strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
_, _, y_train, _ = train_test_split(
    y, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = strata,
)
y_train_std = np.std(y_train)
print(f"  y_train std = {y_train_std:.4f} kWh/m²/yr  (n={len(y_train):,})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Results from ml_pipeline.py run  (ml_pipeline_report.txt)
#    CV R² is from RandomizedSearchCV on 200K-row stratified subsample of train.
#    It serves as the Validation R² — CV held-out folds are never used in fitting.
# ─────────────────────────────────────────────────────────────────────────────
pipeline_results = {
    "Ridge": {
        "CV R² (mean)":   0.836,
        "CV R² (std)":    0.001,
        "Train R²":       0.8368,
        "Test R²":        0.8369,
        "Test RMSE":      50.7,
        "Tune Time (s)":  17.8,
        "Train Time (s)": 7.4,
    },
    "Random Forest": {
        "CV R² (mean)":   0.9327,
        "CV R² (std)":    0.001,
        "Train R²":       0.99,
        "Test R²":        0.947,
        "Test RMSE":      28.91,
        "Tune Time (s)":  2519.0,
        "Train Time (s)": 424.7,
    },
    "LightGBM": {
        "CV R² (mean)":   0.9603,
        "CV R² (std)":    0.001,
        "Train R²":       0.9712,
        "Test R²":        0.9655,
        "Test RMSE":      23.34,
        "Tune Time (s)":  309.9,
        "Train Time (s)": 19.4,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Derive missing metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_train_rmse(train_r2, y_std):
    """RMSE_train = std(y) * sqrt(1 - R²_train)  — exact identity."""
    return round(y_std * np.sqrt(max(0.0, 1.0 - train_r2)), 2)

def verdict(train_r2, test_r2):
    gap = train_r2 - test_r2
    if test_r2 < 0.80:
        return "Underfitting"
    if gap > 0.02 and train_r2 > 0.95:
        return "Overfitting"
    return "Well-fitted"

rows = []
for model, res in pipeline_results.items():
    train_r2 = res["Train R²"]
    test_r2  = res["Test R²"]
    cv_mean  = res["CV R² (mean)"]
    cv_std   = res["CV R² (std)"]
    train_rmse = compute_train_rmse(train_r2, y_train_std)
    v = verdict(train_r2, test_r2)

    rows.append({
        "Model":            model,
        "Train R²":         train_r2,
        "Validation R²\n(CV mean)": cv_mean,
        "Test R²":          test_r2,
        "Train RMSE\n(kWh/m²/yr)":  train_rmse,
        "Test RMSE\n(kWh/m²/yr)":   res["Test RMSE"],
        "CV Mean Score":    cv_mean,
        "CV Std":           cv_std,
        "Train Time (s)":   res["Train Time (s)"],
        "Verdict":          v,
    })

df_table = pd.DataFrame(rows).set_index("Model")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Print table
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("=" * 90)
print("MODEL COMPARISON TABLE — BER Rating Prediction (Regression)")
print("=" * 90)
print(df_table.to_string())
print("=" * 90)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save CSV (flat column names)
# ─────────────────────────────────────────────────────────────────────────────
flat_rows = []
for model, res in pipeline_results.items():
    train_r2 = res["Train R²"]
    test_r2  = res["Test R²"]
    cv_mean  = res["CV R² (mean)"]
    cv_std   = res["CV R² (std)"]
    train_rmse = compute_train_rmse(train_r2, y_train_std)

    flat_rows.append({
        "Model":             model,
        "Train R2":          train_r2,
        "Validation R2 (CV mean)": cv_mean,
        "Test R2":           test_r2,
        "Train RMSE":        train_rmse,
        "Test RMSE":         res["Test RMSE"],
        "CV Mean Score":     cv_mean,
        "CV Std":            cv_std,
        "Train Time (s)":    res["Train Time (s)"],
        "Verdict":           verdict(train_r2, test_r2),
    })

csv_df = pd.DataFrame(flat_rows)
csv_path = OUTPUT_DIR / "model_comparison_full_table.csv"
csv_df.to_csv(csv_path, index=False)
print(f"\nSaved CSV  → {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save text report
# ─────────────────────────────────────────────────────────────────────────────
col_widths = {
    "Model":         14,
    "Train R²":       9,
    "Val R² (CV)":    12,
    "Test R²":         9,
    "Train RMSE":     11,
    "Test RMSE":      10,
    "CV Mean":         9,
    "CV Std":          8,
    "Train Time":     12,
    "Verdict":        14,
}
headers = list(col_widths.keys())
sep = "  ".join("-" * w for w in col_widths.values())

lines = [
    "=" * 110,
    "Irish Home Retrofit — BER Rating Prediction",
    "Model Comparison Table",
    "=" * 110,
    "",
    "Notes:",
    "  • Validation R² = CV mean R² from 5-fold RandomizedSearchCV on 200K-row stratified",
    "    subsample of the training set.  CV folds are held-out during each fit — this IS",
    "    proper out-of-sample validation, not training error.",
    "  • Train RMSE derived as  std(y_train) × √(1 − Train R²)  (exact algebraic identity).",
    f"  • std(y_train) = {y_train_std:.2f} kWh/m²/yr",
    "  • Verdict rules:",
    "      Overfitting  : Train R² > 0.95  AND  (Train R² − Test R²) > 0.02",
    "      Underfitting : Test R² < 0.80",
    "      Well-fitted  : otherwise",
    "",
    sep,
    "  ".join(h.ljust(w) for h, w in zip(headers, col_widths.values())),
    sep,
]

for row in flat_rows:
    verdict_str = row["Verdict"]
    line = "  ".join([
        row["Model"].ljust(14),
        f"{row['Train R2']:.4f}".ljust(9),
        f"{row['Validation R2 (CV mean)']:.4f}".ljust(12),
        f"{row['Test R2']:.4f}".ljust(9),
        f"{row['Train RMSE']:.2f}".ljust(11),
        f"{row['Test RMSE']:.2f}".ljust(10),
        f"{row['CV Mean Score']:.4f}".ljust(9),
        f"{row['CV Std']:.4f}".ljust(8),
        f"{row['Train Time (s)']:.1f}s".ljust(12),
        verdict_str.ljust(14),
    ])
    lines.append(line)

lines += [sep, ""]

# best model
best = max(flat_rows, key=lambda r: r["Test R2"])
lines += [
    f"Best model by Test R²: {best['Model']}",
    f"  Train R²      : {best['Train R2']}",
    f"  Validation R² : {best['Validation R2 (CV mean)']}",
    f"  Test R²       : {best['Test R2']}",
    f"  Train RMSE    : {best['Train RMSE']} kWh/m²/yr",
    f"  Test RMSE     : {best['Test RMSE']} kWh/m²/yr",
    f"  Verdict       : {best['Verdict']}",
    "",
    "=" * 110,
]

report_text = "\n".join(lines)
print("\n" + report_text)
report_path = OUTPUT_DIR / "model_comparison_full_table_report.txt"
report_path.write_text(report_text)
print(f"Saved report → {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot
# ─────────────────────────────────────────────────────────────────────────────
models      = [r["Model"] for r in flat_rows]
colors      = {"LightGBM": "#2196F3", "Random Forest": "#4CAF50", "Ridge": "#FF9800"}
bar_colors  = [colors[m] for m in models]

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle(
    "Model Comparison — BER Rating Prediction (Regression)\n"
    "Dataset: 46_Col_final_with_county.parquet  |  1,324,561 rows  |  80/20 split",
    fontsize=12, fontweight="bold"
)

metric_specs = [
    (
        [r["Train R2"] for r in flat_rows],
        [r["Validation R2 (CV mean)"] for r in flat_rows],
        [r["Test R2"] for r in flat_rows],
        "R² Comparison",
    ),
    (
        [r["Train RMSE"] for r in flat_rows],
        None,
        [r["Test RMSE"] for r in flat_rows],
        "RMSE Comparison (kWh/m²/yr)",
    ),
    (
        [r["CV Mean Score"] for r in flat_rows],
        [r["CV Std"] for r in flat_rows],
        None,
        "CV Mean Score ± Std",
    ),
    (
        [r["Train Time (s)"] for r in flat_rows],
        None,
        None,
        "Training Time (seconds)",
    ),
]

# ── Plot 0: grouped R² bar (Train / Val / Test)
ax = axes[0]
x       = np.arange(len(models))
width   = 0.25
train_r2 = [r["Train R2"] for r in flat_rows]
val_r2   = [r["Validation R2 (CV mean)"] for r in flat_rows]
test_r2  = [r["Test R2"] for r in flat_rows]

b1 = ax.bar(x - width, train_r2, width, label="Train R²",      color="#1565C0", alpha=0.85)
b2 = ax.bar(x,          val_r2,  width, label="Val R² (CV)",   color="#42A5F5", alpha=0.85)
b3 = ax.bar(x + width,  test_r2, width, label="Test R²",       color="#0D47A1", alpha=0.85)

for bar, val in zip(list(b1)+list(b2)+list(b3), train_r2+val_r2+test_r2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_title("R² Comparison\n(Train / Validation / Test)", fontsize=10, fontweight="bold")
ax.set_ylabel("R²")
ax.legend(fontsize=8)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# ── Plot 1: RMSE
ax = axes[1]
train_rmse = [r["Train RMSE"] for r in flat_rows]
test_rmse  = [r["Test RMSE"]  for r in flat_rows]
b1 = ax.bar(x - width/2, train_rmse, width, label="Train RMSE", color="#E65100", alpha=0.85)
b2 = ax.bar(x + width/2, test_rmse,  width, label="Test RMSE",  color="#FF8F00", alpha=0.85)
for bar, val in zip(list(b1)+list(b2), train_rmse+test_rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_title("RMSE Comparison\n(Train / Test, kWh/m²/yr)", fontsize=10, fontweight="bold")
ax.set_ylabel("RMSE (kWh/m²/yr)")
ax.legend(fontsize=8)

# ── Plot 2: CV mean ± std
ax = axes[2]
cv_means = [r["CV Mean Score"] for r in flat_rows]
cv_stds  = [r["CV Std"] for r in flat_rows]
bars = ax.bar(models, cv_means, color=bar_colors, edgecolor="white",
              yerr=cv_stds, capsize=6)
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_title("CV Mean R² ± Std\n(5-fold, 200K subsample)", fontsize=10, fontweight="bold")
ax.set_ylabel("CV R²")

# ── Plot 3: Training time + verdict annotation
ax = axes[3]
train_times = [r["Train Time (s)"] for r in flat_rows]
bars = ax.bar(models, train_times, color=bar_colors, edgecolor="white")
for bar, val, row in zip(bars, train_times, flat_rows):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Verdict below bar label
    v = row["Verdict"]
    vc = "#388E3C" if v == "Well-fitted" else ("#D32F2F" if v == "Overfitting" else "#F57C00")
    ax.text(bar.get_x() + bar.get_width()/2, -35,
            v, ha="center", va="top", fontsize=8, fontweight="bold", color=vc)
ax.set_title("Training Time (seconds)\n+ Final Verdict", fontsize=10, fontweight="bold")
ax.set_ylabel("Seconds")
ax.set_ylim(-50, max(train_times) * 1.25)

plt.tight_layout()
plot_path = OUTPUT_DIR / "model_comparison_full_table.png"
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved plot  → {plot_path}")

print("\nDone.")
