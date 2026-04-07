"""
03_shap_and_retrofit.py
=======================
SHAP explainability + retrofit intervention analysis for the BER model.

What this script does:
  1. Loads the trained LightGBM model and the clean parquet dataset.
  2. Computes SHAP values on a 5,000-row sample (memory-safe).
  3. Saves global feature importance plots + beeswarm summary.
  4. Runs a RETROFIT SIMULATION:
     For each dwelling in a 2,000-row sample it simulates the BER
     improvement from 7 common retrofit interventions (individually
     and as a combined deep-retrofit package) and saves the results.

Retrofit interventions modelled:
  A. Roof insulation upgrade    → UValueRoof = 0.13 (best-practice)
  B. Wall insulation upgrade    → UValueWall = 0.18 (external insulation)
  C. Window upgrade             → UValueWindow = 1.2 (double-low-e)
  D. Heat pump installation     → MainSpaceHeatingFuel = 'Electricity'
                                   + HSMainSystemEfficiency = 300.0 (COP 3)
  E. Solar water heating        → SolarHotWaterHeating = 'YES'
                                   + has_hw_cylinder = 1
  F. Airtightness improvement   → PercentageDraughtStripped = 100.0
                                   + PermeabilityTestResult = 0.0
  G. LED lighting upgrade       → LowEnergyLightingPercent = 100.0
  H. Deep retrofit (A+B+C+D+G) — combined package

Output:
  outputs/shap_summary.png          — global beeswarm plot
  outputs/shap_bar.png              — top-30 mean |SHAP| bar chart
  outputs/shap_values.csv           — raw SHAP values (5K sample)
  outputs/retrofit_results.csv      — per-dwelling retrofit impact
  outputs/retrofit_summary.txt      — aggregate statistics
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import time
from pathlib import Path

import shap
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — safe on all systems
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("outputs")
PARQUET_PATH = OUTPUT_DIR / "clean_data.parquet"
LGBM_PATH    = OUTPUT_DIR / "lgbm_model.pkl"

SHAP_N       = 5_000    # rows for SHAP computation (memory-safe)
RETROFIT_N   = 2_000    # rows for retrofit simulation

RANDOM_SEED  = 42
TARGET       = 'BerRating'


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BER DATASET — SHAP + RETROFIT ANALYSIS")
print("=" * 60)

print(f"\nLoading model from {LGBM_PATH}...")
with open(LGBM_PATH, 'rb') as f:
    artifact = pickle.load(f)

model    = artifact['model']
encoders = artifact['encoders']
cat_cols = artifact['cat_cols']
num_cols = artifact['num_cols']

print(f"Loading data from {PARQUET_PATH}...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Data: {df.shape[0]:,} rows × {df.shape[1]} cols")


# ─────────────────────────────────────────────────────────────
# HELPER: prepare a feature matrix from a DataFrame
# (applies the same ordinal encoding as training)
# ─────────────────────────────────────────────────────────────
def prepare_X(df_input: pd.DataFrame) -> pd.DataFrame:
    """Apply saved encoders to produce the feature matrix for prediction."""
    X = df_input.drop(columns=[TARGET], errors='ignore').copy()

    for col in cat_cols:
        if col in X.columns:
            X[col] = encoders[col].transform(X[[col]])
            X[col] = X[col].astype(np.float32)

    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].astype(np.float32)

    return X


def predict_ber(X: pd.DataFrame) -> np.ndarray:
    """Predict BerRating (original scale) from an encoded feature matrix."""
    log_pred = model.predict(X)
    return np.expm1(log_pred).clip(min=0)


# ─────────────────────────────────────────────────────────────
# SECTION 1: SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1: SHAP Feature Importance")
print("=" * 60)

# Sample SHAP_N rows
rng = np.random.default_rng(RANDOM_SEED)
idx_shap = rng.choice(len(df), size=min(SHAP_N, len(df)), replace=False)
df_shap  = df.iloc[idx_shap].copy()

X_shap = prepare_X(df_shap)

print(f"\nComputing SHAP values for {len(X_shap):,} rows...")
t0 = time.time()

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)
print(f"  Done in {time.time()-t0:.1f}s")

# Convert to DataFrame and save
shap_df = pd.DataFrame(shap_values, columns=X_shap.columns)
shap_df.to_csv(OUTPUT_DIR / "shap_values.csv", index=False)
print(f"  SHAP values saved to {OUTPUT_DIR / 'shap_values.csv'}")

# ── Global bar chart (mean |SHAP|) ────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
fi_shap = pd.DataFrame({
    'feature':    X_shap.columns.tolist(),
    'mean_abs_shap': mean_abs_shap,
}).sort_values('mean_abs_shap', ascending=False)

fig, ax = plt.subplots(figsize=(10, 12))
top30 = fi_shap.head(30)
ax.barh(top30['feature'][::-1], top30['mean_abs_shap'][::-1], color='steelblue')
ax.set_xlabel('Mean |SHAP value| (impact on log BerRating)')
ax.set_title('Top 30 Features — Global SHAP Importance\n(LightGBM, BER Rating Prediction)')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Bar chart saved to {OUTPUT_DIR / 'shap_bar.png'}")

# ── Beeswarm / summary plot ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 14))
shap.summary_plot(
    shap_values, X_shap,
    max_display=30,
    show=False,
    plot_size=(10, 14),
)
plt.title('SHAP Summary — BER Rating Prediction', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Beeswarm plot saved to {OUTPUT_DIR / 'shap_summary.png'}")

# Print top 20
print("\nTop 20 features by mean |SHAP| (log-scale impact):")
print(fi_shap.head(20).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# SECTION 2: RETROFIT SIMULATION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Retrofit Intervention Simulation")
print("=" * 60)

# Sample RETROFIT_N dwellings
idx_ret = rng.choice(len(df), size=min(RETROFIT_N, len(df)), replace=False)
df_ret  = df.iloc[idx_ret].copy().reset_index(drop=True)

# Baseline BER prediction
X_base   = prepare_X(df_ret)
ber_base = predict_ber(X_base)

print(f"\nBaseline BER — mean: {ber_base.mean():.1f}, "
      f"median: {np.median(ber_base):.1f} kWh/m²/yr")


# ─────────────────────────────────────────────────────────────
# DEFINE RETROFIT SCENARIOS
# ─────────────────────────────────────────────────────────────
# Each scenario is a dict of {column: new_value} overrides.
# For categorical columns we must encode them the same way as training.

SCENARIOS = {
    'A_RoofInsulation': {
        'UValueRoof': 0.13,
    },
    'B_WallInsulation': {
        'UValueWall': 0.18,
        'FirstWallUValue': 0.18,
    },
    'C_WindowUpgrade': {
        'UValueWindow': 1.2,
    },
    'D_HeatPump': {
        'MainSpaceHeatingFuel': 'Electricity',
        'MainWaterHeatingFuel': 'Electricity',
        'HSMainSystemEfficiency': 300.0,      # COP ~3 → 300% efficiency
        'WHMainSystemEff': 300.0,
        'IsHeatPump': 1,
    },
    'E_SolarWaterHeating': {
        'SolarHotWaterHeating': 'YES',
        'has_hw_cylinder': 1,
    },
    'F_Airtightness': {
        'PercentageDraughtStripped': 100.0,
        'PermeabilityTestResult': 0.0,
    },
    'G_LEDLighting': {
        'LowEnergyLightingPercent': 100.0,
    },
    'H_DeepRetrofit': {
        # Combines A + B + C + D + G
        'UValueRoof': 0.13,
        'UValueWall': 0.18,
        'FirstWallUValue': 0.18,
        'UValueWindow': 1.2,
        'MainSpaceHeatingFuel': 'Electricity',
        'MainWaterHeatingFuel': 'Electricity',
        'HSMainSystemEfficiency': 300.0,
        'WHMainSystemEff': 300.0,
        'IsHeatPump': 1,
        'LowEnergyLightingPercent': 100.0,
        'PercentageDraughtStripped': 100.0,
    },
}


def apply_scenario(df_input: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Apply retrofit overrides to a copy of df_input and encode."""
    df_mod = df_input.copy()
    for col, val in overrides.items():
        if col in df_mod.columns:
            df_mod[col] = val
    # Re-engineer dependent features
    if 'WallArea' in df_mod.columns and 'WindowArea' in df_mod.columns:
        df_mod['WindowToWallRatio'] = np.where(
            df_mod['WallArea'] > 0,
            df_mod['WindowArea'] / df_mod['WallArea'], 0.0
        ).astype(np.float32)
    if all(c in df_mod.columns for c in
           ['UValueWall','WallArea','UValueRoof','RoofArea',
            'UValueFloor','FloorArea','UValueWindow','WindowArea',
            'UvalueDoor','DoorArea']):
        df_mod['FabricHeatLossProxy'] = (
            df_mod['UValueWall']   * df_mod['WallArea']   +
            df_mod['UValueRoof']   * df_mod['RoofArea']   +
            df_mod['UValueFloor']  * df_mod['FloorArea']  +
            df_mod['UValueWindow'] * df_mod['WindowArea'] +
            df_mod['UvalueDoor']   * df_mod['DoorArea']
        ).astype(np.float32)
        df_mod['FabricHeatLossPerM2'] = np.where(
            df_mod['GroundFloorAreasq_m'] > 0,
            df_mod['FabricHeatLossProxy'] / df_mod['GroundFloorAreasq_m'],
            df_mod['FabricHeatLossProxy']
        ).astype(np.float32)
        df_mod['HasRoofInsulation'] = (df_mod['UValueRoof'] <= 0.16).astype(np.int8)
        df_mod['HasWallInsulation'] = (df_mod['UValueWall'] <= 0.37).astype(np.int8)
        df_mod['HasDoubleGlazing']  = (df_mod['UValueWindow'] <= 2.0).astype(np.int8)

    return prepare_X(df_mod)


# ─────────────────────────────────────────────────────────────
# RUN EACH SCENARIO
# ─────────────────────────────────────────────────────────────
results = pd.DataFrame({'BerRating_baseline': ber_base})
results['DwellingType'] = df_ret['DwellingTypeDescr'].values
results['Year_of_Construction'] = df_ret['Year_of_Construction'].values
results['CountyName'] = df_ret['CountyName'].values

print()
summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("RETROFIT INTERVENTION — AGGREGATE SUMMARY")
summary_lines.append(f"Sample size: {RETROFIT_N:,} dwellings")
summary_lines.append("=" * 60)
summary_lines.append(f"\nBaseline BER: mean={ber_base.mean():.1f}, "
                     f"median={np.median(ber_base):.1f} kWh/m²/yr")
summary_lines.append("")

for scenario_name, overrides in SCENARIOS.items():
    X_mod  = apply_scenario(df_ret, overrides)
    ber_mod = predict_ber(X_mod)

    savings_abs = ber_base - ber_mod
    savings_pct = 100.0 * savings_abs / np.where(ber_base > 0, ber_base, 1.0)

    results[f'BerRating_{scenario_name}'] = ber_mod
    results[f'Saving_abs_{scenario_name}'] = savings_abs
    results[f'Saving_pct_{scenario_name}'] = savings_pct

    line = (f"  {scenario_name:<22s} | "
            f"New BER: {ber_mod.mean():>6.1f} | "
            f"Saving: {savings_abs.mean():>6.1f} kWh/m²/yr "
            f"({savings_pct.mean():>5.1f}%) | "
            f"Median saving: {np.median(savings_abs):>6.1f}")
    print(line)
    summary_lines.append(line)

# Save per-dwelling results
results.to_csv(OUTPUT_DIR / "retrofit_results.csv", index=False)
print(f"\nPer-dwelling results → {OUTPUT_DIR / 'retrofit_results.csv'}")

# ─────────────────────────────────────────────────────────────
# RETROFIT ANALYSIS BY DWELLING TYPE
# ─────────────────────────────────────────────────────────────
summary_lines.append("\n── BY DWELLING TYPE (Deep Retrofit H) ─────────────────")
if 'Saving_abs_H_DeepRetrofit' in results.columns:
    by_type = results.groupby('DwellingType')['Saving_abs_H_DeepRetrofit'].agg(
        ['mean', 'median', 'count']
    ).round(1).sort_values('mean', ascending=False)
    summary_lines.append(by_type.to_string())

# ─────────────────────────────────────────────────────────────
# RETROFIT ANALYSIS BY AGE BAND
# ─────────────────────────────────────────────────────────────
summary_lines.append("\n── BY AGE BAND (Deep Retrofit H) ──────────────────────")
if 'AgeBand' in df_ret.columns and 'Saving_abs_H_DeepRetrofit' in results.columns:
    results['AgeBand'] = df_ret['AgeBand'].values
    by_age = results.groupby('AgeBand')['Saving_abs_H_DeepRetrofit'].agg(
        ['mean', 'median', 'count']
    ).round(1).sort_values('mean', ascending=False)
    summary_lines.append(by_age.to_string())

# ─────────────────────────────────────────────────────────────
# RETROFIT BAR CHART
# ─────────────────────────────────────────────────────────────
scenario_names = list(SCENARIOS.keys())
mean_savings   = [results[f'Saving_abs_{s}'].mean() for s in scenario_names]
labels         = [s.split('_', 1)[1].replace('_', ' ') for s in scenario_names]

fig, ax = plt.subplots(figsize=(10, 6))
colors  = ['#2196F3' if 'Deep' not in s else '#E53935' for s in scenario_names]
bars    = ax.bar(labels, mean_savings, color=colors, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Mean BER Saving (kWh/m²/yr)')
ax.set_title('Expected BER Improvement by Retrofit Intervention\n'
             '(LightGBM prediction, counterfactual simulation)')
ax.set_xticklabels(labels, rotation=30, ha='right')
for bar, val in zip(bars, mean_savings):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "retrofit_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Retrofit bar chart → {OUTPUT_DIR / 'retrofit_bar.png'}")

# ─────────────────────────────────────────────────────────────
# SINGLE DWELLING EXAMPLE
# ─────────────────────────────────────────────────────────────
summary_lines.append("\n── SINGLE DWELLING EXAMPLE ────────────────────────────")
ex = df_ret.iloc[[0]]
ex_base_ber = predict_ber(prepare_X(ex))[0]
summary_lines.append(
    f"Dwelling: {ex['DwellingTypeDescr'].iloc[0]}, "
    f"{ex['Year_of_Construction'].iloc[0]}, "
    f"{ex['CountyName'].iloc[0]}"
)
summary_lines.append(f"Baseline BER: {ex_base_ber:.1f} kWh/m²/yr")

for scenario_name, overrides in SCENARIOS.items():
    X_mod_ex = apply_scenario(ex, overrides)
    ber_mod_ex = predict_ber(X_mod_ex)[0]
    saving = ex_base_ber - ber_mod_ex
    summary_lines.append(
        f"  {scenario_name:<22s} → {ber_mod_ex:>7.1f}  "
        f"(saving: {saving:>+7.1f} kWh/m²/yr)"
    )

# Save summary
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(OUTPUT_DIR / "retrofit_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"\nSummary saved to {OUTPUT_DIR / 'retrofit_summary.txt'}")

print("\nDone. All SHAP and retrofit outputs saved to outputs/")
