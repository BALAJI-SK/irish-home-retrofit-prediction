"""
03_scenario_engine.py
=====================
Retrofit scenario engine for the XAI Scenario Planner.

Applies retrofit overrides to building feature vectors, recomputes
derived features (FabricHeatLossProxy, FabricHeatLossPerM2, etc.),
and returns current vs post-retrofit BER predictions.

Validated retrofit measures (from pipeline-2/reports/05_results.md):
  A. Heat Pump          -66 kWh/m2/yr (-27%)
  B. Wall Insulation    -32 kWh/m2/yr (-11%)
  C. Roof Insulation    -13 kWh/m2/yr (-4%)
  D. Window Upgrade     -13 kWh/m2/yr (-5%)
  E. Floor Insulation   (estimated)
  F. Solar Water        -4 kWh/m2/yr
  G. LED Lighting       -1 kWh/m2/yr
  H. Deep Retrofit      -117 kWh/m2/yr (-45%)

Output:
  outputs/retrofit_results.csv     — per-dwelling results (full dataset)
  outputs/retrofit_bar.png         — mean saving per measure bar chart
  outputs/retrofit_summary.txt     — aggregate statistics by age/type
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

from cli_logger import setup_script_logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
OUTPUT_DIR   = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(OUTPUT_DIR / f"{Path(__file__).stem}.log")
CONFIG_DIR   = BASE_DIR / "config"
PARQUET_PATH = OUTPUT_DIR / "clean_data_55col.parquet"
LGBM_PATH    = OUTPUT_DIR / "lgbm_model.pkl"
MEASURES_PATH = CONFIG_DIR / "retrofit_measures.json"

RETROFIT_N  = None   # use full dataset for retrofit simulation
RANDOM_SEED = 42
TARGET      = 'BerRating'

# BER grade boundaries (kWh/m2/yr)
BER_GRADES = [
    (0,   25,  'A1'), (25,  50,  'A2'), (50,  75,  'A3'),
    (75,  100, 'B1'), (100, 125, 'B2'), (125, 150, 'B3'),
    (150, 175, 'C1'), (175, 200, 'C2'), (200, 225, 'C3'),
    (225, 260, 'D1'), (260, 300, 'D2'),
    (300, 340, 'E1'), (340, 380, 'E2'),
    (380, 450, 'F'),
    (450, 9999,'G'),
]


def ber_grade(kwh: float) -> str:
    """Return BER letter grade for a given kWh/m2/yr value."""
    for lo, hi, grade in BER_GRADES:
        if lo <= kwh < hi:
            return grade
    return 'G'


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BER RETROFIT SCENARIO ENGINE")
print("=" * 60)

print(f"\nLoading model from {LGBM_PATH}...")
with open(LGBM_PATH, 'rb') as f:
    artifact = pickle.load(f)

model         = artifact['model']
encoders      = artifact['encoders']
cat_cols      = artifact['cat_cols']
num_cols      = artifact['num_cols']
feature_names = artifact['feature_names']

print(f"Loading data from {PARQUET_PATH}...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Data: {df.shape[0]:,} rows x {df.shape[1]} cols")

print(f"Loading retrofit measures from {MEASURES_PATH}...")
with open(MEASURES_PATH, 'r') as f:
    MEASURES = json.load(f)
print(f"  Loaded {len(MEASURES)} retrofit measures.")


# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────
def prepare_X(df_input: pd.DataFrame) -> pd.DataFrame:
    """Apply saved encoders and return only training feature columns in order."""
    X = df_input.drop(columns=[TARGET], errors='ignore').copy()

    for col in cat_cols:
        if col in X.columns:
            X[col] = encoders[col].transform(X[[col]]).astype(np.float32)

    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].astype(np.float32)

    # Restrict to exactly the columns the model was trained on, in the same order
    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[feature_names]


def predict_ber(X: pd.DataFrame) -> np.ndarray:
    """Predict BerRating (original scale) from encoded feature matrix."""
    log_pred = model.predict(X)
    return np.expm1(log_pred).clip(min=0)


def recompute_derived_features(row: dict) -> dict:
    """
    Recompute FabricHeatLossProxy, FabricHeatLossPerM2, WindowToWallRatio,
    and binary flag features after U-value overrides.

    Formula (Section 3 of implementation plan):
      FabricHeatLossProxy = sum(U * A) for wall, roof, floor, window, door
      FabricHeatLossPerM2 = FabricHeatLossProxy / GroundFloorAreasq_m
    """
    u_wall   = row.get('UValueWall',   0)
    u_roof   = row.get('UValueRoof',   0)
    u_floor  = row.get('UValueFloor',  0)
    u_window = row.get('UValueWindow', 0)
    u_door   = row.get('UvalueDoor',   0)

    a_wall   = row.get('WallArea',   0)
    a_roof   = row.get('RoofArea',   0)
    a_floor  = row.get('FloorArea',  0)
    a_window = row.get('WindowArea', 0)
    a_door   = row.get('DoorArea',   0)

    proxy = (u_wall * a_wall + u_roof * a_roof + u_floor * a_floor +
             u_window * a_window + u_door * a_door)
    row['FabricHeatLossProxy'] = float(proxy)

    gfa = row.get('GroundFloorAreasq_m', 1.0)
    row['FabricHeatLossPerM2'] = float(proxy / gfa) if gfa > 0 else float(proxy)

    # WindowToWallRatio
    row['WindowToWallRatio'] = (
        float(a_window / a_wall) if a_wall > 0 else 0.0
    )

    # Binary flags
    row['HasRoofInsulation'] = int(row.get('UValueRoof', 1.0)  <= 0.16)
    row['HasWallInsulation'] = int(row.get('UValueWall', 1.0)  <= 0.37)
    row['HasDoubleGlazing']  = int(row.get('UValueWindow', 5.0) <= 2.0)

    return row


def apply_overrides_and_recompute(row: dict, overrides: dict,
                                  recompute: bool) -> dict:
    """
    Apply override dict to a home feature dict, optionally recompute
    derived features (FabricHeatLossProxy, FabricHeatLossPerM2, etc.).
    """
    row = {**row, **overrides}
    if recompute:
        row = recompute_derived_features(row)
    return row


def predict_retrofit(home_features: dict, measure_key: str) -> dict:
    """
    Predict baseline and post-retrofit BER for a single home.

    Parameters
    ----------
    home_features : dict
        Raw (pre-encoded) feature dict for the home.
    measure_key : str
        Key into the MEASURES config (e.g. 'heat_pump', 'wall_insulation').

    Returns
    -------
    dict with keys:
        baseline_ber, retrofit_ber, saving_kwh, saving_pct,
        baseline_grade, retrofit_grade
    """
    if measure_key not in MEASURES:
        raise ValueError(f"Unknown measure '{measure_key}'. "
                         f"Available: {list(MEASURES.keys())}")

    measure    = MEASURES[measure_key]
    overrides  = measure['overrides']
    recompute  = measure.get('recompute_derived', False)

    # Baseline prediction
    base_df  = pd.DataFrame([home_features])
    X_base   = prepare_X(base_df)
    base_ber = float(predict_ber(X_base)[0])

    # Post-retrofit prediction
    retro_features = apply_overrides_and_recompute(
        dict(home_features), overrides, recompute
    )
    retro_df  = pd.DataFrame([retro_features])
    X_retro   = prepare_X(retro_df)
    retro_ber = float(predict_ber(X_retro)[0])

    saving_kwh = base_ber - retro_ber
    saving_pct = 100.0 * saving_kwh / base_ber if base_ber > 0 else 0.0

    return {
        'baseline_ber':   base_ber,
        'retrofit_ber':   retro_ber,
        'saving_kwh':     saving_kwh,
        'saving_pct':     saving_pct,
        'baseline_grade': ber_grade(base_ber),
        'retrofit_grade': ber_grade(retro_ber),
        'measure_name':   measure['name'],
    }


def apply_scenario_df(df_input: pd.DataFrame, overrides: dict,
                      recompute: bool) -> pd.DataFrame:
    """Apply overrides + recompute to an entire DataFrame, return encoded X."""
    df_mod = df_input.copy()
    for col, val in overrides.items():
        if col in df_mod.columns:
            df_mod[col] = val

    if recompute:
        # Vectorised recompute of derived features
        df_mod['FabricHeatLossProxy'] = (
            df_mod.get('UValueWall',   0) * df_mod.get('WallArea',   0) +
            df_mod.get('UValueRoof',   0) * df_mod.get('RoofArea',   0) +
            df_mod.get('UValueFloor',  0) * df_mod.get('FloorArea',  0) +
            df_mod.get('UValueWindow', 0) * df_mod.get('WindowArea', 0) +
            df_mod.get('UvalueDoor',   0) * df_mod.get('DoorArea',   0)
        ).astype(np.float32)

        gfa = df_mod.get('GroundFloorAreasq_m', pd.Series(1.0, index=df_mod.index))
        df_mod['FabricHeatLossPerM2'] = np.where(
            gfa > 0,
            df_mod['FabricHeatLossProxy'] / gfa,
            df_mod['FabricHeatLossProxy']
        ).astype(np.float32)

        if 'WallArea' in df_mod.columns and 'WindowArea' in df_mod.columns:
            df_mod['WindowToWallRatio'] = np.where(
                df_mod['WallArea'] > 0,
                df_mod['WindowArea'] / df_mod['WallArea'], 0.0
            ).astype(np.float32)

        if 'UValueRoof' in df_mod.columns:
            df_mod['HasRoofInsulation'] = (df_mod['UValueRoof']  <= 0.16).astype(np.int8)
        if 'UValueWall' in df_mod.columns:
            df_mod['HasWallInsulation'] = (df_mod['UValueWall']  <= 0.37).astype(np.int8)
        if 'UValueWindow' in df_mod.columns:
            df_mod['HasDoubleGlazing']  = (df_mod['UValueWindow'] <= 2.0).astype(np.int8)

    return prepare_X(df_mod)


# ─────────────────────────────────────────────────────────────
# SIMULATION: full dataset × all measures
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RETROFIT SIMULATION (full dataset x {len(MEASURES)} measures)")
print("=" * 60)

df_ret = df.copy().reset_index(drop=True)

# Baseline
X_base   = prepare_X(df_ret)
ber_base = predict_ber(X_base)
print(f"\nBaseline BER — mean: {ber_base.mean():.1f}  "
      f"median: {np.median(ber_base):.1f} kWh/m2/yr")

results = pd.DataFrame({'BerRating_baseline': ber_base})
if 'DwellingTypeDescr' in df_ret.columns:
    results['DwellingType'] = df_ret['DwellingTypeDescr'].values
if 'Year_of_Construction' in df_ret.columns:
    results['Year_of_Construction'] = df_ret['Year_of_Construction'].values
if 'CountyName' in df_ret.columns:
    results['CountyName'] = df_ret['CountyName'].values
if 'AgeBand' in df_ret.columns:
    results['AgeBand'] = df_ret['AgeBand'].values

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("RETROFIT INTERVENTION — AGGREGATE SUMMARY")
summary_lines.append(f"Sample size: {len(df_ret):,} dwellings")
summary_lines.append(f"Baseline BER: mean={ber_base.mean():.1f}, "
                     f"median={np.median(ber_base):.1f} kWh/m2/yr")
summary_lines.append("=" * 60)
summary_lines.append("")

print()
for measure_key, measure in MEASURES.items():
    overrides  = measure['overrides']
    recompute  = measure.get('recompute_derived', False)
    label      = measure['label']

    X_mod      = apply_scenario_df(df_ret, overrides, recompute)
    ber_mod    = predict_ber(X_mod)
    savings    = ber_base - ber_mod
    savings_pct = 100.0 * savings / np.where(ber_base > 0, ber_base, 1.0)

    results[f'BerRating_{label}']  = ber_mod
    results[f'Saving_abs_{label}'] = savings
    results[f'Saving_pct_{label}'] = savings_pct

    line = (f"  {measure['name']:<28s} | "
            f"New BER: {ber_mod.mean():>6.1f} | "
            f"Saving: {savings.mean():>6.1f} kWh/m2/yr "
            f"({savings_pct.mean():>5.1f}%) | "
            f"Median: {np.median(savings):>6.1f}")
    print(line)
    summary_lines.append(line)

# Save per-dwelling results
results.to_csv(OUTPUT_DIR / "retrofit_results.csv", index=False)
print(f"\nPer-dwelling results -> {OUTPUT_DIR / 'retrofit_results.csv'}")

# ─────────────────────────────────────────────────────────────
# ANALYSIS BY DWELLING TYPE AND AGE BAND
# ─────────────────────────────────────────────────────────────
deep_col = 'Saving_abs_H_DeepRetrofit'

summary_lines.append("")
summary_lines.append("-- BY DWELLING TYPE (Deep Retrofit H) --")
if deep_col in results.columns and 'DwellingType' in results.columns:
    by_type = (results.groupby('DwellingType')[deep_col]
               .agg(['mean', 'median', 'count'])
               .round(1)
               .sort_values('mean', ascending=False))
    summary_lines.append(by_type.to_string())

summary_lines.append("")
summary_lines.append("-- BY AGE BAND (Deep Retrofit H) --")
if deep_col in results.columns and 'AgeBand' in results.columns:
    by_age = (results.groupby('AgeBand')[deep_col]
              .agg(['mean', 'median', 'count'])
              .round(1)
              .sort_values('mean', ascending=False))
    summary_lines.append(by_age.to_string())

# ─────────────────────────────────────────────────────────────
# SINGLE DWELLING EXAMPLE
# ─────────────────────────────────────────────────────────────
summary_lines.append("")
summary_lines.append("-- SINGLE DWELLING EXAMPLE --")
ex_row = df_ret.iloc[0]
ex_dict = ex_row.to_dict()
ex_base = predict_ber(prepare_X(pd.DataFrame([ex_dict])))[0]

summary_lines.append(
    f"Dwelling: {ex_dict.get('DwellingTypeDescr','?')}, "
    f"{int(ex_dict.get('Year_of_Construction', 0))}, "
    f"{ex_dict.get('CountyName','?')}"
)
summary_lines.append(f"Baseline BER: {ex_base:.1f} kWh/m2/yr  [{ber_grade(ex_base)}]")

for measure_key, measure in MEASURES.items():
    result = predict_retrofit(ex_dict, measure_key)
    summary_lines.append(
        f"  {measure['name']:<28s} -> {result['retrofit_ber']:>7.1f} "
        f"[{result['retrofit_grade']}]  "
        f"(saving: {result['saving_kwh']:>+7.1f} kWh/m2/yr, {result['saving_pct']:.1f}%)"
    )

# ─────────────────────────────────────────────────────────────
# RETROFIT BAR CHART
# ─────────────────────────────────────────────────────────────
measure_labels = [m['name'] for m in MEASURES.values()]
measure_label_keys = [m['label'] for m in MEASURES.values()]
mean_savings = [
    results[f'Saving_abs_{lbl}'].mean()
    for lbl in measure_label_keys
    if f'Saving_abs_{lbl}' in results.columns
]

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#E53935' if 'Deep' in lbl else '#2196F3' for lbl in measure_label_keys]
bars   = ax.bar(measure_labels, mean_savings, color=colors, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Mean BER Saving (kWh/m2/yr)')
ax.set_title('Expected BER Improvement by Retrofit Measure\n'
             '(LightGBM XAI Scenario Planner, 55-col honest dataset)')
ax.set_xticklabels(measure_labels, rotation=30, ha='right')
for bar, val in zip(bars, mean_savings):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "retrofit_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Retrofit bar chart -> {OUTPUT_DIR / 'retrofit_bar.png'}")

# Save summary
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(OUTPUT_DIR / "retrofit_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"\nSummary saved to {OUTPUT_DIR / 'retrofit_summary.txt'}")
print("\nDone. All retrofit scenario outputs saved.")
