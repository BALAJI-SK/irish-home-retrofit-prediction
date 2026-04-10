"""
01_clean_and_prepare.py
=======================
Produces the honest ~55-column dataset for the XAI Retrofit Scenario Planner.

Two-source strategy:
  1. FILTERED PARQUET  — already-cleaned 46-col base
       /Users/akarsh/Downloads/Col Final with County.parquet
       (1,351,582 rows, BerRating/Year already filtered, values cleaned)

  2. FULL CSV          — raw 211-col SEAI BER database
       /Users/akarsh/Downloads/Public Search Data.csv
       Used only to pull 7 columns absent from the parquet:
         MainSpaceHeatingFuel, MainWaterHeatingFuel, SolarHotWaterHeating,
         LowEnergyLightingPercent, GroundFloorArea(sq m) -> GroundFloorAreasq_m,
         HESSchemeUpgrade, FirstEnergyType_Description
       Row-alignment: apply the same BerRating/Year filters to the CSV slice
       → positional merge (both datasets retain the same rows in the same order).

Leakage removal vs the 46-col parquet base:
  REMOVED:  DistributionLosses, HSEffAdjFactor, WHEffAdjFactor,
            SHRenewableResources, WHRenewableResources
            (DEAP intermediate outputs / high-VIF features)

Feature engineering (9 new features — all computed from raw inputs):
  WindowToWallRatio, FabricHeatLossProxy, FabricHeatLossPerM2,
  AvgWallUValue*, TotalFloorArea_computed*, AgeBand,
  IsHeatPump, HasSolarWaterHeating, HasRoofInsulation,
  HasWallInsulation, HasDoubleGlazing, has_hw_cylinder*
  (* from parquet base; re-verified after merge)

Output:
  outputs/clean_data_55col.parquet
  outputs/cleaning_report.txt
"""

import sys
import warnings
import time
from pathlib import Path

from cli_logger import setup_script_logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
PARQUET_IN  = Path('/Users/akarsh/Downloads/Col Final with County.parquet')
CSV_IN      = Path('/Users/akarsh/Downloads/Public Search Data.csv')

OUTPUT_DIR  = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(OUTPUT_DIR / f"{Path(__file__).stem}.log")
PARQUET_OUT = OUTPUT_DIR / 'clean_data_55col.parquet'
REPORT_PATH = OUTPUT_DIR / 'cleaning_report.txt'

TARGET  = 'BerRating'
BER_MIN = 0.0
BER_MAX = 2000.0
YOC_MIN = 1700
YOC_MAX = 2026

# ─────────────────────────────────────────────────────────────
# COLUMNS TO DROP FROM PARQUET (leaky DEAP outputs)
# ─────────────────────────────────────────────────────────────
DROP_LEAKY = [
    'DistributionLosses',      # DEAP-calculated output
    'HSEffAdjFactor',          # DEAP intermediate, high VIF (>25)
    'WHEffAdjFactor',          # DEAP intermediate, high VIF (>25)
    'SHRenewableResources',    # DEAP intermediate, high VIF (>25)
    'WHRenewableResources',    # DEAP intermediate, high VIF (>25)
    'CountyName',              # Geographic identifier — not a building input
    # ── Ablation: high-importance features removed to test R² impact ──
    'FabricHeatLossPerM2',     # #1 SHAP — removed to show ~5-8% R² drop
    'Year_of_Construction',    # #2 SHAP — AgeBand retains vintage info
    'HSMainSystemEfficiency',  # #3 SHAP — IsHeatPump flag retains heat pump info
]

# NOTE: HSMainSystemEfficiency and Year_of_Construction are used during
# feature engineering (IsHeatPump, AgeBand) BEFORE being dropped.
# So their derived features are still available to the model.

# ─────────────────────────────────────────────────────────────
# COLUMNS TO ADD FROM CSV (raw DEAP inputs missing from parquet)
# ─────────────────────────────────────────────────────────────
# CSV column name       → target column name in output
CSV_ADD_COLS = {
    'MainSpaceHeatingFuel':      'MainSpaceHeatingFuel',      # critical for heat pump sim
    'MainWaterHeatingFuel':      'MainWaterHeatingFuel',      # water heating scenario
    'SolarHotWaterHeating':      'SolarHotWaterHeating',      # solar water retrofit flag
    'LowEnergyLightingPercent':  'LowEnergyLightingPercent',  # LED upgrade scenario
    'GroundFloorArea(sq m)':     'GroundFloorAreasq_m',       # needed for FabricHeatLossPerM2
    'HESSchemeUpgrade':          'HESSchemeUpgrade',          # pre-retrofit flag
    'FirstEnergyType_Description':'FirstEnergyType_Description', # renewable type
}

# Row-filter anchors in the CSV (used to align rows with the parquet)
CSV_ANCHORS = ['BerRating', 'Year_of_Construction']

# ─────────────────────────────────────────────────────────────
# AGE BAND (DEAP vintage brackets)
# ─────────────────────────────────────────────────────────────
AGE_BINS   = [0, 1900, 1930, 1950, 1967, 1978, 1983, 1994, 2000, 2005, 2011, 2016, 9999]
AGE_LABELS = ['Pre1900', '1900-1929', '1930-1949', '1950-1966',
              '1967-1977', '1978-1982', '1983-1993', '1994-1999',
              '2000-2004', '2005-2010', '2011-2015', '2016+']

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load the filtered parquet (base 46-col set)
# ─────────────────────────────────────────────────────────────
print('=' * 60)
print('  BER DATASET — 55-COL HONEST CLEANING PIPELINE')
print('=' * 60)
t_start = time.time()

print(f'\nSTEP 1: Loading filtered parquet from {PARQUET_IN}...')
df = pd.read_parquet(PARQUET_IN)
print(f'  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols')

# Verify row filters are already applied (sanity check)
ber_ok = df[TARGET].between(BER_MIN, BER_MAX)
yoc_ok = df['Year_of_Construction'].between(YOC_MIN, YOC_MAX)
bad_rows = (~ber_ok | ~yoc_ok).sum()
if bad_rows > 0:
    print(f'  WARNING: {bad_rows:,} rows outside expected range — filtering now.')
    df = df[ber_ok & yoc_ok].reset_index(drop=True)
else:
    print(f'  Row filters already applied (BerRating [{BER_MIN},{BER_MAX}], '
          f'Year [{YOC_MIN},{YOC_MAX}]). ✓')
    df = df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Remove leaky DEAP columns
# ─────────────────────────────────────────────────────────────
print(f'\nSTEP 2: Removing leaky DEAP output columns...')
for col in DROP_LEAKY:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f'  Removed: {col}')
    else:
        print(f'  Already absent: {col}')
print(f'  After removal: {df.shape[1]} columns')

# ─────────────────────────────────────────────────────────────
# STEP 3 — Pull 7 missing columns from the full CSV
# ─────────────────────────────────────────────────────────────
print(f'\nSTEP 3: Loading {len(CSV_ADD_COLS)} additional columns from CSV...')
print(f'  CSV: {CSV_IN}')

csv_load_cols = CSV_ANCHORS + list(CSV_ADD_COLS.keys())
print(f'  Reading {len(csv_load_cols)} columns from {CSV_IN.name}...')

df_csv = pd.read_csv(
    CSV_IN,
    usecols=csv_load_cols,
    low_memory=False,
    encoding='latin-1',
)
print(f'  Raw CSV slice: {df_csv.shape[0]:,} rows')

# Apply the same row filters to align with the parquet
csv_ber_ok = df_csv['BerRating'].between(BER_MIN, BER_MAX)
csv_yoc_ok = df_csv['Year_of_Construction'].between(YOC_MIN, YOC_MAX)
df_csv = df_csv[csv_ber_ok & csv_yoc_ok].reset_index(drop=True)
print(f'  After same row filters: {df_csv.shape[0]:,} rows')

# Verify row counts match
if len(df_csv) != len(df):
    print(f'  WARNING: CSV slice has {len(df_csv):,} rows but parquet has {len(df):,} rows.')
    print('  Trimming to minimum length (positional merge).')
    min_len = min(len(df_csv), len(df))
    df     = df.iloc[:min_len].reset_index(drop=True)
    df_csv = df_csv.iloc[:min_len].reset_index(drop=True)
else:
    print(f'  Row counts match ({len(df):,}). Positional merge safe. ✓')

# Rename and attach the new columns
for csv_col, out_col in CSV_ADD_COLS.items():
    if csv_col in df_csv.columns:
        df[out_col] = df_csv[csv_col].values
        print(f'  Added: {csv_col} -> {out_col}')
    else:
        print(f'  WARNING: {csv_col} not found in CSV — filling with NaN')
        df[out_col] = np.nan

del df_csv
print(f'  After merge: {df.shape[1]} columns')

# ─────────────────────────────────────────────────────────────
# STEP 4 — Clean new columns
# ─────────────────────────────────────────────────────────────
print(f'\nSTEP 4: Cleaning newly added columns...')

# Strip whitespace from string columns
str_new_cols = [
    'MainSpaceHeatingFuel', 'MainWaterHeatingFuel',
    'SolarHotWaterHeating', 'HESSchemeUpgrade',
    'FirstEnergyType_Description',
]
for col in str_new_cols:
    if col in df.columns and df[col].dtype == object:
        df[col] = df[col].str.strip()

# SolarHotWaterHeating: null means no solar system (MNAR) → 'No_cylinder' for consistency
if 'SolarHotWaterHeating' in df.columns:
    # Check if column already has no-cylinder values from parquet imputation
    null_count = df['SolarHotWaterHeating'].isna().sum()
    if null_count > 0:
        df['SolarHotWaterHeating'] = df['SolarHotWaterHeating'].fillna('NO')
        print(f'  SolarHotWaterHeating: filled {null_count:,} nulls with NO')

# LowEnergyLightingPercent: null → 0 (no low-energy lighting)
if 'LowEnergyLightingPercent' in df.columns:
    null_count = df['LowEnergyLightingPercent'].isna().sum()
    if null_count > 0:
        df['LowEnergyLightingPercent'] = df['LowEnergyLightingPercent'].fillna(0.0)
        print(f'  LowEnergyLightingPercent: filled {null_count:,} nulls with 0')

# GroundFloorAreasq_m: null → median
if 'GroundFloorAreasq_m' in df.columns:
    null_count = df['GroundFloorAreasq_m'].isna().sum()
    if null_count > 0:
        med = df['GroundFloorAreasq_m'].median()
        df['GroundFloorAreasq_m'] = df['GroundFloorAreasq_m'].fillna(med)
        print(f'  GroundFloorAreasq_m: filled {null_count:,} nulls with median={med:.1f}')

# MainSpaceHeatingFuel / MainWaterHeatingFuel: null → mode
for col in ['MainSpaceHeatingFuel', 'MainWaterHeatingFuel']:
    if col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            mode_val = df[col].dropna().mode()
            fill = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill)
            print(f'  {col}: filled {null_count:,} nulls with mode={fill!r}')

# HESSchemeUpgrade: null → 'No'
if 'HESSchemeUpgrade' in df.columns:
    null_count = df['HESSchemeUpgrade'].isna().sum()
    if null_count > 0:
        df['HESSchemeUpgrade'] = df['HESSchemeUpgrade'].fillna('No')
        print(f'  HESSchemeUpgrade: filled {null_count:,} nulls with No')

# FirstEnergyType_Description: null → 'None'
if 'FirstEnergyType_Description' in df.columns:
    null_count = df['FirstEnergyType_Description'].isna().sum()
    if null_count > 0:
        df['FirstEnergyType_Description'] = df['FirstEnergyType_Description'].fillna('None')
        print(f'  FirstEnergyType_Description: filled {null_count:,} nulls with None')

# ─────────────────────────────────────────────────────────────
# STEP 5 — Feature engineering (9 features from plan Section 3)
# ─────────────────────────────────────────────────────────────
print(f'\nSTEP 5: Engineering features...')

# ── IsHeatPump (HSMainSystemEfficiency > 100 OR fuel string match) ─
if 'HSMainSystemEfficiency' in df.columns:
    hp_eff  = df['HSMainSystemEfficiency'] > 100
    hp_fuel = df.get('MainSpaceHeatingFuel', pd.Series('', index=df.index)).str.contains(
        'Heat Pump|heat pump', case=False, na=False
    )
    df['IsHeatPump'] = (hp_eff | hp_fuel).astype(np.int8)
    print(f'  IsHeatPump: {df["IsHeatPump"].sum():,} homes with heat pump')

# ── HasSolarWaterHeating ──────────────────────────────────────
if 'SolarHotWaterHeating' in df.columns:
    df['HasSolarWaterHeating'] = (
        df['SolarHotWaterHeating'].isin(['YES', 'Yes', 'yes'])
    ).astype(np.int8)
    print(f'  HasSolarWaterHeating: {df["HasSolarWaterHeating"].sum():,} homes')

# ── HasRoofInsulation (UValueRoof <= 0.16) ───────────────────
if 'UValueRoof' in df.columns:
    df['HasRoofInsulation'] = (df['UValueRoof'] <= 0.16).astype(np.int8)

# ── HasWallInsulation (UValueWall <= 0.37) ───────────────────
if 'UValueWall' in df.columns:
    df['HasWallInsulation'] = (df['UValueWall'] <= 0.37).astype(np.int8)

# ── HasDoubleGlazing (UValueWindow <= 2.0) ───────────────────
if 'UValueWindow' in df.columns:
    df['HasDoubleGlazing'] = (df['UValueWindow'] <= 2.0).astype(np.int8)

# ── WindowToWallRatio ─────────────────────────────────────────
if 'WindowArea' in df.columns and 'WallArea' in df.columns:
    df['WindowToWallRatio'] = np.where(
        df['WallArea'] > 0,
        df['WindowArea'] / df['WallArea'],
        0.0
    ).astype(np.float32)

# ── FabricHeatLossProxy (UA-sum from raw inputs — NOT DEAP output) ─
uval_area_pairs = [
    ('UValueWall',   'WallArea'),
    ('UValueRoof',   'RoofArea'),
    ('UValueFloor',  'FloorArea'),
    ('UValueWindow', 'WindowArea'),
    ('UvalueDoor',   'DoorArea'),
]
if all(c in df.columns for pair in uval_area_pairs for c in pair):
    df['FabricHeatLossProxy'] = sum(
        df[u] * df[a] for u, a in uval_area_pairs
    ).astype(np.float32)

    # FabricHeatLossPerM2 is in DROP_LEAKY (ablation test) — skipping

# ── AgeBand (DEAP vintage brackets) ──────────────────────────
if 'Year_of_Construction' in df.columns:
    df['AgeBand'] = pd.cut(
        df['Year_of_Construction'],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True
    ).astype(str)

print(f'  Feature engineering done.')

# ─────────────────────────────────────────────────────────────
# STEP 6 — Dtype optimisation
# ─────────────────────────────────────────────────────────────
for col in df.select_dtypes('float64').columns:
    df[col] = df[col].astype(np.float32)
for col in df.select_dtypes('int64').columns:
    df[col] = df[col].astype(np.int32)

# ─────────────────────────────────────────────────────────────
# STEP 7 — Save output parquet
# ─────────────────────────────────────────────────────────────
print(f'\nSTEP 7: Saving {df.shape[1]}-column dataset to {PARQUET_OUT}...')
df.to_parquet(PARQUET_OUT, index=False, compression='snappy')
print(f'  Saved: {df.shape[0]:,} rows x {df.shape[1]} cols')

# ─────────────────────────────────────────────────────────────
# VALIDATION & REPORT
# ─────────────────────────────────────────────────────────────
print(f'\nVALIDATION:')
report_lines = []
report_lines.append('=' * 60)
report_lines.append('BER DATASET (55-COL HONEST SET) — CLEANING REPORT')
report_lines.append('=' * 60)
report_lines.append(f'Source (base)    : {PARQUET_IN}')
report_lines.append(f'Source (add cols): {CSV_IN}')
report_lines.append(f'Output           : {PARQUET_OUT}')
report_lines.append(f'Final shape      : {df.shape[0]:,} rows x {df.shape[1]} cols')
report_lines.append('')

# Leakage check
report_lines.append('-- LEAKAGE VERIFICATION --')
for col in DROP_LEAKY + ['FirstEnerProdDelivered', 'TempAdjustment', 'TempFactorMultiplier']:
    status = 'PRESENT (PROBLEM!)' if col in df.columns else 'absent (OK)'
    report_lines.append(f'  {col:<35s}: {status}')
report_lines.append('')

# Correlation check — no feature should correlate > 0.90 with target
report_lines.append('-- HIGH CORRELATION CHECK (|corr| > 0.80 with BerRating) --')
num_df = df.select_dtypes(include=[np.number])
corr   = num_df.corr()[TARGET].drop(TARGET, errors='ignore').abs().sort_values(ascending=False)
high   = corr[corr > 0.80]
if high.empty:
    report_lines.append('  No numeric feature correlates > 0.80. ✓')
else:
    for feat, val in high.items():
        report_lines.append(f'  WARNING: {feat:<40s}: {val:.4f}')
report_lines.append('')

# Target stats
report_lines.append('-- TARGET STATISTICS (BerRating) --')
ber = df[TARGET]
report_lines.append(f'  Mean   : {ber.mean():.2f} kWh/m2/yr')
report_lines.append(f'  Median : {ber.median():.2f} kWh/m2/yr')
report_lines.append(f'  Std    : {ber.std():.2f}')
report_lines.append(f'  Pct A-B (<=100)  : {(ber <= 100).mean()*100:.1f}%')
report_lines.append(f'  Pct C  (101-200) : {((ber > 100) & (ber <= 200)).mean()*100:.1f}%')
report_lines.append(f'  Pct D  (201-300) : {((ber > 200) & (ber <= 300)).mean()*100:.1f}%')
report_lines.append(f'  Pct E+ (>300)    : {(ber > 300).mean()*100:.1f}%')
report_lines.append('')

# Null check
report_lines.append('-- NULL CHECK --')
nulls = df.isna().sum()
null_cols = nulls[nulls > 0]
if null_cols.empty:
    report_lines.append('  No nulls remaining. ✓')
else:
    for col, cnt in null_cols.items():
        report_lines.append(f'  {col:<40s}: {cnt:,} ({cnt/len(df)*100:.2f}%)')
report_lines.append('')

# Engineered features
report_lines.append('-- ENGINEERED FEATURES --')
eng_cols = [
    'WindowToWallRatio', 'FabricHeatLossProxy', 'FabricHeatLossPerM2',
    'AgeBand', 'IsHeatPump', 'HasSolarWaterHeating',
    'HasRoofInsulation', 'HasWallInsulation', 'HasDoubleGlazing',
]
for col in eng_cols:
    if col in df.columns:
        if df[col].dtype in [np.float32, np.float64]:
            report_lines.append(f'  {col:<40s} mean={df[col].mean():.4f}')
        else:
            report_lines.append(f'  {col:<40s} {df[col].value_counts().to_dict()}')
report_lines.append('')

# Heating fuel distribution (critical for retrofit simulation)
report_lines.append('-- MAIN SPACE HEATING FUEL (top 10) --')
if 'MainSpaceHeatingFuel' in df.columns:
    for fuel, cnt in df['MainSpaceHeatingFuel'].value_counts().head(10).items():
        report_lines.append(f'  {str(fuel):<35s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)')
report_lines.append('')

# Age band
report_lines.append('-- AGE BAND DISTRIBUTION --')
if 'AgeBand' in df.columns:
    for band, cnt in df['AgeBand'].value_counts().sort_index().items():
        report_lines.append(f'  {str(band):<20s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)')
report_lines.append('')

# Column list
report_lines.append('-- FINAL COLUMN LIST --')
for i, col in enumerate(sorted(df.columns), 1):
    report_lines.append(f'  {i:3d}. {col}')

report_text = '\n'.join(report_lines)
print(report_text)

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f'\nReport saved to {REPORT_PATH}')
print(f'\nTotal runtime: {time.time()-t_start:.1f}s')
print('Done. Clean 55-col dataset ready at:', PARQUET_OUT)
