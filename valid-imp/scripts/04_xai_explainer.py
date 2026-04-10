"""
04_xai_explainer.py
===================
SHAP-based XAI explainer for the retrofit scenario planner.

For each retrofit measure, this script:
  1. Selects example dwellings from the dataset
  2. Computes SHAP values before and after the retrofit
  3. Produces a delta attribution table (which features drove the change)
  4. Generates SHAP waterfall plots (before/after)
  5. Saves per-scenario CSV outputs

The delta SHAP approach shows *why* a retrofit improved the BER — not just
that it did. This is the core XAI contribution of the project.

SHAP performance note:
  - Global SHAP (10K rows): done in 02_train_model.py
  - Local SHAP (single row): instant with TreeExplainer — used here

Output (in outputs/scenario_reports/):
  {measure}_{idx}_shap_before.png     — waterfall before retrofit
  {measure}_{idx}_shap_after.png      — waterfall after retrofit
  {measure}_{idx}_delta_shap.csv      — feature-level delta attribution
  xai_summary.csv                     — cross-measure summary
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
OUTPUT_DIR    = BASE_DIR / "outputs"
REPORT_DIR    = OUTPUT_DIR / "scenario_reports"
CONFIG_DIR    = BASE_DIR / "config"
PARQUET_PATH  = OUTPUT_DIR / "clean_data_55col.parquet"
LGBM_PATH     = OUTPUT_DIR / "lgbm_model.pkl"
MEASURES_PATH = CONFIG_DIR / "retrofit_measures.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Number of example dwellings to explain per measure
N_EXAMPLES  = 3
RANDOM_SEED = 42
TARGET      = 'BerRating'

# BER grade boundaries
BER_GRADES = [
    (0,   25,  'A1'), (25,  50,  'A2'), (50,  75,  'A3'),
    (75,  100, 'B1'), (100, 125, 'B2'), (125, 150, 'B3'),
    (150, 175, 'C1'), (175, 200, 'C2'), (200, 225, 'C3'),
    (225, 260, 'D1'), (260, 300, 'D2'),
    (300, 340, 'E1'), (340, 380, 'E2'),
    (380, 450, 'F'),
    (450, 9999, 'G'),
]


def ber_grade(kwh: float) -> str:
    for lo, hi, grade in BER_GRADES:
        if lo <= kwh < hi:
            return grade
    return 'G'


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BER XAI EXPLAINER — SHAP BEFORE/AFTER RETROFIT")
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

# ─────────────────────────────────────────────────────────────
# SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────
print("\nInitialising SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
print("  Done.")


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def prepare_X(df_input: pd.DataFrame) -> pd.DataFrame:
    X = df_input.drop(columns=[TARGET], errors='ignore').copy()
    for col in cat_cols:
        if col in X.columns:
            X[col] = encoders[col].transform(X[[col]]).astype(np.float32)
    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].astype(np.float32)
    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[feature_names]


def predict_ber(X: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(X)).clip(min=0)


def recompute_derived(row_dict: dict) -> dict:
    u_wall   = row_dict.get('UValueWall',   0)
    u_roof   = row_dict.get('UValueRoof',   0)
    u_floor  = row_dict.get('UValueFloor',  0)
    u_window = row_dict.get('UValueWindow', 0)
    u_door   = row_dict.get('UvalueDoor',   0)
    a_wall   = row_dict.get('WallArea',   0)
    a_roof   = row_dict.get('RoofArea',   0)
    a_floor  = row_dict.get('FloorArea',  0)
    a_window = row_dict.get('WindowArea', 0)
    a_door   = row_dict.get('DoorArea',   0)

    proxy = (u_wall*a_wall + u_roof*a_roof + u_floor*a_floor +
             u_window*a_window + u_door*a_door)
    row_dict['FabricHeatLossProxy'] = float(proxy)
    gfa = row_dict.get('GroundFloorAreasq_m', 1.0)
    row_dict['FabricHeatLossPerM2'] = float(proxy / gfa) if gfa > 0 else float(proxy)
    row_dict['WindowToWallRatio'] = (
        float(a_window / a_wall) if a_wall > 0 else 0.0
    )
    row_dict['HasRoofInsulation'] = int(row_dict.get('UValueRoof',  1.0) <= 0.16)
    row_dict['HasWallInsulation'] = int(row_dict.get('UValueWall',  1.0) <= 0.37)
    row_dict['HasDoubleGlazing']  = int(row_dict.get('UValueWindow', 5.0) <= 2.0)
    return row_dict


def explain_retrofit(home_row: pd.Series, measure_key: str) -> dict:
    """
    Compute SHAP values before and after a retrofit measure for a single home.

    Returns dict with:
        shap_before  : np.ndarray (1D, one value per feature)
        shap_after   : np.ndarray (1D)
        delta_shap   : np.ndarray (shap_after - shap_before)
        top_drivers  : list of (feature_name, delta_value) sorted by |delta|
        base_ber     : float
        retro_ber    : float
        feature_names: list[str]
        expected_value: float (SHAP base value)
    """
    measure   = MEASURES[measure_key]
    overrides = measure['overrides']
    recompute = measure.get('recompute_derived', False)

    # Before
    home_df = pd.DataFrame([home_row])
    X_before = prepare_X(home_df)
    base_ber = float(predict_ber(X_before)[0])

    sv_before = explainer.shap_values(X_before)
    if isinstance(sv_before, list):
        sv_before = sv_before[0]
    sv_before = sv_before.flatten()

    # After
    retro_dict = home_row.to_dict()
    retro_dict.update(overrides)
    if recompute:
        retro_dict = recompute_derived(retro_dict)

    retro_df = pd.DataFrame([retro_dict])
    # Align columns
    for col in X_before.columns:
        if col not in retro_df.columns:
            retro_df[col] = X_before[col].values
    X_after = prepare_X(retro_df[home_df.drop(columns=[TARGET], errors='ignore').columns])
    retro_ber = float(predict_ber(X_after)[0])

    sv_after = explainer.shap_values(X_after)
    if isinstance(sv_after, list):
        sv_after = sv_after[0]
    sv_after = sv_after.flatten()

    delta_shap   = sv_after - sv_before
    feature_names = X_before.columns.tolist()

    top_drivers = sorted(
        zip(feature_names, delta_shap.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    return {
        'shap_before':   sv_before,
        'shap_after':    sv_after,
        'delta_shap':    delta_shap,
        'top_drivers':   top_drivers,
        'base_ber':      base_ber,
        'retro_ber':     retro_ber,
        'feature_names': feature_names,
        'expected_value': float(explainer.expected_value)
            if not isinstance(explainer.expected_value, np.ndarray)
            else float(explainer.expected_value[0]),
    }


def save_waterfall(shap_vals: np.ndarray, expected_val: float,
                   feature_names: list, title: str, path: Path) -> None:
    """Save a SHAP waterfall-style bar chart for a single prediction."""
    # Use top-15 features by absolute SHAP value
    top_n = 15
    abs_vals   = np.abs(shap_vals)
    top_idx    = np.argsort(abs_vals)[::-1][:top_n]
    top_feats  = [feature_names[i] for i in top_idx]
    top_shaps  = [shap_vals[i] for i in top_idx]

    colors = ['#d32f2f' if v > 0 else '#1565c0' for v in top_shaps]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_feats))
    ax.barh(list(y_pos), top_shaps[::-1], color=colors[::-1])
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_xlabel('SHAP value (impact on log BerRating)')
    ax.set_title(title, fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_delta_table(explanation: dict, measure_key: str,
                     example_idx: int) -> pd.DataFrame:
    """Save delta SHAP attribution table as CSV."""
    measure    = MEASURES[measure_key]
    delta      = explanation['delta_shap']
    feat_names = explanation['feature_names']
    saving     = explanation['base_ber'] - explanation['retro_ber']

    records = []
    for feat, d in zip(feat_names, delta.tolist()):
        pct = 100.0 * abs(d) / abs(saving) if saving != 0 else 0.0
        records.append({
            'feature':    feat,
            'shap_before': float(explanation['shap_before'][feat_names.index(feat)]),
            'shap_after':  float(explanation['shap_after'][feat_names.index(feat)]),
            'delta_shap':  float(d),
            'pct_of_total': pct,
        })

    delta_df = (pd.DataFrame(records)
                .sort_values('delta_shap', key=abs, ascending=False)
                .reset_index(drop=True))

    out_path = REPORT_DIR / f"{measure['label']}_{example_idx}_delta_shap.csv"
    delta_df.to_csv(out_path, index=False)
    return delta_df


# ─────────────────────────────────────────────────────────────
# RUN XAI EXPLANATION FOR EACH MEASURE × N_EXAMPLES
# ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)
# Prefer high-BER homes (most benefit from retrofit)
high_ber_mask = df[TARGET] > 200
candidate_df  = df[high_ber_mask] if high_ber_mask.sum() > N_EXAMPLES else df
example_idx   = rng.choice(len(candidate_df), size=N_EXAMPLES, replace=False)
example_rows  = candidate_df.iloc[example_idx].reset_index(drop=True)

xai_summary = []

for measure_key, measure in MEASURES.items():
    print(f"\n{'='*60}")
    print(f"  Measure: {measure['name']}")
    print(f"{'='*60}")

    for i, (_, home_row) in enumerate(example_rows.iterrows()):
        print(f"  Example {i+1}/{N_EXAMPLES}: "
              f"{home_row.get('DwellingTypeDescr','?')}, "
              f"{int(home_row.get('Year_of_Construction', 0))}, "
              f"{home_row.get('CountyName','?')}")

        try:
            exp = explain_retrofit(home_row, measure_key)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        base_ber  = exp['base_ber']
        retro_ber = exp['retro_ber']
        saving    = base_ber - retro_ber
        saving_pct = 100.0 * saving / base_ber if base_ber > 0 else 0.0

        print(f"    Baseline BER : {base_ber:.1f} [{ber_grade(base_ber)}]")
        print(f"    After retrofit: {retro_ber:.1f} [{ber_grade(retro_ber)}]")
        print(f"    Saving       : {saving:.1f} kWh/m2/yr ({saving_pct:.1f}%)")
        print(f"    Top 5 drivers:")
        for feat, delta in exp['top_drivers'][:5]:
            print(f"      {feat:<40s}: {delta:+.4f}")

        # Save waterfall before
        save_waterfall(
            exp['shap_before'], exp['expected_value'],
            exp['feature_names'],
            f"SHAP Before Retrofit — {measure['name']}\n"
            f"Home: {home_row.get('DwellingTypeDescr','?')}, "
            f"{int(home_row.get('Year_of_Construction', 0))}  "
            f"BER={base_ber:.1f} [{ber_grade(base_ber)}]",
            REPORT_DIR / f"{measure['label']}_{i+1}_shap_before.png"
        )

        # Save waterfall after
        save_waterfall(
            exp['shap_after'], exp['expected_value'],
            exp['feature_names'],
            f"SHAP After Retrofit — {measure['name']}\n"
            f"Home: {home_row.get('DwellingTypeDescr','?')}, "
            f"{int(home_row.get('Year_of_Construction', 0))}  "
            f"BER={retro_ber:.1f} [{ber_grade(retro_ber)}]",
            REPORT_DIR / f"{measure['label']}_{i+1}_shap_after.png"
        )

        # Save delta table
        delta_df = save_delta_table(exp, measure_key, i+1)

        # Collect summary record
        top1_feat, top1_delta = exp['top_drivers'][0]
        xai_summary.append({
            'measure':         measure['name'],
            'example':         i + 1,
            'dwelling_type':   home_row.get('DwellingTypeDescr', '?'),
            'year_built':      int(home_row.get('Year_of_Construction', 0)),
            'county':          home_row.get('CountyName', '?'),
            'baseline_ber':    round(base_ber, 2),
            'baseline_grade':  ber_grade(base_ber),
            'retrofit_ber':    round(retro_ber, 2),
            'retrofit_grade':  ber_grade(retro_ber),
            'saving_kwh':      round(saving, 2),
            'saving_pct':      round(saving_pct, 2),
            'top_driver_feat': top1_feat,
            'top_driver_delta':round(top1_delta, 4),
        })

# Save cross-measure summary
xai_summary_df = pd.DataFrame(xai_summary)
xai_summary_df.to_csv(OUTPUT_DIR / "xai_summary.csv", index=False)
print(f"\nXAI summary saved to {OUTPUT_DIR / 'xai_summary.csv'}")

print(f"\nAll SHAP plots and delta tables saved to {REPORT_DIR}/")
print("Done.")
