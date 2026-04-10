"""
05_scenario_report.py
=====================
Report generator for the XAI Retrofit Scenario Planner.

Generates formatted ASCII/text reports for each retrofit scenario
and each example dwelling. Reports include:
  - Current vs post-retrofit BER and grade
  - SHAP delta attribution table (which features drove the improvement)
  - Policy note based on age band and dwelling type
  - Cross-measure comparison table

Reads from:
  outputs/xai_summary.csv            — produced by 04_xai_explainer.py
  outputs/scenario_reports/*_delta_shap.csv  — per-home delta tables
  outputs/retrofit_results.csv       — aggregate results from 03_scenario_engine.py

Output:
  outputs/scenario_reports/{measure}_{idx}_report.txt   — per-home reports
  outputs/scenario_reports/full_report.txt               — all reports combined
  outputs/scenario_reports/cross_measure_comparison.txt  — aggregate table
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
OUTPUT_DIR  = BASE_DIR / "outputs"
REPORT_DIR  = OUTPUT_DIR / "scenario_reports"
CONFIG_DIR  = BASE_DIR / "config"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MEASURES_PATH = CONFIG_DIR / "retrofit_measures.json"

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


def ber_grade_description(grade: str) -> str:
    desc = {
        'A1': 'Excellent (near-passive)', 'A2': 'Excellent',
        'A3': 'Excellent', 'B1': 'Very good',
        'B2': 'Good', 'B3': 'Good',
        'C1': 'Fair', 'C2': 'Fair', 'C3': 'Fair',
        'D1': 'Poor', 'D2': 'Poor',
        'E1': 'Very poor', 'E2': 'Very poor',
        'F':  'Very poor', 'G':  'Worst',
    }
    return desc.get(grade, '')


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  BER SCENARIO REPORT GENERATOR")
print("=" * 60)

with open(MEASURES_PATH, 'r') as f:
    MEASURES = json.load(f)

xai_summary_path = OUTPUT_DIR / "xai_summary.csv"
retrofit_path    = OUTPUT_DIR / "retrofit_results.csv"

if not xai_summary_path.exists():
    print(f"ERROR: {xai_summary_path} not found. Run 04_xai_explainer.py first.")
    raise SystemExit(1)

xai_summary    = pd.read_csv(xai_summary_path)
retrofit_results = pd.read_csv(retrofit_path) if retrofit_path.exists() else None

print(f"\nLoaded XAI summary: {len(xai_summary)} rows")
if retrofit_results is not None:
    print(f"Loaded retrofit results: {len(retrofit_results)} rows")


# ─────────────────────────────────────────────────────────────
# POLICY NOTES
# ─────────────────────────────────────────────────────────────
POLICY_NOTES = {
    'Heat Pump Installation': (
        "Heat pump installation is the single most impactful retrofit for "
        "pre-2010 homes in Ireland, consistent with SEAI's Home Energy Upgrade "
        "scheme priority. Average savings of 66 kWh/m2/yr align with the SEAI "
        "retrofit grant programme targets."
    ),
    'Wall Insulation': (
        "External wall insulation is most beneficial for pre-1980 detached and "
        "semi-detached houses with solid masonry construction. SEAI grants cover "
        "up to 80% of the cost for qualifying homes."
    ),
    'Roof Insulation': (
        "Attic insulation is the most cost-effective first step for pre-1978 homes "
        "with uninsulated attic spaces. Typical payback period: 3-5 years."
    ),
    'Window Upgrade': (
        "Window upgrades deliver moderate savings (5%) but significantly improve "
        "thermal comfort. Most beneficial for pre-1990 single-glazed homes."
    ),
    'Floor Insulation': (
        "Floor insulation to TGD L 2021 standard (U=0.15 W/m2K) addresses heat "
        "loss through solid ground floors common in pre-1960 Irish housing."
    ),
    'Solar Water Heating': (
        "Solar thermal panels reduce water heating demand by 50-70% in summer "
        "months. Most effective when combined with a heat pump."
    ),
    'LED Lighting Upgrade': (
        "LED upgrades have minimal impact on BER (1%) but are the easiest and "
        "cheapest first step. Typically done alongside other measures."
    ),
    'Deep Retrofit Package': (
        "The deep retrofit package (heat pump + wall/roof/window insulation + LED) "
        "achieves 45% average BER reduction — sufficient to move most pre-1980 "
        "D-rated homes to B-rated, meeting Ireland's 2030 retrofit target. "
        "Pre-1967 stock benefits most (~280 kWh/m2/yr savings). "
        "SEAI's One Stop Shop service supports whole-home retrofit planning."
    ),
}


# ─────────────────────────────────────────────────────────────
# SINGLE-HOME REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_home_report(row: pd.Series, delta_df: pd.DataFrame,
                         measure_name: str) -> str:
    """
    Generate a formatted ASCII report for one home + one retrofit measure.
    """
    W = 65  # report width

    base_ber   = row['baseline_ber']
    retro_ber  = row['retrofit_ber']
    saving     = row['saving_kwh']
    saving_pct = row['saving_pct']
    base_grade = row['baseline_grade']
    retro_grade = row['retrofit_grade']
    dwelling   = row.get('dwelling_type', '?')
    year_built = int(row.get('year_built', 0))
    county     = row.get('county', '?')

    lines = []
    lines.append('=' * W)
    lines.append(f"  RETROFIT SCENARIO REPORT")
    lines.append(f"  {measure_name}")
    lines.append('=' * W)
    lines.append(f"  Home:       {dwelling}, built {year_built}, {county}")
    lines.append('')
    lines.append(f"  Current BER:      {base_grade} ({base_ber:.1f} kWh/m2/yr)  "
                 f"— {ber_grade_description(base_grade)}")
    lines.append(f"  After retrofit:   {retro_grade} ({retro_ber:.1f} kWh/m2/yr)  "
                 f"— {ber_grade_description(retro_grade)}")
    lines.append(f"  Improvement:      -{saving:.1f} kWh/m2/yr  "
                 f"(-{saving_pct:.1f}%)")

    if base_grade != retro_grade:
        lines.append(f"  BER grade:        {base_grade} -> {retro_grade}")
    lines.append('')

    # Delta attribution table
    top_drivers = delta_df[delta_df['delta_shap'] < 0].head(5)  # Improvements only
    if top_drivers.empty:
        top_drivers = delta_df.head(5)

    lines.append("  What drove the improvement (SHAP attribution):")
    lines.append(f"  {'Feature':<38} {'SHAP Impact':>14} {'Share':>7}")
    lines.append("  " + "-" * (38 + 14 + 7 + 4))

    total_impact = top_drivers['delta_shap'].abs().sum()
    for _, dr in top_drivers.iterrows():
        feat  = str(dr['feature'])[:38]
        delta = dr['delta_shap']
        pct   = abs(delta) / total_impact * 100 if total_impact > 0 else 0
        # Convert log-scale SHAP delta to approximate kWh/m2/yr impact
        kwh_impact = delta * base_ber  # rough approximation
        lines.append(f"  {feat:<38} {delta:>+14.4f} {pct:>6.1f}%")

    lines.append('')

    # Policy note
    policy = POLICY_NOTES.get(measure_name, '')
    if policy:
        lines.append("  Policy note:")
        # Word-wrap the policy note at W-4 chars
        words = policy.split()
        line_buf = "    "
        for word in words:
            if len(line_buf) + len(word) + 1 > W - 2:
                lines.append(line_buf)
                line_buf = "    " + word
            else:
                line_buf += (" " if line_buf != "    " else "") + word
        if line_buf.strip():
            lines.append(line_buf)
    lines.append('=' * W)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# GENERATE ALL PER-HOME REPORTS
# ─────────────────────────────────────────────────────────────
all_report_text = []
all_report_text.append("BER XAI RETROFIT SCENARIO PLANNER — FULL REPORT")
all_report_text.append("=" * 65)
all_report_text.append("")

for measure_key, measure in MEASURES.items():
    label = measure['label']
    name  = measure['name']

    # Find rows in XAI summary for this measure
    measure_rows = xai_summary[xai_summary['measure'] == name]
    if measure_rows.empty:
        print(f"  No XAI summary rows for {name} — skipping.")
        continue

    print(f"\nGenerating reports for: {name}")

    for _, row in measure_rows.iterrows():
        ex_num    = int(row['example'])
        delta_path = REPORT_DIR / f"{label}_{ex_num}_delta_shap.csv"

        if not delta_path.exists():
            print(f"  WARNING: {delta_path} not found, skipping example {ex_num}")
            continue

        delta_df = pd.read_csv(delta_path)

        report_text = generate_home_report(row, delta_df, name)
        all_report_text.append(report_text)
        all_report_text.append("")

        # Save individual report
        report_path = REPORT_DIR / f"{label}_{ex_num}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"  Saved: {report_path.name}")


# ─────────────────────────────────────────────────────────────
# CROSS-MEASURE COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\nGenerating cross-measure comparison table...")

cross_lines = []
cross_lines.append("=" * 65)
cross_lines.append("CROSS-MEASURE COMPARISON — MEAN PERFORMANCE")
cross_lines.append("=" * 65)

if retrofit_results is not None:
    cross_lines.append("")
    cross_lines.append(f"{'Measure':<28} {'Mean Saving':>12} {'Median':>9} {'% Saving':>10}")
    cross_lines.append("-" * 65)

    for measure_key, measure in MEASURES.items():
        label = measure['label']
        col   = f'Saving_abs_{label}'
        if col in retrofit_results.columns:
            savings = retrofit_results[col]
            base_col = 'BerRating_baseline'
            base     = retrofit_results[base_col] if base_col in retrofit_results.columns \
                       else pd.Series(np.ones(len(retrofit_results)) * 250)
            pct_col  = f'Saving_pct_{label}'
            pct_mean = retrofit_results[pct_col].mean() \
                       if pct_col in retrofit_results.columns else np.nan
            cross_lines.append(
                f"  {measure['name']:<26} {savings.mean():>10.1f}  "
                f"{savings.median():>8.1f}  {pct_mean:>9.1f}%"
            )

    cross_lines.append("")

# Sub-group findings from XAI summary
cross_lines.append("-- NOTABLE SUB-GROUP FINDINGS --")
cross_lines.append("")
cross_lines.append("From pipeline-2 validated results (05_results.md):")
cross_lines.append("")
cross_lines.append(f"  {'Finding':<55} {'Evidence'}")
cross_lines.append("-" * 65)
findings = [
    ("Pre-1900 homes save ~280 kWh/m2/yr from deep retrofit",
     "Age-band analysis"),
    ("2016+ homes gain ~1 kWh/m2/yr (retrofit waste on new builds)",
     "Age-band analysis"),
    ("Top-floor apartments benefit most by type (-141 kWh/m2/yr)",
     "Dwelling-type analysis"),
    ("Mid-floor apartments benefit least (-80 kWh/m2/yr)",
     "Shared walls = less exposed surface"),
]
for finding, evidence in findings:
    cross_lines.append(f"  {finding:<55} {evidence}")

cross_lines.append("")
cross_lines.append("-- POLICY RECOMMENDATIONS --")
cross_lines.append("")
cross_lines.append("  1. Target grants at pre-1980 detached/semi-detached houses.")
cross_lines.append("  2. Heat pump installation: biggest single intervention.")
cross_lines.append("  3. Deep retrofit package achieves Ireland 2030 B-rating target.")
cross_lines.append("  4. 2016+ homes: no material BER benefit from further insulation.")
cross_lines.append("  5. Top-floor apartments: prioritise roof insulation.")
cross_lines.append("")

cross_text = "\n".join(cross_lines)
cross_path = REPORT_DIR / "cross_measure_comparison.txt"
with open(cross_path, 'w', encoding='utf-8') as f:
    f.write(cross_text)
print(f"  Saved: {cross_path.name}")

all_report_text.append(cross_text)

# Save full combined report
full_text = "\n".join(all_report_text)
full_path = REPORT_DIR / "full_report.txt"
with open(full_path, 'w', encoding='utf-8') as f:
    f.write(full_text)
print(f"\nFull combined report saved to {full_path}")
print("\nDone. All reports generated.")
