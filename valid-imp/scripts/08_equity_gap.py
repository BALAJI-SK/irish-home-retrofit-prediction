"""
08_equity_gap.py
================
Computes the Retrofit Equity Gap Score for each Irish county.

The Equity Gap Score captures the degree to which a county is
underserved by retrofit policy — combining:
  1. Fuel poverty burden        (share of homes at fuel poverty risk)
  2. Retrofit penetration gap   (homes that need retrofit but haven't got it)
  3. Carbon intensity           (mean CO₂ kg/m²/yr)

Formula (normalised 0–100):
    equity_gap = fp_rate × (1 - retrofit_rate) × co2_norm
  where co2_norm = mean_co2 / max_mean_co2 across counties

A high equity gap score means the county has:
  - Many fuel-poor homes (oil/coal/peat heating + BER > 300)
  - Low retrofit uptake (few walls/roofs insulated, no HES upgrade)
  - High CO₂ intensity per floor area

Requires: outputs/county_profiles.csv (from 07_county_profile.py)

Outputs:
  outputs/equity_gap_county.csv   — Equity Gap Score per county
  outputs/equity_gap_bar.png      — Horizontal bar chart sorted by score
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from cli_logger import setup_script_logging

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(OUTPUT_DIR / f"{Path(__file__).stem}.log")

PROFILES_CSV  = OUTPUT_DIR / 'county_profiles.csv'
GAP_CSV       = OUTPUT_DIR / 'equity_gap_county.csv'
GAP_BAR_PNG   = OUTPUT_DIR / 'equity_gap_bar.png'

# ─────────────────────────────────────────────────────────────
# LOAD COUNTY PROFILES
# ─────────────────────────────────────────────────────────────
print('=' * 60)
print('  RETROFIT EQUITY GAP ANALYSIS')
print('=' * 60)
t0 = time.time()

if not PROFILES_CSV.exists():
    raise FileNotFoundError(
        f'{PROFILES_CSV} not found. Run 07_county_profile.py first.'
    )

print(f'\nLoading {PROFILES_CSV}...')
agg = pd.read_csv(PROFILES_CSV)
print(f'  Loaded: {len(agg)} counties')

# ─────────────────────────────────────────────────────────────
# EQUITY GAP SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────
print('\nComputing Equity Gap Scores...')

# Convert percent columns to fractions for calculation
fp_rate      = agg['fuel_poverty_rate']      / 100.0   # fraction
retrofit_rate = agg['retrofit_rate']          / 100.0   # fraction
co2_vals     = agg['mean_co2_kg_m2']

# Normalise CO2 to [0,1]
co2_max = co2_vals.max()
co2_min = co2_vals.min()
co2_norm = (co2_vals - co2_min) / (co2_max - co2_min + 1e-9)

# Core equity gap: fuel poverty × unretrofitted fraction × CO2 intensity
raw_gap = fp_rate * (1.0 - retrofit_rate) * (co2_norm + 0.5)

# Normalise to 0–100 scale
gap_min = raw_gap.min()
gap_max = raw_gap.max()
equity_gap_score = ((raw_gap - gap_min) / (gap_max - gap_min + 1e-9) * 100).round(2)

agg['equity_gap_score'] = equity_gap_score

# Rank counties (1 = highest priority)
agg['equity_gap_rank'] = agg['equity_gap_score'].rank(ascending=False, method='min').astype(int)

# Sort by score descending
result = agg[['CountyName', 'total_homes', 'mean_ber', 'mean_co2_kg_m2',
              'retrofit_rate', 'fuel_poverty_rate',
              'wall_insulation_rate', 'roof_insulation_rate',
              'equity_gap_score', 'equity_gap_rank']] \
    .sort_values('equity_gap_score', ascending=False) \
    .reset_index(drop=True)

print('\n  Top 10 counties by Equity Gap Score (priority for policy intervention):')
print(f'  {"County":<15} {"EqGap":>6} {"FuelPov%":>9} {"Retrofit%":>10} '
      f'{"MeanBER":>8} {"CO2kg/m2":>9}')
print('  ' + '-' * 65)
for _, row in result.head(10).iterrows():
    print(f'  {row["CountyName"]:<15} {row["equity_gap_score"]:>6.1f} '
          f'{row["fuel_poverty_rate"]:>9.1f} {row["retrofit_rate"]:>10.1f} '
          f'{row["mean_ber"]:>8.1f} {row["mean_co2_kg_m2"]:>9.2f}')

result.to_csv(GAP_CSV, index=False)
print(f'\nEquity gap results saved to {GAP_CSV}')

# ─────────────────────────────────────────────────────────────
# EQUITY GAP BAR CHART
# ─────────────────────────────────────────────────────────────
print('\nGenerating equity gap bar chart...')

fig, ax = plt.subplots(figsize=(11, 10))

# Colour bars by fuel poverty rate
fp_norm_plot = (result['fuel_poverty_rate'] - result['fuel_poverty_rate'].min()) / \
               (result['fuel_poverty_rate'].max() - result['fuel_poverty_rate'].min() + 1e-9)
bar_colors = plt.cm.Reds(0.35 + fp_norm_plot * 0.55)

bars = ax.barh(
    result['CountyName'][::-1],
    result['equity_gap_score'][::-1],
    color=bar_colors[::-1],
    edgecolor='#333333',
    linewidth=0.5,
    height=0.72,
)

# Score labels on bars
for bar, score in zip(bars, result['equity_gap_score'][::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{score:.1f}', va='center', ha='left', fontsize=8.5, color='#222222')

# Threshold line — top quartile
q75 = result['equity_gap_score'].quantile(0.75)
ax.axvline(q75, color='#1565c0', linestyle='--', linewidth=1.2,
           label=f'Top-quartile threshold ({q75:.1f})')

# Colourbar legend for fuel poverty
sm = plt.cm.ScalarMappable(
    cmap='Reds',
    norm=plt.Normalize(
        vmin=result['fuel_poverty_rate'].min(),
        vmax=result['fuel_poverty_rate'].max()
    )
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Fuel Poverty Rate (%)', fontsize=10)

ax.set_xlabel('Equity Gap Score (0–100,  higher = greater policy need)', fontsize=12)
ax.set_title(
    'Retrofit Equity Gap Score by County\n'
    'Score = fuel poverty burden × retrofit penetration gap × CO₂ intensity',
    fontsize=13, pad=12
)
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, result['equity_gap_score'].max() * 1.12)

plt.tight_layout()
plt.savefig(GAP_BAR_PNG, dpi=150, bbox_inches='tight')
plt.close()
print(f'Equity gap bar chart saved to {GAP_BAR_PNG}')

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
top5 = result.head(5)['CountyName'].tolist()
print('\n' + '=' * 60)
print('EQUITY GAP SUMMARY')
print('=' * 60)
print(f'  Counties scored      : {len(result)}')
print(f'  Top-5 priority       : {", ".join(top5)}')
print(f'  Score range          : {result["equity_gap_score"].min():.1f} – '
      f'{result["equity_gap_score"].max():.1f}')
print(f'  Top-quartile cutoff  : {q75:.1f} '
      f'({(result["equity_gap_score"] >= q75).sum()} counties)')
print(f'\nTotal runtime: {time.time()-t0:.1f}s')
print('Done.')
