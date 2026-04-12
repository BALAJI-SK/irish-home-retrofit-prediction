"""
07_county_profile.py
====================
Generates 26-county policy profiles from the clean BER dataset.

Requires: outputs/clean_data_55col.parquet (produced by 01_clean_and_prepare.py)
          — must contain CountyName and the policy columns added in that script.

Outputs:
  outputs/county_profiles.csv       — per-county aggregated statistics
  outputs/county_summary_table.md   — markdown table for reports
  outputs/county_bubble_chart.png   — BER vs retrofit rate, sized by homes,
                                       coloured by fuel poverty risk
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cli_logger import setup_script_logging

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(OUTPUT_DIR / f"{Path(__file__).stem}.log")

PARQUET_PATH    = OUTPUT_DIR / 'clean_data_55col.parquet'
PROFILES_CSV    = OUTPUT_DIR / 'county_profiles.csv'
SUMMARY_MD      = OUTPUT_DIR / 'county_summary_table.md'
BUBBLE_PNG      = OUTPUT_DIR / 'county_bubble_chart.png'

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print('=' * 60)
print('  COUNTY PROFILE GENERATOR')
print('=' * 60)
t0 = time.time()

print(f'\nLoading {PARQUET_PATH}...')
df = pd.read_parquet(PARQUET_PATH)
print(f'  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols')

if 'CountyName' not in df.columns:
    raise ValueError(
        'CountyName column not found. Re-run 01_clean_and_prepare.py '
        'to generate the updated parquet with CountyName retained.'
    )

# ─────────────────────────────────────────────────────────────
# REQUIRED POLICY COLUMNS — check or derive
# ─────────────────────────────────────────────────────────────
REQUIRED = [
    'BerRating', 'EstCO2_kg_per_m2', 'Total_Annual_CO2_Tonnes',
    'is_retrofitted', 'fuel_poverty_risk', 'wall_insulated',
    'roof_insulated', 'heating_upgraded',
]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(
        f'Missing policy columns: {missing}. '
        'Re-run 01_clean_and_prepare.py to generate them.'
    )

# ─────────────────────────────────────────────────────────────
# COUNTY AGGREGATION
# ─────────────────────────────────────────────────────────────
print('\nAggregating by CountyName...')

agg = df.groupby('CountyName').agg(
    total_homes           = ('BerRating',              'count'),
    mean_ber              = ('BerRating',              'mean'),
    median_ber            = ('BerRating',              'median'),
    pct_AB                = ('BerRating',              lambda x: (x <= 100).mean() * 100),
    pct_EFG               = ('BerRating',              lambda x: (x > 300).mean()  * 100),
    mean_co2_kg_m2        = ('EstCO2_kg_per_m2',       'mean'),
    total_co2_kt          = ('Total_Annual_CO2_Tonnes', lambda x: x.sum() / 1000),
    retrofit_rate         = ('is_retrofitted',         'mean'),
    fuel_poverty_rate     = ('fuel_poverty_risk',      'mean'),
    wall_insulation_rate  = ('wall_insulated',         'mean'),
    roof_insulation_rate  = ('roof_insulated',         'mean'),
    heating_upgrade_rate  = ('heating_upgraded',       'mean'),
).reset_index()

# Convert rates to percentages
for col in ['retrofit_rate', 'fuel_poverty_rate',
            'wall_insulation_rate', 'roof_insulation_rate',
            'heating_upgrade_rate']:
    agg[col] = (agg[col] * 100).round(2)

agg['mean_ber']       = agg['mean_ber'].round(1)
agg['median_ber']     = agg['median_ber'].round(1)
agg['mean_co2_kg_m2'] = agg['mean_co2_kg_m2'].round(2)
agg['total_co2_kt']   = agg['total_co2_kt'].round(1)
agg['pct_AB']         = agg['pct_AB'].round(1)
agg['pct_EFG']        = agg['pct_EFG'].round(1)

# Sort by mean BER descending (worst first — policy targeting)
agg = agg.sort_values('mean_ber', ascending=False).reset_index(drop=True)

print(f'  Aggregated {len(agg)} counties')
print(f'\n  Top 5 counties by mean BER (worst first):')
for _, row in agg.head(5).iterrows():
    print(f'    {row["CountyName"]:<15s} BER={row["mean_ber"]:.1f}  '
          f'retrofit={row["retrofit_rate"]:.1f}%  '
          f'fuel_poverty={row["fuel_poverty_rate"]:.1f}%')

# Save CSV
agg.to_csv(PROFILES_CSV, index=False)
print(f'\nCounty profiles saved to {PROFILES_CSV}')

# ─────────────────────────────────────────────────────────────
# MARKDOWN SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print('\nGenerating markdown summary table...')

md_lines = [
    '# County Policy Profiles — Irish BER Stock',
    '',
    f'*{len(df):,} BER certificates | {len(agg)} counties | Source: SEAI BER Database*',
    '',
    '| County | Homes | Mean BER | % A-B | % E-G | CO₂ kg/m²/yr | '
    'Retrofit % | Fuel Poverty % | Wall Ins. % | Roof Ins. % |',
    '|--------|------:|--------:|------:|------:|-------------:|'
    '----------:|---------------:|------------:|------------:|',
]

for _, row in agg.iterrows():
    md_lines.append(
        f'| {row["CountyName"]} '
        f'| {int(row["total_homes"]):,} '
        f'| {row["mean_ber"]:.1f} '
        f'| {row["pct_AB"]:.1f}% '
        f'| {row["pct_EFG"]:.1f}% '
        f'| {row["mean_co2_kg_m2"]:.1f} '
        f'| {row["retrofit_rate"]:.1f}% '
        f'| {row["fuel_poverty_rate"]:.1f}% '
        f'| {row["wall_insulation_rate"]:.1f}% '
        f'| {row["roof_insulation_rate"]:.1f}% |'
    )

md_lines += [
    '',
    '**Notes:**',
    '- Mean BER: kWh/m²/yr primary energy (lower = more efficient)',
    '- % A-B: share of homes with BER ≤ 100 kWh/m²/yr',
    '- % E-G: share of homes with BER > 300 kWh/m²/yr (poor performers)',
    '- CO₂: estimated from BER × SEAI fuel emission factor',
    '- Retrofit %: wall insulated OR roof insulated OR heating upgraded OR solar hot water',
    '- Fuel Poverty %: BER > 300 AND oil/coal/peat/solid fuel heating',
]

md_text = '\n'.join(md_lines)
with open(SUMMARY_MD, 'w', encoding='utf-8') as f:
    f.write(md_text)
print(f'Summary table saved to {SUMMARY_MD}')

# ─────────────────────────────────────────────────────────────
# BUBBLE CHART
# ─────────────────────────────────────────────────────────────
print('\nGenerating county bubble chart...')

fig, ax = plt.subplots(figsize=(14, 10))

# Normalise bubble sizes relative to county with most homes
max_homes  = agg['total_homes'].max()
bubble_sz  = (agg['total_homes'] / max_homes * 2500).clip(lower=80)

# Colour by fuel poverty rate (quartile bins)
fp_vals    = agg['fuel_poverty_rate'].values
fp_norm    = (fp_vals - fp_vals.min()) / (fp_vals.max() - fp_vals.min() + 1e-9)
cmap       = plt.cm.RdYlGn_r  # red = high fuel poverty, green = low
colors     = cmap(fp_norm)

sc = ax.scatter(
    agg['mean_ber'],
    agg['retrofit_rate'],
    s=bubble_sz,
    c=fp_norm,
    cmap='RdYlGn_r',
    alpha=0.80,
    edgecolors='#333333',
    linewidths=0.6,
)

# County name annotations
for _, row in agg.iterrows():
    ax.annotate(
        row['CountyName'],
        (row['mean_ber'], row['retrofit_rate']),
        fontsize=7.5,
        ha='center',
        va='bottom',
        xytext=(0, 5),
        textcoords='offset points',
    )

# Colourbar for fuel poverty
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label('Fuel Poverty Risk Rate', fontsize=11)
cbar_ticks = np.linspace(0, 1, 5)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(
    [f'{fp_vals.min() + t*(fp_vals.max()-fp_vals.min()):.1f}%'
     for t in cbar_ticks]
)

# Reference lines
ax.axvline(df['BerRating'].mean(), color='#555555', linestyle='--',
           linewidth=1.0, label=f'National mean BER ({df["BerRating"].mean():.0f})')
ax.axhline(agg['retrofit_rate'].mean(), color='#2255aa', linestyle='--',
           linewidth=1.0, label=f'Avg retrofit rate ({agg["retrofit_rate"].mean():.1f}%)')

# Quadrant shading — annotate policy quadrants
xlim_mid = df['BerRating'].mean()
ylim_mid = agg['retrofit_rate'].mean()
ax.text(xlim_mid * 0.72, agg['retrofit_rate'].max() * 0.92,
        'High retrofit\nLow BER\n✓ Leaders',
        ha='center', va='top', fontsize=8,
        color='#1a6e1a', alpha=0.7)
ax.text(agg['mean_ber'].max() * 0.97, agg['retrofit_rate'].max() * 0.92,
        'High retrofit\nHigh BER\n⚠ In progress',
        ha='right', va='top', fontsize=8,
        color='#b37a00', alpha=0.7)
ax.text(xlim_mid * 0.72, agg['retrofit_rate'].min() * 1.1 + 1,
        'Low retrofit\nLow BER\n● Efficient stock',
        ha='center', va='bottom', fontsize=8,
        color='#555555', alpha=0.7)
ax.text(agg['mean_ber'].max() * 0.97, agg['retrofit_rate'].min() * 1.1 + 1,
        'Low retrofit\nHigh BER\n✗ Priority targets',
        ha='right', va='bottom', fontsize=8,
        color='#cc2222', alpha=0.7)

# Size legend — cap display sizes so labels don't overlap in the legend box
LEGEND_MAX_SZ = 400  # legend marker cap (display only, not data)
for n_homes, label in [(50_000, '50k'), (200_000, '200k'), (400_000, '400k')]:
    sz = n_homes / max_homes * LEGEND_MAX_SZ
    ax.scatter([], [], s=sz, c='#999999', alpha=0.7, edgecolors='#333333',
               linewidths=0.6, label=f'{label} homes')

ax.set_xlabel('Mean BER Rating (kWh/m²/yr primary energy — lower is better)', fontsize=12)
ax.set_ylabel('Retrofit Rate (% of homes with any retrofit measure)', fontsize=12)
ax.set_title('Irish County Building Stock — BER Performance vs Retrofit Uptake\n'
             'Bubble size = number of homes | Colour = fuel poverty risk rate',
             fontsize=13, pad=14)
ax.legend(loc='lower right', fontsize=9, framealpha=0.85,
          labelspacing=1.2, handletextpad=1.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BUBBLE_PNG, dpi=150, bbox_inches='tight')
plt.close()
print(f'Bubble chart saved to {BUBBLE_PNG}')

# ─────────────────────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('COUNTY PROFILE SUMMARY')
print('=' * 60)
print(f'  Counties processed       : {len(agg)}')
print(f'  Total BER certificates   : {agg["total_homes"].sum():,}')
print(f'  National mean BER        : {agg["mean_ber"].mean():.1f} kWh/m2/yr')
print(f'  Best mean BER county     : {agg.iloc[-1]["CountyName"]} ({agg.iloc[-1]["mean_ber"]:.1f})')
print(f'  Worst mean BER county    : {agg.iloc[0]["CountyName"]} ({agg.iloc[0]["mean_ber"]:.1f})')
print(f'  Highest retrofit rate    : {agg.nlargest(1,"retrofit_rate").iloc[0]["CountyName"]} '
      f'({agg["retrofit_rate"].max():.1f}%)')
print(f'  Lowest retrofit rate     : {agg.nsmallest(1,"retrofit_rate").iloc[0]["CountyName"]} '
      f'({agg["retrofit_rate"].min():.1f}%)')
print(f'  Highest fuel poverty     : {agg.nlargest(1,"fuel_poverty_rate").iloc[0]["CountyName"]} '
      f'({agg["fuel_poverty_rate"].max():.1f}%)')
print(f'\nTotal runtime: {time.time()-t0:.1f}s')
print('Done.')
