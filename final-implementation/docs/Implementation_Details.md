# Implementation Details — Irish Home Retrofit Equity Pipeline

**Project:** Mining the Irish National Building Stock for Carbon Reduction Equity  
**Dataset:** SEAI BER Public Search Database (~1.35 million building energy certificates)  
**Language:** Python 3.9 | Libraries: pandas, numpy, LightGBM, scikit-learn, SHAP, matplotlib

---

## Table of Contents

1. [System Architecture and Data Flow](#1-system-architecture-and-data-flow)
2. [Script 01 — Data Cleaning and Feature Engineering](#2-script-01--data-cleaning-and-feature-engineering)
3. [Script 02 — County Policy Profiles](#3-script-02--county-policy-profiles)
4. [Script 03 — Model Training (Regressor + Classifier)](#4-script-03--model-training-regressor--classifier)
5. [Script 04 — Retrofit Equity Gap Analysis](#5-script-04--retrofit-equity-gap-analysis)
6. [Script 05 — XAI Explainer (SHAP)](#6-script-05--xai-explainer-shap)
7. [Utility — cli_logger.py](#7-utility--cli_loggerpy)
8. [Complete Output Artefact Reference](#8-complete-output-artefact-reference)
9. [Key Design Decisions and Rationale](#9-key-design-decisions-and-rationale)

---

## 1. System Architecture and Data Flow

### 1.1 Pipeline Topology

The pipeline is strictly sequential. Each script reads from the previous script's outputs:

```
RAW DATA SOURCES
├── Col Final with County.parquet   (46 cols, ~1.35M rows — pre-filtered BER base)
└── Public Search Data.csv          (211 cols, ~1.6M rows — full SEAI BER database)
         │
         ▼
01_clean_and_prepare.py
         │  outputs/clean_data_55col.parquet  (62 cols, ~1.35M rows)
         │
         ├──► 02_county_profile.py
         │         │  outputs/county_profiles.csv
         │         │  outputs/county_summary_table.md
         │         │  outputs/county_bubble_chart.png
         │         │
         │         └──► 04_equity_gap.py
         │                   outputs/equity_gap_county.csv
         │                   outputs/equity_gap_bar.png
         │
         └──► 03_train_model.py
                   │  outputs/lgbm_model.pkl
                   │  outputs/lgbm_classifier.pkl
                   │  outputs/model_report.txt
                   │  outputs/classifier_report.txt
                   │  outputs/roc_curve.png
                   │  outputs/pr_curve.png
                   │  outputs/shap_bar.png
                   │  outputs/shap_summary.png
                   │  outputs/shap_summary_classifier.png
                   │  outputs/feature_importance.csv
                   │  outputs/shap_values_global.csv
                   │
                   └──► 05_xai_explainer.py
                             outputs/scenario_reports/{measure}_{n}_shap_before.png
                             outputs/scenario_reports/{measure}_{n}_shap_after.png
                             outputs/scenario_reports/{measure}_{n}_delta_shap.csv
                             outputs/xai_summary.csv
                             outputs/shap_summary_classifier.png
                             outputs/county_shap_avg.csv
```

### 1.2 Execution Order

```bash
python3 01_clean_and_prepare.py   # ~2–5 min
python3 02_county_profile.py      # ~30 s
python3 03_train_model.py         # ~10–20 min
python3 04_equity_gap.py          # ~5 s
python3 05_xai_explainer.py       # ~3–5 min
```

### 1.3 Logging

Every script calls `setup_script_logging(log_path)` from `cli_logger.py` immediately after path setup. This redirects both `sys.stdout` and `sys.stderr` to a `Tee` object that writes simultaneously to the terminal and a `.log` file under `outputs/`. Log files are opened in **append** mode so successive runs accumulate history.

---

## 2. Script 01 — Data Cleaning and Feature Engineering

**File:** `01_clean_and_prepare.py`  
**Input:** Two raw sources (parquet + CSV)  
**Output:** `outputs/clean_data_55col.parquet`, `outputs/cleaning_report.txt`

### 2.1 Data Sources and Merge Strategy

The pipeline merges two datasets using a **positional merge** strategy:

| Source | Format | Columns | Rows | Purpose |
|--------|--------|---------|------|---------|
| `Col Final with County.parquet` | Parquet (Snappy) | 46 | ~1.35M | Pre-filtered BER base with CountyName |
| `Public Search Data.csv` | CSV (Latin-1) | 211 | ~1.6M | Supplementary fuel/scheme columns |

**Row alignment method:** Both sources are filtered with identical bounds:
- `BerRating ∈ [0.0, 2000.0]`
- `Year_of_Construction ∈ [1700, 2026]`

After filtering, both DataFrames are `reset_index(drop=True)` so that positional index 0…N-1 refers to the same physical dwelling in both sources. If the row counts differ after filtering (possible if the CSV export date differs from the parquet), the pipeline trims to `min(len_parquet, len_csv)` with a warning.

**Columns pulled from CSV** (7 columns absent from the parquet base):

| CSV Column Name | Target Column Name | Reason |
|-----------------|--------------------|--------|
| `MainSpaceHeatingFuel` | `MainSpaceHeatingFuel` | Required for heat pump simulation and CO₂ estimation |
| `MainWaterHeatingFuel` | `MainWaterHeatingFuel` | Water heating scenario analysis |
| `SolarHotWaterHeating` | `SolarHotWaterHeating` | Solar hot water retrofit flag |
| `LowEnergyLightingPercent` | `LowEnergyLightingPercent` | LED upgrade scenario |
| `GroundFloorArea(sq m)` | `GroundFloorAreasq_m` | Needed for FabricHeatLossPerM2 computation |
| `HESSchemeUpgrade` | `HESSchemeUpgrade` | Pre-retrofit HES scheme participation flag |
| `FirstEnergyType_Description` | `FirstEnergyType_Description` | Renewable energy type descriptor |

### 2.2 Leakage Removal

Eight columns are removed before model training. These are DEAP intermediate outputs or high-VIF features that would constitute data leakage (the model would be learning the output calculation, not physical building properties):

```python
DROP_LEAKY = [
    'DistributionLosses',      # DEAP-calculated output
    'HSEffAdjFactor',          # DEAP intermediate, VIF > 25
    'WHEffAdjFactor',          # DEAP intermediate, VIF > 25
    'SHRenewableResources',    # DEAP intermediate, VIF > 25
    'WHRenewableResources',    # DEAP intermediate, VIF > 25
    'FabricHeatLossPerM2',     # Ablation: top-1 SHAP — replaced by FabricHeatLossProxy
    'Year_of_Construction',    # Ablation: top-2 SHAP — replaced by AgeBand
    'HSMainSystemEfficiency',  # Ablation: top-3 SHAP — replaced by IsHeatPump
]
```

**Important ordering constraint:** `IsHeatPump` and `AgeBand` are derived from `HSMainSystemEfficiency` and `Year_of_Construction` respectively. Feature engineering for these two must occur **before** the drop step (STEP 2 precedes STEP 3 in the pipeline). This is a deliberate structural decision — failing to respect this order would silently produce incorrect (zero) `IsHeatPump` values.

**Note on CountyName:** CountyName is intentionally **not** dropped. It is excluded from the model's feature matrix in `03_train_model.py` via `NON_FEATURE_COLS`, but retained in the parquet for county-level analysis in scripts 02 and 05.

### 2.3 Processing Steps (9 Steps)

#### STEP 1 — Load and Validate Parquet
Load the pre-filtered parquet. Apply row filters as a sanity check; log any unexpected out-of-range rows and filter them.

#### STEP 2 — Early Feature Engineering (Pre-Drop)
Must execute before STEP 3. Derives:

**`IsHeatPump`** (int8, 0/1):
```python
IsHeatPump = (HSMainSystemEfficiency > 100)
```
DEAP assigns heat pumps a system efficiency > 100% (representing the COP effect). This threshold captures air-source and ground-source heat pumps. Later in STEP 6, this flag is OR'd with a string match on `MainSpaceHeatingFuel` to catch any heat pumps not flagged by efficiency alone:
```python
IsHeatPump = IsHeatPump OR ('Heat Pump' in MainSpaceHeatingFuel)
```

**`AgeBand`** (string categorical):
Vintage is encoded as an ordered categorical following DEAP construction period brackets. The bins and labels are:

| Bin Range | Label |
|-----------|-------|
| 0–1900 | Pre1900 |
| 1901–1930 | 1900-1929 |
| 1931–1950 | 1930-1949 |
| 1951–1967 | 1950-1966 |
| 1968–1978 | 1967-1977 |
| 1979–1983 | 1978-1982 |
| 1984–1994 | 1983-1993 |
| 1995–2000 | 1994-1999 |
| 2001–2005 | 2000-2004 |
| 2006–2011 | 2005-2010 |
| 2012–2016 | 2011-2015 |
| 2017+ | 2016+ |

#### STEP 3 — Drop Leaky Columns
Iterates `DROP_LEAKY` and removes each column if present. Logs whether each column was present or already absent.

#### STEP 4 — CSV Supplement Merge
Reads only the 9 needed columns from the 211-column CSV (using `usecols` for memory efficiency). Applies identical row filters. Performs positional merge by assigning `df_csv[col].values` (numpy array, not index-aligned) to `df[out_col]`. Releases the CSV DataFrame with `del df_csv`.

**Memory note:** The full CSV is ~600MB on disk. Using `usecols` reduces the in-memory footprint to ~80MB for the 9 columns. The `low_memory=False` flag prevents pandas from inferring mixed dtypes per-chunk, which would cause incorrect type assignments for numeric-looking string columns.

#### STEP 5 — Null Imputation for New Columns

| Column | Strategy | Rationale |
|--------|----------|-----------|
| `SolarHotWaterHeating` | Fill with 'NO' | MNAR: null means no system installed |
| `LowEnergyLightingPercent` | Fill with 0.0 | MNAR: null means 0% low-energy lighting |
| `GroundFloorAreasq_m` | Fill with median | MAR: assessor omission |
| `MainSpaceHeatingFuel` | Fill with mode | MAR: fuel is usually determinable |
| `MainWaterHeatingFuel` | Fill with mode | MAR: same as above |
| `HESSchemeUpgrade` | Fill with 'No' | MNAR: null means no upgrade received |
| `FirstEnergyType_Description` | Fill with 'None' | MNAR: null means no renewable system |

String columns are stripped of leading/trailing whitespace before imputation.

#### STEP 6 — Secondary Feature Engineering

**`HasSolarWaterHeating`** (int8): `SolarHotWaterHeating.upper() ∈ {YES, Y}`

**`HasRoofInsulation`** (int8): `UValueRoof ≤ 0.16 W/m²K`  
Threshold sourced from Irish Building Regulations Part L (TGD L 2021): maximum permitted U-value for roofs in new dwellings is 0.16 W/m²K.

**`HasWallInsulation`** (int8): `UValueWall ≤ 0.37 W/m²K`  
Threshold reflects the Irish 2011 Building Regulations Part L external wall standard.

**`HasDoubleGlazing`** (int8): `UValueWindow ≤ 2.0 W/m²K`  
Standard double-glazed unit U-value; triple glazing ≈ 0.8–1.2 W/m²K.

**`WindowToWallRatio`** (float32):
```
WindowToWallRatio = WindowArea / WallArea   (0 if WallArea = 0)
```

**`FabricHeatLossProxy`** (float32):
```
FabricHeatLossProxy = Σ (UValue_element × Area_element)
```
Summed over 5 building elements: Wall, Roof, Floor, Window, Door. Units: W/K (total fabric heat loss coefficient). This is the engineering equivalent of the DEAP `FabricHeatLoss` intermediate output, but computed only from raw U-values and areas (not from DEAP's post-processed output), avoiding leakage.

#### STEP 7 — Policy Columns

**`EstCO2_kg_per_m2`** (float32):
```
EstCO2_kg_per_m2 = BerRating × CO2_factor(MainSpaceHeatingFuel)
```
Where `BerRating` is in kWh/m²/yr (primary energy) and `CO2_factor` is the SEAI Ireland emission factor in kg CO₂/kWh. The mapping uses `MainSpaceHeatingFuel` as a proxy for the dominant energy vector. A default factor of 0.25 kg/kWh is used for unlisted fuels.

**SEAI emission factors used (kg CO₂/kWh primary energy):**

| Fuel | Factor | Source |
|------|-------:|--------|
| Mains Gas / Natural Gas | 0.2040 | SEAI Energy in Ireland 2023 |
| Heating Oil | 0.2630 | SEAI |
| Electricity (all tariffs) | 0.2950 | SEAI (Irish grid 2023) |
| Solid Multi-Fuel / House Coal / Anthracite | 0.3400 | SEAI |
| Manufactured Smokeless Fuel | 0.3420 | SEAI |
| Sod Peat / Milled Peat | 0.3550 | SEAI |
| Peat Briquettes | 0.3620 | SEAI |
| Bulk/Bottled LPG | 0.2140 | SEAI |
| Wood Pellets | 0.0150 | SEAI (biogenic) |
| Wood Logs / Wood Chips | 0.0120 | SEAI (biogenic) |
| Biodiesel / Bioethanol | 0.0180 | SEAI (biogenic) |

**`Total_Annual_CO2_Tonnes`** (float32):
```
Total_Annual_CO2_Tonnes = EstCO2_kg_per_m2 × max(FloorArea, 0) / 1000
```

**`wall_insulated`** (int8): `UValueWall ≤ 0.37` — explicit policy label (same threshold as `HasWallInsulation`, different semantic name for policy analysis).

**`roof_insulated`** (int8): `UValueRoof ≤ 0.16` — explicit policy label.

**`heating_upgraded`** (int8):
```
heating_upgraded = IsHeatPump OR (HESSchemeUpgrade ≠ 'No'/'0'/'')
```
`HESSchemeUpgrade` may arrive from the CSV as either a string ('Yes'/'No') or binary integer (1/0). Both cases are handled with a dtype check.

**`is_retrofitted`** (int8) — composite retrofit label used as classifier target:
```
is_retrofitted = wall_insulated OR roof_insulated OR heating_upgraded 
                 OR HasSolarWaterHeating
```
This is the binary classification target for `03_train_model.py`.

**`fuel_poverty_risk`** (int8):
```
fuel_poverty_risk = (BerRating > 300) AND (MainSpaceHeatingFuel ∈ FUEL_POVERTY_FUELS)
```
Where `FUEL_POVERTY_FUELS = {Heating Oil, Oil, Solid Multi-Fuel, House Coal, Anthracite, Manufactured Smokeless Fuel, Sod Peat, Milled Peat, Peat Briquettes, Bulk LPG, Bottled LPG, LPG}`. These fuels are expensive and/or price-volatile, creating risk of energy unaffordability when combined with a poorly-rated home.

#### STEP 8 — Dtype Optimisation
All `float64` columns are downcast to `float32` (halving memory). All `int64` columns are downcast to `int32`.

#### STEP 9 — Save Parquet
Output is written with Snappy compression. Snappy was chosen over gzip/brotli for its faster decompression speed at the cost of slightly larger files — important given that downstream scripts all need to load this file.

### 2.4 Validation Report
The cleaning report (`cleaning_report.txt`) records:
- Leakage verification (confirms dropped columns are absent)
- High-correlation check (flags any numeric feature with |corr| > 0.80 with BerRating)
- Target statistics (mean, median, std, grade band distributions)
- Null check (any remaining nulls)
- Engineered feature summaries
- Policy column value distributions
- Fuel distribution (top 10)
- Age band distribution
- County distribution (top 10)
- Full column list

---

## 3. Script 02 — County Policy Profiles

**File:** `02_county_profile.py`  
**Input:** `outputs/clean_data_55col.parquet`  
**Output:** `county_profiles.csv`, `county_summary_table.md`, `county_bubble_chart.png`

### 3.1 Aggregation

The script groups the full dataset by `CountyName` and computes 12 aggregate statistics per county using a single `groupby().agg()` call:

| Output Column | Source Column | Aggregation |
|---------------|---------------|-------------|
| `total_homes` | BerRating | count |
| `mean_ber` | BerRating | mean |
| `median_ber` | BerRating | median |
| `pct_AB` | BerRating | fraction ≤ 100, × 100 |
| `pct_EFG` | BerRating | fraction > 300, × 100 |
| `mean_co2_kg_m2` | EstCO2_kg_per_m2 | mean |
| `total_co2_kt` | Total_Annual_CO2_Tonnes | sum / 1000 (kilotonnes) |
| `retrofit_rate` | is_retrofitted | mean × 100 |
| `fuel_poverty_rate` | fuel_poverty_risk | mean × 100 |
| `wall_insulation_rate` | wall_insulated | mean × 100 |
| `roof_insulation_rate` | roof_insulated | mean × 100 |
| `heating_upgrade_rate` | heating_upgraded | mean × 100 |

Counties are sorted by `mean_ber` descending (worst-performing first) to facilitate policy targeting.

### 3.2 Bubble Chart Construction

The chart encodes four dimensions simultaneously:
- **X-axis:** `mean_ber` (energy performance)
- **Y-axis:** `retrofit_rate` (retrofit uptake)
- **Bubble area:** proportional to `total_homes` (scaled to max 2500 pts² for the largest county, minimum 80 pts²)
- **Colour:** `fuel_poverty_rate` mapped to the `RdYlGn_r` diverging colormap (red = high fuel poverty)

**Quadrant logic:** Reference lines are drawn at the national mean BER and the mean retrofit rate across counties. Text annotations label four quadrants:

| Quadrant | Condition | Policy label |
|----------|-----------|-------------|
| Top-left | Low BER, High retrofit | Leaders |
| Top-right | High BER, High retrofit | In progress |
| Bottom-left | Low BER, Low retrofit | Efficient stock |
| Bottom-right | High BER, Low retrofit | **Priority targets** |

County names are annotated directly on each bubble using `ax.annotate` with a 5-point vertical offset to avoid overlap with the bubble edge.

---

## 4. Script 03 — Model Training (Regressor + Classifier)

**File:** `03_train_model.py`  
**Input:** `outputs/clean_data_55col.parquet`  
**Output:** Multiple model artefacts, metrics, and plots (see Section 8)

### 4.1 Feature Exclusions

The following columns are excluded from **both** the regressor and classifier feature matrices via `NON_FEATURE_COLS`:

| Column | Reason for Exclusion |
|--------|---------------------|
| `BerRating` | Regression target |
| `is_retrofitted` | Classifier target |
| `CountyName` | Geographic identifier — not a building physical feature |
| `EstCO2_kg_per_m2` | Derived from BerRating — would be leaky for the regressor |
| `Total_Annual_CO2_Tonnes` | Derived from BerRating × FloorArea — leaky |
| `wall_insulated` | Redundant with `HasWallInsulation` (same threshold) |
| `roof_insulated` | Redundant with `HasRoofInsulation` (same threshold) |
| `heating_upgraded` | Component of `is_retrofitted` — would be a direct signal |
| `fuel_poverty_risk` | Derived from BerRating — leaky |

### 4.2 Categorical Encoding

All `object` and `category` dtype columns are encoded with scikit-learn's `OrdinalEncoder`:
```python
OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)
```
Unknown categories at inference time are mapped to -1. LightGBM handles this naturally as it splits on numeric boundaries. The fitted encoders are stored in the pickle artefact for consistent use at inference time in `05_xai_explainer.py`.

### 4.3 Train/Validation/Test Split

A stratified **70/15/15 split** is implemented as two sequential random splits:
```python
# Step 1: 85% trainval / 15% test
X_trainval, X_test = train_test_split(X, test_size=0.15, random_state=42)

# Step 2: within trainval, ~82.4% train / ~17.6% val = 70% / 15% of total
X_train, X_val = train_test_split(X_trainval, test_size=0.15/0.85, random_state=42)
```
All three targets (`y` log-transformed, `y_raw` original scale, `y_clf` binary) are split simultaneously using the same index arrays, ensuring consistency across all downstream evaluations.

### 4.4 LightGBM Regressor

**Target transformation:** `y = log1p(BerRating)`. The right-skewed BER distribution (range 0–2000, median ~175) benefits from log transformation, which stabilises variance and improves RMSE on both high- and low-BER homes.

**Hyperparameters** (validated on a prior 118-column pipeline):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 1500 | Sufficient depth for 1.35M rows with early stopping |
| `learning_rate` | 0.08 | Moderate — balanced with 1500 trees |
| `num_leaves` | 127 | 2^7 − 1; captures complex interactions without overfitting |
| `max_depth` | 8 | Caps tree depth to prevent over-deep splits |
| `min_child_samples` | 100 | Minimum 100 samples per leaf; prevents overfitting on small groups |
| `subsample` | 0.9 | Row subsampling per tree — reduces overfitting |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `reg_alpha` | 0.1 | L1 regularisation |
| `reg_lambda` | 0.1 | L2 regularisation |
| `random_state` | 42 | Reproducibility |

**Training procedure:** Final model is trained on `train + val` combined (85% of data). Early stopping with patience=50 is applied against the validation set to determine the optimal number of trees. This ensures the test set is never seen during training or early stopping.

**5-Fold Cross-Validation:** Performed on the training split only (not train+val). LightGBM is re-instantiated fresh for each fold. Predictions are back-transformed via `expm1` before computing R² on the original kWh/m²/yr scale.

**Metrics reported:** R², RMSE, MAE — all on the original kWh/m²/yr scale after `expm1` back-transformation.

**Comparison models:**
- **Random Forest:** `n_estimators=300, max_depth=20, min_samples_leaf=10` — trained on train split only (not train+val) for honest comparison.
- **Ridge Regression:** `alpha=1.0`, with median imputation and standard scaling in a sklearn `Pipeline` — serves as linear baseline.

### 4.5 LightGBM Classifier (is_retrofitted)

**Target:** Binary `is_retrofitted` (0 = not retrofitted, 1 = retrofitted).

**Class imbalance handling:** `scale_pos_weight` is computed as:
```python
scale_pos_weight = count(negative) / count(positive)
```
This is the recommended LightGBM approach for imbalanced binary classification — it upweights the minority class in the loss function without oversampling.

**Hyperparameters:**

| Parameter | Value | Note |
|-----------|-------|------|
| `n_estimators` | 800 | Fewer trees than regressor (simpler target) |
| `learning_rate` | 0.08 | Same as regressor |
| `num_leaves` | 63 | 2^6 − 1; slightly simpler than regressor |
| `max_depth` | 7 | One level shallower |
| `min_child_samples` | 100 | Same floor as regressor |
| `subsample` | 0.85 | Slightly more aggressive row subsampling |
| `colsample_bytree` | 0.8 | Same as regressor |
| `scale_pos_weight` | computed | Handles class imbalance |

**Evaluation metrics:**
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve — measures rank discrimination regardless of threshold.
- **Average Precision (AP):** Area under the Precision-Recall curve — more informative than ROC-AUC for imbalanced classes.
- **Classification report:** Precision, recall, F1 for both classes at the default 0.5 threshold.

**Curves saved:** `roc_curve.png` and `pr_curve.png` with the no-skill baseline (diagonal for ROC; class prevalence line for PR).

**Classifier SHAP:** SHAP values are computed on 5,000 randomly sampled test rows using `shap.TreeExplainer`. For LightGBM classifiers, `shap_values()` may return a list `[neg_class_shap, pos_class_shap]`; the positive class array (index 1) is used for the beeswarm plot.

### 4.6 SHAP for Regressor (Global)

A stratified random sample of 10,000 rows is drawn from the **full dataset** (not just test set) to provide a global view of feature importance. The sample uses `np.random.default_rng(RANDOM_SEED + 2)` to ensure reproducibility independent of the train/test split seed.

**Critical fix:** The SHAP sample is restricted to exactly `X_train.columns.tolist()` before passing to the explainer — this prevents the `categorical_feature mismatch` error that occurs if policy/county columns are passed to a model that was never trained on them.

Outputs:
- `shap_values_global.csv` — raw SHAP values matrix (10K × n_features)
- `shap_bar.png` — horizontal bar chart of top 30 features by mean |SHAP|
- `shap_summary.png` — SHAP beeswarm (summary plot) for top 30 features

### 4.7 Model Artefact Schema

**`lgbm_model.pkl`** (regressor):
```python
{
    'model':          lgb.LGBMRegressor,    # fitted model
    'encoders':       dict[str, OrdinalEncoder],  # one per categorical column
    'cat_cols':       list[str],            # categorical column names
    'num_cols':       list[str],            # numeric column names
    'feature_names':  list[str],            # ordered feature list
    'params':         dict,                 # LGBM_PARAMS
    'results':        dict,                 # train/val/test R²/RMSE/MAE
    'cv_mean_r2':     float,
    'cv_std_r2':      float,
}
```

**`lgbm_classifier.pkl`** (classifier):
```python
{
    'model':          lgb.LGBMClassifier,
    'encoders':       dict[str, OrdinalEncoder],  # same encoders as regressor
    'cat_cols':       list[str],
    'num_cols':       list[str],
    'feature_names':  list[str],
    'params':         dict,                 # CLF_PARAMS
    'roc_auc':        float,
    'avg_precision':  float,
}
```

---

## 5. Script 04 — Retrofit Equity Gap Analysis

**File:** `04_equity_gap.py`  
**Input:** `outputs/county_profiles.csv`  
**Output:** `equity_gap_county.csv`, `equity_gap_bar.png`

### 5.1 Equity Gap Score Formula

The Equity Gap Score is a composite index designed to identify counties where the intersection of fuel poverty, low retrofit uptake, and high carbon intensity creates the greatest unmet policy need.

**Step 1 — Component normalisation:**
```
fp_rate      = fuel_poverty_rate / 100          ∈ [0, 1]
retrofit_gap = 1 - (retrofit_rate / 100)         ∈ [0, 1]
co2_norm     = (co2 - co2_min) / (co2_max - co2_min)   ∈ [0, 1]
```

A small epsilon (1×10⁻⁹) is added to the denominator of `co2_norm` to prevent division by zero in edge cases.

**Step 2 — Raw composite score:**
```
raw_gap = fp_rate × retrofit_gap × (co2_norm + 0.5)
```

The `+ 0.5` offset on `co2_norm` ensures that even counties with the minimum CO₂ intensity still contribute positively to the score when they have high fuel poverty and low retrofit uptake. Without the offset, a county with `co2_norm = 0` would score zero regardless of fuel poverty or retrofit gap.

**Step 3 — Normalise to 0–100:**
```
equity_gap_score = (raw_gap - min(raw_gap)) / (max(raw_gap) - min(raw_gap)) × 100
```

**Interpretation:**
- Score of 100: maximum fuel poverty burden + maximum retrofit gap + maximum CO₂ intensity among all counties
- Score of 0: minimum combination of the three factors
- Top quartile (score ≥ Q75): counties recommended for priority policy intervention

**County rank:** Counties are ranked 1–26, where rank 1 = highest equity gap score (greatest policy need).

### 5.2 Bar Chart

Counties are displayed in descending order of equity gap score (highest at top). Bar colour intensity maps to `fuel_poverty_rate` using the `Reds` colormap (range 0.35–0.90 of the map, avoiding the very pale low end). A vertical dashed line marks the top-quartile threshold. Numeric score labels are appended to the right of each bar.

---

## 6. Script 05 — XAI Explainer (SHAP)

**File:** `05_xai_explainer.py`  
**Input:** `lgbm_model.pkl`, `lgbm_classifier.pkl`, `clean_data_55col.parquet`, `config/retrofit_measures.json`  
**Output:** Per-scenario SHAP plots + CSVs, classifier global SHAP, county SHAP averages

### 6.1 Section A — Regressor Delta SHAP (Scenario Analysis)

#### Candidate Selection
3 example dwellings are selected from homes with `BerRating > 200` (E-rated or worse) — these are the homes with the greatest retrofit potential and most illustrative SHAP changes. If fewer than 3 such homes exist, the full dataset is used.

#### prepare_X Function
Converts a raw DataFrame row into the feature matrix expected by the model:
1. Drops `BerRating` (target)
2. Applies the saved `OrdinalEncoder` to each categorical column
3. Casts numeric columns to `float32`
4. Fills any missing expected features with 0.0 (defensive padding)
5. Returns columns in the exact order of `feature_names` from the artefact

#### explain_retrofit Function
For each `(home, measure)` pair:

1. **Before state:** Apply `prepare_X` to the original row → `X_before`. Compute `shap_values(X_before)` → `sv_before` (1D array, one value per feature).

2. **After state:** Copy the row dict, apply the measure's `overrides` (e.g., set `UValueWall=0.21, HasWallInsulation=1`). If `recompute_derived=True`, call `recompute_derived()` to recompute `FabricHeatLossProxy`, `WindowToWallRatio`, `HasRoofInsulation`, `HasWallInsulation`, `HasDoubleGlazing` from the updated U-values/areas. Apply `prepare_X` → `X_after`. Compute `shap_values(X_after)` → `sv_after`.

3. **Delta:** `Δ_shap = sv_after − sv_before`. Features with large |Δ_shap| are the primary drivers of the BER change.

4. **Top drivers:** Sorted by |Δ_shap| descending. Top 10 reported.

The `recompute_derived` function recalculates all engineered features that depend on raw U-values and areas — this ensures that applying a wall insulation override (UValueWall: 1.0 → 0.21) also correctly updates `FabricHeatLossProxy` and `HasWallInsulation`.

#### Output Files (per measure × per example)
- `{measure_label}_{n}_shap_before.png` — horizontal bar chart of top 15 features by |SHAP| before retrofit, red bars = positive SHAP (increases BER), blue bars = negative SHAP (decreases BER)
- `{measure_label}_{n}_shap_after.png` — same after retrofit
- `{measure_label}_{n}_delta_shap.csv` — all features with `shap_before`, `shap_after`, `delta_shap`, `pct_of_total` (percentage of total BER saving attributable to each feature's SHAP change)

#### xai_summary.csv
Cross-measure summary table with one row per `(measure, example)` combination:
`measure, example, dwelling_type, year_built, county, baseline_ber, baseline_grade, retrofit_ber, retrofit_grade, saving_kwh, saving_pct, top_driver_feat, top_driver_delta`

BER grades are mapped using the SEAI 15-tier scheme:
A1(0–25), A2(25–50), A3(50–75), B1(75–100), B2(100–125), B3(125–150), C1(150–175), C2(175–200), C3(200–225), D1(225–260), D2(260–300), E1(300–340), E2(340–380), F(380–450), G(450+)

### 6.2 Section B — Classifier Global SHAP and County Attribution

After completing Section A, the script loads `lgbm_classifier.pkl` and computes SHAP on a 10,000-row random sample.

**`prepare_clf_X` function:** Drops the same `NON_FEATURE_COLS` as used in training (BerRating, is_retrofitted, CountyName, policy-derived cols), then applies the saved encoders and returns columns in `clf_feature_names` order.

**SHAP extraction:** `clf_explainer.shap_values(X_clf_shap)` returns either:
- A 2D array directly (newer SHAP versions with LightGBM)
- A list of two 2D arrays `[neg_class, pos_class]` (older SHAP versions)

The positive-class array is selected in both cases.

**County-level SHAP aggregation:**
```python
shap_df = DataFrame(shap_clf_vals, columns=clf_feature_names, index=sample_index)
shap_df['CountyName'] = df.iloc[sample_index]['CountyName']
county_shap = shap_df.groupby('CountyName').mean(numeric_only=True).abs()
county_shap['mean_abs_shap_overall'] = county_shap.mean(axis=1)
```

This produces a table where each row is a county and each column is a feature's mean absolute SHAP value for the retrofit classifier. The `mean_abs_shap_overall` column summarises the overall SHAP "activity" in each county — counties with higher values have building stocks where the classifier's predictions are more feature-driven (higher explainability leverage).

**Outputs:**
- `shap_summary_classifier.png` — beeswarm of top 25 features for positive class (retrofitted)
- `county_shap_avg.csv` — (26 counties × n_features + 1) matrix of mean |SHAP| values

---

## 7. Utility — cli_logger.py

A minimal logging utility that provides dual-stream logging without requiring the Python `logging` module configuration overhead.

### `Tee` Class
Wraps multiple file-like objects and multiplexes `write()` and `flush()` calls to all of them:
```python
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data): [f.write(data) for f in self.files]
    def flush(self): [f.flush() for f in self.files]
    def isatty(self): return any(f.isatty() for f in self.files)
```
The `isatty()` method is important for compatibility with libraries that check whether stdout is a terminal before enabling colour output (e.g., tqdm, rich).

### `setup_script_logging(log_path)` Function
```python
def setup_script_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'a', encoding='utf-8')
    header = f"\n\n=== {datetime.now().isoformat()} {log_path.name} ===\n"
    log_file.write(header)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
```

- Opens the log file in **append** mode (`'a'`), so repeated runs accumulate
- Writes a timestamped header before redirecting
- Replaces both `sys.stdout` and `sys.stderr` so that `print()`, uncaught exceptions, and library warnings all appear in the log
- The original `sys.stdout` is captured before replacement and included in the `Tee`, so output still appears in the terminal

---

## 8. Complete Output Artefact Reference

| File | Script | Format | Description |
|------|--------|--------|-------------|
| `clean_data_55col.parquet` | 01 | Parquet/Snappy | ~1.35M × 62 col cleaned dataset |
| `cleaning_report.txt` | 01 | Text | Validation report: leakage check, nulls, stats |
| `county_profiles.csv` | 02 | CSV | 26-county aggregated statistics (12 metrics) |
| `county_summary_table.md` | 02 | Markdown | Formatted table for reports/papers |
| `county_bubble_chart.png` | 02 | PNG (150 dpi) | BER vs retrofit rate, 4-quadrant policy chart |
| `lgbm_model.pkl` | 03 | Pickle | LightGBM regressor + encoders + metadata |
| `lgbm_classifier.pkl` | 03 | Pickle | LightGBM classifier + encoders + metadata |
| `model_report.txt` | 03 | Text | Full R²/RMSE/MAE for all 3 models + CV results |
| `classifier_report.txt` | 03 | Text | ROC-AUC, Avg Precision, per-class F1 |
| `feature_importance.csv` | 03 | CSV | Gain-based feature importance (all features) |
| `shap_values_global.csv` | 03 | CSV | 10K × n_features SHAP value matrix (regressor) |
| `shap_bar.png` | 03 | PNG (150 dpi) | Top 30 features by mean |SHAP| (bar chart) |
| `shap_summary.png` | 03 | PNG (150 dpi) | SHAP beeswarm — top 30 features (regressor) |
| `shap_summary_classifier.png` | 03/05 | PNG (150 dpi) | SHAP beeswarm — top 25 features (classifier) |
| `roc_curve.png` | 03 | PNG (150 dpi) | ROC curve with AUC annotation |
| `pr_curve.png` | 03 | PNG (150 dpi) | Precision-Recall curve with AP annotation |
| `equity_gap_county.csv` | 04 | CSV | Equity Gap Score + rank per county |
| `equity_gap_bar.png` | 04 | PNG (150 dpi) | Horizontal bar chart sorted by equity gap |
| `xai_summary.csv` | 05 | CSV | Cross-measure SHAP summary table |
| `county_shap_avg.csv` | 05 | CSV | Mean |SHAP| per county (classifier, 26 × n_feat) |
| `scenario_reports/*.png` | 05 | PNG (150 dpi) | Per-measure per-example SHAP before/after |
| `scenario_reports/*_delta_shap.csv` | 05 | CSV | Feature-level delta SHAP attributions |
| `*.log` | all | Text | Timestamped execution logs (append mode) |

---

## 9. Key Design Decisions and Rationale

### 9.1 Why Positional Merge (not key-based join)?

The BER parquet base does not contain a unique dwelling identifier that is also present in the full CSV. The MPRN (meter point reference number) is present in the CSV but was stripped from the public parquet release. The positional merge is safe only because both sources derive from the same underlying SEAI export and are filtered with identical row conditions. A BER value + Year of Construction pair was considered as a composite key but has too many duplicates (~15% of records) to use as a reliable join key.

### 9.2 Why OrdinalEncoder over OneHotEncoder?

LightGBM's split-finding algorithm natively handles ordinal-encoded categoricals by finding optimal binary splits across the integer-encoded levels. OneHotEncoding would create highly sparse columns and prevent LightGBM from grouping related categories. With `handle_unknown='use_encoded_value, unknown_value=-1'`, unseen categories at inference time (e.g., a new fuel type added to the SEAI database) map to -1, which LightGBM treats as a distinct split value rather than crashing.

### 9.3 Why log1p Transform on BerRating?

The BER distribution is right-skewed with a long tail toward high energy use (E–G rated homes). Without transformation, RMSE is dominated by errors on the high-BER tail, and the model is incentivised to sacrifice accuracy on A–C rated homes (the majority) to reduce large errors on a small number of G-rated homes. `log1p` compresses the tail, equalises the loss across the BER range, and produces better-calibrated predictions on the common cases.

### 9.4 Why Feature Engineering Before Leaky Drop?

`HSMainSystemEfficiency` and `Year_of_Construction` are in `DROP_LEAKY` because they are directly used in DEAP's energy calculation (leaky intermediates). However, their derived forms — `IsHeatPump` (boolean threshold) and `AgeBand` (coarse vintage brackets) — preserve the policy-relevant signal without exposing the raw DEAP input value. The early engineering (STEP 2 before STEP 3) ensures these derived columns exist before the source columns are removed.

### 9.5 Why Keep CountyName in the Parquet?

CountyName is a geographic identifier that should not be a model feature (the model should generalise to new homes without memorising county-level BER patterns). However, it is essential for the county equity analysis. The solution — keep it in the parquet, exclude it from `X` in training — cleanly separates the dataset's role as both a modelling input and an analysis resource.

### 9.6 Why the + 0.5 Offset in the Equity Gap Formula?

Without the offset, the county with the lowest CO₂ intensity would always score zero on that dimension, potentially masking genuine fuel poverty and retrofit gap issues. The offset `co2_norm + 0.5` means CO₂ intensity modulates the score between 0.5× and 1.5× of the `fp_rate × retrofit_gap` product, rather than zeroing it out entirely. This reflects the policy judgement that CO₂ intensity is an amplifier, not a gate.

### 9.7 Why Snappy Compression for Parquet?

The clean parquet (~400MB uncompressed) is loaded by three downstream scripts. Snappy's fast decompression (~2–3× faster than gzip) matters more than compression ratio here since disk space is not the constraint. Brotli offers better compression than Snappy but much slower decompression, which is counterproductive for a repeatedly-read intermediate file.

### 9.8 Scale of the Dataset

With ~1.35 million records, the dataset represents approximately **50% of the entire Irish housing stock** (estimated 2.0 million dwellings, 2022 Census). BER certificates are not a random sample — they are biased toward:
- Homes involved in property transactions (purchase/rental)
- Homes that have applied for SEAI grants
- Newer homes (more likely to have certificates)

This selection bias should be acknowledged when generalising findings to the full national stock, particularly for older, rural, owner-occupied homes without recent transactions.
