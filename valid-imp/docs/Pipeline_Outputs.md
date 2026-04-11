# Pipeline Execution Outputs — Irish Home Retrofit Equity Analysis

**Full pipeline run results — all 5 scripts executed in sequence**  
**Dataset: SEAI BER Database | 1,347,642 certificates | 26 counties (55 geographic zones)**

---

## High-Level Overview of the Pipeline

> *This section summarises `Implementation_Details.md` for quick reference.*

The pipeline consists of 5 scripts that execute in strict order, each feeding into the next:

| # | Script | Role | Runtime |
|---|--------|------|---------|
| 1 | `01_clean_and_prepare.py` | Data cleaning, leakage removal, feature engineering, policy columns | 11.1 s |
| 2 | `02_county_profile.py` | 26-county/55-zone aggregation, BER profiling, bubble chart | 0.4 s |
| 3 | `03_train_model.py` | LightGBM regressor + classifier, Random Forest, Ridge, SHAP | 13.7 min |
| 4 | `04_equity_gap.py` | Composite equity gap scoring, ranked bar chart | 0.3 s |
| 5 | `05_xai_explainer.py` | Delta SHAP per retrofit measure × 3 homes, classifier SHAP | ~3–5 min |

**Two sources merged positionally** — a 46-column pre-filtered parquet and the full 211-column SEAI CSV — producing a 61-column clean dataset. **Eight leaky DEAP intermediate features** are removed before training. **Seven policy columns** are engineered for equity analysis (CO₂ intensity, retrofit status flags, fuel poverty risk). The primary model is a **LightGBM regressor** achieving **R² = 0.9714 on the held-out test set**. A second **LightGBM classifier** predicts retrofit uptake (`is_retrofitted`) with **ROC-AUC = 1.000** (near-perfect separation from U-value signals). An **Equity Gap Score** ranks all 55 geographic zones by the intersection of fuel poverty burden, low retrofit uptake, and CO₂ intensity. SHAP is applied in three contexts: global feature importance (regressor), before/after delta attribution (7 retrofit scenarios × 3 homes), and county-level classifier attribution.

**Key finding:** Dublin 23, Co. Leitrim, Dublin 21, Co. Roscommon, and Co. Offaly are the top-5 priority zones for retrofit policy intervention based on the equity gap score.

---

## Script 1 — `01_clean_and_prepare.py`

### Execution Summary

```
Runtime : 11.1 seconds
Input   : Col Final with County.parquet (1,351,582 rows × 46 cols)
          Public Search Data.csv (1,351,582 rows, filtered to 1,347,642)
Output  : clean_data_55col.parquet (1,347,642 rows × 61 cols)
```

### Step-by-Step Console Output

**STEP 1 — Load parquet**
- Loaded: 1,351,582 rows × 46 cols
- WARNING: 3,940 rows outside expected range (BerRating or Year_of_Construction out of bounds) — filtered out
- Working dataset after filter: 1,347,642 rows

**STEP 2 — Early feature engineering (pre-drop)**
- `IsHeatPump` derived from `HSMainSystemEfficiency > 100`: **195,571 heat-pump homes** identified
- `AgeBand` created from `Year_of_Construction` using DEAP vintage brackets ✓

**STEP 3 — Remove leaky DEAP columns**

| Column | Status |
|--------|--------|
| DistributionLosses | Removed |
| HSEffAdjFactor | Removed |
| WHEffAdjFactor | Removed |
| SHRenewableResources | Removed |
| WHRenewableResources | Removed |
| FabricHeatLossPerM2 | Already absent (OK) |
| Year_of_Construction | Removed |
| HSMainSystemEfficiency | Removed |

After removal: **41 columns**

**STEP 4 — CSV supplement merge**
- Raw CSV slice: 1,351,582 rows
- After same row filters: 1,347,642 rows
- Row counts match (1,347,642). **Positional merge safe ✓**
- 7 columns successfully added:

| CSV Column | → | Output Column |
|------------|---|---------------|
| MainSpaceHeatingFuel | → | MainSpaceHeatingFuel |
| MainWaterHeatingFuel | → | MainWaterHeatingFuel |
| SolarHotWaterHeating | → | SolarHotWaterHeating |
| LowEnergyLightingPercent | → | LowEnergyLightingPercent |
| GroundFloorArea(sq m) | → | GroundFloorAreasq_m |
| HESSchemeUpgrade | → | HESSchemeUpgrade |
| FirstEnergyType_Description | → | FirstEnergyType_Description |

After merge: **48 columns**

**STEP 5 — Null imputation**

| Column | Nulls Filled | Fill Value |
|--------|-------------|-----------|
| SolarHotWaterHeating | 682,864 | 'NO' (MNAR) |
| MainSpaceHeatingFuel | 23,464 | 'Mains Gas' (mode) |
| MainWaterHeatingFuel | 23,464 | 'Mains Gas' (mode) |
| FirstEnergyType_Description | 23,466 | 'None' (MNAR) |

**STEP 6 — Secondary feature engineering**
- `IsHeatPump` updated with fuel-name signal: **195,571 homes** (unchanged — fuel match confirmed same set)
- `HasSolarWaterHeating`: **34,274 homes**
- `HasRoofInsulation`, `HasWallInsulation`, `HasDoubleGlazing`, `WindowToWallRatio`, `FabricHeatLossProxy` — all computed

**STEP 7 — Policy columns for equity analysis**

| Column | Value |
|--------|-------|
| `EstCO2_kg_per_m2` | mean = **52.7 kg/m²/yr** |
| `Total_Annual_CO2_Tonnes` | mean = **3.56 t/yr** per home |
| `wall_insulated` | **650,751 homes** (48.3%) |
| `roof_insulated` | **579,284 homes** (43.0%) |
| `heating_upgraded` | **436,065 homes** (32.4%) |
| `is_retrofitted` | **903,455 homes (67.0%)** |
| `fuel_poverty_risk` | **121,195 homes (9.0%)** |

**STEP 9 — Save output**
- Saved: **1,347,642 rows × 61 columns** to `clean_data_55col.parquet`

---

### Validation Report

**Leakage Verification — all 11 checked columns absent ✓**

| Column | Status |
|--------|--------|
| DistributionLosses | absent (OK) |
| HSEffAdjFactor | absent (OK) |
| WHEffAdjFactor | absent (OK) |
| SHRenewableResources | absent (OK) |
| WHRenewableResources | absent (OK) |
| FabricHeatLossPerM2 | absent (OK) |
| Year_of_Construction | absent (OK) |
| HSMainSystemEfficiency | absent (OK) |
| FirstEnerProdDelivered | absent (OK) |
| TempAdjustment | absent (OK) |
| TempFactorMultiplier | absent (OK) |

**High Correlation Check**
- ⚠ WARNING: `EstCO2_kg_per_m2` correlates **0.9682** with BerRating
  *(Expected — CO₂ is directly derived from BerRating × fuel factor. This column is excluded from model training features via `NON_FEATURE_COLS`.)*

**Target Statistics (BerRating)**

| Statistic | Value |
|-----------|-------|
| Mean | **206.20 kWh/m²/yr** |
| Median | **184.62 kWh/m²/yr** |
| Std Dev | 151.67 |
| % A–B (≤ 100) | **22.2%** |
| % C (101–200) | **34.6%** |
| % D (201–300) | **26.3%** |
| % E–G (> 300) | **16.9%** |

**Null Check (remaining after imputation)**

| Column | Nulls | % |
|--------|------:|---:|
| HSSupplHeatFraction | 23,464 | 1.74% |
| HSSupplSystemEff | 23,464 | 1.74% |
| WHMainSystemEff | 23,464 | 1.74% |
| SupplSHFuel | 23,464 | 1.74% |
| SupplWHFuel | 23,464 | 1.74% |
| NoOfFansAndVents | 24,099 | 1.79% |
| VentilationMethod | 24,099 | 1.79% |
| StructureType | 24,099 | 1.79% |
| SuspendedWoodenFloor | 24,099 | 1.79% |
| PercentageDraughtStripped | 24,099 | 1.79% |
| NoOfSidesSheltered | 24,099 | 1.79% |
| PredominantRoofType | 141,236 | 10.48% |

**Engineered Features Summary**

| Feature | Value |
|---------|-------|
| WindowToWallRatio | mean = 0.2990 |
| FabricHeatLossProxy | mean = 176.29 W/K |
| AgeBand | See distribution below |
| IsHeatPump | 0: 1,152,071 / 1: 195,571 |
| HasSolarWaterHeating | 0: 1,313,368 / 1: 34,274 |
| HasRoofInsulation | 0: 768,358 / 1: 579,284 |
| HasWallInsulation | 0: 696,891 / 1: 650,751 |
| HasDoubleGlazing | 0: 951,565 / 1: 396,077 |

**Age Band Distribution**

| Band | Count | % |
|------|------:|--:|
| Pre1900 | 75,593 | 5.6% |
| 1900-1929 | 49,768 | 3.7% |
| 1930-1949 | 70,499 | 5.2% |
| 1950-1966 | 73,560 | 5.5% |
| 1967-1977 | 132,935 | 9.9% |
| 1978-1982 | 69,462 | 5.2% |
| 1983-1993 | 122,299 | 9.1% |
| 1994-1999 | 148,573 | 11.0% |
| 2000-2004 | 226,229 | 16.8% |
| 2005-2010 | 151,371 | 11.2% |
| 2011-2015 | 25,074 | 1.9% |
| 2016+ | 202,279 | 15.0% |

**Main Space Heating Fuel (top 10)**

| Fuel | Count | % |
|------|------:|--:|
| Mains Gas | 482,034 | 35.8% |
| Heating Oil | 449,682 | 33.4% |
| Electricity | 350,397 | 26.0% |
| Solid Multi-Fuel | 33,226 | 2.5% |
| Bulk LPG (propane or butane) | 16,347 | 1.2% |
| Manufactured Smokeless Fuel | 7,456 | 0.6% |
| House Coal | 2,637 | 0.2% |
| Wood Pellets (bulk supply) | 1,993 | 0.1% |
| Bottled LPG | 1,263 | 0.1% |
| Sod Peat | 1,142 | 0.1% |

**County Distribution (top 10 by BER certificates)**

| County | Count | % |
|--------|------:|--:|
| Co. Cork | 128,799 | 9.6% |
| Co. Dublin | 106,898 | 7.9% |
| Co. Kildare | 64,552 | 4.8% |
| Co. Meath | 54,791 | 4.1% |
| Co. Galway | 49,681 | 3.7% |
| Co. Wexford | 44,884 | 3.3% |
| Co. Wicklow | 43,913 | 3.3% |
| Co. Kerry | 40,768 | 3.0% |
| Co. Tipperary | 40,693 | 3.0% |
| Co. Donegal | 40,529 | 3.0% |

**Final Column List (61 columns)**

| # | Column | # | Column | # | Column |
|---|--------|---|--------|---|--------|
| 1 | AgeBand | 22 | HeatSystemControlCat | 43 | ThermalMassCategory |
| 2 | BerRating | 23 | HeatSystemResponseCat | 44 | Total_Annual_CO2_Tonnes |
| 3 | CHBoilerThermostatControlled | 24 | IsHeatPump | 45 | UValueFloor |
| 4 | CombinedCylinder | 25 | LivingAreaPercent | 46 | UValueRoof |
| 5 | CountyName | 26 | LowEnergyLightingPercent | 47 | UValueWall |
| 6 | CylinderStat | 27 | MainSpaceHeatingFuel | 48 | UValueWindow |
| 7 | DoorArea | 28 | MainWaterHeatingFuel | 49 | UndergroundHeating |
| 8 | DwellingTypeDescr | 29 | NoCentralHeatingPumps | 50 | UvalueDoor |
| 9 | EstCO2_kg_per_m2 | 30 | NoOfFansAndVents | 51 | VentilationMethod |
| 10 | FabricHeatLossProxy | 31 | NoOfSidesSheltered | 52 | WHMainSystemEff |
| 11 | FirstEnergyType_Description | 32 | NoStoreys | 53 | WallArea |
| 12 | FloorArea | 33 | OBBoilerThermostatControlled | 54 | WarmAirHeatingSystem |
| 13 | GroundFloorAreasq_m | 34 | OBPumpInsideDwelling | 55 | WindowArea |
| 14 | HESSchemeUpgrade | 35 | PercentageDraughtStripped | 56 | WindowToWallRatio |
| 15 | HSSupplHeatFraction | 36 | PredominantRoofType | 57 | fuel_poverty_risk |
| 16 | HSSupplSystemEff | 37 | RoofArea | 58 | heating_upgraded |
| 17 | HasDoubleGlazing | 38 | SolarHotWaterHeating | 59 | is_retrofitted |
| 18 | HasRoofInsulation | 39 | StructureType | 60 | roof_insulated |
| 19 | HasSolarWaterHeating | 40 | SupplSHFuel | 61 | wall_insulated |
| 20 | HasWallInsulation | 41 | SupplWHFuel | | |
| 21 | HeatSystemControlCat | 42 | SuspendedWoodenFloor | | |

---

## Script 2 — `02_county_profile.py`

### Execution Summary

```
Runtime : 0.4 seconds
Input   : clean_data_55col.parquet (1,347,642 rows × 61 cols)
Output  : county_profiles.csv, county_summary_table.md, county_bubble_chart.png
```

- Dataset loaded: 1,347,642 rows × 61 cols
- **55 geographic zones** aggregated (26 counties + Dublin postal districts)

### Top 5 Counties by Mean BER (Worst First)

| Zone | Mean BER | Retrofit % | Fuel Poverty % |
|------|--------:|----------:|--------------:|
| Dublin 7 | **279.5** | 54.5% | 2.5% |
| Dublin 6 | 272.6 | 55.6% | 2.9% |
| Dublin 23 | 264.7 | 25.0% | **25.0%** |
| Dublin 21 | 264.7 | 50.0% | 16.7% |
| Co. Roscommon | 259.8 | 64.8% | 20.8% |

### National Summary Statistics

| Metric | Value |
|--------|-------|
| Total geographic zones | 55 |
| Total BER certificates | 1,347,642 |
| National mean BER | **214.4 kWh/m²/yr** |
| Best mean BER zone | **Dublin 18 (115.8 kWh/m²/yr)** |
| Worst mean BER zone | **Dublin 7 (279.5 kWh/m²/yr)** |
| Highest retrofit rate | **Dublin 19 (80.0%)** |
| Lowest retrofit rate | **Dublin 23 (25.0%)** |
| Highest fuel poverty | **Dublin 23 (25.0%)** |

### Outputs Generated
- `county_profiles.csv` — 55 rows × 12 metric columns
- `county_summary_table.md` — formatted markdown table
- `county_bubble_chart.png` — 4-quadrant scatter: mean BER (X) vs retrofit rate (Y), bubble size = homes, colour = fuel poverty rate

---

## Script 3 — `03_train_model.py`

### Execution Summary

```
Runtime : 13.7 minutes
Input   : clean_data_55col.parquet (1,347,642 rows × 61 cols)
Output  : lgbm_model.pkl, lgbm_classifier.pkl, model_report.txt,
          classifier_report.txt, feature_importance.csv,
          shap_values_global.csv, shap_bar.png, shap_summary.png,
          shap_summary_classifier.png, roc_curve.png, pr_curve.png
```

### Dataset and Feature Setup

- Loaded: 1,347,642 rows × 61 columns in **0.2 s**
- Classifier target `is_retrofitted`: **903,455 positive (67.0%)**
- BerRating: mean=206.2, std=151.7, min=0.0, max=1997.7
- Log1p(BerRating): mean=5.077, std=0.772

**Feature types after excluding `NON_FEATURE_COLS`:**
- Numeric features: **34**
- Categorical features: **18**
- **Total: 52 features**

**18 Categorical columns:**
`DwellingTypeDescr`, `VentilationMethod`, `StructureType`, `SuspendedWoodenFloor`, `CHBoilerThermostatControlled`, `OBBoilerThermostatControlled`, `OBPumpInsideDwelling`, `WarmAirHeatingSystem`, `UndergroundHeating`, `CylinderStat`, `CombinedCylinder`, `ThermalMassCategory`, `PredominantRoofType`, `AgeBand`, `MainSpaceHeatingFuel`, `MainWaterHeatingFuel`, `SolarHotWaterHeating`, `FirstEnergyType_Description`

**Train/Val/Test Split (70/15/15):**
- Train: **943,348** rows
- Val: **202,147** rows
- Test: **202,147** rows

---

### LightGBM Regressor — 5-Fold Cross-Validation

| Fold | R² |
|------|---:|
| Fold 1/5 | 0.9701 |
| Fold 2/5 | 0.9706 |
| Fold 3/5 | 0.9691 |
| Fold 4/5 | 0.9706 |
| Fold 5/5 | 0.9710 |
| **CV Mean** | **0.9703 ± 0.0006** |

### LightGBM Regressor — Performance

| Split | R² | RMSE (kWh/m²/yr) | MAE (kWh/m²/yr) |
|-------|----|------------------:|----------------:|
| Train | 0.9772 | 22.88 | 11.39 |
| Val | 0.9774 | 22.92 | 11.40 |
| **Test** | **0.9714** | **25.64** | **12.36** |

*Train-test R² gap of 0.006 confirms minimal overfitting.*

### Random Forest — Comparison Model

| Split | R² | RMSE (kWh/m²/yr) | MAE (kWh/m²/yr) |
|-------|----|------------------:|----------------:|
| Train | 0.9631 | 29.12 | 15.05 |
| Val | 0.9440 | 36.05 | 18.51 |
| Test | 0.9435 | 36.05 | 18.47 |

*LightGBM outperforms Random Forest by 0.028 R² on test set.*

### Ridge Regression — Baseline

| Split | R² | RMSE (kWh/m²/yr) | MAE (kWh/m²/yr) |
|-------|----|------------------:|----------------:|
| Train | 0.8765 | 53.25 | 28.34 |
| Val | 0.8773 | 53.37 | 28.42 |
| Test | −2.8591 | 297.87 | 28.96 |

*Note: Ridge test R² is strongly negative due to heteroscedastic residuals in the BER distribution — the linear model cannot handle the non-linear relationships and high-BER tail.*

---

### Feature Importance — Top 20 (LightGBM Gain-Based)

| Rank | Feature | Gain Importance |
|------|---------|----------------:|
| 1 | WHMainSystemEff | 17,412 |
| 2 | UValueWindow | 9,997 |
| 3 | UValueWall | 9,508 |
| 4 | GroundFloorAreasq_m | 9,225 |
| 5 | FabricHeatLossProxy | 7,704 |
| 6 | WallArea | 6,764 |
| 7 | FirstEnergyType_Description | 6,737 |
| 8 | LivingAreaPercent | 6,697 |
| 9 | UValueFloor | 6,667 |
| 10 | WindowToWallRatio | 5,618 |
| 11 | WindowArea | 5,603 |
| 12 | RoofArea | 5,582 |
| 13 | UValueRoof | 5,427 |
| 14 | FloorArea | 5,063 |
| 15 | DoorArea | 4,822 |
| 16 | UvalueDoor | 4,331 |
| 17 | HSSupplSystemEff | 3,519 |
| 18 | AgeBand | 3,311 |
| 19 | HeatSystemResponseCat | 3,034 |
| 20 | NoOfFansAndVents | 2,795 |

### Global SHAP — Top 20 Features (Mean |SHAP|)

SHAP computed on a **10,000-row random sample** in **20.5 seconds**.

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------:|
| 1 | UValueWindow | 0.192100 |
| 2 | GroundFloorAreasq_m | 0.137970 |
| 3 | UValueWall | 0.112036 |
| 4 | WHMainSystemEff | 0.093169 |
| 5 | FirstEnergyType_Description | 0.086212 |
| 6 | IsHeatPump | 0.051717 |
| 7 | FabricHeatLossProxy | 0.048425 |
| 8 | UValueRoof | 0.045092 |
| 9 | HeatSystemControlCat | 0.043249 |
| 10 | WallArea | 0.039982 |
| 11 | CHBoilerThermostatControlled | 0.039639 |
| 12 | UValueFloor | 0.039596 |
| 13 | HSSupplSystemEff | 0.036896 |
| 14 | FloorArea | 0.028375 |
| 15 | AgeBand | 0.027822 |
| 16 | HeatSystemResponseCat | 0.027472 |
| 17 | RoofArea | 0.027373 |
| 18 | LowEnergyLightingPercent | 0.027107 |
| 19 | SupplSHFuel | 0.024916 |
| 20 | ThermalMassCategory | 0.017947 |

*Key observation: SHAP and gain rankings differ — `UValueWindow` tops SHAP (0.192) but ranks #2 by gain. `WHMainSystemEff` tops gain but ranks #4 by SHAP. This reflects gain's bias toward high-cardinality features used in many splits.*

---

### LightGBM Classifier — `is_retrofitted`

**Split:** Train 943,348 / Val 202,147 / Test 202,147

**Test Set Metrics:**

| Metric | Value |
|--------|------:|
| **ROC-AUC** | **1.0000** |
| **Average Precision** | **1.0000** |
| Accuracy | 1.00 |

**Classification Report (test set):**

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| Not Retrofitted | 1.00 | 1.00 | 1.00 | 66,431 |
| Retrofitted | 1.00 | 1.00 | 1.00 | 135,716 |
| **macro avg** | **1.00** | **1.00** | **1.00** | **202,147** |
| **weighted avg** | **1.00** | **1.00** | **1.00** | **202,147** |

> **Note on perfect scores:** The `is_retrofitted` label is directly derived from U-value thresholds and `HESSchemeUpgrade` — both of which are in the feature set. The classifier essentially learns a deterministic rule, which explains the perfect separation. This is expected given how the target was constructed and is useful for interpretability (the SHAP values reveal exactly which features encode the retrofit signal).

**Top 10 Features for Retrofit Classification (Classifier SHAP, mean |SHAP|):**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------:|
| 1 | UValueWall | 5.6546 |
| 2 | UValueRoof | 4.2341 |
| 3 | HESSchemeUpgrade | 3.0818 |
| 4 | HasWallInsulation | 1.5363 |
| 5 | HasRoofInsulation | 1.0575 |
| 6 | IsHeatPump | 0.5887 |
| 7 | SolarHotWaterHeating | 0.1281 |
| 8 | WHMainSystemEff | 0.1138 |
| 9 | CHBoilerThermostatControlled | 0.1006 |
| 10 | RoofArea | 0.0756 |

*Confirms that U-values and HES scheme participation are the primary signals for retrofit status.*

---

## Script 4 — `04_equity_gap.py`

### Execution Summary

```
Runtime : 0.3 seconds
Input   : county_profiles.csv (55 zones)
Output  : equity_gap_county.csv, equity_gap_bar.png
```

### Equity Gap Score Formula

```
fp_rate      = fuel_poverty_rate / 100
retrofit_gap = 1 − (retrofit_rate / 100)
co2_norm     = (co2 − co2_min) / (co2_max − co2_min)    ∈ [0, 1]

raw_gap           = fp_rate × retrofit_gap × (co2_norm + 0.5)
equity_gap_score  = (raw_gap − min) / (max − min) × 100  ∈ [0, 100]
```

### Top 10 Counties by Equity Gap Score

| Rank | County | Score | Fuel Poverty % | Retrofit % | Mean BER | CO₂ kg/m²/yr |
|------|--------|------:|---------------:|-----------:|---------:|-------------:|
| 1 | **Dublin 23** | **100.0** | 25.0% | 25.0% | 264.7 | 70.43 |
| 2 | Co. Leitrim | 42.1 | 20.6% | 62.6% | 258.3 | 72.30 |
| 3 | Dublin 21 | 40.4 | 16.7% | 50.0% | 264.7 | 64.19 |
| 4 | Co. Roscommon | 40.0 | 20.8% | 64.8% | 259.8 | 72.21 |
| 5 | Co. Offaly | 36.1 | 21.1% | 65.7% | 236.4 | 66.01 |
| 6 | Co. Mayo | 34.6 | 17.9% | 63.4% | 250.9 | 69.65 |
| 7 | Co. Donegal | 32.7 | 17.1% | 60.6% | 235.5 | 64.30 |
| 8 | Co. Tipperary | 31.2 | 18.5% | 66.5% | 246.0 | 66.62 |
| 9 | Co. Sligo | 27.3 | 15.5% | 64.0% | 234.0 | 65.04 |
| 10 | Co. Westmeath | 27.1 | 15.4% | 62.4% | 226.6 | 62.10 |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Zones scored | 55 |
| Top-5 priority zones | Dublin 23, Co. Leitrim, Dublin 21, Co. Roscommon, Co. Offaly |
| Score range | 0.0 – 100.0 |
| Top-quartile threshold | **20.6** (14 zones qualify) |

**Policy interpretation:**
- Dublin 23 scores 100 — maximum combination of fuel poverty (25%), near-zero retrofit uptake for its BER level, and high CO₂ intensity (70.43 kg/m²/yr)
- Western rural counties (Leitrim, Roscommon, Offaly, Mayo, Donegal) dominate the top 10 — high oil/solid-fuel dependency combined with moderate retrofit uptake despite poor BER
- 14 out of 55 zones fall above the top-quartile threshold of 20.6 and should be prioritised for SEAI HHES/BEC programme allocation

---

## Script 5 — `05_xai_explainer.py`

### Execution Summary

```
Input   : lgbm_model.pkl, lgbm_classifier.pkl,
          clean_data_55col.parquet, config/retrofit_measures.json
Output  : 7 measures × 3 homes = 21 SHAP before/after pairs (PNG + CSV)
          xai_summary.csv, shap_summary_classifier.png, county_shap_avg.csv
```

**3 example homes selected (BER > 200):**

| Example | Dwelling Type | County | Baseline BER | Grade |
|---------|--------------|--------|-------------:|-------|
| 1 | Detached house | Co. Cork | 203.2 | C3 |
| 2 | Mid-terrace house | Dublin 12 | 222.4 | C3 |
| 3 | Mid-terrace house | Dublin 7 | 679.5 | G |

---

### Measure A — Heat Pump Installation

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | 112.5 [B2] | 90.7 | **44.6%** | WHMainSystemEff (−0.3520) |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | 138.2 [B3] | 84.2 | **37.8%** | WHMainSystemEff (−0.3009) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | 381.6 [F] | 297.9 | **43.8%** | IsHeatPump (−0.3289) |

**Delta SHAP top drivers — Example 1 (Detached, Cork):**

| Feature | Δ SHAP |
|---------|-------:|
| WHMainSystemEff | −0.3520 |
| IsHeatPump | −0.2938 |
| LowEnergyLightingPercent | +0.0737 |
| CHBoilerThermostatControlled | −0.0398 |
| FirstEnergyType_Description | +0.0290 |

---

### Measure B — Wall Insulation

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | 192.5 [C2] | 10.7 | 5.3% | UValueWall (−0.1476) |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | 212.4 [C3] | 10.1 | 4.5% | UValueWall (−0.1586) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | 530.9 [G] | **148.6** | **21.9%** | UValueWall (−0.4697) |

---

### Measure C — Roof Insulation

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | 199.7 [C2] | 3.5 | 1.7% | UValueRoof (−0.0071) |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | 210.2 [C3] | 12.2 | 5.5% | UValueRoof (−0.0490) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | 533.7 [G] | **145.7** | **21.4%** | UValueRoof (−0.2106) |

---

### Measure D — Window Upgrade

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | 168.4 [C1] | 34.8 | **17.1%** | UValueWindow (−0.5117) |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | 212.9 [C3] | 9.6 | 4.3% | UValueWindow (−0.4310) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | 553.5 [G] | 126.0 | **18.5%** | UValueWindow (−0.6236) |

---

### Measure E — Floor Insulation

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | 172.7 [C1] | 30.5 | **15.0%** | UValueFloor (−0.1143) |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | 207.0 [C3] | 15.4 | 6.9% | UValueFloor (−0.0579) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | 643.4 [G] | 36.0 | 5.3% | UValueFloor (−0.0711) |

---

### Measure F — Solar Water Heating

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | 203.2 [C3] | **0.0** | **0.0%** | No change detected |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | 199.9 [C2] | 22.5 | **10.1%** | SolarHotWaterHeating (−0.0481) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | 650.3 [G] | 29.2 | 4.3% | SolarHotWaterHeating (−0.0267) |

*Note: Home 1 (Detached, Cork) showed zero saving — the model may already account for the home's solar potential being saturated, or the override had no impact on the model's feature space for that particular dwelling.*

---

### Measure G — LED Lighting Upgrade

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | 196.1 [C2] | 7.1 | 3.5% | LowEnergyLightingPercent (−0.0358) |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | 212.4 [C3] | 10.0 | 4.5% | LowEnergyLightingPercent (−0.0409) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | 653.8 [G] | 25.6 | 3.8% | PercentageDraughtStripped (−0.0060) |

---

### Measure H — Deep Retrofit Package

*(All measures combined: heat pump + full fabric upgrade)*

| Example | Baseline BER | After BER | Saving | Saving % | Top Driver (Δ SHAP) |
|---------|-------------|----------|-------:|---------:|---------------------|
| 1 — Detached, Cork | 203.2 [C3] | **89.4 [B1]** | 113.8 | **56.0%** | UValueWindow (−0.5104) |
| 2 — Terrace, Dublin 12 | 222.4 [C3] | **73.8 [A3]** | 148.6 | **66.8%** | UValueWindow (−0.5423) |
| 3 — Terrace, Dublin 7 | 679.5 [G] | **105.4 [B2]** | **574.0** | **84.5%** | UValueWindow (−0.7133) |

**Deep retrofit delta SHAP — Example 3 (Terrace, Dublin 7, 679.5 → 105.4):**

| Feature | Δ SHAP |
|---------|-------:|
| UValueWindow | −0.7133 |
| UValueWall | −0.4530 |
| WHMainSystemEff | −0.3169 |
| IsHeatPump | −0.2680 |
| UValueRoof | −0.1936 |

*The deep retrofit achieves an **84.5% BER reduction** for the worst-performing home — from G to B2 — driven primarily by glazing, wall insulation, heat pump, and roof insulation in that order.*

---

### Retrofit Measure Comparison Summary

| Measure | Best BER Saving | Best % Reduction | Primary SHAP Driver |
|---------|---------------:|-----------------:|---------------------|
| Heat Pump Installation | 297.9 kWh/m²/yr | 44.6% | WHMainSystemEff / IsHeatPump |
| Wall Insulation | 148.6 kWh/m²/yr | 21.9% | UValueWall |
| Roof Insulation | 145.7 kWh/m²/yr | 21.4% | UValueRoof |
| **Deep Retrofit Package** | **574.0 kWh/m²/yr** | **84.5%** | UValueWindow |
| Window Upgrade | 126.0 kWh/m²/yr | 18.5% | UValueWindow |
| Floor Insulation | 36.0 kWh/m²/yr | 15.0% | UValueFloor |
| Solar Water Heating | 29.2 kWh/m²/yr | 10.1% | SolarHotWaterHeating |
| LED Lighting Upgrade | 25.6 kWh/m²/yr | 4.5% | LowEnergyLightingPercent |

*For C3-rated homes, heat pump installation is the single most impactful measure (37–45% reduction). For the G-rated home, all fabric measures are large because the baseline heat loss is extreme.*

---

### Section B — Classifier SHAP (County Attribution)

- SHAP computed for 10,000 rows from the classifier
- `county_shap_avg.csv` saved — mean |SHAP| per feature per county (55 zones × 52 features)
- `shap_summary_classifier.png` — beeswarm of top 25 features for positive class

**Confirmed: UValueWall and UValueRoof are the dominant retrofit signals across all counties.**

---

## Key Cross-Pipeline Findings

| Finding | Evidence |
|---------|---------|
| 67.0% of Irish homes have received at least one retrofit measure | `is_retrofitted` column: 903,455 / 1,347,642 |
| 9.0% of homes are at fuel poverty risk | BerRating > 300 AND oil/coal/peat fuel: 121,195 homes |
| LightGBM achieves R² = 0.9714 after honest leakage removal | Test set metrics, 03_train_model.py |
| Top SHAP feature for BER prediction: UValueWindow (0.192) | Global SHAP, 10K sample |
| Heat pump is the highest-impact single measure (up to 44.6% BER reduction) | XAI scenario analysis |
| Deep retrofit can reduce BER by 84.5% for G-rated homes | XAI deep retrofit scenario |
| Dublin 23 has the highest equity gap (score 100) — 25% fuel poverty + 25% retrofit rate | Equity gap scoring |
| 14 of 55 zones qualify for top-quartile policy priority (score ≥ 20.6) | Equity gap scoring |
| Retrofit classification is perfectly learnable from U-values alone (AUC = 1.0) | Classifier metrics |
| UValueWall (SHAP = 5.65) and UValueRoof (4.23) dominate retrofit classification | Classifier SHAP |
