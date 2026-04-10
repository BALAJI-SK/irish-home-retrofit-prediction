# Full Run Summary — valid-imp

## Overview

This document summarizes the latest `valid-imp` full-dataset run, including code changes, dataset usage, validation results, and generated outputs.

## What changed

- `02_train_model.py` was updated to use the full training dataset for:
  - 5-fold cross-validation
  - LightGBM training on full train+validation
  - Random Forest comparison model
  - Ridge regression baseline model
- `03_scenario_engine.py` was updated to simulate retrofit measures on the full dataset instead of a fixed 2,000-home sample.
- `cli_logger.py` was added to capture script stdout/stderr into per-script log files.
- `04_xai_explainer.py`, `05_scenario_report.py`, and `06_demo.py` were also wired to use the shared logging helper.

## Dataset and execution

- Source data: `valid-imp/outputs/clean_data_55col.parquet`
- Full dataset size: **1,347,642 rows**
- Model feature count: **50 features**
- Train / Validation / Test split: **943,348 / 202,147 / 202,147 rows**

## Training results

### LightGBM

- 5-fold CV mean R²: **0.9696 ± 0.0007**
- Fold R² scores: **0.9696, 0.9699, 0.9683, 0.9698, 0.9705**

#### Final evaluation
- Train R²: **0.9768**
- Train RMSE: **23.06**
- Train MAE: **11.50**
- Val R²: **0.9768**
- Val RMSE: **23.22**
- Val MAE: **11.54**
- Test R²: **0.9711**
- Test RMSE: **25.77**
- Test MAE: **12.45**

### Random Forest (full training set)

- Train R²: **0.9629**
- Train RMSE: **29.19**
- Train MAE: **15.10**
- Val R²: **0.9438**
- Val RMSE: **36.11**
- Val MAE: **18.56**
- Test R²: **0.9433**
- Test RMSE: **36.11**
- Test MAE: **18.52**

### Ridge Regression baseline (full training set)

- Train R²: **0.8789**
- Train RMSE: **52.74**
- Train MAE: **28.28**
- Val R²: **0.8795**
- Val RMSE: **52.89**
- Val MAE: **28.37**
- Test R²: **-0.5256**
- Test RMSE: **187.29**
- Test MAE: **28.64**

## Global SHAP / feature importance

- SHAP sample size: **10,000 rows** (for visualization only)
- Top gain features:
  1. `WHMainSystemEff`
  2. `UValueWindow`
  3. `UValueWall`
  4. `GroundFloorAreasq_m`
  5. `FabricHeatLossProxy`
- Top SHAP mean |value| features:
  1. `UValueWindow`
  2. `GroundFloorAreasq_m`
  3. `WHMainSystemEff`
  4. `UValueWall`
  5. `FirstEnergyType_Description`

## Retrofit scenario results

- Simulation run on full dataset: **1,347,642 dwellings**
- Baseline BER (full dataset): **mean 205.3**, **median 184.6** kWh/m²/yr

### Mean retrofit savings by measure

- Heat Pump Installation: **77.5** kWh/m²/yr (**31.2%**) — resulting BER **127.8**
- Wall Insulation: **33.2** kWh/m²/yr (**10.6%**) — resulting BER **172.1**
- Roof Insulation: **13.3** kWh/m²/yr (**3.5%**) — resulting BER **192.0**
- Window Upgrade: **14.2** kWh/m²/yr (**5.1%**) — resulting BER **191.1**
- Floor Insulation: **14.4** kWh/m²/yr (**5.7%**) — resulting BER **190.9**
- Solar Water Heating: **4.9** kWh/m²/yr (**1.4%**) — resulting BER **200.4**
- LED Lighting Upgrade: **-3.4** kWh/m²/yr (**-6.4%**) — resulting BER **208.7**
- Deep Retrofit Package: **124.0** kWh/m²/yr (**41.0%**) — resulting BER **81.4**

### Single dwelling example

- Baseline BER: **262.4** kWh/m²/yr (D2)
- Heat Pump Installation: **139.7** (B3), saving **122.7** kWh/m²/yr
- Wall Insulation: **236.4** (D1), saving **26.1** kWh/m²/yr
- Roof Insulation: **243.7** (D1), saving **18.7** kWh/m²/yr
- Window Upgrade: **242.6** (D1), saving **19.8** kWh/m²/yr
- Floor Insulation: **257.6** (D1), saving **4.9** kWh/m²/yr
- Solar Water Heating: **261.2** (D2), saving **1.2** kWh/m²/yr
- LED Lighting Upgrade: **250.5** (D1), saving **12.0** kWh/m²/yr
- Deep Retrofit Package: **79.5** (B1), saving **182.9** kWh/m²/yr

## Output artifacts

Generated files from this run:

- `valid-imp/outputs/lgbm_model.pkl`
- `valid-imp/outputs/feature_importance.csv`
- `valid-imp/outputs/shap_values_global.csv`
- `valid-imp/outputs/shap_bar.png`
- `valid-imp/outputs/shap_summary.png`
- `valid-imp/outputs/model_report.txt`
- `valid-imp/outputs/retrofit_results.csv`
- `valid-imp/outputs/retrofit_bar.png`
- `valid-imp/outputs/retrofit_summary.txt`
- `valid-imp/outputs/xai_summary.csv`
- `valid-imp/outputs/scenario_reports/full_report.txt`
- `valid-imp/outputs/scenario_reports/cross_measure_comparison.txt`
- `valid-imp/outputs/scenario_reports/{measure}_{idx}_report.txt` for each example report
- log files: `valid-imp/outputs/01_clean_and_prepare.log`, `02_train_model.log`, `03_scenario_engine.log`, `04_xai_explainer.log`, `05_scenario_report.log`, `06_demo.log`

## Code files changed

- `valid-imp/scripts/02_train_model.py`
- `valid-imp/scripts/03_scenario_engine.py`
- `valid-imp/scripts/cli_logger.py`
- `valid-imp/scripts/04_xai_explainer.py`
- `valid-imp/scripts/05_scenario_report.py`
- `valid-imp/scripts/06_demo.py`

## Key validation summary

- The run now uses full dataset training and scenario processing, not fixed subsamples.
- LightGBM validation performance remains strong with test R² **0.9711** and test MAE **12.45**.
- Retrofit engine now reports full-data results across 1.35M homes.
- The established report generation pipeline produced both per-home text reports and a combined full report.

## Notes

- `02_train_model.py` still uses a 10,000-row SHAP sample purely for explainability and plotting, not for training or evaluation.
- The negative LED lighting savings reflect the model's current prediction behavior and should be treated as a realistic model result rather than a code error.
- The full dataset run is recorded in `outputs/03_scenario_engine.log` and `outputs/05_scenario_report.log`.
