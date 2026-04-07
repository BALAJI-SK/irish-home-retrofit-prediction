# RetroFit — BER Prediction Project: Complete Overview

## What This Project Does

This project builds a machine learning model to predict the **Building Energy Rating (BER)** of Irish residential dwellings — a number in kWh/m²/yr that describes how energy-efficient a building is. A lower number means a more efficient building.

Once the model is trained, it is used as a **retrofit simulator**: take any dwelling, change a feature (e.g. replace the heating system with a heat pump), re-predict the BER, and quantify exactly how much the energy rating improves. This lets you answer questions like:

> "If I insulate the walls of a 1970s semi-detached house in Cork, what BER improvement can I expect?"

---

## Project Files

```
RetroFit/
├── scripts/
│   ├── 01_clean_and_prepare.py     # Data cleaning pipeline → parquet
│   ├── 02_train_model.py           # Model training (LightGBM + XGBoost)
│   └── 03_shap_and_retrofit.py     # SHAP explainability + retrofit simulation
│
├── outputs/
│   ├── clean_data.parquet          # 1,350,432 row clean dataset
│   ├── lgbm_model.pkl              # Trained LightGBM model (primary)
│   ├── xgb_model.pkl               # Trained XGBoost model (comparison)
│   ├── feature_importance.csv      # Gain-based feature rankings
│   ├── shap_values.csv             # SHAP values for 5,000 rows
│   ├── shap_bar.png                # Top-30 feature importance bar chart
│   ├── shap_summary.png            # SHAP beeswarm plot
│   ├── retrofit_results.csv        # Per-dwelling retrofit simulation
│   ├── retrofit_bar.png            # Retrofit intervention comparison chart
│   ├── cleaning_report.txt         # Automated cleaning summary
│   └── model_report.txt            # Automated model metrics report
│
├── reports/                        # ← You are here
│   ├── 00_overview.md              # This file
│   ├── 01_dataset.md               # Raw dataset description
│   ├── 02_cleaning.md              # Cleaning decisions and rationale
│   ├── 03_features.md              # Feature engineering
│   ├── 04_model.md                 # Model training and hyperparameters
│   └── 05_results.md               # Results and retrofit findings
│
└── full_report.py                  # Original full-dataset analysis script
```

---

## Quick Results Summary

| Metric | Value |
|--------|-------|
| Dataset (raw) | 1,354,360 rows × 211 columns |
| Dataset (clean) | 1,350,432 rows × 119 columns |
| Model | LightGBM Regressor |
| Test R² | **0.9913** |
| Test RMSE | **14.11 kWh/m²/yr** |
| Test MAE | **7.26 kWh/m²/yr** |
| Best single retrofit | Heat pump installation (−66 kWh/m²/yr, −27%) |
| Best combined retrofit | Deep retrofit package (−117 kWh/m²/yr, −45%) |

---

## Report Index

| Report | Contents |
|--------|----------|
| [01_dataset.md](01_dataset.md) | Raw data description, column types, target variable |
| [02_cleaning.md](02_cleaning.md) | Why 92 columns were dropped, outlier strategy, missing data |
| [03_features.md](03_features.md) | Engineered features and their rationale |
| [04_model.md](04_model.md) | Algorithm choice, hyperparameters, training strategy |
| [05_results.md](05_results.md) | Model accuracy, SHAP importance, retrofit simulation |
