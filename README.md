# Irish Home Retrofit Prediction

Machine learning project to predict Irish home energy ratings (BER) and retrofit likelihood based on physical and efficiency drivers from the SEAI BER database (~1.35 million building energy certificates, ~50% of the national housing stock).

**Language:** Python 3.9 | **Libraries:** pandas, numpy, LightGBM, scikit-learn, SHAP, matplotlib

## Getting Started

### Prerequisites

- **Python 3.9+** (project uses Python 3.9)
- **SEAI Data Files** – Download from SEAI BER database:
  - `Public Search Data.csv` (211 columns, ~1.6M rows) – place in `~/Downloads/`
  - `Col Final with County.parquet` (46 columns, ~1.35M rows — pre-filtered BER base with CountyName) – place in `~/Downloads/`

### Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd irish-home-retrofit-prediction/final-implementation
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or use the pre-configured venv if already initialized.

### Running the Pipeline

Execute the full pipeline from the project root directory:

```bash
cd final-implementation
bash run_pipeline.sh
```

The pipeline runs 5 sequential steps:

| Step | Script | Runtime | Description |
|------|--------|---------|-------------|
| 1 | `01_clean_and_prepare.py` | ~2–5 min | Merges parquet + CSV sources via positional merge, engineers 10+ features (IsHeatPump, AgeBand, FabricHeatLossProxy, CO₂ estimates, policy labels), drops leaky DEAP intermediates, outputs 62-column parquet |
| 2 | `02_county_profile.py` | ~30 s | Aggregates 12 metrics per county, generates 4-quadrant bubble chart (BER vs retrofit rate) |
| 3 | `03_train_model.py` | ~10–20 min | Trains LightGBM regressor (BER prediction) + classifier (retrofit likelihood) with 5-fold CV; computes global SHAP on 10K sample |
| 4 | `04_equity_gap.py` | ~5 s | Computes composite Equity Gap Score per county (fuel poverty × retrofit gap × CO₂ intensity) |
| 5 | `05_xai_explainer.py` | ~3–5 min | Delta-SHAP scenario analysis for retrofit measures; county-level SHAP attribution for classifier |

### Pipeline Output

All outputs are written to `final-implementation/outputs/`:

| File | Script | Description |
|------|--------|-------------|
| `clean_data_55col.parquet` | 01 | ~1.35M × 62 col cleaned & engineered dataset (Snappy) |
| `cleaning_report.txt` | 01 | Leakage verification, null check, target stats, column list |
| `county_profiles.csv` | 02 | 26-county aggregated statistics (12 metrics) |
| `county_summary_table.md` | 02 | Formatted markdown table of county profiles |
| `county_bubble_chart.png` | 02 | BER vs retrofit rate bubble chart (4-quadrant policy view) |
| `lgbm_model.pkl` | 03 | LightGBM regressor + OrdinalEncoders + metadata |
| `lgbm_classifier.pkl` | 03 | LightGBM classifier (retrofit likelihood) + encoders + metadata |
| `model_report.txt` | 03 | R², RMSE, MAE for LightGBM, Random Forest, Ridge + CV results |
| `classifier_report.txt` | 03 | ROC-AUC, Average Precision, per-class F1 |
| `feature_importance.csv` | 03 | Gain-based feature importance (all features) |
| `shap_values_global.csv` | 03 | 10K × n_features SHAP matrix (regressor) |
| `shap_bar.png` | 03 | Top 30 features by mean \|SHAP\| (bar chart) |
| `shap_summary.png` | 03 | SHAP beeswarm — top 30 features (regressor) |
| `shap_summary_classifier.png` | 03/05 | SHAP beeswarm — top 25 features (classifier) |
| `roc_curve.png` | 03 | ROC curve with AUC annotation |
| `pr_curve.png` | 03 | Precision-Recall curve with Average Precision annotation |
| `equity_gap_county.csv` | 04 | Equity Gap Score + rank per county |
| `equity_gap_bar.png` | 04 | Horizontal bar chart sorted by equity gap score |
| `xai_summary.csv` | 05 | Cross-measure delta-SHAP summary (measure × example) |
| `county_shap_avg.csv` | 05 | Mean \|SHAP\| per county for classifier (26 × n_features) |
| `scenario_reports/*.png` | 05 | Per-measure per-example SHAP before/after plots |
| `scenario_reports/*_delta_shap.csv` | 05 | Feature-level delta SHAP attributions |
| `*.log` | all | Timestamped execution logs (append mode) |

### Running Individual Steps

To run a specific pipeline step:

```bash
cd final-implementation
source venv/bin/activate
python scripts/01_clean_and_prepare.py
```

> **Note:** `05_xai_explainer.py` requires `config/retrofit_measures.json` (retrofit scenario definitions) and the outputs of both `03_train_model.py` (`lgbm_model.pkl`, `lgbm_classifier.pkl`) and `01_clean_and_prepare.py` (`clean_data_55col.parquet`).

## Documentation

This repository contains comprehensive documentation covering the entire data engineering and exploratory analysis pipeline:

1. **[Data Cleaning & Feature Selection](data_cleaning.md)**: 
   Outlines the Phase 1 and Phase 2 approaches used to reduce the raw dataset from ~215 columns to a curated set of 45 physical and efficiency drivers (mitigating target leakage and sparsity).

2. **[Row Imputation & Statistical Analysis](imputation_analysis_report.md)**: 
   Documents the methodology for handling missing data, categorizing NMAR vs. MAR blocks, applying Contextual Regression Imputation vs Mode, and using the Interquartile Range (IQR) for outlier suppression to preserve distribution integrity.

3. **[PCA, LDA, MCA & FAMD Dimensionality Reduction](pca_lda_report_v2.md)**: 
   The capstone exploratory data analysis report. It uses four distinct dimensionality reduction methods across continuous and categorical data types to extract the final feature sets. Includes a rigorous VIF/Pearson multicollinearity audit resulting in 35 final features suitable for multiple modelling pathways.

4. **[Initial Row Cleaning Discovery](row_cleaning_balaji_14_mar.md)**: 
   Raw notes, statistical outcomes, and lecture-grounded justifications (DCU MSc) regarding the decision to drop `PredominantRoofType` and other heavily missing blocks.
