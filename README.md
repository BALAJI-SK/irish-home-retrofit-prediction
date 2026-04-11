# Irish Home Retrofit Prediction

Machine learning project to predict Irish home energy ratings (BER) based on physical and efficiency drivers from the SEAI dataset (1.35 million rows).

## Getting Started

### Prerequisites

- **Python 3.9+** (project uses Python 3.9)
- **SEAI Data Files** – Download from SEAI BER database:
  - `Public Search Data.csv` (1.35M rows, 211 columns) – place in `~/Downloads/`
  - `Col Final with County.parquet` (pre-cleaned parquet, 46 columns) – place in `~/Downloads/`

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

1. **01_clean_and_prepare.py** – Merges SEAI data sources, engineers features, filters on BER rating/year
2. **02_county_profile.py** – Generates county-level profiling and bubble chart visualizations
3. **03_train_model.py** – Trains LightGBM classifier on retrofit likelihood
4. **04_equity_gap.py** – Analyzes retrofit equity gaps by county and demography
5. **05_xai_explainer.py** – Generates SHAP explanations for model interpretability

### Pipeline Output

All outputs are written to `final-implementation/outputs/`:

- **Data files:**
  - `clean_data_55col.parquet` – Cleaned & engineered dataset (62 columns)
  - `lgbm_model.pkl` – Trained LightGBM regressor
  - `lgbm_classifier.pkl` – Trained LightGBM classifier (retrofit likelihood)

- **Reports & Visualizations:**
  - `county_profiles.csv` – County-level statistics
  - `county_summary_table.md` – Markdown summary of county profiles
  - `county_bubble_chart.png` – Bubble chart of county retrofit potential
  - `roc_curve.png`, `pr_curve.png` – Model performance curves
  - `shap_summary_classifier.png` – SHAP feature importance

- **Logs:**
  - `*.log` – Step execution logs for debugging

### Running Individual Steps

To run a specific pipeline step:

```bash
cd final-implementation
source venv/bin/activate
python scripts/01_clean_and_prepare.py
```

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
