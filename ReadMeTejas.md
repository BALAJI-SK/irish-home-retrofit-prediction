# BER Rating Prediction — ML Pipeline

**Author:** Tejas Shivakumar
**Input dataset:** `46_Col_final_with_county.parquet` (1.35M rows, fully preprocessed)
**Target:** `BerRating` (kWh/m²/yr) — numerical regression

---

## What this covers

Data cleaning and preprocessing were completed separately by the team. This documents the ML modelling work built on top of that — the pipeline, models, tuning, and results.

---

## Pipeline — `ml_pipeline.py`

```
Load 46_Col_final_with_county.parquet
        │
        ▼
Train / Test Split  (80% / 20%, stratified on 10 BER deciles)
  Train : ~1,081,265 rows
  Test  :   270,317 rows  ← held-out, never seen during tuning
        │
        ▼
Hyperparameter Tuning
  Method  : RandomizedSearchCV
  CV      : 5-fold KFold
  Iterations : 20 per model
  Tuning subset : 200,000 stratified rows (from train set only)
  Scoring : R²
        │
        ▼
Retrain on full train set with best params
        │
        ▼
Evaluate on held-out test set
  Metrics: R², RMSE, MAE
        │
        ▼
Outputs → model_analysis_outputs/
```

### Preprocessing per model type

| Step | Ridge | Random Forest / LightGBM / XGBoost |
|---|---|---|
| Numeric imputation | Median | Passed through |
| Categorical imputation | Mode | Passed through |
| Encoding | OrdinalEncoder | OrdinalEncoder |
| Scaling | StandardScaler | None (tree models are scale-invariant) |

---

## Models

Four models are trained forming a complexity ladder — each one answers a specific question about the data.

### 1. Ridge Regression — Linear Baseline
- Establishes the maximum R² any linear model can achieve on this data
- The gap between Ridge and the tree models quantifies how much non-linearity BER physics contains
- Hyperparameter: `alpha` (L2 regularisation penalty)

### 2. Random Forest — Bagging Ensemble
- Parallel ensemble of decision trees, each on a bootstrap sample
- Comparing RF vs LightGBM isolates the value of boosting over bagging
- Provides OOB score as a free internal validation check

### 3. LightGBM — Gradient Boosting
- Trees built sequentially, each correcting residuals of the previous ensemble
- Histogram binning handles 1.35M rows efficiently
- Leaf-wise tree growth achieves high accuracy with fewer trees

### 4. XGBoost — Gradient Boosting
- Same boosting principle as LightGBM, using `tree_method="hist"` for scale
- Depth-wise tree growth (vs LightGBM's leaf-wise) — direct comparison to measure which strategy suits BER prediction better

---

## Hyperparameter Search Spaces

### Ridge
```python
alpha: np.logspace(-2, 4, 50)   # 0.01 → 10,000
```

### Random Forest
```python
n_estimators:      [100, 200, 300]
max_depth:         [None, 15, 25, 35]
min_samples_split: [2, 5, 10]
min_samples_leaf:  [1, 2, 4]
max_features:      ["sqrt", "log2", 0.5]
```

### LightGBM
```python
n_estimators:      [200, 400, 600, 800]
num_leaves:        [31, 63, 127]
learning_rate:     [0.05, 0.08, 0.1, 0.15]
max_depth:         [-1, 6, 10]
min_child_samples: [10, 20, 50]
subsample:         [0.7, 0.8, 1.0]
colsample_bytree:  [0.7, 0.9, 1.0]
reg_alpha:         [0, 0.1, 0.5]
reg_lambda:        [0, 0.1, 1.0]
```

### XGBoost
```python
n_estimators:     [200, 400, 600, 800]
max_depth:        [4, 6, 8, 10]
learning_rate:    [0.05, 0.08, 0.1, 0.15]
min_child_weight: [1, 5, 10]
subsample:        [0.7, 0.8, 1.0]
colsample_bytree: [0.7, 0.9, 1.0]
reg_alpha:        [0, 0.1, 0.5]
reg_lambda:       [0.5, 1.0, 2.0]
```

---

## Results

> Validation R² = 5-fold CV R² on the 200K stratified tuning subset (no separate validation split).
> All RMSE and MAE values in kWh/m²/yr.

### Full Model Comparison

| Model | CV R² (Validation) | CV Std | Train R² | Test R² | Train RMSE | Test RMSE | Test MAE | Train Time | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **0.8489** | **0.1052** | **0.9816** | **0.9404** | **22.10** | **37.97** | **17.96** | **293s** | **Mild Overfit** |
| LightGBM | 0.8667 | 0.1062 | 0.9670 | 0.9361 | 29.59 | 39.32 | 16.55 | 15s | Well-fitted |
| XGBoost | 0.8638 | 0.1035 | 0.9790 | 0.9252 | 23.58 | 42.55 | 15.26 | 26s | Overfitting |
| Ridge | 0.7238 | 0.0920 | 0.7129 | 0.7795 | 87.26 | 73.06 | 41.23 | 4s | Underfitting |

*Best model bolded by Test R².*

### Verdict rationale

| Model | Train R² | Test R² | Overfit Gap | Train RMSE | Test RMSE | Verdict | Reasoning |
|---|---|---|---|---|---|---|---|
| Ridge | 0.7129 | 0.7795 | −0.0666 | 87.26 | 73.06 | **Underfitting** | Test R² > Train R² and Train RMSE > Test RMSE — model is too simple; the linear form is the binding constraint, not regularisation |
| Random Forest | 0.9816 | 0.9404 | +0.0412 | 22.10 | 37.97 | **Mild Overfit** | Train RMSE (22.10) much lower than Test RMSE (37.97); `max_depth=35` causes memorisation but still generalises well |
| LightGBM | 0.9670 | 0.9361 | +0.0309 | 29.59 | 39.32 | **Well-fitted** | Smallest overfit gap among tree models; Train/Test RMSE spread is moderate and consistent |
| XGBoost | 0.9790 | 0.9252 | +0.0538 | 23.58 | 42.55 | **Overfitting** | Largest overfit gap; Train RMSE (23.58) vs Test RMSE (42.55) — depth-wise growth memorises more than LightGBM's leaf-wise strategy |

### Best hyperparameters found

| Model | Best params |
|---|---|
| Ridge | `alpha = 47.15` |
| Random Forest | `n_estimators=300, max_depth=35, min_samples_split=5, min_samples_leaf=1, max_features=0.5` |
| LightGBM | `n_estimators=800, num_leaves=31, learning_rate=0.08, max_depth=6, subsample=0.7, colsample_bytree=1.0, reg_alpha=0.5, reg_lambda=0.1, min_child_samples=10` |
| XGBoost | `n_estimators=800, max_depth=8, learning_rate=0.05, subsample=1.0, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, min_child_weight=5` |

### Key findings

- **Ridge → Random Forest (+16.1 pp Test R²):** BER physics is fundamentally non-linear — a straight line cannot model U-value interactions and threshold insulation effects
- **Random Forest wins on Test R²** (0.9404) despite LightGBM having a higher CV R² (0.8667 vs 0.8489) — the full 1.08M train set benefits bagging more than the 200K subsample CV suggested
- **LightGBM is the practical winner:** Test R² within 0.004 of Random Forest, trains 20× faster (15s vs 293s), and has the smallest overfit gap (0.031)
- **Ridge underfitting confirmed:** Train RMSE (87.26) is actually *higher* than Test RMSE (73.06) — the model generalises to the test set better than it fits training, a classic underfitting signature
- **Random Forest overfitting:** Train RMSE of 22.10 vs Test RMSE of 37.97 — a 71% jump; `max_depth=35` allows near-perfect memorisation
- **XGBoost worst overfit:** Train RMSE 23.58 → Test RMSE 42.55 (80% jump), largest gap of all tree models

---

## How to Run

```bash
source .venv/bin/activate
python ml_pipeline.py
```

Outputs saved to `model_analysis_outputs/`:
- `ml_pipeline_comparison.png` — bar chart of CV R², Test R², Test RMSE per model
- `ml_pipeline_report.txt` — full metrics and best hyperparameters per model

---

*Updated: April 6, 2026 — full pipeline run with Train RMSE now captured for all 4 models*
