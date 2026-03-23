# Irish Home Retrofit Prediction

Machine learning project to predict Irish home energy ratings (**BerRating** — kWh/m²/yr) based on physical and efficiency drivers from the SEAI dataset (1.35 million rows).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Data Pipeline Documentation](#3-data-pipeline-documentation)
4. [Feature Engineering Summary](#4-feature-engineering-summary)
5. [Dimensionality Reduction](#5-dimensionality-reduction)
6. [Model Selection — Why These 3 Models](#6-model-selection--why-these-3-models)
7. [Model Results & Comparison](#7-model-results--comparison)
8. [Overfitting / Underfitting Analysis](#8-overfitting--underfitting-analysis)
9. [Visual Diagnostics — All Plots](#9-visual-diagnostics--all-plots)
10. [Model Advantages & Disadvantages](#10-model-advantages--disadvantages)
11. [Final Verdict](#11-final-verdict)
12. [Repository Structure](#12-repository-structure)

---

## 1. Project Overview

The **Building Energy Rating (BER)** is a continuous numerical score (kWh/m²/yr) that quantifies the energy performance of Irish residential buildings. It is assessed by the Sustainable Energy Authority of Ireland (SEAI) and maps to letter bands A1 (most efficient) through G (least efficient).

**Objective:** Train and evaluate regression models that can predict the BerRating number from a building's physical and mechanical properties — without relying on derived outputs or target-leaking features.

**Why Regression (not Classification)?**
- `BerRating` is a continuous float (e.g., 184.3 kWh/m²/yr), not a label
- Regression preserves the ordinal magnitude of energy performance
- The BER letter bands are simply binned ranges of this number — predicting the number is strictly more informative
- Downstream retrofit cost estimation requires the numerical value, not just a band

---

## 2. Dataset

| Property | Value |
|----------|-------|
| **Source** | SEAI BER Public Search API |
| **Raw size** | 1,351,582 rows × 215 columns |
| **After cleaning** | 1,351,582 rows × 45 columns |
| **Target variable** | `BerRating` (float32, kWh/m²/yr) |
| **Target range** | −472 to 32,134 (clipped to 1st–99th percentile for modelling) |
| **Target mean** | 205.9 kWh/m²/yr |
| **Target median** | 184.3 kWh/m²/yr |
| **Missing values** | 0 (after imputation) |
| **Feature types** | 29 continuous (float32) + 13 categorical |
| **Data file** | `46_Col_final_with_county.parquet` |

### BER Energy Class Distribution

| Class | Range (kWh/m²/yr) | Typical Building |
|-------|-------------------|-----------------|
| A1–A3 | 0–75 | New passive/near-zero energy |
| B1–B3 | 75–150 | Recent builds, well insulated |
| C1–C3 | 150–225 | Average modern stock |
| D1–D2 | 225–300 | Older stock, some retrofits |
| E1–E2 | 300–380 | Pre-1980 uninsulated |
| F–G   | 380+ | Oldest / worst performing |

---

## 3. Data Pipeline Documentation

Full documentation covering the entire data engineering pipeline:

1. **[Data Cleaning & Feature Selection](data_cleaning.md)**
   Outlines Phase 1 and Phase 2 approaches to reduce the raw dataset from ~215 columns to 45 physical and efficiency drivers — mitigating target leakage and sparsity.

2. **[Row Imputation & Statistical Analysis](imputation_analysis_report.md)**
   Documents the methodology for handling missing data, categorising NMAR vs. MAR blocks, applying Contextual Regression Imputation vs Mode, and using IQR for outlier suppression.

3. **[PCA, LDA, MCA & FAMD Dimensionality Reduction](pca_lda_report_v2.md)**
   Capstone EDA report using four complementary dimensionality reduction techniques. Includes a rigorous VIF/Pearson multicollinearity audit resulting in 35 final features.

4. **[Initial Row Cleaning Discovery](row_cleaning_balaji_14_mar.md)**
   Raw notes and justifications regarding the decision to drop `PredominantRoofType` and other heavily missing blocks.

---

## 4. Feature Engineering Summary

### Phase 1 — Systematic Data Reduction (215 → 76 columns)

| Step | Action | Result |
|------|--------|--------|
| Sparsity filter | Drop columns with >50% nulls | 215 → 138 |
| Suspicious characters | Drop columns with >50% placeholder values | 138 → 97 |
| Metadata removal | Drop all `*Description` text columns | 97 → 92 |
| Cardinality filter | Drop categoricals with >20 unique values | 92 → 85 |
| Target correlation | Drop features with >95% correlation to target | 85 → 76 |

### Phase 2 — Domain-Driven Reduction (76 → 45 columns)

| Step | Columns Dropped | Reason |
|------|----------------|--------|
| Data leakage | `CO2Rating`, `MPCDERValue`, `CPC`, etc. | Directly derived from BerRating |
| Administrative | `TypeofRating`, `PurposeOfRating` | Non-physical metadata |
| Outcome variables | `DeliveredLightingEnergy`, etc. | Results, not causes |
| Redundant components | `FirstWallUValue`, `FirstWallArea`, etc. | Replaced by aggregate |
| Multi-collinearity | `GroundFloorArea`, `FirstFloorArea` | Redundant with `FloorArea` |

### Final Feature Set After VIF Audit (35 features for modelling)

**Drop (VIF > 25 — collinear):** `SHRenewableResources`, `WHRenewableResources`, `HSEffAdjFactor`, `WHEffAdjFactor`

**Drop (NZV categoricals):** `WarmAirHeatingSystem`, `CylinderStat`, `CombinedCylinder`, `CountyName`

**Final 27 continuous features:**
`WHMainSystemEff`, `RoofArea`, `HSMainSystemEfficiency`, `HSSupplHeatFraction`, `WallArea`, `HSSupplSystemEff`, `UValueWall`, `UValueFloor`, `UValueWindow`, `Year_of_Construction`, `WindowArea`, `DistributionLosses`, `LivingAreaPercent`, `UValueRoof`, `NoStoreys`, `HeatSystemControlCat`, `UvalueDoor`, `ThermalBridgingFactor`, `NoCentralHeatingPumps`, `DoorArea`, `SupplWHFuel`, `NoOfSidesSheltered`, `PercentageDraughtStripped`, `HeatSystemResponseCat`, `NoOfFansAndVents`, `SupplSHFuel`, `FloorArea`

**Final 10 categorical features:**
`DwellingTypeDescr`, `VentilationMethod`, `StructureType`, `SuspendedWoodenFloor`, `CHBoilerThermostatControlled`, `OBBoilerThermostatControlled`, `OBPumpInsideDwelling`, `UndergroundHeating`, `ThermalMassCategory`, `PredominantRoofType`

---

## 5. Dimensionality Reduction

Four complementary techniques were applied — each answering a different analytical question:

| Technique | Data Type | Key Finding |
|-----------|-----------|-------------|
| **PCA** | Continuous (31 features) | PC1 = thermal efficiency cluster; PC2 = building size; 19 components → 90% variance |
| **MCA** | Categorical (9 features) | Dim1 = traditional vs modern; Dim2 = detached house profiles; Dim3 = timber-frame MVHR |
| **LDA** | Continuous → `DwellingTypeDescr` | `ThermalBridgingFactor` and `HSSupplHeatFraction` are the strongest dwelling-type discriminators |
| **FAMD** | Mixed (43 features) | Factor 1 aligns with PCA's efficiency cluster; confirms PCA and MCA findings converge |

**Top features by PCA aggregate loading (90% variance, 19 PCs):**

| Rank | Feature | Loading |
|------|---------|---------|
| 1 | `PercentageDraughtStripped` | 3.556 |
| 2 | `HeatSystemResponseCat` | 3.473 |
| 3 | `SupplWHFuel` | 3.289 |
| 4 | `NoOfFansAndVents` | 3.224 |
| 5 | `DoorArea` | 3.143 |

---

## 6. Model Selection — Why These 3 Models

### The Three-Tier Strategy

The three models form a **complexity ladder** — each step is justified by the measurable performance gain it delivers over the step below.

```
Ridge Regression        Random Forest           LightGBM
(Linear Baseline)  →   (Ensemble Baseline)  →  (SOTA Gradient Boost)
       ↓                       ↓                        ↓
"Can a straight line     "Do parallel tree          "How good can
 explain BER?"            ensembles help?"           we actually get?"
```

The R² gap (0.836 → 0.920 → 0.959) at each step **proves** non-linear complexity is real and that sequential correction (boosting) adds value beyond parallel ensembling (bagging).

---

### Model 1 — LightGBM (Gradient Boosting)

**Why chosen:**
- Dataset has 1.35M rows with 37 mixed features — LightGBM was specifically engineered for this scale
- BER involves multiplicative interactions between U-values, heating efficiency, ventilation, and construction year — gradient boosting captures these through sequential residual correction
- Native categorical support — no information loss from encoding `DwellingTypeDescr`, `VentilationMethod`, etc.
- PCA confirmed no single linear direction explains BerRating — multiple non-linear components are needed
- Leaf-wise tree growth finds deeper, more accurate splits than level-wise methods

**How it works:**

```
Tree 1 → coarse prediction → residuals remain
Tree 2 → fits the residuals → smaller residuals remain
Tree 3 → fits those residuals → even smaller
...
Tree 500 → tiny corrections → final prediction
```

Each tree corrects the mistakes of all previous trees. The final prediction is the sum of all 500 trees' outputs.

**Advantages:**

| Advantage | Why It Matters Here |
|-----------|-------------------|
| Handles 1.35M rows efficiently | Histogram-based binning reduces memory usage |
| Native categorical features | No encoding bias for 10 categorical columns |
| Leaf-wise tree growth | Finds deeper, more accurate splits |
| Built-in L1/L2 regularisation | `reg_alpha`, `reg_lambda` control overfitting |
| Early stopping | Halts when validation loss stops improving |
| Handles missing values natively | Learns optimal direction for NaN splits |
| Feature importance built-in | Gain-based importance from boosting process |
| Fastest training | 14s for 160K rows — same as Random Forest but higher accuracy |

**Disadvantages:**

| Disadvantage | Impact |
|-------------|--------|
| Least interpretable | Cannot easily explain a single prediction |
| Many hyperparameters | `num_leaves`, `learning_rate`, `subsample` all interact |
| Can memorise training data | Train R²=0.976 vs Test R²=0.959 — small gap |
| Sensitive to learning rate | Too high diverges; too low needs thousands of rounds |
| Black box | Regulatory/audit contexts may require explainability |
| No uncertainty estimates | Cannot output confidence intervals by default |

---

### Model 2 — Random Forest (Bagging Ensemble)

**Why chosen:**
- Provides a strong, robust ensemble baseline to isolate how much sequential correction (boosting) adds over parallel ensembling (bagging)
- OOB score gives a free internal cross-validation estimate with zero data leakage
- BER depends on multiple independent physical sub-systems (insulation, heating, ventilation) — Random Forest's random feature subsets mirror this independence
- Robust to outlier buildings — extreme BerRating values influence at most a subset of trees

**How it works:**

```
Bootstrap sample 1 → Tree 1 → prediction₁  ─┐
Bootstrap sample 2 → Tree 2 → prediction₂   ├→ Average → Final prediction
Bootstrap sample 3 → Tree 3 → prediction₃  ─┘
...
Bootstrap sample 200 → Tree 200 → prediction₂₀₀
```

Each tree is built independently on a different random sample. Predictions are averaged across all 200 trees.

**Advantages:**

| Advantage | Why It Matters Here |
|-----------|-------------------|
| OOB validation | OOB=0.9212 ≈ Test=0.9198 — zero leakage confirmed |
| Highly parallelisable | All 200 trees built simultaneously (`n_jobs=-1`) |
| Robust to outliers | Extreme BerRating values diluted across bootstrap samples |
| Stable feature importances | MDI consistent across folds (CoV=0.116) |
| No scaling needed | Tree splits are threshold-based — feature scale irrelevant |
| Low hyperparameter sensitivity | `n_estimators=200` saturates performance |

**Disadvantages:**

| Disadvantage | Impact |
|-------------|--------|
| Slower than LightGBM at scale | Same training time but worse accuracy |
| Memory intensive | 200 full trees held in memory simultaneously |
| Cannot extrapolate | Predictions bounded by training range |
| Unlimited depth causes mild overfit | Train R²=0.944 vs Test R²=0.920 (2.38% gap) |
| Biased MDI feature importance | High-cardinality features get inflated importance |
| No sequential correction | Cannot iteratively reduce residuals like boosting |

---

### Model 3 — Ridge Regression (Linear Baseline)

**Why chosen:**
- Acts as the essential lower bound — quantifies how much non-linearity exists in the problem
- Directly tests the linear assumption: the R² gap (0.836 vs 0.959) proves non-linearity is significant
- Extremely fast (0.8s) — instant sanity check
- Interpretable coefficients — shows directional influence of each physical feature on BerRating
- Required for academic completeness: always compare complex models against a simple baseline

**How it works:**

```
BerRating = β₀ + β₁(UValueWall) + β₂(HSMainSystemEff) + β₃(Year) + ... + βₙ(featureₙ)
                                        + λ Σβᵢ²  ← Ridge L2 penalty
```

Every feature gets one fixed coefficient. No interactions, no thresholds, no non-linearity.

**Advantages:**

| Advantage | Why It Matters Here |
|-----------|-------------------|
| Fully interpretable | Coefficients directly quantify marginal effect of each feature |
| Extremely fast | 0.8s training — instant iteration |
| Zero generalisation gap | R² gap of 0.0028 — no overfitting risk |
| Insensitive to hyperparameters | Alpha across 7 orders of magnitude had zero effect |
| Stable CV | std=0.0044 across 5 folds |
| Regulatory transparency | Accepted in audit/compliance contexts |

**Disadvantages:**

| Disadvantage | Impact |
|-------------|--------|
| Cannot model non-linear interactions | BER physics is multiplicative — wrong functional form |
| Structural ceiling at R²=0.836 | 16% variance permanently unexplained regardless of tuning |
| Heteroscedasticity in residuals | Errors grow with BerRating — confirmed in residual plots |
| Ordinal encoding distortion | Treats categorical levels as false numerical relationships |
| Adding data does not help | Learning curve flat from first 3,000 samples |
| Systematic underprediction | High BerRating houses consistently underpredicted |

---

### Side-by-Side Comparison

| Criterion | Ridge Regression | Random Forest | LightGBM |
|-----------|:---:|:---:|:---:|
| Handles non-linearity | ✗ | ✓ | ✓✓ |
| Handles 1.35M rows | ✓ | ✓ | ✓✓ |
| Native categoricals | ✗ | ✗ | ✓✓ |
| Interpretability | ✓✓ | ✓ | ✗ |
| Training speed | ✓✓ | ✓ | ✓✓ |
| Test R² | 0.834 | 0.920 | **0.959** |
| Overfitting risk | None | Low | Very low |
| Hyperparameter sensitivity | Very low | Low | Medium |
| Handles feature interactions | ✗ | ✓ | ✓✓ |
| OOB / internal validation | ✗ | ✓✓ | ✗ |
| **Overall for BER task** | Baseline only | Strong 2nd | **Best choice** |

---

## 7. Model Results & Comparison

All results on 200,000-row stratified sample (80/20 train/test split).

### Test Set Metrics

| Model | Test R² | Test RMSE | Test MAE | MAPE% | Macro F1 (BER bands) | Train Time |
|-------|---------|-----------|----------|-------|---------------------|------------|
| **LightGBM** | **0.9588** | **25.49** | **16.04** | **9.77%** | **0.549** | 14.1s |
| Random Forest | 0.9198 | 35.55 | 22.12 | 12.38% | 0.469 | 14.2s |
| Ridge Regression | 0.8335 | 51.23 | 35.92 | 25.54% | 0.305 | 0.8s |

### Train vs Test Gap

| Model | Train R² | Test R² | R² Gap | Train RMSE | Test RMSE | RMSE Gap |
|-------|----------|---------|--------|------------|-----------|----------|
| LightGBM | 0.9764 | 0.9588 | 0.0176 | 19.33 | 25.49 | 6.16 |
| Random Forest | 0.9436 | 0.9198 | 0.0238 | 29.85 | 35.55 | 5.70 |
| Ridge Regression | 0.8363 | 0.8335 | 0.0028 | 50.86 | 51.23 | 0.36 |

### 5-Fold Cross-Validation Results

| Model | CV Val R² Mean | CV Val R² Std | CV Train R² | CV Gap |
|-------|---------------|--------------|-------------|--------|
| LightGBM | 0.9474 | **0.0011** | 0.9726 | 0.0252 |
| Random Forest | 0.9090 | 0.0031 | 0.9345 | 0.0255 |
| Ridge Regression | 0.8355 | 0.0044 | 0.8362 | 0.0006 |

### Prediction Error Percentiles

| Model | p50 Abs Error | p90 Abs Error |
|-------|--------------|--------------|
| LightGBM | 25 kWh | ~55 kWh |
| Random Forest | 30 kWh | ~75 kWh |
| Ridge Regression | 43 kWh | ~100 kWh |

---

## 8. Overfitting / Underfitting Analysis

### Section 1 — Train vs Test Gap
All three models have R² gaps well below the 5% overfit threshold. LightGBM and Random Forest show mild orange-zone gaps due to model expressiveness; Ridge shows near-zero gap but only because its linear form cannot overfit.

### Section 2 — Cross-Validation Stability
All models are **STABLE** (CV std < 0.01). LightGBM achieves the tightest consistency (std=0.0011) across 5 folds — fold scores vary by only 0.003 in absolute terms.

### Section 3 — Learning Curves
- **LightGBM:** Converging curves → **WELL-FITTED**. Gap narrows as data grows.
- **Random Forest:** Converging curves → **WELL-FITTED**. Gap shrinks but not fully closed — more data would help.
- **Ridge Regression:** Flat parallel curves from the first sample → **STRUCTURAL UNDERFITTING**. Adding more data has zero effect.

### Section 4 — Loss Curve (LightGBM)
RMSE gap stabilises at ~6 kWh from round 50 onwards and never grows. Parallel train/val curves in the final 80 rounds confirm no divergence. Early stopping patience of 50 rounds never triggered — model improved continuously to round 500.

### Section 5 — Residual Analysis

| Model | Residual Std | Mean | Heteroscedasticity | Q-Q r |
|-------|-------------|------|-------------------|-------|
| LightGBM | 25.49 | 0.09 | Minimal | 0.937 |
| Random Forest | 35.55 | 0.19 | Mild | 0.910 |
| Ridge Regression | 51.23 | 0.13 | **Clear fan-shape** | 0.872 |

### Section 6 — Feature Importance Stability

| Model | Mean CoV | Max CoV | Stability |
|-------|----------|---------|-----------|
| LightGBM | **0.082** | 0.566 | High — importances shift 8% across folds |
| Random Forest | 0.116 | 0.607 | Good — consistent top-feature ranking |

Top features (both models agree): `UValueWall`, `WHMainSystemEff`, `HSMainSystemEfficiency`, `Year_of_Construction`, `DistributionLosses`

### Section 7 — OOB Score (Random Forest)
**OOB R²=0.9212 vs Test R²=0.9198** — gap of 0.0014. This near-perfect match confirms the Random Forest generalises without data leakage across all tree counts tested (10–200).

### Section 8 — Early Stopping (LightGBM)
Best iteration = 500/500 (100% of max rounds used). Validation RMSE improved continuously — early stopping never triggered. Recommendation: increase `n_estimators` to 1000 and let early stopping find the true optimal round.

### Final Fit Verdicts

| Model | Verdict | Root Cause |
|-------|---------|-----------|
| LightGBM | **WELL-FITTED** | `num_leaves=127` provides minor memorisation; controlled by regularisation |
| Random Forest | **WELL-FITTED** | `max_depth=None` creates mild variance; bagging absorbs most of it |
| Ridge Regression | **STRUCTURALLY UNDERFITTING** | Linear functional form cannot model non-linear building physics |

### Recommended Fixes

**LightGBM** (to close the 6.2 kWh RMSE gap):
- Reduce `num_leaves`: 127 → **63**
- Increase `n_estimators`: 500 → **1000** with early stopping
- Increase `min_child_samples`: 50 → **100**
- Increase `reg_lambda`: 0.1 → **1.0**
- Expected: RMSE gap 6.2 → ~3.5 kWh, Test R² maintained at 0.960+

**Random Forest** (to reduce the 2.38% R² gap):
- Set `max_depth`: None → **15**
- Increase `min_samples_leaf`: 10 → **30**
- Increase `n_estimators`: 200 → **500**
- Expected: R² gap 2.38% → ~1.5%, Test R² rising to ~0.930

**Ridge Regression** (to overcome structural ceiling):
- Add **degree-2 polynomial features** for top continuous predictors
- Switch categorical encoding from ordinal → **one-hot**
- Expected: Test R² 0.836 → ~0.900+

---

## 9. Visual Diagnostics — All Plots

### Model Comparison Plots

| Plot | File | What It Shows |
|------|------|---------------|
| **Plot 1** | `plot1_master_scorecard.png` | Train vs Test R², RMSE, MAE for all 3 models |
| **Plot 2** | `plot2_generalisation_gap.png` | R² and RMSE generalisation gap (green/orange/red zones) |
| **Plot 3** | `plot3_cv_stability.png` | 5-fold CV train vs val R² per fold with stability verdict |
| **Plot 4** | `plot4_learning_curves.png` | Bias-variance diagnosis as training size increases |
| **Plot 5** | `plot5_lgbm_loss_curve.png` | LightGBM train vs val RMSE per boosting round |
| **Plot 6** | `plot6_actual_vs_predicted.png` | Density hexbin actual vs predicted for all 3 models |
| **Plot 7** | `plot7_residual_analysis.png` | Residuals vs predicted, distribution, Q-Q normality (3×3 grid) |
| **Plot 8** | `plot8_feature_importance_stability.png` | Feature importance ± std across 5 CV folds + CoV heatmap |
| **Plot 9** | `plot9_oob_score.png` | Random Forest OOB vs test score across tree counts |
| **Plot 10** | `plot10_radar_chart.png` | Multi-axis radar chart comparing all 3 models |
| **Plot 11** | `plot11_error_distribution.png` | Violin plot + cumulative error CDF |
| **Plot 12** | `plot12_verdict_scorecard.png` | Colour-coded final scorecard across all 8 diagnostic sections |

### Earlier EDA / Dimensionality Reduction Plots

| Plot | File | What It Shows |
|------|------|---------------|
| PCA Scree | `pca_scree.png` | Variance explained per principal component |
| PCA Scatter | `pca_scatter.png` | PC1 vs PC2 coloured by BerRating |
| PCA Heatmap | `pca_heatmap.png` | Feature loadings on PC1–PC5 |
| PCA Biplot | `pca_biplot.png` | Combined score + loading plot |
| LDA Scatter | `lda_scatter.png` | LD1 vs LD2 coloured by DwellingTypeDescr |
| LDA Variance | `lda_variance.png` | Variance explained per discriminant component |
| MCA Scree | `mca_scree.png` | Inertia per MCA dimension |
| MCA Scatter | `mca_scatter.png` | MCA Dim1 vs Dim2 categorical pattern space |
| FAMD Variance | `famd_variance.png` | Top driver per FAMD component |
| FAMD Scatter | `famd_scatter.png` | FAMD Factor1 vs Factor2 coloured by BerRating |
| VIF Heatmap | `vif_correlation_heatmap.png` | Pearson correlation + VIF flags |
| Cramér's V | `cramers_v_heatmap.png` | Categorical-categorical association strength |
| Eta Squared | `eta_squared_heatmap.png` | Categorical-continuous association (effect size) |

---

## 10. Model Advantages & Disadvantages

### Complete Comparison Table

| Criterion | Ridge Regression | Random Forest | LightGBM |
|-----------|:---:|:---:|:---:|
| Handles non-linearity | ✗ | ✓ | ✓✓ |
| Handles 1.35M rows | ✓ | ✓ | ✓✓ |
| Native categorical features | ✗ | ✗ | ✓✓ |
| Interpretability | ✓✓ | ✓ | ✗ |
| Training speed | ✓✓ | ✓ | ✓✓ |
| Test R² | 0.834 | 0.920 | **0.959** |
| Overfitting risk | None | Low | Very low |
| Hyperparameter sensitivity | Very low | Low | Medium |
| Handles feature interactions | ✗ | ✓ | ✓✓ |
| OOB / internal validation | ✗ | ✓✓ | ✗ |
| Extrapolation beyond training range | ✓ | ✗ | ✗ |
| Regulatory transparency | ✓✓ | ✓ | ✗ |
| Uncertainty quantification | ✗ | ✓ (variance of trees) | ✗ |
| **Overall for BER task** | Baseline only | Strong 2nd | **Best choice** |

---

## 11. Final Verdict

### Performance Summary

```
LightGBM   ████████████████████████  R² = 0.9588  ★ BEST MODEL
Rand Forest ███████████████████░░░░  R² = 0.9198  ✓ GOOD (2nd)
Ridge       ████████████████░░░░░░░  R² = 0.8335  — Baseline only
```

### Why LightGBM Wins

1. **Highest accuracy across all metrics** — R²=0.959, RMSE=25.5 kWh, MAE=16.0 kWh
2. **Tightest cross-validation** — CV std=0.0011, fold scores vary by only 0.003
3. **Most stable feature importances** — Mean CoV=0.082 across 5 folds
4. **Sharpest residual distribution** — σ=25.49, mean≈0, narrowest violin
5. **Best CDF** — 50% of predictions within 25 kWh; 90% within ~55 kWh
6. **Native categorical support** — handles `DwellingTypeDescr` (11 classes), `VentilationMethod` etc. without encoding loss
7. **Loss curve never diverges** — RMSE gap stabilises at 6.2 kWh and remains flat for 450 rounds

### Why Random Forest is a Strong Second

1. R²=0.920 — only 3.9 percentage points behind LightGBM
2. OOB R²=0.9212 ≈ Test R²=0.9198 — the closest thing to a free, unbiased generalisation certificate
3. More interpretable than LightGBM — individual trees can be inspected
4. Parallelisable and robust to outliers through bagging
5. Would be preferred in contexts requiring transparency or uncertainty estimates

### Why Ridge is Baseline Only

1. R²=0.836 — 16% of BerRating variance is structurally unexplained
2. Heteroscedastic residuals prove model misspecification — errors grow with predicted value
3. Alpha tuning across 7 orders of magnitude has zero effect — regularisation is not the bottleneck
4. Learning curve flat from 3,000 samples — adding more data cannot help a linear model on a non-linear problem
5. Use Ridge for: coefficient interpretation, audit/compliance contexts, or as a sanity check

### Final Recommendation

> **Use LightGBM for production BER prediction.**
> Use Random Forest if interpretability or model transparency is required.
> Use Ridge only as a diagnostic baseline or in regulatory contexts demanding linear models.

---

## 12. Repository Structure

```
irish-home-retrofit-prediction/
│
├── Data & Notebooks
│   ├── 46_Col_final_with_county.parquet     # Cleaned dataset (1.35M rows × 46 cols)
│   ├── ber_regression_model_comparison.ipynb # Main model comparison notebook
│   ├── pca_lda_analysis.ipynb               # Dimensionality reduction notebook
│   ├── optimized_data_cleaning.ipynb        # Data cleaning pipeline
│   ├── row_cleaning.ipynb                   # Row-level cleaning
│   └── cleaning_data_reduced_to_76.ipynb    # Phase 1 column reduction
│
├── Documentation
│   ├── README.md                            # This file
│   ├── data_cleaning.md                     # Full data cleaning report
│   ├── imputation_analysis_report.md        # Missing value & imputation report
│   ├── pca_lda_report_v2.md                 # Dimensionality reduction report
│   └── row_cleaning_balaji_14_mar.md        # Initial cleaning notes
│
├── Summary Outputs
│   ├── data_summary.txt                     # Final dataset validation
│   ├── pca_summary_output.txt               # PCA feature importance report
│   ├── multicollinearity_summary.txt        # VIF audit results
│   ├── features_pca_90pct.txt               # Features at 90% PCA variance
│   └── features_pca_95pct.txt               # Features at 95% PCA variance
│
├── Model Comparison Plots
│   ├── plot1_master_scorecard.png
│   ├── plot2_generalisation_gap.png
│   ├── plot3_cv_stability.png
│   ├── plot4_learning_curves.png
│   ├── plot5_lgbm_loss_curve.png
│   ├── plot6_actual_vs_predicted.png
│   ├── plot7_residual_analysis.png
│   ├── plot8_feature_importance_stability.png
│   ├── plot9_oob_score.png
│   ├── plot10_radar_chart.png
│   ├── plot11_error_distribution.png
│   └── plot12_verdict_scorecard.png
│
└── EDA / Dimensionality Reduction Plots
    ├── pca_scree.png / pca_scatter.png / pca_heatmap.png / pca_biplot.png
    ├── lda_scatter.png / lda_variance.png
    ├── mca_scree.png / mca_scatter.png
    ├── famd_scatter.png / famd_variance.png
    ├── vif_correlation_heatmap.png
    ├── cramers_v_heatmap.png
    └── eta_squared_heatmap.png
```

---

## References

- Usman Ali et al. (2024): Refined subset selection for building energy characterisation
- Tripathi & Kumar (2024): Mitigating data leakage in energy performance modelling
- Zhang et al. (2023): Preventing overfitting in physical driver discovery
- Benavente-Peces (2020): Deep learning approaches for energy efficiency in buildings
- SEAI BER Public Dataset: [www.seai.ie](https://www.seai.ie/technologies/seai-research/data-and-statistics/ber-research-tool/)
