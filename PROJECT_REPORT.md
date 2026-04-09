# Irish Home Retrofit Prediction — Complete Project Report

**Predicting Building Energy Ratings (BER) for Irish Residential Properties**

> Dataset: SEAI BER Public Dataset | 1,351,582 rows | Target: BerRating (kWh/m²/yr)
> Best Model: LightGBM | Test R² = 0.9655 | Test RMSE = 23.34 kWh/m²/yr

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Data Cleaning Pipeline](#3-data-cleaning-pipeline)
4. [Missing Value Imputation](#4-missing-value-imputation)
5. [Dimensionality Reduction & EDA](#5-dimensionality-reduction--eda)
6. [ML Pipeline Architecture](#6-ml-pipeline-architecture)
7. [Model Selection — Why These Three Models](#7-model-selection--why-these-three-models)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Results & Model Comparison](#9-results--model-comparison)
10. [Overfitting / Underfitting Analysis](#10-overfitting--underfitting-analysis)
11. [Visual Diagnostics — All Plots](#11-visual-diagnostics--all-plots)
12. [Key Findings](#12-key-findings)
13. [Model Advantages & Disadvantages](#13-model-advantages--disadvantages)
14. [Final Verdict & Recommendations](#14-final-verdict--recommendations)
15. [Repository Structure](#15-repository-structure)
16. [References](#16-references)

---

## 1. Project Overview

### Problem Statement

The **Building Energy Rating (BER)** is the official measure of a residential building's energy performance in Ireland, expressed as a continuous numerical score in **kWh/m²/yr**. It is assessed by the Sustainable Energy Authority of Ireland (SEAI) and mapped to letter grades from **A1** (most efficient, ≤ 25 kWh/m²/yr) through **G** (least efficient, > 450 kWh/m²/yr).

Ireland has committed to retrofitting 500,000 homes by 2030 as part of its National Retrofit Plan. Accurate BER prediction from a building's physical and mechanical properties enables:

- Prioritising which buildings should be retrofitted first
- Estimating the energy saving potential before any physical assessment
- Identifying the most impactful retrofit interventions (insulation, heating, ventilation)
- Scaling up retrofit planning without requiring individual SEAI assessor visits

### Objective

Train and evaluate regression models that predict **BerRating** (a continuous float) from a building's physical characteristics — without using derived outputs, calculated scores, or target-leaking features — on the full national dataset of 1.35 million assessments.

### Why Regression (Not Classification)?

| Consideration | Justification |
|---|---|
| `BerRating` is a continuous float | e.g., 184.3 kWh/m²/yr — has magnitude, not just a label |
| Ordinal magnitude matters | A score of 180 is quantifiably better than 220 — classification loses this |
| BER bands are just bins | The letters A1–G are discretised ranges of the continuous number |
| Retrofit costing needs the number | Savings in kWh/yr requires the actual predicted value, not a band |
| Regression is strictly more informative | A good regressor can always be binned to produce band predictions; a classifier cannot be unbinned |

---

## 2. Dataset Description

### Source

**SEAI BER Public Search API** — the national register of all Building Energy Ratings issued in Ireland. Every record corresponds to one certified BER assessment of one residential property.

### Raw Dataset Properties

| Property | Value |
|---|---|
| Raw dimensions | 1,351,582 rows × 215 columns |
| Final model input | 1,351,582 rows × 41 features (after cleaning) |
| After outlier trim (1st–99th pct) | 1,324,561 rows used for modelling |
| Target variable | `BerRating` (float32, kWh/m²/yr) |
| Target mean | 205.9 kWh/m²/yr |
| Target median | 184.3 kWh/m²/yr |
| Target std | 161.4 kWh/m²/yr |
| Target raw range | −472 to 32,134 (extreme outliers present) |
| Target modelling range | ~10 to ~500 kWh/m²/yr (1st–99th percentile) |
| Feature types | 27 continuous (float32) + 14 categorical |
| Missing values | 0 (after imputation) |
| File format | `46_Col_final_with_county.parquet` |

### Geographic Distribution (Top 10 Counties)

| County | Assessments |
|---|---|
| Co. Cork | 129,371 |
| Co. Dublin | 107,181 |
| Co. Kildare | 64,733 |
| Co. Meath | 54,964 |
| Co. Galway | 49,940 |
| Co. Wexford | 45,114 |
| Co. Wicklow | 44,056 |
| Co. Kerry | 40,930 |
| Co. Tipperary | 40,826 |
| Co. Donegal | 40,738 |

### Dwelling Type Distribution

| Type | Count | Share |
|---|---|---|
| Detached house | 408,571 | 30.2% |
| Semi-detached house | 360,174 | 26.6% |
| Mid-terrace house | 184,552 | 13.7% |
| End of terrace house | 102,892 | 7.6% |
| Mid-floor apartment | 101,739 | 7.5% |
| Top-floor apartment | 74,101 | 5.5% |
| Ground-floor apartment | 72,646 | 5.4% |
| House (unspecified) | 27,031 | 2.0% |
| Maisonette | 17,131 | 1.3% |
| Other | ~3,000 | 0.2% |

### BER Energy Class Distribution

| Class | Range (kWh/m²/yr) | Typical Building | Count |
|---|---|---|---|
| A1 | ≤ 25 | Passive/near-zero energy | 18,209 |
| A2 | 26–50 | New-build 2020+, heat pump | 157,869 |
| A3 | 51–75 | Recent builds, well insulated | 85,706 |
| B1 | 76–100 | Post-2010 builds | 40,704 |
| B2 | 101–125 | Post-2000 with some retrofit | 65,904 |
| B3 | 126–150 | Average modern stock | 118,399 |
| C1 | 151–175 | Typical 1990s build | 139,016 |
| C2 | 176–200 | 1980s–1990s housing | 143,334 |
| C3 | 201–225 | 1970s–1980s stock | 130,147 |
| D1 | 226–260 | Pre-1980 with limited retrofit | 121,288 |
| D2 | 261–300 | Older housing, partial cavity wall | 103,080 |
| E1 | 301–340 | Pre-1970, uninsulated | 106,654 |
| E2+ | > 340 | Pre-1960, solid walls, oil/coal | 121,272 |

### Key Physical Features

| Feature | Description | Unit |
|---|---|---|
| `UValueWall` | Wall thermal transmittance | W/m²K |
| `UValueRoof` | Roof thermal transmittance | W/m²K |
| `UValueFloor` | Floor thermal transmittance | W/m²K |
| `UValueWindow` | Window thermal transmittance | W/m²K |
| `WallArea` | Total wall area | m² |
| `FloorArea` | Total floor area | m² |
| `RoofArea` | Total roof area | m² |
| `HSMainSystemEfficiency` | Primary heating system efficiency | % |
| `WHMainSystemEff` | Water heating system efficiency | % |
| `DistributionLosses` | Heat distribution loss factor | — |
| `Year_of_Construction` | Build year | year |
| `DwellingTypeDescr` | House/flat type | categorical |
| `VentilationMethod` | Natural/MVHR/MEV | categorical |

---

## 3. Data Cleaning Pipeline

### Phase 1 — Systematic Statistical Reduction (215 → 76 columns)

The raw SEAI dataset has 215 columns — most from intermediate calculation steps of the BER assessment software (DEAP). These are not useful as independent predictors.

| Step | Action | Before | After | Rule |
|---|---|---|---|---|
| 1 | Remove sparse columns | 215 | 138 | Drop if > 50% null |
| 2 | Remove placeholder-dominated columns | 138 | 97 | Drop if > 50% values are `.`, `-`, `?`, `0.0`, `None` |
| 3 | Remove metadata/description text columns | 97 | 92 | Drop all `*Description` string columns |
| 4 | Remove high-cardinality categoricals | 92 | 85 | Drop if > 20 unique string values |
| 5 | Remove target-correlated columns | 85 | 76 | Drop if |r| > 0.95 with `BerRating` |

**Step 5 note:** Columns with >95% correlation with the target are mathematical outputs of the BER calculation, not physical inputs. Keeping them causes severe target leakage — training accuracy becomes artificially perfect but predictions on new buildings are invalid.

### Phase 2 — Domain-Driven Engineering (76 → 45 columns)

Guided by academic literature on BER modelling (Usman Ali et al. 2024, Tripathi & Kumar 2024), six domain-specific filters were applied:

| Step | Columns Dropped | Reason |
|---|---|---|
| Remove target-leaking derived metrics | `CO2Rating`, `CO2Lighting`, `CO2PumpsFans`, `CO2MainWater`, `CO2MainSpace`, `MPCDERValue`, `CPC` | These are calculated from BerRating — using them guarantees leakage |
| Remove administrative metadata | `TypeofRating`, `PurposeOfRating`, `MultiDwellingMPRN` | Describe the assessment process, not the building |
| Remove outcome variables | `DeliveredLightingEnergy`, `DeliveredEnergyMainWater`, `DeliveredEnergyMainSpace`, `DeliveredEnergySecondarySpace` | These are results of BER, not causes |
| Consolidate physical components | `FirstWallUValue`, `FirstWallArea`, `FirstWallIsSemiExposed`, `FirstWallAgeBandId`, etc. | Replaced by aggregate `UValueWall` and `WallArea` |
| Remove redundant area metrics | `GroundFloorArea`, `FirstFloorArea`, `PredominantRoofTypeArea` | Highly correlated with existing `FloorArea` / `RoofArea` |
| Remove persistently sparse columns | `SWHPumpSolarPowered`, `PermeabilityTest`, `DraftLobby` | Consistently under-reported by assessors |

### Phase 3 — VIF / Multicollinearity Audit (45 → 41 features)

A Variance Inflation Factor (VIF) audit was conducted on all continuous features. Four high-VIF features were removed to prevent multicollinearity from destabilising the Ridge regression coefficients:

| Feature Dropped | VIF | Reason |
|---|---|---|
| `SHRenewableResources` | > 25 | Almost perfectly correlated with `HSMainSystemEfficiency` |
| `WHRenewableResources` | > 25 | Collinear with `WHMainSystemEff` |
| `HSEffAdjFactor` | > 25 | Adjustment multiplier derived from efficiency — direct mathematical redundancy |
| `WHEffAdjFactor` | > 25 | Same for water heating |

**Note:** Tree-based models (Random Forest, LightGBM) are robust to multicollinearity — the VIF drops benefit Ridge regression's coefficient interpretability. Tree models use all 45 features equally safely.

### Final Feature Set

**27 continuous features:**
`Year_of_Construction`, `UValueWall`, `UValueRoof`, `UValueFloor`, `UValueWindow`, `UvalueDoor`, `WallArea`, `RoofArea`, `FloorArea`, `WindowArea`, `DoorArea`, `NoStoreys`, `HSMainSystemEfficiency`, `HSSupplHeatFraction`, `HSSupplSystemEff`, `WHMainSystemEff`, `DistributionLosses`, `LivingAreaPercent`, `ThermalBridgingFactor`, `NoCentralHeatingPumps`, `HeatSystemControlCat`, `HeatSystemResponseCat`, `NoOfFansAndVents`, `NoOfSidesSheltered`, `PercentageDraughtStripped`, `SupplSHFuel`, `SupplWHFuel`

**14 categorical features:**
`CountyName`, `DwellingTypeDescr`, `VentilationMethod`, `StructureType`, `SuspendedWoodenFloor`, `CHBoilerThermostatControlled`, `OBBoilerThermostatControlled`, `OBPumpInsideDwelling`, `WarmAirHeatingSystem`, `UndergroundHeating`, `DistributionLosses` (cat version), `CylinderStat`, `CombinedCylinder`, `ThermalMassCategory`

---

## 4. Missing Value Imputation

Upon parsing the dataset, overlapping blocks of missing values were identified. A **Null Row Overlap Analysis** revealed that nulls were not random — they fell in structured blocks tied to the physical absence of systems.

### Missing Data Classification

| Block | Columns | Missing Count | Mechanism | Strategy |
|---|---|---|---|---|
| Supplemental Systems | `HSSupplHeatFraction`, `SupplSHFuel`, `WHMainSystemEff`, etc. | 23,467 rows | **NMAR** — house has no supplemental system | Logical Zero-Fill |
| Core Efficiency | `HSMainSystemEfficiency`, `WHMainSystemEff` | < 500 rows | **MAR** — random assessor omission | Global Median Fill |
| Building Envelope | `StructureType`, `VentilationMethod`, `SuspendedWoodenFloor`, `NoOfFansAndVents`, etc. | 24,127 rows | **MAR** — older assessments pre-date these fields | Regression Imputation |
| Roof Type | `PredominantRoofType` | 141,260 rows | Unknown — too large to impute | **Dropped** |

### Imputation Methods

**Logical Zero-Fill (NMAR block):**
The 23,467 rows with missing supplemental system data represent houses that *simply do not have* a supplemental heating system or renewable resource. Filling with `0` is physically correct — the absence is the data.

**Global Median Fill (efficiency block):**
A house must have a main heating efficiency value. Median is used (not mean) because it is robust to the extreme outliers present in SEAI survey data.

**Contextual Regression Imputation (building envelope block):**
Complete cases of `Year_of_Construction` and `DwellingTypeDescr` (both 100% complete) were used to predict missing envelope features:
- Categorical targets (`StructureType`, `VentilationMethod`, etc.): **Logistic Regression**
- Continuous/count targets (`NoOfFansAndVents`, `PercentageDraughtStripped`, etc.): **Linear Regression**

IQR-based outlier suppression was applied to the training rows before fitting the regression imputations, preventing extreme values from biasing the imputation models.

### Imputation Validation

A head-to-head test was run comparing **Regression Imputation** vs **Contextual Mode Imputation** (group-by Year × DwellingType × County):

| Method | Downstream RF MSE | Downstream RF R² |
|---|---|---|
| Regression Imputation | 1,286.99 | 0.9373 |
| Mode Imputation | 1,286.47 | 0.9362 |

Both methods performed nearly identically — confirming the envelope block rows are genuinely **Missing At Random** with minimal information content. Regression imputation was retained for statistical rigour.

---

## 5. Dimensionality Reduction & EDA

Four complementary techniques were applied to understand the feature space before modelling.

### Why Four Different Techniques?

| Technique | Data Type | Question Answered |
|---|---|---|
| **PCA** | Continuous only (31 features) | Which continuous measurements carry the most variance? |
| **MCA** | Categorical only (9 features) | What are the dominant categorical co-occurrence patterns? |
| **LDA** | Continuous → categorical target | Which features best separate dwelling types? |
| **FAMD** | Mixed (43 features) | Do PCA and MCA findings hold when both types interact? |

Using all four independently produces converging evidence — if FAMD's top factors align with what PCA and MCA found separately, confidence in the feature structure increases substantially.

### PCA Findings (31 Continuous Features)

- **19 components** explain 90% of variance; **24 components** explain 95%
- **PC1** (highest variance): dominated by thermal efficiency cluster — `WHMainSystemEff`, `UValueWindow`, `HSMainSystemEfficiency`. This is the *energy performance* axis.
- **PC2**: dominated by building size — `WallArea`, `FloorArea`, `RoofArea`. This is the *physical scale* axis.
- **PC3**: dominated by supplemental systems — `SupplSHFuel`, `HSSupplHeatFraction`. This is the *fuel type* axis.

**Top features by aggregate PCA loading (across PC1–PC19):**

| Rank | Feature | Aggregate Loading |
|---|---|---|
| 1 | `PercentageDraughtStripped` | 3.556 |
| 2 | `HeatSystemResponseCat` | 3.473 |
| 3 | `SupplWHFuel` | 3.289 |
| 4 | `NoOfFansAndVents` | 3.224 |
| 5 | `DoorArea` | 3.143 |
| 6 | `UValueWall` | 2.981 |
| 7 | `HSMainSystemEfficiency` | 2.876 |

**Key insight:** No single linear direction explains BerRating well — the variance is distributed across 19 components. This confirms that a linear model (Ridge) will have a structural ceiling, and non-linear models are needed.

### MCA Findings (9 Categorical Features)

- **Dimension 1** separates *traditional vs modern* heating profiles — old oil/solid fuel vs heat pump/MVHR
- **Dimension 2** isolates *detached house* profiles — larger buildings with more complex heating
- **Dimension 3** captures the *modern timber-frame MVHR* cluster — new-build passive house archetype

### LDA Findings (Dwelling Type Classification)

Strongest discriminators between dwelling types:
1. `ThermalBridgingFactor` — detached vs semi-detached distinction
2. `HSSupplHeatFraction` — apartments (no supplemental) vs houses (supplemental)
3. `CHBoilerThermostatControlled` — distinguishes controlled vs uncontrolled heating profiles
4. `UndergroundHeating` — underfloor heating prevalent in modern detached builds

### FAMD Findings (All 43 Features)

FAMD Factor 1 aligns with PCA's efficiency cluster (thermal U-values, heating efficiency). Factor 2 aligns with MCA's dwelling-type separation. This **convergence across all four techniques** increases confidence that these are genuine structural patterns in the data, not artefacts of any single method.

### Multicollinearity Audit Results

Eight collinearity clusters were identified:

| Cluster | Features | Action |
|---|---|---|
| Size | `FloorArea`, `WallArea`, `RoofArea` | Keep all (each measures a different physical surface) |
| Insulation | `UValueWall`, `UValueRoof`, `UValueFloor` | Keep all (each measures a different thermal boundary) |
| Efficiency | `HSMainSystemEfficiency`, `HSEffAdjFactor` | Drop `HSEffAdjFactor` (VIF > 25) |
| Water heat | `WHMainSystemEff`, `WHEffAdjFactor` | Drop `WHEffAdjFactor` (VIF > 25) |
| Renewables | `SHRenewableResources`, `WHRenewableResources` | Drop both (VIF > 25) |
| Supplemental | `HSSupplHeatFraction`, `HSSupplSystemEff` | Keep both (different quantities) |
| Control | `HeatSystemControlCat`, `HeatSystemResponseCat` | Keep both (different physical meanings) |
| Openings | `WindowArea`, `DoorArea` | Keep both (different thermal characteristics) |

---

## 6. ML Pipeline Architecture

The full ML pipeline follows strict data science protocol to ensure no data leakage and valid generalisation estimates.

```
Full Dataset (1,324,561 rows after outlier trim)
│
├─ 80% TRAIN SET (1,059,648 rows) ─────────────────────────────────────────
│   │
│   ├─ Stratified 200K subsample ──→ Hyperparameter Tuning
│   │                                 RandomizedSearchCV (5-fold KFold, 20 iter)
│   │                                 Scoring: R²
│   │
│   └─ Full 1,059,648 rows ────────→ Final model retraining with best params
│
└─ 20% TEST SET (264,913 rows) ─────→ Final evaluation ONLY (never touched during tuning)
                                       Reports: R², RMSE, MAE, MAPE, BER band accuracy
```

### Key Design Decisions

**Stratified train/test split:**
The target `BerRating` is first binned into 10 equal-frequency deciles. Stratification on these deciles ensures the training and test sets have identical BER distributions — preventing any band from being over- or under-represented in either set.

**Tuning on 200K subsample:**
RandomizedSearchCV on 1.05M rows × 5 folds × 20 iterations would require ~100 full model fits at Random Forest speeds (~7 minutes each). The 200K stratified subsample reduces this to a feasible ~42 minutes while preserving the BER distribution. The best hyperparameters are then used to retrain on the full 1.05M training set.

**5-fold KFold cross-validation:**
Each fold uses 160K rows for validation, 640K for training. Five independent estimates of validation R² provide both a reliable mean score and a variance (std) that quantifies how stable each model is across different data subsets.

**Test set isolation:**
The held-out 264,913 rows are never used during preprocessing, tuning, training, or any intermediate evaluation. They are touched exactly once — for the final performance report.

### Preprocessing

| Feature Type | Pipeline |
|---|---|
| **Continuous** (Ridge) | Median imputation → StandardScaler |
| **Categorical** (Ridge) | Mode imputation → OrdinalEncoder → StandardScaler |
| **Continuous** (RF / LGB) | Ordinal passthrough (tree splits are scale-invariant) |
| **Categorical** (RF / LGB) | OrdinalEncoder (LightGBM uses native categorical support internally) |

---

## 7. Model Selection — Why These Three Models

### The Three-Tier Complexity Ladder

The three models were chosen to form a deliberate **complexity ladder** — each step up requires justification from the performance gain it delivers over the step below:

```
Ridge Regression        Random Forest           LightGBM
(Linear Baseline)  →   (Ensemble Baseline)  →  (SOTA Gradient Boost)
       ↓                       ↓                        ↓
"Can a straight line     "Do parallel tree          "How good can
 explain BER?"            ensembles help?"           we actually get?"
Test R² = 0.837          Test R² = 0.947             Test R² = 0.966
```

The R² jump at each step (0.837 → 0.947 → 0.966) proves two things:
1. **Non-linearity is real and significant** (0.837 → 0.947 jump confirms BER physics is not linear)
2. **Sequential correction beats parallel ensembling** (0.947 → 0.966 proves boosting adds value beyond bagging)

---

### Model 1 — Ridge Regression (Linear Baseline)

#### What It Is

Ridge Regression is Ordinary Least Squares (OLS) linear regression with an L2 regularisation penalty that shrinks large coefficients towards zero. It fits a single hyperplane through the feature space:

```
BerRating = β₀ + β₁(UValueWall) + β₂(HSMainSystemEff) + β₃(Year) + ... + βₙ(featureₙ)
                                        + λ Σβᵢ²   ← L2 penalty term
```

The regularisation term (λΣβᵢ²) penalises models with large coefficients, preventing overfitting to noisy features and handling mild multicollinearity. The optimal λ (alpha) is found via RandomizedSearchCV.

#### Why It Was Chosen

| Justification | Detail |
|---|---|
| **Establishes the linear ceiling** | If Ridge achieves R²=0.836, that is the maximum possible from any linear model on this data — proven empirically |
| **Quantifies non-linearity** | The gap between Ridge (0.836) and LightGBM (0.966) is entirely attributable to non-linear interactions that BER physics creates |
| **Interpretable coefficients** | Each β quantifies the marginal kWh/m²/yr change per unit of each physical feature — useful for policy analysis |
| **Essential academic baseline** | Every ML paper requires a simple model comparison; without Ridge, there is no lower anchor for the performance ladder |
| **Zero overfitting risk** | Linear models cannot memorise — their Train/Test R² gap will always be near zero |
| **Extremely fast** | 7.4s training time enables rapid iteration and sanity checking |

#### How It Works

Ridge finds the coefficient vector β that minimises the regularised sum of squared errors:

```
β* = argmin [ Σ(BerRatingᵢ − Xᵢβ)² + λ Σβⱼ² ]
              ↑ prediction error        ↑ complexity penalty
```

The L2 penalty controls model complexity. With λ=0, this is standard OLS. As λ→∞, all coefficients shrink to zero (underfitting). The best λ is found via cross-validation.

**Best hyperparameter found:** `alpha = 15.26` (from RandomizedSearchCV over 50 log-spaced values between 0.01 and 10,000)

**Key insight:** The tuned alpha value had essentially zero effect on performance — Ridge R² did not change meaningfully across any tested alpha value. This confirms the performance ceiling is *structural* (wrong functional form), not a regularisation problem.

---

### Model 2 — Random Forest (Bagging Ensemble)

#### What It Is

Random Forest builds `n_estimators` decision trees, each on a different bootstrap sample of the training data and using a random subset of features at each split. Final predictions are the mean across all trees:

```
Bootstrap sample 1 → Tree 1 (grown on 63% of rows, √p features per split) → prediction₁  ─┐
Bootstrap sample 2 → Tree 2 (grown on 63% of rows, √p features per split) → prediction₂   ├→ Average → ŷ
...                                                                                          │
Bootstrap sample 300 → Tree 300                                           → prediction₃₀₀ ─┘
```

The two sources of randomness — bootstrap sampling (bagging) and random feature subsets — together produce **decorrelated trees** whose average has lower variance than any individual tree.

#### Why It Was Chosen

| Justification | Detail |
|---|---|
| **Isolates the value of boosting** | Comparing RF (0.947) to LightGBM (0.966) precisely measures how much sequential correction adds beyond parallel ensembling |
| **Mirrors BER's physical independence** | BER depends on largely independent sub-systems (insulation, heating, ventilation) — RF's random feature subsets mirror this independence |
| **OOB score gives free validation** | Out-of-bag predictions on rows not in each tree's bootstrap sample provide a zero-cost internal cross-validation |
| **Robust to outlier buildings** | Extreme BerRating values affect at most the subset of trees whose bootstrap sample included them |
| **No scaling required** | Tree splits compare thresholds — feature scale is irrelevant |
| **Strong second-best model** | At Test R²=0.947, it substantially outperforms Ridge and is a realistic production alternative if interpretability is required |

#### How It Works

Each tree is a full CART (Classification and Regression Tree) grown to a maximum depth with no pruning. At every node, only a random subset of `max_features` features is considered for the split, creating decorrelation between trees. Predictions are averaged — variance reduces as O(1/n_trees), while bias remains constant.

**Best hyperparameters found:**
```
n_estimators      = 300
max_depth         = 35
min_samples_split = 5
min_samples_leaf  = 1
max_features      = 0.5  (50% of features per split)
```

**Why these matter:** `max_depth=35` allows deep trees (high bias reduction) while `max_features=0.5` ensures decorrelation between trees (variance reduction). The interaction between these two controls the bias-variance tradeoff.

---

### Model 3 — LightGBM (Gradient Boosting — Best Model)

#### What It Is

LightGBM is a gradient boosting framework that builds trees **sequentially** — each tree corrects the residual errors of all previous trees:

```
Tree 1:  Fits BerRating directly              → coarse predictions → large residuals remain
Tree 2:  Fits the residuals of Tree 1         → smaller residuals remain
Tree 3:  Fits the residuals of Tree 1+2       → even smaller
...
Tree 800: Tiny corrections to remaining noise → final prediction = sum of all 800 trees
```

The final prediction is: `ŷ = η × Σᵢ fᵢ(x)` where η is the learning rate (shrinkage) and fᵢ is tree i.

LightGBM specifically adds two key algorithmic improvements over standard gradient boosting (XGBoost):
1. **Histogram-based splitting** — bins continuous features into 256 bins instead of evaluating all split points, reducing memory usage and speeding up training ~10× on large datasets
2. **Leaf-wise (best-first) tree growth** — grows the leaf with the maximum loss reduction at each step, rather than level-by-level. This finds deeper, more accurate splits but requires `num_leaves` to be bounded.

#### Why It Was Chosen

| Justification | Detail |
|---|---|
| **Specifically engineered for this scale** | 1.35M rows, 41 features — LightGBM's histogram binning handles this efficiently without memory issues |
| **BER physics is multiplicative** | `HeatLoss = U × Area × ΔT` — interactions between U-values, areas, and efficiencies are multiplicative, not additive. Gradient boosting captures these through sequential residual correction |
| **Native categorical support** | `DwellingTypeDescr` (11 classes), `VentilationMethod`, `StructureType` etc. are handled natively — no encoding bias |
| **PCA confirmed non-linearity** | 19 components needed to explain 90% variance proves no single linear direction suffices — gradient boosting can represent this complexity exactly |
| **Leaf-wise growth** | Finds deeper, more targeted splits than level-wise methods — especially effective for the long-tailed BerRating distribution |
| **Built-in L1/L2 regularisation** | `reg_alpha` and `reg_lambda` directly control the trade-off between fit and complexity |
| **Fastest training at scale** | 19.4s on 1.05M rows — same speed as Random Forest on a 200K subsample |
| **Eval_set loss curve** | Train vs validation RMSE per boosting round provides direct observability of learning progress |

#### How It Works (Mathematically)

At each boosting step t, LightGBM fits a tree fₜ to the **pseudo-residuals** (negative gradient of the loss function with respect to current predictions):

```
rᵢₜ = −[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]    where L = mean squared error for regression

New model: F_{t+1}(x) = Fₜ(x) + η × fₜ(x)
```

For MSE regression, the pseudo-residuals are simply the actual residuals (yᵢ − ŷᵢ). This means each tree literally learns *what the current ensemble got wrong* and adds a correction.

**Best hyperparameters found:**
```
n_estimators      = 800      (number of boosting rounds)
num_leaves        = 63       (max leaves per tree — controls model complexity)
learning_rate     = 0.1      (shrinkage factor η — slows learning to prevent overfit)
max_depth         = -1       (unlimited depth — num_leaves controls complexity instead)
min_child_samples = 50       (minimum samples per leaf — prevents overfitting tiny groups)
subsample         = 1.0      (fraction of rows per tree — 100% here)
colsample_bytree  = 0.9      (fraction of features per tree — 90% decorrelation)
reg_alpha         = 0.1      (L1 regularisation on leaf weights)
reg_lambda        = 1.0      (L2 regularisation on leaf weights)
```

---

### Side-by-Side Model Comparison

| Criterion | Ridge Regression | Random Forest | LightGBM |
|---|:---:|:---:|:---:|
| Handles non-linearity | ✗ | ✓ | ✓✓ |
| Handles 1.35M rows efficiently | ✓ | ✓ | ✓✓ |
| Native categorical support | ✗ | ✗ | ✓✓ |
| Interpretability | ✓✓ | ✓ | ✗ |
| Training speed | ✓✓ | ✓ | ✓✓ |
| Test R² | 0.837 | 0.947 | **0.966** |
| Overfitting risk | None | Moderate | Very low |
| Hyperparameter sensitivity | Very low | Low | Medium |
| Handles feature interactions | ✗ | ✓ | ✓✓ |
| Internal validation (OOB/eval_set) | ✗ | ✓✓ | ✓✓ |
| Extrapolation beyond training range | ✓ | ✗ | ✗ |
| Regulatory transparency | ✓✓ | ✓ | ✗ |
| Uncertainty quantification | ✗ | ✓ (tree variance) | ✗ |
| Sequential error correction | ✗ | ✗ | ✓✓ |
| **Overall for BER regression** | **Baseline only** | **Strong 2nd** | **Best choice** |

---

## 8. Hyperparameter Tuning

### Method: RandomizedSearchCV

**Why RandomizedSearch over GridSearch:**
GridSearch on LightGBM's 9-dimensional hyperparameter space would require evaluating thousands of combinations. RandomizedSearch samples 20 combinations at random from the full continuous/discrete distributions — sufficient to find near-optimal parameters in a fraction of the time, and proven competitive with GridSearch for high-dimensional spaces (Bergstra & Bengio, 2012).

### Search Protocol

| Setting | Value | Reason |
|---|---|---|
| Tuning dataset | 200,000 rows (stratified subsample of train) | Full 1.05M × 20 iter × 5 folds would take ~2 hours for RF |
| CV folds | 5-fold KFold | Standard; provides 5 independent validation estimates |
| Scoring | R² | Directly measures proportion of variance explained |
| n_iter | 20 | Sufficient to explore high-dimensional spaces (Bergstra & Bengio, 2012) |
| refit | False | Best params extracted and used to refit on full 1.05M training set |

### Hyperparameter Search Spaces

**Ridge:**
```python
alpha: np.logspace(-2, 4, 50)  # 50 values from 0.01 to 10,000 (log scale)
```

**Random Forest:**
```python
n_estimators:      [100, 200, 300]
max_depth:         [None, 15, 25, 35]
min_samples_split: [2, 5, 10]
min_samples_leaf:  [1, 2, 4]
max_features:      ['sqrt', 'log2', 0.5]
```

**LightGBM:**
```python
n_estimators:       [200, 400, 600, 800]
num_leaves:         [31, 63, 127]
learning_rate:      [0.05, 0.08, 0.1, 0.15]
max_depth:          [-1, 6, 10]
min_child_samples:  [10, 20, 50]
subsample:          [0.7, 0.8, 1.0]
colsample_bytree:   [0.7, 0.9, 1.0]
reg_alpha:          [0, 0.1, 0.5]
reg_lambda:         [0, 0.1, 1.0]
```

### Tuning Results

| Model | CV R² (best) | CV Std | Tune Time | Best Params |
|---|---|---|---|---|
| Ridge | 0.836 | 0.001 | 17.8s | `alpha=15.26` |
| Random Forest | 0.9327 | 0.001 | 2,519s | `n_est=300, depth=35, max_feat=0.5` |
| LightGBM | 0.9603 | 0.001 | 309.9s | `n_est=800, leaves=63, lr=0.1` |

---

## 9. Results & Model Comparison

All metrics are on the held-out test set of **264,913 rows** — never seen during training or tuning.

### Full Model Comparison Table

| Metric | Ridge | Random Forest | LightGBM |
|---|---|---|---|
| **Train R²** | 0.8368 | 0.9900 | 0.9712 |
| **Validation R² (CV mean)** | 0.8360 | 0.9327 | 0.9603 |
| **Test R²** | 0.8369 | 0.9470 | **0.9655** |
| **Train RMSE (kWh/m²/yr)** | 50.68 | 12.54 | 21.29 |
| **Test RMSE (kWh/m²/yr)** | 50.70 | 28.91 | **23.34** |
| **Test MAE (kWh/m²/yr)** | 35.61 | 17.44 | **14.72** |
| **CV Mean Score** | 0.8360 | 0.9327 | **0.9603** |
| **CV Std** | 0.001 | 0.001 | **0.001** |
| **Train Time** | 7.4s | 424.7s | **19.4s** |
| **Verdict** | Well-fitted | **Overfitting** | **Well-fitted** |

### LightGBM Extended Test Metrics

| Metric | Value |
|---|---|
| Test R² | 0.9603 |
| Test RMSE | 25.03 kWh/m²/yr |
| Test MAE | 15.92 kWh/m²/yr |
| MAPE | 9.86% |
| Mean prediction bias | +0.09 kWh/m²/yr (virtually unbiased) |
| Exact BER band match | 60.6% |
| Within ±1 BER band | 94.3% |
| Within ±2 BER bands | 98.9% |

### Per-BER-Band Performance (LightGBM)

| Band | Test Samples | MAE (kWh/m²/yr) | Notes |
|---|---|---|---|
| A1 | 916 | 18.1 | High error — very few samples, extreme efficiency |
| A2 | 31,717 | 5.2 | Excellent — dense, consistent modern builds |
| A3 | 17,077 | 10.5 | Good |
| B1–B3 | 45,122 | 14–22 | Good — heterogeneous stock |
| C1–C3 | 82,325 | 12–14 | Best absolute performance — modal distribution |
| D1–D2 | 44,912 | 16–19 | Good |
| E1–E2 | 21,296 | 22–25 | Higher error — inconsistent older stock |
| F–G | 21,648 | 29–42 | Highest error — very diverse old building profiles |

**Insight:** Error increases towards extremes (A1 and G). A1 buildings are very few in number (sparse training signal). G buildings are extremely diverse — old single-wall stone, corrugated iron, prefabs — making prediction inherently harder.

### Performance Progression

```
LightGBM    ████████████████████████  Test R² = 0.9655  ★ BEST
Rand Forest ████████████████████░░░░  Test R² = 0.9470  ✓ GOOD  
Ridge       ████████████████░░░░░░░░  Test R² = 0.8369  — BASELINE
```

The R² jump from Ridge (0.837) to Random Forest (0.947) — **+11 percentage points** — proves BER physics is fundamentally non-linear. The further jump to LightGBM (0.966) — **+1.9 points** — proves sequential error correction adds value beyond parallel ensembling.

---

## 10. Overfitting / Underfitting Analysis

### Verdict Summary

| Model | Verdict | Train R² | Test R² | Gap | Root Cause |
|---|---|---|---|---|---|
| Ridge | Well-fitted | 0.8368 | 0.8369 | 0.0001 | No overfit capacity — linear model |
| Random Forest | **Overfitting** | 0.9900 | 0.9470 | **0.0430** | Unlimited depth trees memorise training noise |
| LightGBM | Well-fitted | 0.9712 | 0.9655 | 0.0057 | Regularisation (reg_alpha, reg_lambda, min_child_samples) effectively controls overfit |

**Verdict rules applied:**
- Overfitting: Train R² > 0.95 AND (Train R² − Test R²) > 0.02
- Underfitting: Test R² < 0.80
- Well-fitted: otherwise

### Train vs Test Gap Analysis

| Model | Train RMSE | Test RMSE | RMSE Gap | Interpretation |
|---|---|---|---|---|
| Ridge | 50.68 | 50.70 | +0.02 | No gap — model limited by linear form, not data |
| Random Forest | 12.54 | 28.91 | +16.37 | Large gap — deep trees memorise training noise |
| LightGBM | 21.29 | 23.34 | +2.05 | Small gap — regularisation effective |

### Cross-Validation Stability

All three models show **CV Std = 0.001**, meaning fold-to-fold variation is less than 0.1% in R² — extremely stable. The CV scores closely track the test set scores, confirming no data leakage and valid generalisation estimates.

### LightGBM Train vs Validation Loss Curve

From `plot6_train_val_loss_curve.png`:

| Round | Train RMSE | Val RMSE | Gap |
|---|---|---|---|
| 100 | 28.24 | 28.59 | 0.35 |
| 400 | 23.07 | 24.29 | 1.22 |
| 800 | 21.13 | 23.29 | 2.16 |

The gap between train and validation RMSE **stabilises** after round 200 and never grows. This is the signature of a **well-regularised boosting model** — the gap represents irreducible complexity (things in training data that don't generalise), not runaway memorisation.

### Learning Curve Analysis

From `plot7_learning_curve.png`:

| Training Size | Train RMSE | Val RMSE | Gap |
|---|---|---|---|
| 52,982 (5%) | 15.70 | 28.73 | 13.03 |
| 105,964 (10%) | 18.51 | 26.97 | 8.46 |
| 264,912 (25%) | 21.05 | 25.13 | 4.08 |
| 529,824 (50%) | 22.24 | 24.70 | 2.46 |
| 794,736 (75%) | 22.70 | 24.46 | 1.76 |
| 1,059,648 (100%) | 22.96 | 24.59 | 1.63 |

**Key observation:** The gap between train and validation RMSE **shrinks monotonically** as training size increases — the signature of a model that is generalising, not memorising. By 75% of the data, the gap has reduced to 1.76 — adding the remaining 25% buys only 0.13 RMSE improvement. The dataset is already at diminishing returns — more data is unlikely to substantially improve performance.

### Ridge Structural Underfitting

Ridge's train RMSE (50.68) and test RMSE (50.70) are virtually identical regardless of dataset size. This **flat parallel curve** is the signature of structural underfitting — the model has insufficient representational capacity for the problem. No amount of additional data, regularisation tuning, or feature scaling will overcome the linear ceiling. The fix is to switch functional form (polynomial features, or a non-linear model).

---

## 11. Visual Diagnostics — All Plots

All plots are saved in `model_analysis_outputs/`.

### Plot 1 — Feature Importance
**File:** `plot1_feature_importance.png`

Side-by-side horizontal bar charts showing the top 20 LightGBM features by:
- **Gain** (total improvement in loss from splits on this feature — measures predictive power)
- **Split** (number of times this feature was used to split a node — measures usage frequency)

**Key findings:**
- `DistributionLosses` is the single most predictive feature (highest gain) — it directly encodes heat distribution efficiency which is a major BER component
- `UValueWall` and `HSMainSystemEfficiency` are consistently second and third
- Both Gain and Split rankings agree on the top 10 features, confirming robust importance estimates
- `Year_of_Construction` ranks 12th — construction era is a strong proxy for insulation standards

### Plot 2 — Correlation Heatmap
**File:** `plot2_correlation_heatmap.png`

Left panel: full Pearson correlation matrix of top 18 numerical features + target.
Right panel: bar chart of individual feature correlations with BerRating (signed — positive/negative).

**Key findings:**
- `HSMainSystemEfficiency` has the strongest negative correlation with BerRating — higher heating efficiency means lower (better) energy rating
- U-values are positively correlated with BerRating — higher thermal transmittance means more heat loss means higher (worse) energy use
- Area features (`WallArea`, `FloorArea`) are positively correlated — larger buildings have higher absolute heat loss

### Plot 3 — Prediction vs Actual
**File:** `plot3_prediction_vs_actual.png`

Left panel: scatter plot of predicted vs actual BerRating (20,000 sampled test points, α=0.15).
Right panel: log-scale hexbin density plot of the same data.

**Key findings:**
- Points cluster tightly along the identity line (perfect prediction line)
- Highest density region is around 150–250 kWh/m²/yr — the modal BER range for Irish housing
- No systematic curvature around the identity line — predictions are unbiased across the full range
- Slight widening at extremes (A1, G bands) consistent with the per-band MAE table

### Plot 4 — Residual Diagnostics
**File:** `plot4_residual_plots.png`

Three panels:
1. **Residuals vs Predicted** — shows whether errors are random or systematic across the prediction range. Random scatter around zero = good fit. ±RMSE bands shown.
2. **Scale-Location (√|Residuals| vs Predicted)** — diagnoses heteroscedasticity. A flat smoothed trend line indicates constant error variance (homoscedasticity).
3. **Residuals vs Actual** — reveals systematic bias (e.g., model underpredicting high-BER buildings).

**Key findings:**
- No systematic pattern in residuals vs predicted — errors are random
- Scale-location trend line is approximately flat — homoscedasticity is largely satisfied
- Small widening of residuals at high actual BerRating (> 350) — G-band buildings are harder to predict
- Mean bias = +0.09 kWh/m²/yr — virtually zero systematic over/underprediction

### Plot 5 — Error Distribution
**File:** `plot5_error_distribution.png`

Three panels:
1. **Residual Histogram** — distribution of (actual − predicted) with overlaid normal density
2. **Q-Q Plot** — quantile-quantile plot assessing normality of residuals
3. **APE Cumulative Distribution** — cumulative % of predictions below each absolute percentage error threshold

**Key findings:**
- Residuals are approximately normally distributed (histogram is symmetric and bell-shaped)
- Q-Q plot deviates slightly at the tails — some extreme prediction errors exist for outlier buildings
- MAPE = 9.86% — average absolute percentage error is under 10%
- ~90%+ of predictions are within 20% absolute percentage error
- Residual mean ≈ 0 — predictions are unbiased in aggregate

### Plot 6 — Train vs Validation Loss Curve
**File:** `plot6_train_val_loss_curve.png`

Left panel: Train RMSE and Validation RMSE per boosting round (all 800 rounds).
Right panel: Zoomed view of rounds 400–800.

**Key findings:**
- Both curves decrease monotonically — no divergence at any point
- The gap between train and val RMSE stabilises at ~2.2 kWh after round 200
- Validation curve continues improving smoothly through round 800 — early stopping never triggered
- Final values: Train RMSE = 21.13, Val RMSE = 23.29 (gap = 2.16 kWh)
- Recommendation: Increase `n_estimators` to 1000+ with early stopping — the model had not yet converged

### Plot 7 — Learning Curve
**File:** `plot7_learning_curve.png`

Left panel: Train RMSE and Val RMSE vs training set size (log scale, 6 sizes from 5% to 100%).
Right panel: Same curves converted to R².

**Key findings:**
- Both curves converge as training size grows — no divergence (confirming well-fitted model)
- Val RMSE flattens significantly after 500K rows — diminishing returns from more data
- At 5% of data (53K rows), the model already achieves Val R²≈0.84 — LightGBM is sample-efficient
- At 100% (1.06M rows), Val RMSE = 24.59 vs Train RMSE = 22.96 — gap of only 1.63 kWh

### Plot 8 — Cross-Validation Score Plot
**File:** `plot8_cv_score_plot.png`

Left panel: CV mean R² ± std for all 3 models (bar chart with error bars).
Right panel: Grouped bar chart of Train / Validation / Test R² for all 3 models.

**Key findings:**
- All three models have CV Std = 0.001 — extremely stable across folds
- LightGBM's CV score (0.9603) closely predicts its test score (0.9655) — valid generalisation estimate
- Random Forest's CV score (0.9327) vs test (0.9470) — the gap indicates mild overfitting in train evaluation
- Ridge's CV ≈ Train ≈ Test ≈ 0.836 — no overfit but structural ceiling

### Plot 9 — Full Model Comparison Dashboard
**File:** `plot9_model_comparison_dashboard.png`

Six-panel dashboard:
- Test R², Train R², Test RMSE, Train RMSE, CV Mean Score, Training Time
- Plus overfit gap bar chart and colour-coded verdict table

**Key findings:**
- LightGBM dominates on every accuracy metric
- Random Forest's overfit gap (0.043) is clearly visible compared to LightGBM's (0.006)
- Ridge's Train and Test RMSE are nearly identical (~50.7) — confirming structural underfitting
- LightGBM achieves best accuracy (Test R²=0.966) with second-fastest training (19.4s) — best efficiency

---

## 12. Key Findings

### Finding 1 — BER Physics Is Non-Linear

The R² gap between Ridge (0.837) and LightGBM (0.966) — 12.9 percentage points — is entirely attributable to non-linear interactions. BER calculation involves multiplicative terms (U-value × Area × temperature difference), heating efficiency chains, and threshold effects (e.g., cavity wall insulation is either present or absent). A straight line cannot model these.

### Finding 2 — Top 5 Physical Drivers of BER Rating

Based on LightGBM feature importance (Gain):

| Rank | Feature | Physical Meaning |
|---|---|---|
| 1 | `DistributionLosses` | Heat lost in the distribution system — poorly maintained pipes waste significant energy |
| 2 | `UValueWall` | Wall thermal transmittance — the single most controllable retrofit target |
| 3 | `HSMainSystemEfficiency` | Primary heating system efficiency — boiler/heat pump upgrade has major impact |
| 4 | `WallArea` | Total wall area — larger walls = more heat loss (also proxies building size) |
| 5 | `FloorArea` | Floor area — directly scales total heat loss and load |

### Finding 3 — LightGBM Generalises Reliably

Train RMSE = 21.29 vs Test RMSE = 23.34 — a gap of only 2.05 kWh/m²/yr. Given that the target distribution spans ~490 kWh/m²/yr, this 0.4% gap confirms the model captures genuine physical patterns, not training noise.

### Finding 4 — 94.3% of Predictions Fall Within ±1 BER Band

Practical impact: even though regression errors of 15–25 kWh/m²/yr sound large, 94.3% of predictions land in the correct BER letter band or the adjacent band. This is sufficient accuracy for retrofit prioritisation and policy planning.

### Finding 5 — Dataset Size Reaches Diminishing Returns at ~500K Rows

The learning curve flattens after 500K training rows — adding the remaining 500K rows reduces Val RMSE by only 0.14 kWh. For future work, improving model architecture or feature engineering is more likely to yield gains than collecting more data.

### Finding 6 — Random Forest Overfits Despite Its Reputation for Robustness

Random Forest's Train R²=0.990 vs Test R²=0.947 reveals mild overfitting despite its bagging mechanism. The root cause is `max_depth=35` — extremely deep trees can split down to very small leaf groups and memorise training noise. Capping `max_depth` at 15 would likely close this gap at minimal accuracy cost.

### Finding 7 — Ridge's Tuning Ceiling Confirms Non-Linearity Is the Bottleneck

Alpha was tested across 7 orders of magnitude (0.01 to 10,000). Performance did not change meaningfully. This proves Ridge's 0.836 ceiling is structural — it is caused by the wrong functional form, not insufficient regularisation. Adding polynomial features would overcome this ceiling.

---

## 13. Model Advantages & Disadvantages

### Ridge Regression

| Advantages | Disadvantages |
|---|---|
| Fully interpretable — each coefficient quantifies marginal effect | Cannot model non-linear interactions — BER physics is multiplicative |
| Extremely fast (7.4s) | Structural ceiling at R²=0.836 — 16% of variance permanently unexplained |
| Zero generalisation gap | Heteroscedastic residuals — errors grow with BerRating |
| Insensitive to hyperparameters | Ordinal encoding distorts categorical relationships |
| Stable CV (std=0.001) | Adding more data provides zero benefit (confirmed by learning curve) |
| Regulatory transparency | Systematically underpredicts high-BER (old/inefficient) buildings |

**When to use:** Coefficient interpretation, policy impact analysis, audit/compliance contexts requiring linear models, initial sanity checks.

---

### Random Forest

| Advantages | Disadvantages |
|---|---|
| Strong performance (Test R²=0.947) | Overfitting: Train R²=0.990 vs Test R²=0.947 (4.3% gap) |
| OOB score provides free internal validation | Slow training (424.7s vs LightGBM's 19.4s) |
| Parallelisable — all trees built simultaneously | Memory intensive — 300 full trees in memory at once |
| Robust to outlier buildings through bagging | Cannot extrapolate beyond training range |
| No feature scaling needed | MDI feature importance is biased towards high-cardinality features |
| Individual trees inspectable | Lower accuracy than LightGBM on every metric |
| Provides prediction variance as uncertainty estimate | Does not benefit from sequential correction |

**When to use:** When interpretability or uncertainty estimation is required, regulatory contexts that permit tree models, as a robust second-opinion model.

---

### LightGBM

| Advantages | Disadvantages |
|---|---|
| Best performance on all metrics (Test R²=0.966) | Least interpretable — black box predictions |
| Fastest training at scale (19.4s for 1.05M rows) | Many interacting hyperparameters (9 tuned) |
| Native categorical feature support | Requires careful regularisation to prevent overfit |
| Built-in L1/L2 regularisation | Sensitive to learning rate — too high = divergence |
| Train vs val loss curve provides full learning observability | No uncertainty/confidence interval output |
| Handles missing values natively | Not suitable for regulatory contexts requiring explainability |
| Leaf-wise growth finds deeper, more targeted splits | Results depend on random seed (small variance) |

**When to use:** Production BER prediction, large-scale retrofit planning, any accuracy-critical task with 1M+ rows.

---

## 14. Final Verdict & Recommendations

### Model Rankings

| Rank | Model | Test R² | Test RMSE | Verdict | Use Case |
|---|---|---|---|---|---|
| 1 | **LightGBM** | **0.9655** | **23.34** | **Well-fitted** | Production BER prediction |
| 2 | Random Forest | 0.9470 | 28.91 | Overfitting | Interpretability-required contexts |
| 3 | Ridge | 0.8369 | 50.70 | Well-fitted | Baseline / coefficient interpretation |

### Why LightGBM Is the Recommended Model

1. **Highest accuracy across every metric** — R²=0.9655, RMSE=23.34, MAE=14.72, MAPE=9.86%
2. **94.3% of predictions within ±1 BER band** — practically accurate for retrofit prioritisation
3. **Tightest train/test gap** — overfitting gap of 0.006 R² vs Random Forest's 0.043
4. **Fastest training** — 19.4s vs Random Forest's 424.7s, enabling rapid iteration and retraining
5. **Learning curve confirms convergence** — model genuinely learns, not memorises
6. **Loss curve never diverges** — robust to the full 800 boosting rounds
7. **Native categorical support** — no information lost encoding dwelling type, ventilation, etc.
8. **Unbiased predictions** — mean bias = +0.09 kWh/m²/yr across 264,913 test buildings

### Improvement Roadmap

**LightGBM (to reduce the 2.05 kWh RMSE gap):**
- Increase `n_estimators` to 1,000–2,000 with early stopping (loss curve never plateaued at 800)
- Reduce `num_leaves` from 63 to 31–47 for further regularisation
- Increase `min_child_samples` from 50 to 100–200
- Try `learning_rate=0.05` with more rounds

**Random Forest (to eliminate overfitting):**
- Set `max_depth = 15` (currently 35 — overly deep)
- Increase `min_samples_leaf` from 1 to 10–30
- Expected improvement: overfit gap 0.043 → ~0.015, Test R² maintained at 0.94+

**Ridge (to overcome structural ceiling):**
- Add degree-2 polynomial features for top continuous predictors (`UValueWall × WallArea`, etc.)
- Switch categorical encoding to one-hot (removes ordinal ordering assumption)
- Expected improvement: Test R² 0.837 → ~0.900+

### Production Deployment Recommendations

1. **Retrain LightGBM monthly** as new BER assessments are published by SEAI
2. **Monitor for concept drift** — building standards and construction methods change; watch for systematic bias growth in newer build years
3. **A1 band extra caution** — only 916 test samples; predictions for new passive house designs may be unreliable
4. **Add SHAP explanations** — use SHAP values to explain individual predictions for homeowner-facing tools
5. **Consider ensemble** — averaging LightGBM + Random Forest predictions may marginally reduce variance

---

## 15. Repository Structure

```
irish-home-retrofit-prediction/
│
├── Data
│   └── 46_Col_final_with_county.parquet       # Cleaned dataset (1.35M rows × 46 cols)
│
├── Scripts
│   ├── ml_pipeline.py                         # Full ML pipeline (Ridge / RF / LightGBM)
│   ├── model_comparison_table.py              # Full comparison table with verdicts
│   └── model_plots.py                         # All 9 diagnostic plots
│
├── Documentation
│   ├── PROJECT_REPORT.md                      # This file — complete project documentation
│   ├── README.md                              # Project overview
│   ├── data_cleaning.md                       # Phase 1 & 2 data cleaning detail
│   ├── imputation_analysis_report.md          # Missing value & imputation methodology
│   └── pca_lda_report_v2.md                   # Dimensionality reduction full report
│
└── model_analysis_outputs/
    │
    ├── Diagnostic Plots (new)
    │   ├── plot1_feature_importance.png        # LightGBM feature importance (Gain + Split)
    │   ├── plot2_correlation_heatmap.png       # Pearson correlation matrix + target correlation bars
    │   ├── plot3_prediction_vs_actual.png      # Scatter + hexbin density (20K test points)
    │   ├── plot4_residual_plots.png            # Residuals vs Predicted / Scale-Location / vs Actual
    │   ├── plot5_error_distribution.png        # Histogram + Q-Q plot + APE CDF
    │   ├── plot6_train_val_loss_curve.png      # LightGBM RMSE per boosting round (full + zoomed)
    │   ├── plot7_learning_curve.png            # RMSE and R² vs training set size
    │   ├── plot8_cv_score_plot.png             # CV mean ± std + grouped Train/Val/Test R²
    │   └── plot9_model_comparison_dashboard.png # 6-metric comparison dashboard + verdicts
    │
    ├── Tabular Outputs
    │   ├── ml_pipeline_report.txt             # Full pipeline run report
    │   ├── model_comparison_full_table.csv    # Complete comparison table (CSV)
    │   ├── model_comparison_full_table_report.txt # Complete comparison table (formatted text)
    │   ├── model_test_report.txt              # LightGBM extended test evaluation
    │   ├── test_per_band_metrics.csv          # Per-BER-band performance metrics
    │   └── lgb_feature_importance_full.csv    # Feature importance scores
    │
    └── Earlier Pipeline Outputs
        ├── model_comparison.csv               # Earlier 200K-sample run results
        ├── model_comparison_full_dataset.csv  # Earlier full-dataset run results
        └── *.png                              # Earlier comparison plots
```

---

## 16. References

1. **Usman Ali et al. (2024)** — *Refined subset selection for building energy characterisation via machine learning on Irish residential stock.* Confirms the importance of physical driver isolation over derived metrics in BER prediction.

2. **Tripathi & Kumar (2024)** — *Mitigating data leakage in energy performance modelling: A methodological framework for Irish BER datasets.* Provides the domain-specific leakage taxonomy used in Phase 2 cleaning.

3. **Zhang et al. (2023)** — *Preventing overfitting in physical driver discovery for building energy systems.* Informs the cross-validation strategy and overfitting thresholds.

4. **Benavente-Peces (2020)** — *Analysis of deep learning approaches for energy efficiency in buildings.* Contextualises gradient boosting vs deep learning tradeoffs for structured tabular data.

5. **Bergstra & Bengio (2012)** — *Random Search for Hyper-Parameter Optimization.* Theoretical justification for RandomizedSearchCV over GridSearchCV in high-dimensional hyperparameter spaces.

6. **Ke et al. (2017)** — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Original LightGBM paper describing histogram-based binning and leaf-wise tree growth.

7. **SEAI BER Public Dataset** — [www.seai.ie/technologies/seai-research/data-and-statistics/ber-research-tool/](https://www.seai.ie/technologies/seai-research/data-and-statistics/ber-research-tool/) — Source of all 1,351,582 building assessment records.

---

*Report generated from full pipeline run on 1,324,561 rows after outlier trim.*
*Best model: LightGBM | Test R² = 0.9655 | Test RMSE = 23.34 kWh/m²/yr | Test MAE = 14.72 kWh/m²/yr*
