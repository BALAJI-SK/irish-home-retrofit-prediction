# Row Imputation & Data Cleaning Analysis Report

This document serves as a comprehensive record of the data cleaning methodologies, imputation tests, statistical evaluations, and the rationale behind the final design choices implemented in the Irish Home Retrofit Prediction dataset.

---

## 1. Initial Missing Data Analysis
Upon parsing the dataset (1,351,582 rows), significant chunks of missing values (`NULL`s) were discovered. A precise "Null Row Overlap Analysis" was conducted, indicating that blocks of columns were missing simultaneously on the exact same rows. This led us to classify the missing data into two primary mechanisms:

1. **NMAR (Not Missing At Random)** - Systems Data
2. **MAR (Missing At Random)** - Building Envelope Data

---

## 2. Strategies Implemented by Data Block

### A. The Systems Block (NMAR)
Columns such as `HSSupplHeatFraction`, `SHRenewableResources`, `WHRenewableResources`, `HSSupplSystemEff`, `WHEffAdjFactor`, `HSEffAdjFactor`, `SupplSHFuel`, and `SupplWHFuel` exhibited total overlap in their missingness (23,467 rows). 

* **Conclusion:** These survey fields were skipped because the house simply *did not have* a supplemental system or renewable resource. 
* **Action:** **Logical Zero-Fill**. Missing values represent an absence of the physical feature, so filling with `0` is technically and physically accurate.

### B. The Efficiency Block
Physical drivers that fundamentally *must* exist for a dwelling (e.g., `HSMainSystemEfficiency`, `WHMainSystemEff`). A house cannot have 0% main heating efficiency.
* **Action:** **Global Median Fill**. Chosen over chunk-level medians to prevent statistical skewing across randomized row groups. The median is remarkably robust against extreme outliers compared to the mean.

### C. The Building Envelope Block (MAR)
Columns like `StructureType`, `VentilationMethod`, `SuspendedWoodenFloor`, `PercentageDraughtStripped`, `NoOfSidesSheltered`, and `NoOfFansAndVents` missing in 24,127 rows.

* **Action:** **Iterative Modeling (Regression & Classification Imputation)**. We utilized complete cases of `Year_of_Construction` and `DwellingTypeDescr` to predict these envelope features dynamically:
  - Categorical targets (`StructureType`, etc.) were predicted using **Logistic Regression**.
  - Continuous/count targets (`NoOfFansAndVents`, etc.) were predicted using **Linear Regression**.

> **Note on Feature Dropping:** `PredominantRoofType` was missing in over 141,260 rows. It was permanently dropped because:
> 1. Imputing it introduces a massive synthetic bias burden.
> 2. It shares high redundancy with core physical drivers already present in the dataset (`UValueRoof` and `RoofArea`).

---

## 3. Dealing with Outliers & Noise

Before relying on Regression Imputations to generate synthetic data, we had to ensure our models were not learning from skewed, misrecorded survey data.

* **Methodology:** We applied the **Interquartile Range (IQR) technique** to isolate and suppress outliers solely for the regression targets. Values falling outside of the $[Q1 - 1.5 \times IQR,\; Q3 + 1.5 \times IQR]$ range were stripped out of the training subset.
* **Why?** This prevents extreme values (e.g., a physically impossible number of fans/vents) from dragging the OLS (Ordinary Least Squares) regression line and destroying imputation accuracy.

---

## 4. Tests Conducted: Mode Imputation vs. Regression Imputation

We evaluated whether simply filling the Building Envelope block with Contextual Mode values (Grouped by Year, Type, and County) was better or worse than Machine Learning-based Regression Imputation. 

### Test 1: Goodness-of-Fit (Baseline Random Forest)
A baseline `RandomForestRegressor` was trained to predict the `BerRating` target using both datasets to compare the Mean Squared Error (MSE) and R-squared ($R^2$).

* **Post-Outlier Removal Results:**
  - **Regression Imputation:** MSE = 1286.99 | $R^2$ = 0.9373
  - **Mode Imputation:** MSE = 1286.47 | $R^2$ = 0.9362

Both methods yielded exceptional and virtually identical baseline predictive accuracy once severe outliers were managed.

### Test 2: Kolmogorov-Smirnov (K-S) Distribution Test
We applied the K-S test to observe how radically the imputation altered the dataset's natural distribution.
* **Result:** `K-S Statistic: 0.0174 | p-value: 0.000`
* **Interpretation:** The two methods warp the distribution in statistically significant, fundamentally different ways.

---

## 5. Final Conclusion and Selection

**Chosen Pipeline:** We selected the **Regression Imputation** coupled with **IQR Outlier Suppression** and **Logical Zero-Fills**.

**Why?** 
1. **Physical Reality vs. Statistical Clumping:** While Mode Imputation achieved slightly better initial numerical scores pre-cleaning, it forcefully shoehorns thousands of rows into a singular "most common" value. This destroys variance and artificially compresses the multidimensional feature space.
2. **Preservation of Variance:** Regression Imputation generates a synthetic value that respects the continuous trend lines tied to when the house was built (`Year_of_Construction`) and what kind of house it is (`DwellingTypeDescr`). It respects the natural variance of the population far better than Mode fill.
3. **Model Resilience:** By proactively killing outliers (IQR) internally before predicting missing features, the Regression model achieves high statistical safety, matching the $R^2$ of the Mode approach while generating structurally superior, context-aware data.
