missing values in the dataset:
PredominantRoofType          141260
NoOfSidesSheltered            24127
PercentageDraughtStripped     24127
SuspendedWoodenFloor          24127
StructureType                 24127
VentilationMethod             24127
NoOfFansAndVents              24127
HSSupplHeatFraction           23467
WHRenewableResources          23467
SupplWHFuel                   23467
SupplSHFuel                   23467
WHEffAdjFactor                23467
WHMainSystemEff               23467
HSSupplSystemEff              23467
SHRenewableResources          23467
HSMainSystemEfficiency        23467
HSEffAdjFactor                23467




Processing in chunks to check overlap for 2 groups...

==================================================
NULL ROW OVERLAP ANALYSIS
==================================================

Group Shared Count: 23467
Columns: ['HSMainSystemEfficiency', 'HSEffAdjFactor', 'HSSupplHeatFraction', 'HSSupplSystemEff', 'WHMainSystemEff', 'WHEffAdjFactor', 'SupplSHFuel', 'SupplWHFuel', 'SHRenewableResources', 'WHRenewableResources']
✅ SUCCESS: These columns are missing data in the EXACT SAME 23467 rows.
   (Conclusion: These columns belong to a specific survey block that was skipped.)

Group Shared Count: 24127
Columns: ['NoOfFansAndVents', 'VentilationMethod', 'StructureType', 'SuspendedWoodenFloor', 'PercentageDraughtStripped', 'NoOfSidesSheltered']
✅ SUCCESS: These columns are missing data in the EXACT SAME 24127 rows.
   (Conclusion: These columns belong to a specific survey block that was skipped.)

==================================================

Based on the statistical results and plots you generated, here is the breakdown of which method is better for your BER model, grounded in your lecture materials.

### 1. The Verdict: Mode Imputation is Currently Better

Surprisingly, your **Mode Imputation** results are superior for prediction:

* 
**Lower Error**: Mode MSE (**3011.69**) is lower than Regression MSE (**3338.27**).


* 
**Higher Accuracy**: Mode $R^{2}$ (**0.8760**) is higher than Regression $R^{2}$ (**0.8625**).


* 
**Statistical Significance**: The **Kolmogorov-Smirnov** p-value of **0.0000** confirms that the Regression method significantly distorted the data distribution compared to the Mode method.



### 2. Analysis of Residuals & "Goodness of Fit"

Your residual plots reveal critical insights about your data quality:

* 
**Heteroscedasticity**: Both plots show a "fan" or "cone" shape where error variance increases with the predicted value. This indicates that your model might be struggling with higher energy buildings or outliers.


* 
**Non-Normality**: Your **Q-Q Plot** shows "heavy tails," where the blue dots curve away from the red line at both ends. This means your residuals are not normally distributed, likely due to the "suspicious data" or outliers mentioned in your lectures.


* **Outliers**: You have extreme residuals (e.g., above 8000). According to the notes, these could be "model glitches" or "data miner's dreams," but they currently bias your "Goodness of Fit".



### 3. Handling the "PredominantRoofType" Problem

With **141,260 missing values**, this column is a prime candidate for **Data Reduction**.

* 
**The Risk**: Keeping a variable with this much missingness can lead to the "Curse of Dimensionality," where the data becomes too sparse for the model to find meaningful patterns.


* 
**Action**: Unless this column shows extremely high **Information Gain** (using Entropy), your lecture notes suggest it is safer to **drop it** to "help eliminate irrelevant features and reduce noise".







