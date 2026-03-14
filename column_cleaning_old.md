# Data Cleaning and Feature Engineering Documentation

This document outlines the systematic approach taken to clean and reduce the raw SEAI (Sustainable Energy Authority of Ireland) Building Energy Rating (BER) dataset from its original size to a refined set of features for predictive modeling.

## 1. Initial Dataset Overview
- **Source**: SEAI BER Public Search API dataset.
- **Initial Dimensions**: ~1,351,582 rows across approx. **215 columns**.
- **Objective**: Reduce dimensionality and eliminate noise to improve model performance and prevent data leakage.

## 2. Phase 1: Systematic Data Reduction
The initial cleaning phase focused on technical data quality, sparsity, and statistical redundancy.

### Step 1: Sparsity & Initial Noise Reduction (215 to 138 Columns)
*   **Action**: Analyzed the percentage of null values for every column.
*   **Result**: Removed any column with **more than 50% missing values** (e.g., `StorageLosses`, `SolarHotWaterHeating`).

### Step 2: Suspicious Character Filtering (138 to 97 Columns)
*   **Action**: Identified and removed columns where more than 50% of the entries consisted of placeholder values, zeros, or "suspicious characters".
*   **Suspicious Characters List**: `['.', '-', '*', '?', '0', '0.0', 'None', 'nan', 'none', '?', 'N/A', 'NAN']`.
*   **Result**: Eliminated columns with poor data integrity that would have introduced bias.

### Step 3: Redundant Metadata Removal (97 to 92 Columns)
*   **Action**: Identified and removed all **Description** columns (e.g., `WallDescription`, `RoofDescription`).
*   **Reasoning**: These typically provided natural language metadata that was redundant or difficult to tokenize without significantly increasing the feature space.

### Step 4: Cardinality & Error Filtering (92 to 85 Columns)
*   **Action**: 
    1.  **High Cardinality**: Dropped categorical features with more than 20 unique string values.
    2.  **Error Columns**: Dropped internal error-tracking or derived columns found in the SEAI dataset.
*   **Result**: Prevented model overfitting and removed internal processing artifacts.

### Step 5: Target Correlation Analysis (85 to 76 Columns)
*   **Action**: Calculated the correlation between features and the final **Energy Rating/BER Rating** value.
*   **Result**: Removed features with a correlation higher than **95%**, as these represent mathematical redundancies (Calculated outcomes) rather than independent physical drivers.


## 3. Phase 2: Domain-Driven Feature Engineering (Reduced to 47 Columns)
Guided by academic research (Usman Ali et al. 2024, Tripathi & Kumar 2024), we applied several filters to eliminate redundancy and target leakage.

### Step 1: Removing Data Leakage (Target-Derived Metrics)
Removed columns that are directly derived from or used to calculate the Energy Rating/BER Rating. Keeping these leads to artificially high accuracy.
- **Dropped**: `CO2Rating`, `CO2Lighting`, `CO2PumpsFans`, `CO2MainWater`, `CO2MainSpace`, `CO2SecondarySpace`, `MPCDERValue`, `CPC`.

### Step 2: Removing Administrative & Non-Physical Data
Removed columns that describe the assessment process rather than the building's physical properties.
- **Dropped**: `TypeofRating`, `PurposeOfRating`, `MultiDwellingMPRN`.

### Step 3: Eliminating Outcome Variables
Removed "Calculated Outcomes" which are results of the building's performance, not causes.
- **Dropped**: `DeliveredLightingEnergy`, `DeliveredEnergyPumpsFans`, `DeliveredEnergyMainWater`, `DeliveredEnergyMainSpace`, `DeliveredEnergySecondarySpace`.

### Step 4: Physical Component Consolidation
Refined physical drivers to use representative values instead of granular sub-components that add noise.
- **Dropped**: Granular wall components like `FirstWallUValue`, `FirstWallArea`, `FirstWallIsSemiExposed`, `FirstWallAgeBandId`, `FirstWallTypeId`.

### Step 5: Mitigating Multi-Collinearity
Removed redundant area metrics that were highly correlated with other size-related features.
- **Dropped**: `GroundFloorArea`, `FirstFloorArea`, `PredominantRoofTypeArea`.

### Step 6: Final Sparse Column Removal
Removed remaining columns that were consistently under-reported by assessors.
- **Dropped**: `SWHPumpSolarPowered`, `PermeabilityTest`, `DraftLobby`.

## 4. Final Feature Set (47 Columns)
The final dataset consists of the following 47 physical and efficiency-related drivers:

1.  **DwellingTypeDescr**: Description of the dwelling type.
2.  **Year_of_Construction**: Construction year.
3.  **BerRating**: The target energy efficiency rating value.
4.  **UValueWall**: Wall thermal transmittance.
5.  **UValueRoof**: Roof thermal transmittance.
6.  **UValueFloor**: Floor thermal transmittance.
7.  **UValueWindow**: Window thermal transmittance.
8.  **UvalueDoor**: Door thermal transmittance.
9.  **WallArea**: Total wall area.
10. **RoofArea**: Total roof area.
11. **FloorArea**: Total floor area.
12. **WindowArea**: Total window area.
13. **DoorArea**: Total door area.
14. **NoStoreys**: Number of storeys.
15. **HSMainSystemEfficiency**: Heating system efficiency.
16. **HSEffAdjFactor**: Efficiency adjustment factor.
17. **HSSupplHeatFraction**: Supplemental heat fraction.
18. **HSSupplSystemEff**: Supplemental system efficiency.
19. **WHMainSystemEff**: Water heating efficiency.
20. **WHEffAdjFactor**: Water heating adjustment factor.
21. **SupplSHFuel**: Supplemental space heating fuel type ID.
22. **SupplWHFuel**: Supplemental water heating fuel type ID.
23. **SHRenewableResources**: Space heating renewable resource usage.
24. **WHRenewableResources**: Water heating renewable resource usage.
25. **NoOfFansAndVents**: Quantity of ventilation openings.
26. **VentilationMethod**: Method of ventilation (natural/mechanical).
27. **StructureType**: Building construction material/structure.
28. **SuspendedWoodenFloor**: Boolean for floor type.
29. **PercentageDraughtStripped**: Level of draught prevention.
30. **NoOfSidesSheltered**: Level of exposure protection.
31. **HeatSystemControlCat**: Control category for heating.
32. **HeatSystemResponseCat**: Response category for heating.
33. **NoCentralHeatingPumps**: Number of pumps in the system.
34. **CHBoilerThermostatControlled**: Boolean for thermostat control.
35. **OBBoilerThermostatControlled**: Boolean for out-of-building boiler control.
36. **OBPumpInsideDwelling**: Boolean for pump location.
37. **WarmAirHeatingSystem**: Boolean for warm air systems.
38. **UndergroundHeating**: Boolean for underfloor heating.
39. **DistributionLosses**: Loss factor for energy distribution.
40. **CylinderStat**: Hot water cylinder status.
41. **CombinedCylinder**: Hot water cylinder type.
42. **GroundFloorHeight**: Floor-to-ceiling height (Ground).
43. **FirstFloorHeight**: Floor-to-ceiling height (First).
44. **ThermalBridgingFactor**: Thermal bridging impact value.
45. **ThermalMassCategory**: Thermal capacity of the building.
46. **PredominantRoofType**: Type of roof construction.
47. **LivingAreaPercent**: Percentage of space dedicated to living areas.

## References
- Usman Ali et al. (2024): Refined subset selection for building energy characterization.
- Tripathi & Kumar (2024): Mitigating data leakage in energy performance modeling.
- Zhang et al. (2023): Preventing overfitting in physical driver discovery.
- Benavente-Peces (2020): Deep learning approaches for energy efficiency in buildings.
