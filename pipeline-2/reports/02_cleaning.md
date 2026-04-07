# Report 02 — Data Cleaning

## Summary

| Stage | Rows | Columns |
|-------|------|---------|
| Raw CSV | 1,354,360 | 211 |
| After column selection | 1,354,360 | 104 |
| After outlier removal | 1,350,432 | 104 |
| After imputation + engineering | **1,350,432** | **119** |

**Rows retained: 99.71%**  
**Columns retained: 104 input → 119 (with 15 engineered features added)**

---

## Step 1: Column Selection — What We Dropped and Why

### 1A. Data Leakage Columns (25 dropped)

These columns are **outputs of the DEAP calculation**, meaning they are computed *after* `BerRating` is determined. Including them would let the model cheat — it would learn to reverse-engineer a formula instead of learning from physical building properties.

**How we confirmed leakage:**  
- Domain knowledge: DEAP methodology documentation shows these as outputs, not inputs
- McGarry (2023, Paper 6) explicitly confirms that EPC, CPC, RER, and all delivered/primary energy columns are DEAP outputs
- Pearson correlation: these columns correlate with BerRating at r > 0.99

| Column | Type | Why Dropped |
|--------|------|-------------|
| `EnergyRating` | Letter band A1–G | Direct categorical re-encoding of BerRating |
| `CO2Rating` | kg CO₂/m²/yr | DEAP output, computed from BerRating |
| `MPCDERValue` | Carbon metric | DEAP output |
| `DeliveredEnergyMainSpace` | kWh/yr | DEAP output |
| `DeliveredEnergyMainWater` | kWh/yr | DEAP output |
| `DeliveredEnergyPumpsFans` | kWh/yr | DEAP output |
| `DeliveredLightingEnergy` | kWh/yr | DEAP output |
| `DeliveredEnergySecondarySpace` | kWh/yr | DEAP output |
| `DeliveredEnergySupplementaryWater` | kWh/yr | DEAP output |
| `PrimaryEnergyMainSpace` | kWh/yr | DEAP output |
| `PrimaryEnergyMainWater` | kWh/yr | DEAP output |
| `PrimaryEnergyPumpsFans` | kWh/yr | DEAP output |
| `PrimaryEnergyLighting` | kWh/yr | DEAP output |
| `PrimaryEnergySecondarySpace` | kWh/yr | DEAP output |
| `PrimaryEnergySupplementaryWater` | kWh/yr | DEAP output |
| `CO2MainSpace` | kg CO₂/yr | DEAP output |
| `CO2MainWater` | kg CO₂/yr | DEAP output |
| `CO2Lighting` | kg CO₂/yr | DEAP output |
| `CO2PumpsFans` | kg CO₂/yr | DEAP output |
| `CO2SecondarySpace` | kg CO₂/yr | DEAP output |
| `CO2SupplementaryWater` | kg CO₂/yr | DEAP output |
| `TotalDeliveredEnergy` | kWh/yr | DEAP output (also 65.85% null) |
| `RER` | Ratio | Renewable Energy Ratio — DEAP output |
| `EPC` | Coefficient | Energy Performance Coefficient — DEAP output |
| `CPC` | Coefficient | Carbon Performance Coefficient — DEAP output |
| `DistributionLosses` | kWh/yr | DEAP lookup value derived from system type |
| `RenewEPnren` | kWh/yr | DEAP output |
| `RenewEPren` | kWh/yr | DEAP output |

### 1B. Near-Null Columns >80% Missing (38 dropped)

A column with 80%+ missing values cannot be reliably imputed — you would be making up 80% of the data. These were dropped entirely.

| Column Group | % Null | Reason Missing |
|-------------|--------|---------------|
| Solar collector columns (ApertureArea, ZeroLossCollectorEff, etc.) | 95.81% | Only ~4% of dwellings have solar thermal |
| `VolumeOfPreHeatStore` | 98.34% | Extremely rare system type + single value |
| `gsd*` columns (gsdHSSupplHeatFraction etc.) | 98.2–99.4% | Rare community heating systems |
| `CHP*` columns (CHPUnitHeatFraction etc.) | 98.2% | Combined heat & power — very rare |
| `SupplHSFuelTypeID`, `DistLossFactor` | 98.2% | Rare supplementary systems |
| Third/Second boiler columns | 98.2% | Rare multi-boiler setups |
| `ThirdWall*` columns (all 7) | 82–86% | Third wall type — most dwellings have 1–2 |
| `ThirdEner*` columns (all) | 98.2% | Third renewable system — extremely rare |
| `SecondEnerConsumed*` columns | 98.2% | Near-zero population |
| `*Comment` free-text columns | 87–100% | Unstructured text, not useful for ML |
| `SolarHeatFraction`, `SolarSpaceHeatingSystem` | 98.2% | Rare |
| `TotalPrimaryEnergyFact`, `TotalCO2Emissions` | 98.2% | DEAP outputs + near-null |

### 1C. Administrative / Identifier Columns (4 dropped)

| Column | Reason |
|--------|--------|
| `DateOfAssessment` | Assessment date, not a physical building property |
| `SA_Code` | Small Area code — too high cardinality (>5000 unique), geographic info already in `CountyName` |
| `prob_smarea_error_0corr` | Statistical modeling artifact, not a building feature |
| `prob_smarea_error_100corr` | Statistical modeling artifact |

### 1D. Free-Text Comment Columns (7 dropped)

Assessor free-text notes. Unstructured, high cardinality, not useful for ML.

`FirstWallDescription`, `SecondWallDescription`, `FirstEnerProdComment`, `SecondEnerProdComment`, `ThirdEnerProdComment`, `FirstEnerConsumedComment`, `SecondEnerConsumedComment`

### 1E. Redundant ID Columns (4 dropped)

Numeric ID versions of columns already kept as description strings.

`FirstWallTypeId` → kept `FirstWallType_Description`  
`SecondWallTypeId` → kept `SecondWallType_Description`  
`FirstEnergyTypeId` → kept `FirstEnergyType_Description`  
`SecondEnergyTypeId` → kept `SecondEnergyType_Description`

### 1F. DEAP Contribution Outputs (4 dropped)

`FirstPartLTotalContribution`, `SecondPartLTotalContribution` — calculated total energy contributions, DEAP outputs

`FirstEnerConsumedDelivered`, `SecondEnerConsumedDelivered` — near-zero population + DEAP-derived

---

## Step 2: Outlier Filtering

### BerRating Outliers

**Rule:** Remove rows where `BerRating < 0` or `BerRating > 2000`

**Justification:** Curtis et al. (2014, Paper 1) explicitly define BerRating > 2000 as a data entry error. Negative values are physically impossible (you cannot have negative energy consumption per unit area under DEAP standardised assumptions).

| Category | Count | Action |
|----------|-------|--------|
| BerRating < 0 | ~1,800 rows | Dropped |
| BerRating > 2,000 | ~2,124 rows | Dropped |
| **Total dropped** | **3,924** | |

### Year of Construction Outliers

**Rule:** Remove rows where `Year_of_Construction < 1700` or `> 2026`

**Justification:** Ireland's oldest inhabited buildings date to the early 1700s. Years above 2026 are future dates — data entry errors. Note: the raw dataset contains values as low as 1753 (legitimate historic buildings) and as high as 2104 (clearly invalid).

| Category | Count | Action |
|----------|-------|--------|
| Year < 1700 | ~4 rows | Dropped |
| Year > 2026 | 0 rows after 2104 found | Dropped |
| **Total dropped** | **4** | |

**Total rows retained after outlier filtering: 1,350,432 (99.71%)**

---

## Step 3: Missing Data — Strategy per Group

### 3A. The MNAR Group (Missing Not At Random) — 15 columns, 51.19% null

**What these columns are:** The hot water cylinder group — `StorageLosses`, `WaterStorageVolume`, `InsulationType`, `InsulationThickness`, `CombiBoiler`, `SolarHotWaterHeating`, `ElecImmersionInSummer`, `ManuLossFactorAvail`, `KeepHotFacility`, `DeclaredLossFactor`, `TempFactorUnadj`, `TempFactorMultiplier`, `PrimaryCircuitLoss`, `CombiBoilerAddLoss`, `ElecConsumpKeepHot`

**Why 51.19% are null:** When a dwelling has a **combi boiler**, there is no separate hot water storage cylinder. DEAP does not ask for cylinder details if there is no cylinder. The missingness is not random — it is structurally caused by the heating system type. This is textbook **MNAR (Missing Not At Random)**.

**Wrong approach:** Standard imputation (mean/mode) would invent cylinder data for houses that have no cylinder. This is physically wrong.

**Correct approach:**
1. Create a binary flag: `has_hw_cylinder = 1` if these columns are not null, `0` if null
2. Fill categorical MNAR nulls with `'No_cylinder'` — a new category that the model can learn from
3. Fill numeric MNAR nulls with `0.0` — no cylinder means no storage volume, no insulation thickness, no losses

This way the model learns: "when `has_hw_cylinder = 0`, the water heating behaviour is different" — which is true.

### 3B. Second Wall Group — 15 columns, ~51–67% null

**Why null:** Most dwellings have only one wall construction type. The second wall is null not because data is missing but because there is no second wall type.

**Strategy:** Fill categorical with `'None'`, fill numeric (area, U-value, age band) with `0.0`

### 3C. Renewable Energy Production — 2 columns, 1.8% null

`FirstEnerProdDelivered`, `SecondEnerProdDelivered` — null when no renewable system is installed.

**Strategy:** Fill with `0.0` (no system = 0 kWh delivered)

### 3D. General Small-Percentage Nulls — ~30 columns, 1–10% null

These are random missingness (MAR — Missing At Random). The assessor simply did not enter the value.

**Strategy:**
- **Categorical columns** (e.g. `PredominantRoofType` 10.52% null, `FirstWallType_Description` 23.21% null, `StructureType` ~7% null after cleaning): fill with **mode** computed from 200,000-row sample
- **Numeric columns** (e.g. heating system efficiency, ventilation counts): fill with **median** computed from 200,000-row sample

Using sample-computed statistics rather than chunk-by-chunk statistics ensures consistency across all chunks.

### 3D Special — `StructureType` Placeholder

76,506 rows (5.6%) contain `'Please select'` in `StructureType` — this is not truly a structural type, it means the assessor forgot to select. Replaced with `NaN` before imputation, then filled with mode (`'Masonry'`).

---

## Step 4: String Cleaning

All categorical columns had trailing whitespace stripped. Examples from raw data:
- `'Existing       '` → `'Existing'`
- `'Masonry                       '` → `'Masonry'`
- `'Heating Oil                   '` → `'Heating Oil'`
- `'YES'` / `'YES '` / `'Yes'` → standardised after strip

---

## Post-Cleaning Null Check

**Zero nulls remaining** in the clean parquet file across all 119 columns and 1,350,432 rows.

---

## Memory Strategy

The raw CSV is 1.4 GB. The machine has 8 GB RAM. Loading the full file at once would consume all available memory and likely crash.

**Solution: Two-pass chunked processing**

- **Pass 1:** Read 4 chunks (200,000 rows) → compute global median and mode for every column → free memory
- **Pass 2:** Read full file in 50,000-row chunks → clean each chunk → write immediately to parquet using `pyarrow.parquet.ParquetWriter` (streaming, never accumulates in RAM)

Each 50,000-row chunk uses ~50 MB RAM. Peak usage during processing was well under 2 GB.

**Output format:** Apache Parquet with Snappy compression — reads back in 3.9 seconds vs ~60 seconds for the CSV.
