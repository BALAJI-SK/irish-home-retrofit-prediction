# Report 01 — Raw Dataset Description

## Source

**Provider:** Sustainable Energy Authority of Ireland (SEAI)  
**Dataset:** BER (Building Energy Rating) Residential Public Register  
**File:** `BER_Residential_Data.csv`  
**Size on disk:** ~1.4 GB  
**Rows:** 1,354,360  
**Columns:** 211  

This is Ireland's national register of all BER certificates ever issued. Every time a dwelling is sold, rented, or built, an accredited BER assessor surveys it, enters measurements into the DEAP software, and the certificate is registered here.

---

## What is a BER Rating?

BER stands for **Building Energy Rating**. It is expressed in **kWh/m²/yr** — the primary energy a standardised occupant would use per square metre of floor area per year, under standardised weather and occupancy assumptions.

| Rating | kWh/m²/yr |
|--------|-----------|
| A1 | < 25 |
| A2 | 25–50 |
| A3 | 50–75 |
| B1 | 75–100 |
| B2 | 100–125 |
| B3 | 125–150 |
| C1 | 150–175 |
| C2 | 175–200 |
| C3 | 200–225 |
| D1 | 225–260 |
| D2 | 260–300 |
| E1 | 300–340 |
| E2 | 340–380 |
| F | 380–450 |
| G | > 450 |

**Critical point:** BER is an *asset rating*, not a measure of actual energy consumption. It uses standardised assumptions (same weather, same occupancy pattern, same thermostat setting for every dwelling). This means BER tells you about the *building fabric and systems*, not the actual occupant behaviour. This is by design — it makes dwellings comparable.

---

## How BER is Calculated — DEAP

BER is calculated by the **DEAP** engine (Dwelling Energy Assessment Procedure), Ireland's national implementation of the EU Energy Performance of Buildings Directive. An assessor physically surveys the dwelling and enters:
- Wall, roof, floor, window U-values
- Areas of each element
- Heating system type and efficiency
- Hot water cylinder details
- Ventilation system details
- Renewable energy systems

DEAP then calculates energy demand deterministically. The output — `BerRating` — is the column we are predicting.

---

## Target Variable: `BerRating`

| Statistic | Value |
|-----------|-------|
| Count | 1,354,360 |
| Mean | 205.6 kWh/m²/yr |
| Median | 183.4 kWh/m²/yr |
| Std Dev | 161.4 kWh/m²/yr |
| Min | −472.99 (invalid) |
| Max | 32,134.94 (invalid) |
| Valid range (Paper 1) | 0–2000 kWh/m²/yr |

The raw distribution is **right-skewed** — most dwellings cluster between 100–300, but extreme outliers (data entry errors) push the mean up. The model is trained on `log1p(BerRating)` to handle this skew.

---

## Column Categories (211 total)

### 1. DEAP Input Features (~100 columns)
Physical measurements entered by the assessor. These are the legitimate predictors.

Examples: `UValueWall`, `UValueRoof`, `WallArea`, `WindowArea`, `HSMainSystemEfficiency`, `MainSpaceHeatingFuel`, `Year_of_Construction`

### 2. DEAP Output / Derived Columns (~35 columns) — DATA LEAKAGE
These are calculated *by DEAP* after `BerRating` is determined. Using them as features would be circular — the model would learn to invert a calculation rather than learn from physical inputs.

| Column | Why it's leakage |
|--------|-----------------|
| `EnergyRating` | Letter-band version of BerRating (A1, A2, … G) |
| `CO2Rating` | CO₂ kg/m²/yr — DEAP output derived from BerRating |
| `MPCDERValue` | DEAP carbon performance metric |
| `DeliveredEnergyMainSpace` | Total space heating kWh/yr — DEAP output |
| `DeliveredEnergyMainWater` | Total water heating kWh/yr — DEAP output |
| `PrimaryEnergyMainSpace` | Primary energy kWh/yr — DEAP output |
| `CO2MainSpace` | CO₂ kg/yr — DEAP output |
| `RER` | Renewable Energy Ratio — DEAP output |
| `EPC` | Energy Performance Coefficient — DEAP output |
| `CPC` | Carbon Performance Coefficient — DEAP output |
| `TotalDeliveredEnergy` | Sum of all delivered energy — DEAP output |
| *(+18 more)* | All `Delivered*`, `PrimaryEnergy*`, `CO2*` columns |

### 3. Near-Null Columns (>80% missing) — 38 columns
Columns with so few values they cannot train a model reliably.

Examples:
- `ApertureArea` — 95.81% null (solar collector aperture, very rare)
- `VolumeOfPreHeatStore` — 98.34% null + single value
- `ThirdWall*` columns — 82%+ null (third wall construction type)
- `CHP*` columns — 98.2% null (combined heat and power, very rare)
- `gsd*` columns — 98.2–99.36% null

### 4. Free-Text Comment Columns — 7 columns
Unstructured assessor notes. Not useful for ML.

Examples: `FirstWallDescription`, `FirstEnerProdComment`, `SecondEnerConsumedComment`

### 5. Administrative / Identifier Columns — 4 columns
Not physical properties of the dwelling.

| Column | Reason to drop |
|--------|---------------|
| `DateOfAssessment` | Assessment date, not a building property |
| `SA_Code` | Small Area geographic code (too high cardinality, covered by CountyName) |
| `prob_smarea_error_0corr` | Statistical probability artifact |
| `prob_smarea_error_100corr` | Statistical probability artifact |

### 6. Redundant ID Columns — 4 columns
Same information as a description column, just in numeric ID form.

`FirstWallTypeId` (same as `FirstWallType_Description`), `SecondWallTypeId`, `FirstEnergyTypeId`, `SecondEnergyTypeId`

---

## Dwelling Type Distribution (raw)

| Type | Count | % |
|------|-------|---|
| Detached house | 409,537 | 30.2% |
| Semi-detached house | 360,277 | 26.6% |
| Mid-terrace house | 184,490 | 13.6% |
| Mid-floor apartment | 102,932 | 7.6% |
| End of terrace house | 102,927 | 7.6% |
| Top-floor apartment | 74,351 | 5.5% |
| Ground-floor apartment | 72,836 | 5.4% |
| House | 27,095 | 2.0% |
| Maisonette | 17,172 | 1.3% |
| Apartment | 2,243 | 0.2% |
| Basement Dwelling | 500 | 0.04% |

All dwelling types are included in the model. `DwellingTypeDescr` is a feature, not a filter.

---

## Key Data Quality Issues Found

| Issue | Affected Rows | Column |
|-------|--------------|--------|
| Negative BerRating | ~1,800 rows | `BerRating` |
| BerRating > 2,000 | ~2,124 rows | `BerRating` |
| Year of construction 2104 | a few rows | `Year_of_Construction` |
| Year of construction 1753 | a few rows | `Year_of_Construction` |
| 51.19% structural nulls | 693,242 rows | Hot water cylinder group (15 cols) |
| 'Please select' in StructureType | 76,506 rows | `StructureType` |
| Trailing whitespace in strings | Most object cols | All categorical |

---

## Research Papers Used

This project is grounded in 8 peer-reviewed papers:

| # | Paper | Key Contribution |
|---|-------|-----------------|
| 1 | Curtis et al. 2014 — ESRI Working Paper | Irish BER dataset characterisation; outlier threshold BER > 2000; log-transform rationale |
| 2 | Ali et al. 2024 — Energy & Buildings | WindowToWallRatio feature; LightGBM performance on urban BER |
| 3 | Benavente-Peces & Ibadah 2020 — Energies | ML classifier benchmarks for EPC |
| 4 | Dinmohammadi et al. 2023 — Energies | PSO-RF stacking; feature importance validation |
| 5 | Tripathi & Kumar 2024 — Irish BER + LightGBM | Direct precedent: LightGBM on Irish SEAI data, retrofit analysis |
| 6 | McGarry 2023 — TU Dublin | DEAP gap vs actual consumption; confirms BerRating is deterministic |
| 7 | Zhang et al. 2023 — Energy | 70/15/15 split strategy; LightGBM + SHAP methodology |
| 8 | Bilous et al. 2018 — J. Building Engineering | Multi-regression baseline for BER prediction |
