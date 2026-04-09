# DEAP-Informed Analysis: Feature Classification & Implementation Corrections

> **Source:** Cross-referencing DEAP Manual v4.2.7, Introduction to DEAP for Professionals,
> `45_cols_models_output.md`, and `pipeline-2/reports/03_features.md` (118-col feature list).
>
> **Goal:** Determine exactly which features are raw DEAP *inputs* vs DEAP *calculated outputs*,
> fix the U-value override values in `new_implemention.md`, and identify gaps in the plan.

---

## 1. What DEAP Actually Calculates — The Formula Chain

From the DEAP Manual v4.2.7 and the "Introduction to DEAP for Professionals", the pipeline is:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — SURVEY INPUTS (Raw measurements entered on-site — SAFE as features)  │
│                                                                                 │
│  Fabric:          UValueWall, UValueRoof, UValueFloor, UValueWindow, UvalueDoor │
│  Areas:           WallArea, RoofArea, FloorArea, WindowArea, DoorArea           │
│  Geometry:        NoStoreys, GroundFloorAreasq_m, GroundFloorHeight, LivingArea │
│  Heating inputs:  MainSpaceHeatingFuel, HSMainSystemEfficiency, HSSupplHeatFr.  │
│  Ventilation:     NoOfChimneys, NoOfFans, DraftLobby, VentilationMethod         │
│  Controls:        HeatSystemControlCat, HeatSystemResponseCat, ThermostatFlag   │
│  Water heat:      WHMainSystemEff, WaterStorageVolume, SolarHotWaterHeating      │
│  Renewables:      FirstEnergyType_Description (type only — not kWh produced)    │
│  Misc:            Year_of_Construction, DwellingTypeDescr, ThermalMassCategory  │
└─────────────────────────────────────────────────────────────────────────────────┘
                            ↓  DEAP calculates ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — DEAP INTERMEDIATE CALCULATIONS (outputs of tabs — RISKY as features) │
│                                                                                 │
│  Tab: Dist. system losses and gains                                             │
│    → DistributionLosses   ← CALCULATED from HeatSystemControlCat +              │
│                             HeatSystemResponseCat + pump/emitter settings        │
│    → TempAdjustment       ← CALCULATED from control/emitter type table (Tab4a)  │
│    → HeatSystemResponseCat in combination feeds into DistributionLosses          │
│                                                                                 │
│  Tab: Energy requirements                                                       │
│    → HSEffAdjFactor       ← CALCULATED from weather/load compensation installed │
│    → WHEffAdjFactor       ← CALCULATED from HW controls                        │
│    → SHRenewableResources ← CALCULATED from renewable type + fraction delivered │
│    → WHRenewableResources ← CALCULATED from solar thermal contribution           │
│                                                                                 │
│  Tab: Results                                                                   │
│    → TempFactorMultiplier ← CALCULATED cylinder temperature factor (Table 2)    │
│    → TempFactorUnadj      ← CALCULATED cylinder unadjusted temperature          │
│    → DeclaredLossFactor   ← CALCULATED from cylinder insulation type/thickness  │
│    → StorageLosses        ← CALCULATED cylinder heat loss                       │
│    → PrimaryCircuitLoss   ← CALCULATED pipe heat loss                           │
│    → FirstEnerProdDelivered ← CALCULATED kWh from PV: 0.80 × kWp × S × ZPV      │
│    → SecondEnerProdDelivered ← Same for second system                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                            ↓  DEAP sums ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — FINAL DEAP OUTPUTS (pure outputs — NEVER use as features)            │
│                                                                                 │
│  BerRating, EnergyRating, CO2Rating, MPCDERValue, CPC, EPC                      │
│  DeliveredEnergyMainSpace, DeliveredEnergyMainWater                             │
│  TotalDeliveredEnergy, PrimaryEnergyMainSpace, CO2MainSpace, RER                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Revelation from the DEAP Manual

The DEAP manual (Appendix S13, Table S12 "Data to be collected") lists exactly what a
BER assessor enters on-site. **`DistributionLosses` is listed as a survey item** in
Table S12 — meaning it IS something an assessor records, not a purely calculated value.

However, the "Dist. system losses and gains" TAB in DEAP calculates several outputs from it.
The field itself in the SEAI database likely reflects whether the pipes/cylinder are insulated
(a binary/categorical survey question), **not the calculated loss number**. This matters:

- `DistributionLosses` in the SEAI CSV: likely a **lookup value** from a table given
  heating system + pipe insulation description — treat as **borderline** (survey-derived
  default, not a pure formula output). It is safer to remove it than keep it.

---

## 2. Complete Feature Classification for Both Pipelines

### 2A. Features in the 45-col Set — Status from DEAP Manual

| Column | DEAP Stage | Status | Action |
|---|---|---|---|
| `DwellingTypeDescr` | Stage 1 — Survey | ✅ Safe input | Keep |
| `Year_of_Construction` | Stage 1 — Survey | ✅ Safe input | Keep |
| `UValueWall` | Stage 1 — Survey | ✅ Safe input | Keep |
| `UValueRoof` | Stage 1 — Survey | ✅ Safe input | Keep |
| `UValueFloor` | Stage 1 — Survey | ✅ Safe input | Keep |
| `UValueWindow` | Stage 1 — Survey | ✅ Safe input | Keep |
| `UvalueDoor` | Stage 1 — Survey | ✅ Safe input | Keep |
| `WallArea` | Stage 1 — Survey | ✅ Safe input | Keep |
| `RoofArea` | Stage 1 — Survey | ✅ Safe input | Keep |
| `FloorArea` | Stage 1 — Survey | ✅ Safe input | Keep |
| `WindowArea` | Stage 1 — Survey | ✅ Safe input | Keep |
| `DoorArea` | Stage 1 — Survey | ✅ Safe input | Keep |
| `NoStoreys` | Stage 1 — Survey | ✅ Safe input | Keep |
| `HSMainSystemEfficiency` | Stage 1 — Survey (from HARP DB) | ✅ Safe input | Keep |
| `HSSupplHeatFraction` | Stage 1 — Survey | ✅ Safe input | Keep |
| `HSSupplSystemEff` | Stage 1 — Survey | ✅ Safe input | Keep |
| `WHMainSystemEff` | Stage 1 — Survey (from HARP DB) | ✅ Safe input | Keep |
| `SupplSHFuel` | Stage 1 — Survey | ✅ Safe input | Keep |
| `SupplWHFuel` | Stage 1 — Survey | ✅ Safe input | Keep |
| `NoOfFansAndVents` | Stage 1 — Survey | ✅ Safe input | Keep |
| `VentilationMethod` | Stage 1 — Survey | ✅ Safe input | Keep |
| `StructureType` | Stage 1 — Survey | ✅ Safe input | Keep |
| `SuspendedWoodenFloor` | Stage 1 — Survey | ✅ Safe input | Keep |
| `PercentageDraughtStripped` | Stage 1 — Survey | ✅ Safe input | Keep |
| `NoOfSidesSheltered` | Stage 1 — Survey | ✅ Safe input | Keep |
| `HeatSystemControlCat` | Stage 1 — Survey | ✅ Safe input | Keep |
| `HeatSystemResponseCat` | Stage 1 — Survey | ✅ Safe input | Keep |
| `NoCentralHeatingPumps` | Stage 1 — Survey | ✅ Safe input | Keep |
| `CHBoilerThermostatControlled` | Stage 1 — Survey | ✅ Safe input | Keep |
| `OBBoilerThermostatControlled` | Stage 1 — Survey | ✅ Safe input | Keep |
| `OBPumpInsideDwelling` | Stage 1 — Survey | ✅ Safe input | Keep |
| `WarmAirHeatingSystem` | Stage 1 — Survey | ✅ Safe input | Keep |
| `UndergroundHeating` | Stage 1 — Survey | ✅ Safe input | Keep |
| `CylinderStat` | Stage 1 — Survey | ✅ Safe input | Keep |
| `CombinedCylinder` | Stage 1 — Survey | ✅ Safe input | Keep |
| `GroundFloorHeight` | Stage 1 — Survey | ✅ Safe input | Keep |
| `FirstFloorHeight` | Stage 1 — Survey | ✅ Safe input | Keep |
| `ThermalBridgingFactor` | Stage 1 — Survey (default from table or calculated) | ✅ Keep (it's an official DEAP input) | Keep |
| `ThermalMassCategory` | Stage 1 — Survey (Table S10 lookup) | ✅ Safe input | Keep |
| `PredominantRoofType` | Stage 1 — Survey | ✅ Safe input | Keep |
| `LivingAreaPercent` | Stage 1 — Survey | ✅ Safe input | Keep |
| `CountyName` | Stage 1 — Admin | ✅ Safe (geographic) | Keep |
| **`DistributionLosses`** | **Stage 2 — Lookup table from controls** | ⚠️ **Borderline leaky** | **Remove** |
| **`HSEffAdjFactor`** | **Stage 2 — Weather comp. calculation** | ❌ **DEAP calculates** | **Remove** |
| **`WHEffAdjFactor`** | **Stage 2 — HW controls calculation** | ❌ **DEAP calculates** | **Remove** |
| **`SHRenewableResources`** | **Stage 2 — Renewable fraction** | ❌ **DEAP calculates** | **Remove** |
| **`WHRenewableResources`** | **Stage 2 — Solar thermal fraction** | ❌ **DEAP calculates** | **Remove** |

**Net result: 45-col → 40 safe columns after removing 5**

### 2B. Critical Missing Features — Must Add from Raw Data

From DEAP Manual Appendix S13 Table S12 "Data to be collected" — these are legitimate
survey inputs that are in the raw SEAI CSV and in the 118-col pipeline:

| Column | DEAP Source | Why Critical |
|---|---|---|
| `MainSpaceHeatingFuel` | Stage 1 — Primary heating fuel | **Cannot simulate heat pump without this**. DEAP manual explicitly lists "Fuel for main heating" as a mandatory survey item |
| `MainWaterHeatingFuel` | Stage 1 — Water heating fuel | Needed for heat pump + solar water scenarios |
| `SolarHotWaterHeating` | Stage 1 — "Solar panel (yes/no)" | Direct survey item (Table S12, row "Solar water heating") |
| `LowEnergyLightingPercent` | Stage 1 — Lighting survey | DEAP Appendix L uses % low-energy lamps directly as a survey input |
| `GroundFloorAreasq_m` | Stage 1 — Total floor area | Required for `FabricHeatLossPerM2` normalisation |
| `HESSchemeUpgrade` | Stage 1 — Grant scheme flag | Identifies homes that received SEAI retrofit grants |
| `FirstEnergyType_Description` | Stage 1 — Renewable system *type* | Survey input (what type of system is installed, not how much it produces) |
| `NoOfChimneys` | Stage 1 — Ventilation survey | "Count of chimneys" is listed in Table S12 as a required ventilation survey item |

**Important: `FirstEnergyType_Description` YES, `FirstEnerProdDelivered` NO**
The DEAP manual (Appendix M1) confirms `FirstEnerProdDelivered` is the *output* of:
`E_PV = 0.80 × kWp × S × Z_PV`
This is a calculated kWh value — it is a DEAP Stage 2 output. The *type* of renewable
system (`Solar PV`, `Wind`) is a survey input. The *kWh delivered* is calculated.

### 2C. Features in the 118-col Set — DEAP Status

Additional features from `pipeline-2/reports/03_features.md` that need review:

| Column | DEAP Status | Decision |
|---|---|---|
| `TempAdjustment` | ❌ DEAP Stage 2 output — intermittent heating correction calculated from Table 4e | **Remove from 118-col** |
| `TempFactorMultiplier` | ❌ DEAP Stage 2 output — cylinder temperature factor from Table 2 | **Remove from 118-col** |
| `TempFactorUnadj` | ❌ DEAP Stage 2 calculated | **Remove from 118-col** |
| `FabricHeatLossPerM2` | ⚠️ Engineered from raw inputs — safe IF we compute it from U×A ourselves | **Re-engineer, don't inherit from DEAP** |
| `FabricHeatLossProxy` | ⚠️ Same — must compute ourselves | **Re-engineer from scratch** |
| `WindowToWallRatio` | ✅ Computed from survey inputs only | **Keep if computed from WallArea/WindowArea** |
| `AgeBand` | ✅ Derived from Year_of_Construction — safe | **Keep** |
| `IsHeatPump` | ✅ Derived from HSMainSystemEfficiency > 100 | **Keep** |
| `HasSolarWaterHeating` | ✅ Derived from survey input | **Keep** |
| `HasRoofInsulation` | ✅ Derived from survey input | **Keep** |
| `HasWallInsulation` | ✅ Derived from survey input | **Keep** |
| `HasDoubleGlazing` | ✅ Derived from survey input | **Keep** |
| `AvgWallUValue` | ✅ Area-weighted calculation from survey inputs | **Keep** |
| `TotalFloorArea_computed` | ✅ Sum of floor areas — all survey inputs | **Keep** |
| `FanPowerManuDeclaredValue` | ✅ Survey input (mechanical ventilation fan spec) | **Keep** |
| `HeatExchangerEff` | ✅ Survey input (MVHR heat recovery %) | **Keep** |
| `PermeabilityTestResult` | ✅ Survey input (blower door test result) | **Keep** |
| `TGDLEdition` | ✅ Survey input (which regulation edition) | **Keep** |
| `FirstWallType_Description` | ✅ Survey input | **Keep** |
| `FirstWallAgeBandId` | ✅ Survey input (age band of wall construction) | **Keep** |
| `has_hw_cylinder` | ✅ Engineered from cylinder group (structural NULL = combi) | **Keep** |
| `DeclaredLossFactor` | ❌ Depends on cylinder insulation thickness — DEAP Table 2 **calculated** value | **Remove** |
| `StorageLosses` | ❌ DEAP Table 2 calculated value | **Borderline — survey descriptor or calculated?** |
| `PrimaryCircuitLoss` | ⚠️ DEAP Table 2 — primary circuit heat loss descriptor | **Keep as categorical descriptor, remove if numeric** |
| `FirstEnerProdDelivered` | ❌ DEAP Appendix M1 formula output: `0.80 × kWp × S × Z_PV` | **Remove** |
| `SecondEnerProdDelivered` | ❌ Same | **Remove** |
| `MultiDwellingMPRN` | ✅ Survey/admin input | **Keep** |
| `HESSchemeUpgrade` | ✅ Survey input | **Keep** |

---

## 3. Corrections to `new_implemention.md` — U-Value Override Values

The current plan uses some U-value targets. Cross-referencing the DEAP Manual:

### DEAP Table S2 — Building Regulations by Age Band (official Irish targets)

| Year of Regulations | Age Band | Roof (W/m²K) | Wall (W/m²K) | Floor (W/m²K) |
|---|---|---|---|---|
| 1976 (Draft) | F | 0.40 | 1.10 | 0.60 |
| 1981 (Draft) | G | 0.40 | 0.60 | 0.60 |
| 1991 | H | 0.35 | 0.55 | 0.45 |
| 1997 | I | 0.35 | 0.55 | 0.45 |
| 2002/2005 | J | 0.25 | 0.37 | 0.37 |
| 2008 | K | 0.22 | 0.27 | 0.25 |
| **2011 (TGD L)** | **L** | **0.20** | **0.21** | **0.21** |

### DEAP Table S9 — Window Default U-Values

| Glazing Type | Period | U-value (W/m²K) |
|---|---|---|
| Single glazed (Wood/PVC) | Any | 4.80 |
| Double, pre-2004, air-filled, no Low-E | pre-2004 | 3.10 |
| Double, 2004–2009, Low-E, air-filled | 2004–2009 | 2.20 |
| Double, 2010+, Low-E, argon-filled | 2010+ | **2.00** |
| Triple, pre-2010 | pre-2010 | 1.70 |
| Triple, 2010+ | 2010+ | **1.50** |

### DEAP Table S4 — Roof U-Values by Loft Insulation Thickness

| Insulation Thickness (mm) | Roof U-value (W/m²K) |
|---|---|
| None | 2.30 |
| 100mm | 0.40 |
| 150mm | 0.26 |
| 200mm | 0.20 |
| **250mm** | **0.16** |
| ≥ 300mm | **0.13** |

### DEAP Table S3 — Pre-retrofit Wall U-Values (typical for uninsulated pre-1967 stock)

| Wall Construction | Pre-1967 bands (A–E) | Typical U-value |
|---|---|---|
| Stone | A–E | **2.10** W/m²K |
| 225mm solid brick | A–E | **2.10** W/m²K |
| 300mm cavity (uninsulated) | A–E | **1.78–2.10** W/m²K |
| Timber frame | A | 2.50, B–C: 1.90, D–E: 1.10 |

### Corrected Override Values for `new_implemention.md`

The current plan's overrides are **directionally correct** but need these precise corrections:

| Retrofit Measure | Current Plan Value | DEAP-Correct Value | Source |
|---|---|---|---|
| Wall insulation → target | `UValueWall = 0.21` | ✅ Correct — TGD L 2011 | Table S2, Band L |
| Roof insulation → target | `UValueRoof = 0.16` | ✅ Correct — 250mm loft insulation | Table S4 |
| Floor insulation → target | `UValueFloor = 0.15` | **Change to 0.21** — TGD L 2011 floor standard | Table S2, Band L |
| Window upgrade → target  | `UValueWindow = 1.40`| **Change to 1.50** — DEAP triple glazing 2010+ default | Table S9 |
| Window "double" upgrade  | `UValueWindow = 2.0` | ✅ Correct — DEAP 2010+ double glazing default | Table S9 |
| Door upgrade → target    | `UvalueDoor = 1.40`  | Use `UvalueDoor = 1.40` (from TGD L Table 6a) | DEAP Table 6a |

**Pre-retrofit baseline to document (better than made-up numbers):**

| Element | Typical pre-retrofit value | DEAP source |
|---|---|---|
| Stone wall (pre-1900) | **2.10** W/m²K | Table S3: Stone, Band A |
| 300mm cavity wall (uninsulated, 1930–1966) | **1.78** W/m²K | Table S3: 300mm cavity, Band C/D |
| Uninsulated timber frame (pre-1950) | **1.90** W/m²K | Table S3: Timber frame, Band B/C |
| Uninsulated pitched roof | **2.30** W/m²K | Table S4: None |
| 100mm loft insulation (typical 1980-installed) | **0.40** W/m²K | Table S4 |
| Single glazed window | **4.80** W/m²K | Table S9: Single, Wood/PVC |
| Double, pre-2004 (air, no Low-E) | **3.10** W/m²K | Table S9 |
| Double, 2004-2009 | **2.20** W/m²K | Table S9 |
| Double, 2010+ | **2.00** W/m²K | Table S9 |

---

## 4. Age Band Impact on Feature Engineering

The DEAP Manual Table S1 defines 12 age bands (A–L). This is directly relevant to our
`AgeBand` engineered feature. The plan already uses these bins correctly:

```python
# Correct DEAP-aligned bins (from Table S1):
bins = [-inf, 1900, 1930, 1950, 1967, 1978, 1983, 1994, 2000, 2005, 2010, 2014, inf]
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
```

**Important note from Table S1:** Band L starts at 2014 (not 2016 as the current plan states).
The 45_cols_models_output.md uses 2016+ as its cut-off. The DEAP-correct split is **2014+**.
This affects the "2016+ homes gain ~1 kWh from deep retrofit" finding — it should be reframed
as "Band L (2014+) homes gain minimal BER improvement from deep retrofit."

---

## 5. New Insight: Why `DistributionLosses` Is Borderline, Not Definitely Leaky

From the DEAP Manual Appendix S13, Table S12, row "Water heating (cont.)":
> "Distribution losses?"

This confirms `DistributionLosses` is something a BER assessor **records from site survey**
about whether the primary circuit / pipes are insulated. It is NOT a calculated kWh value.
However, DEAP then **looks up the numeric loss factor** from a table given the pipe
insulation description — so what ends up in the SEAI database could be:
a) The survey categorical (insulated/not/partially), OR
b) The numeric lookup value from DEAP's tables

If it's the numeric lookup, it is table-derived — still remove it (borderline leaky).
If it's the categorical, it's safe. Without inspecting the raw data distribution, **safer to remove**.

---

## 6. New Insight: `HSMainSystemEfficiency` — How It Distinguishes Heat Pumps

From the DEAP Manual section on heating systems and the HARP database:
> "Heating system efficiencies are based on Gross Calorific Values and generally are a
> seasonal value. Heat pump efficiencies > 100% are possible as they move more heat
> than the electrical energy consumed."

This **confirms** the plan's use of `HSMainSystemEfficiency > 100` as the `IsHeatPump` flag.
For a COP 3 heat pump: 300% efficiency is entered. For `HSMainSystemEfficiency = 350`, this
means COP 3.5 — the override value in the plan is correct.

However: In the SEAI data, **heat pump systems are recorded as Electricity fuel type**
with efficiency > 100. When simulating a heat pump retrofit:
```python
# DEAP-correct heat pump override:
overrides = {
    "MainSpaceHeatingFuel": "Electricity",  # NOT "Electricity - Heat Pump"
    "HSMainSystemEfficiency": 350,           # COP 3.5 — correct
    "IsHeatPump": 1,                          # binary flag
    "MainWaterHeatingFuel": "Electricity",
    "WHMainSystemEff": 350,
}
# The label "Electricity - Heat Pump" does NOT appear in SEAI data
# Heat pumps appear as Electricity + high efficiency
```

---

## 7. New Insight: PV & Solar — What Makes `FirstEnerProdDelivered` Leaky

From DEAP Manual Appendix M1, the formula for PV electricity produced is:
```
E_PV = 0.80 × kWp × S × Z_PV
```
Where:
- `kWp` = peak power (surveyor enters from manufacturer data)
- `S` = solar radiation from Table H2 (orientation + pitch lookup)
- `Z_PV` = overshading factor from Table H3

`FirstEnerProdDelivered` is the **output** of this formula. It is not a raw measurement.
The inputs to this formula (`kWp`, orientation, pitch, overshading) are what an assessor
records. **The kWh value is DEAP's calculation, not the assessor's measurement.**

This means the plan is correct to:
1. Include `FirstEnergyType_Description` (solar PV vs wind — a survey input)
2. Exclude `FirstEnerProdDelivered` (a DEAP formula output)

The honest limitation: when simulating "add solar PV", the model will infer the benefit
from the *type* descriptor only, not the kWh. This underestimates the benefit. Document it.

---

## 8. Why `TempAdjustment` and `TempFactorMultiplier` Are DEAP Outputs

From the DEAP Manual:
- **`TempAdjustment`** — tabulated from DEAP Table 4e based on `HeatSystemResponseCat`
  and `HeatSystemControlCat`. It is a **lookup table output**, not a survey measurement.
- **`TempFactorMultiplier`** — from DEAP Table 2 (hot water cylinder temperature factor).
  It depends on cylinder thermostat + timer settings, computed by DEAP.

Both of these appear in the 118-col pipeline's features as SHAP rank #6 and #10 respectively.
Their high SHAP importance is **because they encode control/responsiveness information that
DEAP has already processed** — they're shortcut representations of DEAP intermediate steps.

Removing them forces the model to learn this from the raw inputs (`HeatSystemControlCat`,
`HeatSystemResponseCat`, `CylinderStat`), which is the honest approach.

---

## 9. What Changes Must Be Made to `new_implemention.md`

### Confirmed Changes (from DEAP manual reading):

| # | Change | Reason |
|---|---|---|
| 1 | `MainSpaceHeatingFuel` override = `"Electricity"` not `"Electricity - Heat Pump"` | SEAI data records heat pumps as Electricity + high efficiency |
| 2 | `UValueFloor` retrofit target = **0.21** not 0.15 | DEAP Table S2 TGD L 2011 floor standard is 0.21 |
| 3 | Triple glazing target = **1.50** not 1.40 | DEAP Table S9: triple, 2010+ = 1.50 W/m²K |
| 4 | Age band cut-off: **2014+** is Band L (not 2016+) | DEAP Table S1 |
| 5 | Add `NoOfChimneys` to features to add — it's a DEAP survey item affecting infiltration losses | DEAP Appendix S13, Table S12 |
| 6 | Document pre-retrofit U-value baselines using DEAP Table S3/S4/S9 values | More defensible than assumed values |
| 7 | Consider also adding `PermeabilityTestResult` and `PermeabilityTest` — blower door test result is a direct DEAP input affecting infiltration | DEAP Appendix S, ventilation section |

### New Retrofit Measures to Add (from DEAP Manual):

The DEAP manual appendix S12 (Improvement measures) and the DEAP software "Graphical Energy
Representation" (Figure 9) shows that space heating dominates BER for most dwellings.
Two additional measures worth adding:

**Measure I: Airtightness Improvement**
```python
# DEAP Appendix S, Ventilation:
# "Draught proofing on windows and doors" → PercentageDraughtStripped
# "Count of chimneys, open flues" → NoOfFansAndVents / NoOfChimneys
overrides = {
    "PercentageDraughtStripped": 100,  # Fully draught stripped
    "NoOfChimneys": max(0, current - 1),  # Block one chimney
}
```

**Measure J: Cylinder Insulation Upgrade**
```python
# DEAP Manual S11.1: cylinder insulation from Table S11
# Age band I/J: 35mm foam; Band K: 50mm foam
# Already captured by WaterStorageVolume + InsulationThickness if included
overrides = {
    "InsulationThickness": 50,        # 50mm foam insulation
}
```

---

## 10. Final Definitive Feature Decision Table

### Features for the ~55-column "honest" dataset:

| Status | Feature | Source | In 45-col? | In 118-col? |
|---|---|---|---|---|
| ✅ Keep | `DwellingTypeDescr` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `Year_of_Construction` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | All 5 U-values | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | All 5 Areas | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `NoStoreys`, `LivingAreaPercent` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `HSMainSystemEfficiency`, `HSSupplHeatFraction`, `HSSupplSystemEff` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `WHMainSystemEff` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `SupplSHFuel`, `SupplWHFuel` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `NoOfFansAndVents`, `VentilationMethod`, `PercentageDraughtStripped` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `NoOfSidesSheltered`, `HeatSystemControlCat`, `HeatSystemResponseCat` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `NoCentralHeatingPumps`, `CHBoilerThermostatControlled`, `OBBoilerThermostatControlled` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `ThermalBridgingFactor`, `ThermalMassCategory`, `PredominantRoofType` | Survey/lookup Stage 1 | ✓ | ✓ |
| ✅ Keep | `CylinderStat`, `CombinedCylinder`, `WarmAirHeatingSystem`, `UndergroundHeating` | Survey Stage 1 | ✓ | ✓ |
| ✅ Keep | `StructureType`, `SuspendedWoodenFloor` | Survey Stage 1 | ✓ | ✓ |
| ✅ Add | `MainSpaceHeatingFuel` | Survey Stage 1 | ✗ | ✓ |
| ✅ Add | `MainWaterHeatingFuel` | Survey Stage 1 | ✗ | ✓ |
| ✅ Add | `SolarHotWaterHeating` | Survey Stage 1 | ✗ | ✓ |
| ✅ Add | `LowEnergyLightingPercent` | Survey Stage 1 (Appendix L) | ✗ | ✓ |
| ✅ Add | `GroundFloorAreasq_m` | Survey Stage 1 | ✗ | ✓ |
| ✅ Add | `HESSchemeUpgrade` | Survey Stage 1 | ✗ | ✓ |
| ✅ Add | `FirstEnergyType_Description` | Survey Stage 1 (type, not kWh) | ✗ | ✓ |
| ✅ Add | `NoOfChimneys` | Survey Stage 1 (Table S12) | ✗ | ✓ |
| ✅ Engineer | `IsHeatPump` (`HSMainSystemEfficiency > 100`) | Derived | New | New |
| ✅ Engineer | `HasSolarWaterHeating` (`SolarHotWaterHeating == 'YES'`) | Derived | New | New |
| ✅ Engineer | `HasRoofInsulation` (`UValueRoof <= 0.16`) | Derived | New | New |
| ✅ Engineer | `HasWallInsulation` (`UValueWall <= 0.37`) | Derived | New | New |
| ✅ Engineer | `HasDoubleGlazing` (`UValueWindow <= 2.0`) | Derived | New | New |
| ✅ Engineer | `WindowToWallRatio` (`WindowArea / WallArea`) | Derived from Stage 1 | New | ✓ |
| ✅ Engineer | `FabricHeatLossProxy` (`Σ U×A`) | Derived from Stage 1 | New | ✓ (re-compute) |
| ✅ Engineer | `FabricHeatLossPerM2` (`FabricHeatLossProxy / GroundFloorAreasq_m`) | Derived | New | ✓ (re-compute) |
| ✅ Engineer | `AgeBand` (DEAP Table S1 bins, 2014+ = Band L) | Derived | New | ✓ |
| ❌ Remove | `DistributionLosses` | Stage 2 borderline | ✓ | ✓ |
| ❌ Remove | `HSEffAdjFactor` | Stage 2 calculated | ✓ | ✓ |
| ❌ Remove | `WHEffAdjFactor` | Stage 2 calculated | ✓ | ✓ |
| ❌ Remove | `SHRenewableResources` | Stage 2 calculated | ✓ | ✓ |
| ❌ Remove | `WHRenewableResources` | Stage 2 calculated | ✓ | ✓ |
| ❌ Remove | `FirstEnerProdDelivered` | Stage 2 PV formula output | ✗ | ✓ |
| ❌ Remove | `SecondEnerProdDelivered` | Stage 2 formula output | ✗ | ✓ |
| ❌ Remove | `TempAdjustment` | Stage 2 Table 4e lookup | ✗ | ✓ |
| ❌ Remove | `TempFactorMultiplier` | Stage 2 Table 2 lookup | ✗ | ✓ |
| ❌ Remove | `DeclaredLossFactor` | Stage 2 calculated from insulation | ✗ | ✓ |
| ❌ Remove | `CO2Rating`, `MPCDERValue`, `CPC`, `EPC`, etc. | Stage 3 final outputs | Already removed | — |

---

## 11. Honest Expected Performance After Changes

After removing all Stage 2 outputs and adding the 8 new survey inputs:

| Metric | Prediction | Reasoning |
|---|---|---|
| **LightGBM R²** | **0.93–0.96** | Top SHAP features (`FabricHeatLossPerM2`, `FirstEnerProdDelivered`, `TempAdjustment`) account for ~30% of SHAP importance; removing them lowers R² from 0.9913 but the model still sees all raw building inputs |
| **Random Forest R²** | 0.89–0.93 | Consistent ~3pp gap observed in 45-col run |
| **Ridge R²** | 0.83–0.87 | Structural ceiling — adding more raw inputs helps slightly vs. the 45-col run |
| **Retrofit simulation accuracy** | High — mechanism unchanged | Override-and-predict mechanism doesn't depend on which features we excluded; it still sees UValues and heating efficiency |

---

## 12. Summary: Key Differences from `new_implemention.md`

| Aspect | `new_implemention.md` | This Document's Correction |
|---|---|---|
| Floor insulation target | 0.15 W/m²K | **0.21 W/m²K** (DEAP TGD L 2011) |
| Triple glazing target | 1.40 W/m²K | **1.50 W/m²K** (DEAP Table S9) |
| Heat pump fuel string | `"Electricity - Heat Pump"` | **`"Electricity"`** (how SEAI data actually records it) |
| Age band cut for new builds | `2016+` | **`2014+`** (DEAP Table S1 Band L) |
| Features to add | 7 features | **8 features** (add `NoOfChimneys`) |
| `DistributionLosses` classification | "DEAP-calculated output" | **More nuanced: survey-informed lookup** — but still remove |
| `TempAdjustment` and `TempFactorMultiplier` | Correctly identified as leaky | Confirmed by DEAP Appendix S analysis |
| `FirstEnerProdDelivered` | Correctly identified as leaky | Confirmed by DEAP Appendix M1 formula: `0.80 × kWp × S × Z_PV` |
| Retrofit measures count | 8 | Can add 2 more: **Airtightness** and **Cylinder insulation** |
| Pre-retrofit baselines | Assumptions | Now grounded in DEAP Tables S3/S4/S9 |
