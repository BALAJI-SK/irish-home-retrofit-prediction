# Report 03 — Features

## Final Feature Set: 118 Features

After cleaning, the model is trained on 118 features (119 columns minus the target `BerRating`). They fall into the following groups:

---

## Feature Groups

### Group 1: Geographic and Administrative (8 features)

| Feature | Type | Description | Why Kept |
|---------|------|-------------|----------|
| `CountyName` | Categorical | 55 Irish counties | Regional climate differences (coastal vs inland) |
| `DwellingTypeDescr` | Categorical | Detached/Semi/Apartment etc. | Fundamental building form — biggest structural difference |
| `TypeofRating` | Categorical | Existing / Final / Provisional | Assessment type affects which features are available |
| `Year_of_Construction` | Integer | 1700–2026 | Building vintage is one of the strongest BER predictors |
| `PurposeOfRating` | Categorical | Sale / Grant / Social housing / etc. | Proxy for dwelling condition profile |
| `HESSchemeUpgrade` | Binary (0/1) | Whether a Home Energy Scheme upgrade was done | Directly relevant to retrofit status |
| `TGDLEdition` | Integer (0–4) | Building Regulation edition in force when built | Captures regulatory insulation standards of the era |
| `MultiDwellingMPRN` | Binary | Shared metering | Controls for apartment block effects |

---

### Group 2: Building Geometry (11 features)

| Feature | Type | Description |
|---------|------|-------------|
| `GroundFloorAreasq_m` | Float | Total floor area — key size predictor |
| `NoStoreys` | Integer | Number of floors |
| `LivingAreaPercent` | Float | % of floor area in the living zone |
| `GroundFloorArea` | Float | Ground floor area (m²) |
| `GroundFloorHeight` | Float | Ground floor ceiling height (m) |
| `FirstFloorArea` | Float | First floor area (m²) |
| `FirstFloorHeight` | Float | First floor ceiling height (m) |
| `SecondFloorArea` | Float | Second floor area (m²) |
| `SecondFloorHeight` | Float | Second floor ceiling height (m) |
| `ThirdFloorArea` | Float | Third floor area (mostly 0) |
| `ThirdFloorHeight` | Float | Third floor ceiling height (mostly 0) |
| `RoomInRoofArea` | Float | Room-in-roof area (m²) |

---

### Group 3: Building Fabric Areas (6 features)

| Feature | Type | Description |
|---------|------|-------------|
| `WallArea` | Float | Total exposed wall area (m²) |
| `RoofArea` | Float | Total roof area (m²) |
| `FloorArea` | Float | Total exposed floor area (m²) |
| `WindowArea` | Float | Total glazing area (m²) |
| `DoorArea` | Float | Total door area (m²) |
| `PredominantRoofTypeArea` | Float | Area of main roof section (m²) |

---

### Group 4: Thermal Performance — U-Values (7 features)

U-value (W/m²K) measures how quickly heat passes through a building element. Lower = better insulated.

| Feature | Typical Values | Description |
|---------|---------------|-------------|
| `UValueWall` | 0.15–2.1 | Wall thermal transmittance |
| `UValueRoof` | 0.13–2.3 | Roof thermal transmittance |
| `UValueFloor` | 0.12–0.73 | Floor thermal transmittance |
| `UValueWindow` | 1.2–4.8 | Window thermal transmittance |
| `UvalueDoor` | 0.94–3.0 | Door thermal transmittance |
| `GroundFloorUValue` | 0.0–0.41 | Ground floor specific value |
| `ThermalBridgingFactor` | 0.03–0.15 | y-value for thermal bridges |

---

### Group 5: Fabric Construction Type (11 features)

| Feature | Type | Description |
|---------|------|-------------|
| `ThermalMassCategory` | Categorical | Low / Medium-low / Medium / Medium-high / High |
| `StructureType` | Categorical | Masonry / Timber Frame / Insulated Concrete |
| `SuspendedWoodenFloor` | Categorical | No / Yes (Sealed) / Yes (Unsealed) |
| `PredominantRoofType` | Categorical | Pitch/Flat/Room-in-roof etc. |
| `FirstWallType_Description` | Categorical | 300mm Cavity / Solid Brick / Stone / Timber Frame etc. |
| `FirstWallArea` | Float | Area of main wall type (m²) |
| `FirstWallUValue` | Float | U-value of main wall type |
| `FirstWallIsSemiExposed` | Categorical | Whether the wall faces an unheated space |
| `FirstWallAgeBandId` | Integer | Age band of main wall (0–12) |
| `SecondWallType_Description` | Categorical | Second wall construction type ('None' if only one) |
| `SecondWallArea` | Float | Area of second wall type (0 if only one) |
| `SecondWallUValue` | Float | U-value of second wall type (0 if only one) |
| `SecondWallIsSemiExposed` | Categorical | |
| `SecondWallAgeBandId` | Integer | |

---

### Group 6: Airtightness and Ventilation (13 features)

| Feature | Type | Description |
|---------|------|-------------|
| `NoOfChimneys` | Float | Number of chimneys (infiltration) |
| `NoOfOpenFlues` | Float | Number of open flues |
| `NoOfFansAndVents` | Float | Number of fans and passive vents |
| `NoOfFluelessGasFires` | Float | Number of flueless gas fires |
| `DraftLobby` | Categorical | Whether there is a draft lobby (YES/NO) |
| `VentilationMethod` | Categorical | Natural / Whole house extract / MVHR / etc. |
| `FanPowerManuDeclaredValue` | Float | Mechanical vent fan power (W/l/s) |
| `HeatExchangerEff` | Float | MVHR heat recovery efficiency (%) |
| `PercentageDraughtStripped` | Float | % of openings draught-stripped (0–100) |
| `NoOfSidesSheltered` | Float | Number of sheltered sides (0–4) |
| `PermeabilityTest` | Categorical | Whether a blower door test was done |
| `PermeabilityTestResult` | Float | Air permeability m³/h/m² at 50 Pa |
| `TempAdjustment` | Float | Temperature adjustment for intermittent heating |

---

### Group 7: Heating System (22 features)

| Feature | Type | Description |
|---------|------|-------------|
| `MainSpaceHeatingFuel` | Categorical | Mains Gas / Oil / Electricity / Solid Fuel / LPG |
| `MainWaterHeatingFuel` | Categorical | Same options as above |
| `HSMainSystemEfficiency` | Float | Main heating system efficiency (%) — >100 = heat pump |
| `HSEffAdjFactor` | Float | Heating efficiency adjustment (0.95/1.0/1.02) |
| `HSSupplHeatFraction` | Float | Fraction of heat from supplementary system |
| `HSSupplSystemEff` | Float | Supplementary system efficiency |
| `WHMainSystemEff` | Float | Water heating system efficiency |
| `WHEffAdjFactor` | Float | Water heating efficiency adjustment |
| `SupplSHFuel` | Float | Supplementary space heating fuel ID |
| `SupplWHFuel` | Float | Supplementary water heating fuel ID |
| `SHRenewableResources` | Float | Space heating renewable resource type |
| `WHRenewableResources` | Float | Water heating renewable resource type |
| `HeatSystemControlCat` | Integer | Heating control category (0–3) |
| `HeatSystemResponseCat` | Integer | Heating response category (0–5) |
| `NoCentralHeatingPumps` | Integer | Number of central heating pumps |
| `CHBoilerThermostatControlled` | Categorical | CH boiler thermostat (YES/NO) |
| `NoOilBoilerHeatingPumps` | Integer | Number of oil boiler pumps |
| `OBBoilerThermostatControlled` | Categorical | Oil boiler thermostat (YES/NO) |
| `OBPumpInsideDwelling` | Categorical | Oil boiler pump location |
| `NoGasBoilerHeatingPumps` | Integer | Number of gas boiler pumps |
| `WarmAirHeatingSystem` | Categorical | Whether warm air system present |
| `UndergroundHeating` | Categorical | Whether underfloor heating present |

**Key feature:** `HSMainSystemEfficiency` — values > 100% indicate a heat pump (COP > 1). This distinguishes heat pumps from resistance electric heating even though both use 'Electricity' as fuel.

---

### Group 8: Hot Water Cylinder (16 features including engineered flag)

These 15 raw columns are the MNAR group — null when a combi boiler is present. Plus the engineered `has_hw_cylinder` flag.

| Feature | Type | Description |
|---------|------|-------------|
| `has_hw_cylinder` | Binary | **Engineered:** 1 = cylinder exists, 0 = combi boiler |
| `StorageLosses` | Categorical | Whether cylinder has storage losses |
| `ManuLossFactorAvail` | Categorical | Manufacturer loss factor available |
| `SolarHotWaterHeating` | Categorical | Solar water heating present (YES/NO/No_cylinder) |
| `ElecImmersionInSummer` | Categorical | Immersion in summer (YES/NO/No_cylinder) |
| `CombiBoiler` | Categorical | Combi boiler type (None/Instantaneous/Storage) |
| `KeepHotFacility` | Categorical | Keep-hot facility type |
| `WaterStorageVolume` | Float | Cylinder volume in litres (0 = no cylinder) |
| `DeclaredLossFactor` | Float | Cylinder declared loss factor |
| `TempFactorUnadj` | Float | Cylinder temperature factor |
| `TempFactorMultiplier` | Float | Temperature factor multiplier |
| `InsulationType` | Categorical | Cylinder insulation type |
| `InsulationThickness` | Float | Cylinder insulation thickness (mm) |
| `PrimaryCircuitLoss` | Categorical | Primary circuit heat loss description |
| `CombiBoilerAddLoss` | Float | Additional combi boiler losses |
| `ElecConsumpKeepHot` | Float | Electricity consumption for keep-hot |

---

### Group 9: Solar and Water Heating Controls (4 features)

| Feature | Type | Description |
|---------|------|-------------|
| `CylinderStat` | Categorical | Whether cylinder thermostat is fitted |
| `CombinedCylinder` | Categorical | Whether cylinder is combined type |
| `SWHPumpSolarPowered` | Categorical | Whether solar pump is solar-powered |
| `ChargingBasisHeatConsumed` | Categorical | Charging basis for heat consumed |

---

### Group 10: Lighting and Renewables (3 features)

| Feature | Type | Description |
|---------|------|-------------|
| `LowEnergyLightingPercent` | Float | % of fixed lights that are LED/low-energy (0–100) |
| `FirstEnergyType_Description` | Categorical | Primary renewable system type |
| `FirstEnerProdDelivered` | Float | kWh delivered by primary renewable system |
| `SecondEnergyType_Description` | Categorical | Secondary renewable system type |
| `SecondEnerProdDelivered` | Float | kWh delivered by secondary renewable system |

---

## Engineered Features (15 new columns added)

These features were created from existing columns to improve model accuracy. None of them constitute leakage because they are computed solely from DEAP inputs.

### 1. WindowToWallRatio
```
WindowToWallRatio = WindowArea / WallArea
```
**Source:** Ali et al. (2024, Paper 2) — demonstrated to be a strong predictor of energy performance in urban BER studies. Captures glazing intensity relative to building envelope.

**Mean value:** 0.300 (30% of wall area is window)

---

### 2. FabricHeatLossProxy
```
FabricHeatLossProxy = (UValueWall × WallArea) +
                      (UValueRoof × RoofArea) +
                      (UValueFloor × FloorArea) +
                      (UValueWindow × WindowArea) +
                      (UvalueDoor × DoorArea)
```
**Purpose:** Approximates the total UA-value (W/K) of the building envelope — the most fundamental determinant of space heating demand in building physics. Combining U-values with areas into a single number captures the interaction effect that a model might otherwise need many trees to discover separately.

**Mean value:** 176.2 W/K

---

### 3. FabricHeatLossPerM2
```
FabricHeatLossPerM2 = FabricHeatLossProxy / GroundFloorAreasq_m
```
**Purpose:** Normalises the heat loss by floor area, making large and small dwellings directly comparable. A large house and a small apartment with the same insulation standards will have similar `FabricHeatLossPerM2` even though their absolute losses differ.

**SHAP rank: #1 most important feature globally**  
**Mean value:** 1.55 W/K/m²

---

### 4. AvgWallUValue
```
AvgWallUValue = (FirstWallUValue × FirstWallArea + SecondWallUValue × SecondWallArea)
                / (FirstWallArea + SecondWallArea)
```
**Purpose:** Area-weighted average wall U-value across both wall types. More accurate than `UValueWall` (which is the primary wall's U-value) for dwellings with extensions or mixed construction.

**Mean value:** 0.633 W/m²K

---

### 5. TotalFloorArea_computed
```
TotalFloorArea_computed = GroundFloorArea + FirstFloorArea + SecondFloorArea + ThirdFloorArea
```
**Purpose:** Bottom-up calculation of total floor area from individual floor measurements. Cross-checks against `GroundFloorAreasq_m` and captures multi-storey detail.

**Mean value:** 115.3 m²

---

### 6. AgeBand
```
AgeBand = pd.cut(Year_of_Construction, bins=[...DEAP vintage brackets...])
```
**Bins:** Pre1900, 1900–1929, 1930–1949, 1950–1966, 1967–1977, 1978–1982, 1983–1993, 1994–1999, 2000–2004, 2005–2010, 2011–2015, 2016+

**Purpose:** DEAP uses these exact vintage brackets to assign default U-values when no measurement is provided. The age band is therefore a direct feature of the DEAP calculation and a strong categorical predictor.

---

### 7. IsHeatPump (Binary)
```
IsHeatPump = 1 if 'heat pump' in MainSpaceHeatingFuel (case-insensitive)
```
**Note:** In this dataset, all heat pumps are recorded as fuel = `'Electricity'`, not explicitly labelled "Heat Pump". The `HSMainSystemEfficiency > 100` already captures the distinction. This flag is kept for retrofit simulation logic.

**Value in data:** 0 for all rows (label not used in SEAI data — efficiency distinguishes HPs)

---

### 8–12. Binary Threshold Flags

| Feature | Rule | Purpose |
|---------|------|---------|
| `HasSolarWaterHeating` | `SolarHotWaterHeating == 'YES'` | Direct retrofit indicator |
| `HasRoofInsulation` | `UValueRoof ≤ 0.16` | Best-practice roof threshold |
| `HasWallInsulation` | `UValueWall ≤ 0.37` | Filled-cavity / external insulation threshold |
| `HasDoubleGlazing` | `UValueWindow ≤ 2.0` | Double-low-e glazing threshold |

These flags make the interaction between U-value levels explicit, reducing the number of trees needed to capture threshold effects.

---

## Feature Encoding

Categorical features are encoded using `OrdinalEncoder` with `handle_unknown='use_encoded_value'` (unknown categories → −1). LightGBM natively handles ordinal-encoded categoricals as unordered categories internally using its native categorical split logic when `categorical_feature` is specified, or by treating integer-encoded values as continuous (which still works well in practice given the tree structure).

All features are stored as `float32` / `int32` to minimise memory usage.
