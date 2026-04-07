"""
01_clean_and_prepare.py
=======================
Comprehensive data cleaning and feature engineering pipeline for the
Irish SEAI BER Residential dataset (1,354,360 rows, 211 columns).

Memory-safe: two-pass chunked processing (never loads full 1.4 GB into RAM).
  Pass 1 — compute global imputation statistics from a 200K-row sample.
  Pass 2 — full dataset: clean, impute, engineer features, write parquet.

Output: outputs/clean_data.parquet
        outputs/cleaning_report.txt

Author: RetroFit project
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_PATH    = r"C:\Users\achal\Downloads\BER_Residential_Data.csv"
OUTPUT_DIR   = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
PARQUET_PATH = OUTPUT_DIR / "clean_data.parquet"
REPORT_PATH  = OUTPUT_DIR / "cleaning_report.txt"
CHUNK_SIZE   = 50_000

TARGET  = 'BerRating'
BER_MIN = 0.0       # Paper 1 domain threshold
BER_MAX = 2000.0    # Paper 1 domain threshold
YOC_MIN = 1700      # Earliest plausible Irish construction
YOC_MAX = 2026      # Assessment year upper bound

# ─────────────────────────────────────────────────────────────
# COLUMNS TO KEEP  (all leakage/admin/near-null columns dropped)
# ─────────────────────────────────────────────────────────────
#
# DROPPED categories:
#  - Pure DEAP leakage outputs: EnergyRating, CO2Rating, MPCDERValue,
#    TotalDeliveredEnergy, Delivered*, PrimaryEnergy*, CO2* columns,
#    RER, RenewEPnren, RenewEPren, CPC, EPC, DistributionLosses
#  - Admin/identifiers: DateOfAssessment, SA_Code,
#    prob_smarea_error_0corr, prob_smarea_error_100corr
#  - Near-null (>80% missing): ApertureArea, ZeroLossCollectorEff,
#    CollectorHeatLossCoEff, AnnualSolarRadiation, OvershadingFactor,
#    SolarStorageVolume, VolumeOfPreHeatStore, ElectricityConsumption,
#    all gsd* columns, all CHP* columns, all Third/SecondBoiler*,
#    SolarSpaceHeatingSystem, TotalPrimaryEnergyFact, TotalCO2Emissions,
#    all Third* wall columns, all ThirdEner*/SecondEnerConsumed* columns,
#    FirstEnerConsumedComment, SecondEnerConsumedComment, SolarHeatFraction
#  - Free text comment fields: First/Second/ThirdEnerProdComment,
#    First/SecondWallDescription, ThirdWallDescription
#  - Redundant ID columns: FirstEnergyTypeId, SecondEnergyTypeId,
#    FirstWallTypeId, SecondWallTypeId
#  - DEAP total-contribution outputs: FirstPartLTotalContribution,
#    SecondPartLTotalContribution, FirstEnerConsumedDelivered,
#    SecondEnerConsumedDelivered (and their ConvFactor/CO2 siblings)

KEEP_COLS = [
    # ── TARGET ────────────────────────────────────────────────
    'BerRating',

    # ── ADMINISTRATIVE CONTEXT ────────────────────────────────
    'CountyName',               # 55 Irish counties
    'DwellingTypeDescr',        # Detached / Semi / Apartment etc.
    'TypeofRating',             # Existing / Final / Provisional
    'Year_of_Construction',     # Building vintage
    'PurposeOfRating',          # Sale / Grant / Social housing etc.
    'HESSchemeUpgrade',         # Home Energy Scheme upgrade flag
    'TGDLEdition',              # Building Reg edition (proxy for standards era)
    'MultiDwellingMPRN',        # Multi-dwelling indicator

    # ── GEOMETRY ──────────────────────────────────────────────
    'GroundFloorAreasq_m',      # Total floor area (m²) — key size predictor
    'NoStoreys',
    'LivingAreaPercent',
    'GroundFloorArea',
    'GroundFloorHeight',
    'FirstFloorArea',
    'FirstFloorHeight',
    'SecondFloorArea',
    'SecondFloorHeight',
    'ThirdFloorArea',
    'ThirdFloorHeight',
    'RoomInRoofArea',

    # ── FABRIC AREAS ──────────────────────────────────────────
    'WallArea',
    'RoofArea',
    'FloorArea',
    'WindowArea',
    'DoorArea',
    'PredominantRoofTypeArea',

    # ── U-VALUES (thermal performance) ────────────────────────
    'UValueWall',
    'UValueRoof',
    'UValueFloor',
    'UValueWindow',
    'UvalueDoor',               # Note: lowercase 'v' — exact column name
    'GroundFloorUValue',
    'ThermalBridgingFactor',    # y-value

    # ── FABRIC TYPE / CONSTRUCTION ────────────────────────────
    'ThermalMassCategory',
    'StructureType',
    'SuspendedWoodenFloor',
    'PredominantRoofType',
    'FirstWallType_Description',
    'FirstWallArea',
    'FirstWallUValue',
    'FirstWallIsSemiExposed',
    'FirstWallAgeBandId',
    'SecondWallType_Description',   # Null when only one wall type (~67%)
    'SecondWallArea',
    'SecondWallUValue',
    'SecondWallIsSemiExposed',
    'SecondWallAgeBandId',

    # ── AIRTIGHTNESS / VENTILATION ────────────────────────────
    'NoOfChimneys',
    'NoOfOpenFlues',
    'NoOfFansAndVents',
    'NoOfFluelessGasFires',
    'DraftLobby',
    'VentilationMethod',
    'FanPowerManuDeclaredValue',
    'HeatExchangerEff',
    'PercentageDraughtStripped',
    'NoOfSidesSheltered',
    'PermeabilityTest',
    'PermeabilityTestResult',
    'TempAdjustment',

    # ── HEATING SYSTEM ────────────────────────────────────────
    'MainSpaceHeatingFuel',
    'MainWaterHeatingFuel',
    'HSMainSystemEfficiency',
    'HSEffAdjFactor',
    'HSSupplHeatFraction',
    'HSSupplSystemEff',
    'WHMainSystemEff',
    'WHEffAdjFactor',
    'SupplSHFuel',
    'SupplWHFuel',
    'SHRenewableResources',
    'WHRenewableResources',
    'HeatSystemControlCat',
    'HeatSystemResponseCat',
    'NoCentralHeatingPumps',
    'CHBoilerThermostatControlled',
    'NoOilBoilerHeatingPumps',
    'OBBoilerThermostatControlled',
    'OBPumpInsideDwelling',
    'NoGasBoilerHeatingPumps',
    'WarmAirHeatingSystem',
    'UndergroundHeating',

    # ── HOT WATER CYLINDER GROUP (MNAR: null = combi boiler) ──
    # These 15 columns are NULL for the 51.19% of dwellings that
    # have a combi boiler (no separate hot water cylinder).
    # We handle with has_hw_cylinder binary flag + domain-fill.
    'StorageLosses',
    'ManuLossFactorAvail',
    'SolarHotWaterHeating',
    'ElecImmersionInSummer',
    'CombiBoiler',
    'KeepHotFacility',
    'WaterStorageVolume',
    'DeclaredLossFactor',
    'TempFactorUnadj',
    'TempFactorMultiplier',
    'InsulationType',
    'InsulationThickness',
    'PrimaryCircuitLoss',
    'CombiBoilerAddLoss',
    'ElecConsumpKeepHot',

    # ── SOLAR / WATER HEATING CONTROLS ────────────────────────
    'CylinderStat',
    'CombinedCylinder',
    'SWHPumpSolarPowered',
    'ChargingBasisHeatConsumed',

    # ── LIGHTING ──────────────────────────────────────────────
    'LowEnergyLightingPercent',

    # ── RENEWABLE ENERGY PRODUCTION ───────────────────────────
    'FirstEnergyType_Description',  # Solar PV / Wind / Thermal etc.
    'FirstEnerProdDelivered',       # kWh delivered by primary renewable
    'SecondEnergyType_Description',
    'SecondEnerProdDelivered',
]

# MNAR group — which sub-lists get which imputation
MNAR_CATEGORICAL = [
    'StorageLosses', 'ManuLossFactorAvail', 'SolarHotWaterHeating',
    'ElecImmersionInSummer', 'CombiBoiler', 'KeepHotFacility',
    'InsulationType', 'PrimaryCircuitLoss',
]
MNAR_NUMERIC = [
    'WaterStorageVolume', 'DeclaredLossFactor', 'TempFactorUnadj',
    'TempFactorMultiplier', 'InsulationThickness',
    'CombiBoilerAddLoss', 'ElecConsumpKeepHot',
]

# String columns that need whitespace stripping
STR_COLS = [
    'CountyName', 'DwellingTypeDescr', 'TypeofRating', 'PurposeOfRating',
    'MultiDwellingMPRN', 'ThermalMassCategory', 'StructureType',
    'SuspendedWoodenFloor', 'PredominantRoofType',
    'FirstWallType_Description', 'FirstWallIsSemiExposed',
    'SecondWallType_Description', 'SecondWallIsSemiExposed',
    'DraftLobby', 'VentilationMethod', 'PermeabilityTest',
    'CHBoilerThermostatControlled', 'OBBoilerThermostatControlled',
    'OBPumpInsideDwelling', 'WarmAirHeatingSystem', 'UndergroundHeating',
    'StorageLosses', 'ManuLossFactorAvail', 'SolarHotWaterHeating',
    'ElecImmersionInSummer', 'CombiBoiler', 'KeepHotFacility',
    'InsulationType', 'PrimaryCircuitLoss',
    'CylinderStat', 'CombinedCylinder', 'SWHPumpSolarPowered',
    'ChargingBasisHeatConsumed',
    'MainSpaceHeatingFuel', 'MainWaterHeatingFuel',
    'FirstEnergyType_Description', 'SecondEnergyType_Description',
]

# Placeholder values that are effectively null
PLACEHOLDER_MAP = {
    'StructureType':       ['Please select'],
    'PredominantRoofType': ['Select Roof Type', ''],
}


# ─────────────────────────────────────────────────────────────
# AGE BAND CONFIGURATION  (DEAP-aligned vintage brackets)
# ─────────────────────────────────────────────────────────────
AGE_BINS   = [0, 1900, 1930, 1950, 1967, 1978, 1983, 1994, 2000, 2005, 2011, 2016, 9999]
AGE_LABELS = ['Pre1900', '1900-1929', '1930-1949', '1950-1966',
              '1967-1977', '1978-1982', '1983-1993', '1994-1999',
              '2000-2004', '2005-2010', '2011-2015', '2016+']


# ─────────────────────────────────────────────────────────────
# HELPER: strip + replace placeholders in a single chunk
# ─────────────────────────────────────────────────────────────
def _preprocess_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and replace known placeholder strings with NaN."""
    for col in STR_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()

    for col, bad_vals in PLACEHOLDER_MAP.items():
        if col in df.columns:
            df[col] = df[col].replace(bad_vals, np.nan)

    return df


# ─────────────────────────────────────────────────────────────
# PASS 1 — Compute imputation statistics from a ~200K sample
# ─────────────────────────────────────────────────────────────
def compute_imputation_stats():
    """
    Read up to 200K rows (4 chunks), compute per-column median and mode
    for use as global imputation fill values in Pass 2.
    Returns (medians: dict, modes: dict).
    """
    print("=" * 60)
    print("PASS 1: Computing imputation statistics (200K sample)...")
    print("=" * 60)

    sample_parts = []
    rows = 0
    for chunk in pd.read_csv(
        DATA_PATH, usecols=KEEP_COLS, chunksize=CHUNK_SIZE,
        low_memory=False, encoding='latin-1'
    ):
        chunk = _preprocess_strings(chunk)
        # Only filter BerRating here so statistics reflect valid rows
        chunk = chunk[
            (chunk[TARGET] >= BER_MIN) & (chunk[TARGET] <= BER_MAX)
        ].copy()
        sample_parts.append(chunk)
        rows += len(chunk)
        if rows >= 200_000:
            break

    sample = pd.concat(sample_parts, ignore_index=True)
    del sample_parts

    medians: dict = {}
    modes:   dict = {}

    for col in KEEP_COLS:
        if col == TARGET or col not in sample.columns:
            continue
        if sample[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            val = sample[col].dropna().median()
            medians[col] = float(val) if pd.notna(val) else 0.0
        else:
            mode_series = sample[col].dropna().mode()
            modes[col] = mode_series.iloc[0] if len(mode_series) > 0 else 'Unknown'

    del sample
    print(f"  Stats computed from {rows:,} rows "
          f"({len(medians)} numeric, {len(modes)} categorical columns).\n")
    return medians, modes


# ─────────────────────────────────────────────────────────────
# CLEAN CHUNK — applied to every chunk in Pass 2
# ─────────────────────────────────────────────────────────────
def clean_chunk(
    df: pd.DataFrame,
    medians: dict,
    modes:   dict,
) -> pd.DataFrame:
    """
    Full cleaning + imputation + feature engineering for one chunk.
    Returns cleaned DataFrame (may be shorter than input after filtering).
    """

    # 1. STRING PREPROCESSING
    df = _preprocess_strings(df)

    # 2. OUTLIER FILTERING
    # BerRating: Paper 1 domain threshold [0, 2000]
    df = df[(df[TARGET] >= BER_MIN) & (df[TARGET] <= BER_MAX)].copy()
    if df.empty:
        return df

    # Year of construction: impossible values
    df = df[
        (df['Year_of_Construction'] >= YOC_MIN) &
        (df['Year_of_Construction'] <= YOC_MAX)
    ].copy()
    if df.empty:
        return df

    # Wall area of exactly 0 with non-zero floor area is suspicious;
    # keep them — LightGBM handles edge cases gracefully and zero-area
    # apartments are valid (e.g. ground floor mid-terrace).

    # 3. MNAR HOT WATER CYLINDER GROUP
    # Create binary flag BEFORE imputation so model knows which rows
    # had no cylinder (the nulls are structurally meaningful).
    df['has_hw_cylinder'] = df[MNAR_CATEGORICAL[0]].notna().astype(np.int8)

    # Categorical MNAR → 'No_cylinder' (domain-meaningful category)
    for col in MNAR_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna('No_cylinder')

    # Numeric MNAR → 0.0 (no cylinder = no storage losses/volume)
    for col in MNAR_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 4. SECOND WALL: null means only one wall construction type
    for col in ['SecondWallArea', 'SecondWallUValue', 'SecondWallAgeBandId']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    for col in ['SecondWallType_Description', 'SecondWallIsSemiExposed']:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # 5. RENEWABLE ENERGY: null means no system installed → 0 delivered
    for col in ['FirstEnerProdDelivered', 'SecondEnerProdDelivered']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 6. GENERAL IMPUTATION for remaining nulls
    for col in df.columns:
        if col in (TARGET, 'has_hw_cylinder'):
            continue
        if not df[col].isna().any():
            continue
        if col in medians:
            df[col] = df[col].fillna(medians[col])
        elif col in modes:
            df[col] = df[col].fillna(modes[col])

    # 7. FEATURE ENGINEERING
    # ── Window-to-Wall Ratio (Paper 2) ────────────────────────
    df['WindowToWallRatio'] = np.where(
        df['WallArea'] > 0,
        df['WindowArea'] / df['WallArea'],
        0.0
    ).astype(np.float32)

    # ── Fabric Heat Loss Proxy (UA-sum, kWh-style) ─────────────
    # Sum of U-value × Area for each opaque/transparent element.
    # Highly predictive of space-heating demand without being a leakage.
    df['FabricHeatLossProxy'] = (
        df['UValueWall']    * df['WallArea']   +
        df['UValueRoof']    * df['RoofArea']   +
        df['UValueFloor']   * df['FloorArea']  +
        df['UValueWindow']  * df['WindowArea'] +
        df['UvalueDoor']    * df['DoorArea']
    ).astype(np.float32)

    # Normalise by total floor area → comparable across dwelling sizes
    df['FabricHeatLossPerM2'] = np.where(
        df['GroundFloorAreasq_m'] > 0,
        df['FabricHeatLossProxy'] / df['GroundFloorAreasq_m'],
        df['FabricHeatLossProxy']
    ).astype(np.float32)

    # ── Weighted Average Wall U-Value ─────────────────────────
    total_wall_area = df['FirstWallArea'] + df['SecondWallArea']
    df['AvgWallUValue'] = np.where(
        total_wall_area > 0,
        (df['FirstWallUValue'] * df['FirstWallArea'] +
         df['SecondWallUValue'] * df['SecondWallArea']) / total_wall_area,
        df['UValueWall']
    ).astype(np.float32)

    # ── Total Computed Floor Area (sum of individual floors) ───
    df['TotalFloorArea_computed'] = (
        df['GroundFloorArea'] +
        df['FirstFloorArea']  +
        df['SecondFloorArea'] +
        df['ThirdFloorArea']
    ).astype(np.float32)

    # ── Building Age Band (DEAP vintage brackets) ──────────────
    df['AgeBand'] = pd.cut(
        df['Year_of_Construction'],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True
    ).astype(str)

    # ── Is Heat Pump? ─────────────────────────────────────────
    # Retrofit intervention flag — useful for SHAP analysis.
    if 'MainSpaceHeatingFuel' in df.columns:
        df['IsHeatPump'] = df['MainSpaceHeatingFuel'].str.contains(
            'Heat Pump|heat pump|HEAT PUMP', case=False, na=False
        ).astype(np.int8)

    # ── Has Solar Water Heating? ───────────────────────────────
    if 'SolarHotWaterHeating' in df.columns:
        df['HasSolarWaterHeating'] = (
            df['SolarHotWaterHeating'].isin(['YES', 'Yes', 'yes'])
        ).astype(np.int8)

    # ── Has Roof Insulation? (proxy: UValueRoof <= 0.16) ──────
    df['HasRoofInsulation'] = (df['UValueRoof'] <= 0.16).astype(np.int8)

    # ── Has Cavity Wall Insulation? (proxy: UValueWall <= 0.37) ──
    df['HasWallInsulation'] = (df['UValueWall'] <= 0.37).astype(np.int8)

    # ── Has Double/Triple Glazing? (proxy: UValueWindow <= 2.0) ─
    df['HasDoubleGlazing'] = (df['UValueWindow'] <= 2.0).astype(np.int8)

    # 8. DTYPE OPTIMISATION (reduce memory for parquet)
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes('int64').columns:
        df[col] = df[col].astype(np.int32)

    return df


# ─────────────────────────────────────────────────────────────
# PASS 2 — Full chunked cleaning → parquet
# ─────────────────────────────────────────────────────────────
def run_cleaning_pass(medians: dict, modes: dict) -> dict:
    """
    Read the full CSV in 50K-row chunks, clean each, write to parquet
    using PyArrow's streaming writer (avoids loading entire dataset).
    Returns stats dict for the report.
    """
    print("=" * 60)
    print("PASS 2: Full dataset cleaning → parquet...")
    print("=" * 60)

    writer = None
    schema = None
    stats  = {
        'rows_read':    0,
        'rows_written': 0,
        'rows_dropped_ber':  0,
        'rows_dropped_yoc':  0,
        'chunks':       0,
    }
    t0 = time.time()

    for chunk in pd.read_csv(
        DATA_PATH, usecols=KEEP_COLS, chunksize=CHUNK_SIZE,
        low_memory=False, encoding='latin-1'
    ):
        stats['chunks'] += 1
        n_raw = len(chunk)
        stats['rows_read'] += n_raw

        # Count outliers before cleaning
        n_ber_bad = int(((chunk[TARGET] < BER_MIN) | (chunk[TARGET] > BER_MAX)).sum())
        yoc_valid = (chunk['Year_of_Construction'] >= YOC_MIN) & \
                    (chunk['Year_of_Construction'] <= YOC_MAX)
        n_yoc_bad = int((~yoc_valid).sum())

        stats['rows_dropped_ber'] += n_ber_bad
        stats['rows_dropped_yoc'] += n_yoc_bad

        chunk = clean_chunk(chunk, medians, modes)

        if chunk.empty:
            continue

        # Convert to Arrow table and write
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(
                str(PARQUET_PATH), schema, compression='snappy'
            )

        writer.write_table(table)
        stats['rows_written'] += len(chunk)

        # Progress update every 10 chunks (~500K rows)
        if stats['chunks'] % 10 == 0:
            elapsed = time.time() - t0
            rate    = stats['rows_read'] / elapsed
            print(f"  Chunk {stats['chunks']:3d} | "
                  f"Read: {stats['rows_read']:>10,} | "
                  f"Written: {stats['rows_written']:>10,} | "
                  f"Speed: {rate:,.0f} rows/s")

    if writer:
        writer.close()

    elapsed = time.time() - t0
    print(f"\n  Finished in {elapsed:.1f}s  |  "
          f"{stats['rows_written']:,} rows written to {PARQUET_PATH}\n")
    return stats


# ─────────────────────────────────────────────────────────────
# VALIDATION — quick sanity check on the output parquet
# ─────────────────────────────────────────────────────────────
def validate_parquet(stats: dict) -> None:
    """Load the output parquet and print a cleaning summary report."""
    print("=" * 60)
    print("VALIDATION: Checking output parquet...")
    print("=" * 60)

    df = pd.read_parquet(PARQUET_PATH)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("BER DATASET — CLEANING REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Original rows read   : {stats['rows_read']:>12,}")
    report_lines.append(f"Dropped (BerRating)  : {stats['rows_dropped_ber']:>12,}  "
                        f"[outside {BER_MIN}–{BER_MAX}]")
    report_lines.append(f"Dropped (Year)       : {stats['rows_dropped_yoc']:>12,}  "
                        f"[outside {YOC_MIN}–{YOC_MAX}]")
    report_lines.append(f"Clean rows written   : {stats['rows_written']:>12,}")
    report_lines.append(f"Retention rate       : "
                        f"{stats['rows_written']/stats['rows_read']*100:.2f}%")
    report_lines.append("")
    report_lines.append(f"Columns in parquet   : {df.shape[1]}")
    report_lines.append("")

    report_lines.append("── TARGET STATISTICS (BerRating) ──────────────────────")
    ber = df[TARGET]
    report_lines.append(f"  Mean   : {ber.mean():.2f} kWh/m²/yr")
    report_lines.append(f"  Median : {ber.median():.2f} kWh/m²/yr")
    report_lines.append(f"  Std    : {ber.std():.2f}")
    report_lines.append(f"  Min    : {ber.min():.2f}")
    report_lines.append(f"  Max    : {ber.max():.2f}")
    report_lines.append(f"  Pct A-B (≤100): {(ber <= 100).mean()*100:.1f}%")
    report_lines.append(f"  Pct C  (101-200): {((ber > 100) & (ber <= 200)).mean()*100:.1f}%")
    report_lines.append(f"  Pct D  (201-300): {((ber > 200) & (ber <= 300)).mean()*100:.1f}%")
    report_lines.append(f"  Pct E+ (>300): {(ber > 300).mean()*100:.1f}%")
    report_lines.append("")

    report_lines.append("── NULL CHECK (post-cleaning) ────────────────────────")
    null_counts = df.isna().sum()
    null_cols   = null_counts[null_counts > 0]
    if null_cols.empty:
        report_lines.append("  No nulls remaining. ✓")
    else:
        for col, cnt in null_cols.items():
            report_lines.append(f"  {col:<40s}: {cnt:,} ({cnt/len(df)*100:.2f}%)")
    report_lines.append("")

    report_lines.append("── ENGINEERED FEATURES ───────────────────────────────")
    eng_cols = ['WindowToWallRatio', 'FabricHeatLossProxy', 'FabricHeatLossPerM2',
                'AvgWallUValue', 'TotalFloorArea_computed', 'AgeBand',
                'IsHeatPump', 'HasSolarWaterHeating',
                'HasRoofInsulation', 'HasWallInsulation', 'HasDoubleGlazing',
                'has_hw_cylinder']
    for col in eng_cols:
        if col in df.columns:
            if df[col].dtype in [np.float32, np.float64]:
                report_lines.append(
                    f"  {col:<40s} mean={df[col].mean():.4f}"
                )
            else:
                report_lines.append(
                    f"  {col:<40s} values={df[col].value_counts().to_dict()}"
                )
    report_lines.append("")

    report_lines.append("── DWELLING TYPE DISTRIBUTION ────────────────────────")
    if 'DwellingTypeDescr' in df.columns:
        for dtype, cnt in df['DwellingTypeDescr'].value_counts().items():
            report_lines.append(f"  {dtype:<35s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")
    report_lines.append("")

    report_lines.append("── HEATING FUEL DISTRIBUTION ─────────────────────────")
    if 'MainSpaceHeatingFuel' in df.columns:
        for fuel, cnt in df['MainSpaceHeatingFuel'].value_counts().head(10).items():
            report_lines.append(f"  {fuel:<35s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")
    report_lines.append("")

    report_lines.append("── BUILDING AGE BAND DISTRIBUTION ───────────────────")
    if 'AgeBand' in df.columns:
        for band, cnt in df['AgeBand'].value_counts().sort_index().items():
            report_lines.append(f"  {band:<20s}: {cnt:>8,} ({cnt/len(df)*100:.1f}%)")

    report_lines.append("")
    report_lines.append("── COLUMN LIST ───────────────────────────────────────")
    for i, col in enumerate(sorted(df.columns), 1):
        report_lines.append(f"  {i:3d}. {col}")

    report_text = "\n".join(report_lines)

    print(report_text)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved to {REPORT_PATH}")
    del df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_start = time.time()

    print("\n" + "=" * 60)
    print("  BER DATASET — DATA CLEANING PIPELINE")
    print("  Input  :", DATA_PATH)
    print("  Output :", PARQUET_PATH)
    print("  Chunk  :", f"{CHUNK_SIZE:,} rows")
    print("=" * 60 + "\n")

    # Pass 1: statistics
    medians, modes = compute_imputation_stats()

    # Pass 2: full clean
    stats = run_cleaning_pass(medians, modes)

    # Validate and report
    validate_parquet(stats)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("Done. Clean dataset is ready at:", PARQUET_PATH)
