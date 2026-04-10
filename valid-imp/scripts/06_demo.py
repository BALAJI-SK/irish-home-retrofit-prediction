"""
06_demo.py
==========
Interactive XAI Retrofit Scenario Planner — CLI demo.

Lets a user enter their home's properties, predicts the current BER,
then lets them select a retrofit measure and see the predicted improvement
with SHAP-based explanation of what drove the change.

Usage:
  python 06_demo.py               — interactive CLI
  python 06_demo.py --streamlit   — launch Streamlit web UI (if installed)
  python 06_demo.py --example     — run with a pre-filled example home (non-interactive)

Requirements:
  - outputs/lgbm_model.pkl        (from 02_train_model.py)
  - config/retrofit_measures.json
  - config/uvalue_standards.json
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

from cli_logger import setup_script_logging

import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
OUTPUT_DIR    = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
setup_script_logging(OUTPUT_DIR / f"{Path(__file__).stem}.log")
CONFIG_DIR    = BASE_DIR / "config"
LGBM_PATH     = OUTPUT_DIR / "lgbm_model.pkl"
MEASURES_PATH = CONFIG_DIR / "retrofit_measures.json"
STANDARDS_PATH = CONFIG_DIR / "uvalue_standards.json"

# BER grades
BER_GRADES = [
    (0,   25,  'A1'), (25,  50,  'A2'), (50,  75,  'A3'),
    (75,  100, 'B1'), (100, 125, 'B2'), (125, 150, 'B3'),
    (150, 175, 'C1'), (175, 200, 'C2'), (200, 225, 'C3'),
    (225, 260, 'D1'), (260, 300, 'D2'),
    (300, 340, 'E1'), (340, 380, 'E2'),
    (380, 450, 'F'),
    (450, 9999, 'G'),
]


def ber_grade(kwh: float) -> str:
    for lo, hi, grade in BER_GRADES:
        if lo <= kwh < hi:
            return grade
    return 'G'


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
def load_artifacts():
    if not LGBM_PATH.exists():
        print(f"ERROR: Model not found at {LGBM_PATH}")
        print("Run 02_train_model.py first.")
        sys.exit(1)

    with open(LGBM_PATH, 'rb') as f:
        artifact = pickle.load(f)

    with open(MEASURES_PATH, 'r') as f:
        measures = json.load(f)

    with open(STANDARDS_PATH, 'r') as f:
        standards = json.load(f)

    return artifact, measures, standards


# ─────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────
def prepare_X(home_dict: dict, artifact: dict) -> pd.DataFrame:
    """Convert a home feature dict to an encoded DataFrame for prediction."""
    encoders = artifact['encoders']
    cat_cols = artifact['cat_cols']
    num_cols = artifact['num_cols']

    df = pd.DataFrame([home_dict])

    for col in cat_cols:
        if col in df.columns:
            try:
                df[col] = encoders[col].transform(df[[col]]).astype(np.float32)
            except Exception:
                df[col] = -1.0

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)

    # Ensure all expected columns are present (fill missing with 0 or -1)
    all_cols = cat_cols + num_cols
    for col in all_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[all_cols]


def predict_ber(home_dict: dict, artifact: dict) -> float:
    X = prepare_X(home_dict, artifact)
    log_pred = artifact['model'].predict(X)
    return float(np.expm1(log_pred).clip(min=0)[0])


def recompute_derived(home: dict) -> dict:
    """Recompute FabricHeatLossProxy, FabricHeatLossPerM2 etc. after U-value overrides."""
    u = {k: home.get(k, 0) for k in
         ['UValueWall', 'UValueRoof', 'UValueFloor', 'UValueWindow', 'UvalueDoor']}
    a = {k: home.get(k, 0) for k in
         ['WallArea', 'RoofArea', 'FloorArea', 'WindowArea', 'DoorArea']}

    proxy = (u['UValueWall'] * a['WallArea'] +
             u['UValueRoof'] * a['RoofArea'] +
             u['UValueFloor'] * a['FloorArea'] +
             u['UValueWindow'] * a['WindowArea'] +
             u['UvalueDoor'] * a['DoorArea'])

    home['FabricHeatLossProxy'] = float(proxy)
    gfa = home.get('GroundFloorAreasq_m', 1.0)
    home['FabricHeatLossPerM2'] = float(proxy / gfa) if gfa > 0 else float(proxy)
    home['WindowToWallRatio'] = (
        float(a['WindowArea'] / a['WallArea']) if a['WallArea'] > 0 else 0.0
    )
    home['HasRoofInsulation'] = int(u['UValueRoof']  <= 0.16)
    home['HasWallInsulation'] = int(u['UValueWall']  <= 0.37)
    home['HasDoubleGlazing']  = int(u['UValueWindow'] <= 2.0)
    return home


def apply_retrofit(home: dict, measure: dict) -> dict:
    """Apply retrofit overrides and optionally recompute derived features."""
    retro = dict(home)
    retro.update(measure['overrides'])
    if measure.get('recompute_derived', False):
        retro = recompute_derived(retro)
    return retro


def get_shap_explanation(home_dict: dict, retro_dict: dict,
                         artifact: dict) -> list:
    """
    Compute SHAP delta between before and after retrofit.
    Returns top-5 drivers as list of (feature, delta_shap).
    """
    try:
        explainer = shap.TreeExplainer(artifact['model'])
        X_before  = prepare_X(home_dict, artifact)
        X_after   = prepare_X(retro_dict, artifact)

        sv_before = explainer.shap_values(X_before)
        sv_after  = explainer.shap_values(X_after)

        if isinstance(sv_before, list):
            sv_before = sv_before[0]
        if isinstance(sv_after, list):
            sv_after = sv_after[0]

        delta = sv_after.flatten() - sv_before.flatten()
        feature_names = X_before.columns.tolist()

        top5 = sorted(
            zip(feature_names, delta.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        return top5
    except Exception as e:
        return [("(SHAP unavailable)", str(e))]


# ─────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────
def print_report(home: dict, base_ber: float, retro_ber: float,
                 measure_name: str, top_drivers: list) -> None:
    saving     = base_ber - retro_ber
    saving_pct = 100.0 * saving / base_ber if base_ber > 0 else 0.0
    W = 65

    print()
    print('=' * W)
    print(f"  RETROFIT SCENARIO REPORT — {measure_name}")
    print('=' * W)
    print(f"  Home:  {home.get('DwellingTypeDescr','?')}, "
          f"built {int(home.get('Year_of_Construction', 0))}")
    print()
    print(f"  Current BER:      {ber_grade(base_ber)} ({base_ber:.1f} kWh/m2/yr)")
    print(f"  After retrofit:   {ber_grade(retro_ber)} ({retro_ber:.1f} kWh/m2/yr)")
    print(f"  Improvement:      -{saving:.1f} kWh/m2/yr  (-{saving_pct:.1f}%)")
    print()

    print("  Top drivers of improvement (SHAP attribution):")
    print(f"  {'Feature':<40} {'SHAP Delta':>12}")
    print("  " + "-" * 55)
    for feat, delta in top_drivers:
        bar = '|' * min(int(abs(delta) * 200), 20)
        direction = 'improvement' if delta < 0 else 'worsening'
        print(f"  {feat:<40} {delta:>+12.4f}  {bar}")
    print('=' * W)
    print()


# ─────────────────────────────────────────────────────────────
# USER INPUT COLLECTION
# ─────────────────────────────────────────────────────────────
DWELLING_TYPES = [
    'Detached house',
    'Semi-detached house',
    'End of terrace house',
    'Mid-terrace house',
    'Apartment',
    'Top-floor apartment',
    'Mid-floor apartment',
    'Ground-floor apartment',
    'Basement dwelling',
    'Maisonette',
]

HEATING_FUELS = [
    'Mains Gas',
    'Oil',
    'Electricity',
    'Electricity - Heat Pump',
    'Solid Fuel - Coal',
    'Solid Fuel - Peat',
    'Wood Pellets',
    'LPG',
]


def prompt(text: str, default=None, cast=str):
    """Prompt user with a default value."""
    if default is not None:
        text = f"{text} [{default}]: "
    else:
        text = f"{text}: "
    val = input(text).strip()
    if val == '' and default is not None:
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        return default


def prompt_choice(text: str, options: list, default: int = 0) -> str:
    """Display numbered list and prompt user to choose."""
    print(f"\n{text}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i:2d}. {opt}")
    choice = prompt(f"Enter number (1-{len(options)})", default=str(default+1))
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except (ValueError, TypeError):
        pass
    return options[default]


def collect_user_inputs() -> dict:
    """Collect home features from user via CLI prompts."""
    print("\n" + "=" * 65)
    print("  ENTER YOUR HOME DETAILS")
    print("  (Press Enter to accept default values)")
    print("=" * 65)

    dwelling_type = prompt_choice("Dwelling type", DWELLING_TYPES, default=1)
    year_built    = prompt("Year of construction", default=1972, cast=int)
    floor_area    = prompt("Ground floor area (m2)", default=110.0, cast=float)
    no_storeys    = prompt("Number of storeys", default=2, cast=int)

    print("\n-- U-values (thermal conductance, W/m2K) --")
    print("  (Typical uninsulated: Wall=1.2, Roof=0.5, Floor=0.6, Window=4.0)")
    u_wall   = prompt("UValueWall   (W/m2K)", default=0.55, cast=float)
    u_roof   = prompt("UValueRoof   (W/m2K)", default=0.35, cast=float)
    u_floor  = prompt("UValueFloor  (W/m2K)", default=0.45, cast=float)
    u_window = prompt("UValueWindow (W/m2K)", default=2.8, cast=float)
    u_door   = prompt("UValueDoor   (W/m2K)", default=3.0, cast=float)

    print("\n-- Areas (m2) --")
    wall_area   = prompt("Wall area   (m2)", default=150.0, cast=float)
    roof_area   = prompt("Roof area   (m2)", default=60.0,  cast=float)
    floor_area2 = prompt("Floor area  (m2)", default=60.0,  cast=float)
    window_area = prompt("Window area (m2)", default=20.0,  cast=float)
    door_area   = prompt("Door area   (m2)", default=4.0,   cast=float)

    print("\n-- Heating system --")
    main_fuel = prompt_choice("Main space heating fuel", HEATING_FUELS, default=1)
    eff_map   = {
        'Mains Gas': 92, 'Oil': 86, 'Electricity': 100,
        'Electricity - Heat Pump': 350,
        'Solid Fuel - Coal': 60, 'Solid Fuel - Peat': 60,
        'Wood Pellets': 80, 'LPG': 86,
    }
    default_eff = eff_map.get(main_fuel, 90)
    hs_eff = prompt(f"Main system efficiency (%)", default=default_eff, cast=float)

    solar_water = prompt("Solar water heating? (YES/NO)", default="NO")
    led_pct     = prompt("Low energy lighting percent (%)", default=30, cast=int)

    # Derived features
    proxy = (u_wall * wall_area + u_roof * roof_area + u_floor * floor_area2 +
             u_window * window_area + u_door * door_area)
    gfa   = floor_area if floor_area > 0 else 1.0

    home = {
        'DwellingTypeDescr':      dwelling_type,
        'Year_of_Construction':   year_built,
        'GroundFloorAreasq_m':    floor_area,
        'NoStoreys':              no_storeys,
        'UValueWall':             u_wall,
        'UValueRoof':             u_roof,
        'UValueFloor':            u_floor,
        'UValueWindow':           u_window,
        'UvalueDoor':             u_door,
        'WallArea':               wall_area,
        'RoofArea':               roof_area,
        'FloorArea':              floor_area2,
        'WindowArea':             window_area,
        'DoorArea':               door_area,
        'MainSpaceHeatingFuel':   main_fuel,
        'HSMainSystemEfficiency': hs_eff,
        'MainWaterHeatingFuel':   main_fuel,
        'WHMainSystemEff':        hs_eff,
        'SolarHotWaterHeating':   solar_water.upper(),
        'LowEnergyLightingPercent': led_pct,
        'FabricHeatLossProxy':    proxy,
        'FabricHeatLossPerM2':    proxy / gfa,
        'WindowToWallRatio':      window_area / wall_area if wall_area > 0 else 0.0,
        'IsHeatPump':             int(hs_eff > 100),
        'HasSolarWaterHeating':   int(str(solar_water).upper() == 'YES'),
        'HasRoofInsulation':      int(u_roof <= 0.16),
        'HasWallInsulation':      int(u_wall <= 0.37),
        'HasDoubleGlazing':       int(u_window <= 2.0),
        # Fill remaining with sensible defaults (model handles unknown via -1 encoding)
        'CountyName':             'Dublin',
        'TypeofRating':           'Existing dwelling',
        'PurposeOfRating':        'Marketed sale',
        'TGDLEdition':            0,
        'MultiDwellingMPRN':      'No',
        'LivingAreaPercent':      20.0,
        'GroundFloorArea':        floor_area,
        'GroundFloorHeight':      2.4,
        'FirstFloorArea':         floor_area,
        'FirstFloorHeight':       2.4,
        'SecondFloorArea':        0.0,
        'SecondFloorHeight':      0.0,
        'ThirdFloorArea':         0.0,
        'ThirdFloorHeight':       0.0,
        'RoomInRoofArea':         0.0,
        'PredominantRoofTypeArea':roof_area,
        'GroundFloorUValue':      u_floor,
        'ThermalBridgingFactor':  0.15,
        'ThermalMassCategory':    'Medium',
        'StructureType':          'Masonry',
        'SuspendedWoodenFloor':   'No',
        'PredominantRoofType':    'Pitched',
        'FirstWallType_Description': 'Cavity',
        'FirstWallArea':          wall_area,
        'FirstWallUValue':        u_wall,
        'FirstWallIsSemiExposed': 'No',
        'FirstWallAgeBandId':     5,
        'SecondWallType_Description': 'None',
        'SecondWallArea':         0.0,
        'SecondWallUValue':       0.0,
        'SecondWallIsSemiExposed': 'None',
        'SecondWallAgeBandId':    0,
        'NoOfChimneys':           1,
        'NoOfOpenFlues':          0,
        'NoOfFansAndVents':       2,
        'NoOfFluelessGasFires':   0,
        'DraftLobby':             'No',
        'VentilationMethod':      'Natural',
        'FanPowerManuDeclaredValue': 0.0,
        'HeatExchangerEff':       0.0,
        'PercentageDraughtStripped': 50.0,
        'NoOfSidesSheltered':     2,
        'PermeabilityTest':       'No',
        'PermeabilityTestResult': 10.0,
        'HSSupplHeatFraction':    0.0,
        'HSSupplSystemEff':       0.0,
        'SupplSHFuel':            'None',
        'SupplWHFuel':            'None',
        'HeatSystemControlCat':   2,
        'HeatSystemResponseCat':  2,
        'NoCentralHeatingPumps':  1,
        'CHBoilerThermostatControlled': 'Yes',
        'NoOilBoilerHeatingPumps':0,
        'OBBoilerThermostatControlled': 'No',
        'OBPumpInsideDwelling':   'No',
        'NoGasBoilerHeatingPumps':0,
        'WarmAirHeatingSystem':   'No',
        'UndergroundHeating':     'No',
        'StorageLosses':          'No_cylinder',
        'ManuLossFactorAvail':    'No_cylinder',
        'ElecImmersionInSummer':  'No_cylinder',
        'CombiBoiler':            'Yes',
        'KeepHotFacility':        'No_cylinder',
        'WaterStorageVolume':     0.0,
        'DeclaredLossFactor':     0.0,
        'TempFactorUnadj':        0.0,
        'InsulationType':         'No_cylinder',
        'InsulationThickness':    0.0,
        'PrimaryCircuitLoss':     'No_cylinder',
        'CombiBoilerAddLoss':     0.0,
        'ElecConsumpKeepHot':     0.0,
        'CylinderStat':           'No',
        'CombinedCylinder':       'No',
        'SWHPumpSolarPowered':    'No',
        'ChargingBasisHeatConsumed': 'No',
        'FirstEnergyType_Description': 'None',
        'HESSchemeUpgrade':       'No',
        'has_hw_cylinder':        0,
        'AvgWallUValue':          u_wall,
        'TotalFloorArea_computed':floor_area,
        'AgeBand':                '1967-1977' if 1967 <= year_built <= 1977 else
                                  '1978-1982' if 1978 <= year_built <= 1982 else
                                  '1983-1993' if 1983 <= year_built <= 1993 else
                                  '1994-1999' if 1994 <= year_built <= 1999 else
                                  '2000-2004' if 2000 <= year_built <= 2004 else
                                  '2005-2010' if 2005 <= year_built <= 2010 else
                                  '2011-2015' if 2011 <= year_built <= 2015 else
                                  '2016+'     if year_built >= 2016 else
                                  'Pre1900'   if year_built < 1900 else
                                  '1900-1929' if year_built < 1930 else
                                  '1930-1949' if year_built < 1950 else
                                  '1950-1966',
    }
    return home


# ─────────────────────────────────────────────────────────────
# EXAMPLE HOME (non-interactive mode)
# ─────────────────────────────────────────────────────────────
def get_example_home() -> dict:
    """Return a pre-filled 1972 semi-detached Dublin home."""
    u_wall, u_roof, u_floor, u_window, u_door = 0.55, 0.35, 0.45, 2.8, 3.0
    wall_area, roof_area, floor_area = 140.0, 55.0, 55.0
    window_area, door_area, gfa = 18.0, 3.5, 110.0

    proxy = (u_wall*wall_area + u_roof*roof_area + u_floor*floor_area +
             u_window*window_area + u_door*door_area)

    return {
        'DwellingTypeDescr':       'Semi-detached house',
        'Year_of_Construction':    1972,
        'GroundFloorAreasq_m':     gfa,
        'NoStoreys':               2,
        'UValueWall':              u_wall,
        'UValueRoof':              u_roof,
        'UValueFloor':             u_floor,
        'UValueWindow':            u_window,
        'UvalueDoor':              u_door,
        'WallArea':                wall_area,
        'RoofArea':                roof_area,
        'FloorArea':               floor_area,
        'WindowArea':              window_area,
        'DoorArea':                door_area,
        'MainSpaceHeatingFuel':    'Oil',
        'HSMainSystemEfficiency':  86.0,
        'MainWaterHeatingFuel':    'Oil',
        'WHMainSystemEff':         86.0,
        'SolarHotWaterHeating':    'No_cylinder',
        'LowEnergyLightingPercent': 20,
        'FabricHeatLossProxy':     proxy,
        'FabricHeatLossPerM2':     proxy / gfa,
        'WindowToWallRatio':       window_area / wall_area,
        'IsHeatPump':              0,
        'HasSolarWaterHeating':    0,
        'HasRoofInsulation':       0,
        'HasWallInsulation':       0,
        'HasDoubleGlazing':        1,
        'CountyName':              'Dublin',
        'TypeofRating':            'Existing dwelling',
        'PurposeOfRating':         'Marketed sale',
        'TGDLEdition':             0,
        'MultiDwellingMPRN':       'No',
        'LivingAreaPercent':       20.0,
        'GroundFloorArea':         gfa,
        'GroundFloorHeight':       2.4,
        'FirstFloorArea':          gfa,
        'FirstFloorHeight':        2.4,
        'SecondFloorArea':         0.0,
        'SecondFloorHeight':       0.0,
        'ThirdFloorArea':          0.0,
        'ThirdFloorHeight':        0.0,
        'RoomInRoofArea':          0.0,
        'PredominantRoofTypeArea': roof_area,
        'GroundFloorUValue':       u_floor,
        'ThermalBridgingFactor':   0.15,
        'ThermalMassCategory':     'Medium',
        'StructureType':           'Masonry',
        'SuspendedWoodenFloor':    'No',
        'PredominantRoofType':     'Pitched',
        'FirstWallType_Description': 'Cavity',
        'FirstWallArea':           wall_area,
        'FirstWallUValue':         u_wall,
        'FirstWallIsSemiExposed':  'No',
        'FirstWallAgeBandId':      5,
        'SecondWallType_Description': 'None',
        'SecondWallArea':          0.0,
        'SecondWallUValue':        0.0,
        'SecondWallIsSemiExposed': 'None',
        'SecondWallAgeBandId':     0,
        'NoOfChimneys':            1,
        'NoOfOpenFlues':           0,
        'NoOfFansAndVents':        2,
        'NoOfFluelessGasFires':    0,
        'DraftLobby':              'No',
        'VentilationMethod':       'Natural',
        'FanPowerManuDeclaredValue': 0.0,
        'HeatExchangerEff':        0.0,
        'PercentageDraughtStripped': 50.0,
        'NoOfSidesSheltered':      2,
        'PermeabilityTest':        'No',
        'PermeabilityTestResult':  10.0,
        'HSSupplHeatFraction':     0.0,
        'HSSupplSystemEff':        0.0,
        'SupplSHFuel':             'None',
        'SupplWHFuel':             'None',
        'HeatSystemControlCat':    2,
        'HeatSystemResponseCat':   2,
        'NoCentralHeatingPumps':   1,
        'CHBoilerThermostatControlled': 'Yes',
        'NoOilBoilerHeatingPumps': 0,
        'OBBoilerThermostatControlled': 'No',
        'OBPumpInsideDwelling':    'No',
        'NoGasBoilerHeatingPumps': 0,
        'WarmAirHeatingSystem':    'No',
        'UndergroundHeating':      'No',
        'StorageLosses':           'No_cylinder',
        'ManuLossFactorAvail':     'No_cylinder',
        'ElecImmersionInSummer':   'No_cylinder',
        'CombiBoiler':             'Yes',
        'KeepHotFacility':         'No_cylinder',
        'WaterStorageVolume':      0.0,
        'DeclaredLossFactor':      0.0,
        'TempFactorUnadj':         0.0,
        'InsulationType':          'No_cylinder',
        'InsulationThickness':     0.0,
        'PrimaryCircuitLoss':      'No_cylinder',
        'CombiBoilerAddLoss':      0.0,
        'ElecConsumpKeepHot':      0.0,
        'CylinderStat':            'No',
        'CombinedCylinder':        'No',
        'SWHPumpSolarPowered':     'No',
        'ChargingBasisHeatConsumed': 'No',
        'FirstEnergyType_Description': 'None',
        'HESSchemeUpgrade':        'No',
        'has_hw_cylinder':         0,
        'AvgWallUValue':           u_wall,
        'TotalFloorArea_computed': gfa,
        'AgeBand':                 '1967-1977',
    }


# ─────────────────────────────────────────────────────────────
# CLI MAIN
# ─────────────────────────────────────────────────────────────
def run_cli(artifact, measures, standards, example_mode: bool = False) -> None:
    print("\n" + "=" * 65)
    print("  IRISH HOME RETROFIT XAI SCENARIO PLANNER")
    print("  LightGBM + SHAP — 55-column honest dataset")
    print("=" * 65)

    # Step 1: Get home inputs
    if example_mode:
        print("\n[Example mode] Using pre-filled 1972 semi-detached Dublin home.")
        home = get_example_home()
    else:
        home = collect_user_inputs()

    # Step 2: Predict current BER
    base_ber = predict_ber(home, artifact)
    print(f"\nCurrent BER prediction: {ber_grade(base_ber)} ({base_ber:.1f} kWh/m2/yr)")

    # Step 3: Select retrofit measure
    measure_keys  = list(measures.keys())
    measure_names = [measures[k]['name'] for k in measure_keys]

    selected_key = prompt_choice("Select a retrofit measure to simulate",
                                 measure_names, default=0)
    sel_key = measure_keys[measure_names.index(selected_key)]
    measure = measures[sel_key]

    # Step 4: Apply retrofit and predict
    retro_home = apply_retrofit(home, measure)
    retro_ber  = predict_ber(retro_home, artifact)

    # Step 5: Get SHAP explanation
    print("\nComputing SHAP explanation...")
    top_drivers = get_shap_explanation(home, retro_home, artifact)

    # Step 6: Print report
    print_report(home, base_ber, retro_ber, measure['name'], top_drivers)

    # Step 7: Ask to try another measure
    if not example_mode:
        again = input("Try another retrofit measure? (y/n) [n]: ").strip().lower()
        if again == 'y':
            run_cli(artifact, measures, standards, example_mode=False)


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────
def run_streamlit(artifact, measures, standards) -> None:
    """Streamlit web UI for the XAI Scenario Planner."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        print("Falling back to CLI mode...")
        run_cli(artifact, measures, standards)
        return

    st.set_page_config(
        page_title="Irish Home Retrofit XAI Planner",
        page_icon="house",
        layout="wide"
    )

    st.title("Irish Home Retrofit XAI Scenario Planner")
    st.markdown(
        "Enter your home's properties, select a retrofit measure, "
        "and see the predicted BER improvement explained by SHAP feature attribution."
    )

    # ── Sidebar: home inputs ──────────────────────────────────
    st.sidebar.header("Your Home")

    dwelling_type = st.sidebar.selectbox("Dwelling type", DWELLING_TYPES, index=1)
    year_built    = st.sidebar.slider("Year of construction", 1700, 2025, 1972)
    floor_area    = st.sidebar.number_input("Ground floor area (m2)", 20.0, 500.0, 110.0)
    no_storeys    = st.sidebar.number_input("Number of storeys", 1, 5, 2)

    st.sidebar.subheader("U-values (W/m2K)")
    u_wall   = st.sidebar.slider("Wall U-value",   0.1, 2.5, 0.55, step=0.01)
    u_roof   = st.sidebar.slider("Roof U-value",   0.1, 1.5, 0.35, step=0.01)
    u_floor  = st.sidebar.slider("Floor U-value",  0.1, 1.5, 0.45, step=0.01)
    u_window = st.sidebar.slider("Window U-value", 0.5, 5.5, 2.80, step=0.1)

    st.sidebar.subheader("Heating system")
    main_fuel = st.sidebar.selectbox("Main heating fuel", HEATING_FUELS, index=1)
    hs_eff    = st.sidebar.slider("Heating efficiency (%)", 50, 400,
                                   350 if 'Heat Pump' in main_fuel else 86)
    led_pct   = st.sidebar.slider("LED lighting (%)", 0, 100, 20)

    # ── Main area: measure selection ──────────────────────────
    st.header("Retrofit Measure")
    measure_names = [v['name'] for v in measures.values()]
    sel_name = st.selectbox("Select a retrofit measure", measure_names)
    sel_key  = [k for k, v in measures.items() if v['name'] == sel_name][0]

    if st.button("Simulate Retrofit", type="primary"):
        # Build home dict
        wall_area, roof_area = floor_area * 1.3, floor_area * 0.6
        window_area, door_area, floor_area2 = floor_area * 0.18, 3.5, floor_area * 0.6
        proxy = (u_wall*wall_area + u_roof*roof_area + u_floor*floor_area2 +
                 u_window*window_area + 3.0*door_area)
        gfa = floor_area if floor_area > 0 else 1.0

        home = get_example_home()
        home.update({
            'DwellingTypeDescr':       dwelling_type,
            'Year_of_Construction':    year_built,
            'GroundFloorAreasq_m':     gfa,
            'NoStoreys':               no_storeys,
            'UValueWall':              u_wall,
            'UValueRoof':              u_roof,
            'UValueFloor':             u_floor,
            'UValueWindow':            u_window,
            'WallArea':                wall_area,
            'RoofArea':                roof_area,
            'FloorArea':               floor_area2,
            'WindowArea':              window_area,
            'MainSpaceHeatingFuel':    main_fuel,
            'HSMainSystemEfficiency':  float(hs_eff),
            'MainWaterHeatingFuel':    main_fuel,
            'WHMainSystemEff':         float(hs_eff),
            'LowEnergyLightingPercent': led_pct,
            'FabricHeatLossProxy':     proxy,
            'FabricHeatLossPerM2':     proxy / gfa,
            'IsHeatPump':              int(hs_eff > 100),
            'HasWallInsulation':       int(u_wall <= 0.37),
            'HasRoofInsulation':       int(u_roof <= 0.16),
            'HasDoubleGlazing':        int(u_window <= 2.0),
        })

        base_ber  = predict_ber(home, artifact)
        retro_home = apply_retrofit(home, measures[sel_key])
        retro_ber  = predict_ber(retro_home, artifact)
        saving     = base_ber - retro_ber
        saving_pct = 100.0 * saving / base_ber if base_ber > 0 else 0.0

        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("Current BER",
                    f"{ber_grade(base_ber)} ({base_ber:.1f} kWh/m2/yr)")
        col2.metric("After Retrofit",
                    f"{ber_grade(retro_ber)} ({retro_ber:.1f} kWh/m2/yr)",
                    delta=f"-{saving:.1f} kWh/m2/yr")
        col3.metric("Improvement", f"{saving_pct:.1f}%")

        # SHAP explanation
        st.subheader("What drove the improvement?")
        top_drivers = get_shap_explanation(home, retro_home, artifact)
        driver_df = pd.DataFrame(top_drivers, columns=['Feature', 'SHAP Delta'])
        driver_df['Direction'] = driver_df['SHAP Delta'].apply(
            lambda x: 'Improvement' if x < 0 else 'Worsening'
        )
        st.dataframe(driver_df.style.background_gradient(
            subset=['SHAP Delta'], cmap='RdYlGn_r'
        ))

        st.info(f"**Measure description:** {measures[sel_key]['description']}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Irish Home Retrofit XAI Scenario Planner'
    )
    parser.add_argument('--streamlit', action='store_true',
                        help='Launch Streamlit web UI')
    parser.add_argument('--example', action='store_true',
                        help='Run with pre-filled example home (non-interactive)')
    args = parser.parse_args()

    artifact, measures, standards = load_artifacts()

    if args.streamlit:
        run_streamlit(artifact, measures, standards)
    else:
        run_cli(artifact, measures, standards, example_mode=args.example)
