"""
Configuration setup for Multiple Single Family Houses
Each building has its own HVAC and DER systems with varied parameters
"""

import logging
import os
import random
from bestopt.env.core.config_manager import ConfigurationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

# ===========================
# CONFIGURATION PARAMETERS
# ===========================

# Number of buildings to configure
N_BUILDINGS = 5

# Base parameters that will be varied for each building
BASE_PARAMS = {
    "hvac": {
        "base_cooling": 24.0,  # Will vary ±2°C
        "base_heating": 18.0,  # Will vary ±2°C
        "cooling_power_max": 4000.0,  # Will vary ±1000W
        "heating_power_max": 4000.0,  # Will vary ±1000W
        "chiller_capacity": 3500,  # Will vary ±1000W
        "chiller_cop": 4.5,  # Will vary ±0.5
    },
    "der": {
        "pv_capacity_kW": 10,  # Will vary ±5kW
        "battery_capacity_kWh": 8,  # Will vary ±2kWh
        "battery_initial_soc": 0.3,  # Will vary ±0.2
        "ev1_capacity_kWh": 60,  # Will vary ±20kWh
        "ev2_capacity_kWh": 40,  # Will vary ±10kWh
    },
    "electrical":
        {
        "lighting_daytime": 600,    # W
        "lighting_nighttime": 1800, # W
        "appliance_cooking": 2000,  # W
        "appliance_tv": 200,        # W
        "appliance_pc": 400,        # W
        "appliance_dishwashing": 1000,  # W
        }
}

# Variation ranges for parameters (as percentage or absolute)
VARIATION_RANGES = {
    "hvac": {
        "base_cooling": {"min": -2, "max": 2},  # ±2°C
        "base_heating": {"min": -2, "max": 2},  # ±2°C
        "cooling_power_max": {"min": -1000, "max": 1000},  # ±1000W
        "heating_power_max": {"min": -1000, "max": 1000},  # ±1000W
        "chiller_capacity": {"min": -1000, "max": 1000},  # ±1000W
        "chiller_cop": {"min": -0.5, "max": 0.5},  # ±0.5
    },
    "der": {
        "pv_capacity_kW": {"min": -5, "max": 5},  # ±5kW
        "battery_capacity_kWh": {"min": -2, "max": 2},  # ±2kWh
        "battery_initial_soc": {"min": -0.2, "max": 0.2},  # ±0.2
        "ev1_capacity_kWh": {"min": -20, "max": 20},  # ±20kWh
        "ev2_capacity_kWh": {"min": -10, "max": 10},  # ±10kWh
    },
    "electrical": {
        "lighting_daytime": {"min": -0.30, "max": 0.30, "mode": "percent"},
        "lighting_nighttime": {"min": -0.30, "max": 0.30, "mode": "percent"},
        "appliance_cooking": {"min": -0.20, "max": 0.20, "mode": "percent"},
        "appliance_tv": {"min": -0.20, "max": 0.20, "mode": "percent"},
        "appliance_pc": {"min": -0.20, "max": 0.20, "mode": "percent"},
        "appliance_dishwashing": {"min": -0.20, "max": 0.20, "mode": "percent"},
    }
}


# ===========================
# HELPER FUNCTIONS
# ===========================

def generate_varied_param(base_value, variation_range, rng, clamp=None):
    """
    Draw a varied parameter.
    variation_range: dict with keys:
        - min, max (floats)
        - mode: "abs" or "percent" (default "abs")
    rng: random.Random instance (per-building)
    clamp: optional (min_val, max_val) tuple to clamp final value
    """
    mode = variation_range.get("mode", "abs")
    delta = rng.uniform(variation_range["min"], variation_range["max"])
    if mode == "percent":
        varied = base_value * (1.0 + delta)
    else:
        varied = base_value + delta
    if clamp is not None:
        lo, hi = clamp
        varied = max(lo, min(hi, varied))
    return round(varied, 2)



def create_building_config(cm, building_id, cluster_id, building_seed=None):
    """Create configuration for a single building with its own systems."""

    rng = random.Random(building_seed)

    hvac_params = {}
    der_params = {}
    elec_params = {}

    for param, base_value in BASE_PARAMS["hvac"].items():
        hvac_params[param] = generate_varied_param(
            base_value,
            VARIATION_RANGES["hvac"][param],
            rng=rng
        )

    for param, base_value in BASE_PARAMS["der"].items():
        # Clamp SOC if this is the SOC param
        clamp = (0.1, 0.9) if param == "battery_initial_soc" else None
        der_params[param] = generate_varied_param(
            base_value,
            VARIATION_RANGES["der"][param],
            rng=rng,
            clamp=clamp
        )

    for param, base_value in BASE_PARAMS["electrical"].items():
        elec_params[param] = generate_varied_param(
            base_value,
            VARIATION_RANGES["electrical"][param],
            rng=rng
        )

    # Add building
    cm.add_building(
        cluster_id=cluster_id,
        building_id=building_id,
        parameters={
            "building_type": "single_family_home",
            "building_number": int(building_id.split("_")[1]),
        },
        thermal_zones=["zone0"],
        electrical_zones=["zone0"],
    )

    # Add thermal zone module (same model, same path for all buildings)
    cm.add_thermal_zone_module(
        building_id=building_id,
        zone_id="zone0",
        parameters={
            "model_args": {
                "para": {"Int_h": 8, "Ext_h": 14, "epochs": 20},
                "modeltype": "PI-modnn",
                "startday": 1,
                "trainday": 180,
                "testday": 1,
                "datapath": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
                "temp_unit": "C",
                "device": "cuda:0",
                "save_name": "SFH_1"  # Using same trained model for all buildings
            },
            "model_path": os.path.join(PROJECT_ROOT_PATH, "examples", "Saved", "SFH_1",
                                       "Trained_mdlEnco48_Deco96", "PI-modnn_180daysTest_on07-01.pth"),
            "scaler_path": os.path.join(PROJECT_ROOT_PATH, "examples", "Scaler", "SFH_1", "ModNN_scaler.pkl"),
            "historical_data_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
            "encoder_length": 48,
            "retrain": "Off",
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.env.modules.building.thermalzone.ThermalDynamicsModule"
    )

    # Add electrical zone module (same for all buildings)
    cm.add_electrical_zone_module(
        building_id=building_id,
        zone_id="zone0",
        parameters={
            "lighting": {
                "daytime": elec_params["lighting_daytime"],
                "nighttime": elec_params["lighting_nighttime"],
            },
            "appliance": {
                "cooking": elec_params["appliance_cooking"],
                "tv": elec_params["appliance_tv"],
                "pc": elec_params["appliance_pc"],
                "dishwashing": elec_params["appliance_dishwashing"],
            },
        },
        class_path="bestopt.env.modules.building.electricalzone.ElectricalDynamicModule"
    )

    # Create unique HVAC system for this building
    hvac_system_id = f"hvac_system_{building_id}"
    cm.add_system(
        cluster_id=cluster_id,
        system_id=hvac_system_id,
        system_type="hvac_systems",
        parameters={
            "system_name": f"FCU System for {building_id}",
            "system_config": {
                "fan": {"rated_flow_m3s": 1, "rated_power_W": 1000},
                "fan_ctrl": {"ctrl_type": "linear"},
                "coil": {"epsilon": 0.8},
                "pump": {"rated_flow_m3s": 0.005, "rated_power_W": 500},
                "chiller": {
                    "rated_capacity_W": hvac_params["chiller_capacity"],
                    "rated_cop": hvac_params["chiller_cop"]
                },
                "tower": {
                    "rated_capacity_W": hvac_params["chiller_capacity"],
                    "rated_fan_power_W": 2000,
                    "pump_power_per_flow": 1800,
                    "min_approach_C": 3.0,
                    "max_approach_C": 7.0
                }
            }
        },
        class_path="bestopt.env.modules.hvac.system.FCU.FCUModule"
    )

    # Create unique DER system for this building
    der_system_id = f"der_system_{building_id}"
    cm.add_system(
        cluster_id=cluster_id,
        system_id=der_system_id,
        system_type="der_systems",
        parameters={
            "system_name": f"PV-Battery-EV System for {building_id}",
            "system_config": {
                "pv": {"rated_capacity_kW": der_params["pv_capacity_kW"]},
                "bat": {
                    "rated_capacity_kWh": der_params["battery_capacity_kWh"],
                    "initial_soc": der_params["battery_initial_soc"],
                    "charge_speed": 0.25,
                    "discharge_speed": 0.5,
                    "charge_efficiency": 0.95,
                },
                "evs": [
                    {
                        "id": f"ev_tesla_{building_id}",
                        "rated_capacity_kWh": der_params["ev1_capacity_kWh"],
                        "initial_soc": 0.2,
                        "charge_speed": 0.25,
                        "discharge_speed": 0.5,
                        "charge_efficiency": 0.95,
                        "initially_connected": True
                    },
                    {
                        "id": f"ev_nissan_{building_id}",
                        "rated_capacity_kWh": der_params["ev2_capacity_kWh"],
                        "initial_soc": 0.8,
                        "charge_speed": 0.25,
                        "discharge_speed": 0.5,
                        "charge_efficiency": 0.95,
                        "initially_connected": False
                    }
                ]
            }
        },
        class_path="bestopt.env.modules.ders.system.der.DERModule"
    )

    # Assign systems to building (one-to-one mapping)
    cm.assign_system_to_buildings(hvac_system_id, [building_id])
    cm.assign_system_to_buildings(der_system_id, [building_id])

    # Create HVAC controller for this building's system
    hvac_controller_id = f"hvac_controller_{building_id}"
    cm.add_system_controller(
        controller_id=hvac_controller_id,
        system_id=hvac_system_id,
        parameters={
            "domain": "thermal",
            "type": "rule-based",
            "mode": "cooling",
            "precooling": {"degree": 0, "hours": 0},
            "base_cooling": hvac_params["base_cooling"],
            "base_heating": hvac_params["base_heating"],
            "deadband": 0.5,
            "cooling_power_max": hvac_params["cooling_power_max"],
            "heating_power_max": hvac_params["heating_power_max"]
        },
        class_path="bestopt.env.controllers.thermal.SupervisoryController"
    )

    # Create DER controller for this building's system
    der_controller_id = f"der_controller_{building_id}"
    cm.add_system_controller(
        controller_id=der_controller_id,
        system_id=der_system_id,
        parameters={
            "domain": "electrical",
            "type": "rule-based",
            "mode": "self_consumption",
            "bat_soc_min": 0.1,
            "bat_soc_max": 0.9,
            "ev_soc_min": 0.2,
            "ev_soc_target": 0.8,
            "ev_v2g_enabled": True,
            "max_grid_import": 10000,
            "max_grid_export": 5000,
            "system_config": {
                "pv": {"rated_capacity_kW": der_params["pv_capacity_kW"]},
                "bat": {
                    "rated_capacity_kWh": der_params["battery_capacity_kWh"],
                    "initial_soc": der_params["battery_initial_soc"]
                },
                "ev": {"rated_capacity_kWh": 5, "initial_soc": 0.3}
            }
        },
        class_path="bestopt.env.controllers.electrical.SupervisoryController"
    )

    return {
        "hvac_system": hvac_system_id,
        "der_system": der_system_id,
        "hvac_controller": hvac_controller_id,
        "der_controller": der_controller_id,
        "hvac_params": hvac_params,
        "der_params": der_params,
        "elec_params": elec_params,
    }

# ===========================
# MAIN CONFIGURATION
# ===========================

def main():
    # Initialize configuration manager
    cm = ConfigurationManager()
    logging.info(f"Starting configuration for {N_BUILDINGS} buildings")

    # Create cluster
    cluster_id = "residential_cluster_multi"
    cm.add_cluster(cluster_id, parameters={"location": "Syracuse, NY"})

    # Track all created entities
    all_buildings = []
    all_systems = []
    all_controllers = []
    building_configs = {}

    # Create configuration for each building
    for i in range(1, N_BUILDINGS + 1):
        building_id = f"SFH_{i}"
        logging.info(f"Configuring building {building_id}")

        # Use building index as seed for reproducible variation
        config = create_building_config(cm, building_id, cluster_id, building_seed=i * 100)

        all_buildings.append(building_id)
        all_systems.extend([config["hvac_system"], config["der_system"]])
        all_controllers.extend([config["hvac_controller"], config["der_controller"]])
        building_configs[building_id] = config

        # Log the varied parameters for this building
        logging.info(f"  HVAC: Cooling={config['hvac_params']['base_cooling']}°C, "
                     f"Chiller={config['hvac_params']['chiller_capacity']}W")
        logging.info(f"  DER: PV={config['der_params']['pv_capacity_kW']}kW, "
                     f"Battery={config['der_params']['battery_capacity_kWh']}kWh")

    # Add shared disturbances (all buildings use same weather, occupancy, price)
    cm.add_disturbance(
        "weather",
        parameters={
            "file_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "DIST", "weather", "weather.csv"),
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.env.disturbances.weather.WeatherModule"
    )

    cm.add_disturbance(
        "occupancy",
        parameters={
            "file_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "DIST", "occupancy", "occupancy.csv"),
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.env.disturbances.occupancy.OccupancyModule"
    )

    cm.add_disturbance(
        "price",
        parameters={},
        class_path="bestopt.env.disturbances.price.PriceModule"
    )

    # Add environment
    cm.add_environment(
        parameters={
            "resolution": 900,  # 15 minutes
            "duration": 86400 * 2,  # 3 days
            "enable_history": True,
            "logging_level": "INFO",
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.environment.BESTOptEnvironment"
    )

    # Select all components
    cm.select_cluster(cluster_id)
    cm.select_buildings(all_buildings)
    cm.select_systems(all_systems)

    # Assign controllers to their respective systems
    for building_id, config in building_configs.items():
        cm.select_controller_for_system(config["hvac_system"], config["hvac_controller"])
        cm.select_controller_for_system(config["der_system"], config["der_controller"])

    # Select disturbances and environment
    cm.select_disturbances(["weather", "occupancy", "price"])
    cm.select_environment()

    # Validate configuration
    warnings = cm.validate_configuration()
    if warnings:
        print(f"\n⚠ Warnings found: {warnings}")
    else:
        print(f"\n✓ Configuration validation passed for {N_BUILDINGS} buildings")

    # Print summary
    cm.print_summary()

    # Print parameter variations summary
    print(f"\n{'=' * 50}")
    print("Parameter Variations Summary")
    print(f"{'=' * 50}")
    for building_id, config in building_configs.items():
        print(f"\n{building_id}:")
        print(f"  HVAC Parameters:")
        print(f"    - Cooling Setpoint: {config['hvac_params']['base_cooling']}°C")
        print(f"    - Heating Setpoint: {config['hvac_params']['base_heating']}°C")
        print(f"    - Chiller Capacity: {config['hvac_params']['chiller_capacity']}W")
        print(f"    - Chiller COP: {config['hvac_params']['chiller_cop']}")
        print(f"  DER Parameters:")
        print(f"    - PV Capacity: {config['der_params']['pv_capacity_kW']}kW")
        print(f"    - Battery Capacity: {config['der_params']['battery_capacity_kWh']}kWh")
        print(f"    - Battery Initial SOC: {config['der_params']['battery_initial_soc']}")

    # Save configuration
    output_file = f"config_setup_{N_BUILDINGS}buildings.json"
    cm.save_final_configuration(output_file)
    print(f"\n✓ Saved simulation configuration as {output_file}")

    return cm, building_configs


if __name__ == "__main__":
    cm, configs = main()