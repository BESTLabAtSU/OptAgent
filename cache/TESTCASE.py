"""
Toy Test Case For Proof-of Concept, need to refine later.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TestCategory(Enum):
    """Categories of benchmark tests - 4 combinations"""
    SINGLE_AGENT_SINGLE_TOOL = "single_agent_single_tool"
    SINGLE_AGENT_MULTI_TOOL = "single_agent_multi_tool"
    MULTI_AGENT_SINGLE_TOOL = "multi_agent_single_tool"
    MULTI_AGENT_MULTI_TOOL = "multi_agent_multi_tool"


@dataclass
class ExpectedStep:
    """Definition of an expected step in the execution plan"""
    step_order: int
    agent_id: str
    required_tools: List[str]
    expected_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class BenchmarkTestCase:
    """Single test case definition"""
    test_id: str
    name: str
    category: TestCategory
    request: str
    expected_agents: List[str]
    expected_tools: List[str]
    expected_steps: List[ExpectedStep]
    alternative_valid_tools: List[List[str]] = field(default_factory=list)
    description: str = ""

    def get_expected_step_sequence(self) -> List[tuple]:
        """Return ordered list of (agent_id, tools) tuples"""
        sorted_steps = sorted(self.expected_steps, key=lambda s: s.step_order)
        return [(s.agent_id, s.required_tools) for s in sorted_steps]


# =============================================================================
# SINGLE AGENT SINGLE TOOL TEST CASES
# =============================================================================
#
SINGLE_AGENT_SINGLE_TOOL_CASES = [
    # ---- Configuration Tools ----
    BenchmarkTestCase(
        test_id="SAST_CFG_001",
        name="Config Create",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Initialize a new configuration named 'test_config'",
        expected_agents=["config_agent"],
        expected_tools=["config_create"],
        expected_steps=[
                ExpectedStep(
                    step_order=1,
                    agent_id="config_agent",
                    required_tools=["config_create"],
                    expected_parameters={
                        "config_create": {
                            "name": "test_config"
                        }
                    }
                )
            ],
        description="Create a new configuration"
    ),
    BenchmarkTestCase(
        test_id="SAST_CFG_002",
        name="Config Validate",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Validate the completeness and correctness of current configuration",
        expected_agents=["config_agent"],
        expected_tools=["config_validate"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="config_agent",
                         required_tools=["config_validate"],
                         expected_parameters={
                         }
                         )
        ],
        description="Validate existing configuration"
    ),
    BenchmarkTestCase(
        test_id="SAST_CFG_003",
        name="Config Save",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Save the current configuration to a new name 'new_config'",
        expected_agents=["config_agent"],
        expected_tools=["config_save"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="config_agent",
                         required_tools=["config_save"],
                         expected_parameters={
                                             "config_save": {
                                                 "name": "new_config"
                                             }
                                         }
                         )
        ],
        description="Save configuration to JSON file"
    ),
    BenchmarkTestCase(
        test_id="SAST_CFG_004",
        name="Config List",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="List all available configurations",
        expected_agents=["config_agent"],
        expected_tools=["config_list"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="config_agent",
                         required_tools=["config_list"],
                         expected_parameters={
                                               }
                         )
        ],
        description="List all configurations"
    ),

    # ---- Building Tools ----
    BenchmarkTestCase(
        test_id="SAST_BLD_001",
        name="Building Add",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add a new building called 'COE'. "
                "'COE' is an office building with 5 floors, it locates at Syracuse, NY."
                "It belongs to 'downtown' cluster",
        expected_agents=["building_agent"],
        expected_tools=["building_add"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="building_agent",
                         required_tools=["building_add"],
                         expected_parameters={
                             "building_add": {
                                 "building_id": "COE",
                                 "cluster_id": "downtown",
                                 "building_type": "office"
                             }
                         }
                         )
        ],
        description="Add a single building"
    ),
    BenchmarkTestCase(
        test_id="SAST_BLD_002",
        name="Building Query",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Can you describe all buildings in this case?",
        expected_agents=["building_agent"],
        expected_tools=["building_query"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="building_agent",
                         required_tools=["building_query"],
                         expected_parameters={
                                              }
                         )
        ],
        description="Query all buildings"
    ),
    BenchmarkTestCase(
        test_id="SAST_BLD_003",
        name="Building Update",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="The retrofit program updates the building 'SFH_1' to an 'apartment', "
                "it has five thermal zones which are 'Tzone1', 'Tzone2' ... to 'Tzone5'",
        expected_agents=["building_agent"],
        expected_tools=["building_update"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="building_agent",
                         required_tools=["building_update"],
                         expected_parameters={
                                  "building_update": {
                                      "building_id": "SFH_1",
                                      "updates": {"parameters": {
                                                "building_type": "apartment"
                                                              },
                                                 "thermal_zones": ["Tzone1", "Tzone2", "Tzone3", "Tzone4", "Tzone5"]},
                                                  }
                                              }
                         )
        ],
        description="Update current building setup"
    ),
    BenchmarkTestCase(
        test_id="SAST_BLD_004",
        name="Building Thermal Zone Module",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add another thermal zone 'Tzone_core' to existing building 'SFH_1', below is the hype-parameter"
                "The training data used is 90 days and test on 7 days, move the model to 'cuda:0' "
                "the temperature unit now is F and we use LSTM",
        expected_agents=["building_agent"],
        expected_tools=["building_add_thermal_zone"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="building_agent",
                         required_tools=["building_add_thermal_zone"],
                         expected_parameters={
                             "building_add_thermal_zone": {
                                 "building_id": "SFH_1",
                                 "zone_id": "Tzone_core",
                                 "parameters": {
                                            "model_args": {
                                                "modeltype": "LSTM",
                                                "trainday": 90,
                                                "testday": 7,
                                                "temp_unit": "F",
                                                "device": "cuda:0",
                                            },
                                        },
                             }
                         }
                         )
        ],
        description="Add a new thermal zone in existing building"
    ),
    BenchmarkTestCase(
        test_id="SAST_BLD_005",
        name="Building Electric Zone Module",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add another electrical zone 'Ezone_core' to existing building 'SFH_1'",
        expected_agents=["building_agent"],
        expected_tools=["building_add_electrical_zone"],
        expected_steps=[
            ExpectedStep(step_order=1,
                         agent_id="building_agent",
                         required_tools=["building_add_electrical_zone"],
                         expected_parameters={
                             "building_add_electrical_zone": {
                                 "building_id": "SFH_1",
                                 "zone_id": "Ezone_core",
                             }
                         }
                         )
        ],
        description="Add a new electrical zone in existing building"
    ),

    # ---- Cluster Tools ----
    BenchmarkTestCase(
        test_id="SAST_CLU_001",
        name="Cluster Add",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add a new cluster named 'RESI' located at Syracuse, NY for residential buildings",
        expected_agents=["cluster_agent"],
        expected_tools=["cluster_add"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="cluster_agent",
                required_tools=["cluster_add"],
                expected_parameters={
                    "cluster_add": {
                        "cluster_id": "RESI",
                        "parameters": {
                            "location": "Syracuse, NY"
                        }
                    }
                }
            )
        ],
        description="Add a new cluster to the configuration"
    ),

    # ---- HVAC System Tools ----
    BenchmarkTestCase(
        test_id="SAST_HVAC_001",
        name="HVAC Add FCU System",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add an HVAC system 'hvac2small_office' to cluster 'commercial_cluster'. "
                "Configure it as an FCU system with: fan rated flow 0.5 m3/s and power 500W in constant mode, "
                "coil effectiveness 0.75, pump flow 0.02 m3/s with 2000W power, "
                "chiller capacity 20kW with COP 4.0, "
                "and cooling tower capacity 20kW with fan power 500W",
        expected_agents=["hvac_agent"],
        expected_tools=["hvac_add"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="hvac_agent",
                required_tools=["hvac_add"],
                expected_parameters={
                    "hvac_add": {
                        "system_id": "hvac2small_office",
                        "cluster_id": "commercial_cluster",
                        "system_config": {
                            "fan": {"rated_flow_m3s": 0.5, "rated_power_W": 500},
                            "fan_ctrl": {"ctrl_type": "constant", "rated_flow_m3s": 0.5},
                            "coil": {"effectiveness": 0.75},
                            "pump": {"rated_flow_m3s": 0.02, "rated_power_W": 2000},
                            "chiller": {"rated_capacity_W": 20000, "rated_cop": 4.0},
                            "tower": {"rated_capacity_W": 20000, "rated_fan_power_W": 500}
                        }
                    }
                }
            )
        ],
        description="Add HVAC FCU system with detailed configuration"
    ),
    BenchmarkTestCase(
        test_id="SAST_HVAC_002",
        name="HVAC Query All",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Show me all HVAC systems in the current configuration",
        expected_agents=["hvac_agent"],
        expected_tools=["hvac_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="hvac_agent",
                required_tools=["hvac_query"],
                expected_parameters={}
            )
        ],
        description="Query all HVAC systems"
    ),
    BenchmarkTestCase(
        test_id="SAST_HVAC_003",
        name="HVAC Update Chiller",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Update HVAC system 'hvac_system_1': change chiller COP to 5.0 and capacity to 18kW",
        expected_agents=["hvac_agent"],
        expected_tools=["hvac_update"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="hvac_agent",
                required_tools=["hvac_update"],
                expected_parameters={
                    "hvac_update": {
                        "system_id": "hvac_system_1",
                        "updates": {
                            "system_config": {
                                "chiller": {"rated_cop": 5.0, "rated_capacity_W": 18000}
                            }
                        }
                    }
                }
            )
        ],
        description="Update HVAC system chiller parameters"
    ),
    BenchmarkTestCase(
        test_id="SAST_HVAC_004",
        name="Remove HVAC System",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Remove HVAC system named 'hvac_system_1'",
        expected_agents=["hvac_agent"],
        expected_tools=["hvac_remove"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="hvac_agent",
                required_tools=["hvac_remove"],
                expected_parameters={
                    "hvac_remove": {
                        "system_id": "hvac_system_1",
                    }
                }
            )
        ],
        description="Remove existing HVAC system"
    ),
    # ---- DER System Tools ----
    BenchmarkTestCase(
        test_id="SAST_DER_001",
        name="DER Add PV-Battery System",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add a DER system 'der_residential_1' to 'residential_cluster_1'. "
                "Configure it with: PV rated capacity 15kW, "
                "battery 10kWh capacity with 40% initial SOC, 0.3 charge speed, 0.4 discharge speed, 0.92 charge efficiency",
        expected_agents=["der_agent"],
        expected_tools=["der_add"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="der_agent",
                required_tools=["der_add"],
                expected_parameters={
                    "der_add": {
                        "system_id": "der_residential_1",
                        "cluster_id": "residential_cluster_1",
                        "system_config": {
                            "pv": {"rated_capacity_kW": 15},
                            "bat": {
                                "rated_capacity_kWh": 10,
                                "initial_soc": 0.4,
                                "charge_speed": 0.3,
                                "discharge_speed": 0.4,
                                "charge_efficiency": 0.92
                            }
                        }
                    }
                }
            )
        ],
        description="Add DER system with PV and battery"
    ),
    BenchmarkTestCase(
        test_id="SAST_DER_002",
        name="DER Add with EVs",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add DER system 'der_ev_hub' to 'parking_cluster' with two EVs: "
                "first EV 'ev_model3' with 75kWh capacity, 30% initial SOC, charge_speed 0.5, discharge_speed 2, charge_efficiency 0.95, initially connected; "
                "second EV 'ev_leaf' with 40kWh capacity, 60% initial SOC, charge_speed 0.3, discharge_speed 1, charge_efficiency 0.9, initially not connected",
        expected_agents=["der_agent"],
        expected_tools=["der_add"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="der_agent",
                required_tools=["der_add"],
                expected_parameters={
                    "der_add": {
                        "system_id": "der_ev_hub",
                        "cluster_id": "parking_cluster",
                        "system_config": {
                            "evs": [
                                {
                                    "id": "ev_model3",
                                    "rated_capacity_kWh": 75,
                                    "initial_soc": 0.3,
                                    "charge_speed": 0.5,
                                    "discharge_speed": 2,
                                    "charge_efficiency": 0.95,
                                    "initially_connected": True
                                },
                                {
                                    "id": "ev_leaf",
                                    "rated_capacity_kWh": 40,
                                    "initial_soc": 0.6,
                                    "charge_speed": 0.3,
                                    "discharge_speed": 1,
                                    "charge_efficiency": 0.9,
                                    "initially_connected": False
                                }
                            ]
                        }
                    }
                }
            )
        ],
        description="Add DER system with multiple EVs"
    ),
    BenchmarkTestCase(
        test_id="SAST_DER_003",
        name="DER Query All",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Show me the detailed information about DER systems 'der_system_1'",
        expected_agents=["der_agent"],
        expected_tools=["der_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="der_agent",
                required_tools=["der_query"],
                expected_parameters={"der_query": {
                            "system_id": "der_system_1"}}
            )
        ],
        description="Query all DER systems"
    ),
    BenchmarkTestCase(
        test_id="SAST_DER_004",
        name="DER Update Battery",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="I just increased my battery capacity to 20kWh for DER system 'der_system_1'"
                "The new Battery has charge speed of 0.35",
        expected_agents=["der_agent"],
        expected_tools=["der_update"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="der_agent",
                required_tools=["der_update"],
                expected_parameters={
                    "der_update": {
                        "system_id": "der_system_1",
                        "updates": {
                            "system_config": {
                                "bat": {
                                    "rated_capacity_kWh": 20,
                                    "charge_speed": 0.35
                                }
                            }
                        }
                    }
                }
            )
        ],
        description="Update DER battery configuration"
    ),

    # ---- Controller Tools ----
    BenchmarkTestCase(
        test_id="SAST_CTRL_001",
        name="Controller Add HVAC",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add an HVAC controller 'controller_hvac_1' for system 'hvac_system_1'. "
                "Set it as rule-based cooling mode with base cooling setpoint 23.5°C, "
                "base heating 19°C, deadband 0.3, and max cooling power 5000W",
        expected_agents=["controller_agent"],
        expected_tools=["controller_add_hvac"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="controller_agent",
                required_tools=["controller_add_hvac"],
                expected_parameters={
                    "controller_add_hvac": {
                        "controller_id": "controller_hvac_1",
                        "system_id": "hvac_system_1",
                        "parameters": {
                            "type": "rule-based",
                            "mode": "cooling",
                            "base_cooling": 23.5,
                            "base_heating": 19.0,
                            "deadband": 0.3,
                            "cooling_power_max": 5000.0
                        }
                    }
                }
            )
        ],
        description="Add HVAC controller with detailed parameters"
    ),
    BenchmarkTestCase(
        test_id="SAST_CTRL_002",
        name="Controller Add DER",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add DER controller 'ctrl_der_1' for 'der_system_1'. "
                "Configure as self-consumption mode with battery SOC limits 0.15-0.85, "
                "EV target SOC 0.9, V2G enabled, max grid import 8000W, max export 4000W",
        expected_agents=["controller_agent"],
        expected_tools=["controller_add_der"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="controller_agent",
                required_tools=["controller_add_der"],
                expected_parameters={
                    "controller_add_der": {
                        "controller_id": "ctrl_der_1",
                        "system_id": "der_system_1",
                        "parameters": {
                            "mode": "self_consumption",
                            "bat_soc_min": 0.15,
                            "bat_soc_max": 0.85,
                            "ev_soc_target": 0.9,
                            "ev_v2g_enabled": True,
                            "max_grid_import": 8000,
                            "max_grid_export": 4000
                        }
                    }
                }
            )
        ],
        description="Add DER controller with detailed parameters"
    ),
    BenchmarkTestCase(
        test_id="SAST_CTRL_003",
        name="Controller Query All",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Show me all controllers in the current configuration",
        expected_agents=["controller_agent"],
        expected_tools=["controller_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="controller_agent",
                required_tools=["controller_query"],
                expected_parameters={}
            )
        ],
        description="Query all controllers"
    ),
    BenchmarkTestCase(
        test_id="SAST_CTRL_004",
        name="Controller Update",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Update controller 'hvac_controller_1': change base cooling to 25°C, deadband to 0.8°C, and apply 2°C pre-cooling for 2 hours",
        expected_agents=["controller_agent"],
        expected_tools=["controller_update"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="controller_agent",
                required_tools=["controller_update"],
                expected_parameters={
                    "controller_update": {
                        "controller_id": "hvac_controller_1",
                        "updates": {
                            "precooling": {"degree": 2, "hours": 2},
                            "base_cooling": 25.0,
                            "deadband": 0.8
                        }
                    }
                }
            )
        ],
        description="Update controller parameters"
    ),

    # ---- Disturbance Tools ----
    BenchmarkTestCase(
        test_id="SAST_DIST_001",
        name="Disturbance Add Weather",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add weather disturbance data from file '/data/weather/syracuse_2023.csv' "
                "starting from '2023-07-15 00:00:00'",
        expected_agents=["disturbance_agent"],
        expected_tools=["disturbance_add_weather"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="disturbance_agent",
                required_tools=["disturbance_add_weather"],
                expected_parameters={
                    "disturbance_add_weather": {
                        "file_path": "/data/weather/syracuse_2023.csv",
                        "simulation_start": "2023-07-15 00:00:00"
                    }
                }
            )
        ],
        description="Add weather disturbance"
    ),

    BenchmarkTestCase(
        test_id="SAST_DIST_002",
        name="Disturbance Add Occupancy",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Add occupancy data from '/data/occupancy/office_schedule.csv' starting '2023-08-01 00:00:00'",
        expected_agents=["disturbance_agent"],
        expected_tools=["disturbance_add_occupancy"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="disturbance_agent",
                required_tools=["disturbance_add_occupancy"],
                expected_parameters={
                    "disturbance_add_occupancy": {
                        "file_path": "/data/occupancy/office_schedule.csv",
                        "simulation_start": "2023-08-01 00:00:00"
                    }
                }
            )
        ],
        description="Add occupancy disturbance"
    ),

    BenchmarkTestCase(
        test_id="SAST_DIST_003",
        name="Disturbance Query",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="List all disturbances configured for this simulation",
        expected_agents=["disturbance_agent"],
        expected_tools=["disturbance_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="disturbance_agent",
                required_tools=["disturbance_query"],
                expected_parameters={}
            )
        ],
        description="Query all disturbances"
    ),
    # ---- Environment Tools ----
    BenchmarkTestCase(
        test_id="SAST_ENV_001",
        name="Environment Setup",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Setup the simulation environment with 15-minute resolution (900 seconds), "
                "3-day duration (259200 seconds), starting from '2023-08-15 00:00:00'",
        expected_agents=["environment_agent"],
        expected_tools=["environment_setup"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="environment_agent",
                required_tools=["environment_setup"],
                expected_parameters={
                    "environment_setup": {
                        "resolution": 900,
                        "duration": 259200,
                        "simulation_start": "2023-08-15 00:00:00"
                    }
                }
            )
        ],
        description="Setup simulation environment"
    ),
    BenchmarkTestCase(
        test_id="SAST_ENV_002",
        name="Environment Update",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Update environment: change resolution to 5 minutes (300 seconds)",
        expected_agents=["environment_agent"],
        expected_tools=["environment_update"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="environment_agent",
                required_tools=["environment_update"],
                expected_parameters={
                    "environment_update": {
                        "updates": {"resolution": 300}
                    }
                }
            )
        ],
        description="Update environment parameters"
    ),
    BenchmarkTestCase(
        test_id="SAST_ENV_003",
        name="Environment Query",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="What is the current environment configuration?",
        expected_agents=["environment_agent"],
        expected_tools=["environment_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="environment_agent",
                required_tools=["environment_query"],
                expected_parameters={}
            )
        ],
        description="Query environment configuration"
    ),
    # ---- Simulation Tools ----
    BenchmarkTestCase(
        test_id="SAST_SIM_001",
        name="Simulation Run",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Run simulation for 96 timesteps and save as 'case_study'",
        expected_agents=["simulation_agent"],
        expected_tools=["simulation_run"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "case_study"
                    }
                }
            )
        ],
        description="Run simulation with specific steps"
    ),

    BenchmarkTestCase(
        test_id="SAST_SIM_002",
        name="Simulation Get Status",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="What is the current simulation status?",
        expected_agents=["simulation_agent"],
        expected_tools=["simulation_get_status"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="simulation_agent",
                required_tools=["simulation_get_status"],
                expected_parameters={}
            )
        ],
        description="Get simulation status"
    ),
    BenchmarkTestCase(
        test_id="SAST_SIM_003",
        name="Simulation List Results",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Show me all available simulation results",
        expected_agents=["simulation_agent"],
        expected_tools=["simulation_list_results"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="simulation_agent",
                required_tools=["simulation_list_results"],
                expected_parameters={}
            )
        ],
        description="List simulation results"
    ),

    # ---- Analysis Tools ----
    BenchmarkTestCase(
        test_id="SAST_ANL_001",
        name="Analysis Comfort",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Analyze thermal comfort metrics for simulation 'case_study'",
        expected_agents=["analysis_agent"],
        expected_tools=["analysis_comfort"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="analysis_agent",
                required_tools=["analysis_comfort"],
                expected_parameters={
                    "analysis_comfort": {"simulation_name": "case_study"}
                }
            )
        ],
        description="Analyze comfort metrics"
    ),
    BenchmarkTestCase(
        test_id="SAST_ANL_002",
        name="Analysis Energy",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Analyze energy consumption for simulation 'case_study'",
        expected_agents=["analysis_agent"],
        expected_tools=["analysis_energy"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="analysis_agent",
                required_tools=["analysis_energy"],
                expected_parameters={
                    "analysis_energy": {"simulation_name": "case_study"}
                }
            )
        ],
        description="Analyze energy metrics"
    ),
    BenchmarkTestCase(
        test_id="SAST_ANL_003",
        name="Analysis Cost",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="What were the costs in simulation 'case_study'?",
        expected_agents=["analysis_agent"],
        expected_tools=["analysis_cost"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="analysis_agent",
                required_tools=["analysis_cost"],
                expected_parameters={
                    "analysis_cost": {"simulation_name": "case_study"}
                }
            )
        ],
        description="Analyze cost metrics"
    ),
    BenchmarkTestCase(
        test_id="SAST_ANL_004",
        name="Analysis Flexibility",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Analyze grid flexibility and battery cycling for 'case_study'",
        expected_agents=["analysis_agent"],
        expected_tools=["analysis_flexibility"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="analysis_agent",
                required_tools=["analysis_flexibility"],
                expected_parameters={
                    "analysis_flexibility": {"simulation_name": "case_study"}
                }
            )
        ],
        description="Analyze flexibility metrics"
    ),
    # ---- Comparison Tools ----
    BenchmarkTestCase(
        test_id="SAST_CMP_001",
        name="Comparison Comfort",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Compare thermal comfort between simulations 'case_study_1' and 'case_study_2'",
        expected_agents=["comparison_agent"],
        expected_tools=["comparison_comfort"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="comparison_agent",
                required_tools=["comparison_comfort"],
                expected_parameters={
                    "comparison_comfort": {
                        "sim1": "case_study_1",
                        "sim2": "case_study_2"
                    }
                }
            )
        ],
        description="Compare comfort metrics"
    ),
    BenchmarkTestCase(
        test_id="SAST_CMP_002",
        name="Comparison Energy",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Compare energy performance between 'case_study_1' and 'case_study_2'",
        expected_agents=["comparison_agent"],
        expected_tools=["comparison_energy"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="comparison_agent",
                required_tools=["comparison_energy"],
                expected_parameters={
                    "comparison_energy": {
                        "sim1": "case_study_1",
                        "sim2": "case_study_2"
                    }
                }
            )
        ],
        description="Compare energy metrics"
    ),
    BenchmarkTestCase(
        test_id="SAST_CMP_003",
        name="Comparison Comprehensive",
        category=TestCategory.SINGLE_AGENT_SINGLE_TOOL,
        request="Do a full comparison of 'case_study_1' vs 'case_study_2",
        expected_agents=["comparison_agent"],
        expected_tools=["comparison_comprehensive"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="comparison_agent",
                required_tools=["comparison_comprehensive"],
                expected_parameters={
                    "comparison_comprehensive": {
                        "sim1": "case_study_1",
                        "sim2": "case_study_2"
                    }
                }
            )
        ],
        description="Comprehensive comparison"
    ),

  ]

SINGLE_AGENT_MULTI_TOOL_CASES = [
    # ---- HVAC Lifecycle: Add -> Update -> Query ----
    BenchmarkTestCase(
        test_id="SAMT_HVAC_001",
        name="HVAC Add-Update-Query Lifecycle",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "Add an HVAC system 'hvac_office_main' to cluster 'office_cluster' with "
            "fan rated flow 0.8 m3/s and power 750W, chiller capacity 30kW with COP 3.5. "
            "Then update the chiller COP to 4.2 and capacity to 35kW. "
            "Finally, query information about the updated HVAC system."
        ),
        expected_agents=["hvac_agent"],
        expected_tools=["hvac_add", "hvac_update", "hvac_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="hvac_agent",
                required_tools=["hvac_add"],
                expected_parameters={
                    "hvac_add": {
                        "system_id": "hvac_office_main",
                        "cluster_id": "office_cluster",
                        "system_config": {
                            "fan": {"rated_flow_m3s": 0.8, "rated_power_W": 750},
                            "chiller": {"rated_capacity_W": 30000, "rated_cop": 3.5}
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="hvac_agent",
                required_tools=["hvac_update"],
                expected_parameters={
                    "hvac_update": {
                        "system_id": "hvac_office_main",
                        "updates": {
                            "system_config": {
                                "chiller": {"rated_cop": 4.2, "rated_capacity_W": 35000}
                            }
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="hvac_agent",
                required_tools=["hvac_query"],
                expected_parameters={
                    "hvac_query": {"system_id": "hvac_office_main"}
                }
            )
        ],
        description="Complete HVAC lifecycle: add system, update parameters, query result"
    ),

    # ---- DER Lifecycle: Add -> Update Battery -> Query ----
    BenchmarkTestCase(
        test_id="SAMT_DER_001",
        name="DER Add-Update-Query Lifecycle",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "Add a DER system 'der_home_1' to 'residential_cluster' with PV capacity 10kW "
            "and battery 8kWh at 50% initial SOC with charge efficiency 0.9. "
            "Then upgrade the battery to 15kWh capacity with charge speed 0.4. "
            "Query the information about the latest DER system."
            "Don't save the file."
        ),
        expected_agents=["der_agent"],
        expected_tools=["der_add", "der_update", "der_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="der_agent",
                required_tools=["der_add"],
                expected_parameters={
                    "der_add": {
                        "system_id": "der_home_1",
                        "cluster_id": "residential_cluster",
                        "system_config": {
                            "pv": {"rated_capacity_kW": 10},
                            "bat": {
                                "rated_capacity_kWh": 8,
                                "initial_soc": 0.5,
                                "charge_efficiency": 0.9
                            }
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="der_agent",
                required_tools=["der_update"],
                expected_parameters={
                    "der_update": {
                        "system_id": "der_home_1",
                        "updates": {
                            "system_config": {
                                "bat": {
                                    "rated_capacity_kWh": 15,
                                    "charge_speed": 0.4
                                }
                            }
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="der_agent",
                required_tools=["der_query"],
                expected_parameters={
                    "der_query": {"system_id": "der_home_1"}
                }
            )
        ],
        description="Complete DER lifecycle: add system, upgrade battery, query result"
    ),

    # ---- Config Lifecycle: Create -> Validate -> Save ----
    BenchmarkTestCase(
        test_id="SAMT_CFG_001",
        name="Config Create-Validate-Save Workflow",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "Create a new configuration named 'summer_scenario'. "
            "Validate that the configuration is complete and correct. "
            "Then save it as 'summer_scenario_v1'."
        ),
        expected_agents=["config_agent"],
        expected_tools=["config_create", "config_validate", "config_save"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="config_agent",
                required_tools=["config_create"],
                expected_parameters={
                    "config_create": {"name": "summer_scenario"}
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="config_agent",
                required_tools=["config_validate"],
                expected_parameters={}
            ),
            ExpectedStep(
                step_order=3,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {"name": "summer_scenario_v1"}
                }
            )
        ],
        description="Configuration workflow: create, validate, and save"
    ),

    # ---- Controller Lifecycle: Add HVAC Controller -> Update -> Query ----
    BenchmarkTestCase(
        test_id="SAMT_CTRL_001",
        name="Controller Add-Update-Query Lifecycle",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "Add an HVAC controller 'ctrl_hvac_zone1' for system 'hvac_system_1' "
            "as rule-based cooling mode with base cooling 24°C and deadband 0.5. "
            "Update it to use base cooling 22°C with 3°C pre-cooling for 1 hour. "
            "Then show all controllers."
        ),
        expected_agents=["controller_agent"],
        expected_tools=["controller_add_hvac", "controller_update", "controller_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="controller_agent",
                required_tools=["controller_add_hvac"],
                expected_parameters={
                    "controller_add_hvac": {
                        "controller_id": "ctrl_hvac_zone1",
                        "system_id": "hvac_system_1",
                        "parameters": {
                            "type": "rule-based",
                            "mode": "cooling",
                            "base_cooling": 24.0,
                            "deadband": 0.5
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="controller_agent",
                required_tools=["controller_update"],
                expected_parameters={
                    "controller_update": {
                        "controller_id": "ctrl_hvac_zone1",
                        "updates": {
                            "base_cooling": 22.0,
                            "precooling": {"degree": 3, "hours": 1}
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="controller_agent",
                required_tools=["controller_query"],
                expected_parameters={}
            )
        ],
        description="Controller lifecycle: add, update settings, query all"
    ),

    # ---- Building: Add -> Add Thermal Zone -> Query ----
    BenchmarkTestCase(
        test_id="SAMT_BLD_001",
        name="Building Add with Thermal Zone",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "Add a new building 'Lab_Building' as a 'laboratory' type in cluster named 'campus_cluster' "
            "located in Boston, MA with 3 floors. "
            "Then call tool 'building_add_thermal_zone' to add a thermal zone "
            "Named 'Tzone_lab1', with 60 training days, "
            "14 test days, temperature unit C, device cuda:0. "
            "Finally query the building information details."
        ),
        expected_agents=["building_agent"],
        expected_tools=["building_add", "building_add_thermal_zone", "building_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="building_agent",
                required_tools=["building_add"],
                expected_parameters={
                    "building_add": {
                        "building_id": "Lab_Building",
                        "cluster_id": "campus_cluster",
                        "building_type": "laboratory"
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="building_agent",
                required_tools=["building_add_thermal_zone"],
                expected_parameters={
                    "building_add_thermal_zone": {
                        "building_id": "Lab_Building",
                        "zone_id": "Tzone_lab1",
                        "parameters": {
                            "model_args": {
                                "modeltype": "LSTM",
                                "trainday": 60,
                                "testday": 14,
                                "temp_unit": "C",
                                "device": "cuda:0"
                            }
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="building_agent",
                required_tools=["building_query"],
                expected_parameters={"building_query": {"building_id": "Lab_Building"}}
            )
        ],
        description="Building setup with thermal zone configuration"
    ),

    # ---- Simulation: Create Environment -> Run -> Get Status ----
    BenchmarkTestCase(
        test_id="SAMT_SIM_001",
        name="Simulation Run-Status Workflow",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "Run the simulation for 48 timesteps and save name as 'quick_test'. "
            "Then check the simulation status."
        ),
        expected_agents=["simulation_agent"],
        expected_tools=["simulation_run", "simulation_get_status"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 48,
                        "save_name": "quick_test"
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="simulation_agent",
                required_tools=["simulation_get_status"],
                expected_parameters={}
            )
        ],
        description="Simulation workflow: run and check status"
    ),

    # ---- Analysis: Energy -> Cost -> Flexibility (Full Analysis) ----
    BenchmarkTestCase(
        test_id="SAMT_ANL_001",
        name="Comprehensive Analysis Suite",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "For simulation 'baseline_run', analyze the energy consumption, "
            "calculate the costs, and evaluate grid flexibility metrics."
        ),
        expected_agents=["analysis_agent"],
        expected_tools=["analysis_energy", "analysis_cost", "analysis_flexibility"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="analysis_agent",
                required_tools=["analysis_energy"],
                expected_parameters={
                    "analysis_energy": {"simulation_name": "baseline_run"}
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="analysis_agent",
                required_tools=["analysis_cost"],
                expected_parameters={
                    "analysis_cost": {"simulation_name": "baseline_run"}
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="analysis_agent",
                required_tools=["analysis_flexibility"],
                expected_parameters={
                    "analysis_flexibility": {"simulation_name": "baseline_run"}
                }
            )
        ],
        description="Complete analysis: energy, cost, and flexibility metrics"
    ),

    # ---- Environment: Setup -> Update -> Query ----
    BenchmarkTestCase(
        test_id="SAMT_ENV_001",
        name="Environment Setup-Update-Query",
        category=TestCategory.SINGLE_AGENT_MULTI_TOOL,
        request=(
            "Setup simulation environment with 10-minute resolution (600 seconds), "
            "1-day duration (86400 seconds), starting '2024-01-15 00:00:00'. "
            "Then change resolution to 5 minutes (300 seconds). "
            "Query the final environment configuration."
        ),
        expected_agents=["environment_agent"],
        expected_tools=["environment_setup", "environment_update", "environment_query"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="environment_agent",
                required_tools=["environment_setup"],
                expected_parameters={
                    "environment_setup": {
                        "resolution": 600,
                        "duration": 86400,
                        "simulation_start": "2024-01-15 00:00:00"
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="environment_agent",
                required_tools=["environment_update"],
                expected_parameters={
                    "environment_update": {
                        "updates": {"resolution": 300}
                    }
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="environment_agent",
                required_tools=["environment_query"],
                expected_parameters={}
            )
        ],
        description="Environment configuration workflow"
    ),
]

# =============================================================================
# MULTI AGENT SINGLE TOOL TEST CASES
# =============================================================================

MULTI_AGENT_SINGLE_TOOL_CASES = [
    # ---- Config Agent + Simulation Agent: Validate -> Create Env ----
    BenchmarkTestCase(
        test_id="MAST_001",
        name="Config Validation then Simulation Run",
        category=TestCategory.MULTI_AGENT_SINGLE_TOOL,
        request=(
            "First validate the current configuration to ensure it's complete. "
            "Then run a quick 48-step simulation saved as 'validated_run'."
        ),
        expected_agents=["config_agent", "simulation_agent"],
        expected_tools=["config_validate", "simulation_run"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="config_agent",
                required_tools=["config_validate"],
                expected_parameters={}
            ),
            ExpectedStep(
                step_order=2,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 48,
                        "save_name": "validated_run"
                    }
                }
            )
        ],
        description="Validate config before running simulation"
    ),

    # ---- HVAC Agent + Controller Agent: Add HVAC -> Add Controller ----
    BenchmarkTestCase(
        test_id="MAST_002",
        name="HVAC System with Controller Setup",
        category=TestCategory.MULTI_AGENT_SINGLE_TOOL,
        request=(
            "Add HVAC system 'hvac_retail' to 'retail_cluster' with fan flow 1.2 m3/s, "
            "power 1000W, and chiller capacity 50kW with COP 3.8. "
            "Then add a controller 'ctrl_retail' for this HVAC system in cooling mode "
            "with base cooling 25°C and deadband 0.4."
        ),
        expected_agents=["hvac_agent", "controller_agent"],
        expected_tools=["hvac_add", "controller_add_hvac"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="hvac_agent",
                required_tools=["hvac_add"],
                expected_parameters={
                    "hvac_add": {
                        "system_id": "hvac_retail",
                        "cluster_id": "retail_cluster",
                        "system_config": {
                            "fan": {"rated_flow_m3s": 1.2, "rated_power_W": 1000},
                            "chiller": {"rated_capacity_W": 50000, "rated_cop": 3.8}
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="controller_agent",
                required_tools=["controller_add_hvac"],
                expected_parameters={
                    "controller_add_hvac": {
                        "controller_id": "ctrl_retail",
                        "system_id": "hvac_retail",
                        "parameters": {
                            "mode": "cooling",
                            "base_cooling": 25.0,
                            "deadband": 0.4
                        }
                    }
                }
            )
        ],
        description="Setup HVAC system and its controller across two agents"
    ),

    # ---- DER Agent + Controller Agent: Add DER -> Add DER Controller ----
    BenchmarkTestCase(
        test_id="MAST_003",
        name="DER System with Controller Setup",
        category=TestCategory.MULTI_AGENT_SINGLE_TOOL,
        request=(
            "Add DER system 'der_commercial' to 'commercial_cluster' with PV 25kW "
            "and battery 20kWh at 40% SOC with charge efficiency 0.92. "
            "Then add a DER controller 'ctrl_der_comm' for this system in self_consumption mode "
            "with battery SOC limits 0.2-0.9 and max grid import 10000W."
        ),
        expected_agents=["der_agent", "controller_agent"],
        expected_tools=["der_add", "controller_add_der"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="der_agent",
                required_tools=["der_add"],
                expected_parameters={
                    "der_add": {
                        "system_id": "der_commercial",
                        "cluster_id": "commercial_cluster",
                        "system_config": {
                            "pv": {"rated_capacity_kW": 25},
                            "bat": {
                                "rated_capacity_kWh": 20,
                                "initial_soc": 0.4,
                                "charge_efficiency": 0.92
                            }
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="controller_agent",
                required_tools=["controller_add_der"],
                expected_parameters={
                    "controller_add_der": {
                        "controller_id": "ctrl_der_comm",
                        "system_id": "der_commercial",
                        "parameters": {
                            "mode": "self_consumption",
                            "bat_soc_min": 0.2,
                            "bat_soc_max": 0.9,
                            "max_grid_import": 10000
                        }
                    }
                }
            )
        ],
        description="Setup DER system and its controller across two agents"
    ),

    # ---- Simulation Agent + Analysis Agent: Run -> Analyze Energy ----
    BenchmarkTestCase(
        test_id="MAST_004",
        name="Run Simulation then Analyze",
        category=TestCategory.MULTI_AGENT_SINGLE_TOOL,
        request=(
            "Run the simulation for 96 timesteps and save as 'energy_study'. "
            "Then analyze the energy consumption of this simulation."
        ),
        expected_agents=["simulation_agent", "analysis_agent"],
        expected_tools=["simulation_run", "analysis_energy"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "energy_study"
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="analysis_agent",
                required_tools=["analysis_energy"],
                expected_parameters={
                    "analysis_energy": {"simulation_name": "energy_study"}
                }
            )
        ],
        description="Run simulation and analyze energy consumption"
    ),
]

# =============================================================================
# MULTI AGENT MULTI TOOL TEST CASES
# =============================================================================

MULTI_AGENT_MULTI_TOOL_CASES = [
    # ---- HVAC Tuning Comparison Study ----
    BenchmarkTestCase(
        test_id="MAMT_001",
        name="HVAC Controller Tuning Comparison",
        category=TestCategory.MULTI_AGENT_MULTI_TOOL,
          request=(
              "I want to compare two HVAC control strategies for 'hvac_controller_1'. "
              "First, set up a baseline with cooling setpoint 24°C and deadband 0.5, "
              "save it as 'hvac_baseline_config', and run a 96-step simulation called 'hvac_baseline'. "
              "Then test a pre-cooling strategy: lower the setpoint to 22°C and apply "
              "2°C pre-cooling for 2 hours before peak. Save that as 'hvac_precool_config' "
              "and run another 96-step simulation called 'hvac_precool'. "
              "Compare the energy consumption and thermal comfort between both scenarios."
          ),
        expected_agents=["controller_agent", "config_agent", "simulation_agent", "comparison_agent"],
        expected_tools=[
            "controller_update", "config_save", "simulation_run",
            "comparison_energy", "comparison_comfort"
        ],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="controller_agent",
                required_tools=["controller_update"],
                expected_parameters={
                    "controller_update": {
                        "controller_id": "hvac_controller_1",
                        "updates": {
                            "base_cooling": 24.0,
                            "deadband": 0.5
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "hvac_baseline_config"
                    }
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "hvac_baseline"
                    }
                }
            ),
            ExpectedStep(
                step_order=4,
                agent_id="controller_agent",
                required_tools=["controller_update"],
                expected_parameters={
                    "controller_update": {
                        "controller_id": "hvac_controller_1",
                        "updates": {
                            "base_cooling": 22.0,
                            "precooling": {"degree": 2, "hours": 2}
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=5,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "hvac_precool_config"
                    }
                }
            ),
            ExpectedStep(
                step_order=6,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "hvac_precool"
                    }
                }
            ),
            ExpectedStep(
                step_order=7,
                agent_id="comparison_agent",
                required_tools=["comparison_energy"],
                expected_parameters={
                    "comparison_energy": {
                        "sim1": "hvac_baseline",
                        "sim2": "hvac_precool"
                    }
                }
            ),
            ExpectedStep(
                step_order=8,
                agent_id="comparison_agent",
                required_tools=["comparison_comfort"],
                expected_parameters={
                    "comparison_comfort": {
                        "sim1": "hvac_baseline",
                        "sim2": "hvac_precool"
                    }
                }
            )
        ],
        description="Compare HVAC controller settings: baseline vs pre-cooling strategy"
    ),

    # ---- DER Battery Sizing Study ----
    BenchmarkTestCase(
        test_id="MAMT_002",
        name="DER Battery Sizing Comparison",
        category=TestCategory.MULTI_AGENT_MULTI_TOOL,
        request=(
            "Update DER system 'der_system_1' battery to 10kWh capacity with charge speed 0.3. "
            "Save the configuration as 'der_small_battery_config'. "
            "Run simulation for 96 steps and save as 'der_small_battery'. "
            "Then update the battery to 20kWh with charge speed 0.4. "
            "Save the configuration as 'der_large_battery_config'. "
            "Run simulation for 96 steps and save as 'der_large_battery'. "
            "Perform a comprehensive comparison between these scenarios."
        ),
        expected_agents=["der_agent", "config_agent", "simulation_agent", "comparison_agent"],
        expected_tools=["der_update", "config_save", "simulation_run", "comparison_comprehensive"],
        expected_steps=[
            ExpectedStep(
                step_order=1,
                agent_id="der_agent",
                required_tools=["der_update"],
                expected_parameters={
                    "der_update": {
                        "system_id": "der_system_1",
                        "updates": {
                            "system_config": {
                                "bat": {
                                    "rated_capacity_kWh": 10,
                                    "charge_speed": 0.3
                                }
                            }
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=2,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "der_small_battery_config"
                    }
                }
            ),
            ExpectedStep(
                step_order=3,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "der_small_battery"
                    }
                }
            ),
            ExpectedStep(
                step_order=4,
                agent_id="der_agent",
                required_tools=["der_update"],
                expected_parameters={
                    "der_update": {
                        "system_id": "der_system_1",
                        "updates": {
                            "system_config": {
                                "bat": {
                                    "rated_capacity_kWh": 20,
                                    "charge_speed": 0.4
                                }
                            }
                        }
                    }
                }
            ),
            ExpectedStep(
                step_order=5,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "der_large_battery_config"
                    }
                }
            ),
            ExpectedStep(
                step_order=6,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "der_large_battery"
                    }
                }
            ),
            ExpectedStep(
                step_order=7,
                agent_id="comparison_agent",
                required_tools=["comparison_comprehensive"],
                expected_parameters={
                    "comparison_comprehensive": {
                        "sim1": "der_small_battery",
                        "sim2": "der_large_battery"
                    }
                }
            )
        ],
        description="Compare DER performance with different battery sizes"
    ),

    # ---- Comprehensive HVAC + DER Upgrade Study ----
    BenchmarkTestCase(
        test_id="MAMT_003",
        name="HVAC and DER System Upgrade Comparison",
        category=TestCategory.MULTI_AGENT_MULTI_TOOL,
        request=(
            "Save the current configuration as 'baseline_config'. "
            "Run the simulation for 96 steps and save as 'baseline'. "
            "Then upgrade the system: update HVAC system 'hvac_system_1' chiller COP to 5.0, "
            "update HVAC controller 'hvac_controller_1' to use 2-hour pre-cooling with 2°C offset, "
            "and update DER system 'der_system_1' battery capacity to 25kWh. "
            "Save the upgraded configuration as 'upgrade_config'. "
            "Run the upgraded simulation for 96 steps as 'upgrade'. "
            "Finally perform a comprehensive comparison between baseline and upgrade."
        ),
        expected_agents=[
            "config_agent", "simulation_agent", "hvac_agent",
            "controller_agent", "der_agent", "comparison_agent"
        ],
        expected_tools=[
            "config_save", "simulation_run", "hvac_update",
            "controller_update", "der_update", "comparison_comprehensive"
        ],
        expected_steps=[
            # Step 1: Save baseline configuration
            ExpectedStep(
                step_order=1,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "baseline_config"
                    }
                }
            ),
            # Step 2: Run baseline simulation
            ExpectedStep(
                step_order=2,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "baseline"
                    }
                }
            ),
            # Step 3: Update HVAC chiller COP
            ExpectedStep(
                step_order=3,
                agent_id="hvac_agent",
                required_tools=["hvac_update"],
                expected_parameters={
                    "hvac_update": {
                        "system_id": "hvac_system_1",
                        "updates": {
                            "system_config": {
                                "chiller": {"rated_cop": 5.0}
                            }
                        }
                    }
                }
            ),
            # Step 4: Update HVAC controller with pre-cooling
            ExpectedStep(
                step_order=4,
                agent_id="controller_agent",
                required_tools=["controller_update"],
                expected_parameters={
                    "controller_update": {
                        "controller_id": "hvac_controller_1",
                        "updates": {
                            "precooling": {"degree": 2, "hours": 2}
                        }
                    }
                }
            ),
            # Step 5: Update DER battery capacity (auto-syncs to controller)
            ExpectedStep(
                step_order=5,
                agent_id="der_agent",
                required_tools=["der_update"],
                expected_parameters={
                    "der_update": {
                        "system_id": "der_system_1",
                        "updates": {
                            "system_config": {
                                "bat": {"rated_capacity_kWh": 25}
                            }
                        }
                    }
                }
            ),
            # Step 6: Save upgraded configuration
            ExpectedStep(
                step_order=6,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "upgrade_config"
                    }
                }
            ),
            # Step 7: Run upgraded simulation
            ExpectedStep(
                step_order=7,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "upgrade"
                    }
                }
            ),
            # Step 8: Comprehensive comparison
            ExpectedStep(
                step_order=8,
                agent_id="comparison_agent",
                required_tools=["comparison_comprehensive"],
                expected_parameters={
                    "comparison_comprehensive": {
                        "sim1": "baseline",
                        "sim2": "upgrade"
                    }
                }
            )
        ],
        description="Comprehensive baseline vs upgrade study: HVAC efficiency + pre-cooling + larger battery"
    ),

# ---- Comprehensive HVAC + DER Upgrade Study ----
    BenchmarkTestCase(
        test_id="MAMT_004",
        name="HVAC and DER System Upgrade Comparison",
        category=TestCategory.MULTI_AGENT_MULTI_TOOL,
          request=(
              "I want to evaluate a building retrofit package against the current baseline. "
              "First, save the current setup as 'baseline_config' and run a 96-step simulation "
              "called 'baseline'. "
              "The retrofit includes three upgrades: "
              "upgrading the chiller COP to 5.0 on 'hvac_system_1', "
              "adding 2°C pre-cooling for 2 hours on 'hvac_controller_1', "
              "and increasing the battery capacity to 25kWh on 'der_system_1'. "
              "Save the retrofitted configuration as 'upgrade_config', "
              "run a 96-step simulation called 'upgrade', "
              "and give me a comprehensive comparison of baseline vs retrofit performance."
          ),
        expected_agents=[
            "config_agent", "simulation_agent", "hvac_agent",
            "controller_agent", "der_agent", "comparison_agent"
        ],
        expected_tools=[
            "config_save", "simulation_run", "hvac_update",
            "controller_update", "der_update", "comparison_comprehensive"
        ],
        expected_steps=[
            # Step 1: Save baseline configuration
            ExpectedStep(
                step_order=1,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "baseline_config"
                    }
                }
            ),
            # Step 2: Run baseline simulation
            ExpectedStep(
                step_order=2,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "baseline"
                    }
                }
            ),
            # Step 3: Update HVAC chiller COP
            ExpectedStep(
                step_order=3,
                agent_id="hvac_agent",
                required_tools=["hvac_update"],
                expected_parameters={
                    "hvac_update": {
                        "system_id": "hvac_system_1",
                        "updates": {
                            "system_config": {
                                "chiller": {"rated_cop": 5.0}
                            }
                        }
                    }
                }
            ),
            # Step 4: Update HVAC controller with pre-cooling
            ExpectedStep(
                step_order=4,
                agent_id="controller_agent",
                required_tools=["controller_update"],
                expected_parameters={
                    "controller_update": {
                        "controller_id": "hvac_controller_1",
                        "updates": {
                            "precooling": {"degree": 2, "hours": 2}
                        }
                    }
                }
            ),
            # Step 5: Update DER battery capacity (auto-syncs to controller)
            ExpectedStep(
                step_order=5,
                agent_id="der_agent",
                required_tools=["der_update"],
                expected_parameters={
                    "der_update": {
                        "system_id": "der_system_1",
                        "updates": {
                            "system_config": {
                                "bat": {"rated_capacity_kWh": 25}
                            }
                        }
                    }
                }
            ),
            # Step 6: Save upgraded configuration
            ExpectedStep(
                step_order=6,
                agent_id="config_agent",
                required_tools=["config_save"],
                expected_parameters={
                    "config_save": {
                        "name": "upgrade_config"
                    }
                }
            ),
            # Step 7: Run upgraded simulation
            ExpectedStep(
                step_order=7,
                agent_id="simulation_agent",
                required_tools=["simulation_run"],
                expected_parameters={
                    "simulation_run": {
                        "steps": 96,
                        "save_name": "upgrade"
                    }
                }
            ),
            # Step 8: Comprehensive comparison
            ExpectedStep(
                step_order=8,
                agent_id="comparison_agent",
                required_tools=["comparison_comprehensive"],
                expected_parameters={
                    "comparison_comprehensive": {
                        "sim1": "baseline",
                        "sim2": "upgrade"
                    }
                }
            )
        ],
        description="Comprehensive baseline vs upgrade study: HVAC efficiency + pre-cooling + larger battery"
    ),
]


# =============================================================================
# HELPER FUNCTION TO GET ALL TEST CASES
# =============================================================================

def get_all_test_cases() -> Dict[TestCategory, List[BenchmarkTestCase]]:
    """Returns all test cases organized by category"""
    return {
        TestCategory.SINGLE_AGENT_SINGLE_TOOL: SINGLE_AGENT_SINGLE_TOOL_CASES,
        TestCategory.SINGLE_AGENT_MULTI_TOOL: SINGLE_AGENT_MULTI_TOOL_CASES,
        TestCategory.MULTI_AGENT_SINGLE_TOOL: MULTI_AGENT_SINGLE_TOOL_CASES,
        TestCategory.MULTI_AGENT_MULTI_TOOL: MULTI_AGENT_MULTI_TOOL_CASES,
    }


def get_test_cases_by_category(category: TestCategory) -> List[BenchmarkTestCase]:
    """Returns test cases for a specific category"""
    all_cases = get_all_test_cases()
    return all_cases.get(category, [])


def get_test_case_by_id(test_id: str) -> Optional[BenchmarkTestCase]:
    """Returns a specific test case by ID"""
    all_cases = get_all_test_cases()
    for cases in all_cases.values():
        for case in cases:
            if case.test_id == test_id:
                return case
    return None


def get_test_cases_summary() -> Dict[str, Any]:
    """Returns summary statistics of test cases"""
    all_cases = get_all_test_cases()
    return {
        "total_cases": sum(len(cases) for cases in all_cases.values()),
        "by_category": {
            cat.value: len(cases) for cat, cases in all_cases.items()
        },
        "categories": [cat.value for cat in TestCategory]
    }