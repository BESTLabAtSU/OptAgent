"""
BESTOpt MCP Server
Provides building simulation and configuration tools via MCP protocol
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import pickle
import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import main modules
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bestopt-mcp_center")

class BESTOptMCPServer:
    """MCP Server for BESTOpt platform"""

    def __init__(self, project_root: str = None):
        self.server = Server("bestopt-server")

        # Set project root
        if project_root:
            self.project_root = Path(project_root)
        else:
            self.project_root = Path(__file__).parent.parent

        # Storage for simulation results
        self.results_dir = self.project_root / "simulation_results"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Track active simulations
        self.active_simulations = {}

        # Register all tools
        self._register_tools()

    def _register_tools(self):
        """Register all available MCP tools"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="create_single_building",
                    description="Create configuration for a single building with HVAC and DER systems",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "building_id": {"type": "string", "default": "SFH_1"},
                            "cluster_id": {"type": "string", "default": "residential_cluster"},
                            "thermal_zones": {"type": "array", "items": {"type": "string"}, "default": ["zone0"]},
                            "hvac_cooling_setpoint": {"type": "number", "default": 24.0},
                            "hvac_heating_setpoint": {"type": "number", "default": 18.0},
                            "pv_capacity_kw": {"type": "number", "default": 10},
                            "battery_capacity_kwh": {"type": "number", "default": 10},
                            "ev_configs": {"type": "array", "default": []},
                            "save_config": {"type": "boolean", "default": True}
                        },
                        "required": ["building_id"]
                    }
                ),
                Tool(
                    name="create_multi_building",
                    description="Create configuration for multiple buildings with varied parameters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "num_buildings": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                            "cluster_id": {"type": "string", "default": "residential_cluster_multi"},
                            "vary_parameters": {"type": "boolean", "default": True},
                            "base_pv_capacity_kw": {"type": "number", "default": 10},
                            "base_battery_capacity_kwh": {"type": "number", "default": 8},
                            "variation_percentage": {"type": "number", "default": 0.3},
                            "save_config": {"type": "boolean", "default": True}
                        },
                        "required": ["num_buildings"]
                    }
                ),
                Tool(
                    name="run_simulation",
                    description="Run a building simulation with specified configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "config_path": {"type": "string"},
                            "simulation_name": {"type": "string"},
                            "duration_hours": {"type": "number", "default": 24},
                            "save_results": {"type": "boolean", "default": True}
                        },
                        "required": ["config_path", "simulation_name"]
                    }
                ),
                Tool(
                    name="analyze_simulation",
                    description="Analyze simulation results and get key metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "simulation_name": {"type": "string"},
                            "metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": ["comfort", "cost", "efficiency", "flexibility"]
                            }
                        },
                        "required": ["simulation_name"]
                    }
                ),
                Tool(
                    name="compare_simulations",
                    description="Compare two simulation results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "simulation_1": {"type": "string"},
                            "simulation_2": {"type": "string"},
                            "comparison_metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": ["comfort", "cost", "efficiency"]
                            }
                        },
                        "required": ["simulation_1", "simulation_2"]
                    }
                ),
                Tool(
                    name="update_configuration",
                    description="Update existing configuration parameters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "config_path": {"type": "string"},
                            "updates": {"type": "object"},
                            "save_as": {"type": "string"}
                        },
                        "required": ["config_path", "updates"]
                    }
                ),
                Tool(
                    name="get_simulation_status",
                    description="Get current status of active simulations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "simulation_name": {"type": "string", "default": "all"}
                        }
                    }
                ),
                Tool(
                    name="list_configurations",
                    description="List all available configuration files",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "default": "."}
                        }
                    }
                ),
                Tool(
                    name="list_simulations",
                    description="List all completed simulation results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 10}
                        }
                    }
                ),
                Tool(
                    name="optimize_control",
                    description="Suggest optimized control parameters based on simulation results",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "simulation_name": {"type": "string"},
                            "optimization_target": {
                                "type": "string",
                                "enum": ["comfort", "cost", "efficiency", "balanced"],
                                "default": "balanced"
                            }
                        },
                        "required": ["simulation_name"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict) -> List[TextContent]:
            try:
                if name == "create_single_building":
                    result = await self._create_single_building(**arguments)
                elif name == "create_multi_building":
                    result = await self._create_multi_building(**arguments)
                elif name == "run_simulation":
                    result = await self._run_simulation(**arguments)
                elif name == "analyze_simulation":
                    result = await self._analyze_simulation(**arguments)
                elif name == "compare_simulations":
                    result = await self._compare_simulations(**arguments)
                elif name == "update_configuration":
                    result = await self._update_configuration(**arguments)
                elif name == "get_simulation_status":
                    result = await self._get_simulation_status(**arguments)
                elif name == "list_configurations":
                    result = await self._list_configurations(**arguments)
                elif name == "list_simulations":
                    result = await self._list_simulations(**arguments)
                elif name == "optimize_control":
                    result = await self._optimize_control(**arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]

    async def _create_single_building(
        self,
        building_id: str,
        cluster_id: str = "residential_cluster",
        thermal_zones: List[str] = None,
        hvac_cooling_setpoint: float = 24.0,
        hvac_heating_setpoint: float = 18.0,
        pv_capacity_kw: float = 10,
        battery_capacity_kwh: float = 8,
        ev_configs: List[Dict] = None,
        save_config: bool = True
    ) -> Dict:
        """Create a single building configuration"""

        thermal_zones = thermal_zones or ["zone0"]
        cm = ConfigurationManager()

        # Create cluster and building
        cm.add_cluster(cluster_id, parameters={"location": "Syracuse, NY"})
        cm.add_building(
            cluster_id=cluster_id,
            building_id=building_id,
            parameters={"building_type": "single_family_home"},
            thermal_zones=thermal_zones,
            electrical_zones=["zone0"]
        )

        # Add thermal zone
        for zone in thermal_zones:
            cm.add_thermal_zone_module(
                building_id=building_id,
                zone_id=zone,
                parameters=self._get_thermal_zone_params(),
                class_path="bestopt.env.modules.building.thermalzone.ThermalDynamicsModule"
            )

        # Add electrical zone
        cm.add_electrical_zone_module(
            building_id=building_id,
            zone_id="zone0",
            parameters=self._get_electrical_zone_params(),
            class_path="bestopt.env.modules.building.electricalzone.ElectricalDynamicModule"
        )

        # Add HVAC system
        hvac_system_id = f"hvac_{building_id}"
        cm.add_system(
            cluster_id=cluster_id,
            system_id=hvac_system_id,
            system_type="hvac_systems",
            parameters=self._get_hvac_params(),
            class_path="bestopt.env.modules.hvac.system.FCU.FCUModule"
        )

        # Add DER system
        der_system_id = f"der_{building_id}"
        cm.add_system(
            cluster_id=cluster_id,
            system_id=der_system_id,
            system_type="der_systems",
            parameters=self._get_der_params(pv_capacity_kw, battery_capacity_kwh, ev_configs),
            class_path="bestopt.env.modules.ders.system.der.DERModule"
        )

        # Assign systems
        cm.assign_system_to_buildings(hvac_system_id, [building_id])
        cm.assign_system_to_buildings(der_system_id, [building_id])

        # Add controllers
        self._add_controllers(cm, hvac_system_id, der_system_id,
                            hvac_cooling_setpoint, hvac_heating_setpoint)

        # Add disturbances and environment
        self._add_disturbances(cm)
        self._add_environment(cm)

        # Select components
        cm.select_cluster(cluster_id)
        cm.select_buildings([building_id])
        cm.select_systems([hvac_system_id, der_system_id])
        cm.select_controller_for_system(hvac_system_id, f"ctrl_{hvac_system_id}")
        cm.select_controller_for_system(der_system_id, f"ctrl_{der_system_id}")
        cm.select_disturbances(["weather", "occupancy", "price"])
        cm.select_environment()

        # Save configuration
        config_path = None
        if save_config:
            config_path = str(self.results_dir / f"config_{building_id}_{datetime.now():%Y%m%d_%H%M%S}.json")
            cm.save_final_configuration(config_path)

        return {
            "status": "success",
            "building_id": building_id,
            "config_path": config_path,
            "summary": {
                "thermal_zones": thermal_zones,
                "hvac_setpoints": {"cooling": hvac_cooling_setpoint, "heating": hvac_heating_setpoint},
                "pv_capacity_kw": pv_capacity_kw,
                "battery_capacity_kwh": battery_capacity_kwh
            }
        }

    async def _create_multi_building(
        self,
        num_buildings: int,
        cluster_id: str = "residential_cluster_multi",
        vary_parameters: bool = True,
        base_pv_capacity_kw: float = 10,
        base_battery_capacity_kwh: float = 8,
        variation_percentage: float = 0.3,
        save_config: bool = True
    ) -> Dict:
        """Create configuration for multiple buildings"""

        import random
        cm = ConfigurationManager()
        cm.add_cluster(cluster_id, parameters={"location": "Syracuse, NY"})

        building_configs = []
        all_systems = []

        for i in range(1, num_buildings + 1):
            building_id = f"SFH_{i}"
            rng = random.Random(i * 100)  # Reproducible randomness

            # Vary parameters if requested
            if vary_parameters:
                pv_capacity = base_pv_capacity_kw * (1 + rng.uniform(-variation_percentage, variation_percentage))
                battery_capacity = base_battery_capacity_kwh * (1 + rng.uniform(-variation_percentage, variation_percentage))
                cooling_sp = 24.0 + rng.uniform(-2, 2)
                heating_sp = 18.0 + rng.uniform(-2, 2)
            else:
                pv_capacity = base_pv_capacity_kw
                battery_capacity = base_battery_capacity_kwh
                cooling_sp = 24.0
                heating_sp = 18.0

            # Create building
            cm.add_building(
                cluster_id=cluster_id,
                building_id=building_id,
                parameters={"building_type": "single_family_home", "building_number": i},
                thermal_zones=["zone0"],
                electrical_zones=["zone0"]
            )

            # Add zones
            cm.add_thermal_zone_module(
                building_id=building_id,
                zone_id="zone0",
                parameters=self._get_thermal_zone_params(),
                class_path="bestopt.env.modules.building.thermalzone.ThermalDynamicsModule"
            )

            cm.add_electrical_zone_module(
                building_id=building_id,
                zone_id="zone0",
                parameters=self._get_electrical_zone_params(),
                class_path="bestopt.env.modules.building.electricalzone.ElectricalDynamicModule"
            )

            # Create systems
            hvac_id = f"hvac_{building_id}"
            der_id = f"der_{building_id}"

            cm.add_system(
                cluster_id=cluster_id,
                system_id=hvac_id,
                system_type="hvac_systems",
                parameters=self._get_hvac_params(),
                class_path="bestopt.env.modules.hvac.system.FCU.FCUModule"
            )

            cm.add_system(
                cluster_id=cluster_id,
                system_id=der_id,
                system_type="der_systems",
                parameters=self._get_der_params(pv_capacity, battery_capacity),
                class_path="bestopt.env.modules.ders.system.der.DERModule"
            )

            cm.assign_system_to_buildings(hvac_id, [building_id])
            cm.assign_system_to_buildings(der_id, [building_id])

            # Add controllers
            self._add_controllers(cm, hvac_id, der_id, cooling_sp, heating_sp)

            all_systems.extend([hvac_id, der_id])
            building_configs.append({
                "building_id": building_id,
                "pv_capacity_kw": round(pv_capacity, 2),
                "battery_capacity_kwh": round(battery_capacity, 2),
                "cooling_setpoint": round(cooling_sp, 1),
                "heating_setpoint": round(heating_sp, 1)
            })

        # Add shared disturbances and environment
        self._add_disturbances(cm)
        self._add_environment(cm)

        # Select all components
        cm.select_cluster(cluster_id)
        cm.select_buildings([f"SFH_{i}" for i in range(1, num_buildings + 1)])
        cm.select_systems(all_systems)

        for i in range(1, num_buildings + 1):
            hvac_id = f"hvac_SFH_{i}"
            der_id = f"der_SFH_{i}"
            cm.select_controller_for_system(hvac_id, f"ctrl_{hvac_id}")
            cm.select_controller_for_system(der_id, f"ctrl_{der_id}")

        cm.select_disturbances(["weather", "occupancy", "price"])
        cm.select_environment()

        # Save configuration
        config_path = None
        if save_config:
            config_path = str(self.results_dir / f"config_{num_buildings}buildings_{datetime.now():%Y%m%d_%H%M%S}.json")
            cm.save_final_configuration(config_path)

        return {
            "status": "success",
            "num_buildings": num_buildings,
            "config_path": config_path,
            "building_configs": building_configs
        }

    async def _run_simulation(
        self,
        config_path: str,
        simulation_name: str,
        duration_hours: float = 24,
        save_results: bool = True
    ) -> Dict:
        """Run simulation with given configuration"""

        # Load configuration
        cm = ConfigurationManager(config_path)
        env = BESTOptEnvironment(cm.config)

        # Calculate steps (15-minute intervals)
        total_steps = int(duration_hours * 4)

        # Initialize data storage
        data = {
            'timesteps': [],
            'zone_temperatures': {},
            'hvac_power': {},
            'pv_generation': {},
            'battery_soc': {},
            'grid_import': {},
            'grid_export': {},
            'total_load': {}
        }

        # Track active simulation
        self.active_simulations[simulation_name] = {
            "status": "running",
            "progress": 0,
            "start_time": datetime.now()
        }

        # Run simulation
        for step in range(min(total_steps, env.total_step)):
            observations, done, info = env.step()

            # Collect basic metrics for all buildings
            for building_id in cm.selected_buildings:
                if building_id not in data['zone_temperatures']:
                    data['zone_temperatures'][building_id] = []
                    data['hvac_power'][building_id] = []
                    data['pv_generation'][building_id] = []
                    data['battery_soc'][building_id] = []
                    data['grid_import'][building_id] = []
                    data['total_load'][building_id] = []

                # Extract metrics (simplified for MCP)
                cluster_id = list(cm.selected_clusters.keys())[0]
                building_state = env.cluster_states[cluster_id].thermal.systems.get(f'{building_id}_building')

                if building_state:
                    zone_temp = building_state.components['zone0'].temperature
                    data['zone_temperatures'][building_id].append(zone_temp)

                    # Get HVAC power
                    hvac_system_id = env.building_system_map[building_id].get('thermal')
                    if hvac_system_id:
                        hvac_system = env.system_modules.get(hvac_system_id)
                        if hvac_system:
                            data['hvac_power'][building_id].append(hvac_system.FCU_power_total_W / 1000)

                    # Get DER metrics
                    der_system_id = env.building_system_map[building_id].get('electrical')
                    if der_system_id:
                        der_system = env.system_modules.get(der_system_id)
                        if der_system:
                            # PV generation
                            pv_gen = sum(pv.generation_w for pv in der_system.pv_states.values()) / 1000
                            data['pv_generation'][building_id].append(pv_gen)

                            # Battery SOC
                            for bat in der_system.battery_states.values():
                                data['battery_soc'][building_id].append(bat.soc * 100)
                                break

            data['timesteps'].append(step)

            # Update progress
            self.active_simulations[simulation_name]["progress"] = (step + 1) / total_steps * 100

            if done:
                break

        # Mark simulation as complete
        self.active_simulations[simulation_name]["status"] = "completed"
        self.active_simulations[simulation_name]["end_time"] = datetime.now()

        # Save results if requested
        save_path = None
        if save_results:
            save_path = str(self.results_dir / f"{simulation_name}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'metadata': {
                        'simulation_name': simulation_name,
                        'config_path': config_path,
                        'duration_hours': duration_hours,
                        'total_steps': len(data['timesteps']),
                        'timestamp': datetime.now().isoformat()
                    }
                }, f)

        return {
            "status": "success",
            "simulation_name": simulation_name,
            "total_steps": len(data['timesteps']),
            "save_path": save_path,
            "summary": {
                "avg_temperature": np.mean([np.mean(temps) for temps in data['zone_temperatures'].values()]),
                "total_pv_generation_kwh": sum(np.sum(gen) * 0.25 for gen in data['pv_generation'].values()),
                "avg_battery_soc": np.mean([np.mean(soc) for soc in data['battery_soc'].values() if soc])
            }
        }

    async def _analyze_simulation(
        self,
        simulation_name: str,
        metrics: List[str] = None
    ) -> Dict:
        """Analyze simulation results"""

        metrics = metrics or ["comfort", "cost", "efficiency", "flexibility"]

        # Load simulation data
        load_path = self.results_dir / f"{simulation_name}.pkl"
        if not load_path.exists():
            return {"error": f"Simulation {simulation_name} not found"}

        with open(load_path, 'rb') as f:
            saved_data = pickle.load(f)

        data = saved_data['data']
        analysis = {}

        # Comfort analysis
        if "comfort" in metrics:
            temps = []
            for building_temps in data['zone_temperatures'].values():
                temps.extend(building_temps)

            analysis['comfort'] = {
                "avg_temperature": np.mean(temps),
                "min_temperature": np.min(temps),
                "max_temperature": np.max(temps),
                "temperature_std": np.std(temps),
                "violations": sum(1 for t in temps if t < 18 or t > 26)  # Simple violation check
            }

        # Cost analysis (simplified)
        if "cost" in metrics:
            # Assume simple electricity price
            electricity_price = 0.15  # $/kWh
            total_import = sum(np.sum(imports) * 0.25 for imports in data.get('grid_import', {}).values())
            total_export = sum(np.sum(exports) * 0.25 for exports in data.get('grid_export', {}).values())

            analysis['cost'] = {
                "import_cost": total_import * electricity_price,
                "export_revenue": total_export * electricity_price * 0.5,  # Feed-in tariff
                "net_cost": (total_import * electricity_price) - (total_export * electricity_price * 0.5)
            }

        # Efficiency analysis
        if "efficiency" in metrics:
            total_pv = sum(np.sum(gen) * 0.25 for gen in data['pv_generation'].values())
            total_hvac = sum(np.sum(power) * 0.25 for power in data['hvac_power'].values())

            analysis['efficiency'] = {
                "total_pv_generation_kwh": total_pv,
                "total_hvac_consumption_kwh": total_hvac,
                "renewable_fraction": (total_pv / total_hvac * 100) if total_hvac > 0 else 0
            }

        # Flexibility analysis
        if "flexibility" in metrics:
            battery_socs = []
            for soc_list in data['battery_soc'].values():
                if soc_list:
                    battery_socs.extend(soc_list)

            analysis['flexibility'] = {
                "avg_battery_soc": np.mean(battery_socs) if battery_socs else 0,
                "min_battery_soc": np.min(battery_socs) if battery_socs else 0,
                "max_battery_soc": np.max(battery_socs) if battery_socs else 0,
                "battery_utilization": np.std(battery_socs) if battery_socs else 0
            }

        return {
            "status": "success",
            "simulation_name": simulation_name,
            "analysis": analysis
        }

    async def _compare_simulations(
        self,
        simulation_1: str,
        simulation_2: str,
        comparison_metrics: List[str] = None
    ) -> Dict:
        """Compare two simulations"""

        comparison_metrics = comparison_metrics or ["comfort", "cost", "efficiency"]

        # Analyze both simulations
        analysis_1 = await self._analyze_simulation(simulation_1, comparison_metrics)
        analysis_2 = await self._analyze_simulation(simulation_2, comparison_metrics)

        if "error" in analysis_1 or "error" in analysis_2:
            return {"error": "One or both simulations not found"}

        comparison = {
            "simulation_1": simulation_1,
            "simulation_2": simulation_2,
            "differences": {}
        }

        # Calculate differences
        for metric in comparison_metrics:
            if metric in analysis_1['analysis'] and metric in analysis_2['analysis']:
                comparison['differences'][metric] = {}
                for key in analysis_1['analysis'][metric]:
                    val1 = analysis_1['analysis'][metric][key]
                    val2 = analysis_2['analysis'][metric][key]

                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        diff = val2 - val1
                        pct_change = (diff / abs(val1) * 100) if val1 != 0 else 0
                        comparison['differences'][metric][key] = {
                            "value_1": val1,
                            "value_2": val2,
                            "difference": diff,
                            "percent_change": pct_change
                        }

        # Determine winner
        improvements = 0
        if "comfort" in comparison['differences']:
            if comparison['differences']['comfort'].get('violations', {}).get('difference', 0) < 0:
                improvements += 1
        if "cost" in comparison['differences']:
            if comparison['differences']['cost'].get('net_cost', {}).get('difference', 0) < 0:
                improvements += 1
        if "efficiency" in comparison['differences']:
            if comparison['differences']['efficiency'].get('renewable_fraction', {}).get('difference', 0) > 0:
                improvements += 1

        comparison['summary'] = {
            "improvements": improvements,
            "better_simulation": simulation_2 if improvements > len(comparison_metrics) / 2 else simulation_1
        }

        return comparison

    async def _update_configuration(
        self,
        config_path: str,
        updates: Dict,
        save_as: str = None
    ) -> Dict:
        """Update configuration parameters"""

        # Load existing configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Apply updates recursively
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v

        update_dict(config, updates)

        # Save updated configuration
        save_path = save_as or config_path.replace('.json', '_updated.json')
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)

        return {
            "status": "success",
            "original_config": config_path,
            "updated_config": save_path,
            "updates_applied": updates
        }

    async def _optimize_control(
        self,
        simulation_name: str,
        optimization_target: str = "balanced"
    ) -> Dict:
        """Suggest optimized control parameters"""

        # Analyze current simulation
        analysis = await self._analyze_simulation(simulation_name, ["comfort", "cost", "efficiency"])

        if "error" in analysis:
            return {"error": f"Simulation {simulation_name} not found"}

        suggestions = {}

        # Based on optimization target, suggest parameter adjustments
        if optimization_target == "comfort":
            suggestions["hvac"] = {
                "cooling_setpoint": 23.0,  # More aggressive cooling
                "heating_setpoint": 20.0,   # More aggressive heating
                "deadband": 0.3             # Tighter control
            }
        elif optimization_target == "cost":
            suggestions["hvac"] = {
                "cooling_setpoint": 26.0,  # Less cooling
                "heating_setpoint": 18.0,   # Less heating
                "precooling": {"degree": 2, "hours": 2}  # Use off-peak
            }
            suggestions["der"] = {
                "battery_charge_hours": "off_peak",
                "ev_v2g_enabled": True
            }
        elif optimization_target == "efficiency":
            suggestions["der"] = {
                "pv_capacity_increase": "20%",
                "battery_capacity_increase": "25%"
            }
            suggestions["hvac"] = {
                "mode": "economizer_when_available"
            }
        else:  # balanced
            avg_temp = analysis['analysis']['comfort']['avg_temperature']
            if avg_temp > 24:
                suggestions["hvac"] = {"cooling_setpoint": 23.5}
            elif avg_temp < 20:
                suggestions["hvac"] = {"heating_setpoint": 20.5}

            suggestions["der"] = {
                "mode": "self_consumption_priority",
                "battery_reserve": "20%"
            }

        return {
            "status": "success",
            "simulation_analyzed": simulation_name,
            "optimization_target": optimization_target,
            "current_performance": analysis['analysis'],
            "suggested_parameters": suggestions,
            "expected_improvements": {
                "comfort": "5-10% reduction in violations",
                "cost": "10-20% reduction in energy costs",
                "efficiency": "15-25% increase in self-sufficiency"
            }
        }

    async def _get_simulation_status(self, simulation_name: str = "all") -> Dict:
        """Get status of active simulations"""

        if simulation_name == "all":
            return {"active_simulations": self.active_simulations}
        elif simulation_name in self.active_simulations:
            return {simulation_name: self.active_simulations[simulation_name]}
        else:
            return {"error": f"Simulation {simulation_name} not found"}

    async def _list_configurations(self, directory: str = ".") -> Dict:
        """List available configuration files"""

        config_dir = Path(directory) if directory != "." else self.results_dir
        config_files = list(config_dir.glob("config_*.json"))

        configs = []
        for config_file in config_files:
            configs.append({
                "filename": config_file.name,
                "path": str(config_file),
                "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat(),
                "size_kb": config_file.stat().st_size / 1024
            })

        return {
            "total_configs": len(configs),
            "configurations": configs
        }

    async def _list_simulations(self, limit: int = 10) -> Dict:
        """List completed simulation results"""

        sim_files = list(self.results_dir.glob("*.pkl"))
        sim_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        sim_files = sim_files[:limit]

        simulations = []
        for sim_file in sim_files:
            # Skip non-simulation files
            if "config" in sim_file.name:
                continue

            simulations.append({
                "name": sim_file.stem,
                "path": str(sim_file),
                "created": datetime.fromtimestamp(sim_file.stat().st_mtime).isoformat(),
                "size_mb": sim_file.stat().st_size / (1024 * 1024)
            })

        return {
            "total_simulations": len(simulations),
            "simulations": simulations
        }

    # Helper methods for default parameters
    def _get_thermal_zone_params(self) -> Dict:
        """Get default thermal zone parameters"""
        import os
        return {
            "model_args": {
                "para": {"Int_h": 8, "Ext_h": 14, "epochs": 20},
                "modeltype": "PI-modnn",
                "startday": 1,
                "trainday": 180,
                "testday": 1,
                "datapath": os.path.join(self.project_root, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
                "temp_unit": "C",
                "device": "cpu",
                "save_name": "SFH_1"
            },
            "model_path": os.path.join(self.project_root, "examples", "Saved", "SFH_1",
                                       "Trained_mdlEnco48_Deco96", "PI-modnn_180daysTest_on07-01.pth"),
            "scaler_path": os.path.join(self.project_root, "examples", "Scaler", "SFH_1", "ModNN_scaler.pkl"),
            "historical_data_path": os.path.join(self.project_root, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
            "encoder_length": 48,
            "retrain": "Off",
            "simulation_start_time": "2023-08-01 00:00:00"
        }

    def _get_electrical_zone_params(self) -> Dict:
        """Get default electrical zone parameters"""
        return {
            "lighting": {"daytime": 600, "nighttime": 1800},
            "appliance": {
                "cooking": 2000,
                "tv": 200,
                "pc": 400,
                "dishwashing": 1000
            }
        }

    def _get_hvac_params(self) -> Dict:
        """Get default HVAC system parameters"""
        return {
            "system_name": "FCU System",
            "system_config": {
                "fan": {"rated_flow_m3s": 0.4, "rated_power_W": 400},
                "fan_ctrl": {"ctrl_type": "constant", "rated_flow_m3s": 0.4},
                "coil": {"effectiveness": 0.7},
                "pump": {"rated_flow_m3s": 0.01, "rated_power_W": 1500},
                "chiller": {"rated_capacity_W": 15000, "rated_cop": 4.5},
                "tower": {
                    "rated_capacity_W": 15000,
                    "rated_fan_power_W": 400,
                    "pump_power_per_flow": 85000,
                    "min_approach_C": 3.0,
                    "max_approach_C": 7.0
                }
            }
        }

    def _get_der_params(self, pv_capacity_kw: float, battery_capacity_kwh: float, ev_configs: List[Dict] = None) -> Dict:
        """Get DER system parameters"""

        if not ev_configs:
            ev_configs = [
                {
                    "id": "ev_tesla",
                    "rated_capacity_kWh": 60,
                    "initial_soc": 0.2,
                    "charge_speed": 0.25,
                    "discharge_speed": 0.5,
                    "charge_efficiency": 0.95,
                    "initially_connected": True
                },
                {
                    "id": "ev_nissan",
                    "rated_capacity_kWh": 40,
                    "initial_soc": 0.8,
                    "charge_speed": 0.25,
                    "discharge_speed": 0.5,
                    "charge_efficiency": 0.95,
                    "initially_connected": False
                }
            ]

        return {
            "system_name": "PV-Battery-EV System",
            "system_config": {
                "pv": {"rated_capacity_kW": pv_capacity_kw},
                "bat": {
                    "rated_capacity_kWh": battery_capacity_kwh,
                    "initial_soc": 0.3,
                    "charge_speed": 0.25,
                    "discharge_speed": 0.5,
                    "charge_efficiency": 0.95
                },
                "evs": ev_configs
            }
        }

    def _add_controllers(self, cm, hvac_system_id: str, der_system_id: str,
                        cooling_setpoint: float, heating_setpoint: float):
        """Add controllers for HVAC and DER systems"""

        # HVAC controller
        cm.add_system_controller(
            controller_id=f"ctrl_{hvac_system_id}",
            system_id=hvac_system_id,
            parameters={
                "domain": "thermal",
                "type": "rule-based",
                "mode": "cooling",
                "precooling": {"degree": 0, "hours": 0},
                "base_cooling": cooling_setpoint,
                "base_heating": heating_setpoint,
                "deadband": 0.5,
                "cooling_power_max": 4000.0,
                "heating_power_max": 4000.0
            },
            class_path="bestopt.env.controllers.thermal.SupervisoryController"
        )

        # DER controller
        cm.add_system_controller(
            controller_id=f"ctrl_{der_system_id}",
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
                "max_grid_export": 5000
            },
            class_path="bestopt.env.controllers.electrical.SupervisoryController"
        )

    def _add_disturbances(self, cm):
        """Add disturbances to configuration"""
        import os

        cm.add_disturbance(
            "weather",
            parameters={
                "file_path": os.path.join(self.project_root, "data", "SFH", "DIST", "weather", "weather.csv"),
                "simulation_start_time": "2023-08-01 00:00:00"
            },
            class_path="bestopt.env.disturbances.weather.WeatherModule"
        )

        cm.add_disturbance(
            "occupancy",
            parameters={
                "file_path": os.path.join(self.project_root, "data", "SFH", "DIST", "occupancy", "occupancy.csv"),
                "simulation_start_time": "2023-08-01 00:00:00"
            },
            class_path="bestopt.env.disturbances.occupancy.OccupancyModule"
        )

        cm.add_disturbance(
            "price",
            parameters={},
            class_path="bestopt.env.disturbances.price.PriceModule"
        )

    def _add_environment(self, cm):
        """Add environment configuration"""
        cm.add_environment(
            parameters={
                "resolution": 900,  # 15 minutes
                "duration": 86400,  # 1 day
                "enable_history": True,
                "logging_level": "INFO",
                "simulation_start_time": "2023-08-01 00:00:00"
            },
            class_path="bestopt.environment.BESTOptEnvironment"
        )

    async def run(self):
        """Run the MCP server"""
        init_opts = {
            "serverInfo": {"name": "bestopt-server", "version": "0.1.0"},
            "projectRoot": str(self.project_root),
            "resultsDir": str(self.results_dir),
        }
        logger.info("BESTOpt MCP server started. Waiting for an MCP client on stdio...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, init_opts)


# Main entry point
if __name__ == "__main__":
    import sys

    # Get project root from command line or use default
    project_root = sys.argv[1] if len(sys.argv) > 1 else None

    # Create and run server
    server = BESTOptMCPServer(project_root)
    asyncio.run(server.run())