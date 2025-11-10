"""
Simulation Tool for running building energy simulations
"""
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bestopt.env.core.config_manager import ConfigurationManager
from .base_tool import BaseTool, ToolResult


class SimulationTool(BaseTool):
    """
    Tool for running building energy simulations and calculating KPIs
    """

    def __init__(self, config_dir: Optional[Path] = None, simulation_module=None):
        super().__init__(
            name="simulation",
            description="Run building energy simulations and analyze results",
            parameters_schema={
                "action": {
                    "type": "string",
                    "enum": ["run", "analyze", "compare", "get_kpis"],
                    "description": "Action to perform"
                },
                "config": {
                    "type": "object",
                    "description": "Configuration to use for simulation"
                },
                "config_file": {
                    "type": "string",
                    "description": "Path to configuration file (alternative to config object)"
                },
                "parameters": {
                    "type": "object",
                    "description": "Simulation parameters",
                    "properties": {
                        "start_time": {"type": "string"},
                        "end_time": {"type": "string"},
                        "timestep": {"type": "integer"},
                        "save_results": {"type": "boolean"}
                    }
                },
                "baseline_results": {
                    "type": "object",
                    "description": "Baseline results for comparison"
                },
                "comparison_results": {
                    "type": "object",
                    "description": "Comparison results for analysis"
                }
            }
        )

        self.config_dir = config_dir or Path(".")
        self.config_manager = ConfigurationManager(self.config_dir)
        self.simulation_module = simulation_module  # Your actual simulation module
        self.last_results = None
        self.results_cache = {}

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute simulation operation"""
        try:
            action = parameters.get("action", "run")

            if action == "run":
                return await self._run_simulation(parameters)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Simulation operation failed: {str(e)}"
            )

    async def _run_simulation(self, parameters: Dict[str, Any]) -> ToolResult:
        """Run a building energy simulation"""
        try:
            PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))
            from bestopt.env.core.environment import BESTOptEnvironment
            from bestopt.env.core.config_manager import ConfigurationManager
            from bestopt.scripts.runtime_plotter import create_hvac_dashboard
            from bestopt.scripts.elec_plotter import create_electrical_dashboard
            config_path = os.path.join(PROJECT_ROOT_PATH, "bestopt", "examples", "SFH_1_Building", "config_setup.json")
            cm = ConfigurationManager(config_path)
            env = BESTOptEnvironment(cm.config)
            hvac_plotter = create_hvac_dashboard(max_points=96 * 1, window_title="HVAC System Monitor")
            electrical_plotter = create_electrical_dashboard(max_points=96 * 1,
                                                             window_title="Electrical System Monitor")
            # Run Simulation
            for timestep in range(env.total_step):
                # Step the environment
                observations, done, info = env.step()
                # Get building and system IDs
                cluster_id = 'residential_cluster_1'
                building_id = 'SFH_1'
                hvac_system_id = env.building_system_map[building_id].get('thermal')
                der_system_id = env.building_system_map[building_id].get('electrical')
                # Extract zone temperature
                zone_temperature = env.cluster_states[cluster_id].thermal.systems['SFH_1_building'].components[
                    'zone0'].temperature
                # Get supervisory setpoints from action
                cooling_setpoint = env.cluster_actions[cluster_id].thermal.system_actions[
                    'hvac_system_1'].cooling_setpoint_c
                heating_setpoint = env.cluster_actions[cluster_id].thermal.system_actions[
                    'hvac_system_1'].heating_setpoint_c
                # Get HVAC system module
                hvac_system = env.system_modules[hvac_system_id]
                der_system = env.system_modules[der_system_id]
                # Get supply air parameters (actual values from HVAC system)
                supply_air_temp_real = hvac_system.SAT_actual_C
                supply_air_flow_real = hvac_system.SA_flow_actual_m3s
                # Get HVAC performance metrics
                HVAC_power = hvac_system.FCU_power_total_W
                HVAC_thermal_load = hvac_system.Q_zone_actual_W
                # Get supervisory supply air setpoints
                supply_air_temp_setpt = env.cluster_actions[cluster_id].thermal.system_actions[
                    'hvac_system_1'].supply_temp_setpoint_c
                supply_air_flow_setpt = env.cluster_actions[cluster_id].thermal.system_actions[
                    'hvac_system_1'].supply_airflow_setpoint_m3s
                # Get electrical system data
                bat_soc = der_system.battery_states['bat_1'].soc
                ev_tesla_soc = der_system.ev_states['ev_tesla'].soc
                ev_nissan_soc = der_system.ev_states['ev_nissan'].soc
                pv_generation = der_system.pv_states['pv_1'].generation_w / 1000  # Convert to kW
                # Get loads (all in kW)
                building_total_load = HVAC_power / 1000 + \
                                      env.cluster_states['residential_cluster_1'].electrical.systems[
                                          'SFH_1_building'].components['electrical'].building_power_w / 1000
                HVAC_power_kw = HVAC_power / 1000  # Convert HVAC power to kW
                cooking = env.electrical_zone_modules['SFH_1.zone0'].cooking_power / 1000
                pc = env.electrical_zone_modules['SFH_1.zone0'].pc_power / 1000
                tv = env.electrical_zone_modules['SFH_1.zone0'].tv_power / 1000
                lighting = env.electrical_zone_modules['SFH_1.zone0'].lighting_power / 1000
                # Get peak signal
                is_peak = env.disturbance.prices.peaksignal  # True or False
                # Update the HVAC dashboard
                hvac_plotter.add_data_point(
                    timestep=timestep,
                    zone_temperature=zone_temperature,
                    hvac_thermal_load=HVAC_thermal_load,
                    hvac_power=HVAC_power,
                    supervisory_cooling_setpoint=cooling_setpoint,
                    supervisory_heating_setpoint=heating_setpoint,
                    supply_air_temp_real=supply_air_temp_real,
                    supply_air_temp_setpt=supply_air_temp_setpt,
                    supply_air_flow_real=supply_air_flow_real,
                    supply_air_flow_setpt=supply_air_flow_setpt
                )
                # Update the Electrical dashboard
                electrical_plotter.add_data_point(
                    timestep=timestep,
                    pv_generation_kw=pv_generation,
                    battery_soc=bat_soc,
                    ev_tesla_soc=ev_tesla_soc,
                    ev_nissan_soc=ev_nissan_soc,
                    total_load_kw=building_total_load,
                    hvac_power_kw=HVAC_power_kw,
                    cooking_kw=cooking,
                    pc_kw=pc,
                    tv_kw=tv,
                    lighting_kw=lighting,
                    is_peak=is_peak
                )
                # Print progress every 100 steps
                if timestep % 100 == 0:
                    print(f"Step {timestep}/{env.total_step}: "
                          f"Temp={zone_temperature:.1f}°C, "
                          f"PV={pv_generation:.2f}kW, "
                          f"Load={building_total_load:.2f}kW, "
                          f"Peak={'Yes' if is_peak else 'No'}")
                if done:
                    break

            result_id = f"sim_{datetime.now().timestamp()}"
            return ToolResult(
                success=True,
                data={
                    "simulation_id": result_id,
                    "results": None,
                    "kpis": None,
                    "parameters": None,
                    "message": "Simulation completed successfully"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to run simulation: {str(e)}"
            )

