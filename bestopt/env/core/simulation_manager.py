"""
Revised SimulationManager class optimized for MCP Server integration
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class SimulationManager:
    """Simplified SimulationManager for MCP Server integration"""

    def __init__(self, base_path: str = "./simulation_results"):
        """Initialize the simulation manager

        Parameters:
        -----------
        base_path : str
            Base directory for saving simulation results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Store running simulations data in memory
        self.simulation_data = {}
        self.analysis_cache = {}

    def extract_simulation_data(self, env, timestep: int) -> Dict[str, Any]:
        """Extract data from environment at current timestep

        Parameters:
        -----------
        env : BESTOptEnvironment
            The environment object
        timestep : int
            Current simulation timestep

        Returns:
        --------
        data : dict
            Extracted data for this timestep
        """
        # Get building and system IDs
        cluster_id = 'residential_cluster_1'
        building_id = 'SFH_1'

        # Initialize data dict for this timestep
        step_data = {}

        try:
            hvac_system_id = env.building_system_map[building_id].get('thermal')
            der_system_id = env.building_system_map[building_id].get('electrical')

            # Extract zone data
            zone_temperature = env.cluster_states[cluster_id].thermal.systems['SFH_1_building'].components[
                'zone0'].temperature
            cooling_setpoint = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].cooling_setpoint_c
            heating_setpoint = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].heating_setpoint_c

            # Get system modules
            hvac_system = env.system_modules[hvac_system_id]
            der_system = env.system_modules[der_system_id]

            # Get HVAC system parameters
            HVAC_power = hvac_system.FCU_power_total_W
            HVAC_thermal_load = hvac_system.Q_zone_actual_W

            # Get supply air parameters
            supply_air_temp_actual = hvac_system.SAT_actual_C
            supply_air_flow_actual = hvac_system.SA_flow_actual_m3s
            supply_air_temp_setpoint = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].supply_temp_setpoint_c
            supply_air_flow_setpoint = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].supply_airflow_setpoint_m3s

            # Get water flow parameters
            water_flow_actual = hvac_system.CHW_flow_actual_m3s
            chiller_supply_water_temp = hvac_system.CHW_supply_temp_C

            # Get electrical data
            bat_soc = der_system.battery_states['bat_1'].soc
            ev_tesla_soc = der_system.ev_states['ev_tesla'].soc
            ev_nissan_soc = der_system.ev_states['ev_nissan'].soc

            # Get loads in kW
            HVAC_power_kw = HVAC_power / 1000
            cooking = env.electrical_zone_modules['SFH_1.zone0'].cooking_power / 1000
            pc = env.electrical_zone_modules['SFH_1.zone0'].pc_power / 1000
            tv = env.electrical_zone_modules['SFH_1.zone0'].tv_power / 1000
            lighting = env.electrical_zone_modules['SFH_1.zone0'].lighting_power / 1000

            building_total_load = HVAC_power_kw + env.cluster_states['residential_cluster_1'].electrical.systems[
                'SFH_1_building'].components['electrical'].building_power_w / 1000

            # Get peak signal and price
            is_peak = env.disturbance.prices.peaksignal
            electricity_price = env.disturbance.prices.electricity_price

            # Get power flows
            grid2ev_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2ev
            grid2battery_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2battery
            grid2ev = sum(grid2ev_dict.values())
            grid2battery = sum(grid2battery_dict.values())
            grid2building = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2building

            pv_generation = der_system.pv_states['pv_1'].generation_w / 1000
            grid_import = grid2ev + grid2battery + grid2building

            # Get PV distribution
            pv2ev_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2ev
            pv2battery_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2battery
            pv2ev = sum(pv2ev_dict.values())
            pv2battery = sum(pv2battery_dict.values())
            pv2building = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2building
            pv2grid = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2grid

            # Store all data
            step_data = {
                'timestep': timestep,
                'zone_temperature': zone_temperature,
                'cooling_setpoint': cooling_setpoint,
                'heating_setpoint': heating_setpoint,
                'hvac_thermal_load': HVAC_thermal_load,
                'hvac_power': HVAC_power,
                'supply_air_temp_setpoint': supply_air_temp_setpoint,
                'supply_air_temp_actual': supply_air_temp_actual,
                'supply_air_flow_setpoint': supply_air_flow_setpoint,
                'supply_air_flow_actual': supply_air_flow_actual,
                'chilled_water_flow_actual': water_flow_actual,
                'chiller_supply_water_temp': chiller_supply_water_temp,
                'battery_soc': bat_soc,
                'ev_tesla_soc': ev_tesla_soc,
                'ev_nissan_soc': ev_nissan_soc,
                'lighting': lighting,
                'cooking': cooking,
                'pc': pc,
                'tv': tv,
                'hvac_power_kw': HVAC_power_kw,
                'total_building_load': building_total_load,
                'is_peak': is_peak,
                'electricity_price': electricity_price,
                'grid2battery': grid2battery,
                'grid2ev': grid2ev,
                'pv_generation': pv_generation,
                'pv2building': pv2building,
                'pv2battery': pv2battery,
                'pv2ev': pv2ev,
                'pv2grid': pv2grid,
                'grid2building': grid2building,
                'grid_import': grid_import
            }
        except Exception as e:
            print(f"Error extracting data at timestep {timestep}: {str(e)}")
            step_data = {'timestep': timestep, 'error': str(e)}

        return step_data

    def initialize_simulation(self, simulation_name: str) -> None:
        """Initialize a new simulation data structure

        Parameters:
        -----------
        simulation_name : str
            Name for this simulation
        """
        if simulation_name not in self.simulation_data:
            self.simulation_data[simulation_name] = {
                'data': [],
                'metadata': {
                    'simulation_name': simulation_name,
                    'start_time': datetime.now().isoformat(),
                    'status': 'initialized'
                }
            }

    def add_simulation_step(self, simulation_name: str, env, timestep: int) -> Dict:
        """Add a single simulation step data

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation
        env : BESTOptEnvironment
            The environment object
        timestep : int
            Current timestep

        Returns:
        --------
        step_data : dict
            Data from this timestep
        """
        if simulation_name not in self.simulation_data:
            self.initialize_simulation(simulation_name)

        step_data = self.extract_simulation_data(env, timestep)
        self.simulation_data[simulation_name]['data'].append(step_data)
        return step_data

    def finalize_simulation(self, simulation_name: str) -> Dict:
        """Finalize simulation and convert to arrays

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation

        Returns:
        --------
        data_dict : dict
            Dictionary with arrays for each variable
        """
        if simulation_name not in self.simulation_data:
            return {}

        sim_data = self.simulation_data[simulation_name]
        sim_data['metadata']['end_time'] = datetime.now().isoformat()
        sim_data['metadata']['status'] = 'completed'
        sim_data['metadata']['total_steps'] = len(sim_data['data'])

        # Convert list of dicts to dict of arrays
        data_dict = {}
        if sim_data['data']:
            keys = sim_data['data'][0].keys()
            for key in keys:
                if key != 'error':
                    data_dict[key] = np.array([d.get(key, np.nan) for d in sim_data['data']])

        # Save to disk
        save_path = self.base_path / "operation" / f"{simulation_name}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'data': data_dict, 'metadata': sim_data['metadata']}, f)

        # Also save as CSV
        if data_dict:
            df = pd.DataFrame(data_dict)
            df.to_csv(self.base_path / "operation" / f"{simulation_name}.csv", index=False)

        return data_dict

    def analyze_comfort(self, simulation_name: str) -> Dict:
        """Analyze comfort metrics for a simulation

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation

        Returns:
        --------
        comfort_analysis : dict
            Comfort metrics
        """
        data_dict = self.get_simulation_data(simulation_name)
        if not data_dict:
            return {"error": "Simulation not found"}

        comfort = {}
        zone_temp = data_dict.get('zone_temperature', np.array([]))
        cool_sp = data_dict.get('cooling_setpoint', np.array([]))
        heat_sp = data_dict.get('heating_setpoint', np.array([]))

        if len(zone_temp) == 0:
            return {"error": "No temperature data available"}

        # Temperature violations
        comfort['temp_above_cooling_frq'] = int(np.sum(zone_temp > cool_sp))
        comfort['temp_below_heating_frq'] = int(np.sum(zone_temp < heat_sp))
        comfort['total_violations_frq'] = comfort['temp_above_cooling_frq'] + comfort['temp_below_heating_frq']
        comfort['violation_rate'] = float(comfort['total_violations_frq'] / len(zone_temp) * 100)

        comfort['temp_above_cooling_c_h'] = float(np.sum(np.maximum(zone_temp - cool_sp, 0)) * 0.25)
        comfort['temp_below_heating_c_h'] = float(np.sum(np.maximum(heat_sp - zone_temp, 0)) * 0.25)
        comfort['total_violation_c_h'] = comfort['temp_above_cooling_c_h'] + comfort['temp_below_heating_c_h']

        # Comfort metrics
        comfort['avg_temperature'] = float(np.mean(zone_temp))
        comfort['min_temperature'] = float(np.min(zone_temp))
        comfort['max_temperature'] = float(np.max(zone_temp))
        comfort['temp_std'] = float(np.std(zone_temp))
        comfort['temp_range'] = float(np.max(zone_temp) - np.min(zone_temp))

        return comfort

    def analyze_energy(self, simulation_name: str) -> Dict:
        """Analyze energy metrics for a simulation

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation

        Returns:
        --------
        energy_analysis : dict
            Energy metrics
        """
        data_dict = self.get_simulation_data(simulation_name)
        if not data_dict:
            return {"error": "Simulation not found"}

        energy = {}
        time_step_hours = 0.25

        # Total consumption
        total_load = data_dict.get('total_building_load', np.array([]))
        if len(total_load) > 0:
            energy['total_consumption_kwh'] = float(np.sum(total_load) * time_step_hours)
            energy['peak_demand_kw'] = float(np.max(total_load))
            energy['avg_demand_kw'] = float(np.mean(total_load))
            energy['load_factor'] = float(np.mean(total_load) / np.max(total_load) * 100) if np.max(
                total_load) > 0 else 0.0

        # PV metrics
        pv_gen = data_dict.get('pv_generation', np.array([]))
        if len(pv_gen) > 0:
            energy['pv_total_generation_kwh'] = float(np.sum(pv_gen) * time_step_hours)
            energy['pv_peak_output_kw'] = float(np.max(pv_gen))
            energy['pv_capacity_factor'] = float(np.mean(pv_gen) / np.max(pv_gen) * 100) if np.max(pv_gen) > 0 else 0.0

        # Self-consumption
        pv2building = data_dict.get('pv2building', np.array([]))
        pv2battery = data_dict.get('pv2battery', np.array([]))
        pv2ev = data_dict.get('pv2ev', np.array([]))

        if len(pv2building) > 0:
            self_consumption = np.sum(pv2building + pv2battery + pv2ev) * time_step_hours
            energy['self_consumption_kwh'] = float(self_consumption)
            energy['self_consumption_rate'] = float(
                self_consumption / energy.get('pv_total_generation_kwh', 1) * 100) if energy.get(
                'pv_total_generation_kwh', 0) > 0 else 0.0

        # Grid interaction
        grid_import = data_dict.get('grid_import', np.array([]))
        pv2grid = data_dict.get('pv2grid', np.array([]))

        if len(grid_import) > 0:
            energy['grid_import_total_kwh'] = float(np.sum(grid_import) * time_step_hours)
            energy['grid_export_total_kwh'] = float(np.sum(pv2grid) * time_step_hours)
            energy['net_grid_energy_kwh'] = energy['grid_import_total_kwh'] - energy['grid_export_total_kwh']

        # Renewable fraction
        grid2building = data_dict.get('grid2building', np.array([]))
        if len(grid2building) > 0 and len(pv2building) > 0:
            grid2building_energy = np.sum(grid2building) * time_step_hours
            pv2building_energy = np.sum(pv2building) * time_step_hours
            building_energy = energy.get('total_consumption_kwh', 1)

            energy['grid_dependency'] = float(
                (grid2building_energy / building_energy) * 100) if building_energy > 0 else 0.0
            energy['renewable_fraction'] = float(
                (pv2building_energy / building_energy) * 100) if building_energy > 0 else 0.0

        return energy

    def analyze_cost(self, simulation_name: str) -> Dict:
        """Analyze cost metrics for a simulation

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation

        Returns:
        --------
        cost_analysis : dict
            Cost metrics
        """
        data_dict = self.get_simulation_data(simulation_name)
        if not data_dict:
            return {"error": "Simulation not found"}

        cost = {}
        time_step_hours = 0.25

        grid_import = data_dict.get('grid_import', np.array([]))
        pv2grid = data_dict.get('pv2grid', np.array([]))
        electricity_price = data_dict.get('electricity_price', np.ones_like(grid_import) * 0.15)
        is_peak = data_dict.get('is_peak', np.zeros_like(grid_import)).astype(bool)

        if len(grid_import) > 0:
            # Energy costs
            cost['import_cost'] = float(np.sum(grid_import * electricity_price * time_step_hours) / 100)

            # Export revenue (assuming feed-in tariff is 50% of import price)
            feed_in_tariff = electricity_price * 0.5
            cost['export_revenue'] = float(np.sum(pv2grid * feed_in_tariff * time_step_hours) / 100)
            cost['net_cost'] = cost['import_cost'] - cost['export_revenue']

            # Peak charges
            if is_peak.any():
                cost['peak_consumption_kwh'] = float(np.sum(grid_import[is_peak]) * time_step_hours)
                cost['off_peak_consumption_kwh'] = float(np.sum(grid_import[~is_peak]) * time_step_hours)
                cost['peak_charges'] = float(cost['peak_consumption_kwh'] * 0.05)  # Additional peak charge
            else:
                cost['peak_consumption_kwh'] = 0.0
                cost['off_peak_consumption_kwh'] = float(np.sum(grid_import) * time_step_hours)
                cost['peak_charges'] = 0.0

        return cost

    def analyze_flexibility(self, simulation_name: str) -> Dict:
        """Analyze flexibility metrics for a simulation

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation

        Returns:
        --------
        flexibility_analysis : dict
            Flexibility metrics
        """
        data_dict = self.get_simulation_data(simulation_name)
        if not data_dict:
            return {"error": "Simulation not found"}

        flexibility = {}

        def _battery_efc_from_soc(soc: np.ndarray) -> float:
            if soc is None or len(soc) < 2:
                return 0.0
            soc = np.asarray(soc, dtype=float)
            soc = np.clip(soc, 0.0, 1.0)
            throughput_soc = np.sum(np.abs(np.diff(soc)))
            efc = throughput_soc / 2.0
            return float(efc)

        # Battery flexibility
        battery_soc = data_dict.get('battery_soc', np.array([]))
        if len(battery_soc) > 0:
            flexibility['battery_soc_avg'] = float(np.mean(battery_soc) * 100)
            flexibility['battery_soc_min'] = float(np.min(battery_soc) * 100)
            flexibility['battery_soc_max'] = float(np.max(battery_soc) * 100)
            flexibility['battery_cycles_efc'] = _battery_efc_from_soc(battery_soc)

        # EV flexibility
        ev_tesla_soc = data_dict.get('ev_tesla_soc', np.array([]))
        ev_nissan_soc = data_dict.get('ev_nissan_soc', np.array([]))
        if len(ev_tesla_soc) > 0:
            flexibility['ev_tesla_cycles_efc'] = _battery_efc_from_soc(ev_tesla_soc)
        if len(ev_nissan_soc) > 0:
            flexibility['ev_nissan_cycles_efc'] = _battery_efc_from_soc(ev_nissan_soc)

        # Load shifting capability
        total_load = data_dict.get('total_building_load', np.array([]))
        if len(total_load) > 0:
            flexibility['load_factor'] = float(np.mean(total_load) / np.max(total_load) * 100) if np.max(
                total_load) > 0 else 0.0
            flexibility['peak_to_average_ratio'] = float(np.max(total_load) / np.mean(total_load)) if np.mean(
                total_load) > 0 else 0.0

        # Peak reduction potential
        is_peak = data_dict.get('is_peak', np.array([])).astype(bool)
        grid_import = data_dict.get('grid_import', np.array([]))
        if len(grid_import) > 0 and is_peak.any():
            flexibility['peak_max_kw'] = float(np.max(grid_import[is_peak]))
            flexibility['off_peak_max_kw'] = float(np.max(grid_import[~is_peak])) if (~is_peak).any() else 0.0
            flexibility['peak_reduction'] = float(
                (1 - flexibility['peak_max_kw'] / np.max(grid_import)) * 100) if np.max(grid_import) > 0 else 0.0

        return flexibility

    def get_simulation_data(self, simulation_name: str) -> Dict:
        """Get simulation data either from memory or disk

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation

        Returns:
        --------
        data_dict : dict
            Simulation data dictionary
        """
        # Check if in memory
        if simulation_name in self.simulation_data:
            return self.finalize_simulation(simulation_name)

        # Try to load from disk
        load_path = self.base_path / 'operation' / f"{simulation_name}.pkl"
        if load_path.exists():
            with open(load_path, 'rb') as f:
                saved_data = pickle.load(f)
                return saved_data.get('data', {})

        return {}

    def compare_simulations(self, sim1: str, sim2: str) -> Dict:
        """Compare two simulations

        Parameters:
        -----------
        sim1 : str
            Name of first simulation
        sim2 : str
            Name of second simulation

        Returns:
        --------
        comparison : dict
            Comparison results
        """
        # Get analyses for both simulations
        comfort1 = self.analyze_comfort(sim1)
        comfort2 = self.analyze_comfort(sim2)
        energy1 = self.analyze_energy(sim1)
        energy2 = self.analyze_energy(sim2)
        cost1 = self.analyze_cost(sim1)
        cost2 = self.analyze_cost(sim2)
        flexibility1 = self.analyze_flexibility(sim1)
        flexibility2 = self.analyze_flexibility(sim2)

        comparison = {
            'simulation_1': sim1,
            'simulation_2': sim2,
            'comfort_differences': {},
            'energy_differences': {},
            'cost_differences': {},
            'flexibility_differences': {}
        }

        # Comfort differences
        if 'error' not in comfort1 and 'error' not in comfort2:
            comparison['comfort_differences'] = {
                'violation_rate_diff': comfort2['violation_rate'] - comfort1['violation_rate'],
                'avg_temperature_diff': comfort2['avg_temperature'] - comfort1['avg_temperature'],
                'total_violation_c_h_diff': comfort2['total_violation_c_h'] - comfort1['total_violation_c_h']
            }

        # Energy differences
        if 'error' not in energy1 and 'error' not in energy2:
            comparison['energy_differences'] = {
                'total_consumption_diff': energy2.get('total_consumption_kwh', 0) - energy1.get('total_consumption_kwh',
                                                                                                0),
                'peak_demand_diff': energy2.get('peak_demand_kw', 0) - energy1.get('peak_demand_kw', 0),
                'renewable_fraction_diff': energy2.get('renewable_fraction', 0) - energy1.get('renewable_fraction', 0),
                'self_consumption_rate_diff': energy2.get('self_consumption_rate', 0) - energy1.get(
                    'self_consumption_rate', 0)
            }

        # Cost differences
        if 'error' not in cost1 and 'error' not in cost2:
            comparison['cost_differences'] = {
                'net_cost_diff': cost2.get('net_cost', 0) - cost1.get('net_cost', 0),
                'import_cost_diff': cost2.get('import_cost', 0) - cost1.get('import_cost', 0),
                'export_revenue_diff': cost2.get('export_revenue', 0) - cost1.get('export_revenue', 0)
            }

        # Flexibility differences
        if 'error' not in flexibility1 and 'error' not in flexibility2:
            comparison['flexibility_differences'] = {
                'battery_cycles_diff': flexibility2.get('battery_cycles_efc', 0) - flexibility1.get(
                    'battery_cycles_efc', 0),
                'load_factor_diff': flexibility2.get('load_factor', 0) - flexibility1.get('load_factor', 0),
                'peak_reduction_diff': flexibility2.get('peak_reduction', 0) - flexibility1.get('peak_reduction', 0)
            }

        return comparison