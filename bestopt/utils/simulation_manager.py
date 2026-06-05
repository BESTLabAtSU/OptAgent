import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

class SimulationManager:
    """Manager class for building simulation operations"""

    def __init__(self, base_path: str = "./simulation_results"):
        """Initialize the simulation manager

        Parameters:
        -----------
        base_path : str
            Base directory for saving simulation results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Define publication-quality color schemes
        self.thermal_colors = {
            'zone_temp': '#2C3E50',  # Dark blue-gray
            'setpoint': '#7F8C8D',  # Gray
            'thermal_load': '#E74C3C',  # Red
            'power': '#3498DB',  # Blue
            'supply_temp': '#16A085',  # Teal
            'flow': '#8E44AD'  # Purple
        }

        self.electrical_colors = plt.cm.viridis(np.linspace(0.2, 0.9, 10))
        self.pv_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 5))

    def run_and_save_simulation(self,
                                env,
                                total_steps: int = 96,
                                simulation_name: str = None,
                                config_info: Dict = None) -> Tuple[Dict, str]:
        """
        Function 1: Run simulation and store the data (Enhanced with HVAC parameters)

        Parameters:
        -----------
        env : BESTOptEnvironment
            The environment object
        total_steps : int
            Number of simulation steps (default 96 for one day)
        simulation_name : str
            Name for this simulation run
        config_info : dict
            Additional configuration information to store

        Returns:
        --------
        data_dict : dict
            Dictionary containing all simulation data
        save_path : str
            Path where data was saved
        """
        if simulation_name is None:
            simulation_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Starting simulation: {simulation_name}")

        # Initialize data storage (including new HVAC parameters)
        data_dict = {
            # Thermal zone data
            'zone_temperature': [],
            'cooling_setpoint': [],
            'heating_setpoint': [],
            'hvac_thermal_load': [],
            'hvac_power': [],

            # New HVAC system data
            'supply_air_temp_setpoint': [],
            'supply_air_temp_actual': [],
            'supply_air_flow_setpoint': [],
            'supply_air_flow_actual': [],
            'chilled_water_flow_actual': [],
            'chiller_supply_water_temp': [],

            # Electrical system data
            'battery_soc': [],
            'ev_tesla_soc': [],
            'ev_nissan_soc': [],
            'lighting': [],
            'cooking': [],
            'pc': [],
            'tv': [],
            'hvac_power_kw': [],
            'total_building_load': [],

            # Grid interaction
            'grid2battery': [],
            'grid2ev': [],
            'grid2building': [],
            'grid_import': [],
            'is_peak': [],
            'electricity_price': [],

            # PV system
            'pv_generation': [],
            'pv2building': [],
            'pv2battery': [],
            'pv2ev': [],
            'pv2grid': [],

            'timesteps': []
        }

        # Run simulation
        for timestep in range(min(total_steps, env.total_step)):
            # Step the environment
            observations, done, info = env.step()

            # Get building and system IDs
            cluster_id = 'residential_cluster_1'
            building_id = 'SFH_1'
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

            # Get supply air parameters (actual values)
            supply_air_temp_actual = hvac_system.SAT_actual_C
            supply_air_flow_actual = hvac_system.SA_flow_actual_m3s

            # Get supply air setpoints
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
            data_dict['timesteps'].append(timestep)
            data_dict['zone_temperature'].append(zone_temperature)
            data_dict['cooling_setpoint'].append(cooling_setpoint)
            data_dict['heating_setpoint'].append(heating_setpoint)
            data_dict['hvac_thermal_load'].append(HVAC_thermal_load)
            data_dict['hvac_power'].append(HVAC_power)

            # Store new HVAC parameters
            data_dict['supply_air_temp_setpoint'].append(supply_air_temp_setpoint)
            data_dict['supply_air_temp_actual'].append(supply_air_temp_actual)
            data_dict['supply_air_flow_setpoint'].append(supply_air_flow_setpoint)
            data_dict['supply_air_flow_actual'].append(supply_air_flow_actual)
            data_dict['chilled_water_flow_actual'].append(water_flow_actual)
            data_dict['chiller_supply_water_temp'].append(chiller_supply_water_temp)

            # Store electrical data
            data_dict['battery_soc'].append(bat_soc)
            data_dict['ev_tesla_soc'].append(ev_tesla_soc)
            data_dict['ev_nissan_soc'].append(ev_nissan_soc)
            data_dict['lighting'].append(lighting)
            data_dict['cooking'].append(cooking)
            data_dict['pc'].append(pc)
            data_dict['tv'].append(tv)
            data_dict['hvac_power_kw'].append(HVAC_power_kw)
            data_dict['total_building_load'].append(building_total_load)
            data_dict['is_peak'].append(is_peak)
            data_dict['electricity_price'].append(electricity_price)
            data_dict['grid2battery'].append(grid2battery)
            data_dict['grid2ev'].append(grid2ev)
            data_dict['pv_generation'].append(pv_generation)
            data_dict['pv2building'].append(pv2building)
            data_dict['pv2battery'].append(pv2battery)
            data_dict['pv2ev'].append(pv2ev)
            data_dict['pv2grid'].append(pv2grid)
            data_dict['grid2building'].append(grid2building)
            data_dict['grid_import'].append(grid_import)

            if timestep % 20 == 0:
                print(f"Step {timestep}/{total_steps}: Temp={zone_temperature:.1f}°C, "
                      f"SAT={supply_air_temp_actual:.1f}°C, Battery SOC={bat_soc:.1%}")

            if done:
                break

        # Convert lists to numpy arrays
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

        # Save LLM metrics if available
        try:
            llm_matrix = env.system_controllers["hvac_system_1"].llm_controller.metrics_history
            save_path = Path(self.base_path) / "llm" / f"{simulation_name}.pkl"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'llm_matrix': llm_matrix}, f)
        except:
            pass

        # Add metadata
        metadata = {
            'simulation_name': simulation_name,
            'timestamp': datetime.now().isoformat(),
            'total_steps': len(data_dict['timesteps']),
            'config_info': config_info or {}
        }

        # Save data
        save_path = Path(self.base_path) / "operation" / f"{simulation_name}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'data': data_dict, 'metadata': metadata}, f)

        # Also save as CSV for easy viewing
        df = pd.DataFrame(data_dict)
        df.to_csv(self.base_path / "operation" / f"{simulation_name}.csv", index=False)

        print(f"\nSimulation completed: {len(data_dict['timesteps'])} steps")
        print(f"Data saved to: {save_path}")

        return data_dict, str(save_path)

    def plot_thermal_results(self,
                             simulation_name: str = None,
                             data_dict: Dict = None,
                             save_figure: bool = True,
                             show_figure: bool = True) -> plt.Figure:
        """
        Create publication-quality thermal system plots with proper colors
        """
        if data_dict is None:
            data_dict, metadata = self.load_simulation_data(simulation_name)

        # Create time axis
        timesteps = data_dict['timesteps']
        time_hours = timesteps * 0.25

        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(6, 8), dpi=300, facecolor='white')

        # Set common properties
        for i, ax in enumerate(axes):
            ax.tick_params(labelsize=7)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.1, linestyle='--')
            if i < 3:
                ax.set_xticklabels([])

        # 1. Zone Temperature and Setpoints
        ax1 = axes[0]
        ax1.plot(time_hours, data_dict['zone_temperature'],
                 color='#2C3E50', linewidth=2, label='Zone Temperature')
        ax1.plot(time_hours, data_dict['cooling_setpoint'], '--',
                 color='#E74C3C', linewidth=1.5, label='Cooling Setpoint', alpha=0.7)
        ax1.plot(time_hours, data_dict['heating_setpoint'], '--',
                 color='#3498DB', linewidth=1.5, label='Heating Setpoint', alpha=0.7)
        ax1.set_ylabel('Temperature (°C)', fontsize=8)
        ax1.set_title('Zone Temperature Control', fontsize=9, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True, shadow=True)

        # 2. HVAC Thermal Load and Power
        ax2 = axes[1]
        ax2.plot(time_hours, data_dict['hvac_thermal_load'] / 1000,
                 color='#E74C3C', linewidth=2, label='Thermal Load')
        ax2.plot(time_hours, data_dict['hvac_power'] / 1000,
                 color='#3498DB', linewidth=2, label='Power', linestyle='-.')
        ax2.set_ylabel('Power (kW)', fontsize=8)
        ax2.set_title('HVAC Performance', fontsize=9, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True, shadow=True)

        # 3. Supply Air Temperature
        ax3 = axes[2]
        ax3.plot(time_hours, data_dict['supply_air_temp_setpoint'], '--',
                 color='#7F8C8D', linewidth=1.5, label='Setpoint', alpha=0.7)
        ax3.plot(time_hours, data_dict['supply_air_temp_actual'],
                 color='#16A085', linewidth=2, label='Actual')
        ax3.set_ylabel('Supply Air Temp (°C)', fontsize=8)
        ax3.set_title('Supply Air Temperature', fontsize=9, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True, shadow=True)

        # 4. Supply Air Flow
        ax4 = axes[3]
        ax4.plot(time_hours, data_dict['supply_air_flow_setpoint'] * 1000, '--',
                 color='#7F8C8D', linewidth=1.5, label='Setpoint', alpha=0.7)
        ax4.plot(time_hours, data_dict['supply_air_flow_actual'] * 1000,
                 color='#8E44AD', linewidth=2, label='Actual')
        ax4.set_ylabel('Supply Air Flow (L/s)', fontsize=8)
        ax4.set_xlabel('Time (hours)', fontsize=8)
        ax4.set_title('Supply Air Flow Rate', fontsize=9, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True, shadow=True)

        plt.suptitle(f'Thermal System Performance - {simulation_name if simulation_name else "Current"}',
                     fontsize=11, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.99])

        if save_figure:
            fig_path = self.base_path / f"{simulation_name}_thermal_plots.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Thermal plot saved to: {fig_path}")

        if show_figure:
            plt.show()

        return fig

    def plot_electrical_results(self,
                                simulation_name: str = None,
                                data_dict: Dict = None,
                                save_figure: bool = True,
                                show_figure: bool = True) -> plt.Figure:
        """
        Create publication-quality electrical system plots with proper colors
        """
        if data_dict is None:
            data_dict, metadata = self.load_simulation_data(simulation_name)

        # Create time axis
        timesteps = data_dict['timesteps']
        time_hours = timesteps * 0.25

        # Get peak periods
        peak_mask = data_dict['is_peak'].astype(bool)

        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(6, 8), dpi=300, facecolor='white')

        # Set common properties
        for i, ax in enumerate(axes):
            ax.tick_params(labelsize=7)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.1, linestyle='--')
            if i < 3:
                ax.set_xticklabels([])

        # Define colors for loads
        load_colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

        # 1. Disaggregated Building Loads
        ax1 = axes[0]
        ax1.stackplot(time_hours,
                      data_dict['hvac_power_kw'],
                      data_dict['lighting'],
                      data_dict['cooking'],
                      data_dict['pc'],
                      data_dict['tv'],
                      labels=['HVAC', 'Lighting', 'Cooking', 'PC', 'TV'],
                      colors=load_colors,
                      alpha=0.7)
        ax1.plot(time_hours, data_dict['total_building_load'],
                 color='#2C3E50', linewidth=2, label='Total Load')
        ax1.set_ylabel('Power (kW)', fontsize=8)
        ax1.set_title('Building Load Breakdown', fontsize=9, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=7, ncol=2, frameon=True, fancybox=True, shadow=True)

        # 2. Battery and EV SOC with peak shading
        ax2 = axes[1]
        if peak_mask.any():
            ax2.fill_between(time_hours, 0, 100, where=peak_mask,
                             alpha=0.15, color='#FF6B6B', label='Peak Period')
        ax2.plot(time_hours, data_dict['battery_soc'] * 100,
                 color='#2980B9', linewidth=2, label='Battery')
        ax2.plot(time_hours, data_dict['ev_tesla_soc'] * 100,
                 color='#27AE60', linewidth=2, label='EV Tesla')
        ax2.plot(time_hours, data_dict['ev_nissan_soc'] * 100,
                 color='#E67E22', linewidth=2, label='EV Nissan')
        ax2.set_ylabel('State of Charge (%)', fontsize=8)
        ax2.set_ylim([0, 100])
        ax2.set_title('Energy Storage Systems', fontsize=9, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True, shadow=True)

        # 3. PV Generation and Distribution
        ax3 = axes[2]
        # PV generation as shaded area
        ax3.fill_between(time_hours, 0, data_dict['pv_generation'],
                         alpha=0.3, color='#FDB813', label='PV Generation')

        # Stacked area for PV distribution
        pv_colors = ['#FF9800', '#FF5722', '#795548', '#607D8B']
        pv_components = ['pv2building', 'pv2battery', 'pv2ev', 'pv2grid']
        pv_labels = ['PV→Building', 'PV→Battery', 'PV→EV', 'PV→Grid']

        bottom = np.zeros(len(time_hours))
        for i, (comp, label, color) in enumerate(zip(pv_components, pv_labels, pv_colors)):
            ax3.fill_between(time_hours, bottom, bottom + data_dict[comp],
                             color=color, alpha=0.6, label=label)
            bottom += data_dict[comp]

        ax3.set_ylabel('Power (kW)', fontsize=8)
        ax3.set_title('PV Generation and Distribution', fontsize=9, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=7, ncol=2, frameon=True, fancybox=True, shadow=True)

        # 4. Grid Net Load with peak shading
        ax4 = axes[3]
        if peak_mask.any():
            y_max = max(np.max(data_dict['grid_import']), np.max(data_dict['pv2grid'])) * 1.1
            y_min = -np.max(data_dict['pv2grid']) * 1.1
            ax4.fill_between(time_hours, y_min, y_max, where=peak_mask,
                             alpha=0.15, color='#FF6B6B', label='Peak Period')

        # Net grid load
        net_grid = data_dict['grid_import'] - data_dict['pv2grid']
        ax4.plot(time_hours, net_grid, color='#2C3E50', linewidth=2, label='Net Grid Load')
        ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax4.fill_between(time_hours, 0, net_grid, where=(net_grid > 0),
                         color='#E74C3C', alpha=0.4, label='Import')
        ax4.fill_between(time_hours, 0, net_grid, where=(net_grid < 0),
                         color='#27AE60', alpha=0.4, label='Export')
        ax4.set_ylabel('Power (kW)', fontsize=8)
        ax4.set_xlabel('Time (hours)', fontsize=8)
        ax4.set_title('Grid Interaction', fontsize=9, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=7, frameon=True, fancybox=True, shadow=True)

        plt.suptitle(f'Electrical System Performance - {simulation_name if simulation_name else "Current"}',
                     fontsize=11, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.99])

        if save_figure:
            fig_path = self.base_path / f"{simulation_name}_electrical_plots.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Electrical plot saved to: {fig_path}")

        if show_figure:
            plt.show()

        return fig

    def load_simulation_data(self, simulation_name: str) -> Tuple[Dict, Dict]:
        """
        Load saved simulation data

        Parameters:
        -----------
        simulation_name : str
            Name of the simulation to load

        Returns:
        --------
        data_dict : dict
            Dictionary containing all simulation data
        metadata : dict
            Metadata about the simulation
        """
        load_path = self.base_path / 'operation' / f"{simulation_name}.pkl"

        if not load_path.exists():
            # Try without extension
            load_path = self.base_path / simulation_name
            if not load_path.exists():
                raise FileNotFoundError(f"Simulation data not found: {simulation_name}")

        with open(load_path, 'rb') as f:
            saved_data = pickle.load(f)

        return saved_data['data'], saved_data.get('metadata', {})

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


    def analyze_simulation(self,
                           simulation_name: str = None,
                           data_dict: Dict = None) -> Dict:
        """
        Function 3: Load saved data and perform comprehensive analysis

        Parameters:
        -----------
        simulation_name : str
            Name of simulation to analyze
        data_dict : dict
            Pre-loaded data dictionary (optional)

        Returns:
        --------
        analysis : dict
            Comprehensive analysis results
        """
        if data_dict is None:
            data_dict, metadata = self.load_simulation_data(simulation_name)

        analysis = {}

        # 1. COMFORT ANALYSIS
        comfort = {}
        zone_temp = data_dict['zone_temperature']
        cool_sp = data_dict['cooling_setpoint']
        heat_sp = data_dict['heating_setpoint']

        # Temperature violations
        comfort['temp_above_cooling_frq'] = np.sum(zone_temp > cool_sp)
        comfort['temp_below_heating_frq'] = np.sum(zone_temp < heat_sp)
        comfort['total_violations_frq'] = comfort['temp_above_cooling_frq'] + comfort['temp_below_heating_frq']
        comfort['violation_rate'] = comfort['total_violations_frq'] / len(zone_temp) * 100

        comfort['temp_above_cooling_c_h'] = np.sum(np.maximum(zone_temp - cool_sp, 0)) * 0.25
        comfort['temp_below_heating_c_h'] = np.sum(np.maximum(heat_sp - zone_temp, 0)) * 0.25
        comfort['total_violation_c_h'] = comfort['temp_above_cooling_c_h'] + comfort['temp_below_heating_c_h']

        # Comfort metrics
        comfort['avg_temperature'] = np.mean(zone_temp)
        comfort['min_temperature'] = np.min(zone_temp)
        comfort['max_temperature'] = np.max(zone_temp)
        comfort['temp_std'] = np.std(zone_temp)
        comfort['temp_range'] = np.max(zone_temp) - np.min(zone_temp)

        analysis['comfort'] = comfort

        # 2. COST ANALYSIS
        cost = {}
        grid_import = data_dict['grid_import']
        pv2grid = data_dict['pv2grid']
        electricity_price = data_dict.get('electricity_price', np.ones_like(grid_import) * 0.15)
        is_peak = data_dict['is_peak'].astype(bool)

        # Energy costs
        time_step_hours = 0.25
        cost['import_cost'] = np.sum(grid_import * electricity_price * time_step_hours) / 100

        # Export revenue (assuming feed-in tariff is 50% of import price)
        feed_in_tariff = electricity_price * 0.5
        cost['export_revenue'] = np.sum(pv2grid * feed_in_tariff * time_step_hours) / 100
        cost['net_cost'] = cost['import_cost'] - cost['export_revenue']

        analysis['cost'] = cost

        # 3. FLEXIBILITY ANALYSIS
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
        battery_soc = data_dict['battery_soc']
        flexibility['battery_soc_avg'] = np.mean(battery_soc) * 100
        flexibility['battery_soc_min'] = np.min(battery_soc) * 100
        flexibility['battery_soc_max'] = np.max(battery_soc) * 100
        battery_efc = _battery_efc_from_soc(battery_soc)
        flexibility['battery_cycles_efc'] = battery_efc

        # EV flexibility
        ev_tesla_soc = data_dict['ev_tesla_soc']
        ev_nissan_soc = data_dict['ev_nissan_soc']
        flexibility['ev_tesla_cycles_efc'] = _battery_efc_from_soc(ev_tesla_soc)
        flexibility['ev_nissan_cycles_efc'] = _battery_efc_from_soc(ev_nissan_soc)

        # Load shifting capability
        total_load = data_dict['total_building_load']
        flexibility['load_factor'] = np.mean(total_load) / np.max(total_load) * 100
        flexibility['peak_to_average_ratio'] = np.max(total_load) / np.mean(total_load)

        # Peak vs off-peak energy consumption
        flexibility['peak_consumption'] = np.sum(grid_import[is_peak]) * time_step_hours
        flexibility['off_peak_consumption'] = np.sum(grid_import[~is_peak]) * time_step_hours
        flexibility['total_consumption'] = flexibility['peak_consumption'] + flexibility['off_peak_consumption']

        flexibility['peak_max'] = np.max(grid_import[is_peak])
        flexibility['off_peak_max'] = np.max(grid_import[~is_peak])

        analysis['flexibility'] = flexibility

        # 4. SYSTEM OPERATION ANALYSIS
        operation = {}

        # HVAC operation
        hvac_power = data_dict['hvac_power_kw']
        operation['hvac_runtime_hours'] = np.sum(hvac_power > 0.01) * time_step_hours
        operation['hvac_energy_kwh'] = np.sum(hvac_power) * time_step_hours
        operation['hvac_avg_power'] = np.mean(hvac_power[hvac_power > 0.01]) if any(hvac_power > 0.01) else 0
        operation['hvac_avg_cop'] = (np.sum(np.abs(data_dict['hvac_thermal_load'])) /
                                 np.sum(data_dict['hvac_power'])) if np.sum(data_dict['hvac_power']) > 0 else 0

        # PV operation
        pv_gen = data_dict['pv_generation']
        operation['pv_total_generation'] = np.sum(pv_gen) * time_step_hours
        operation['pv_peak_output'] = np.max(pv_gen)
        operation['pv_capacity_factor'] = (np.mean(pv_gen) / np.max(pv_gen) * 100
                                           if np.max(pv_gen) > 0 else 0)

        # Self-consumption
        pv2building = data_dict['pv2building']
        pv2battery = data_dict['pv2battery']
        pv2ev = data_dict['pv2ev']
        operation['self_consumption'] = np.sum(pv2building + pv2battery + pv2ev) * time_step_hours
        operation['self_consumption_rate'] = (operation['self_consumption'] /
                                              operation['pv_total_generation'] * 100
                                              if operation['pv_total_generation'] > 0 else 0)

        # Grid interaction
        operation['grid_import_total'] = np.sum(grid_import) * time_step_hours
        operation['grid_export_total'] = np.sum(pv2grid) * time_step_hours
        operation['net_grid_energy'] = operation['grid_import_total'] - operation['grid_export_total']

        analysis['operation'] = operation

        # 5. EFFICIENCY METRICS
        efficiency = {}

        building_energy = np.sum(total_load) * time_step_hours  # kWh
        grid2building_energy = np.sum(data_dict['grid2building']) * time_step_hours  # kWh
        pv2building_energy = np.sum(data_dict['pv2building']) * time_step_hours  # kWh

        efficiency['total_consumption'] = building_energy  # building load energy (kWh)
        efficiency['grid_dependency'] = (
            (grid2building_energy / building_energy) * 100 if building_energy > 0 else 0.0
        )
        efficiency['renewable_fraction'] = (
            (pv2building_energy / building_energy) * 100 if building_energy > 0 else 0.0
        )

        # Load diversity
        loads = {
            'HVAC': np.sum(hvac_power) * time_step_hours,
            'Lighting': np.sum(data_dict['lighting']) * time_step_hours,
            'Cooking': np.sum(data_dict['cooking']) * time_step_hours,
            'PC': np.sum(data_dict['pc']) * time_step_hours,
            'TV': np.sum(data_dict['tv']) * time_step_hours
        }
        efficiency['load_breakdown'] = loads

        analysis['efficiency'] = efficiency

        # 6. SUMMARY STATISTICS
        summary = {
            'simulation_duration_hours': len(data_dict['timesteps']) * time_step_hours,
            'total_cost': cost['net_cost'],
            'comfort_violation_rate': comfort['violation_rate'],
            'renewable_usage': efficiency['renewable_fraction'],
            'peak_demand': np.max(total_load),
            'average_demand': np.mean(total_load),
            'grid_dependency': efficiency['grid_dependency'],
            'self_consumption_rate': operation['self_consumption_rate'],
            'battery_cycles': flexibility['battery_cycles_efc'],
            'peak_consumption_ratio': (flexibility['peak_consumption'] /
                                       flexibility['total_consumption'] * 100
                                       if flexibility['total_consumption'] > 0 else 0)
        }
        analysis['summary'] = summary

        # Print analysis results
        self._print_analysis(analysis)

        # Save analysis to file
        if simulation_name:
            analysis_path = self.base_path / f"{simulation_name}_analysis.json"
            with open(analysis_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_analysis = self._convert_to_json_serializable(analysis)
                json.dump(json_analysis, f, indent=2)
            print(f"\nAnalysis saved to: {analysis_path}")


        return analysis

    def compare_simulations(self,
                            sim_name1: str,
                            sim_name2: str,
                            comparison_name: str = None) -> Dict:
        """
        Function 4: Load and compare two simulations

        Parameters:
        -----------
        sim_name1 : str
            Name of first simulation
        sim_name2 : str
            Name of second simulation
        comparison_name : str
            Name for the comparison output

        Returns:
        --------
        comparison : dict
            Comparison results
        """
        # Load both simulations
        data1, meta1 = self.load_simulation_data(sim_name1)
        data2, meta2 = self.load_simulation_data(sim_name2)

        # Analyze both simulations (suppress printing)
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        analysis1 = self.analyze_simulation(data_dict=data1)
        analysis2 = self.analyze_simulation(data_dict=data2)
        sys.stdout = old_stdout

        comparison = {
            'simulation_1': sim_name1,
            'simulation_2': sim_name2,
            'timestamp': datetime.now().isoformat()
        }

        # Compare key metrics
        metrics_comparison = {}

        # Comfort comparison
        metrics_comparison['comfort'] = {
            'violation_rate_diff': analysis2['comfort']['violation_rate'] -
                                   analysis1['comfort']['violation_rate'],
            'total_violations_frq_diff': analysis2['comfort']['total_violations_frq'] -
                                         analysis1['comfort']['total_violations_frq'],
            'total_violation_c_h_diff': analysis2['comfort']['total_violation_c_h'] -
                                        analysis1['comfort']['total_violation_c_h'],
            'avg_temp_diff': analysis2['comfort']['avg_temperature'] -
                             analysis1['comfort']['avg_temperature'],
            'temp_range_diff': analysis2['comfort']['temp_range'] -
                               analysis1['comfort']['temp_range']
        }

        # Cost comparison
        metrics_comparison['cost'] = {
            'net_cost_diff': analysis2['cost']['net_cost'] - analysis1['cost']['net_cost'],
            'net_cost_change_pct': ((analysis2['cost']['net_cost'] -
                                     analysis1['cost']['net_cost']) /
                                    abs(analysis1['cost']['net_cost']) * 100
                                    if analysis1['cost']['net_cost'] != 0 else 0),
            'import_cost_diff': analysis2['cost']['import_cost'] - analysis1['cost']['import_cost'],
            'export_revenue_diff': analysis2['cost']['export_revenue'] - analysis1['cost']['export_revenue']
        }

        # Flexibility comparison
        metrics_comparison['flexibility'] = {
            'battery_cycles_diff': analysis2['flexibility']['battery_cycles_efc'] -
                                   analysis1['flexibility']['battery_cycles_efc'],
            'load_factor_diff': analysis2['flexibility']['load_factor'] -
                                analysis1['flexibility']['load_factor'],
            'peak_consumption_diff': analysis2['flexibility']['peak_consumption'] -
                                     analysis1['flexibility']['peak_consumption'],
            'peak_max_diff': analysis2['flexibility']['peak_max'] -
                             analysis1['flexibility']['peak_max']
        }

        # Operation comparison
        metrics_comparison['operation'] = {
            'self_consumption_rate_diff': analysis2['operation']['self_consumption_rate'] -
                                          analysis1['operation']['self_consumption_rate'],
            'pv_total_generation_diff': analysis2['operation']['pv_total_generation'] -
                                        analysis1['operation']['pv_total_generation'],
            'hvac_energy_diff': analysis2['operation']['hvac_energy_kwh'] -
                                analysis1['operation']['hvac_energy_kwh'],
            'hvac_avg_cop_diff': analysis2['operation']['hvac_avg_cop'] -
                                 analysis1['operation']['hvac_avg_cop']
        }

        # Efficiency comparison
        metrics_comparison['efficiency'] = {
            'grid_dependency_diff': analysis2['efficiency']['grid_dependency'] -
                                    analysis1['efficiency']['grid_dependency'],
            'renewable_fraction_diff': analysis2['efficiency']['renewable_fraction'] -
                                       analysis1['efficiency']['renewable_fraction'],
            'total_consumption_diff': analysis2['efficiency']['total_consumption'] -
                                      analysis1['efficiency']['total_consumption']
        }

        comparison['metrics_comparison'] = metrics_comparison

        # Create comparison plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Temperature profiles
        ax = axes[0, 0]
        time_hours1 = data1['timesteps'] * 0.25
        time_hours2 = data2['timesteps'] * 0.25
        ax.plot(time_hours1, data1['zone_temperature'], 'b-', label=f'{sim_name1}', alpha=0.7)
        ax.plot(time_hours2, data2['zone_temperature'], 'r-', label=f'{sim_name2}', alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Zone Temperature Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Total load profiles
        ax = axes[0, 1]
        ax.plot(time_hours1, data1['total_building_load'], 'b-', label=f'{sim_name1}', alpha=0.7)
        ax.plot(time_hours2, data2['total_building_load'], 'r-', label=f'{sim_name2}', alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Total Building Load Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Battery SOC
        ax = axes[1, 0]
        ax.plot(time_hours1, data1['battery_soc'] * 100, 'b-', label=f'{sim_name1}', alpha=0.7)
        ax.plot(time_hours2, data2['battery_soc'] * 100, 'r-', label=f'{sim_name2}', alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('SOC (%)')
        ax.set_title('Battery SOC Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Grid import
        ax = axes[1, 1]
        ax.plot(time_hours1, data1['grid_import'], 'b-', label=f'{sim_name1}', alpha=0.7)
        ax.plot(time_hours2, data2['grid_import'], 'r-', label=f'{sim_name2}', alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power (kW)')
        ax.set_title('Grid Import Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Key metrics bar chart
        ax = axes[2, 0]
        metrics = ['Violation\nRate', 'Net\nCost', 'Battery\nCycles',
                   'Renewable\nFraction']
        values1 = [analysis1['comfort']['violation_rate'],
                   analysis1['cost']['net_cost'],
                   analysis1['flexibility']['battery_cycles_efc'],
                   analysis1['efficiency']['renewable_fraction']]
        values2 = [analysis2['comfort']['violation_rate'],
                   analysis2['cost']['net_cost'],
                   analysis2['flexibility']['battery_cycles_efc'],
                   analysis2['efficiency']['renewable_fraction']]

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, values1, width, label=sim_name1, alpha=0.7)
        ax.bar(x + width / 2, values2, width, label=sim_name2, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title('Key Metrics Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Summary table
        ax = axes[2, 1]
        ax.axis('tight')
        ax.axis('off')

        table_data = [
            ['Metric', sim_name1, sim_name2, 'Difference'],
            ['Violation Rate (%)', f"{analysis1['comfort']['violation_rate']:.1f}",
             f"{analysis2['comfort']['violation_rate']:.1f}",
             f"{metrics_comparison['comfort']['violation_rate_diff']:+.1f}"],
            ['Net Cost ($)', f"{analysis1['cost']['net_cost']:.2f}",
             f"{analysis2['cost']['net_cost']:.2f}",
             f"{metrics_comparison['cost']['net_cost_diff']:+.2f}"],
            ['Peak Demand (kW)', f"{analysis1['summary']['peak_demand']:.2f}",
             f"{analysis2['summary']['peak_demand']:.2f}",
             f"{analysis2['summary']['peak_demand'] - analysis1['summary']['peak_demand']:+.2f}"],
            ['Self-Consumption (%)', f"{analysis1['operation']['self_consumption_rate']:.1f}",
             f"{analysis2['operation']['self_consumption_rate']:.1f}",
             f"{metrics_comparison['operation']['self_consumption_rate_diff']:+.1f}"],
            ['Grid Dependency (%)', f"{analysis1['efficiency']['grid_dependency']:.1f}",
             f"{analysis2['efficiency']['grid_dependency']:.1f}",
             f"{metrics_comparison['efficiency']['grid_dependency_diff']:+.1f}"]
        ]

        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color difference column based on improvement
        for i in range(1, len(table_data)):
            val = float(table_data[i][3])
            if i == 1:  # Violation Rate (lower is better)
                color = '#90EE90' if val < 0 else '#FFB6C1'
            elif i == 2:  # Cost (lower is better)
                color = '#90EE90' if val < 0 else '#FFB6C1'
            elif i == 5:  # Grid Dependency (lower is better)
                color = '#90EE90' if val < 0 else '#FFB6C1'
            else:  # Others (higher is better)
                color = '#90EE90' if val > 0 else '#FFB6C1'
            table[(i, 3)].set_facecolor(color)

        plt.suptitle(f'Simulation Comparison: {sim_name1} vs {sim_name2}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save comparison
        if comparison_name is None:
            comparison_name = f"{sim_name1}_vs_{sim_name2}"

        # Save figure
        fig_path = self.base_path / f"{comparison_name}_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {fig_path}")

        # Save comparison data
        comparison_path = self.base_path / f"{comparison_name}_comparison.json"
        with open(comparison_path, 'w') as f:
            json_comparison = self._convert_to_json_serializable(comparison)
            json.dump(json_comparison, f, indent=2)
        print(f"Comparison data saved to: {comparison_path}")

        plt.show()

        # Print comparison summary
        print("\n" + "=" * 60)
        print(f"COMPARISON SUMMARY: {sim_name1} vs {sim_name2}")
        print("=" * 60)

        print("\n📊 Key Performance Indicators:")
        print(f"  Comfort (Violation Rate): {metrics_comparison['comfort']['violation_rate_diff']:+.1f}% "
              f"{'✅' if metrics_comparison['comfort']['violation_rate_diff'] < 0 else '❌'}")
        print(f"  Cost: ${metrics_comparison['cost']['net_cost_diff']:+.2f} "
              f"({metrics_comparison['cost']['net_cost_change_pct']:+.1f}%) "
              f"{'✅' if metrics_comparison['cost']['net_cost_diff'] < 0 else '❌'}")
        print(f"  Battery Cycles: {metrics_comparison['flexibility']['battery_cycles_diff']:+.2f} "
              f"{'❌' if abs(metrics_comparison['flexibility']['battery_cycles_diff']) > 0.5 else '✅'}")
        print(f"  Renewable Usage: {metrics_comparison['efficiency']['renewable_fraction_diff']:+.1f}% "
              f"{'✅' if metrics_comparison['efficiency']['renewable_fraction_diff'] > 0 else '❌'}")
        print(f"  Grid Dependency: {metrics_comparison['efficiency']['grid_dependency_diff']:+.1f}% "
              f"{'✅' if metrics_comparison['efficiency']['grid_dependency_diff'] < 0 else '❌'}")

        # Determine winner
        improvements = sum([
            metrics_comparison['comfort']['violation_rate_diff'] < 0,  # Lower is better
            metrics_comparison['cost']['net_cost_diff'] < 0,  # Lower is better
            abs(metrics_comparison['flexibility']['battery_cycles_diff']) < 0.5,  # Minimal degradation
            metrics_comparison['efficiency']['renewable_fraction_diff'] > 0,  # Higher is better
            metrics_comparison['efficiency']['grid_dependency_diff'] < 0  # Lower is better
        ])

        if improvements >= 3:
            print(f"\n✅ {sim_name2} shows better overall performance ({improvements}/5 metrics improved)")
        elif improvements < 3:
            print(f"\n✅ {sim_name1} shows better overall performance ({5 - improvements}/5 metrics improved)")
        else:
            print(f"\n⚖️ Mixed results - each simulation has its strengths")

        comparison['detailed_analysis'] = {
            'analysis_1': analysis1,
            'analysis_2': analysis2
        }

        return comparison

    def _print_analysis(self, analysis: Dict):
        """Helper function to print analysis results in a formatted way"""
        print("\n" + "=" * 60)
        print("SIMULATION ANALYSIS RESULTS")
        print("=" * 60)

        print("\n🌡️ COMFORT ANALYSIS:")
        print(f"  Temperature Violations (frequency): {analysis['comfort']['total_violations_frq']} "
              f"({analysis['comfort']['violation_rate']:.1f}%)")
        print(f"  Temperature Violations (°C·h): {analysis['comfort']['total_violation_c_h']:.2f}")
        print(f"    - Above cooling: {analysis['comfort']['temp_above_cooling_c_h']:.2f} °C·h")
        print(f"    - Below heating: {analysis['comfort']['temp_below_heating_c_h']:.2f} °C·h")
        print(f"  Average Temperature: {analysis['comfort']['avg_temperature']:.1f}°C "
              f"(±{analysis['comfort']['temp_std']:.2f}°C)")
        print(f"  Temperature Range: {analysis['comfort']['min_temperature']:.1f}°C - "
              f"{analysis['comfort']['max_temperature']:.1f}°C")

        print("\n💰 COST ANALYSIS:")
        print(f"  Import Cost: ${analysis['cost']['import_cost']:.2f}")
        print(f"  Export Revenue: ${analysis['cost']['export_revenue']:.2f}")
        print(f"  Net Cost: ${analysis['cost']['net_cost']:.2f}")

        print("\n⚡ FLEXIBILITY ANALYSIS:")
        print(f"  Battery SOC (avg/min/max): {analysis['flexibility']['battery_soc_avg']:.1f}% / "
              f"{analysis['flexibility']['battery_soc_min']:.1f}% / "
              f"{analysis['flexibility']['battery_soc_max']:.1f}%")
        print(f"  Battery Cycles (EFC): {analysis['flexibility']['battery_cycles_efc']:.2f}")
        print(f"  EV Tesla Cycles (EFC): {analysis['flexibility']['ev_tesla_cycles_efc']:.2f}")
        print(f"  EV Nissan Cycles (EFC): {analysis['flexibility']['ev_nissan_cycles_efc']:.2f}")
        print(f"  Load Factor: {analysis['flexibility']['load_factor']:.1f}%")
        print(f"  Peak-to-Average Ratio: {analysis['flexibility']['peak_to_average_ratio']:.2f}")
        print(f"  Peak Consumption: {analysis['flexibility']['peak_consumption']:.1f} kWh "
              f"(max: {analysis['flexibility']['peak_max']:.2f} kW)")
        print(f"  Off-Peak Consumption: {analysis['flexibility']['off_peak_consumption']:.1f} kWh "
              f"(max: {analysis['flexibility']['off_peak_max']:.2f} kW)")

        print("\n🔧 SYSTEM OPERATION:")
        print(f"  HVAC Runtime: {analysis['operation']['hvac_runtime_hours']:.1f} hours")
        print(f"  HVAC Energy: {analysis['operation']['hvac_energy_kwh']:.1f} kWh")
        print(f"  HVAC Average Power: {analysis['operation']['hvac_avg_power']:.2f} kW")
        print(f"  HVAC Average COP: {analysis['operation']['hvac_avg_cop']:.2f}")
        print(f"  PV Generation: {analysis['operation']['pv_total_generation']:.1f} kWh")
        print(f"  PV Peak Output: {analysis['operation']['pv_peak_output']:.2f} kW")
        print(f"  PV Capacity Factor: {analysis['operation']['pv_capacity_factor']:.1f}%")
        print(f"  Self-Consumption: {analysis['operation']['self_consumption']:.1f} kWh "
              f"({analysis['operation']['self_consumption_rate']:.1f}%)")
        print(f"  Grid Import Total: {analysis['operation']['grid_import_total']:.1f} kWh")
        print(f"  Grid Export Total: {analysis['operation']['grid_export_total']:.1f} kWh")
        print(f"  Net Grid Energy: {analysis['operation']['net_grid_energy']:.1f} kWh")

        print("\n📈 EFFICIENCY METRICS:")
        print(f"  Total Consumption: {analysis['efficiency']['total_consumption']:.1f} kWh")
        print(f"  Grid Dependency: {analysis['efficiency']['grid_dependency']:.1f}%")
        print(f"  Renewable Fraction: {analysis['efficiency']['renewable_fraction']:.1f}%")

        print("\n📊 LOAD BREAKDOWN:")
        for load_type, value in analysis['efficiency']['load_breakdown'].items():
            print(f"  {load_type}: {value:.1f} kWh")


# Example usage
if __name__ == "__main__":
    def set_seed(seed):
        import torch
        import random
        import numpy as np
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    seed_value = 142857
    set_seed(seed_value)

    for level in ['no']: #0,1,2,3,
        # Initialize the simulation manager
        sim_manager = SimulationManager(base_path="./simulation_results")

        # Example of how to use with your environment

        # 1. Run and save simulation
        from bestopt.env.core.config_manager import ConfigurationManager
        from bestopt.env.core.environment import BESTOptEnvironment
        import os
        from pathlib import Path

        PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

        # Load configuration
        config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_1_Building", f"llm_level_{level}.json")
        cm = ConfigurationManager(config_path)
        env = BESTOptEnvironment(cm.config)

        # Run simulation and save
        data_dict, save_path = sim_manager.run_and_save_simulation(
            env=env,
            total_steps=96,
            simulation_name=f"llm_level_{level}",
            config_info={"Running": f"llm_level_{level}"}
        )

        # 2. Load and plot saved simulation
        thermal_fig = sim_manager.plot_thermal_results(f"llm_level_{level}")
        electrical_fig = sim_manager.plot_electrical_results(f"llm_level_{level}")

