import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams


def create_paper_figure(timesteps, data_dict, save_path='simulation_results.pdf'):
    """
    Create a static figure for paper with 4 subplots showing one day simulation results.

    Parameters:
    -----------
    timesteps : array-like
        Time steps (0 to 95 for 15-minute intervals over 24 hours)
    data_dict : dict
        Dictionary containing all the data arrays with keys:
        - zone_temperature, cooling_setpoint, heating_setpoint
        - hvac_thermal_load, hvac_power
        - battery_soc, ev_tesla_soc, ev_nissan_soc
        - lighting, cooking, pc, tv, hvac_power_kw
        - total_building_load, battery_power, ev_power
        - is_peak (boolean array for peak hours)
    save_path : str
        Path to save the PDF figure
    """

    # Set global font sizes
    rcParams['font.size'] = 7
    rcParams['axes.labelsize'] = 7
    rcParams['axes.titlesize'] = 7
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 7
    rcParams['figure.titlesize'] = 7

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(6, 8), dpi=300)
    fig.subplots_adjust(hspace=0.3, top=0.95, bottom=0.08, left=0.1, right=0.95)

    # Convert timesteps to hours for x-axis
    hours = timesteps * 0.25  # Assuming 15-minute intervals

    # Extract peak hours for shading
    peak_regions = []
    if 'is_peak' in data_dict:
        is_peak = data_dict['is_peak']
        # Find continuous peak regions
        peak_start = None
        for i in range(len(is_peak)):
            if is_peak[i] and peak_start is None:
                peak_start = hours[i]
            elif not is_peak[i] and peak_start is not None:
                peak_regions.append((peak_start, hours[i - 1] if i > 0 else hours[i]))
                peak_start = None
        if peak_start is not None:
            peak_regions.append((peak_start, hours[-1]))

    # ===== Subplot 1: Zone Temperature and Setpoints =====
    ax1 = axes[0]
    ax1.plot(hours, data_dict['zone_temperature'], 'b-', linewidth=1.2, label='Zone Temperature')
    ax1.plot(hours, data_dict['cooling_setpoint'], '--', color='gray', linewidth=1, label='Cooling Setpoint')
    ax1.plot(hours, data_dict['heating_setpoint'], '--', color='gray', linewidth=1, label='Heating Setpoint')

    ax1.set_ylabel('Temperature (°C)', fontsize=7)
    ax1.set_xlim([0, 24])
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.legend(loc='lower left', frameon=False, fancybox=False, framealpha=0.9)
    ax1.set_title('(a) Zone Temperature Control', fontsize=7, fontweight='bold')

    # ===== Subplot 2: HVAC Thermal Load and Power =====
    ax2 = axes[1]
    ax2.plot(hours, data_dict['hvac_thermal_load'] / 1000, 'b-', linewidth=1.2, label='Thermal Load')
    ax2.plot(hours, data_dict['hvac_power'] / 1000, 'r-', linewidth=1.2, label='HVAC Power')

    ax2.set_ylabel('Power (kW)', fontsize=7)
    ax2.set_xlim([0, 24])
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.legend(loc='lower left', frameon=False, fancybox=False, framealpha=0.9)
    ax2.set_title('(b) HVAC Performance', fontsize=7, fontweight='bold')

    # ===== Subplot 3: Battery and EV SOC with Peak Shading + PV Gen/Dispatch =====
    ax3 = axes[2]

    # Peak hour shading (draw on ax3 so it sits "behind" lines)
    for start, end in peak_regions:
        ax3.axvspan(start, end, alpha=0.2, color='red',
                    label='Peak Hours' if start == peak_regions[0][0] else '')

    # SOC lines (left axis)
    soc_bat = np.asarray(data_dict['battery_soc']) * 100.0
    soc_tesla = np.asarray(data_dict['ev_tesla_soc']) * 100.0
    soc_nissan = np.asarray(data_dict['ev_nissan_soc']) * 100.0

    l_bat, = ax3.plot(hours, soc_bat, '-', linewidth=1.2, color='#2ca02c', label='Battery SOC')
    l_tesla, = ax3.plot(hours, soc_tesla, '-', linewidth=1.2, color='#1f77b4', label='EV Tesla SOC')
    l_nissan, = ax3.plot(hours, soc_nissan, '-', linewidth=1.2, color='#17becf', label='EV Nissan SOC')

    ax3.set_ylabel('State of Charge (%)', fontsize=7)
    ax3.set_xlim([0, 24])
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    ax3.set_title('(c) Energy Storage State of Charge & PV Dispatch', fontsize=7, fontweight='bold')

    # Secondary y-axis for PV power (kW)
    ax3b = ax3.twinx()

    def _to_kw(arr):
        arr = np.asarray(arr)
        # Heuristic: if the magnitude ever exceeds ~200, assume W and convert to kW
        # (tweak threshold if your data is smaller/bigger)
        return arr / 1000.0 if np.nanmax(np.abs(arr)) > 200 else arr
    pv_gen = _to_kw(data_dict.get('pv_generation', np.zeros_like(hours)))
    pv2bld = _to_kw(data_dict.get('pv2building', np.zeros_like(hours)))
    pv2bat = _to_kw(data_dict.get('pv2battery', np.zeros_like(hours)))
    pv2ev = _to_kw(data_dict.get('pv2ev', np.zeros_like(hours)))

    # Ensure non-negative / finite
    pv_gen = np.nan_to_num(np.maximum(pv_gen, 0.0))
    pv2bld = np.nan_to_num(np.maximum(pv2bld, 0.0))
    pv2bat = np.nan_to_num(np.maximum(pv2bat, 0.0))
    pv2ev = np.nan_to_num(np.maximum(pv2ev, 0.0))

    # Stacked dispatch areas on the right axis
    # Order: building -> battery -> ev (adjust if you prefer a different stack order)
    base0 = np.zeros_like(hours)
    a_bld = ax3b.fill_between(hours, base0, pv2bld, alpha=0.35, label='PV→Building')
    base1 = pv2bld
    a_bat = ax3b.fill_between(hours, base1, base1 + pv2bat, alpha=0.35, label='PV→Battery')
    base2 = base1 + pv2bat
    a_ev = ax3b.fill_between(hours, base2, base2 + pv2ev, alpha=0.35, label='PV→EV')

    # PV generation line (compare with stacked areas)
    l_pv, = ax3b.plot(hours, pv_gen, '-', linewidth=1.2, color='k', label='PV Generation')

    # Right axis formatting
    ax3b.set_ylabel('PV Power (kW)', fontsize=7)
    y_max = max(1e-6, float(np.nanmax([pv_gen.max(), (pv2bld + pv2bat + pv2ev).max()])))
    ax3b.set_ylim([0, y_max * 1.15])

    # Make SOC lines draw above the filled areas
    ax3.set_zorder(ax3b.get_zorder() + 1)
    ax3.patch.set_visible(False)

    # One combined legend (left axis location)
    h1, lab1 = ax3.get_legend_handles_labels()
    h2, lab2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, lab1 + lab2, loc='lower left', frameon=False, fancybox=False, framealpha=0.9, ncol=2)

    # ===== Subplot 4: Disaggregated Load with Stacked Area =====
    ax4 = axes[3]

    # Prepare load components for stacked area plot
    lighting = data_dict.get('lighting', np.zeros_like(hours))
    cooking = data_dict.get('cooking', np.zeros_like(hours))
    pc = data_dict.get('pc', np.zeros_like(hours))
    tv = data_dict.get('tv', np.zeros_like(hours))
    hvac = data_dict.get('hvac_power_kw', np.zeros_like(hours))

    # Create stacked area plot for disaggregated loads
    ax4.fill_between(hours, 0, lighting,
                     alpha=0.7, color='#FFD700', label='Lighting')
    ax4.fill_between(hours, lighting, lighting + cooking,
                     alpha=0.7, color='#FF6347', label='Cooking')
    ax4.fill_between(hours, lighting + cooking, lighting + cooking + pc,
                     alpha=0.7, color='#4169E1', label='PC')
    ax4.fill_between(hours, lighting + cooking + pc, lighting + cooking + pc + tv,
                     alpha=0.7, color='#32CD32', label='TV')
    ax4.fill_between(hours, lighting + cooking + pc + tv,
                     lighting + cooking + pc + tv + hvac,
                     alpha=0.7, color='#9370DB', label='HVAC')

    # Add lines for total loads
    total_building = data_dict.get('total_building_load',
                                   lighting + cooking + pc + tv + hvac)
    ax4.plot(hours, total_building, 'k-', linewidth=1.5, label='Total Building Load')

    # Add battery and EV power if available
    # if 'grid2battery' in data_dict:
    #     total_with_battery = total_building + data_dict['grid2battery']
    #     ax4.plot(hours, total_with_battery, 'r--', linewidth=1.2, label='Total + Battery')

    # if 'grid2ev' in data_dict:
    #     total_with_ev = total_building + data_dict.get('grid2battery', 0) + data_dict['grid2ev']
    #     ax4.plot(hours, total_with_ev, 'b--', linewidth=1.2, label='Grid Net Import')

    ax4.set_xlabel('Time (hours)', fontsize=7)
    ax4.set_ylabel('Power (kW)', fontsize=7)
    ax4.set_xlim([0, 24])
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    ax4.legend(loc='upper left', ncol=4, frameon=False, fancybox=False,
               framealpha=0.9, columnspacing=1)
    ax4.set_title('(d) Disaggregated Load Profile', fontsize=7, fontweight='bold')

    # Set x-axis ticks for all subplots
    for ax in axes:
        ax.set_xticks(np.arange(0, 25, 3))
        ax.tick_params(axis='both', which='major', labelsize=7)

    # Only show x-axis label on bottom plot
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    # Save figure
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def run_simulation_and_plot(env, total_steps=96):  # 96 steps = 24 hours with 15-min intervals
    """
    Run simulation for one day and create the paper figure.

    Parameters:
    -----------
    env : BESTOptEnvironment
        The environment object
    total_steps : int
        Number of simulation steps (default 96 for one day with 15-min intervals)
    """

    # Initialize data storage
    data_dict = {
        'zone_temperature': [],
        'cooling_setpoint': [],
        'heating_setpoint': [],
        'hvac_thermal_load': [],
        'hvac_power': [],
        'battery_soc': [],
        'ev_tesla_soc': [],
        'ev_nissan_soc': [],
        'lighting': [],
        'cooking': [],
        'pc': [],
        'tv': [],
        'hvac_power_kw': [],
        'total_building_load': [],
        'grid2battery': [],  # Power flow from grid to battery
        'grid2ev': [],  # Power flow from grid to EV
        'grid2building': [],
        'grid_import': [],
        'is_peak': [],
        'pv_generation': [],
        'pv2building': [],
        'pv2battery': [],
        'pv2ev': [],
        'pv2grid': [],
        'electricity_price': [],
    }

    timesteps = []

    # Run simulation
    for timestep in range(min(total_steps, env.total_step)):
        # Step the environment
        observations, done, info = env.step()

        # Get building and system IDs
        cluster_id = 'residential_cluster_1'
        building_id = 'SFH_1'
        hvac_system_id = env.building_system_map[building_id].get('thermal')
        der_system_id = env.building_system_map[building_id].get('electrical')

        # Extract all required data
        zone_temperature = env.cluster_states[cluster_id].thermal.systems['SFH_1_building'].components[
            'zone0'].temperature
        cooling_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].cooling_setpoint_c
        heating_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].heating_setpoint_c

        hvac_system = env.system_modules[hvac_system_id]
        der_system = env.system_modules[der_system_id]

        HVAC_power = hvac_system.FCU_power_total_W
        HVAC_thermal_load = hvac_system.Q_zone_actual_W

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

        # Get peak signal
        is_peak = env.disturbance.prices.peaksignal
        electricity_price = env.disturbance.prices.electricity_price  # ¢/kWh

        # Get grid to battery and EV power flows
        grid2ev_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2ev
        grid2battery_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2battery
        grid2ev = sum(grid2ev_dict.values())
        grid2battery = sum(grid2battery_dict.values())
        grid2building = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2building
        der_system = env.system_modules[der_system_id]
        pv_generation = der_system.pv_states['pv_1'].generation_w / 1000
        grid_import = grid2ev+grid2battery+grid2building

        # Get PV generation
        pv2ev_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2ev
        pv2battery_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2battery
        pv2ev = sum(pv2ev_dict.values())
        pv2battery = sum(pv2battery_dict.values())
        pv2building = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2building
        pv2grid = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].pv2grid

        # Store data
        timesteps.append(timestep)
        data_dict['zone_temperature'].append(zone_temperature)
        data_dict['cooling_setpoint'].append(cooling_setpoint)
        data_dict['heating_setpoint'].append(heating_setpoint)
        data_dict['hvac_thermal_load'].append(HVAC_thermal_load)
        data_dict['hvac_power'].append(HVAC_power)
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


        if timestep % 10 == 0:
            print(f"Step {timestep}/{total_steps}: Temp={zone_temperature:.1f}°C")

        if done:
            break

    # Convert lists to numpy arrays
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])

    # Create the figure
    timesteps = np.array(timesteps)
    fig = create_paper_figure(timesteps, data_dict, save_path='one_day_simulation.pdf')

    print(f"\nSimulation completed: {len(timesteps)} steps")
    print("Figure saved as 'one_day_simulation.pdf'")

    return fig, data_dict

from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
import os
from pathlib import Path
PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))
# Load configuration
config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_1_Building", "config_setup.json")
cm = ConfigurationManager(config_path)
env = BESTOptEnvironment(cm.config)
fig, data_dict = run_simulation_and_plot(env, total_steps=96)