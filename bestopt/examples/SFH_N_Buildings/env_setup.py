"""
Multi-Building Simulation Runner with Individual DER Systems
"""
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Get project root path
PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))


def run_multi_building_simulation(config_path: str = None, max_steps: int = None):
    """Run simulation for multiple buildings with individual DER systems."""

    # Load configuration
    cm = ConfigurationManager(config_path)
    env = BESTOptEnvironment(cm.config)

    # Use max_steps or total_step from environment
    total_steps = max_steps if max_steps else env.total_step

    # Initialize data storage for all buildings
    building_data = {}
    building_ids = ['SFH_1', 'SFH_2', 'SFH_3', 'SFH_4', 'SFH_5']

    for building_id in building_ids:
        building_data[building_id] = {
            'timestamps': [],
            'zone_temperature': [],
            'cooling_setpoint': [],
            'heating_setpoint': [],
            'hvac_power_kw': [],
            'hvac_thermal_load_kw': [],
            'pv_generation_kw': [],
            'battery_soc': [],
            'battery_power_kw': [],
            'total_load_kw': [],
            'grid_power_kw': [],
            'ev_socs': {},
            'building_load_kw': [],
            'cooking_kw': [],
            'pc_kw': [],
            'tv_kw': [],
            'lighting_kw': [],
            'electricity_price': [],
            'is_peak': [],
        }
    # Simulation loop
    print(f"\nStarting multi-building simulation for {total_steps} steps...")
    print("=" * 60)

    for timestep in range(total_steps):
        # Step the environment
        observations, done, info = env.step()

        # Calculate timestamp
        current_time = datetime.strptime(env.simulation_start_time, "%Y-%m-%d %H:%M:%S") + \
                       timedelta(seconds=timestep * env.res)

        # Collect data for each building
        cluster_id = 'residential_cluster_multi'

        for i, building_id in enumerate(building_ids):
            hvac_system_id = f'hvac_system_{building_id}'
            der_system_id = f'der_system_{building_id}'

            # Get building system state
            building_system_id = f'{building_id}_building'
            building_state = env.cluster_states[cluster_id].thermal.systems.get(building_system_id)

            if not building_state:
                print(f"Warning: Building state not found for {building_id}")
                continue

            # Zone temperature
            zone_temp = building_state.components['zone0'].temperature if 'zone0' in building_state.components else 20.0

            # Get setpoints from controller action
            hvac_action = env.cluster_actions[cluster_id].thermal.system_actions.get(hvac_system_id)
            if hvac_action:
                cooling_setpoint = hvac_action.cooling_setpoint_c
                heating_setpoint = hvac_action.heating_setpoint_c
            else:
                print(f"Warning: Setpoint not found for {building_id}")

            # HVAC metrics
            hvac_system = env.system_modules.get(hvac_system_id)
            if hvac_system and hasattr(hvac_system, 'FCU_power_total_W'):
                hvac_power = hvac_system.FCU_power_total_W
                hvac_thermal = hvac_system.Q_zone_actual_W if hasattr(hvac_system, 'Q_zone_actual_W') else 0
            else:
                hvac_power = 0
                hvac_thermal = 0

            # DER metrics
            der_system = env.system_modules.get(der_system_id)

            # PV generation
            pv_generation = 0
            if der_system and hasattr(der_system, 'pv_states'):
                for pv_id, pv_state in der_system.pv_states.items():
                    if hasattr(pv_state, 'generation_w'):
                        pv_generation += pv_state.generation_w

            # Battery SOC and power
            battery_soc = 0
            battery_power = 0
            if der_system and hasattr(der_system, 'battery_states'):
                for bat_id, bat_state in der_system.battery_states.items():
                    if hasattr(bat_state, 'soc'):
                        battery_soc = bat_state.soc  # Take first battery
                    if hasattr(bat_state, 'power_w'):
                        battery_power += bat_state.power_w

            # EV SOCs
            ev_socs = {}
            if der_system and hasattr(der_system, 'ev_states'):
                for ev_id, ev_state in der_system.ev_states.items():
                    if hasattr(ev_state, 'soc'):
                        ev_socs[ev_id] = ev_state.soc

            # Building electrical load details
            electrical_zone = env.cluster_states[cluster_id].electrical.systems.get(building_system_id)
            if electrical_zone:
                electrical_comp = electrical_zone.components.get('electrical')
                building_load = electrical_comp.building_power_w if electrical_comp else 0
            else:
                building_load = 0

            # Get detailed loads from electrical zone module
            zone_key = f"{building_id}.zone0"
            if zone_key in env.electrical_zone_modules:
                elec_module = env.electrical_zone_modules[zone_key]
                cooking = elec_module.cooking_power if hasattr(elec_module, 'cooking_power') else 0
                pc = elec_module.pc_power if hasattr(elec_module, 'pc_power') else 0
                tv = elec_module.tv_power if hasattr(elec_module, 'tv_power') else 0
                lighting = elec_module.lighting_power if hasattr(elec_module, 'lighting_power') else 0
            else:
                cooking = pc = tv = lighting = 0

            total_load = (hvac_power + building_load) / 1000  # Convert to kW

            # Grid power calculation
            der_action = env.cluster_actions[cluster_id].electrical.system_actions.get(der_system_id)
            grid2building = der_action.grid2building if hasattr(der_action, 'grid2building') else 0
            grid2ev = sum(der_action.grid2ev.values()) if hasattr(der_action, 'grid2ev') else 0
            grid2battery = sum(der_action.grid2battery.values()) if hasattr(der_action, 'grid2battery') else 0
            grid_power = (grid2building + grid2ev + grid2battery) / 1000

            # Price signal
            electricity_price = env.disturbance.prices.electricity_price # ¢/kWh
            is_peak = env.disturbance.prices.peaksignal # Bool

            # Store data
            data = building_data[building_id]
            data['timestamps'].append(current_time)
            data['zone_temperature'].append(zone_temp)
            data['cooling_setpoint'].append(cooling_setpoint)
            data['heating_setpoint'].append(heating_setpoint)
            data['hvac_power_kw'].append(hvac_power / 1000)
            data['hvac_thermal_load_kw'].append(hvac_thermal / 1000)
            data['pv_generation_kw'].append(pv_generation / 1000)
            data['battery_soc'].append(battery_soc*100)
            data['battery_power_kw'].append(battery_power / 1000)
            data['total_load_kw'].append(total_load)
            data['grid_power_kw'].append(grid_power)
            data['building_load_kw'].append(building_load / 1000)
            data['cooking_kw'].append(cooking / 1000)
            data['pc_kw'].append(pc / 1000)
            data['tv_kw'].append(tv / 1000)
            data['lighting_kw'].append(lighting / 1000)
            data['electricity_price'].append(electricity_price)
            data['is_peak'].append(is_peak)

            # Store EV SOCs
            for ev_id, soc in ev_socs.items():
                if ev_id not in data['ev_socs']:
                    data['ev_socs'][ev_id] = []
                data['ev_socs'][ev_id].append(soc*100)

        # Progress update
        if timestep % 4 == 0:
            avg_temp = np.mean([building_data[bid]['zone_temperature'][-1] for bid in building_ids])
            total_pv = sum([building_data[bid]['pv_generation_kw'][-1] for bid in building_ids])
            total_load_all = sum([building_data[bid]['total_load_kw'][-1] for bid in building_ids])
            print(f"Step {timestep:4d}/{total_steps} | Avg Temp: {avg_temp:5.1f}°C | "
                  f"Total PV: {total_pv:6.2f}kW | Total Load: {total_load_all:6.2f}kW")

        if done:
            break

    print("=" * 60)
    print(f"✓ Simulation completed: {timestep + 1} steps")
    return building_data, env


#@todo Claude generated plot function, revise later
def plot_multi_building_results(building_data: dict):
    """Create comprehensive plots for multi-building simulation results."""

    # Create figure with subplots
    fig, axes = plt.subplots(4, 5, figsize=(24, 16), dpi=300)
    fig.suptitle('Multi-Building Simulation Results', fontsize=16, fontweight='bold')

    buildings = list(building_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(buildings)))

    for idx, (building_id, data) in enumerate(building_data.items()):
        col = idx
        time_hours = np.arange(len(data['timestamps'])) * 0.25  # Assuming 15-min intervals

        # Row 1: Temperature Control
        ax = axes[0, col]
        ax.plot(time_hours, data['zone_temperature'], 'b-', label='Zone Temp', linewidth=2)
        ax.plot(time_hours, data['cooling_setpoint'], 'r--', label='Cool SP', alpha=0.7)
        ax.plot(time_hours, data['heating_setpoint'], 'g--', label='Heat SP', alpha=0.7)
        ax.fill_between(time_hours, data['heating_setpoint'], data['cooling_setpoint'],
                        alpha=0.2, color='gray', label='Comfort Zone')
        ax.set_title(f'{building_id}\nTemperature Control')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim([18, 27])

        # Row 2: Power Generation & Storage
        ax = axes[1, col]
        ax.plot(time_hours, data['pv_generation_kw'], 'gold', label='PV Gen', linewidth=2)
        ax.fill_between(time_hours, 0, data['pv_generation_kw'], alpha=0.3, color='gold')
        ax2 = ax.twinx()
        ax2.plot(time_hours, data['battery_soc'], 'green', label='Battery SOC', linewidth=2)
        ax.set_ylabel('PV Generation (kW)', color='gold')
        ax2.set_ylabel('Battery SOC (%)', color='green')
        ax.set_title('Generation & Storage')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(10, max(data['pv_generation_kw']) * 1.1)])
        ax2.set_ylim([0, 100])

        # Row 3: Load Breakdown
        ax = axes[2, col]
        ax.plot(time_hours, data['total_load_kw'], 'k-', label='Total', linewidth=2)
        ax.plot(time_hours, data['hvac_power_kw'], 'b-', label='HVAC', alpha=0.7)
        ax.plot(time_hours, data['building_load_kw'], 'g-', label='Building', alpha=0.7)
        ax.fill_between(time_hours, 0, data['hvac_power_kw'], alpha=0.3, color='blue')
        ax.fill_between(time_hours, data['hvac_power_kw'], data['total_load_kw'],
                        alpha=0.3, color='green')
        ax.set_ylabel('Load (kW)')
        ax.set_title('Load Breakdown')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        # Row 4: Grid Exchange
        ax = axes[3, col]
        positive_grid = np.maximum(data['grid_power_kw'], 0)
        negative_grid = np.minimum(data['grid_power_kw'], 0)
        ax.fill_between(time_hours, 0, positive_grid, color='red', alpha=0.5, label='Grid Import')
        ax.fill_between(time_hours, 0, negative_grid, color='green', alpha=0.5, label='Grid Export')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Grid Power (kW)')
        ax.set_title('Grid Exchange')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    return fig

def create_summary_dashboard(building_data: dict):
    """Create a summary dashboard comparing all buildings."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multi-Building Comparison Dashboard', fontsize=16, fontweight='bold')

    buildings = list(building_data.keys())
    building_labels = [b.replace('Building_', 'B') for b in buildings]

    # Calculate summary statistics for each building
    summary_stats = {}
    for building_id, data in building_data.items():
        # Filter out any zero or NaN values
        valid_temps = [t for t in data['zone_temperature'] if t > 0]
        summary_stats[building_id] = {
            'avg_temp': np.mean(valid_temps) if valid_temps else 20.0,
            'temp_std': np.std(valid_temps) if valid_temps else 0,
            'total_pv': np.sum(data['pv_generation_kw']) * 0.25,  # kWh
            'total_load': np.sum(data['total_load_kw']) * 0.25,  # kWh
            'peak_load': np.max(data['total_load_kw']) if data['total_load_kw'] else 0,
            'avg_battery_soc': np.mean([s for s in data['battery_soc'] if s > 0]),
            'total_grid_import': np.sum(np.maximum(data['grid_power_kw'], 0)) * 0.25,  # kWh
            'total_grid_export': np.sum(np.abs(np.minimum(data['grid_power_kw'], 0))) * 0.25  # kWh
        }

    # Plot 1: Temperature Performance
    ax = axes[0, 0]
    temps = [summary_stats[b]['avg_temp'] for b in buildings]
    temp_stds = [summary_stats[b]['temp_std'] for b in buildings]
    bars = ax.bar(range(len(buildings)), temps, yerr=temp_stds, capsize=5,
                  color='skyblue', edgecolor='navy', linewidth=1.5)
    ax.set_xticks(range(len(buildings)))
    ax.set_xticklabels(building_labels)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Average Zone Temperature (±std)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([15, 30])

    # Add values on bars
    for bar, temp in zip(bars, temps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{temp:.1f}°C', ha='center', va='bottom', fontsize=9)

    # Plot 2: Energy Balance
    ax = axes[0, 1]
    x = np.arange(len(buildings))
    width = 0.35
    loads = [summary_stats[b]['total_load'] for b in buildings]
    pvs = [summary_stats[b]['total_pv'] for b in buildings]

    bars1 = ax.bar(x - width / 2, loads, width, label='Load', color='orangered', alpha=0.8)
    bars2 = ax.bar(x + width / 2, pvs, width, label='PV Gen', color='gold', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(building_labels)
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('24h Energy Balance')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add self-sufficiency percentage
    for i, (load, pv) in enumerate(zip(loads, pvs)):
        self_suff = (pv / load * 100) if load > 0 else 0
        ax.text(i, max(load, pv) + 1, f'{self_suff:.0f}%',
                ha='center', va='bottom', fontsize=8, color='green')

    # Plot 3: Peak Load Analysis
    ax = axes[0, 2]
    peaks = [summary_stats[b]['peak_load'] for b in buildings]
    bars = ax.bar(range(len(buildings)), peaks, color='darkred', alpha=0.7)
    ax.set_xticks(range(len(buildings)))
    ax.set_xticklabels(building_labels)
    ax.set_ylabel('Power (kW)')
    ax.set_title('Peak Load')
    ax.grid(True, axis='y', alpha=0.3)

    # Add values
    for bar, peak in zip(bars, peaks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{peak:.1f}kW', ha='center', va='bottom', fontsize=9)

    # Plot 4: Battery Utilization
    ax = axes[1, 0]
    socs = [summary_stats[b]['avg_battery_soc'] for b in buildings]
    bars = ax.bar(range(len(buildings)), socs, color='green', alpha=0.7)
    ax.set_xticks(range(len(buildings)))
    ax.set_xticklabels(building_labels)
    ax.set_ylabel('SOC (%)')
    ax.set_title('Average Battery SOC')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    # Plot 5: Grid Exchange
    ax = axes[1, 1]
    imports = [summary_stats[b]['total_grid_import'] for b in buildings]
    exports = [summary_stats[b]['total_grid_export'] for b in buildings]

    x = np.arange(len(buildings))
    width = 0.35

    bars1 = ax.bar(x - width / 2, imports, width, label='Import', color='red', alpha=0.7)
    bars2 = ax.bar(x + width / 2, exports, width, label='Export', color='green', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(building_labels)
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Grid Exchange (24h)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Plot 6: Performance Metrics Summary
    ax = axes[1, 2]
    # Create a performance score based on self-sufficiency and peak reduction
    perf_scores = []
    for b in buildings:
        self_suff = (summary_stats[b]['total_pv'] / summary_stats[b]['total_load'] * 100) \
            if summary_stats[b]['total_load'] > 0 else 0
        peak_ratio = summary_stats[b]['peak_load'] / (summary_stats[b]['total_load'] / 24) \
            if summary_stats[b]['total_load'] > 0 else 1
        score = self_suff / peak_ratio  # Higher is better
        perf_scores.append(score)

    bars = ax.bar(range(len(buildings)), perf_scores, color='purple', alpha=0.7)
    ax.set_xticks(range(len(buildings)))
    ax.set_xticklabels(building_labels)
    ax.set_ylabel('Performance Score')
    ax.set_title('Overall Performance\n(Self-Sufficiency / Peak Ratio)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, summary_stats


# Main execution
if __name__ == "__main__":
    print("MULTI-BUILDING SIMULATION")
    PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))
    # Load configuration
    config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_N_Buildings", "config_setup_5buildings.json")
    building_data, env = run_multi_building_simulation(config_path=config_path, max_steps=96)

    # Create plots
    print("\nGenerating visualizations...")
    fig1 = plot_multi_building_results(building_data)
    fig2, stats = create_summary_dashboard(building_data)

    # Save plots
    output_dir = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_N_Buildings", "results")
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(os.path.join(output_dir, 'multi_building_results.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'multi_building_summary.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {output_dir}")

    # Print summary statistics
    print("SIMULATION SUMMARY")
    for building_id, stat in stats.items():
        print(f"\n{building_id}:")
        print(f"  Average Temperature: {stat['avg_temp']:.1f}°C (±{stat['temp_std']:.1f}°C)")
        print(f"  Total PV Generation: {stat['total_pv']:.1f} kWh")
        print(f"  Total Load: {stat['total_load']:.1f} kWh")
        print(f"  Peak Load: {stat['peak_load']:.1f} kW")
        print(f"  Self-sufficiency: {(stat['total_pv'] / stat['total_load'] * 100):.1f}%")
        print(f"  Grid Import: {stat['total_grid_import']:.1f} kWh")
        print(f"  Grid Export: {stat['total_grid_export']:.1f} kWh")
        print(f"  Net Grid: {(stat['total_grid_import'] - stat['total_grid_export']):.1f} kWh")

    # Calculate totals
    print("AGGREGATE METRICS:")
    total_pv = sum(stats[b]['total_pv'] for b in stats)
    total_load = sum(stats[b]['total_load'] for b in stats)
    total_import = sum(stats[b]['total_grid_import'] for b in stats)
    total_export = sum(stats[b]['total_grid_export'] for b in stats)

    print(f"  Total PV Generation: {total_pv:.1f} kWh")
    print(f"  Total Load: {total_load:.1f} kWh")
    print(f"  Overall Self-sufficiency: {(total_pv / total_load * 100):.1f}%")
    print(f"  Total Grid Import: {total_import:.1f} kWh")
    print(f"  Total Grid Export: {total_export:.1f} kWh")
    print(f"  Net Grid Energy: {(total_import - total_export):.1f} kWh")

    print("\n✓ Simulation complete!")

    # Show plots if not running in headless mode
    try:
        plt.show()
    except:
        print("Note: Unable to display plots (headless mode). Check saved files.")