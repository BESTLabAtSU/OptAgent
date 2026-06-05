from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.scripts.runtime_plotter import create_hvac_dashboard
from bestopt.scripts.elec_plotter import create_electrical_dashboard  # Import the new dashboard
import os
from pathlib import Path
PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))
# Load configuration
config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_1_Building", "config_setup.json")
cm = ConfigurationManager(config_path)
env = BESTOptEnvironment(cm.config)
# Create both dashboards
hvac_plotter = create_hvac_dashboard(max_points=96 * 1, window_title="HVAC System Monitor")
electrical_plotter = create_electrical_dashboard(max_points=96 * 1, window_title="Electrical System Monitor")
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
    zone_temperature = env.cluster_states[cluster_id].thermal.systems['SFH_1_building'].components['zone0'].temperature
    # Get supervisory setpoints from action
    cooling_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].cooling_setpoint_c
    heating_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].heating_setpoint_c
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
    supply_air_temp_setpt = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].supply_temp_setpoint_c
    supply_air_flow_setpt = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].supply_airflow_setpoint_m3s
    
    water_flow_real = hvac_system.CHW_flow_actual_m3s
    # water_flow_setpt = env.cluster_actions[cluster_id].thermalFCUModule.system_actions['hvac_system_1'].
    
    chiller_supply_water_temp =  hvac_system.CHW_supply_temp_C
        
    # Get electrical system data
    bat_soc = der_system.battery_states['bat_1'].soc
    ev_tesla_soc = der_system.ev_states['ev_tesla'].soc
    ev_nissan_soc = der_system.ev_states['ev_nissan'].soc
    pv_generation = der_system.pv_states['pv_1'].generation_w / 1000  # Convert to kW
    # Get loads (all in kW)
    building_total_load = HVAC_power / 1000+env.cluster_states['residential_cluster_1'].electrical.systems['SFH_1_building'].components['electrical'].building_power_w / 1000
    HVAC_power_kw = HVAC_power / 1000  # Convert HVAC power to kW
    cooking = env.electrical_zone_modules['SFH_1.zone0'].cooking_power / 1000
    pc = env.electrical_zone_modules['SFH_1.zone0'].pc_power / 1000
    tv = env.electrical_zone_modules['SFH_1.zone0'].tv_power / 1000
    lighting = env.electrical_zone_modules['SFH_1.zone0'].lighting_power / 1000
    #
    grid2building = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2building
    grid2ev_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2ev
    grid2battery_dict = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1'].grid2battery
    grid2ev = sum(grid2ev_dict.values())
    grid2battery = sum(grid2battery_dict.values())
    # Get peak signal
    is_peak = env.disturbance.prices.peaksignal  # True or False
    electricity_price = env.disturbance.prices.electricity_price # ¢/kWh
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
        supply_air_flow_setpt=supply_air_flow_setpt,
        water_flow_real=water_flow_real,                     # m³/s (pass 0 if zero)
        chiller_supply_water_temp=chiller_supply_water_temp  # °C
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
# Stop plotters and save GIFs
print("\n:bar_chart: Stopping dashboards and saving animations...")
hvac_plotter.stop()
hvac_plotter.save_as_gif('sfh1_hvac_dashboard.gif', fps=20)
electrical_plotter.stop()
electrical_plotter.save_as_gif('sfh1_electrical_dashboard.gif', fps=20)
print(f"\n:white_check_mark: Simulation completed: {env.current_step} steps")
print(":file_folder: Saved animations: sfh1_hvac_dashboard.gif, sfh1_electrical_dashboard.gif")