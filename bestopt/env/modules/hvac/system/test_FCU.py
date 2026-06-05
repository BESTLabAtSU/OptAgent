import numpy as np
from bestopt.env.modules.hvac.system.FCU import FCUModule

from bestopt.env.core.data_structure import (
    State, Action, Disturbance, Observation,
    BatteryState, PVState, BLDGEState, EVState,
    HVACState, BLDGTState, TESState,
    WaterHeaterState,WeatherData,
    ThermalAction, ElectricalAction, WaterAction, ThermalDomainState
)
# === FCU System Config ===
fcu_config = {
    "fan": {"rated_flow_m3s": 8, "rated_power_W": 8*1000}, #fan_power_per_flow ≈ 1,000 – 1,500 W per m³/s
    "fan_ctrl": {"ctrl_type": "linear"},
    "pump": {"rated_flow_m3s": 0.01, "rated_power_W": 0.01*100_000}, # pump_power_per_flow = 100,000 W per m³/s
    "chiller": {"rated_capacity_W": 90_000, "rated_cop": 5.5},
    "tower": {"rated_capacity_W": 100_000}
}

# === Instantiate FCU system ===
fcu = FCUModule(config=fcu_config)
fcu.initialize()

# === Setup inputs ===
timestep_sec = 3600  # 60 minutes
n_steps = 24

# Fake weather & disturbance
weather = WeatherData()
weather.outdoor_wet_bulb_temperature = 24.0
disturbance = Disturbance(weather=weather)
state = State(building_id='SFH_1')
# === Simulate 24 steps ===
print("=== FCU 24-step Simulation ===")
for t in range(n_steps):
    action = ThermalAction()

    state.thermal = ThermalDomainState(hvac_systems={
        'fcu': HVACState(component_id='fcu', component_type='hvac', domain='thermal', timestamp=0.0, is_active=True,
                         thermal_load=0.0)}, thermal_zones={
        'zone0': BLDGTState(component_id='zone0', component_type='bldg_t', domain='thermal', timestamp=0.0,
                            is_active=True, temperature=22.47007248919996, humidity=50.0, internal_heat_gain=0.0)},
                       thermal_storage={}, total_heating_load=0.0, total_cooling_load=0.0)
    # === Fake Supervisory Controller Outputs ===
    action.thermal_load = -8_0000 + 2000 * (t % 6)  # Cooling demand (negative)
    action.supervisory_supply_air_temperature = 15.0       # SAT setpoint
    action.supervisory_supply_air_flow_rate = abs(action.thermal_load)/1005/1.225/13 # m³/s
    action.chws_temp_c_sp = 5.0              # CHW setpoint
    action.condenser_temp_c_sp = 25.0        # CW setpoint
    action.return_air_temperature = 25.0 + 1 * (t % 6)           # Simulated zone return air

    result = fcu.step(state.thermal, action=action, disturbance=disturbance, timestep=timestep_sec)

    print(
        f"Step {t:02d} | "
        f"Q_set={action.thermal_load:6.0f} W | "
        f"Q_actual={result['Q_zone_actual_W']:6.0f} W | "
        f"SAT_set={action.supervisory_supply_air_temperature:4.1f}°C | "
        f"SAT_actual={result['SAT_actual_C']:4.1f}°C | "
        f"Air_flow={result['SA_flow_actual_m3s']:.2f} m³/s | "
        f"CHW_flow={result['CHW_flow_actual_m3s']:.5f} m³/s | "
        
        f"CHW_out={result['CHW_supply_temp_C']:4.1f}°C | "

        f"Total_Power={result['FCU_power_total_W']:6.0f} W | "
        f"Fan_Power={result['FCU_fan_power_W']:6.0f} W | "
        f"Pump_Power={result['FCU_pump_power_W']:6.0f} W | "
        f"Chiller_Power={result['FCU_chiller_power_W']:6.0f} W | "
        f"Tower_Fan_Power={result['FCU_tower_fan_power_W']:6.0f} W | "
        f"Tower_Pump_Power={result['FCU_tower_pump_power_W']:6.0f} W | "
        
        f"Sys_COP={abs(result['Q_zone_actual_W'])/result['FCU_power_total_W']:4.2f} | "
        f"E_cum={result['FCU_energy_cumulative_J']/1000/3600:4.1f} kWh"
    )

# === Summary ===
print("\nFinal cumulative energy consumed (J):", result["FCU_energy_cumulative_J"])
