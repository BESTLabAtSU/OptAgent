from bestopt.env.modules.hvac.component.cooling_tower import CoolingTowerModule
from bestopt.env.core.data_structure import CoolingTowerState, ChillerState, ThermalAction, WeatherData

# === Configuration ===
cfg = {
    "rated_capacity_W": 200_000,
    "rated_fan_power_W": 5000,
    "min_approach_C": 3.0,
    "max_approach_C": 7.0,
    "temp_range_C": 5.0,
    "enable_history": True
}

# === Instantiate and initialize module ===
tower = CoolingTowerModule(cfg, name="tower1")
tower.initialize()

# === Initialize state objects ===
tower_state = CoolingTowerState(component_id="tower1", component_type="", domain="")
weather = WeatherData()
action = ThermalAction()

# === Simulated input profiles for 24 timesteps ===
cooling_W_profile = [80_000 + 2_000 * (i % 6) for i in range(24)]  # Stepwise increase
cop_profile = [5.5 - 0.05 * (i % 6) for i in range(24)]             # Decreasing COP
wet_bulb_profile = [24.0 + 0.2 * (i % 4) for i in range(24)]        # Small variations
cw_sp_temp = 30.0                                                  # CW setpoint constant

timestep_sec = 900  # 15 minutes

# === Simulation loop ===
print("=== 24-Step Cooling Tower Simulation ===")
for t in range(24):
    chiller_state = ChillerState(
        component_id="chiller1",
        cooling_W=cooling_W_profile[t],
        cop=cop_profile[t]
    )
    weather.outdoor_wet_bulb_temperature = wet_bulb_profile[t]
    action.condenser_temp_c_sp = cw_sp_temp

    tower.step(state=tower_state, action=action, chiller_state=chiller_state, weather=weather, timestep=timestep_sec)

    print(f"Step {t:02d} | Q_rej={tower_state.heat_rejected_W:6.0f} W | CW_out={tower_state.cw_supply_temp_c:4.1f}°C | "
          f"CW_in={tower_state.cw_return_temp_c:4.1f}°C | "
          f"Flow={tower_state.cw_flow_m3s:.4f} m³/s | Fan={tower_state.fan_power_W:5.0f} W | "
          f"Pump={tower_state.pump_power_W:4.0f} W | E_cum={tower_state.energy_J_cum:.0f} J")

# === Summary ===
print("\nTotal fan energy consumed (J):", round(tower_state.energy_J_cum, 2))
