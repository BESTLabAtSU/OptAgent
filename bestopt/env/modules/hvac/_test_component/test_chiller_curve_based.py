from bestopt.env.modules.hvac.component.chiller_curve_based import ChillerCurveBased
from bestopt.env.core.data_structure import ChillerState, ThermalAction, PumpState, CoilState

# Define configuration using default E+ curves
cfg = {
    "rated_capacity_W": 150_000,   # 150 kW
    "rated_cop": 6.0,
    "min_plr": 0.15,
    "max_plr": 1.05,
    "enable_history": True
}

# Instantiate module
chiller = ChillerCurveBased(cfg, name="chiller_curve_based")
chiller.initialize()

# Create state objects
cs = ChillerState(component_id="chiller_curve_based", component_type="", domain="")
coil_state = CoilState()
pump_state = PumpState()

# Simulated schedules for 24 timesteps (e.g., 15-min intervals for 6 hours)
chw_return_schedule = [12.0 - 0.1 * i for i in range(24)]           # gradually cooling return temp
chw_flow_schedule = [0.02] * 24                                     # constant 0.02 m³/s
chws_sp_schedule = [7.0 + 0.2 * (i % 3) for i in range(24)]         # slight setpoint variation
cond_temp_schedule = [30.0 + 0.5 * (i % 4) for i in range(24)]      # varying condenser temp

dt_sec = 900  # 15 minutes

# === Run 24-timestep simulation ===
print("=== 24-Step ChillerCurveBased Simulation ===")
for t in range(24):
    coil_state.water_outlet_temp_C = chw_return_schedule[t]
    pump_state.waterflow_m3s = chw_flow_schedule[t]
    action = ThermalAction(
        chws_temp_c_sp=chws_sp_schedule[t],
        condenser_temp_c_sp=cond_temp_schedule[t]
    )

    chiller.step(cs, action=action, coil_state=coil_state, pump_state=pump_state, timestep=dt_sec)

    print(f"Step {t:02d} | Q={cs.cooling_W:.1f} W | COP={cs.cop:.2f} | T_chws={cs.chws_temp_c:.1f}°C | "
          f"Flow={cs.chw_flow_m3s:.3f} m³/s | Power={cs.power_W:.1f} W | E_cum={cs.energy_J_cum:.1f} J")

# === Summary ===
print("\nTotal Energy Consumed (J):", round(cs.energy_J_cum, 2))
