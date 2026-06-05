from bestopt.env.modules.hvac.component.ice_tank import IceTankModule
from bestopt.env.core.data_structure import IceTankState, ThermalAction

cfg = {
    "capacity_kWh": 500,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "max_rate_W": 50_000
}

tank = IceTankModule(cfg, name="ice_tank1")
tank.initialize()

state = IceTankState(component_id="ice_tank1", component_type="", domain="")

# Simulate charging for 2 hours (8 steps of 900s)
for i in range(8):
    act = ThermalAction(ice_tank_mode="charge", ice_tank_power_W_sp=30000)
    tank.step(state=state, action=act, timestep=900)
    print(f"Step {i:02d} | Mode: charge | SOC: {state.soc:.3f} | Q: {state.q_actual_W:.0f} W | E: {state.energy_J_cum:.0f} J")

# Simulate discharging for 1 hour (4 steps)
for i in range(4):
    act = ThermalAction(ice_tank_mode="discharge", ice_tank_power_W_sp=30000)
    tank.step(state=state, action=act, timestep=900)
    print(f"Step {i+8:02d} | Mode: discharge | SOC: {state.soc:.3f} | Q: {state.q_actual_W:.0f} W | E: {state.energy_J_cum:.0f} J")