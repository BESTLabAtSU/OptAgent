from bestopt.env.modules.hvac.component.coil import CoilModule
from bestopt.env.core.data_structure import CoilState, Disturbance

# Config: pick an effectiveness (0..1) and default properties
cfg = {
    "effectiveness": 0.9,
    "enable_history": True,
}
coil = CoilModule(cfg, name="cooling_coil")
coil.initialize()

# Build a CoilState; set flows and inlet temperatures (constants for this test)
cs = CoilState(component_id="CC-1", component_type="", domain="")
cs.airflow_m3s = 1.2        # m^3/s (air)
cs.waterflow_m3s = 0.01     # m^3/s (water) ~ 10 L/s
cs.air_inlet_temp_C = 30.0  # °C
cs.water_inlet_temp_C = 7.0 # °C chilled water

# One step (15 min); action is unused so we pass None
dt = 900  # s
coil.step(state=cs, action=None, disturbance=Disturbance(), timestep=dt)

print(f"Air:  Tin={30.0:.1f}C  Tout={cs.air_outlet_temp_C:.2f}C")
print(f"Water:Tin={7.0:.1f}C  Tout={cs.water_outlet_temp_C:.2f}C")
print(f"Q={cs.Q_W:.2f}W")



# Multi-step example with varying air inlet (water kept constant)
for Ta_in in [30, 28, 26, 24, 22]:
    cs.air_inlet_temp_C = float(Ta_in)
    coil.step(state=cs, action=None, disturbance=Disturbance(), timestep=dt)
    print(f"[Ta={Ta_in:>2}C], Air_out={cs.air_outlet_temp_C:.2f}C, Water_out={cs.water_outlet_temp_C:.2f}C")

