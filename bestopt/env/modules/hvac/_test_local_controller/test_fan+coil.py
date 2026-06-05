# bestopt/env/controllers/test_fan_local_controller.py
from dataclasses import dataclass

from bestopt.env.modules.hvac.local_controller.fan_local_controller0 import FanLocalController
from bestopt.env.core.data_structure import ThermalAction, HVACLocalAction

# One step (15 min); action is unused so we pass None
dt = 900  # s

fan_local_controller = FanLocalController({}, name="fan_local_controller")
fan_local_controller.initialize()


ip_fan_local_ctrl = ThermalAction(supervisory_supply_air_flow_rate=1.0)

op_fan_local_ctrl = fan_local_controller.step(action=ip_fan_local_ctrl, timestep=dt)

print(op_fan_local_ctrl.fan_supply_air_flow_rate)   




# test fan_local_controller and fan

from bestopt.env.modules.hvac.component.fan import FanModule
from bestopt.env.core.data_structure import FanState, ThermalAction, HVACLocalAction

# single step

cfg = {"rated_flow": 2.0, "rated_power_W": 4000.0, "enable_history": True}
fan = FanModule(cfg, name="supply_fan")
fan.initialize()

# create a standalone FanState (not stored under ThermalDomainState)
fs0 = FanState(component_id="supply_fan", component_type="", domain="")  # __post_init__ will set domain/type

# one step
op_fan = fan.step(state=fs0, action=op_fan_local_ctrl, timestep=dt)

print(op_fan.airflow_m3s)        
print(op_fan.power_W)         
print(op_fan.energy_J_cum)   




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
cs0 = CoilState(component_id="CC-1", component_type="", domain="")

cs0.airflow_m3s = op_fan.airflow_m3s       
cs0.waterflow_m3s = 0.01       # Should from pump state
cs0.air_inlet_temp_C = ThermalAction(return_air_temperature=30).return_air_temperature
cs0.water_inlet_temp_C = 10.0    # Should from chiller state

print(cs0)

# One step (15 min); action is unused so we pass None
op_coil = coil.step(state=cs0, action=None, timestep=dt)

print(f"Air:  Tin={cs0.air_inlet_temp_C:.1f}C  Tout={op_coil.air_outlet_temp_C:.2f}C")
print(f"Water:Tin={cs0.water_inlet_temp_C:.1f}C  Tout={op_coil.water_outlet_temp_C:.2f}C")
print(f"Q={cs0.Q_W:.2f}W")



