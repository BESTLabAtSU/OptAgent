# bestopt/env/controllers/test_fan_local_controller.py
from dataclasses import dataclass

from bestopt.env.modules.hvac.local_controller.fan_local_controller0 import FanLocalController
from bestopt.env.core.data_structure import ThermalAction, HVACLocalAction

fan_local_controller = FanLocalController({"ctrl_type": "staged", "rated_flow_m3s": 2.0}, name="fan_local_controller")
fan_local_controller.initialize()


ip_act = ThermalAction(supervisory_supply_air_flow_rate=0.6)

op_act = fan_local_controller.step(action=ip_act, timestep=900)

print(op_act.fan_supply_air_flow_rate)   




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
fs = fan.step(state=fs0, action=op_act, timestep=900)

print(fs.airflow_m3s)        
print(fs.power_W)         
print(fs.energy_J_cum)   



