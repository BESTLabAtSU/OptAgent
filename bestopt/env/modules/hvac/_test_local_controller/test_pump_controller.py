#%%
import sys, os
# Add project root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# bestopt/env/controllers/test_pump_local_controller.py

from dataclasses import dataclass

from bestopt.env.modules.hvac.local_controller.pump_local_controller import PumpLocalController
from bestopt.env.core.data_structure import ThermalAction, HVACLocalAction

pump_local_controller = PumpLocalController({}, name="pump_local_controller")
pump_local_controller.initialize()

ip_act = ThermalAction(thermal_load=4186*5*1000*80)

op_act = pump_local_controller.step(action=ip_act, timestep=900)

print(op_act.pump_flowrate)   
#%%
# test pump_local_controller and pump

from bestopt.env.modules.hvac.component.pump import PumpModule
from bestopt.env.core.data_structure import PumpState, ThermalAction, HVACLocalAction

# single step
cfg = {"rated_flow": 82.0, "rated_power_W": 4000.0, "enable_history": True}
pump = PumpModule(cfg, name="pump")
pump.initialize()

# create a standalone PumpState (not stored under ThermalDomainState)
fs0 = PumpState(component_id="pump", component_type="", domain="")  # __post_init__ will set domain/type

# one step
fs = pump.step(state=fs0, action=op_act, timestep=900)

print(fs.waterflow_m3s)        
print(fs.power_W)         
print(fs.energy_J_cum)   