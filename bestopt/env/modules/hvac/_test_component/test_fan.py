from bestopt.env.modules.hvac.component.fan import FanModule
from bestopt.env.core.data_structure import FanState, ThermalAction, HVACLocalAction

cfg = {"rated_flow": 2.0, "rated_power_W": 4000.0, "enable_history": True}
fan = FanModule(cfg, name="supply_fan")
fan.initialize()

# create a standalone FanState (not stored under ThermalDomainState)
fs0 = FanState(component_id="supply_fan", component_type="", domain="")  # __post_init__ will set domain/type

# create an action with a setpoint
act = HVACLocalAction(fan_supply_air_flow_rate=1.2)

# one step
fs = fan.step(state=fs0, action=act, timestep=900)

print(fs.airflow_m3s)        
print(fs.power_W)         
print(fs.energy_J_cum)   




# Many steps

flow_schedule = [1.0, 1.2, 0.8, 1.5, 1.0, 1.2, 0.8, 1.5]

for i, q in enumerate(flow_schedule, 1):
    act.fan_supply_air_flow_rate = q

    ret = fan.step(state=fs0, action=act, timestep=900)
    if ret is not None:
        fs = ret  

    # 打印当前步结果（功率W、累计能量J与kWh）
    print(f"step {i:02d}: "
          f"q={fs.airflow_m3s:.3f} m^3/s, "
          f"P={fs.power_W:.2f} W, "
          f"E_cum={fs.energy_J_cum:.2f} J ({fs.energy_J_cum/3.6e6:.6f} kWh)")



