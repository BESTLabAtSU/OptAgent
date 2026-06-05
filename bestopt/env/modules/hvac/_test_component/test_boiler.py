# bestopt/env/modules/hvac/test_boiler.py

from bestopt.env.modules.hvac.component.boiler import BoilerModule
from bestopt.env.core.data_structure import BoilerState, Disturbance


def test_boiler():
    config = {"capacity_W": 8000, "efficiency": 0.9}
    boiler = BoilerModule(config)
    state = BoilerState(
        inlet_temp_C=40.0,
        flow_m3s=0.05,
        outlet_temp_set_C=60.0,
    )
    disturbance = Disturbance()
    dt = 3600  # 1 hour

    boiler.step({}, state, disturbance, dt)

    print("Outlet temp [°C]:", state.outlet_temp_C)
    print("Thermal power [W]:", state.thermal_power_W)
    print("Fuel power [W]:", state.fuel_power_W)
    print("Cumulative energy [kWh]:", state.energy_J_cum / 3.6e6)


if __name__ == "__main__":
    test_boiler()
