# bestopt/env/modules/hvac/boiler.py
"""
Boiler module.
"""

from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import BoilerState, Disturbance


class BoilerModule(BaseModule):
    """
    Boiler module that heats water to meet an outlet temperature setpoint.
    Updates a BoilerState in place.

    Inputs (from BoilerState):
        - inlet_temp_C
        - flow_m3s
        - outlet_temp_set_C

    Outputs:
        - outlet_temp_C
        - thermal_power_W
        - fuel_power_W
        - energy_J_cum

    Model:
        Q_demand = m * cp * (T_set − T_in)
        Q_actual = clamp(Q_demand, 0, capacity)
        Fuel power = Q_actual / efficiency
    """

    def __init__(self, config: Dict[str, Any], name: str = "boiler"):
        super().__init__(config, name)
        self.capacity = config.get("capacity_W", 10000.0)   # [W]
        self.efficiency = config.get("efficiency", 0.9)     # 90% default
        self.cp = config.get("cp", 4180.0)                  # J/kg-K (water)
        self.rho = config.get("rho", 1000.0)                # kg/m³ (water)

    def step(
        self,
        action: Dict[str, Any],        # not used here
        state: BoilerState,
        disturbance: Disturbance,
        dt: float,
    ):
        # Mass flow rate [kg/s]
        m_flow = state.flow_m3s * self.rho

        # Demand based on setpoint
        Q_demand = m_flow * self.cp * (state.outlet_temp_set_C - state.inlet_temp_C)

        # Boiler only supplies heat (no cooling)
        Q_actual = max(0.0, min(self.capacity, Q_demand))

        # Fuel consumption
        fuel_power = Q_actual / self.efficiency if self.efficiency > 0 else 0.0

        # Update outlet temp
        if m_flow > 0:
            state.outlet_temp_C = state.inlet_temp_C + Q_actual / (m_flow * self.cp)
        else:
            state.outlet_temp_C = state.inlet_temp_C

        # Update powers
        state.thermal_power_W = Q_actual
        state.fuel_power_W = fuel_power
        state.energy_J_cum += fuel_power * dt

    def reset(self, state: BoilerState):
        state.outlet_temp_C = 0.0
        state.thermal_power_W = 0.0
        state.fuel_power_W = 0.0
        state.energy_J_cum = 0.0

    def initialize(self, state: BoilerState):
        self.reset(state)
