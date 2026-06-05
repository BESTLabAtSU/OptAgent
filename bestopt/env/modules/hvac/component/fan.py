"""
Ideal HVAC module.
"""

from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import FanComponentAction, FanComponentState

class FanModule(BaseModule):
    """
    Supply fan module that CONSUMES an airflow setpoint and UPDATES a FanState in place.

    Input (from action): ThermalAction.fan_supply_air_flow_rate [m^3/s]

    Output (written in-place to FanState):
    - state.flow_m3s
    - state.power_W
    - state.energy_J_cum  (accumulated over steps)

    Model
    - VSD: power_W = self.rated_power_W * (0.00153 + 0.0052*PLR + 1.1086*(PLR)^2 - 0.1164*(PLR)^3)
    - Config keys:
        * rated_flow_m3s  (or rated_flow)  Q_rated [m^3/s]
        * rated_power_W                   P at Q_rated [W]

    State
    - Expects a FanState instance passed as `state`.
        * state.airflow_m3s     : echoed airflow [m^3/s]
        * state.power_W        : electric power via affinity law [W]
        * state.energy_J_cum  : accumulated energy over steps [W]

    Action
        No action.

    """

    def __init__(self, config: Dict[str, Any], name: str = "supply_fan"):
        super().__init__(config, name)
        q_rated = config.get("rated_flow_m3s", config.get("rated_flow", 1.0))
        self.rated_flow_m3s: float = float(q_rated)                 
        self.rated_power_W:  float = float(config.get("rated_power_W", 1000.0))   

        if self.rated_flow_m3s <= 0.0:
            self.logger.warning(f"{self.name}: rated_flow_m3s <= 0, power will be forced to 0.")

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "FanComponentState",
        action: "FanComponentAction",
        timestep: float
    ) -> FanComponentState:

        sp = action.airflow_setpoint_m3s
        flow = 0.0 if sp is None else float(sp)
        if flow < 0.0:
            self.logger.warning(f"{self.name}: negative flow received; clamped to 0.0")
            flow = 0.0

        if self.rated_flow_m3s > 0:
            flow = min(flow, self.rated_flow_m3s)

        if flow > 1e-6 and self.rated_flow_m3s > 0 and self.rated_power_W > 0:
            PLR = max(0.0, min(1.0, flow / self.rated_flow_m3s))
            power_W = self.rated_power_W * (0.00153 + 0.0052*PLR + 1.1086*(PLR**2) - 0.1164*(PLR**3))
        else:
            power_W = 0.0

        energy_J = power_W * (timestep if (timestep and timestep > 0.0) else 0.0)

        prev = getattr(state, "energy_J_cum", 0.0) or 0.0
        state.airflow_m3s  = flow          
        state.power_W      = power_W
        state.energy_J_cum = prev + energy_J

        self._record_state({"airflow_m3s": flow, "power_W": power_W, "energy_J": energy_J})
        return state




    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()