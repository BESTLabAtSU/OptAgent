"""
Ideal HVAC module.
"""

from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ThermalAction, FanState, Action, Disturbance  # 'state' not used here

class FanModule(BaseModule):
    """
    Supply fan module that CONSUMES an airflow setpoint and UPDATES a FanState in place.

    Input (from action): Thermal Action supplyfan_flow_sp [m^3/s]

    Output (written in-place to FanState):
    - state.flow_m3s
    - state.power_W
    - state.energy_J_cum  (accumulated over steps)

    Model
    - Fan affinity law:  P = P_rated * (Q / Q_rated)^exponent
    - Config keys:
        * rated_flow_m3s  (or rated_flow)  Q_rated [m^3/s]
        * rated_power_W                   P at Q_rated [W]
        * power_exponent (default 3.0)     cube-law exponent

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
        self.power_exponent: float = float(config.get("power_exponent", 3.0))    

        if self.rated_flow_m3s <= 0.0:
            self.logger.warning(f"{self.name}: rated_flow_m3s <= 0, power will be forced to 0.")

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "FanState",
        action: Any,      # not used
        disturbance: "Disturbance",
        timestep: float
    ) -> Dict[str, Any]:
        """
        One step (SI):
          - read airflow setpoint [m^3/s]
          - compute power [W] via affinity law
          - accumulate step energy [J] = W * s
          - write results IN-PLACE into FanState
        """
        
        # 1) airflow setpoint [m^3/s]
        sp = getattr(action, "supplyfan_flow_sp", None)
        if sp is None and hasattr(action, "thermal"):
            sp = getattr(action.thermal, "supplyfan_flow_sp", None)

        flow = 0.0 if sp is None else float(sp)
        if flow < 0.0:
            self.logger.warning(f"{self.name}: negative flow received; clamped to 0.0")
            flow = 0.0

        # 2) power via affinity law [W]
        if self.rated_flow_m3s > 0.0 and self.rated_power_W >= 0.0:
            ratio = flow / self.rated_flow_m3s
            power_W = self.rated_power_W * (ratio ** self.power_exponent)
        else:
            power_W = 0.0

        # 3) step energy [J]; timestep in seconds
        energy_J = power_W * (timestep if (timestep and timestep > 0.0) else 0.0)

        # 4) in-place update (pure SI)
        state.airflow_m3s = flow
        state.power_W = power_W
        state.energy_J_cum = getattr(state, "energy_J_cum", 0.0) + energy_J

        # 5) optional history
        self._record_state({"airflow_m3s": flow, "power_W": power_W, "energy_J": energy_J})
        
        return {}


    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()