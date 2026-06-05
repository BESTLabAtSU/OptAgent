from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import PumpComponentState, PumpComponentAction


class PumpModule(BaseModule):
    """
    Pump module that consumes a pump water flow setpoint and updates a PumpState in place.

    Input (action):
        - ThermalAction.pump_flowrate [m^3/s]

    Output (written in-place to PumpState):
        - state.waterflow_m3s : actual water flow [m^3/s]
        - state.power_W       : electric power [W]
        - state.energy_J_cum  : accumulated energy [J]

    Model:
        Variable speed drive pump.
        power_W = rated_power_W * (0.00153 + 0.0052*PLR + 1.1086*PLR^2 - 0.1164*PLR^3)

    Config:
        - rated_flow_m3s (float, default 1.0): Rated flow [m^3/s]
        - rated_power_W  (float, default 1000.0): Power at rated flow [W]
    """

    def __init__(self, config: Dict[str, Any], name: str = "supply_pump"):
        super().__init__(config, name)
        q_rated = config.get("rated_flow_m3s", config.get("rated_flow", 1.0))
        self.rated_flow_m3s: float = float(q_rated)
        self.rated_power_W: float = float(config.get("rated_power_W", 1000.0))

        if self.rated_flow_m3s <= 0.0:
            self.logger.warning(f"{self.name}: rated_flow_m3s <= 0, power will be forced to 0.")

        # Internal state variables
        self.current_flow: float = 0.0
        self.current_power: float = 0.0
        self.energy_J_cum: float = 0.0

    # ------------------- lifecycle methods -------------------

    def initialize(self) -> None:
        """Prepare module before simulation starts."""
        self.current_flow = 0.0
        self.current_power = 0.0
        self.energy_J_cum = 0.0
        self._initialized = True

    def step(
        self,
        state: PumpComponentState,
        action: PumpComponentAction,
        timestep: float
    ) -> PumpComponentState:
        """
        One step (SI units):
          - Read water flow setpoint [m^3/s]
          - Compute power [W] via affinity law
          - Accumulate energy [J] = W * timestep
          - Write results in-place into PumpState
        """
        # 1) waterflow setpoint
        sp = action.flow_setpoint_m3s
        flow = 0.0 if sp is None else float(sp)
        if flow < 0.0:
            self.logger.warning(f"{self.name}: negative flow received; clamped to 0.0")
            flow = 0.0
        flow = min(flow, self.rated_flow_m3s)
        # 2) power computation
        # 2) power via affinity law [W]
        if flow > 1e-8 * self.rated_power_W and self.rated_power_W >= 0.0:
            PLR = max(0.0, min(1.0, flow / self.rated_flow_m3s))
            power_W = self.rated_power_W * (0.00153 + 0.0052*PLR + 1.1086*(PLR)**2 - 0.1164*(PLR)**3)
            power_W = max(power_W, 0.0)   # clamp to zero
        else:
            power_W = 0.0

        # 3) step energy [J]
        energy_J = power_W * (timestep if timestep and timestep > 0.0 else 0.0)

        # 4) update internal state
        self.current_flow = flow
        self.current_power = power_W
        self.energy_J_cum += energy_J

        # 5) write to external state
        state.waterflow_m3s = self.current_flow
        state.power_W = self.current_power
        state.energy_J_cum = self.energy_J_cum

        # record history
        self._record_state({
            "waterflow_m3s": flow,
            "power_W": power_W,
            "energy_J": energy_J
        })

        return state

    def reset(self) -> None:
        """Return to initial conditions for new episode."""
        self.clear_history()
        self.current_flow = 0.0
        self.current_power = 0.0
        self.energy_J_cum = 0.0
        self._initialized = True
