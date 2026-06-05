"""
Ice Tank TES module.
"""

from typing import Dict, Any
import numpy as np

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ThermalAction, IceTankState


class IceTankModule(BaseModule):
    """
    Ice-based Thermal Energy Storage (TES) tank.

    Inputs:
        - ThermalAction.ice_tank_mode        : str ["charge", "discharge", "idle"]
        - ThermalAction.ice_tank_power_W_sp  : float (positive scalar)

    Outputs:
        - IceTankState.soc                   : 0.0 - 1.0
        - IceTankState.q_actual_W            : heat absorbed or released (W)
        - IceTankState.energy_J_cum          : energy charged/discharged (J)
    """

    def __init__(self, config: Dict[str, Any], name: str = "ice_tank"):
        super().__init__(config, name)
        self.capacity_kWh = float(config.get("capacity_kWh", 500.0))
        self.capacity_J = self.capacity_kWh * 3.6e6  # Convert to joules
        self.charge_eff = float(config.get("charge_efficiency", 0.95))
        self.discharge_eff = float(config.get("discharge_efficiency", 0.95))
        self.max_rate_W = float(config.get("max_rate_W", 50_000))  # 50 kW default

    def initialize(self) -> None:
        self._initialized = True

    def step(self, state: "IceTankState", action: "ThermalAction", timestep: float) -> Dict[str, Any]:
        mode = getattr(action, "ice_tank_mode", "idle").lower()
        power_sp = float(getattr(action, "ice_tank_power_W_sp", 0.0))
        power_sp = np.clip(power_sp, 0.0, self.max_rate_W)

        soc = getattr(state, "soc", 0.0)
        q_actual = 0.0
        energy_J = 0.0

        if mode == "charge" and soc < 1.0:
            energy_J = power_sp * timestep * self.charge_eff
            new_energy = soc * self.capacity_J + energy_J
            new_energy = min(new_energy, self.capacity_J)
            q_actual = power_sp
        elif mode == "discharge" and soc > 0.0:
            energy_J = -power_sp * timestep / self.discharge_eff
            new_energy = soc * self.capacity_J + energy_J
            new_energy = max(new_energy, 0.0)
            q_actual = -power_sp
        else:
            new_energy = soc * self.capacity_J
            q_actual = 0.0

        new_soc = np.clip(new_energy / self.capacity_J, 0.0, 1.0)

        state.soc = new_soc
        state.q_actual_W = q_actual
        state.energy_J_cum += abs(energy_J)

        self._record_state({
            "SOC": new_soc,
            "Q_Actual_W": q_actual,
            "Energy_J": abs(energy_J)
        })

        return {}

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()