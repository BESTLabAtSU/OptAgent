from typing import Dict, Any, Optional
from math import inf

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACSystemAction, HVACSystemState, PumpComponentAction, ComponentType
from bestopt.env.core.constants import WATER_DENSITY, WATER_SPECIFIC_HEAT


class PumpLocalController(BaseModule):
    """
    Local pump controller. Water flow is controlled based on a constant 
    temperature difference across the coil.

    Config:
        - delta_T (float, default 5.0): Desired temperature difference across the coil [K].
        - pump_flowrate_max (float, default inf): Maximum allowable flowrate [m^3/s].

    Input:
        - ThermalAction.thermal_load [W] 

    Output:
        - ThermalAction.pump_flowrate [m^3/s] = thermal_load / (delta_T * rho * cp)
    """

    def __init__(self, config: Dict[str, Any], name: str = "PumpLocalController"):
        super().__init__(config, name)
        self.delta_T: float = float(config.get("delta_T", 5.0))  # K
        self.coil_effectiveness: float = float(config.get("coil_effectiveness", 0.8))
        self.Tcw_sp: float = float(config.get("Tcw_setpoint", 5))   # C
        # if self.delta_T <= 0:
        #     raise ValueError("delta_T must be a positive number.")

        self.pump_flowrate_max: float = float(config.get("pump_flowrate_max", inf))  # m^3/s
        if self.pump_flowrate_max <= 0:
            raise ValueError("pump_flowrate_max must be a positive number.")

    def initialize(self) -> None:
        """Initialize controller state."""
        self._initialized = True
        self.current_flowrate = None

    def step(
        self,
        state: HVACSystemState,
        action: HVACSystemAction,
        timestep: float
    ) -> PumpComponentAction:
        """
        Compute local pump command and return a local-controller action.

        Args:
            action (ThermalAction): Contains the thermal load in Watts.
            timestep (float): Current timestep [s].

        Returns:
            ThermalAction: The computed pump flowrate command.
        """
        #@ TODO I use flowrate and temperature setpoint here to estimate thermal demand, is it OK?
        # Calculate required flowrate
        #@ TODO should delta T update across time
        # pump_flowrate = abs(action.hvac_thermal_load_demand / (self.delta_T * WATER_DENSITY * WATER_SPECIFIC_HEAT))

        pump_flowrate = abs(action.hvac_thermal_load_demand / (self.coil_effectiveness * (24 - self.Tcw_sp) * WATER_DENSITY * WATER_SPECIFIC_HEAT))

        # Enforce maximum limit
        pump_flowrate = min(pump_flowrate, self.pump_flowrate_max)

        self.current_flowrate = pump_flowrate

        local_action = PumpComponentAction(
            component_id=state.system_id,
            component_type=ComponentType.PUMP.value
        )
        local_action.flow_setpoint_m3s = pump_flowrate

        # Optionally record state for debugging/monitoring
        # self._record_state({
        #     "thermal_load": thermal_load,
        #     "pump_flowrate": pump_flowrate
        # })

        return local_action

    def reset(self) -> None:
        """Reset controller state."""
        self.clear_history()
        self.current_flowrate = None
        self._initialized = True