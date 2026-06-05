import numpy as np
from typing import Dict, Any, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACState, ThermalAction, Disturbance


class FanModule(BaseModule):

    def __init__(self, config: Dict[str, Any], name: str = "FanModule"):
        """Initialize fan module with configuration."""
        super().__init__(config, name)

        # Fan configuration
        self.fan_type = config.get("fan_type", "ideal")  # ideal, constant, staged, variable

        if self.fan_type == "constant":
            self.design_air_flow_rate = config.get("design_air_flow_rate", 0.5)

        elif self.fan_type == "staged":
            self.stage_air_flow_rate = config.get("stage_air_flow_rate")

        elif self.fan_type == "variable":
            self.min_air_flow_rate = config.get("min_air_flow_rate", 0.1)
            self.max_air_flow_rate = config.get("max_air_flow_rate", 1.0)

        #@TODO need add some paras to calculate the fan power

    def initialize(self) -> None:
        """Initialize the fan module."""
        pass

    def step(self, state: HVACState, action: Any, disturbance: Disturbance,
             timestep: float) -> None:
        """
        Execute fan dynamics for one timestep.
        """
        try:
            # Determine actual fan action based on fan type
            if self.local_controller:
                fan_action = action
            else:
                print("warning, no local fan controller defined")

            # Execute fan dynamics based on type
            if self.fan_type == "ideal":
                self._execute_ideal_fan(fan_action, state)
            elif self.fan_type == "constant":
                self._execute_constant_fan(fan_action, state)
            elif self.fan_type == "staged":
                self._execute_staged_fan(fan_action, state)
            elif self.fan_type == "variable":
                self._execute_variable_fan(fan_action, state)
            else:
                self.logger.warning(f"Unknown fan type: {self.fan_type}")

            # Update state with fan outputs
            state.supply_air_flow_rate = self.current_flow_rate
            #state.fan_power = self.current_power

        except Exception as e:
            self.logger.error(f"Error in fan step: {e}")

    def _execute_ideal_fan(self, action: Dict[str, Any], state: HVACState) -> None:
        pass

    def _execute_constant_fan(self, action: Dict[str, Any], state: HVACState) -> None:
        pass

    def _execute_staged_fan(self, action: Dict[str, Any], state: HVACState) -> None:
        """
        """
        self.current_flow_rate = self.stage_air_flow_rate[action['stage']]

    def _execute_variable_fan(self, action: Dict[str, Any], state: HVACState) -> None:
        pass

    def reset(self) -> None:
        """Reset fan to initial state."""
        pass
        self.logger.debug(f"Reset fan: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        pass

    def set_state(self, state: Dict[str, Any]) -> None:
        pass