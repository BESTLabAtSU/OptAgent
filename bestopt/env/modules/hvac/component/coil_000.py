import numpy as np
from typing import Dict, Any, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACState, ThermalAction, Disturbance


class CoolingCoilModule(BaseModule):
    def __init__(self, config: Dict[str, Any], name: str = "CoolingCoilModule"):
        """Initialize fan module with configuration."""
        super().__init__(config, name)
        pass

    def initialize(self) -> None:
        pass

    def step(self, state: HVACState, action: Any, disturbance: Disturbance,
             timestep: float) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        pass

    def set_state(self, state: Dict[str, Any]) -> None:
        pass


class HeatingCoilModule(BaseModule):
    def __init__(self, config: Dict[str, Any], name: str = "HeatingCoilModule"):
        """Initialize fan module with configuration."""
        super().__init__(config, name)
        pass

    def initialize(self) -> None:
        pass

    def step(self, state: HVACState, action: Any, disturbance: Disturbance,
             timestep: float) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        pass

    def set_state(self, state: Dict[str, Any]) -> None:
        pass