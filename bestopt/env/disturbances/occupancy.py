"""
Disturbance occupancy module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from ..core.base import BaseModule
from ..core.data_structure import OccupancyDisturbance




class OccupancyModule(BaseModule):
    """
    Occupancy model.

    Features:
    # @TODO
    -
    -
    -
    """

    def __init__(self, config: Dict[str, Any], name: str = "Occupancy"):
        """
        Initialize occupancy module.

        Args:
            config: Occupancy configuration parameters
            name: Module name
        """
        super().__init__(config, name)
        self.current_occ = OccupancyDisturbance()

    def initialize(self) -> None:
        leave = int(np.random.normal(8 * 4, 2 * 4))
        back = int(np.random.normal(17 * 4, 2 * 4))
        daily = np.ones(96)
        daily[leave:back]=0
        self.daily = daily

    def step(self, current_step: int) -> Optional[OccupancyDisturbance]:
        step_of_day = current_step % 96
        self.current_occ.occupancy_fraction = self.daily[step_of_day]
        self.current_occ.step_of_day = step_of_day
        return self.current_occ

    def reset(self) -> None:
        pass
