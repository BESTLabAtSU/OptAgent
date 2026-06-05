"""
Disturbance price module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from ..core.base import BaseModule
from ..core.data_structure import PriceSignals


class PriceModule(BaseModule):
    """
    Price energy storage model.

    Features:
    # @TODO
    -
    -
    -
    """

    def __init__(self, config: Dict[str, Any], name: str = "Price"):
        """
        Initialize price module.

        Args:
            config: Price configuration parameters
            name: Module name
        """
        super().__init__(config, name)
        self.price_signal = PriceSignals()

    def initialize(self) -> None:
        self.peak_start = 17 * 4
        self.peak_end = 21 * 4
        daily_price = np.ones(96) * 10
        daily_price[self.peak_start:self.peak_end] = 15
        self.daily_price = daily_price
        daily_peaksignal = np.zeros(96, dtype=bool)
        daily_peaksignal[self.peak_start:self.peak_end] = True
        self.daily_peaksignal = daily_peaksignal

    def step(self, current_step: int) -> Optional[PriceSignals]:
        step_of_day = current_step % 96
        self.price_signal.electricity_price = self.daily_price[step_of_day]
        self.price_signal.peaksignal = self.daily_peaksignal[step_of_day]
        self.price_signal.peak_start = self.peak_start
        self.price_signal.peak_end = self.peak_end
        self.price_signal.forecast_peaksignal = self.daily_peaksignal[step_of_day:step_of_day+96]

        return self.price_signal

    def reset(self) -> None:
        pass
