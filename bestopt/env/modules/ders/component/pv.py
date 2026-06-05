"""
PV module.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import PVComponentState, Disturbance


class PVModule(BaseModule):
    """

    """

    def __init__(self, config: Dict[str, Any], name: str = "PV"):
        """

        """
        super().__init__(config, name)
        # https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
        # PV system specifications
        self.rated_power_kw = config.get("rated_power_kw", 5.0)  # Nominal power in kW
        self.panel_area_m2 = config.get("panel_area_m2", 25.0)  # Total panel area in m²
        self.efficiency_stc = config.get("efficiency_stc", 0.20)  # Efficiency at STC (20%)

        # Temperature coefficients
        self.temp_coeff_power = config.get("temp_coeff_power", -0.004)  # %/°C
        self.noct = config.get("noct", 45.0)  # Nominal Operating Cell Temperature (°C)
        self.stc_temp = config.get("stc_temp", 25.0)  # Standard Test Conditions temp (°C)
        self.stc_irradiance = config.get("stc_irradiance", 1000.0)  # W/m²

        # Degradation and losses
        self.annual_degradation = config.get("annual_degradation", 0.005)  # 0.5% per year
        self.soiling_factor = config.get("soiling_factor", 0.98)  # 2% soiling loss
        self.shading_factor = config.get("shading_factor", 1.0)  # No shading by default
        self.inverter_efficiency = config.get("inverter_efficiency", 0.97)  # 97% inverter efficiency
        self.dc_losses = config.get("dc_losses", 0.98)  # 2% DC wiring losses
        self.min_irradiance = config.get("min_irradiance", 10.0)  # Minimum irradiance for operation (W/m²)
        self.max_power_output = self.rated_power_kw * 1000  # Convert to Watts

        # Installation details
        self.tilt_angle = config.get("tilt_angle", 30.0)  # degrees
        self.azimuth = config.get("azimuth", 180.0)  # degrees (180 = south-facing)

        # Disturbance
        self.ambient_temp = None
        self.irradiance = None

        # Tracking
        self.lifetime_energy_kwh = 0.0
        self.operating_hours = 0.0
        self.age_years = config.get("initial_age_years", 0.0)

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the PV module."""
        self.logger.info(f"Initialized PV module: {self.name}")
        self.logger.info(f"Rated power: {self.rated_power_kw} kW")
        self.logger.info(f"Panel area: {self.panel_area_m2} m²")
        self._initialized = True

    def step(self,
             state: PVComponentState,
             disturbance: Disturbance,
             resolution: int,
             timestep: int) -> PVComponentState:
        """

        """
        try:
            # Get weather conditions
            self.irradiance = disturbance.weather.solar_radiation_w_m2  # W/m²
            self.ambient_temp = disturbance.weather.outdoor_dry_bulb_temp  # °C

            # Calculate cell temperature using NOCT model
            cell_temp = self._calculate_cell_temperature()

            # Calculate degradation factor based on age
            degradation_factor = self._calculate_degradation_factor()

            # Calculate efficiency
            efficiency = self._calculate_efficiency(cell_temp)

            # Calculate losses
            losses = self._calculate_losses(degradation_factor)

            if self.irradiance < self.min_irradiance:
                power = 0.0
            else:
                power = self.rated_power_kw * 1000 * efficiency * losses
                power = min(power, self.max_power_output * 1.1)

            # Update state
            state.generation_w = power
            state.cell_temperature_c = cell_temp
            state.degradation_factor = degradation_factor
            state.efficiency = efficiency
            state.losses = losses

            # Update tracking metrics
            energy_kwh = (power / 1000) * (resolution / 3600)
            self.lifetime_energy_kwh += energy_kwh
            if self.irradiance > self.min_irradiance:
                self.operating_hours += resolution / 3600

            # Log generation details
            if power > 1:
                self.logger.debug(
                    f"PV Generation: {power:.1f}W, "
                    f"Irradiance: {self.irradiance:.1f}W/m², "
                    f"Cell Temp: {cell_temp:.1f}°C, "
                    f"Efficiency: {state.efficiency:.1%}, "
                    f"Losses: {state.losses:.1%}, "
                )

            return state

        except Exception as e:
            self.logger.error(f"Error in PV step calculation: {e}")
            state.power_generation = 0.0
            return state

    def _calculate_cell_temperature(self) -> float:
        """
        Calculate PV cell temperature using NOCT model.

        Returns:
            Cell temperature in °C
        """
        # NOCT model: Tc = Ta + (NOCT - 20) * (G / 800)
        if self.irradiance <= 0:
            return self.ambient_temp

        cell_temp = self.ambient_temp + (self.noct - 20) * (self.irradiance / 800)
        return cell_temp

    def _calculate_degradation_factor(self) -> float:
        """
        Calculate degradation factor based on PV system age.

        Returns:
            Degradation factor (0-1)
        """
        # Linear degradation model
        degradation = 1.0 - (self.age_years * self.annual_degradation)
        return max(0.7, degradation)  # Minimum 70% of original capacity

    def _calculate_losses(self, degradation_factor: float) -> float:
        """PV losses."""
        return (
                self.soiling_factor *
                self.shading_factor *
                self.inverter_efficiency *
                self.dc_losses *
                degradation_factor
        )

    def _calculate_efficiency(self, cell_temp: float) -> float:
        """
        Calculate overall system efficiency factor
        """
        if self.irradiance < self.min_irradiance:
            return 0.0

        # Irradiance effect (normalized to STC)
        irradiance_factor = self.irradiance / self.stc_irradiance

        # Temperature derating
        temp_derating = 1.0 + self.temp_coeff_power * (cell_temp - self.stc_temp)

        # Low-light adjustment
        low_light_factor = 0.95 if self.irradiance < 200 else 1.0

        # Combined efficiency factor
        efficiency_factor = irradiance_factor * temp_derating * low_light_factor

        return max(0.0, min(efficiency_factor, 1.1))

    def update_age(self, time_delta_hours: float):
        """Update system age."""
        self.age_years += time_delta_hours / 8760

    def reset(self) -> None:
        pass
        self.logger.debug(f"Reset PV module: {self.name}")

    def get_state_summary(self) -> Dict[str, Any]:
        pass
