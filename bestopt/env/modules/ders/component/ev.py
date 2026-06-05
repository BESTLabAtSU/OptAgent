"""
Electric Vehicle (EV) charging module.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import EVComponentState, DERSystemAction, Disturbance


class EVStatus(Enum):
    """EV operational status."""
    CONNECTED_IDLE = "connected_idle"
    CHARGING = "charging"
    DISCHARGING = "discharging"  # V2G/V2B
    DISCONNECTED = "disconnected"
    FAULT = "fault"


class EVModule(BaseModule):
    """
    Electric Vehicle energy storage and charging model.

    Features:
    - State of Charge (SOC) tracking
    - Connection/disconnection scheduling
    - Charging/discharging efficiency
    - C-rate limitations
    - V2G/V2B capability
    - Simplified degradation model
    """

    def __init__(self, config: Dict[str, Any], name: str = "EV"):
        """
        Initialize EV module.

        Args:
            config: EV configuration parameters
            name: Module name
        """
        super().__init__(config, name)

        # EV specifications
        self.capacity_kwh = config.get("rated_capacity_kWh", 60.0)  # Battery capacity in kWh
        self.nominal_voltage = config.get("nominal_voltage", 400.0)  # Volts
        self.max_charge_power_kw = config.get("max_charge_power_kw", 11.0)  # AC charging power (kW)
        self.max_discharge_power_kw = config.get("max_discharge_power_kw", 11.0)  # V2G/V2B power (kW)

        # SOC limits
        self.soc_min = config.get("soc_min", 0.2)  # Minimum SOC (20% - preserve battery life)
        self.soc_max = config.get("soc_max", 0.9)  # Maximum SOC (90% - preserve battery life)
        self.soc_initial = config.get("initial_soc", 0.5)  # Initial SOC
        self.soc_departure_target = config.get("soc_departure_target", 0.8)  # Target SOC for departure

        # Efficiency parameters
        self.charge_efficiency = config.get("charge_efficiency", 0.95)  # AC charging efficiency
        self.discharge_efficiency = config.get("discharge_efficiency", 0.95)  # V2G efficiency
        self.self_discharge_rate = config.get("self_discharge_rate", 0.00005)  # per hour (0.005%)

        # C-rate limits (simplified compared to battery)
        self.max_c_rate_charge = config.get("max_c_rate_charge", 0.5)  # 0.5C charging
        self.max_c_rate_discharge = config.get("max_c_rate_discharge", 0.5)  # 0.5C discharging

        # Connection schedule (list of tuples: (arrival_hour, departure_hour))
        self.connection_schedule = config.get("connection_schedule", [(18, 7)])  # Default: 6PM to 7AM
        self.always_connected = config.get("always_connected", False)  # Override for always connected

        # V2G/V2B settings
        self.v2g_enabled = config.get("v2g_enabled", True)  # Allow discharging
        self.v2g_min_soc = config.get("v2g_min_soc", 0.3)  # Don't discharge below this SOC

        # State tracking
        self.current_soc = self.soc_initial
        self.is_connected = config.get("initially_connected", True)
        self.last_connection_time = 0.0
        self.last_disconnection_time = 0.0
        self.total_energy_charged_kwh = 0.0
        self.total_energy_discharged_kwh = 0.0
        self.total_sessions = 0

        # Operational status
        self.status = EVStatus.CONNECTED_IDLE if self.is_connected else EVStatus.DISCONNECTED
        self.fault_message = ""

        # Power limits in Watts
        self.max_charge_power_w = self.max_charge_power_kw * 1000
        self.max_discharge_power_w = self.max_discharge_power_kw * 1000

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the EV module."""
        self.logger.info(f"Initialized EV module: {self.name}")
        self.logger.info(f"Capacity: {self.capacity_kwh} kWh")
        self.logger.info(f"Max Charge Power: {self.max_charge_power_kw} kW")
        self.logger.info(f"Initial SOC: {self.soc_initial:.1%}")
        self.logger.info(f"V2G Enabled: {self.v2g_enabled}")

        # Set initial state
        self.current_soc = self.soc_initial

    def step(self,
             state: EVComponentState,
             action: Dict,
             disturbance: Disturbance,
             timestep: int) -> EVComponentState:
        """
        Execute EV charging/discharging for current timestep.

        Args:
            state: Current EV state
            action: Dictionary with 'net_power_kw' key
            disturbance: Current disturbances
            timestep: Current simulation timestep

        Returns:
            Updated EV state
        """
        try:
            # Update connection status based on schedule
            # self._update_connection_status(state, timestep)
            #
            # # If disconnected, no power transfer possible
            # if not state.is_active:
            #     state.power_w = 0.0
            #     state.operation_mode = "DISCONNECTED"
            #     self.status = EVStatus.DISCONNECTED
            #     return state

            # Get power command from action
            power_command_kw = action.get('net_power_kw', 0.0)

            # Apply operational constraints
            # power_actual_kw = self._apply_constraints(power_command_kw, state.soc)
            power_actual_kw = power_command_kw
            if power_actual_kw==0.0 and disturbance.occupancy.occupancy_fraction==0:
                power_actual_kw = -10

            # Calculate timestep duration in hours
            timestep_hours = 900 / 3600  # 15 minutes = 0.25 hours

            # Calculate energy transferred
            energy_delta_kwh = self._calculate_energy_transfer(power_actual_kw, timestep_hours)

            # Update SOC
            new_soc = self._update_soc(state.soc, energy_delta_kwh)

            # Apply self-discharge
            new_soc = self._apply_self_discharge(new_soc, timestep_hours)

            # Update state
            state.soc = new_soc
            # state.power_w = power_actual_kw * 1000  # Convert to Watts

            # Update operation mode and status
            # if power_actual_kw > 0.1:  # Charging (threshold to avoid noise)
            #     state.operation_mode = "CHARGING"
            #     self.status = EVStatus.CHARGING
            # elif power_actual_kw < -0.1:  # Discharging
            #     state.operation_mode = "V2G"
            #     self.status = EVStatus.DISCHARGING
            # else:
            #     state.operation_mode = "IDLE"
            #     self.status = EVStatus.CONNECTED_IDLE
            #
            # # Update cumulative metrics
            # if power_actual_kw > 0:
            #     self.total_energy_charged_kwh += energy_delta_kwh
            # else:
            #     self.total_energy_discharged_kwh += abs(energy_delta_kwh)
            #
            # # Log significant operations
            # if abs(power_actual_kw) > 1.0:  # Only log > 1kW operations
            #     operation = "Charging" if power_actual_kw > 0 else "V2G"
            #     self.logger.debug(
            #         f"{operation}: {abs(power_actual_kw):.1f}kW, "
            #         f"SOC: {new_soc:.1%}, "
            #         f"Status: {self.status.value}"
            #     )

        except Exception as e:
            self.logger.error(f"Error in EV step calculation: {e}")
            self.status = EVStatus.FAULT
            self.fault_message = str(e)
            state.power_w = 0.0

        return state

    def _update_connection_status(self, state: EVComponentState, timestep: int) -> None:
        """
        Update EV connection status based on schedule.

        Args:
            state: Current EV state
            timestep: Current simulation timestep in seconds
        """
        if self.always_connected:
            state.is_active = True
            self.is_connected = True
            return

        # Convert timestep to hour of day (0-24)
        hour_of_day = (timestep / 3600) % 24

        # Check if EV should be connected based on schedule
        was_connected = self.is_connected
        self.is_connected = False

        for arrival_hour, departure_hour in self.connection_schedule:
            if departure_hour > arrival_hour:
                # Normal case: arrival before departure (e.g., 18-23)
                if arrival_hour <= hour_of_day < departure_hour:
                    self.is_connected = True
                    break
            else:
                # Overnight case: departure after midnight (e.g., 18-7)
                if hour_of_day >= arrival_hour or hour_of_day < departure_hour:
                    self.is_connected = True
                    break

        # Update state
        state.is_active = self.is_connected

        # Track connection events
        if self.is_connected and not was_connected:
            self.last_connection_time = timestep
            self.total_sessions += 1
            self.logger.info(f"EV connected at hour {hour_of_day:.1f}")
        elif not self.is_connected and was_connected:
            self.last_disconnection_time = timestep
            self.logger.info(f"EV disconnected at hour {hour_of_day:.1f}, SOC: {state.soc:.1%}")

    def _apply_constraints(self, power_command_kw: float, current_soc: float) -> float:
        """
        Apply EV operational constraints.

        Args:
            power_command_kw: Requested power in kW (positive=charge, negative=discharge)
            current_soc: Current state of charge (0-1)

        Returns:
            Actual power after applying constraints (kW)
        """
        if power_command_kw > 0:  # Charging
            # Check SOC limit
            if current_soc >= self.soc_max:
                return 0.0

            # Apply power and C-rate limits
            max_charge = min(
                self.max_charge_power_kw,
                self.max_c_rate_charge * self.capacity_kwh
            )

            # SOC-based derating (slower charging near full)
            if current_soc > 0.8:
                soc_factor = 1.0 - (current_soc - 0.8) * 5  # Rapid taper above 80%
                max_charge *= max(0.1, soc_factor)

            return min(power_command_kw, max_charge)

        elif power_command_kw < 0:  # Discharging (V2G/V2B)
            # Check if V2G is enabled
            if not self.v2g_enabled:
                return 0.0

            # Check SOC limit for V2G
            if current_soc <= self.v2g_min_soc:
                return 0.0

            # Apply power and C-rate limits
            max_discharge = min(
                self.max_discharge_power_kw,
                self.max_c_rate_discharge * self.capacity_kwh
            )

            # SOC-based derating (protect battery at low SOC)
            if current_soc < 0.4:
                soc_factor = (current_soc - self.v2g_min_soc) / (0.4 - self.v2g_min_soc)
                max_discharge *= max(0.1, soc_factor)

            return max(power_command_kw, -max_discharge)

        return 0.0

    def _calculate_energy_transfer(self, power_kw: float, timestep_hours: float) -> float:
        """
        Calculate energy transferred accounting for efficiency.

        Args:
            power_kw: Power in kW
            timestep_hours: Timestep duration in hours

        Returns:
            Energy change in kWh (positive = added to battery)
        """
        if power_kw > 0:  # Charging
            energy_kwh = power_kw * timestep_hours * self.charge_efficiency
        elif power_kw < 0:  # Discharging
            energy_kwh = power_kw * timestep_hours / self.discharge_efficiency
        else:
            energy_kwh = 0.0

        return energy_kwh

    def _update_soc(self, current_soc: float, energy_delta_kwh: float) -> float:
        """
        Update SOC based on energy transfer.

        Args:
            current_soc: Current SOC (0-1)
            energy_delta_kwh: Energy change in kWh

        Returns:
            New SOC (0-1)
        """
        # Calculate SOC change
        soc_delta = energy_delta_kwh / self.capacity_kwh
        new_soc = current_soc + soc_delta

        # Enforce limits
        new_soc = max(self.soc_min, min(self.soc_max, new_soc))

        return new_soc

    def _apply_self_discharge(self, soc: float, timestep_hours: float) -> float:
        """
        Apply self-discharge losses (minimal for EVs).

        Args:
            soc: Current SOC
            timestep_hours: Timestep in hours

        Returns:
            Updated SOC after self-discharge
        """
        discharge_factor = 1.0 - (self.self_discharge_rate * timestep_hours)
        return soc * discharge_factor

    def get_available_charge_power(self, soc: float) -> float:
        """
        Get available charging power at current SOC.

        Args:
            soc: Current SOC

        Returns:
            Available charge power in kW
        """
        if not self.is_connected or soc >= self.soc_max:
            return 0.0

        power = self.max_charge_power_kw

        # SOC-based derating
        if soc > 0.8:
            power *= (1.0 - (soc - 0.8) * 5)

        return power

    def get_available_discharge_power(self, soc: float) -> float:
        """
        Get available V2G discharge power at current SOC.

        Args:
            soc: Current SOC

        Returns:
            Available discharge power in kW
        """
        if not self.is_connected or not self.v2g_enabled or soc <= self.v2g_min_soc:
            return 0.0

        power = self.max_discharge_power_kw

        # SOC-based derating
        if soc < 0.4:
            power *= (soc - self.v2g_min_soc) / (0.4 - self.v2g_min_soc)

        return power

    def reset(self) -> None:
        """Reset EV module to initial state."""
        self.current_soc = self.soc_initial
        self.is_connected = self.config.get("initially_connected", True)
        self.last_connection_time = 0.0
        self.last_disconnection_time = 0.0
        self.total_energy_charged_kwh = 0.0
        self.total_energy_discharged_kwh = 0.0
        self.total_sessions = 0
        self.status = EVStatus.CONNECTED_IDLE if self.is_connected else EVStatus.DISCONNECTED
        self.fault_message = ""
        self.logger.debug(f"Reset EV module: {self.name}")

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of EV state.

        Returns:
            Dictionary with key metrics
        """
        return {
            "soc": self.current_soc,
            "capacity_kwh": self.capacity_kwh,
            "is_connected": self.is_connected,
            "status": self.status.value,
            "total_charged_kwh": self.total_energy_charged_kwh,
            "total_discharged_kwh": self.total_energy_discharged_kwh,
            "total_sessions": self.total_sessions,
            "v2g_enabled": self.v2g_enabled,
            "available_charge_power_kw": self.get_available_charge_power(self.current_soc),
            "available_discharge_power_kw": self.get_available_discharge_power(self.current_soc),
            "time_to_target_soc_hours": self._estimate_time_to_target()
        }

    def _estimate_time_to_target(self) -> float:
        """
        Estimate time to reach departure target SOC.

        Returns:
            Hours needed to reach target SOC
        """
        if not self.is_connected or self.current_soc >= self.soc_departure_target:
            return 0.0

        energy_needed = (self.soc_departure_target - self.current_soc) * self.capacity_kwh
        avg_charge_power = self.max_charge_power_kw * 0.8  # Assume 80% of max on average

        return energy_needed / (avg_charge_power * self.charge_efficiency)