"""
Battery Energy Storage System (BESS) module.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import BatteryComponentState, DERSystemAction, Disturbance


class BatteryStatus(Enum):
    """Battery operational status."""
    NORMAL = "normal"
    WARNING = "warning"
    FAULT = "fault"
    MAINTENANCE = "maintenance"


class BatteryModule(BaseModule):
    """
    Battery energy storage model with detailed physics.

    Features:
    - State of Charge (SOC) tracking with coulomb counting
    - Charging/discharging efficiency curves
    - C-rate limitations and voltage-based constraints
    - Thermal effects on performance
    - Degradation modeling (cycle and calendar aging)
    - Safety limits and protection mechanisms
    """

    def __init__(self, config: Dict[str, Any], name: str = "Battery"):
        """
        Initialize battery module.

        Args:
            config: Battery configuration parameters
            name: Module name
        """
        super().__init__(config, name)

        # Battery specifications
        self.capacity_kwh = config.get("rated_capacity_kWh", 10.0)  # Nominal capacity in kWh
        self.capacity_ah = config.get("capacity_ah", self.capacity_kwh * 1000 / 400)  # Ah (assuming 400V nominal)
        self.nominal_voltage = config.get("nominal_voltage", 400.0)  # Volts
        self.max_power_kw = config.get("max_power_kw", 5.0)  # Maximum charge/discharge power

        # SOC limits
        self.soc_min = config.get("soc_min", 0.1)  # Minimum SOC (10%)
        self.soc_max = config.get("soc_max", 0.9)  # Maximum SOC (90%)
        self.soc_initial = config.get("initial_soc", 0.5)  # Initial SOC

        # Efficiency parameters
        self.charge_efficiency = config.get("charge_efficiency", 0.95)  # Charging efficiency
        self.discharge_efficiency = config.get("discharge_efficiency", 0.95)  # Discharging efficiency
        self.roundtrip_efficiency = self.charge_efficiency * self.discharge_efficiency
        self.self_discharge_rate = config.get("self_discharge_rate", 0.0001)  # per hour (0.01%)

        # C-rate limits
        self.max_c_rate_charge = config.get("max_c_rate_charge", 1.0)  # 1C charging
        self.max_c_rate_discharge = config.get("max_c_rate_discharge", 2.0)  # 2C discharging

        # Thermal parameters
        self.temp_optimal = config.get("temp_optimal", 25.0)  # °C
        self.temp_min = config.get("temp_min", -20.0)  # °C
        self.temp_max = config.get("temp_max", 45.0)  # °C
        self.temp_derating_factor = config.get("temp_derating_factor", 0.01)  # per °C from optimal

        # Degradation parameters
        self.cycle_life = config.get("cycle_life", 5000)  # Number of equivalent full cycles
        self.calendar_life_years = config.get("calendar_life_years", 10)  # Years
        self.end_of_life_capacity = config.get("end_of_life_capacity", 0.8)  # 80% of original

        # State tracking
        self.current_soc = self.soc_initial
        self.current_capacity_kwh = self.capacity_kwh  # Current capacity (degrades over time)
        self.current_temperature = self.temp_optimal
        self.total_cycles = 0.0
        self.age_years = 0.0
        self.total_energy_charged_kwh = 0.0
        self.total_energy_discharged_kwh = 0.0

        # Operational status
        self.status = BatteryStatus.NORMAL
        self.fault_message = ""

        # Power limits in Watts
        self.max_charge_power_w = self.max_power_kw * 1000
        self.max_discharge_power_w = self.max_power_kw * 1000

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the battery module."""
        self.logger.info(f"Initialized Battery module: {self.name}")
        self.logger.info(f"Capacity: {self.capacity_kwh} kWh")
        self.logger.info(f"Max Power: {self.max_power_kw} kW")
        self.logger.info(f"Initial SOC: {self.soc_initial:.1%}")

        # Set initial state
        self.current_soc = self.soc_initial

    def step(self,
             state: BatteryComponentState,
             action: Dict,
             disturbance: Disturbance,
             timestep: int) -> BatteryComponentState:
        """
        Execute battery charging/discharging for current timestep.

        Args:
            state: Current battery state
            action: Electrical action with power commands
            disturbance: Current disturbances
            timestep: Current simulation timestep

        Returns:
            Dictionary with battery operation metrics
        """
        try:
            # Get temperature from disturbance (could be indoor or outdoor)
            if hasattr(disturbance.weather, 'outdoor_dry_bulbtemperature'):
                self.current_temperature = disturbance.weather.outdoor_dry_bulb_temp

            # Extract power command from action
            # power_command = self._get_power_command(action)
            #I am using simplifoed for now
            power_command = action['net_power_kw']

            # Apply operational constraints
            # power_actual = self._apply_constraints(power_command, state.soc)
            power_actual = power_command # apply constraint later
            # Calculate energy transferred (assuming 1-hour timestep, adjust as needed)
            # Note: You should get the actual timestep duration from the environment
            timestep_hours = 900 / 3600  # Convert seconds to hours if timestep is in seconds
            # Or get from environment configuration

            # Update SOC based on power flow
            energy_delta_kwh = self._calculate_energy_transfer(power_actual, timestep_hours)
            new_soc = self._update_soc(state.soc, energy_delta_kwh)

            # Apply self-discharge
            new_soc = self._apply_self_discharge(new_soc, timestep_hours)

            # Update state
            state.soc = new_soc
            state.temperature = self.current_temperature
        except:
            print("error")

        return state

        #
        #     # Update degradation
        #     self._update_degradation(abs(energy_delta_kwh))
        #
        #     # Update cumulative metrics
        #     if power_actual > 0:
        #         self.total_energy_charged_kwh += energy_delta_kwh
        #     else:
        #         self.total_energy_discharged_kwh += abs(energy_delta_kwh)
        #
        #     # Log operation
        #     if abs(power_actual) > 10:  # Only log significant operations
        #         operation = "Charging" if power_actual > 0 else "Discharging"
        #         self.logger.debug(
        #             f"{operation}: {abs(power_actual):.1f}W, "
        #             f"SOC: {new_soc:.1%}, "
        #             f"Status: {self.status.value}"
        #         )
        #
        #     # Check for warnings
        #     self._check_status(new_soc, self.current_temperature)
        #
        #     return {
        #         "power_actual": power_actual,
        #         "energy_transferred_kwh": energy_delta_kwh,
        #         "soc": new_soc,
        #         "available_charge_power": self._get_available_charge_power(new_soc),
        #         "available_discharge_power": self._get_available_discharge_power(new_soc),
        #         "status": self.status.value
        #     }
        #
        # except Exception as e:
        #     self.logger.error(f"Error in Battery step calculation: {e}")
        #     self.status = BatteryStatus.FAULT
        #     self.fault_message = str(e)
        #     return {"power_actual": 0.0, "error": str(e), "status": BatteryStatus.FAULT.value}

    def _get_power_command(self, action: DERSystemAction) -> float:
        """
        Extract power command from electrical action.

        Args:
            action: Electrical action containing battery commands

        Returns:
            Power command in Watts (positive = charge, negative = discharge)
        """
        # The controller should set battery power in the action
        # Check various possible action attributes

        # Net power to/from battery
        power = 0.0

        # Charging sources
        if hasattr(action, 'grid2battery'):
            power += action.grid2battery
        if hasattr(action, 'pv2battery'):
            power += action.pv2battery

        # Discharging destinations
        if hasattr(action, 'battery2building'):
            power -= action.battery2building
        if hasattr(action, 'battery2grid'):
            power -= action.battery2grid
        if hasattr(action, 'battery2ev'):
            power -= action.battery2ev

        return power

    def _apply_constraints(self, power_command: float, current_soc: float) -> float:
        """
        Apply battery operational constraints.

        Args:
            power_command: Requested power (W)
            current_soc: Current state of charge (0-1)

        Returns:
            Actual power after applying constraints (W)
        """
        # Temperature derating
        temp_derate = self._calculate_temp_derating()

        if power_command > 0:  # Charging
            # Check SOC limit
            if current_soc >= self.soc_max:
                return 0.0

            # Apply C-rate limit
            max_charge = min(
                self.max_charge_power_w,
                self.max_c_rate_charge * self.current_capacity_kwh * 1000
            )

            # Apply temperature derating
            max_charge *= temp_derate

            # Apply SOC-based derating (slower charging near full)
            if current_soc > 0.8:
                soc_derate = 1.0 - (current_soc - 0.8) * 2.5  # Linear reduction from 80% to 90%
                max_charge *= max(0.2, soc_derate)

            return min(power_command, max_charge)

        elif power_command < 0:  # Discharging
            # Check SOC limit
            if current_soc <= self.soc_min:
                return 0.0

            # Apply C-rate limit
            max_discharge = min(
                self.max_discharge_power_w,
                self.max_c_rate_discharge * self.current_capacity_kwh * 1000
            )

            # Apply temperature derating
            max_discharge *= temp_derate

            # Apply SOC-based derating (reduced power at low SOC)
            if current_soc < 0.2:
                soc_derate = current_soc * 5  # Linear reduction below 20%
                max_discharge *= max(0.2, soc_derate)

            return max(power_command, -max_discharge)

        return 0.0

    def _calculate_temp_derating(self) -> float:
        """
        Calculate temperature-based derating factor.

        Returns:
            Derating factor (0-1)
        """
        if self.current_temperature < self.temp_min or self.current_temperature > self.temp_max:
            return 0.0  # Battery protection - no operation outside limits

        # Linear derating from optimal temperature
        temp_diff = abs(self.current_temperature - self.temp_optimal)
        derating = 1.0 - (temp_diff * self.temp_derating_factor)

        return max(0.2, min(1.0, derating))

    def _calculate_energy_transfer(self, power: float, timestep_hours: float) -> float:
        """
        Calculate energy transferred accounting for efficiency.

        Args:
            power: Power in Watts
            timestep_hours: Timestep duration in hours

        Returns:
            Energy change in kWh (positive = added to battery)
        """
        if power > 0:  # Charging
            energy_kwh = (power ) * timestep_hours * self.charge_efficiency
        else:  # Discharging
            energy_kwh = (power) * timestep_hours / self.discharge_efficiency

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
        soc_delta = energy_delta_kwh / self.current_capacity_kwh
        new_soc = current_soc + soc_delta

        # Enforce limits
        new_soc = max(self.soc_min, min(self.soc_max, new_soc))

        return new_soc

    def _apply_self_discharge(self, soc: float, timestep_hours: float) -> float:
        """
        Apply self-discharge losses.

        Args:
            soc: Current SOC
            timestep_hours: Timestep in hours

        Returns:
            Updated SOC after self-discharge
        """
        discharge_factor = 1.0 - (self.self_discharge_rate * timestep_hours)
        return soc * discharge_factor

    def _update_degradation(self, energy_throughput_kwh: float) -> None:
        """
        Update battery degradation based on usage.

        Args:
            energy_throughput_kwh: Energy cycled in kWh
        """
        # Cycle aging
        cycle_increment = energy_throughput_kwh / (2 * self.capacity_kwh)  # Full cycle = charge + discharge
        self.total_cycles += cycle_increment

        # Calculate capacity fade
        cycle_fade = min(0.2, (self.total_cycles / self.cycle_life) * 0.2)  # 20% max fade from cycling
        calendar_fade = min(0.2, (self.age_years / self.calendar_life_years) * 0.2)  # 20% max fade from age

        total_fade = cycle_fade + calendar_fade
        self.current_capacity_kwh = self.capacity_kwh * (1.0 - total_fade)

    def _check_status(self, soc: float, temperature: float) -> None:
        """
        Check battery status and set warnings/faults.

        Args:
            soc: Current SOC
            temperature: Current temperature
        """
        self.status = BatteryStatus.NORMAL
        self.fault_message = ""

        # Check SOC warnings
        if soc < 0.15:
            self.status = BatteryStatus.WARNING
            self.fault_message = "Low SOC warning"
        elif soc > 0.95:
            self.status = BatteryStatus.WARNING
            self.fault_message = "High SOC warning"

        # Check temperature warnings
        if temperature < 0 or temperature > 40:
            self.status = BatteryStatus.WARNING
            self.fault_message = "Temperature outside optimal range"

        # Check degradation
        if self.current_capacity_kwh < self.capacity_kwh * self.end_of_life_capacity:
            self.status = BatteryStatus.MAINTENANCE
            self.fault_message = "Battery reached end of life"

    def _get_available_charge_power(self, soc: float) -> float:
        """
        Get available charging power at current SOC.

        Args:
            soc: Current SOC

        Returns:
            Available charge power in Watts
        """
        if soc >= self.soc_max:
            return 0.0

        power = self.max_charge_power_w

        # SOC-based derating
        if soc > 0.8:
            power *= (1.0 - (soc - 0.8) * 2.5)

        # Temperature derating
        power *= self._calculate_temp_derating()

        return power

    def _get_available_discharge_power(self, soc: float) -> float:
        """
        Get available discharging power at current SOC.

        Args:
            soc: Current SOC

        Returns:
            Available discharge power in Watts
        """
        if soc <= self.soc_min:
            return 0.0

        power = self.max_discharge_power_w

        # SOC-based derating
        if soc < 0.2:
            power *= soc * 5

        # Temperature derating
        power *= self._calculate_temp_derating()

        return power

    def reset(self) -> None:
        """Reset battery module to initial state."""
        self.current_soc = self.soc_initial
        self.current_capacity_kwh = self.capacity_kwh
        self.current_temperature = self.temp_optimal
        self.total_cycles = 0.0
        self.total_energy_charged_kwh = 0.0
        self.total_energy_discharged_kwh = 0.0
        self.status = BatteryStatus.NORMAL
        self.fault_message = ""
        self.logger.debug(f"Reset Battery module: {self.name}")

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of battery state.

        Returns:
            Dictionary with key metrics
        """
        return {
            "soc": self.current_soc,
            "capacity_kwh": self.current_capacity_kwh,
            "original_capacity_kwh": self.capacity_kwh,
            "degradation": 1.0 - (self.current_capacity_kwh / self.capacity_kwh),
            "total_cycles": self.total_cycles,
            "age_years": self.age_years,
            "total_charged_kwh": self.total_energy_charged_kwh,
            "total_discharged_kwh": self.total_energy_discharged_kwh,
            "status": self.status.value,
            "temperature": self.current_temperature,
            "available_charge_power_w": self._get_available_charge_power(self.current_soc),
            "available_discharge_power_w": self._get_available_discharge_power(self.current_soc)
        }