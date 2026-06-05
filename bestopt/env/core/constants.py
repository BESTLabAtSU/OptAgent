"""
This module provides basic physical constants and unit conversions.
"""

from typing import Dict, Final

# Air properties (standard conditions)
AIR_DENSITY: Final[float] = 1.225  # kg/m³ at 15°C, 101.325 kPa
AIR_SPECIFIC_HEAT: Final[float] = 1005.0  # J/(kg·K)
AIR_THERMAL_CONDUCTIVITY: Final[float] = 0.0257  # W/(m·K)

# Water properties
WATER_DENSITY: Final[float] = 1000.0  # kg/m³
WATER_SPECIFIC_HEAT: Final[float] = 4186.0  # J/(kg·K)
WATER_THERMAL_CONDUCTIVITY: Final[float] = 0.6  # W/(m·K)

# Solar and atmospheric constants
SOLAR_CONSTANT: Final[float] = 1361.0  # W/m²
STEFAN_BOLTZMANN: Final[float] = 5.67e-8  # W/(m²·K⁴)
STANDARD_ATMOSPHERIC_PRESSURE: Final[float] = 101325.0  # Pa
STANDARD_TEMPERATURE: Final[float] = 273.15  # K (0°C)
STANDARD_GRAVITY: Final[float] = 9.80665  # m/s²


# Temperature conversions
def celsius_to_kelvin(temp_c: float) -> float:
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k: float) -> float:
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15


def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


# Energy conversions
KWH_TO_J: Final[float] = 3.6e6  # J/kWh
J_TO_KWH: Final[float] = 1.0 / KWH_TO_J
BTU_TO_J: Final[float] = 1055.06  # J/BTU
J_TO_BTU: Final[float] = 1.0 / BTU_TO_J
THERM_TO_J: Final[float] = 1.05506e8  # J/therm (US)

# Power conversions
HP_TO_W: Final[float] = 745.7  # W/hp
W_TO_HP: Final[float] = 1.0 / HP_TO_W
TON_TO_W: Final[float] = 3516.85  # W/ton (refrigeration)

# Time conversions
SECONDS_PER_HOUR: Final[float] = 3600.0
HOURS_PER_DAY: Final[float] = 24.0
DAYS_PER_YEAR: Final[float] = 365  # @TODO leap year
SECONDS_PER_DAY: Final[float] = SECONDS_PER_HOUR * HOURS_PER_DAY

# Simulation default
SIMULATION_DEFAULTS: Final[Dict[str, float]] = {
    'resolution': 15*60,  # use second for consistency
}

