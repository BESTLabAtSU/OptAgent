"""
Weather disturbance module.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os
import pandas as pd
from ..core.base import BaseModule
from ..core.data_structure import WeatherDisturbance

REQUIRED_COLS_CSV = {"outdoor_temperature", "solar_radiation"}

class WeatherModule(BaseModule):
    """
    Weather module that provides outdoor temperature and solar radiation.
    Supports CSV (with columns: outdoor_temperature, solar_radiation)
    and EPW (basic fields parsed).
    """

    def __init__(self, config: Dict[str, Any], name: str = "Weather"):
        super().__init__(config, name)
        self.weather_data: Optional[pd.DataFrame] = None
        self.weather = WeatherDisturbance()
        self.current_timestep = 0
        # @TODO for future large scale evaluation
        self.location = config.get("location", "Syracuse, NY")

    def initialize(self) -> None:
        """Initialize weather data source."""
        file_path = self.config.get("file_path")
        self.sim_start_time = self.config.get("simulation_start_time")
        if file_path:
            try:
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"No such file: {file_path}")
                self._load_weather_file(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to load weather file {file_path}: {e}")
                self.logger.info("Falling back to no weather data (module will return None).")
                self.weather_data = None
        else:
            self.logger.info("No weather data provided (config['file_path'] missing).")
            self.weather_data = None

        # Initialize current weather
        self.weather = WeatherDisturbance(outdoor_dry_bulb_temp=0.0, outdoor_wet_bulb_temp=0.0, solar_radiation_w_m2=0.0)
        self.logger.info(f"Weather module initialized: {self.name}")

    def step(self, current_step: int) -> Optional[WeatherDisturbance]:
        self.current_timestep = current_step

        if self.weather_data is None or self.weather_data.empty:
            self.logger.error("Weather data not loaded; step() returning None.")
            return None

        if not (0 <= current_step < len(self.weather_data)):
            self.logger.error(
                f"Requested step {current_step} out of range [0, {len(self.weather_data)-1}]."
            )
            return None

        self._get_weather_from_data(current_step)
        #@todo use my previous cnn-lstm-baysien model instead
        self._get_weather_forecast_from_data(current_step)

        return self.weather

    def _load_weather_file(self, file_path: str) -> None:
        """Load weather data from CSV or EPW file into self.weather_data."""
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
            # Ensure required columns exist
            missing = REQUIRED_COLS_CSV - set(df.columns)
            if missing:
                raise ValueError(f"CSV is missing required columns: {missing}")
            self.weather_data = df
            self.logger.info(f"Loaded CSV weather data: {len(self.weather_data)} records.")

        elif file_path.lower().endswith(".epw"):
            # Likely to get an error for EPW file unless it follows a standard format...
            names = [
                "year","month","day","hour","minute","data_source",
                "dry_bulb_temp","dew_point_temp","relative_humidity","atmospheric_pressure",
                "extraterrestrial_horizontal_radiation","extraterrestrial_direct_radiation",
                "horizontal_infrared_radiation","global_horizontal_radiation",
                "direct_normal_radiation","diffuse_horizontal_radiation"
            ] + [f"field_{i}" for i in range(16, 35)]
            try:
                epw = pd.read_csv(file_path, skiprows=8, header=None, names=names)
                epw["hour"] = epw["hour"].clip(1, 24) - 1
                epw["minute"] = epw.get("minute", 0).fillna(0).astype(int)
                epw["datetime"] = pd.to_datetime(
                    epw[["year", "month", "day", "hour", "minute"]],
                    errors="coerce"
                )
                if epw["datetime"].isna().any():
                    n_bad = int(epw["datetime"].isna().sum())
                    self.logger.warning(f"{n_bad} EPW rows had invalid datetimes and will be dropped.")
                    epw = epw.dropna(subset=["datetime"])

                epw = epw.set_index("datetime").sort_index()

                # Map to required columns
                epw["outdoor_temperature"] = epw["dry_bulb_temp"].astype(float)
                epw["solar_radiation"] = epw["global_horizontal_radiation"].astype(float)

                self.weather_data = epw[["outdoor_temperature", "solar_radiation"]]
                self.logger.info(f"Loaded EPW weather data: {len(self.weather_data)} records.")
            except Exception as e:
                raise ValueError(f"Failed to parse EPW file: {e}") from e

        else:
            raise ValueError(f"Unsupported weather file format: {file_path}")

    def _get_weather_forecast_from_data(self, current_step: int):
        self.weather_data['Time'] = pd.to_datetime(self.weather_data['Time'])
        # @ todo hard coding now, need update
        sim_start = pd.Timestamp("2023-08-01 00:00:00")
        sim_data = self.weather_data[self.weather_data['Time'] >= sim_start]
        forecast = sim_data.iloc[current_step:current_step+96] # let's say 96 steps now, all of them need to be parametrized
        temp_forecast = ((forecast["outdoor_temperature"] - 32) * 5 / 9).to_numpy()  # @TODO need to use standard unit, use hard coding for now
        # @TODO need to seperate dry/wet bulb temperature later
        # also need to update the data format process
        solar_forecast = forecast["solar_radiation"].to_numpy()
        self.weather.forecast_outdoor_dry_bulb_temp = temp_forecast
        self.weather.forecast_outdoor_wet_bulb_temp = temp_forecast
        self.weather.forecast_solar_radiation_w_m2 = solar_forecast

    def _get_weather_from_data(self, current_step: int):
        self.weather_data['Time'] = pd.to_datetime(self.weather_data['Time'])
        #@ todo hard coding now, need update
        sim_start = pd.Timestamp("2023-08-01 00:00:00")
        sim_data = self.weather_data[self.weather_data['Time'] >= sim_start]
        row = sim_data.iloc[current_step]
        ot = (float(row["outdoor_temperature"])-32)*5/9  # @TODO need to use standard unit, use hard coding for now
        # @TODO need to seperate dry/wet bulb temperature later
        # also need to update the data format process
        sr = float(row["solar_radiation"])
        self.weather.outdoor_dry_bulb_temp = ot
        self.weather.outdoor_wet_bulb_temp = ot
        self.weather.solar_radiation_w_m2 = sr


    def reset(self) -> None:
        self.current_timestep = 0
        self.current_weather = WeatherDisturbance(outdoor_dry_bulb_temp=0, outdoor_wet_bulb_temp=0, solar_radiation_w_m2=0)
        self.logger.debug(f"Reset weather module: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_timestep": self.current_timestep,
            "outdoor_dry_bulbtemperature": self.current_weather.outdoor_dry_bulb_temp,
            "outdoor_wet_bulb_temperature": self.current_weather.outdoor_wet_bulb_temp,
            "solar_radiation_w_m2": self.current_weather.solar_radiation_w_m2,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.current_timestep = int(state.get("current_timestep", 0))
        self.current_weather.outdoor_dry_bulb_temp = float(state.get("outdoor_dry_bulbtemperature", 0.0))
        self.current_weather.outdoor_wet_bulb_temp = float(state.get("outdoor_wet_bulb_temperature", 0.0))
        self.current_weather.solar_radiation_w_m2 = float(state.get("solar_radiation_w_m2", 0.0))
