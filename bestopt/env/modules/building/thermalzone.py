"""
Building thermal dynamic module.
"""
import numpy as np
import torch
import pickle
import pandas as pd
from datetime import timedelta
from typing import Dict, Any, Optional, Tuple
from collections import deque
import logging
from modnn.Config import _args  # Using version 3.0.7
from modnn.utils import Mod
from ...core.base import BaseModule
from ...core.data_structure import ThermalZoneComponentState, DomainAction, Disturbance, BuildingSystemState
import os

class ThermalDynamicsModule(BaseModule):
    """
    Building thermal dynamic module.
    Features:
    - Neural network-based thermal dynamics prediction
    - Encoder-decoder architecture with historical data warmup
    - Real-time temperature prediction and state updates
    """

    def __init__(self, config: Dict[str, Any], name: str = ""):
        """
        Initialize building thermal dynamic module.

        Args:
            config: building thermal dynamic configuration parameters
                Required keys:
                - model_args: Neural network model configuration
                - scaler_path: Path to data scalers pickle file
                - historical_data_path: Path to historical data CSV file
                Optional keys:
                - encoder_length: Length of encoder sequence (default: 48)
                - resolution: Time resolution in seconds (default: 900)
                - zone_id: Thermal zone identifier (default: "main_zone")
            name: Module name
        """
        super().__init__(config, name)

        # Configuration parameters
        self.encoder_length = config.get("encoder_length", 48)
        self.resolution = config.get("resolution", 900)  # 15 minutes default
        self.zone_id = config.get("zone_id", "main_zone")
        self.scaler_path = config["scaler_path"]  # Required
        self.historical_data_path = config["historical_data_path"]  # Required
        simulation_start_time = config["simulation_start_time"]
        # grab init temperature when register
        df = pd.read_csv(self.historical_data_path, index_col=0)
        sim_data = df.loc[simulation_start_time:]
        self.initial_temp = sim_data["temp_room"].values[0]

        # Model components
        self.mdl = None
        self.dynamic = None
        self.scalers = None
        self.historical_df = None

        # Historical data storage (encoder sequence)
        self.history_buffer = deque(maxlen=self.encoder_length)

        # Simulation state
        self.current_timestep = 0
        self.is_prepared = False

        # Logging
        self.logger = logging.getLogger(f"ThermalDynamics.{name}")



    def initialize(self) -> None:
        try:
            self.logger.info("Initializing thermal dynamics model...")

            # Normalize paths
            for key in ["model_path", "scaler_path", "historical_data_path"]:
                if key in self.config:
                    self.config[key] = os.path.abspath(self.config[key])
                    self.logger.info(f"Resolved {key} → {self.config[key]}")

            args = _args(**self.config.get("model_args"))
            self.mdl = Mod(args=args)
            self.mdl.data_ready()

            model_path = self.config.get("model_path")
            retrain = self.config.get("retrain")

            if retrain == "On":
                self.mdl.train()
                self.mdl.test()
                # self.mdl.check()
                # self.mdl.dynamiccheck()
                # self.mdl.check_show()
            else:
                if model_path:
                    # @TODO I feel we need a clean way to do it?
                    # if we only need to transfer from cuda to cpu, the package can take this args directly
                    abs_model_path = os.path.abspath(model_path)

                    # --- Monkey patch step_mdl’s torch.load to always use abs_model_path ---
                    orig_step_mdl = self.mdl.step_mdl
                    def patched_step_mdl(*args, **kwargs):
                        import torch as _torch
                        orig_torch_load = _torch.load

                        def cpu_load(_, *a, **kw):
                            return orig_torch_load(abs_model_path, map_location=_torch.device("cpu"), *a, **kw)

                        _torch.load = cpu_load
                        try:
                            return orig_step_mdl(*args, **kwargs)
                        finally:
                            _torch.load = orig_torch_load

                    self.mdl.step_mdl = patched_step_mdl
                    self.logger.info(f"Patched step_mdl() to use: {abs_model_path}")

                else:
                    self.mdl.train()
                    self.mdl.test()
                    self.mdl.check()
                    self.mdl.dynamiccheck()
                    self.mdl.check_show()
                    raise ValueError("model_path is required in configuration")
                
            # --- Call it to initialize ---
            self.dynamic = self.mdl.step_mdl()
            # Load data scalers
            self._load_scalers()
            # Load historical data
            self._load_historical_data()

            self.logger.info("Thermal dynamics model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize thermal dynamics model: {e}")
            raise

    def _load_scalers(self) -> None:
        """Load data preprocessing scalers."""
        try:
            with open(self.scaler_path, "rb") as f:
                self.scalers = pickle.load(f)
            self.logger.info(f"Loaded scalers from: {self.scaler_path}")

            # Verify required scalers exist
            required_scalers = ["temp", "solar", "flux", "occ"]
            for scaler_name in required_scalers:
                if scaler_name not in self.scalers:
                    raise ValueError(f"Required scaler '{scaler_name}' not found")

        except Exception as e:
            self.logger.error(f"Failed to load scalers: {e}")
            raise

    def _load_historical_data(self) -> None:
        """Load historical data for warmup."""
        try:
            self.historical_df = pd.read_csv(self.historical_data_path, index_col=0)
            self.historical_df.index = pd.to_datetime(self.historical_df.index)

            # Add time features if not present
            if 'day_sin' not in self.historical_df.columns:
                time_hours = self.historical_df.index.hour + self.historical_df.index.minute / 60
                self.historical_df["day_sin"] = np.sin(2 * np.pi * time_hours / 24) / 2 + 0.5
                self.historical_df["day_cos"] = np.cos(2 * np.pi * time_hours / 24) / 2 + 0.5

            self.logger.info(f"Loaded historical data from: {self.historical_data_path}")
            self.logger.info(f"Historical data range: {self.historical_df.index[0]} to {self.historical_df.index[-1]}")

        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            raise

    def prepare_for_simulation(self, sim_start_time: str) -> None:
        """
        Prepare the module for simulation by warming up with historical data.

        Args:
            sim_start_time: Start time of simulation (e.g., "2024-01-01 00:00:00")
        """
        try:
            sim_start = pd.to_datetime(sim_start_time) - timedelta(minutes=15)
            self.logger.info(f"Preparing for simulation starting at: {sim_start}")

            # Calculate warmup period (encoder_length timesteps before simulation start)
            warmup_duration = pd.Timedelta(seconds=self.encoder_length * self.resolution)
            warmup_start = sim_start - warmup_duration

            self.logger.info(f"Warmup period: {warmup_start} to {sim_start}")

            # Get warmup data from historical dataset
            # @TODO check the index carefully, I feel the current version has one step mismatch
            # Fixed
            warmup_data = self.historical_df.loc[warmup_start:sim_start]

            if len(warmup_data) < self.encoder_length:
                raise ValueError(
                    f"Insufficient historical data for warmup. "
                    f"Need {self.encoder_length} timesteps, got {len(warmup_data)}. "
                    f"Historical data must cover period from {warmup_start} to {sim_start}"
                )

            # Use exactly the last encoder_length timesteps
            warmup_data = warmup_data.iloc[-self.encoder_length:]

            # Fill encoder buffer with historical data
            self.history_buffer.clear()
            for _, row in warmup_data.iterrows():
                timestep_data = {
                    'temp_room': row['temp_room'],
                    'temp_amb': row['temp_amb'],
                    'solar': row['solar'],
                    'phvac': row['phvac'],
                    'occ': row['occ'],
                    'day_sin': row['day_sin'],
                    'day_cos': row['day_cos']
                }
                self.history_buffer.append(timestep_data)

            self.is_prepared = True
            self.current_timestep = 0  # Reset for new simulation

            self.logger.info(f"Successfully warmed up with {len(warmup_data)} historical timesteps")

        except Exception as e:
            self.logger.error(f"Failed to prepare for simulation: {e}")
            raise

    def _compute_time_features(self, timestep: int) -> Tuple[float, float]:
        """Compute cyclic time features from simulation timestep."""
        time_hours = (timestep * self.resolution / 3600) % 24
        day_sin = np.sin(2 * np.pi * time_hours / 24) / 2 + 0.5
        day_cos = np.cos(2 * np.pi * time_hours / 24) / 2 + 0.5
        return day_sin, day_cos

    def _prepare_encoder_sequence(self) -> np.ndarray:
        """Prepare the encoder sequence for model input."""
        sequence_data = []

        for timestep_data in self.history_buffer:
            # Scale the inputs
            scaled_temp_room = self.scalers["temp"].transform([[timestep_data['temp_room']]])[0, 0]
            scaled_temp_amb = self.scalers["temp"].transform([[timestep_data['temp_amb']]])[0, 0]
            scaled_solar = self.scalers["solar"].transform([[timestep_data['solar']]])[0, 0]
            scaled_phvac = self.scalers["flux"].transform([[timestep_data['phvac']]])[0, 0]
            scaled_occ = self.scalers["occ"].transform([[timestep_data['occ']]])[0, 0]

            # Combine features
            features = [
                scaled_temp_room,
                scaled_temp_amb,
                scaled_solar,
                timestep_data['day_sin'],
                timestep_data['day_cos'],
                scaled_occ,
                scaled_phvac
            ]

            sequence_data.append(features)

        return np.array(sequence_data)

    def _update_history_buffer(self, current_data: Dict[str, float]) -> None:
        """Update the history buffer with current timestep data."""
        self.history_buffer.append(current_data.copy())

    def step(self, state: ThermalZoneComponentState, action: Any,
             disturbance: Disturbance, timestep: int) -> Dict[str, Any]:
        """
        Execute one simulation step for thermal dynamics.

        Args:
            state: Current building state
            action: Control actions (thermal domain)
            disturbance: Environmental disturbances
            timestep: Current simulation timestep

        Returns:
            Dictionary containing updated thermal state variables
        """
        if not self.is_prepared:
            raise RuntimeError(
                "Module not prepared for simulation. Call prepare_for_simulation() first."
            )

        try:
            self.current_timestep = timestep

            # Extract inputs from disturbance and action
            outdoor_temp = disturbance.weather.outdoor_dry_bulb_temp
            solar_radiation = disturbance.weather.solar_radiation_w_m2
            occupancy = disturbance.occupancy.occupancy_fraction
            hvac_thermal_load = action.hvac_thermal_load

            current_zone_state = state
            current_temp = current_zone_state.temperature

            # Compute time features
            day_sin, day_cos = self._compute_time_features(timestep)

            # Prepare current timestep data
            current_data = {
                'temp_room': current_temp,
                'temp_amb': outdoor_temp,
                'solar': solar_radiation,
                'phvac': hvac_thermal_load,
                'occ': occupancy,
                'day_sin': day_sin,
                'day_cos': day_cos
            }

            # Update history buffer
            self._update_history_buffer(current_data)

            # Prepare encoder sequence
            encoder_sequence = self._prepare_encoder_sequence()

            # Reshape for model input (batch_size=1, sequence_length, features)
            model_input = encoder_sequence.reshape(1, self.encoder_length, -1)
            model_input = torch.from_numpy(model_input).float()

            # Move to appropriate device if using GPU
            if torch.cuda.is_available():
                try:
                    device = next(self.mdl.model.parameters()).device
                    model_input = model_input.to(device)
                except:
                    # Fallback to CPU if GPU setup fails
                    pass

            # Model prediction
            with torch.no_grad():
                predicted_temp_scaled, _, _ = self.dynamic(model_input)

                # Convert back to CPU if necessary
                if predicted_temp_scaled.is_cuda:
                    predicted_temp_scaled = predicted_temp_scaled.cpu()

                # Inverse transform to get actual temperature
                predicted_temp = self.scalers["temp"].inverse_transform(
                    predicted_temp_scaled.detach().numpy().reshape(-1, 1)
                )[0, 0]

            # Update the zone temperature in state
            current_zone_state.temperature = float(predicted_temp)
            current_zone_state.temperature_buffer = self.history_buffer

            # Return step results
            return {
                'predicted_temperature': predicted_temp,
                'zone_id': self.zone_id,
                'timestep': timestep
            }

        except Exception as e:
            self.logger.error(f"Error in thermal dynamics step: {e}")
            raise

    def reset(self) -> None:
        """Reset the module to initial state."""
        self.current_timestep = 0
        self.is_prepared = False
        self.history_buffer.clear()
        self.logger.info("Thermal dynamics module reset")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current module state."""
        if self.history_buffer and self.is_prepared:
            latest_data = self.history_buffer[-1]
            return {
                'timestep': self.current_timestep,
                'encoder_length': self.encoder_length,
                'buffer_size': len(self.history_buffer),
                'current_step_room_temp': self.initial_temp,
                'last_step_room_temp': latest_data.get('temp_room', 0),
                'last_step_ambient_temp': latest_data.get('temp_amb', 0),
                'last_step_hvac_thermal_load': latest_data.get('phvac', 0),
                'zone_id': self.zone_id,
                'is_prepared': self.is_prepared
            }
        else:
            return {
                'status': 'not_prepared',
                'is_prepared': self.is_prepared,
                'buffer_size': len(self.history_buffer)
            }