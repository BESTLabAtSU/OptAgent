"""
Revised Electrical Supervisory Controller with improved structure and bug fixes
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
import json
from dataclasses import dataclass, field

from ..core.base import BaseModule
from ..core.data_structure import (
    DERSystemState, Disturbance, ClusterObservation, DERSystemAction, DomainState,SystemType,
    PVComponentState, BatteryComponentState, EVComponentState, DERMode, ComponentType
)
from ..controllers.llm_controller import LLMDERController

class SupervisoryController(BaseModule):
    """
    Revised rule-based electrical supervisory controller for DER power flow management.

    Improvements:
    - Handles multiple components of same type properly
    - Clear power flow allocation strategy
    - Better state management
    - Robust error handling
    """

    def __init__(self, config: Dict[str, Any], name: str = "SupervisoryController"):
        """Initialize Electrical Supervisory Controller."""
        super().__init__(config, name)

        # Domain configuration
        self.domain = config.get("domain", "electrical")
        self.control_mode = DERMode(config.get("mode", "self-consumption"))

        # System configuration
        self.system_config = config.get("system_config", {})

        # SOC thresholds
        self.bat_soc_min = config.get("bat_soc_min", 0.1)
        self.bat_soc_max = config.get("bat_soc_max", 0.9)
        self.bat_soc_reserve = config.get("bat_soc_reserve", 0.2)

        self.ev_soc_min = config.get("ev_soc_min", 0.2)
        self.ev_soc_max = config.get("ev_soc_max", 0.8)
        self.ev_v2g_enabled = config.get("ev_v2g_enabled", True)

        # Power limits
        self.max_grid_import = config.get("max_grid_import", 20000)  # Watts
        self.max_grid_export = config.get("max_grid_export", 5000)  # Watts

        # Default charging/discharging rates (will be overridden by component configs)
        self.default_bat_charge_rate = config.get("bat_charge_rate", 2000)
        self.default_bat_discharge_rate = config.get("bat_discharge_rate", 2000)
        self.default_ev_charge_rate = config.get("ev_charge_rate", 3000)
        self.default_ev_discharge_rate = config.get("ev_discharge_rate", 2000)

        # Grid status
        self.grid_connected = True

        # Add LLM controller initialization
        self.use_llm = config.get("use_llm", False)
        if self.use_llm:
            api_key = "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"
            if not api_key:
                raise ValueError("OpenAI API key required for LLM control")

            self.llm_controller = LLMDERController(
                api_key=api_key,
                max_grid_import=self.max_grid_import / 1000,  # Convert to kW
                max_grid_export=self.max_grid_export / 1000,
                bat_soc_min=self.bat_soc_min,
                bat_soc_max=self.bat_soc_max,
                bat_soc_reserve=self.bat_soc_reserve,
                ev_soc_min=self.ev_soc_min,
                ev_soc_max=self.ev_soc_max,
                ev_v2g_enabled=self.ev_v2g_enabled,
                timestep_hours=0.25  # 15-minute timesteps
            )

        # Component detection and configuration
        self._detect_and_configure_components()

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the controller."""
        self.logger.info(f"Initialized electrical controller: {self.name}")
        self.logger.info(f"Components detected - PV: {len(self.pv_configs)}, "
                        f"Batteries: {len(self.battery_configs)}, EVs: {len(self.ev_configs)}")

    def step(self,
             state: DomainState,
             # observation: ClusterObservation,
             observation: Any, #@todo the building and system need to be properly managed in observation follow standard format, just use simplfied format for now
             disturbance: Disturbance,
             timestep: float) -> DERSystemAction:
        """
        Determine electrical control action based on current conditions.

        Handles multiple components properly by aggregating states and
        distributing commands proportionally.
        """
        try:
            system_state, building_state = observation
            building_load = self._get_building_load(building_state)/1000  # kw
            pv_generation = self._get_pv_generation(system_state)/1000  # kw
            DER_state = system_state.components

            # Get control signals
            is_peak = self._is_peak_period(disturbance)
            self.grid_connected = self._get_grid_status(disturbance)
            action = DERSystemAction(
                system_id=system_state.system_id,
                system_type=SystemType.DER.value)

            if self.use_llm:
                # Get forecasts
                forecast_peaksignal = disturbance.prices.forecast_peaksignal
                forecast_outdoor_temp = disturbance.weather.forecast_outdoor_dry_bulb_temp
                forecast_solar = disturbance.weather.forecast_solar_radiation_w_m2

                # Get or generate load forecast
                load_forecast = self._get_load_forecast()

                # Use LLM controller - pass the action object to be modified
                action, reasoning = self.llm_controller.get_control_action(
                    action=action,  # Pass the pre-defined action
                    building_load=building_load,
                    pv_generation=pv_generation,
                    der_state=DER_state,
                    is_peak=is_peak,
                    forecast_peaksignal=forecast_peaksignal,
                    forecast_outdoor_temp=forecast_outdoor_temp,
                    forecast_solar=forecast_solar,
                    load_forecast=load_forecast
                )

                # Log reasoning
                self.logger.info(f"LLM Strategy: {reasoning.get('strategy', 'N/A')}")
                self.logger.debug(f"LLM Reasoning: {json.dumps(reasoning, indent=2)}")

            else:
                # Use existing rule-based control
                action = self.tou_control(
                    building_load, pv_generation, DER_state, is_peak, action
                )

            return action

        except Exception as e:
            self.logger.error(f"Error in electrical controller step: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return DERSystemAction(
            system_id=state['der_system_1'].system_id,
            system_type=SystemType.DER.value
        )

    def _get_load_forecast(self, noise_level: float = 0.05, bias: float = 0.02):
        true_load = np.array([
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
            0.6, 0.6, 0.6, 1.65225412, 2.22834069, 3.599572, 4.27484051,
            2.26060921, 3.599572, 2.30473647, 4.28630795, 3.599572,
            4.3266618, 4.31140015, 3.599572, 4.35855564, 4.34306867, 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.26954275,
            1.2393136, 1.23620264, 0.95178742, 1.24185186, 0.80799366,
            1.24646148, 0., 0., 0., 0., 0., 0., 1.22235219, 1.19179398,
            1.18802125, 0.88245864, 1.18345206, 0., 0., 0., 0., 0., 0.,
            1.1485885, 1.11523261, 1.10815843, 0.80480693, 0., 5.51868879,
            6.21435277, 3.64574435, 5.06863093, 3.50055708, 4.799572,
            3.51071597, 3.48493728, 2.799572, 3.51071597, 3.48493728, 2.4,
            2.4, 2.4, 2.4, 2.4, 2.4, 2.4
        ])
        np.random.seed(42)  # for reproducibility
        noise = np.random.normal(0, noise_level * np.mean(true_load), size=true_load.shape)
        forecast = true_load * (1 + bias) + noise
        forecast = np.clip(forecast, 0, None)  # no negative loads
        return forecast


    def _detect_and_configure_components(self) -> None:
        """Detect and configure all components in the system."""
        # Parse PV configurations
        self.pv_configs = self._parse_component_config('pv_systems', 'pv')
        self.has_pv = len(self.pv_configs) > 0

        # Parse battery configurations
        self.battery_configs = self._parse_component_config('batteries', 'bat')
        self.has_battery = len(self.battery_configs) > 0

        # Parse EV configurations
        self.ev_configs = self._parse_component_config('evs', 'ev')
        self.has_ev = len(self.ev_configs) > 0

    def _parse_component_config(self, plural_key: str, singular_key: str) -> Dict[str, Dict[str, Any]]:
        """Parse component configuration supporting both single and multiple formats."""
        configs = {}

        # Check for multiple components
        if plural_key in self.system_config:
            multi_config = self.system_config[plural_key]
            if isinstance(multi_config, list):
                for idx, comp_config in enumerate(multi_config):
                    comp_id = comp_config.get('id', f"{singular_key}_{idx + 1}")
                    configs[comp_id] = comp_config
            elif isinstance(multi_config, dict):
                configs = multi_config

        # Check for single component (backward compatibility)
        elif singular_key in self.system_config:
            single_config = self.system_config[singular_key]
            comp_id = single_config.get('id', f"{singular_key}_1")
            configs[comp_id] = single_config

        return configs

    def _get_building_load(self, state: Any) -> float:
        building_load = state.components['electrical'].total_load_w
        return building_load  # Default fallback

    def _get_pv_generation(self, state: Any) -> float:
        pv_generation = state.components['pv_1'].generation_w
        return pv_generation


    def _get_all_battery_states(self, state: DERSystemState) -> Dict[str, Dict[str, Any]]:
        """Get states of all batteries."""
        battery_states = {}
        for bat_id, bat_state in state['der_system_1'].components.items():
            if bat_state.component_type == ComponentType.BATTERY:
                bat_config = self.battery_configs.get(bat_id, {})
                battery_states[bat_id] = {
                    'soc': bat_state.soc,
                    'capacity_kwh': bat_config.get('rated_capacity_kWh', 5.0),
                    'max_charge_rate': bat_config.get('max_charge_rate', self.default_bat_charge_rate),
                    'max_discharge_rate': bat_config.get('max_discharge_rate', self.default_bat_discharge_rate),
                    'state': bat_state
                }

        return battery_states

    def _get_all_ev_states(self, der_state: DERSystemState) -> Dict[str, Dict[str, Any]]:
        """Get states of all EVs."""
        ev_states = {}

        if not self.has_ev or not der_state.evs:
            return ev_states

        for ev_id, ev_state in der_state.evs.items():
            ev_config = self.ev_configs.get(ev_id, {})
            ev_states[ev_id] = {
                'soc': ev_state.ev_soc,
                'connected': ev_state.is_connected,
                'capacity_kwh': ev_config.get('rated_capacity_kWh', 40.0),
                'max_charge_rate': ev_config.get('max_charge_rate', self.default_ev_charge_rate),
                'max_discharge_rate': ev_config.get('max_discharge_rate', self.default_ev_discharge_rate),
                'state': ev_state
            }

        return ev_states

    def tou_control(self, building_load, pv_generation, DER_state, is_peak, action):
        """
        DERSystemAction with all power flow decisions (kW everywhere).
        """
        # Organize components by type
        batteries = {k: v for k, v in DER_state.items() if 'bat' in k.lower()}
        evs = {k: v for k, v in DER_state.items() if 'ev' in k.lower()}

        # Sort for consistent priority
        battery_ids = sorted(batteries.keys())
        ev_ids = sorted(evs.keys())

        # Grid constrain
        grid_import_constrain = float(self.max_grid_import) / 1000.0
        grid_export_constrain = float(self.max_grid_export) / 1000.0

        # Helper: capacity with SOC bounds
        def get_capacity(component, comp_type: str):
            """
            Returns (max_charge_kw, max_discharge_kw) permitted this step.

            Considers:
              - C-rate power limits (kW)
              - Energy window to SOC bounds (kWh) mapped to kW for a 1h step

            """
            capacity_kwh = float(getattr(component, "capacity_kwh"))
            soc = float(getattr(component, "soc"))
            charge_c = float(getattr(component, "charge_speed"))  # C-rate
            discharge_c = float(getattr(component, "discharge_speed"))  # C-rate

            # SOC limits (batteries use reserve during peak)
            if comp_type == "bat":
                soc_min = self.bat_soc_min
                soc_max = self.bat_soc_max
            else:  # "ev"
                soc_min = self.ev_soc_min
                soc_max = self.ev_soc_max

            # Clamp
            soc_min = max(0.0, min(1.0, soc_min))
            soc_max = max(0.0, min(1.0, soc_max))

            # kW limits from C-rate
            rate_charge_kw = charge_c * capacity_kwh
            rate_discharge_kw = discharge_c * capacity_kwh

            # Energy windows to bounds (kWh)
            energy_to_max = max(0.0, (soc_max - soc) * capacity_kwh)
            energy_above_min = max(0.0, (soc - soc_min) * capacity_kwh)

            # @todo replace 4 by resolution later
            max_charge_kw = min(rate_charge_kw, energy_to_max/4)
            max_discharge_kw = min(rate_discharge_kw, energy_above_min/4)

            return max_charge_kw, max_discharge_kw

        # Step 1: PV -> Building (self-consumption)
        remaining_pv = max(0.0, pv_generation)  # kW
        remaining_load = max(0.0, building_load)  # kW

        if remaining_pv > 0 and remaining_load > 0:
            pv_to_building = min(remaining_pv, remaining_load)
            action.pv2building = pv_to_building
            remaining_pv -= pv_to_building
            remaining_load -= pv_to_building

        # Step 2: Off-peak vs Peak behavior
        if not is_peak:
            # -------- OFF-PEAK: charge up to SOC max --------
            # Excess PV -> Batteries
            for bat_id in battery_ids:
                if remaining_pv <= 0: break
                bat = batteries[bat_id]
                max_charge_kw, _ = get_capacity(bat, "bat")
                if max_charge_kw > 0:
                    charge_power = min(max_charge_kw, remaining_pv)
                    action.pv2battery[bat_id] = charge_power
                    remaining_pv -= charge_power

            # Excess PV -> EVs (only if connected/active)
            for ev_id in ev_ids:
                if remaining_pv <= 0: break
                ev = evs[ev_id]
                if getattr(ev, "is_connected", True) and getattr(ev, "is_active", True):
                    max_charge_kw, _ = get_capacity(ev, "ev")
                    if max_charge_kw > 0:
                        charge_power = min(max_charge_kw, remaining_pv)
                        action.pv2ev[ev_id] = charge_power
                        remaining_pv -= charge_power

            # Grid -> Batteries (respect import cap and SOC max)
            for bat_id in battery_ids:
                if grid_import_constrain <= 0: break
                bat = batteries[bat_id]
                max_charge_kw, _ = get_capacity(bat, "bat")
                if max_charge_kw > 0:
                    charge_power = min(max_charge_kw, grid_import_constrain)
                    action.grid2battery[bat_id] = charge_power
                    grid_import_constrain -= charge_power

            # Grid -> EVs
            for ev_id in ev_ids:
                if grid_import_constrain <= 0: break
                ev = evs[ev_id]
                if getattr(ev, "is_connected", True) and getattr(ev, "is_active", True):
                    max_charge_kw, _ = get_capacity(ev, "ev")
                    if max_charge_kw > 0:
                        charge_power = min(max_charge_kw, grid_import_constrain)
                        action.grid2ev[ev_id] = charge_power
                        grid_import_constrain -= charge_power

            # Any remaining PV -> Grid (export cap)
            if remaining_pv > 0 and grid_export_constrain > 0:
                to_grid = min(remaining_pv, grid_export_constrain)
                action.pv2grid = to_grid
                remaining_pv -= to_grid
                grid_export_constrain -= to_grid

        else:
            # -------- PEAK: discharge down to reserve/floor --------
            # Batteries -> Building
            for bat_id in battery_ids:
                if remaining_load <= 0: break
                bat = batteries[bat_id]
                _, max_discharge_kw = get_capacity(bat, "bat")
                if max_discharge_kw > 0:
                    discharge_power = min(max_discharge_kw, remaining_load)
                    action.battery2building[bat_id] = discharge_power
                    remaining_load -= discharge_power

            # EVs -> Building (V2G must be enabled + connected/active)
            if self.ev_v2g_enabled:
                for ev_id in ev_ids:
                    if remaining_load <= 0: break
                    ev = evs[ev_id]
                    if getattr(ev, "is_connected", True) and getattr(ev, "is_active", True):
                        _, max_discharge_kw = get_capacity(ev, "ev")
                        if max_discharge_kw > 0:
                            discharge_power = min(max_discharge_kw, remaining_load)
                            action.ev2building[ev_id] = discharge_power
                            remaining_load -= discharge_power

            # During peak, don't charge; excess PV -> grid
            if remaining_pv > 0 and grid_export_constrain > 0:
                to_grid = min(remaining_pv, grid_export_constrain)
                action.pv2grid = to_grid
                remaining_pv -= to_grid
                grid_export_constrain -= to_grid

        # Step 3: Remaining load -> Grid (import cap)
        if remaining_load > 0 and grid_import_constrain > 0:
            grid_take = min(remaining_load, grid_import_constrain)
            action.grid2building = grid_take
            remaining_load -= grid_take
            grid_import_constrain -= grid_take

        return action

    def _is_peak_period(self, disturbance: Disturbance) -> bool:
        """Check if current time is peak period."""
        if hasattr(disturbance, 'prices') and hasattr(disturbance.prices, 'peaksignal'):
            return disturbance.prices.peaksignal
        return False

    def _get_grid_status(self, disturbance: Disturbance) -> bool:
        """Check grid connection status."""
        if hasattr(disturbance, 'grid_connected'):
            return disturbance.grid_connected
        return True  # Default to connected

    def _log_control_decision(self, action: DERSystemAction,
                            building_load: float, pv_generation: float) -> None:
        """Log control decision for debugging."""
        self.logger.debug(f"Control decision - Load: {building_load:.1f}W, "
                         f"PV: {pv_generation:.1f}W")
        self.logger.debug(f"Power flows - PV2Building: {action.pv2building:.1f}W, "
                         f"Battery2Building: {action.battery2building:.1f}W, "
                         f"Grid2Building: {action.grid2building:.1f}W")

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.grid_connected = True
        self.logger.debug(f"Reset electrical controller: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return {
            'control_mode': self.control_mode.value,
            'grid_connected': self.grid_connected,
            'num_batteries': len(self.battery_configs),
            'num_evs': len(self.ev_configs),
            'num_pvs': len(self.pv_configs)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set controller state."""
        if 'control_mode' in state:
            self.control_mode = DERMode(state['control_mode'])
        if 'grid_connected' in state:
            self.grid_connected = state['grid_connected']