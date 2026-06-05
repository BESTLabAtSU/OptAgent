"""
LLM-based Building Control System
Uses OpenAI GPT-4 to optimize HVAC control for energy efficiency while maintaining comfort
"""

import json
from typing import Tuple, List, Dict, Any
from collections import deque
import openai
import logging
import numpy as np


class LLMThermalController:
    """
    LLM-based controller for building HVAC optimization.
    Converts HVAC power (phvac) decisions to supply air flow rate and temperature.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize the LLM controller.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            temperature: LLM temperature parameter for response variability
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

        # Physical constraints for HVAC system
        self.constraints = {
            'supply_air_flow_rate': {
                'min': 0.0,  # m³/s or CFM - adjust based on your units
                'max': 2.0,  # Maximum air flow rate
            },
            'supply_air_temperature': {
                'min': 12.0,  # °C - Minimum supply air temp (cooling)
                'max': 35.0,  # °C - Maximum supply air temp (heating)
            }
        }

        # Conversion factors (adjust based on your system)
        self.cp_air = 1.006  # kJ/kg·K - Specific heat of air
        self.rho_air = 1.2  # kg/m³ - Density of air

    def _extract_recent_history(self, history_info: deque, num_steps: int = 8) -> List[Dict]:
        """
        Extract the most recent history steps for analysis.

        Args:
            history_info: Full history deque
            num_steps: Number of recent steps to extract (default: 8 for 2 hours)

        Returns:
            List of recent history dictionaries
        """
        # Get the last num_steps entries (most recent at the end)
        recent_history = list(history_info)[-num_steps:] if len(history_info) >= num_steps else list(history_info)
        return recent_history

    def _create_pattern_pairs(self, history: List[Dict]) -> str:
        """
        Create state-action-result pairs from history for pattern recognition.

        Args:
            history: Recent history data

        Returns:
            Formatted string describing patterns
        """
        patterns = []
        for i in range(len(history) - 1):
            current = history[i]
            next_state = history[i + 1]

            pattern = {
                'current_state': {
                    'zone_temp': current['temp_room'],
                    'ambient_temp': current['temp_amb'],
                    'solar_gain': current['solar'],
                    'occupancy': current['occ']
                },
                'action_taken': {
                    'phvac': current['phvac']  # Negative means cooling, positive means heating
                },
                'resulting_state': {
                    'next_zone_temp': next_state['temp_room'],
                    'temp_change': next_state['temp_room'] - current['temp_room']
                }
            }
            patterns.append(pattern)

        return json.dumps(patterns, indent=2)

    def _phvac_to_supply_params(self, phvac: float, current_temp: float,
                                setpoint_avg: float) -> Tuple[float, float]:
        """
        Convert HVAC power to supply air flow rate and temperature.
        This is a physics-based approximation that the LLM can override.

        Args:
            phvac: HVAC power (negative for cooling, positive for heating)
            current_temp: Current zone temperature
            setpoint_avg: Average of heating and cooling setpoints

        Returns:
            Tuple of (flow_rate, supply_temp)
        """
        # Base conversion using thermodynamic principles
        # Q = m * cp * ΔT = ρ * V * cp * ΔT

        if abs(phvac) < 100:  # Near zero power - minimal flow
            return 0.1, current_temp

        # Determine if heating or cooling
        is_cooling = phvac < 0
        power_magnitude = abs(phvac) / 1000  # Convert to kW

        if is_cooling:
            # Cooling mode
            supply_temp = max(self.constraints['supply_air_temperature']['min'],
                              current_temp - 10)  # Start with 10°C below room temp
            temp_diff = current_temp - supply_temp
        else:
            # Heating mode
            supply_temp = min(self.constraints['supply_air_temperature']['max'],
                              current_temp + 10)  # Start with 10°C above room temp
            temp_diff = supply_temp - current_temp

        # Calculate required flow rate
        if temp_diff != 0:
            # Q = ρ * V * cp * ΔT
            # V = Q / (ρ * cp * ΔT)
            flow_rate = power_magnitude / (self.rho_air * self.cp_air * abs(temp_diff))
            flow_rate = np.clip(flow_rate,
                                self.constraints['supply_air_flow_rate']['min'],
                                self.constraints['supply_air_flow_rate']['max'])
        else:
            flow_rate = 0.1

        return flow_rate, supply_temp

    def _build_prompt(self, current_temp: float, cooling_setpoint: float,
                      heating_setpoint: float, history_patterns: str,
                      timestep: int) -> str:
        """
        Build the prompt for the LLM.

        Args:
            current_temp: Current zone temperature
            cooling_setpoint: Upper temperature limit
            heating_setpoint: Lower temperature limit
            history_patterns: JSON string of historical patterns
            timestep: Current simulation timestep

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert HVAC controller optimizing for minimal energy consumption while maintaining thermal comfort.

CURRENT STATE:
- Zone Temperature: {current_temp:.2f}°C
- Cooling Setpoint (max): {cooling_setpoint}°C
- Heating Setpoint (min): {heating_setpoint}°C
- Timestep: {timestep}

COMFORT ZONE: Temperature must stay between {heating_setpoint}°C and {cooling_setpoint}°C

HISTORICAL PATTERNS (last 2 hours):
{history_patterns}

CONTROL CONSTRAINTS:
- Supply Air Flow Rate: {self.constraints['supply_air_flow_rate']['min']} to {self.constraints['supply_air_flow_rate']['max']} m³/s
- Supply Air Temperature: {self.constraints['supply_air_temperature']['min']}°C to {self.constraints['supply_air_temperature']['max']}°C

PHYSICAL RELATIONSHIPS:
- Negative phvac values indicate cooling (heat removal)
- Positive phvac values indicate heating (heat addition)
- Higher flow rates increase heat transfer but consume more fan energy
- Larger temperature differences increase heat transfer but may cause discomfort

OPTIMIZATION OBJECTIVES (in priority order):
1. Maintain zone temperature within setpoint range
2. Minimize energy consumption (prefer natural drift when possible)
3. Anticipate disturbances (solar gains, occupancy changes)
4. Avoid frequent switching between heating and cooling

Based on the historical patterns, predict upcoming disturbances and determine optimal control actions.

Provide your response in the following JSON format:
{{
    "supply_air_flow_rate": <float between {self.constraints['supply_air_flow_rate']['min']} and {self.constraints['supply_air_flow_rate']['max']}>,
    "supply_air_temperature": <float between {self.constraints['supply_air_temperature']['min']} and {self.constraints['supply_air_temperature']['max']}>,
    "reasoning": {{
        "current_analysis": "<analysis of current state>",
        "pattern_insights": "<insights from historical patterns>",
        "disturbance_forecast": "<predicted disturbances>",
        "control_strategy": "<chosen strategy and why>",
        "energy_consideration": "<how this minimizes energy>"
    }}
}}"""

        return prompt

    def _parse_llm_response(self, response_text: str) -> Tuple[float, float, Dict]:
        """
        Parse the LLM response and extract control values.

        Args:
            response_text: Raw LLM response

        Returns:
            Tuple of (flow_rate, supply_temp, reasoning)
        """
        try:
            # Clean the response text - remove markdown code blocks if present
            cleaned_text = response_text.strip()

            # Remove ```json and ``` markers if present
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]  # Remove ```

            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]  # Remove trailing ```

            cleaned_text = cleaned_text.strip()

            # Parse JSON response
            response_data = json.loads(cleaned_text)

            # Extract and validate values
            flow_rate = float(response_data['supply_air_flow_rate'])
            supply_temp = float(response_data['supply_air_temperature'])
            reasoning = response_data['reasoning']

            # Apply constraints
            flow_rate = np.clip(flow_rate,
                                self.constraints['supply_air_flow_rate']['min'],
                                self.constraints['supply_air_flow_rate']['max'])
            supply_temp = np.clip(supply_temp,
                                  self.constraints['supply_air_temperature']['min'],
                                  self.constraints['supply_air_temperature']['max'])

            return flow_rate, supply_temp, reasoning

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to safe defaults
            print(f"Error parsing LLM response: {e}")
            return self._get_fallback_control(response_text)

    def _get_fallback_control(self, response_text: str) -> Tuple[float, float, Dict]:
        """
        Provide fallback control values if LLM response parsing fails.

        Args:
            response_text: Failed response text for logging

        Returns:
            Safe default control values
        """
        return (
            0.5,  # Moderate flow rate
            22.0,  # Neutral temperature
            {
                "current_analysis": "Fallback mode activated",
                "pattern_insights": "Unable to parse LLM response",
                "disturbance_forecast": "Unknown",
                "control_strategy": "Using safe defaults",
                "energy_consideration": "Moderate energy consumption",
                "error": response_text[:200]  # Log partial response for debugging
            }
        )

    def get_control_action(self, current_temp: float, history_info: deque,
                           cooling_setpoint: float, heating_setpoint: float,
                           timestep: int) -> Tuple[float, float, Dict]:
        """
        Main method to get HVAC control actions from the LLM.

        Args:
            current_temp: Current zone temperature
            history_info: Historical building state information
            cooling_setpoint: Upper temperature limit
            heating_setpoint: Lower temperature limit
            timestep: Current simulation timestep

        Returns:
            Tuple of (supply_air_flow_rate, supply_air_temperature, reasoning)
        """
        try:
            # Extract recent history
            recent_history = self._extract_recent_history(history_info)

            # Create pattern pairs for LLM
            patterns = self._create_pattern_pairs(recent_history)

            # Build prompt
            prompt = self._build_prompt(
                current_temp=current_temp,
                cooling_setpoint=cooling_setpoint,
                heating_setpoint=heating_setpoint,
                history_patterns=patterns,
                timestep=timestep
            )

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert HVAC control system focused on energy efficiency."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )

            # Parse response
            response_text = response.choices[0].message.content
            flow_rate, supply_temp, reasoning = self._parse_llm_response(response_text)

            # Add metadata to reasoning
            reasoning['timestamp'] = timestep
            reasoning['current_temp'] = current_temp

            return flow_rate, supply_temp, reasoning

        except Exception as e:
            print(f"Error in LLM control: {e}")
            # Return safe defaults
            return self._get_fallback_control(str(e))


class LLMDERController:
    """
    LLM-based DER controller for intelligent demand response.
    Optimizes storage usage based on forecasts to minimize costs.
    """

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o",
                 temperature: float = 0.3,
                 max_grid_import: float = 20.0,  # kW
                 max_grid_export: float = 5.0,  # kW
                 bat_soc_min: float = 0.1,
                 bat_soc_max: float = 0.9,
                 bat_soc_reserve: float = 0.2,
                 ev_soc_min: float = 0.2,
                 ev_soc_max: float = 0.9,
                 ev_v2g_enabled: bool = True,
                 timestep_hours: float = 0.25):  # 15-minute timesteps
        """
        Initialize the LLM DER controller.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            temperature: LLM temperature for response consistency
            max_grid_import: Maximum grid import power (kW)
            max_grid_export: Maximum grid export power (kW)
            bat_soc_min: Minimum battery SOC
            bat_soc_max: Maximum battery SOC
            bat_soc_reserve: Battery SOC reserve for peak (used when peak)
            ev_soc_min: Minimum EV SOC
            ev_soc_max: Maximum EV SOC
            ev_v2g_enabled: Whether V2G is enabled for EVs
            timestep_hours: Duration of each timestep in hours
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

        # Grid constraints
        self.max_grid_import = max_grid_import
        self.max_grid_export = max_grid_export

        # SOC constraints
        self.bat_soc_min = bat_soc_min
        self.bat_soc_max = bat_soc_max
        self.bat_soc_reserve = bat_soc_reserve
        self.ev_soc_min = ev_soc_min
        self.ev_soc_max = ev_soc_max
        self.ev_v2g_enabled = ev_v2g_enabled

        # Timestep duration for energy calculations
        self.timestep_hours = timestep_hours

        self.logger = logging.getLogger(__name__)

    def _get_capacity(self, component: Any, comp_type: str, is_peak: bool = False) -> Tuple[float, float]:
        """
        Calculate max charge/discharge capacity for a component.
        Matches the logic from rule-based controller.

        Args:
            component: Component state object
            comp_type: 'battery' or 'ev'
            is_peak: Whether currently in peak period (affects battery reserve)

        Returns:
            Tuple of (max_charge_kw, max_discharge_kw)
        """
        # Extract component properties
        soc = float(component.soc)

        if comp_type == 'battery':
            capacity_kwh = float(component.capacity_kwh)
            charge_c = float(component.charge_speed)
            discharge_c = float(component.discharge_speed)

            # SOC limits (use reserve during peak for batteries)
            if is_peak:
                soc_min = self.bat_soc_reserve  # Can discharge to reserve during peak
            else:
                soc_min = self.bat_soc_min
            soc_max = self.bat_soc_max

        else:  # 'ev'
            # Handle capacity in Wh or kWh
            if hasattr(component, 'capacity_kwh'):
                capacity_kwh = float(component.capacity_kwh)
            elif hasattr(component, 'capacity_wh'):
                capacity_kwh = float(component.capacity_wh) / 1000 if component.capacity_wh > 0 else 40.0
            else:
                capacity_kwh = 40.0  # Default

            charge_c = float(component.charge_speed)
            discharge_c = float(component.discharge_speed)

            # EV SOC limits
            soc_min = self.ev_soc_min
            soc_max = self.ev_soc_max

        # Clamp SOC limits
        soc_min = max(0.0, min(1.0, soc_min))
        soc_max = max(0.0, min(1.0, soc_max))

        # Power limits from C-rate (kW)
        rate_charge_kw = charge_c * capacity_kwh
        rate_discharge_kw = discharge_c * capacity_kwh

        # Energy windows to bounds (kWh)
        energy_to_max = max(0.0, (soc_max - soc) * capacity_kwh)
        energy_above_min = max(0.0, (soc - soc_min) * capacity_kwh)

        # Convert energy to power considering timestep duration
        # For 15-min timestep (0.25 hours), multiply by 4 to get kW
        max_charge_kw = min(rate_charge_kw, energy_to_max / self.timestep_hours)
        max_discharge_kw = min(rate_discharge_kw, energy_above_min / self.timestep_hours)

        return max_charge_kw, max_discharge_kw

    def _extract_der_capabilities(self, der_state: Dict, is_peak: bool) -> Dict:
        """
        Extract capabilities and constraints from DER state.
        Dynamically handles any component IDs.

        Args:
            der_state: Current DER component states
            is_peak: Current peak status

        Returns:
            Dictionary of capabilities for each component
        """
        capabilities = {
            'batteries': {},
            'evs': {},
            'pvs': {},
            'summary': {
                'total_battery_charge_kw': 0,
                'total_battery_discharge_kw': 0,
                'total_ev_charge_kw': 0,
                'total_ev_discharge_kw': 0,
                'battery_ids': [],
                'ev_ids': [],
                'pv_ids': []
            }
        }

        for component_id, state in der_state.items():
            if 'bat' in component_id.lower():
                # Battery capabilities
                max_charge_kw, max_discharge_kw = self._get_capacity(state, 'battery', is_peak)

                capabilities['batteries'][component_id] = {
                    'soc': float(state.soc),
                    'capacity_kwh': float(state.capacity_kwh),
                    'max_charge_kw': max_charge_kw,
                    'max_discharge_kw': max_discharge_kw,
                    'energy_available_kwh': max_discharge_kw * self.timestep_hours,
                    'energy_capacity_kwh': max_charge_kw * self.timestep_hours
                }

                capabilities['summary']['battery_ids'].append(component_id)
                capabilities['summary']['total_battery_charge_kw'] += max_charge_kw
                capabilities['summary']['total_battery_discharge_kw'] += max_discharge_kw

            elif 'ev' in component_id.lower():
                # Check if EV is connected and active
                is_connected = getattr(state, 'initially_connected', True)
                is_active = getattr(state, 'is_active', True)

                if is_connected and is_active:
                    max_charge_kw, max_discharge_kw = self._get_capacity(state, 'ev', is_peak)

                    # Disable discharge if V2G not enabled
                    if not self.ev_v2g_enabled:
                        max_discharge_kw = 0
                else:
                    max_charge_kw = 0
                    max_discharge_kw = 0

                capabilities['evs'][component_id] = {
                    'soc': float(state.soc),
                    'max_charge_kw': max_charge_kw,
                    'max_discharge_kw': max_discharge_kw,
                    'is_connected': is_connected,
                    'is_active': is_active,
                    'v2g_enabled': self.ev_v2g_enabled and is_connected
                }

                if is_connected and is_active:
                    capabilities['summary']['ev_ids'].append(component_id)
                    capabilities['summary']['total_ev_charge_kw'] += max_charge_kw
                    capabilities['summary']['total_ev_discharge_kw'] += max_discharge_kw

            elif 'pv' in component_id.lower():
                capabilities['pvs'][component_id] = {
                    'current_generation_kw': float(state.generation_w) / 1000
                }
                capabilities['summary']['pv_ids'].append(component_id)

        return capabilities

    def _analyze_forecast_windows(self,
                                  forecast_peaksignal: np.ndarray,
                                  forecast_solar: np.ndarray,
                                  load_forecast: np.ndarray) -> Dict:
        """
        Analyze forecasts to identify key time windows for optimization.

        Args:
            forecast_peaksignal: 96-step peak signal forecast
            forecast_solar: 96-step solar radiation forecast (W/m²)
            load_forecast: 96-step load forecast (kW)

        Returns:
            Analysis of forecast windows
        """
        # Find peak periods
        peak_indices = np.where(forecast_peaksignal)[0]
        peak_start = int(peak_indices[0]) if len(peak_indices) > 0 else -1
        peak_duration = len(peak_indices)

        # Estimate PV generation from solar radiation
        # Rough estimation: 5kW system with 15% efficiency
        pv_capacity_kw = 5.0
        panel_area = 25  # m² for 5kW system
        efficiency = 0.15
        estimated_pv = forecast_solar * panel_area * efficiency / 1000  # kW

        # Calculate energy needs during peak
        if len(peak_indices) > 0:
            peak_load_kwh = np.sum(load_forecast[peak_indices]) * self.timestep_hours
            peak_solar_kwh = np.sum(estimated_pv[peak_indices]) * self.timestep_hours
            net_peak_energy_needed = peak_load_kwh - peak_solar_kwh
        else:
            peak_load_kwh = 0
            peak_solar_kwh = 0
            net_peak_energy_needed = 0

        # Find optimal pre-charging windows (high solar, low load, before peak)
        if peak_start > 0:
            pre_peak_window = slice(max(0, peak_start - 20), peak_start)
            pre_peak_solar_kwh = np.sum(estimated_pv[pre_peak_window]) * self.timestep_hours
            pre_peak_load_kwh = np.sum(load_forecast[pre_peak_window]) * self.timestep_hours
            pre_peak_net_solar = pre_peak_solar_kwh - pre_peak_load_kwh
        else:
            pre_peak_solar_kwh = 0
            pre_peak_load_kwh = 0
            pre_peak_net_solar = 0

        # Calculate next 4 hours (16 timesteps)
        next_4h_solar_kwh = np.sum(estimated_pv[:16]) * self.timestep_hours
        next_4h_load_kwh = np.sum(load_forecast[:16]) * self.timestep_hours

        return {
            'peak_starts_in': peak_start,
            'peak_duration_steps': peak_duration,
            'peak_load_kwh': round(peak_load_kwh, 2),
            'peak_solar_kwh': round(peak_solar_kwh, 2),
            'net_peak_energy_needed': round(net_peak_energy_needed, 2),
            'pre_peak_net_solar_kwh': round(pre_peak_net_solar, 2),
            'next_4h_solar_kwh': round(next_4h_solar_kwh, 2),
            'next_4h_load_kwh': round(next_4h_load_kwh, 2),
            'current_is_peak': bool(forecast_peaksignal[0]),
            'next_4h_has_peak': bool(np.any(forecast_peaksignal[:16])),
            'high_solar_coming_2h': bool(np.max(forecast_solar[:8]) > 500),
            'high_solar_coming_4h': bool(np.max(forecast_solar[:16]) > 500)
        }

    def _build_dynamic_json_template(self, capabilities: Dict) -> str:
        """
        Build a dynamic JSON template based on available components.

        Args:
            capabilities: Component capabilities dictionary

        Returns:
            JSON template string
        """
        battery_dict = {bid: "<float>" for bid in capabilities['summary']['battery_ids']}
        ev_dict = {eid: "<float>" for eid in capabilities['summary']['ev_ids']}

        template = {
            "pv2building": "<float>",
            "pv2battery": battery_dict if battery_dict else {},
            "pv2ev": ev_dict if ev_dict else {},
            "pv2grid": "<float>",
            "battery2building": battery_dict if battery_dict else {},
            "grid2battery": battery_dict if battery_dict else {},
            "ev2building": ev_dict if ev_dict else {},
            "grid2ev": ev_dict if ev_dict else {},
            "grid2building": "<float>",
            "reasoning": {
                "strategy": "<current strategy>",
                "storage_decision": "<why charge/discharge now>",
                "forecast_consideration": "<how forecast affects decision>",
                "cost_optimization": "<how this minimizes cost>"
            }
        }

        return json.dumps(template, indent=2)

    def _build_optimization_prompt(self,
                                   building_load: float,
                                   pv_generation: float,
                                   capabilities: Dict,
                                   forecast_analysis: Dict,
                                   is_peak: bool) -> str:
        """
        Build the optimization prompt for the LLM.

        Args:
            building_load: Current building load (kW)
            pv_generation: Current PV generation (kW)
            capabilities: DER component capabilities
            forecast_analysis: Analysis of forecasts
            is_peak: Current peak status

        Returns:
            Formatted prompt string
        """
        # Create dynamic JSON template
        json_template = self._build_dynamic_json_template(capabilities)

        prompt = f"""You are an expert DER controller optimizing for minimum electricity cost through intelligent storage management.

CURRENT STATE:
- Building Load: {building_load:.2f} kW
- PV Generation: {pv_generation:.2f} kW
- Peak Period: {'YES (expensive)' if is_peak else 'NO (cheap)'}
- Net Load: {(building_load - pv_generation):.2f} kW

STORAGE CAPABILITIES:
Batteries: {json.dumps(capabilities['batteries'], indent=2)}
EVs: {json.dumps(capabilities['evs'], indent=2)}

TOTAL AVAILABLE:
- Battery Charge Capacity: {capabilities['summary']['total_battery_charge_kw']:.2f} kW
- Battery Discharge Capacity: {capabilities['summary']['total_battery_discharge_kw']:.2f} kW
- EV Charge Capacity: {capabilities['summary']['total_ev_charge_kw']:.2f} kW
- EV Discharge Capacity: {capabilities['summary']['total_ev_discharge_kw']:.2f} kW

FORECAST ANALYSIS:
- Peak starts in: {forecast_analysis['peak_starts_in']} steps (15 min each)
- Peak duration: {forecast_analysis['peak_duration_steps']} steps
- Net energy needed during peak: {forecast_analysis['net_peak_energy_needed']} kWh
- Available solar before peak: {forecast_analysis['pre_peak_net_solar_kwh']} kWh
- Next 4h solar generation: {forecast_analysis['next_4h_solar_kwh']} kWh
- Next 4h load: {forecast_analysis['next_4h_load_kwh']} kWh
- High solar in 2h: {forecast_analysis['high_solar_coming_2h']}
- High solar in 4h: {forecast_analysis['high_solar_coming_4h']}

GRID CONSTRAINTS:
- Max Import: {self.max_grid_import} kW
- Max Export: {self.max_grid_export} kW

OPTIMIZATION RULES:
1. The max_charge_kw and max_discharge_kw values ALREADY include SOC constraints - don't reduce them further
2. During OFF-PEAK: 
   - If peak coming soon AND net_peak_energy_needed > current storage: charge aggressively
   - If high solar coming soon: charge conservatively to leave room for PV
   - Use excess PV first, then grid if needed

3. During PEAK:
   - Discharge storage to minimize grid import (most expensive)
   - Prioritize batteries over EVs for reliability
   - Export excess PV if profitable

4. SMART DECISIONS:
   - Calculate if pre-charging is needed: compare net_peak_energy_needed vs available storage
   - Don't overcharge if solar will provide energy during peak
   - Consider round-trip efficiency (~90%) when pre-charging

CRITICAL: 
- All values must be non-negative
- Don't exceed the max_charge_kw or max_discharge_kw limits (they already account for SOC)
- Ensure power balance: sources ≈ sinks
- Only use component IDs that exist in the capabilities

Respond with this exact JSON structure (use actual component IDs):
{json_template}"""

        return prompt

    def _parse_llm_response(self, response_text: str, capabilities: Dict) -> Tuple[Dict, Dict]:
        """
        Parse LLM response into action dictionary.

        Args:
            response_text: Raw LLM response
            capabilities: Component capabilities for validation

        Returns:
            Tuple of (action_dict, reasoning)
        """
        try:
            # Clean response text
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            data = json.loads(cleaned)

            # Extract action components, filtering by available components
            action_dict = {}

            # Single value fields
            for field in ['pv2building', 'pv2grid', 'grid2building']:
                action_dict[field] = max(0, float(data.get(field, 0)))

            # Dictionary fields - only include components that exist
            battery_ids = capabilities['summary']['battery_ids']
            ev_ids = capabilities['summary']['ev_ids']

            # Battery-related flows
            for field in ['pv2battery', 'battery2building', 'grid2battery']:
                action_dict[field] = {}
                if field in data and isinstance(data[field], dict):
                    for bat_id in battery_ids:
                        if bat_id in data[field]:
                            action_dict[field][bat_id] = max(0, float(data[field][bat_id]))

            # EV-related flows
            for field in ['pv2ev', 'ev2building', 'grid2ev']:
                action_dict[field] = {}
                if field in data and isinstance(data[field], dict):
                    for ev_id in ev_ids:
                        if ev_id in data[field]:
                            action_dict[field][ev_id] = max(0, float(data[field][ev_id]))

            reasoning = data.get('reasoning', {})

            return action_dict, reasoning

        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._get_fallback_action_dict(), {"error": str(e)}

    def _get_fallback_action_dict(self) -> Dict:
        """
        Provide fallback action dictionary for error cases.

        Returns:
            Safe default action dictionary
        """
        return {
            'pv2building': 0.0,
            'pv2battery': {},
            'pv2ev': {},
            'pv2grid': 0.0,
            'battery2building': {},
            'grid2battery': {},
            'ev2building': {},
            'grid2ev': {},
            'grid2building': 1.0  # Default minimal grid import
        }

    def _validate_and_apply_action(self,
                                   action: Any,
                                   action_dict: Dict,
                                   building_load: float,
                                   pv_generation: float,
                                   capabilities: Dict) -> Any:
        """
        Validate action dictionary and apply to DERSystemAction object.

        Args:
            action: Pre-defined DERSystemAction object
            action_dict: Action dictionary from LLM
            building_load: Current building load
            pv_generation: Current PV generation
            capabilities: Component capabilities

        Returns:
            Modified action object with validated values
        """
        # First, apply all values from action_dict to action object
        action.pv2building = action_dict.get('pv2building', 0)
        action.pv2grid = action_dict.get('pv2grid', 0)
        action.grid2building = action_dict.get('grid2building', 0)

        # Apply battery flows with validation
        for bat_id, charge_kw in action_dict.get('pv2battery', {}).items():
            if bat_id in capabilities['batteries']:
                max_charge = capabilities['batteries'][bat_id]['max_charge_kw']
                action.pv2battery[bat_id] = min(charge_kw, max_charge)

        for bat_id, charge_kw in action_dict.get('grid2battery', {}).items():
            if bat_id in capabilities['batteries']:
                max_charge = capabilities['batteries'][bat_id]['max_charge_kw']
                action.grid2battery[bat_id] = min(charge_kw, max_charge)

        for bat_id, discharge_kw in action_dict.get('battery2building', {}).items():
            if bat_id in capabilities['batteries']:
                max_discharge = capabilities['batteries'][bat_id]['max_discharge_kw']
                action.battery2building[bat_id] = min(discharge_kw, max_discharge)

        # Apply EV flows with validation
        for ev_id, charge_kw in action_dict.get('pv2ev', {}).items():
            if ev_id in capabilities['evs']:
                max_charge = capabilities['evs'][ev_id]['max_charge_kw']
                action.pv2ev[ev_id] = min(charge_kw, max_charge)

        for ev_id, charge_kw in action_dict.get('grid2ev', {}).items():
            if ev_id in capabilities['evs']:
                max_charge = capabilities['evs'][ev_id]['max_charge_kw']
                action.grid2ev[ev_id] = min(charge_kw, max_charge)

        for ev_id, discharge_kw in action_dict.get('ev2building', {}).items():
            if ev_id in capabilities['evs']:
                max_discharge = capabilities['evs'][ev_id]['max_discharge_kw']
                action.ev2building[ev_id] = min(discharge_kw, max_discharge)

        # Validate PV allocation doesn't exceed generation
        total_pv_out = (action.pv2building + action.pv2grid +
                        sum(action.pv2battery.values()) + sum(action.pv2ev.values()))

        if total_pv_out > pv_generation * 1.01:  # 1% tolerance
            # Scale down PV allocations
            if total_pv_out > 0:
                scale = pv_generation / total_pv_out
                action.pv2building *= scale
                action.pv2grid *= scale
                for k in action.pv2battery:
                    action.pv2battery[k] *= scale
                for k in action.pv2ev:
                    action.pv2ev[k] *= scale

        # Validate grid constraints
        total_grid_import = action.grid2building + sum(action.grid2battery.values()) + sum(action.grid2ev.values())
        if total_grid_import > self.max_grid_import:
            # Scale down grid imports
            scale = self.max_grid_import / total_grid_import if total_grid_import > 0 else 0
            action.grid2building *= scale
            for k in action.grid2battery:
                action.grid2battery[k] *= scale
            for k in action.grid2ev:
                action.grid2ev[k] *= scale

        action.pv2grid = min(action.pv2grid, self.max_grid_export)

        # Ensure building load is met
        total_to_building = (action.pv2building + action.grid2building +
                             sum(action.battery2building.values()) +
                             sum(action.ev2building.values()))

        if total_to_building < building_load * 0.99:  # 1% tolerance
            # Increase grid import to meet load
            deficit = building_load - total_to_building
            action.grid2building += deficit

        return action

    def get_control_action(self,
                           action: Any,  # Pre-defined DERSystemAction
                           building_load: float,
                           pv_generation: float,
                           der_state: Dict,
                           is_peak: bool,
                           forecast_peaksignal: np.ndarray,
                           forecast_outdoor_temp: np.ndarray,
                           forecast_solar: np.ndarray,
                           load_forecast: np.ndarray) -> Tuple[Any, Dict]:
        """
        Main method to get DER control action from LLM.

        Args:
            action: Pre-defined DERSystemAction object to populate
            building_load: Current building load (kW)
            pv_generation: Current PV generation (kW)
            der_state: Current state of all DER components
            is_peak: Current peak period status
            forecast_peaksignal: 96-step peak signal forecast
            forecast_outdoor_temp: 96-step temperature forecast
            forecast_solar: 96-step solar radiation forecast (W/m²)
            load_forecast: 96-step load forecast (kW)

        Returns:
            Tuple of (modified action object, reasoning)
        """
        try:
            # Extract capabilities with current peak status
            capabilities = self._extract_der_capabilities(der_state, is_peak)

            # Analyze forecasts
            forecast_analysis = self._analyze_forecast_windows(
                forecast_peaksignal, forecast_solar, load_forecast
            )

            # Build prompt
            prompt = self._build_optimization_prompt(
                building_load, pv_generation, capabilities,
                forecast_analysis, is_peak
            )

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert DER optimization controller focused on minimizing electricity costs while respecting all constraints."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=800
            )

            # Parse response
            response_text = response.choices[0].message.content
            action_dict, reasoning = self._parse_llm_response(response_text, capabilities)

            # Validate and apply to action object
            action = self._validate_and_apply_action(
                action, action_dict, building_load, pv_generation, capabilities
            )

            # Add metadata to reasoning
            reasoning['capabilities'] = capabilities
            reasoning['forecast_analysis'] = forecast_analysis

            return action, reasoning

        except Exception as e:
            self.logger.error(f"Error in LLM DER control: {e}")
            # Return action with minimal grid import as fallback
            action.grid2building = building_load
            return action, {"error": str(e), "fallback": True}


# Integration helper for your existing code
def create_llm_der_controller(api_key: str, **kwargs) -> LLMDERController:
    """
    Factory function to create LLM DER controller.

    Args:
        api_key: OpenAI API key
        **kwargs: Additional configuration parameters

    Returns:
        Configured LLMDERController instance
    """
    return LLMDERController(api_key=api_key, **kwargs)
