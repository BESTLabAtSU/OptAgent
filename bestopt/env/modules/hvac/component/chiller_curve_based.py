"""
ChillerCurveBased module using EnergyPlus-style three performance curves.

This chiller model adjusts its performance (cooling capacity and power use)
based on:
1. Entering chilled water temp (from coil return)
2. Condenser water temp
3. Part Load Ratio (PLR)

Performance is influenced by:
- A capacity modifier curve (vs temp)
- An EIR (energy input ratio) modifier curve (vs temp)
- An EIR modifier curve (vs PLR)

All performance inputs/outputs are in SI units.
"""

from typing import Dict, Any
import numpy as np

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ChillerState, ThermalAction, PumpState, CoilState


class ChillerCurveBased(BaseModule):
    """
    EnergyPlus-style chiller with 3 performance modifier curves.

    Inputs (from action + state):
        - CoilState.water_outlet_temp_C       : CHW inlet temp [°C]
        - ThermalAction.chws_temp_c_sp        : CHW outlet temp setpoint [°C]
        - PumpState.waterflow_m3s             : CHW flow rate [m³/s]
        - ThermalAction.condenser_temp_c_sp   : Condenser water inlet temp [°C]

    Outputs (in ChillerState):
        - cooling_W                           : Cooling provided [W]
        - cop                                 : Coefficient of performance [-]
        - power_W                             : Chiller power input [W]
        - energy_J_cum                        : Cumulative energy use [J]
        - chws_temp_c                         : CHW supply temperature [°C]
        - chw_flow_m3s                        : Chilled water flow rate [m³/s]
        
    Model:
      - Follows EnergyPlus electric chiller model using 3 empirical modifier curves:
          1. Capacity modifier curve (function of evap temp and cond temp)
          2. EIR modifier curve (function of evap temp and cond temp)
          3. EIR modifier curve (function of part-load ratio, PLR)
      - Actual cooling output:
          Q = RatedCapacity × CapMod × PLR
      - Power consumption:
          Power = (Q / RatedCOP) × EIRModTemp × EIRModPLR
      - COP is computed as: COP = Q / Power
      - PLR is clipped between min_plr and max_plr
      - CHW supply temperature is assumed to match setpoint (no outlet curve model)

    State:
      - Reads from:
          * CoilState.water_outlet_temp_C
          * PumpState.waterflow_m3s
          * ThermalAction.chws_temp_c_sp
          * ThermalAction.condenser_temp_c_sp
      - Writes IN-PLACE to ChillerState:
          * ChillerState.cooling_W
          * ChillerState.cop
          * ChillerState.chws_temp_c
          * ChillerState.chw_flow_m3s
          * ChillerState.power_W
          * ChillerState.energy_J_cum

    Action:
      - Requires ThermalAction instances containing:
          * chws_temp_c_sp
          * condenser_temp_c_sp
      - Typically triggered within an HVAC system manager or building MPC loop

    Notes:
      - All temperatures in °C; internal calculations convert to Kelvin only if needed
      - Modifier curves are passed as Python functions or use built-in defaults
      - No explicit control logic—cooling load is driven by delta-T and flow
    """

    def __init__(self, config: Dict[str, Any], name: str = "chiller_curve"):
        super().__init__(config, name)

        # Rated values
        self.rated_capacity_W = float(config.get("rated_capacity_W", 150_000.0))  # 150 kW
        self.rated_cop = float(config.get("rated_cop", 6.0))  # Typical for water-cooled chiller
        self.min_plr = float(config.get("min_plr", 0.15))
        self.max_plr = float(config.get("max_plr", 1.05))

        # Fluid properties
        self.rho = 1000.0  # Water density [kg/m³]
        self.cp = 4180.0   # Water specific heat [J/kg-K]

        # Optional override for curve functions
        self.cap_mod_temp = config.get("cap_mod_temp", self.default_cap_mod_temp)
        self.eir_mod_temp = config.get("eir_mod_temp", self.default_eir_mod_temp)
        self.eir_mod_plr = config.get("eir_mod_plr", self.default_eir_mod_plr)

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "ChillerState",
        action: "ThermalAction",
        coil_state: "CoilState",
        pump_state: "PumpState",
        timestep: float
    ) -> Dict[str, Any]:

        # === Inputs from components ===
        t_in = coil_state.water_outlet_temp_C            # CHW return temp
        t_out_sp = action.chws_temp_c_sp                 # CHW supply temp (setpoint)
        t_cond = action.condenser_temp_c_sp              # Condenser water temp
        flow_m3s = pump_state.waterflow_m3s              # Chilled water flow rate

        # === Cooling load requested based on flow and delta-T ===
        mass_flow_kg_s = self.rho * flow_m3s
        q_requested_W = max(mass_flow_kg_s * self.cp * (t_in - t_out_sp), 0.0)

        # === Capacity modifier curve (CapModFnTemp) ===
        cap_mod = self.cap_mod_temp(t_out_sp, t_cond)
        cap_available_W = self.rated_capacity_W * cap_mod

        # === Part load ratio ===
        plr = np.clip(q_requested_W / cap_available_W, self.min_plr, self.max_plr)

        # === Actual cooling output ===
        q_actual_W = cap_available_W * plr

        # === EIR modifier curves ===
        eir_mod_temp = self.eir_mod_temp(t_out_sp, t_cond)
        eir_mod_plr = self.eir_mod_plr(plr)

        # === Power consumption and COP ===
        power_W = (q_actual_W / self.rated_cop) * eir_mod_temp * eir_mod_plr
        cop = q_actual_W / power_W if power_W > 0 else 0.0
        energy_J = power_W * timestep

        # === Update state ===
        state.cooling_W = q_actual_W
        state.cop = cop
        state.power_W = power_W
        state.energy_J_cum += energy_J
        state.chws_temp_c = t_out_sp
        state.chw_flow_m3s = flow_m3s

        # === Log internal state ===
        self._record_state({
            "cooling_W": q_actual_W,
            "cop": cop,
            "power_W": power_W,
            "energy_J": energy_J,
            "PLR": plr,
            "cap_mod_temp": cap_mod,
            "eir_mod_temp": eir_mod_temp,
            "eir_mod_plr": eir_mod_plr
        })

        return {}

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()

    # ===== Default EnergyPlus-style curves =====

    def default_cap_mod_temp(self, T_evap_C: float, T_cond_C: float) -> float:
        """Default biquadratic curve for capacity modifier."""
        a, b, c, d, e, f = 0.997, -0.01, 0.0005, 0.002, -0.0001, 0.0
        return a + b*T_evap_C + c*T_evap_C**2 + d*T_cond_C + e*T_cond_C**2 + f*T_evap_C*T_cond_C

    def default_eir_mod_temp(self, T_evap_C: float, T_cond_C: float) -> float:
        """Default biquadratic curve for EIR modifier."""
        a, b, c, d, e, f = 1.02, -0.01, 0.0003, 0.015, -0.0001, 0.0
        return a + b*T_evap_C + c*T_evap_C**2 + d*T_cond_C + e*T_cond_C**2 + f*T_evap_C*T_cond_C

    def default_eir_mod_plr(self, plr: float) -> float:
        """Default quadratic curve for EIR modifier vs PLR."""
        a, b, c = 0.9, 0.2, 0.1
        return a + b*plr + c*plr**2
