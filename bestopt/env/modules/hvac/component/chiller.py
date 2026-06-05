"""
Chiller module (Carnot-based COP with CHW flow and return temp inputs).
"""

from typing import Dict, Any
import numpy as np

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ChillerComponentAction, ChillerComponentState, PumpComponentState, CoilComponentState, CoolingTowerComponentState


class ChillerModule(BaseModule):
    """
    Chiller module with Carnot-based COP, cooling output, and CHW flow rate.

    Input (from CoilState, PumpState, CoolingTowerState, ThermalAction):
        - CoilState.water_outlet_temp_C       : CHW inlet temp [°C]
        - ThermalAction.chws_temp_c_sp        : CHW outlet temp setpoint [°C]
        - PumpState.waterflow_m3s             : CHW flow rate [m³/s]
        - CoolingTowerState.cw_supply_temp_c  : Condenser inlet temp [°C]

    Output (written in-place to ChillerState):
        - cooling_W                           : Cooling output [W]
        - cop                                 : Coefficient of performance [-]
        - chws_temp_c                         : CHW outlet temperature [°C]
        - chw_flow_m3s                        : CHW flow rate [m³/s]
        - power_W                             : Electrical power consumption [W]
        - energy_J_cum                        : Cumulative energy consumption [J]
        
    Model:
      - Carnot-based COP with fixed effectiveness factor:
          COP = η_carnot * (T_cw / (T_cw - T_chw)), using Kelvin temps
      - Cooling output calculated as:
          Q = m_dot * c_p * (T_return - T_supply)
      - Power = Q / COP
      - All temperatures are assumed to be in °C and converted to K internally

    State:
      - Reads from:
          * CoilState.water_outlet_temp_C
          * PumpState.waterflow_m3s
          * ThermalAction.chws_temp_c_sp
          * CoolingTowerState.cw_supply_temp_c
      - Writes IN-PLACE to ChillerState:
          * ChillerState.cooling_W
          * ChillerState.cop
          * ChillerState.chws_temp_c
          * ChillerState.chw_flow_m3s
          * ChillerState.power_W
          * ChillerState.energy_J_cum

    Action:
      - Expects ThermalAction instances with:
          * chws_temp_c_sp

      - Indirectly affects condenser-side heat rejection (used in cooling tower model)
    """

    def __init__(self, config: Dict[str, Any], name: str = "chiller"):
        super().__init__(config, name)
        self.rated_capacity_W = float(config.get("rated_capacity_W", 120_000.0))  # default: 120 kW
        self.rated_cop = float(config.get("rated_cop", 5.5))  # typical COP
        self.eta_carnot = float(config.get("eta_carnot", 0.4))
        self.min_cop = float(config.get("min_cop", 2.0))
        self.max_cop = float(config.get("max_cop", 10.0))
        self.min_chws_temp = float(config.get("min_chws_temp_c", 5.0))
        self.max_chws_temp = float(config.get("max_chws_temp_c", 10.0))
        self.rho = 1000.0  # kg/m³
        self.cp = 4180.0   # J/kg-K

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "ChillerComponentState",
        action: "ChillerComponentAction",
        coil_state: "CoilComponentState",
        pump_state: "PumpComponentState",
        cooling_tower_state: "CoolingTowerComponentState",
        timestep: float
    ) -> Dict[str, Any]:
        t_in = float(getattr(coil_state, "water_outlet_temp_C", 12.0))  # chiller chw inlet temp        t_out_k_lag1 = float(getattr(coil_state, "water_inlet_temp_C", 5.0))  # chiller last-step outlet temp
        t_out_sp = np.clip(float(getattr(action, "chws_temp_setpoint_c", 7.0)),self.min_chws_temp, self.max_chws_temp)
        flow_m3s = float(getattr(pump_state, "waterflow_m3s"))
        # t_cond_sp = float(getattr(action, "condenser_temp_c_sp", 35.0))
        t_cond = float(getattr(cooling_tower_state, "cw_supply_temp_c", 35.0))  

        mass_flow_kg_s = self.rho * flow_m3s
        q_cooling_W = mass_flow_kg_s * self.cp * (t_in - t_out_sp)
        q_cooling_W = min(max(q_cooling_W, 0.0), self.rated_capacity_W*1.2) # assume maximum 120% overload
        
        t_out = t_in - q_cooling_W / (mass_flow_kg_s * self.cp) if mass_flow_kg_s > 0 else t_in # update chw out temp based on actual Q; else if no flow, t_out = t_in
        #coil_state.Q_W = q_cooling_W # update coil state for actual cooling provided by chiller

        # COP Calculation (Carnot)
        T_evap_K = t_out + 273.15
        T_cond_K = t_cond + 273.15
        delta_T = max(T_cond_K - T_evap_K, 0.5)
        cop_carnot = T_evap_K / delta_T
        cop = np.clip(self.eta_carnot * cop_carnot, self.min_cop, self.max_cop)
        cop = min(cop, self.rated_cop*1.2)  # limit COP to 120% of rated

        power_W = q_cooling_W / cop if cop > 0 else 0.0
        energy_J = power_W * timestep

        # Update state
        state.cooling_W = q_cooling_W
        state.cop = cop
        state.chws_temp_c = t_out
        state.chw_flow_m3s = flow_m3s
        state.power_W = power_W
        state.energy_J_cum += energy_J

        self._record_state({
            "cooling_W": q_cooling_W,
            "cop": cop,
            "power_W": power_W,
            "chws_temp_c": t_out,
            "chw_flow_m3s": flow_m3s,
            "energy_J": energy_J
        })

        return {}

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
