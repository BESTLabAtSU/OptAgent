"""
Heat Pump module (two-port, source & load).
"""

from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACSystemAction, HVACMode, HeatPumpComponentAction, HeatPumpComponentState, Disturbance


class HeatPumpModule(BaseModule):
    def __init__(self, config: Dict[str, Any], name: str = "heat_pump"):
        super().__init__(config, name)
        # Separate rated capacities for heating and cooling 
        self.Q_ref_heating = float(config.get("rated_capacity_heating_W", 0.0))
        self.Q_ref_cooling = float(config.get("rated_capacity_cooling_W", 0.0))

        # Separate COPs for heating and cooling 
        self.cop_ref_heating = float(config.get("rated_heating_cop", 1.0))
        self.cop_ref_cooling = float(config.get("rated_cooling_cop", 1.0))
        
        self.load_flow_ref = float(config.get("rated_load_flow_m3s", 1.0))
        self.source_flow_ref = float(config.get("rated_source_flow_m3s", 1.0))
        
        self.plr_min = float(config.get("min_part_load_ratio", 0.0))
        self.plr_max = float(config.get("max_part_load_ratio", 1.0))
        
        # ensure numeric types for physical properties
        self.cp_load = float(config.get("cp_load", 4180.0))                 # load side J/kg-K (default water)
        self.cp_source = float(config.get("cp_source", 1000.0))             # source side J/kg-K (default air)
        self.rho_load = float(config.get("rho_load", 1000.0))               # load side  kg/m³ (default water)
        self.rho_source = float(config.get("rho_source", 1.2))              # source side kg/m³ (default air)
        
        # Polynomial coefficients for chiller-like performance curves (defaults are neutral)
        # CAPFT / EIRFT are biquadratic forms in (T_cw_l, T_cond_e):
        #   a + b*T_cw_l + c*T_cw_l^2 + d*T_cond_e + e*T_cond_e^2 + f*(T_cw_l*T_cond_e)
        # EIRFPLR is quadratic in PLR: a + b*PLR + c*PLR^2
        
        # Performance curves: always require separate sets for heating and cooling
        default_biquad = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        default_quad = [1.0, 0.0, 0.0]

        # Capacity/Performance factors (biquadratic) - explicitly separate heating & cooling
        self.CAPFT_coeffs_heating = config.get("CAPFT_coeffs_heating", default_biquad)
        self.CAPFT_coeffs_cooling = config.get("CAPFT_coeffs_cooling", default_biquad)

        # EIR temperature factors (biquadratic) - explicitly separate heating & cooling
        self.EIRFT_coeffs_heating = config.get("EIRFT_coeffs_heating", default_biquad)
        self.EIRFT_coeffs_cooling = config.get("EIRFT_coeffs_cooling", default_biquad)

        # EIR part-load factors (quadratic) - explicitly separate heating & cooling
        self.EIRFPLR_coeffs_heating = config.get("EIRFPLR_coeffs_heating", default_quad)
        self.EIRFPLR_coeffs_cooling = config.get("EIRFPLR_coeffs_cooling", default_quad)

    def step(
        self,
        state: HeatPumpComponentState,
        hp_action: HeatPumpComponentAction,
        sys_action: HVACSystemAction,
        # timestep: float
    ) -> HeatPumpComponentState:
        # hp_action: heat-pump specific action
        # sys_action: higher-level HVAC system action 
        
        m_load = state.load_flow_m3s * self.rho_load
        m_source = state.source_flow_m3s * self.rho_source

        sys_mode = sys_action.mode # use system-level action to determine the heat pump operation mode 
        # normalize mode to support Enum members or plain strings
        if hasattr(sys_mode, "value"):
            sys_mode_val = sys_mode.value
        else:
            sys_mode_val = str(sys_mode)

        # @TODO: 251023: check if sys_action can update as expected

        if sys_mode_val.casefold() == HVACMode.OFF.value.casefold() or m_load <= 1e-4*self.load_flow_ref*self.rho_load:
            # No flow or system off: no operation            
            state.mode = HVACMode.OFF
        else:
            # Determine mode based on temperature setpoint
            if sys_mode_val.casefold() == HVACMode.HEATING.value.casefold():
                load_outlet_temp_setpoint_C = hp_action.hw_temp_setpoint_c
                if load_outlet_temp_setpoint_C > state.load_inlet_temp_C + 0.5:
                    state.mode = HVACMode.HEATING
                    Q_ref = abs(self.Q_ref_heating)
                    cop_ref = self.cop_ref_heating
                    P_ref = Q_ref / cop_ref
                    CAPFT_coeffs = self.CAPFT_coeffs_heating
                    EIRFT_coeffs = self.EIRFT_coeffs_heating
                    EIRFPLR_coeffs = self.EIRFPLR_coeffs_heating
                else:
                    state.mode = HVACMode.OFF                    
            elif sys_mode_val.casefold() == HVACMode.COOLING.value.casefold():
                load_outlet_temp_setpoint_C = hp_action.chw_temp_setpoint_c
                if load_outlet_temp_setpoint_C < state.load_inlet_temp_C - 0.5:
                    state.mode = HVACMode.COOLING
                    Q_ref = -abs(self.Q_ref_cooling)
                    cop_ref = self.cop_ref_cooling
                    P_ref = -Q_ref / cop_ref
                    CAPFT_coeffs = self.CAPFT_coeffs_cooling
                    EIRFT_coeffs = self.EIRFT_coeffs_cooling
                    EIRFPLR_coeffs = self.EIRFPLR_coeffs_cooling
                else:
                    state.mode = HVACMode.OFF
            else:
                # Unknown system mode -> raise to expose the configuration/usage problem
                raise ValueError(f"Unknown HVAC system mode: {sys_mode}")
        
        if state.mode == HVACMode.OFF:
            state.source_outlet_temp_C = state.source_inlet_temp_C
            state.load_outlet_temp_C = state.load_inlet_temp_C
            state.thermal_output_W = 0.0
            state.power_W = 0.0
            state.cop = 0.0
            return state
        else:
            # Calculate required thermal output
            Q_load_req = m_load * self.cp_load * (load_outlet_temp_setpoint_C - state.load_inlet_temp_C) # heating (+) or cooling (-)

            CAPFT_val = self._evaluate_polynomial(CAPFT_coeffs, load_outlet_temp_setpoint_C, state.source_inlet_temp_C)
            Q_avail = Q_ref * CAPFT_val  # available thermal capacity at current conditions # heating (+) or cooling (-)
            
            # Calculate part-load ratio
            if abs(Q_avail) < abs(1e-4*Q_ref):
                plr = 0.0
            else:
                plr = max(self.plr_min, min(Q_load_req / Q_avail, self.plr_max)) 
            
            # Calculate delivered thermal output
            Q_load = plr*Q_avail # heating (+) or cooling (-)   
            
            EIRFT_val = self._evaluate_polynomial(EIRFT_coeffs, load_outlet_temp_setpoint_C, state.source_inlet_temp_C)
            EIRFPLR_val = self._evaluate_polynomial(EIRFPLR_coeffs, plr)
            
            eir = (1.0/cop_ref) * EIRFT_val * EIRFPLR_val  # Energy Input Ratio
            power = P_ref*CAPFT_val*EIRFT_val*EIRFPLR_val # power input in W
            
            if state.mode == HVACMode.COOLING:
                Q_source = power - Q_load  # heat rejected to source side
            elif state.mode == HVACMode.HEATING:
                Q_source = - (Q_load - power)  # heat extracted from source side
            else:
                raise ValueError(f"Unknown heat pump mode: {state.mode}")
            
            state.load_outlet_temp_C = state.load_inlet_temp_C + Q_load / (m_load * self.cp_load)
            # guard against zero source mass flow to avoid division by zero
            if m_source <= 0:
                state.source_outlet_temp_C = state.source_inlet_temp_C
            else:
                state.source_outlet_temp_C = state.source_inlet_temp_C + Q_source / (m_source * self.cp_source)
            
            state.thermal_output_W = Q_load
            state.power_W = power            
            state.cop = abs(Q_load) / power if power > 1e-4*P_ref else 0.0
            
            return state
    
    def reset(self, state: HeatPumpComponentState):
        """
        Reset dynamic/cumulative variables.
        """
        state.mode = HVACMode.OFF
        state.source_outlet_temp_C = 0.0
        state.load_outlet_temp_C = 0.0
        state.thermal_output_W = 0.0
        state.power_W = 0.0
        state.cop = 0.0
        
    def initialize(self, state: HeatPumpComponentState):
        """
        Initialize state (required by BaseModule).
        """
        self.reset(state)

    # --- Chiller-like performance curve helpers ---
    def _evaluate_polynomial(self, coeffs, x1, x2=None):
        """Evaluate biquadratic or quadratic polynomials.

        If coeffs length == 6, treat as biquadratic:
            a + b*x1 + c*x1^2 + d*x2 + e*x2^2 + f*(x1*x2)
        If coeffs length == 3, treat as quadratic in single variable x1:
            a + b*x1 + c*x1^2
        """
        if coeffs is None:
            return 1.0
        if len(coeffs) == 6 and x2 is not None:
            a, b, c, d, e, f = coeffs
            return a + b * x1 + c * (x1 ** 2) + d * x2 + e * (x2 ** 2) + f * (x1 * x2)
        elif len(coeffs) == 3:
            a, b, c = coeffs
            return a + b * x1 + c * (x1 ** 2)
        else:
            # fallback: if first coeff is 1 assume neutral
            return coeffs[0] if len(coeffs) > 0 else 1.0