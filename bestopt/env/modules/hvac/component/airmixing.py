# Simplified water–air coil (heating/cooling) with effectiveness method.
# Writes outlet temperatures IN-PLACE to CoilState. No action is required.

import numpy as np
from typing import Dict, Any, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import CoilState, Disturbance, Disturbance



class CoilModule(BaseModule):
    """
    Standalone coil module.

    Inputs (read from CoilState):
      - airflow_m3s          [m^3/s]
      - waterflow_m3s        [m^3/s]
      - air_inlet_temp_C     [°C]
      - water_inlet_temp_C   [°C]

    Outputs (written in-place to CoilState):
      - air_outlet_temp_C    [°C]
      - water_outlet_temp_C  [°C]

    Model
    - Simplified effectiveness method (placeholder for NTU–ε):
        Q = ε * C_min * (T_hot,in − T_cold,in),  with  C = m_dot * c_p
    - Properties are treated as constants (configurable):
        ρ_air, c_p,air,  ρ_water, c_p,water
    - Limitations: no phase change/condensation, no bypass, ε is constant.

    State
    - Expects a CoilState instance passed as `state`.
    - Reads (and does NOT modify):
        * state.airflow_m3s         [m^3/s]
        * state.waterflow_m3s       [m^3/s]
        * state.air_inlet_temp_C    [°C]
        * state.water_inlet_temp_C  [°C]
    - Writes IN-PLACE:
        * state.air_outlet_temp_C   [°C]
        * state.water_outlet_temp_C [°C]

    Action
    - None required for now (pass `None` is fine). Future versions may consume valve positions or setpoints.
    """

    def __init__(self, config: Dict[str, Any], name: str = "coil"):
        super().__init__(config, name)

        # --- Effectiveness and properties (defaults OK for quick testing) ---
        self.effectiveness: float = float(config.get("effectiveness", 0.7))
        self.effectiveness = max(0.0, min(1.0, self.effectiveness))

        # Air properties (approx. near 20–25°C)
        self.rho_air: float = float(config.get("rho_air", 1.2))         # kg/m^3
        self.cp_air_JkgK = float(config.get("cp_air_JkgK", 1005.0))  # J/(kg·K)

        # Water properties (approx. liquid water near room temp)
        self.rho_water: float = float(config.get("rho_water", 997.0))   # kg/m^3
        self.cp_water_JkgK = float(config.get("cp_water_JkgK", 4186.0))  # J/(kg·K)

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "CoilState",
        action: Any,                      # not used
        disturbance: "Disturbance",
        timestep: float
    ) -> Dict[str, Any]:
        """Compute outlet temps from inlet temps and flows; write in-place to state. (pure SI)"""
        # 0) Inputs (do NOT modify)
        Va = float(getattr(state, "airflow_m3s", 0.0))          # [m^3/s]
        Vw = float(getattr(state, "waterflow_m3s", 0.0))        # [m^3/s]
        Ta_in = float(getattr(state, "air_inlet_temp_C", 0.0))  # [°C]
        Tw_in = float(getattr(state, "water_inlet_temp_C", 0.0))# [°C]

        # 1) Capacity rates: C = m_dot * cp  →  [W/K]
        mdot_air = max(0.0, Va) * self.rho_air                  # [kg/s]
        mdot_wat = max(0.0, Vw) * self.rho_water               # [kg/s]
        C_air = mdot_air * self.cp_air_JkgK                    # [W/K]
        C_wat = mdot_wat * self.cp_water_JkgK                  # [W/K]

        # No flow or no driving ΔT → no heat transfer
        if C_air <= 0.0 or C_wat <= 0.0 or Ta_in == Tw_in:
            state.air_outlet_temp_C = Ta_in
            state.water_outlet_temp_C = Tw_in
            self._record_state({"Ta_out": Ta_in, "Tw_out": Tw_in, "Q_W": 0.0})
            return {}

        # 2) Identify hot/cold side by inlet temps
        if Ta_in >= Tw_in:
            Th_in, Tc_in = Ta_in, Tw_in
            C_hot, C_cold = C_air, C_wat
            hot_side = "air"
        else:
            Th_in, Tc_in = Tw_in, Ta_in
            C_hot, C_cold = C_wat, C_air
            hot_side = "water"

        C_min = C_hot if C_hot < C_cold else C_cold            # [W/K]
        dT_in = Th_in - Tc_in                                   # [K] (°C diff)

        # 3) Effectiveness method: q in [W]
        eps = self.effectiveness
        Q_W = eps * C_min * dT_in                               # [W], hot → cold

        # 4) Outlet temps by energy balance
        Th_out = Th_in - Q_W / C_hot
        Tc_out = Tc_in + Q_W / C_cold

        # 5) Map back to air/water
        if hot_side == "air":
            state.air_outlet_temp_C = Th_out
            state.water_outlet_temp_C = Tc_out
        else:
            state.water_outlet_temp_C = Th_out
            state.air_outlet_temp_C = Tc_out

        state.Q_W = Q_W

        self._record_state({
            "Ta_in": Ta_in, "Tw_in": Tw_in,
            "Ta_out": state.air_outlet_temp_C, "Tw_out": state.water_outlet_temp_C,
            "Q_W": Q_W, "eps": eps
        })
        return {}


    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
