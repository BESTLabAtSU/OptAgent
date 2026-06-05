from typing import Dict, Any
from math import inf

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    HVACSystemAction, HVACSystemState, PumpComponentAction, ComponentType
)

class PumpLocalController(BaseModule):
    """P-only flow control with airflow gating and soft-boot."""

    def __init__(self, config: Dict[str, Any], name: str = "PumpLocalController"):
        super().__init__(config, name)

        # P control + limits
        self.Kp: float = float(config.get("Kp_flow_per_K", 2e-4))            # (m^3/s)/K
        self.flow_min: float = float(config.get("flow_min_m3s", 0.0))
        self.flow_max: float = float(config.get("pump_flowrate_max", inf))
        self.deadband: float = float(config.get("deadband_K", 0.2))          # K
        self.rate_limit: float = float(config.get("rate_limit_m3s_per_s", 5e-5))  # m^3/s/s
        if self.flow_max <= 0:
            raise ValueError("pump_flowrate_max must be positive.")
        if self.Kp < 0:
            raise ValueError("Kp_flow_per_K must be non-negative.")

        # Airflow gating + boot
        self.airflow_off_thr: float = float(config.get("airflow_off_threshold_m3s", 1e-3))
        self.boot_flow: float = float(config.get("boot_flow_m3s", 2e-3))
        self.boot_rate_limit: float = float(config.get("boot_rate_limit_m3s_per_s", 2e-4))

        # Optional cap by air-side capacity
        self.cap_by_air: bool = bool(config.get("cap_by_air_capacity", True))
        self.rho_air: float = float(config.get("rho_air", 1.2))
        self.cp_air: float  = float(config.get("cp_air_JkgK", 1005.0))
        self.cp_w: float    = float(config.get("cp_w_JkgK", 4186.0))

        self.current_flowrate: float | None = None
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True
        self.current_flowrate = None

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def step(
        self,
        state: HVACSystemState,
        action: HVACSystemAction,
        timestep: float
    ) -> PumpComponentAction:
        # inputs
        T_sa_prev = float(getattr(state, "air_outlet_temp_c", getattr(state, "air_outlet_temp_C", 0.0)))
        T_sa_sp   = float(getattr(action, "supply_temp_setpoint_c", getattr(action, "supply_temp_setpoint_C", T_sa_prev)))
        V_air_sp  = float(getattr(action, "supply_airflow_setpoint_m3s", 0.0))

        # airflow gate: airflow off -> water off
        if V_air_sp <= self.airflow_off_thr:
            flow_cmd = 0.0
            self.current_flowrate = flow_cmd
            la = PumpComponentAction(component_id=state.system_id, component_type=ComponentType.PUMP.value)
            la.flow_setpoint_m3s = flow_cmd
            return la

        # previous flow; on rising edge give a boot floor
        flow_prev = self.current_flowrate if self.current_flowrate is not None else self.flow_min
        if flow_prev == 0.0:
            flow_prev = max(self.boot_flow, self.flow_min)

        # P on TSA error
        e = T_sa_prev - T_sa_sp
        delta = 0.0 if abs(e) < self.deadband else self.Kp * e
        flow_target = self._clip(flow_prev + delta, self.flow_min, self.flow_max)

        # optional: cap by air capacity (keep Cw < Cair)
        if self.cap_by_air:
            C_air = (self.rho_air * V_air_sp) * self.cp_air
            flow_cap = 0.98 * C_air / (1000 * self.cp_w)
            flow_target = min(flow_target, flow_cap)

        # rate limit (looser on first on-step)
        rl = self.boot_rate_limit if self.current_flowrate in (None, 0.0) else self.rate_limit
        if rl and timestep and timestep > 0:
            max_step = rl * float(timestep)
            flow_lo  = flow_prev - max_step
            flow_hi  = flow_prev + max_step
            flow_cmd = self._clip(flow_target, max(flow_lo, self.flow_min), min(flow_hi, self.flow_max))
        else:
            flow_cmd = flow_target

        self.current_flowrate = flow_cmd

        la = PumpComponentAction(component_id=state.system_id, component_type=ComponentType.PUMP.value)
        la.flow_setpoint_m3s = flow_cmd
        return la

    def reset(self) -> None:
        self.clear_history()
        self.current_flowrate = None
        self._initialized = True

