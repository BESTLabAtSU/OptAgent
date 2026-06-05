from typing import Dict, Any, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACSystemAction, FanComponentAction, FanComponentState, ComponentType


class FanLocalController(BaseModule):
    """
    Local fan controller.

    Input:
      - action.supply_airflow_setpoint_m3s [m^3/s]

    Output:
      - FanComponentAction.airflow_setpoint_m3s [m^3/s]

    Config:
      - gain (float, default 1.0): attenuation/efficiency factor
      - ctrl_type (str, default "continous"): "continous" or "staged"
          continous : cmd = max(0, gain * upstream)
          staged: snap to stage_levels based on stage_breaks, scaled by rated_flow_m3s
      - rated_flow_m3s (float, default 1.0): rated flow used for staging normalization
      - stage_breaks (List[float], default [0.25, 0.5, 0.75, 1.0])  # (0,1]
      - stage_levels (List[float], default [0.25, 0.5, 0.75, 1.0])  # (0,1]
    """

    def __init__(self, config: Dict[str, Any], name: str = "FanLocalController"):
        super().__init__(config, name)

        self.gain: float = float(config.get("gain", 1.0))

        ctrl = str(config.get("ctrl_type", "continous")).strip().lower()
        if ctrl not in ("continous", "staged"):
            self.logger.warning(f'{self.name}: ctrl_type must be "continous" or "staged"; fallback to "continous".')
            ctrl = "continous"
        self.ctrl_type: str = ctrl

        self.rated_flow_m3s: float = float(config.get("rated_flow_m3s", 1.0))
        self.stage_breaks: list[float] = list(config.get("stage_breaks", [0.25, 0.5, 0.75, 1.0]))
        self.stage_levels: list[float] = list(config.get("stage_levels", [0.25, 0.5, 0.75, 1.0]))

        if self.ctrl_type == "staged":
            if self.rated_flow_m3s <= 0:
                self.logger.warning(f"{self.name}: rated_flow_m3s <= 0, disable staged control.")
                self.ctrl_type = "continous"
            elif not self.stage_breaks or not self.stage_levels or len(self.stage_levels) != len(self.stage_breaks):
                self.logger.warning(f"{self.name}: invalid stage config, fallback to defaults.")
                self.stage_breaks = [0.25, 0.5, 0.75, 1.0]
                self.stage_levels = [0.25, 0.5, 0.75, 1.0]

    def initialize(self) -> None:
        self._initialized = True

    def _staged_cmd(self, want: float) -> float:
        q_norm = 0.0 if self.rated_flow_m3s <= 0 else max(0.0, min(1.0, want / self.rated_flow_m3s))
        idx = 0
        for i, ub in enumerate(self.stage_breaks):
            if q_norm <= ub:
                idx = i
                break
        q_out_norm = self.stage_levels[idx]
        return max(0.0, q_out_norm * want)

    def step(
        self,
        state: FanComponentState,
        action: HVACSystemAction,
        timestep: float,
    ) -> "FanComponentAction":
        """Compute local fan command and return a local-controller action."""

        upstream_flow = action.supply_airflow_setpoint_m3s  
        if upstream_flow is None:
            cmd = None
        else:
            want = max(0.0, self.gain * float(upstream_flow))
            if self.ctrl_type == "staged" and want > 0.0:
                cmd = self._staged_cmd(want)
            else:
                cmd = want

        local_action = FanComponentAction(
            component_id=state.system_id,
            component_type=ComponentType.FAN.value
        )
        local_action.airflow_setpoint_m3s = cmd

        # self._record_state({"upstream": upstream_flow, "cmd": cmd})
        return local_action

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
