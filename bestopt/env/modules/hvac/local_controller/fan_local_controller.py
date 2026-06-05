from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACSystemAction, FanComponentAction, FanComponentState, ComponentType




class FanLocalController(BaseModule):
    """
    Simplified fan local controller with three modes:
      - "vfd"      : continuous output tracking upstream command
      - "staged"   : discrete stages evenly distributed
      - "constant" : ON/OFF based on upstream command presence
    """

    def __init__(self, config: Dict[str, Any], name: str = "FanLocalController"):
        super().__init__(config, name)

        # Basic parameters
        self.gain: float = float(config.get("gain", 0.97))
        self.rated_flow_m3s: float = max(0.0, float(config.get("rated_flow_m3s", 1.0)))
        self.off_threshold_m3s: float = float(config.get("off_threshold_m3s", 1e-4))

        # Controller type
        ctrl = str(config.get("ctrl_type", "vfd")).strip().lower()
        if ctrl not in ("vfd", "staged", "constant"):
            self.logger.warning(
                f'{self.name}: ctrl_type must be "vfd" | "staged" | "constant"; defaulting to "vfd".')
            ctrl = "vfd"
        self.ctrl_type: str = ctrl

        # Staged mode settings
        if self.ctrl_type == "staged":
            num_stages = int(config.get("stages", 4))
            num_stages = max(2, min(num_stages, 10))  # Limit to 2-10 stages
            # Create evenly distributed stages including 0 and 1
            self.stage_levels = [i / (num_stages - 1) for i in range(num_stages)]

        # Constant mode settings
        if self.ctrl_type == "constant":
            self.on_fraction: float = float(config.get("on_fraction", 1.0))
            self.on_fraction = max(0.0, min(1.0, self.on_fraction))
            self.on_cmd_threshold: float = float(config.get("on_cmd_threshold", 0.05))

    def initialize(self) -> None:
        self._initialized = True

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()

    def _compute_vfd_output(self, upstream_cmd: float) -> float:
        """VFD: Continuous output proportional to upstream command"""
        if upstream_cmd is None or upstream_cmd <= 0:
            return 0.0

        # Apply gain and clamp to rated flow
        output = self.gain * upstream_cmd
        output = min(output, self.rated_flow_m3s)

        # Turn off if below threshold
        return 0.0 if output < self.off_threshold_m3s else output

    def _compute_staged_output(self, upstream_cmd: float) -> float:
        """Staged: Select nearest discrete stage"""
        if upstream_cmd is None or upstream_cmd <= 0:
            return 0.0

        # Normalize command to 0-1 range
        normalized = min(upstream_cmd / self.rated_flow_m3s, 1.0) if self.rated_flow_m3s > 0 else 0.0

        # Find closest stage
        closest_stage = min(self.stage_levels, key=lambda x: abs(x - normalized))

        # Convert back to flow rate
        output = closest_stage * self.rated_flow_m3s
        return 0.0 if output < self.off_threshold_m3s else output

    def _compute_constant_output(self, upstream_cmd: float) -> float:
        """Constant: ON/OFF based on upstream command"""
        if upstream_cmd is None or upstream_cmd <= self.on_cmd_threshold:
            return 0.0

        # ON at the configured fraction of rated flow
        return self.on_fraction * self.rated_flow_m3s

    def step(
            self,
            state: FanComponentState,
            action: HVACSystemAction,
            timestep: float,
    ) -> FanComponentAction:
        """Main control step"""
        # Get upstream command
        upstream = getattr(action, "supply_airflow_setpoint_m3s", None)
        upstream = 0.0 if upstream is None else max(0.0, float(upstream))

        # Compute output based on control type
        if self.ctrl_type == "vfd":
            output = self._compute_vfd_output(upstream)
        elif self.ctrl_type == "staged":
            output = self._compute_staged_output(upstream)
        else:  # constant
            output = self._compute_constant_output(upstream)

        # Create action
        fan_action = FanComponentAction(
            component_id=state.system_id,
            component_type=ComponentType.FAN.value
        )
        fan_action.airflow_setpoint_m3s = output

        # Logging
        self.logger.debug(f"[{self.name}] mode={self.ctrl_type} upstream={upstream:.3f} output={output:.3f}")

        return fan_action
