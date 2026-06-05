"""
Fan Coil Unit (FCU) system module
"""

from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    HVACSystemState, ComponentType, DomainType,
    FanComponentState, CoilComponentState, PumpComponentState,
    ChillerComponentState, CoolingTowerComponentState,
    Disturbance, HVACSystemAction
)

from bestopt.env.modules.hvac.component.fan import FanModule
from bestopt.env.modules.hvac.component.coil import CoilModule
from bestopt.env.modules.hvac.component.pump import PumpModule
from bestopt.env.modules.hvac.component.chiller import ChillerModule
from bestopt.env.modules.hvac.component.cooling_tower import CoolingTowerModule

from bestopt.env.modules.hvac.local_controller.fan_local_controller import FanLocalController
from bestopt.env.modules.hvac.local_controller.pump_local_controller import PumpLocalController


class FCUModule(BaseModule):
    """
    Fan Coil Unit (FCU) system that includes:
      - 1x Fan (airflow control)
      - 1x Coil (cooling)
      - 1x Pump (CHW)
      - 1x Chiller
      - 1x Cooling Tower
    """

    def __init__(self, config: Dict[str, Any], name: str = "fcu_system"):
        super().__init__(config, name)

        cfg = config.get("system_config", config)

        self.fan      = FanModule(cfg.get("fan", {}), name="supply_fan")
        self.fan_ctrl = FanLocalController(cfg.get("fan_ctrl", {}), name="fan_controller")
        self.coil     = CoilModule(cfg.get("coil", {}), name="cooling_coil")
        self.pump     = PumpModule(cfg.get("pump", {}), name="chilled_water_pump")
        self.pump_ctrl= PumpLocalController(cfg.get("pump_ctrl", {}), name="pump_controller")
        self.chiller  = ChillerModule(cfg.get("chiller", {}), name="chiller")
        self.tower    = CoolingTowerModule(cfg.get("tower", {}), name="cooling_tower")

        # Return air temperature
        self.return_air_temperature = None

    def register_component_state(self, state: HVACSystemState):
        """Register component states in the HVAC system state."""
        # @TODO we are hard coding the system right now, it might be better if we can handle it automatically from config file directly
        # Create fan component state
        fan_state = FanComponentState(
            component_id='fan',
            component_type=ComponentType.FAN,
            system_id=state.system_id
        )
        state.components['fan'] = fan_state

        # Create coil component state
        coil_state = CoilComponentState(
            component_id='coil',
            component_type=ComponentType.COIL,
            system_id=state.system_id
        )
        state.components['coil'] = coil_state

        # Create pump component state
        pump_state = PumpComponentState(
            component_id='pump',
            component_type=ComponentType.PUMP,
            system_id=state.system_id
        )
        state.components['pump'] = pump_state

        # Create chiller component state
        chiller_state = ChillerComponentState(
            component_id='chiller',
            component_type=ComponentType.CHILLER,
            system_id=state.system_id
        )
        state.components['chiller'] = chiller_state

        # Create cooling tower component state
        tower_state = CoolingTowerComponentState(
            component_id='tower',
            component_type=ComponentType.COOLING_TOWER,
            system_id=state.system_id
        )
        state.components['tower'] = tower_state

        self.logger.debug(f"Registered {len(state.components)} components for HVAC system {state.system_id}")

    def initialize(self) -> None:
        """Initialize all submodules."""
        self.fan.initialize()
        self.fan_ctrl.initialize()
        self.coil.initialize()
        self.pump.initialize()
        self.pump_ctrl.initialize()
        self.chiller.initialize()
        self.tower.initialize()
        self._initialized = True
        self.logger.info(f"FCU system initialized: {self.name}")

    def reset(self) -> None:
        """Reset all submodules."""
        self.fan.reset()
        self.fan_ctrl.reset()
        self.coil.reset()
        self.pump.reset()
        self.pump_ctrl.reset()
        self.chiller.reset()
        self.tower.reset()
        self.return_air_temperature = None
        self._initialized = False
        self.initialize()

    def update_return_air_temperature(self, temp: float):
        """Update return air temperature from environment."""
        self.return_air_temperature = temp

    def step(self, state: HVACSystemState, action: HVACSystemAction,
             disturbance: Disturbance, timestep: float) -> Dict[str, Any]:
        """
        Step all submodules in FCU loop.
        """

        # Get component states from system state
        fan_state = state.components.get('fan')
        coil_state = state.components.get('coil')
        pump_state = state.components.get('pump')
        chiller_state = state.components.get('chiller')
        tower_state = state.components.get('tower')

        # Validate components exist
        if not all([fan_state, coil_state, pump_state, chiller_state, tower_state]):
            self.logger.error("Missing component states in HVAC system")
            missing = []
            if not fan_state: missing.append('fan')
            if not coil_state: missing.append('coil')
            if not pump_state: missing.append('pump')
            if not chiller_state: missing.append('chiller')
            if not tower_state: missing.append('tower')
            self.logger.error(f"Missing components: {missing}")
            return self._empty_result()

        # === 1. Fan control (local) ===
        fan_local_cmd = self.fan_ctrl.step(fan_state, action, timestep)

        # === 2. Fan actuation ===
        self.fan.step(fan_state, fan_local_cmd, timestep)

        # === 3. Pump control (local) ===
        pump_local_cmd = self.pump_ctrl.step(coil_state, action, timestep)
        # pump_local_cmd = self.pump_ctrl.step(coil_state, action, fan_local_cmd, timestep)
        
        # === 4. Pump step ===
        self.pump.step(pump_state, pump_local_cmd, timestep)

        # === 5. Coil step (pure thermodynamics) ===
        # Transfer flow rates from fan and pump to coil
        coil_state.airflow_m3s = fan_state.airflow_m3s
        coil_state.waterflow_m3s = pump_state.waterflow_m3s

        # Use zone temperature (should be updated from environment)
        coil_state.air_inlet_temp_c = self.return_air_temperature
        coil_state.water_inlet_temp_c = chiller_state.chws_temp_c
        self.coil.step(coil_state, action, timestep)

        # === 6. Chiller step ===
        self.chiller.step(chiller_state, action, coil_state, pump_state, tower_state, timestep)

        # === 7. Cooling tower step ===
        self.tower.step(tower_state, action, chiller_state, disturbance.weather, timestep)

        # === Update component domain impacts ===
        # Fan impacts
        fan_state.set_domain_impact(DomainType.ELECTRICAL, 'power', fan_state.power_W)
        fan_state.set_domain_impact(DomainType.THERMAL, 'airflow', fan_state.airflow_m3s)

        # Coil impacts
        coil_state.set_domain_impact(DomainType.THERMAL, 'cooling', -coil_state.Q_W)

        # Pump impacts
        pump_state.set_domain_impact(DomainType.ELECTRICAL, 'power', pump_state.power_W)
        pump_state.set_domain_impact(DomainType.WATER, 'flow', pump_state.waterflow_m3s)

        # Chiller impacts
        chiller_state.set_domain_impact(DomainType.ELECTRICAL, 'power', chiller_state.power_W)
        chiller_state.set_domain_impact(DomainType.THERMAL, 'cooling', -chiller_state.cooling_capacity_w)
        chiller_state.set_domain_impact(DomainType.WATER, 'flow', chiller_state.water_flow_m3s)

        # Cooling tower impacts
        #@todo

        # Update system-level domain metrics
        state.update_domain_metrics()

        # === Aggregate outputs ===
        power_total_W = (
            fan_state.power_W +
            pump_state.power_W +
            chiller_state.power_W +
            tower_state.fan_power_W +
            tower_state.pump_power_W
        )

        # Calculate cumulative energy
        energy_total_J = 0
        if hasattr(fan_state, 'energy_j_cum'):
            energy_total_J += fan_state.energy_j_cum
        if hasattr(pump_state, 'energy_J_cum'):
            energy_total_J += pump_state.energy_J_cum
        if hasattr(chiller_state, 'energy_j_cum'):
            energy_total_J += chiller_state.energy_j_cum
        if hasattr(tower_state, 'energy_j_cum'):
            energy_total_J += tower_state.energy_j_cum

        # Store key values for easy access
        self.Q_zone_actual_W = -coil_state.Q_W
        self.SAT_actual_C = coil_state.air_outlet_temp_c
        self.SA_flow_actual_m3s = fan_state.airflow_m3s
        self.CHW_flow_actual_m3s = pump_state.waterflow_m3s
        self.FCU_power_total_W = power_total_W
        self.FCU_energy_cumulative_J = energy_total_J
        self.CHW_supply_temp_C = chiller_state.chws_temp_c

        return {
            "Q_zone_actual_W": self.Q_zone_actual_W,
            "SAT_actual_C": self.SAT_actual_C,
            "SA_flow_actual_m3s": self.SA_flow_actual_m3s,
            "CHW_flow_actual_m3s": self.CHW_flow_actual_m3s,
            "CHW_supply_temp_C":  self.CHW_supply_temp_C,
            "CHW_return_temp_C": chiller_state.chwr_temp_c,
            "FCU_power_total_W": power_total_W,
            "FCU_fan_power_W": fan_state.power_W,
            "FCU_pump_power_W": pump_state.power_W,
            "FCU_chiller_power_W": chiller_state.power_W,
            "FCU_energy_cumulative_J": energy_total_J
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when components are missing."""
        return {
            "Q_zone_actual_W": 0.0,
            "SAT_actual_C": 22.0,
            "SA_flow_actual_m3s": 0.0,
            "CHW_flow_actual_m3s": 0.0,
            "CHW_supply_temp_C": 7.0,
            "CHW_return_temp_C": 12.0,
            "FCU_power_total_W": 0.0,
            "FCU_fan_power_W": 0.0,
            "FCU_pump_power_W": 0.0,
            "FCU_chiller_power_W": 0.0,
            "FCU_energy_cumulative_J": 0.0
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the FCU system."""
        return {
            "return_air_temperature": self.return_air_temperature,
            "Q_zone_actual_W": getattr(self, 'Q_zone_actual_W', 0.0),
            "SAT_actual_C": getattr(self, 'SAT_actual_C', 22.0),
            "power_total_W": getattr(self, 'FCU_power_total_W', 0.0)
        }