"""
DER System Module - Orchestrates PV, Battery, and EV component modules
"""
from typing import Dict, Any, Tuple, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    DERSystemAction, Disturbance, DERSystemState,
    PVComponentState, BatteryComponentState, EVComponentState,
    ComponentType, ComponentState
)

from bestopt.env.modules.ders.component.pv import PVModule
from bestopt.env.modules.ders.component.battery import BatteryModule
from bestopt.env.modules.ders.component.ev import EVModule


class ComponentRegistry:
    """Registry that manages component-state pairs."""

    @staticmethod
    def create_component(component_type: ComponentType,
                         component_id: str,
                         config: Dict[str, Any],
                         system_id: str) -> Tuple[BaseModule, ComponentState]:
        """Create both module and state together."""

        if component_type == ComponentType.PV:
            module = PVModule(config=config, name=component_id)
            state = PVComponentState(
                component_id=component_id,
                component_type=component_type,
                system_id=system_id
            )
            return module, state
        elif component_type == ComponentType.BATTERY:
            module = BatteryModule(config=config, name=component_id)
            state = BatteryComponentState(
                component_id=component_id,
                component_type=component_type,
                system_id=system_id
            )
            return module, state
        elif component_type == ComponentType.EV:
            module = EVModule(config=config, name=component_id)
            state = EVComponentState(
                component_id=component_id,
                component_type=component_type,
                system_id=system_id
            )
            return module, state
        else:
            raise ValueError(f"Unsupported component type: {component_type}")


class DERModule(BaseModule):
    """
    DER System using ComponentRegistry for module-state management.
    """

    def __init__(self, config: Dict[str, Any], name: str = "der_system"):
        super().__init__(config, name)
        self.system_config = config.get("system_config", {})

        # Parse configurations
        self.pv_configs = self._parse_component_config('pvs', 'pv')
        self.battery_configs = self._parse_component_config('batteries', 'bat')
        self.ev_configs = self._parse_component_config('evs', 'ev')

        # Module containers
        self.pv_modules: Dict[str, Any] = {}
        self.battery_modules: Dict[str, Any] = {}
        self.ev_modules: Dict[str, Any] = {}

        # State containers
        self.pv_states: Dict[str, Any] = {}
        self.battery_states: Dict[str, Any] = {}
        self.ev_states: Dict[str, Any] = {}

        # Create registry instance
        self.registry = ComponentRegistry()

        self.logger.info(f"DER system configured with: "
                         f"{len(self.pv_configs)} PV configs, "
                         f"{len(self.battery_configs)} battery configs, "
                         f"{len(self.ev_configs)} EV configs")

    def register_component_state(self, state: DERSystemState):
        # Create PV components
        for pv_id, pv_config in self.pv_configs.items():
            module, component_state = self.registry.create_component(
                component_type=ComponentType.PV,
                component_id=pv_id,
                config=pv_config,
                system_id=state.system_id
            )

            # Store module and state
            self.pv_modules[pv_id] = module
            self.pv_states[pv_id] = component_state
            state.components[pv_id] = component_state

            self.logger.debug(f"Registered PV component: {pv_id}")

        # Create Battery components
        for bat_id, bat_config in self.battery_configs.items():
            module, component_state = self.registry.create_component(
                component_type=ComponentType.BATTERY,
                component_id=bat_id,
                config=bat_config,
                system_id=state.system_id,
            )
            component_state.soc = self.battery_configs[bat_id]['initial_soc']
            component_state.capacity_kwh = self.battery_configs[bat_id]['rated_capacity_kWh']
            component_state.charge_speed = self.battery_configs[bat_id]['charge_speed']
            component_state.discharge_speed = self.battery_configs[bat_id]['discharge_speed']
            component_state.charge_efficiency = self.battery_configs[bat_id]['charge_efficiency']

            self.battery_modules[bat_id] = module
            self.battery_states[bat_id] = component_state
            state.components[bat_id] = component_state

            self.logger.debug(f"Registered Battery component: {bat_id}")

        # Create EV components
        for ev_id, ev_config in self.ev_configs.items():
            module, component_state = self.registry.create_component(
                component_type=ComponentType.EV,
                component_id=ev_id,
                config=ev_config,
                system_id=state.system_id,
            )
            component_state.soc = self.ev_configs[ev_id]['initial_soc']
            component_state.capacity_kwh = self.ev_configs[ev_id]['rated_capacity_kWh']
            component_state.charge_speed = self.ev_configs[ev_id]['charge_speed']
            component_state.discharge_speed = self.ev_configs[ev_id]['discharge_speed']
            component_state.charge_efficiency = self.ev_configs[ev_id]['charge_efficiency']
            self.ev_modules[ev_id] = module
            self.ev_states[ev_id] = component_state
            state.components[ev_id] = component_state

            self.logger.debug(f"Registered EV component: {ev_id}")

        self.logger.info(f"Registered {len(state.components)} components for DER system {state.system_id}")

    def initialize(self) -> None:
        """Initialize all component modules that have been created."""

        # Initialize PV modules
        for pv_id, pv_module in self.pv_modules.items():
            pv_module.initialize()
            self.logger.debug(f"Initialized PV module: {pv_id}")

        # Initialize battery modules
        for bat_id, bat_module in self.battery_modules.items():
            bat_module.initialize()
            self.logger.debug(f"Initialized battery module: {bat_id}")

        # Initialize EV modules
        for ev_id, ev_module in self.ev_modules.items():
            ev_module.initialize()
            self.logger.debug(f"Initialized EV module: {ev_id}")

        self._initialized = True
        self.logger.info(f"DER system fully initialized: {self.name}")

    def calculate_pv_generation(self, state: DERSystemState, disturbance, timestep):
        """
        Calculate PV generation for all PV components.

        Args:
            state: The DER system state containing component states
            disturbance: Current disturbances (weather, etc.)
            timestep: Current simulation timestep

        Returns:
            Total PV generation in Watts
        """
        total_generation = 0.0

        for pv_id, pv_module in self.pv_modules.items():
            # Get the PV state from the system state, not from local storage
            pv_state = state.components.get(pv_id)

            if pv_state and pv_state.component_type == ComponentType.PV:
                pv_state = pv_module.step(
                    state=pv_state,  # Pass the actual state from the system
                    disturbance=disturbance,
                    resolution=900,
                    timestep=timestep,
                )

                total_generation += pv_state.generation_w
                self.logger.debug(f"PV {pv_id}: {pv_state.generation_w:.1f}W")

        return total_generation

    def step(self, state: DERSystemState, action: DERSystemAction, disturbance: Disturbance,
             resolution: int, timestep: float) -> Dict[str, Any]:

        results = {
            'pv_results': {},
            'battery_results': {},
            'ev_results': {},
            'total_grid_import': 0.0,
            'total_grid_export': 0.0,
            'total_building_supply': 0.0,
            'power_flows': {}
        }

        for bat_id, bat_module in self.battery_modules.items():
            bat_state = state.components.get(bat_id)
            if bat_state and bat_state.component_type == ComponentType.BATTERY:
                # Calculate net power for this battery from action
                power_in = action.pv2battery.get(bat_id, 0.0) + action.grid2battery.get(bat_id, 0.0)
                power_out = action.battery2building.get(bat_id, 0.0)
                net_power_kw = power_in - power_out

                # Let battery module handle its own dynamics
                bat_state = bat_module.step(
                    state=bat_state,
                    action={'net_power_kw': net_power_kw},  # Pass simplified action
                    disturbance=disturbance,
                    timestep=timestep
                )
                # print(bat_state.soc)
                # results['battery_results'][bat_id] = battery_results

        # Step 3: Process EV modules
        for ev_id, ev_module in self.ev_modules.items():
            ev_state = state.components.get(ev_id)
            if ev_state and ev_state.component_type == ComponentType.EV:
                # Calculate net power for this EV from action
                power_in = action.pv2ev.get(ev_id, 0.0) + action.grid2ev.get(ev_id, 0.0)
                power_out = action.ev2building.get(ev_id, 0.0)
                net_power_kw = power_in - power_out

                # Let EV module handle its own dynamics
                ev_state = ev_module.step(
                    state=ev_state,
                    action={'net_power_kw': net_power_kw},  # Pass simplified action
                    disturbance=disturbance,
                    timestep=timestep
                )


        return results

    def reset(self) -> None:
        """Reset all component modules."""
        for module in self.pv_modules.values():
            module.reset()
        for module in self.battery_modules.values():
            module.reset()
        for module in self.ev_modules.values():
            module.reset()

        self.logger.debug(f"DER system reset: {self.name}")

    def _parse_component_config(self, plural_key: str, singular_key: str) -> Dict[str, Dict[str, Any]]:
        """Parse component configuration (same as before)."""
        configs = {}

        if plural_key in self.system_config:
            multi_config = self.system_config[plural_key]

            if isinstance(multi_config, list):
                for idx, comp_config in enumerate(multi_config):
                    comp_id = comp_config.get('id', f"{singular_key}_{idx + 1}")
                    configs[comp_id] = comp_config
            elif isinstance(multi_config, dict):
                for comp_id, comp_config in multi_config.items():
                    if isinstance(comp_config, dict):
                        comp_config['id'] = comp_id
                    configs[comp_id] = comp_config

        elif singular_key in self.system_config:
            single_config = self.system_config[singular_key]
            comp_id = single_config.get('id', f"{singular_key}_1")
            configs[comp_id] = single_config

        return configs


    def get_state(self) -> Dict[str, Any]:
        """Get current state of all component modules."""
        state = {
            'pv_modules': {},
            'battery_modules': {},
            'ev_modules': {},
        }

        # Get state from each component module
        for pv_id, pv_module in self.pv_modules.items():
            state['pv_modules'][pv_id] = pv_module.get_state()

        for bat_id, bat_module in self.battery_modules.items():
            state['battery_modules'][bat_id] = bat_module.get_state()

        for ev_id, ev_module in self.ev_modules.items():
            state['ev_modules'][ev_id] = ev_module.get_state()

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state of all component modules."""
        # Set state for PV modules
        if 'pv_modules' in state:
            for pv_id, pv_state in state['pv_modules'].items():
                if pv_id in self.pv_modules:
                    self.pv_modules[pv_id].set_state(pv_state)

        # Set state for battery modules
        if 'battery_modules' in state:
            for bat_id, bat_state in state['battery_modules'].items():
                if bat_id in self.battery_modules:
                    self.battery_modules[bat_id].set_state(bat_state)

        # Set state for EV modules
        if 'ev_modules' in state:
            for ev_id, ev_state in state['ev_modules'].items():
                if ev_id in self.ev_modules:
                    self.ev_modules[ev_id].set_state(ev_state)

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all components in the system."""
        return {
            'pv_count': len(self.pv_modules),
            'battery_count': len(self.battery_modules),
            'ev_count': len(self.ev_modules),
            'pv_ids': list(self.pv_modules.keys()),
            'battery_ids': list(self.battery_modules.keys()),
            'ev_ids': list(self.ev_modules.keys()),
            'total_pv_capacity': sum(
                m.config.get('rated_capacity_kW', 0) for m in self.pv_modules.values()
            ),
            'total_battery_capacity': sum(
                m.config.get('rated_capacity_kWh', 0) for m in self.battery_modules.values()
            ),
            'total_ev_capacity': sum(
                m.config.get('rated_capacity_kWh', 0) for m in self.ev_modules.values()
            )
        }