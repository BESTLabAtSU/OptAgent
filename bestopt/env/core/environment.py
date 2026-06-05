"""
BESTOpt Runtime environment
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from .base import BaseModule
from .statemanager import StateManager
from .data_structure import (
    ClusterState, ClusterAction, ClusterObservation, Disturbance,
    DomainType, SystemType, ComponentType,
    HVACSystemState, DERSystemState, BuildingSystemState, WaterSystemState,
    FanComponentState, ChillerComponentState, PumpComponentState,
    BoilerComponentState, HeatPumpComponentState, CoilComponentState,
    CoolingTowerComponentState, IceTankComponentState,
    PVComponentState, BatteryComponentState, EVComponentState,
    ThermalZoneComponentState, ElectricalZoneComponentState, WaterZoneComponentState,
    propagate_cross_domain_effects, SystemObservation
)
import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_value = 142857
set_seed(seed_value)


class BESTOptEnvironment:
    """Runtime environment with hierarchical cluster-domain-system-component architecture."""

    def __init__(self, configuration: Dict[str, Any]):
        """
        Initialize environment with configuration.

        Args:
            configuration: Configuration dictionary from ConfigurationManager
        """
        self.config = configuration
        self.logger = logging.getLogger("BESTOptEnvironment")

        # Environment config
        env_config = self.config.get('environment', {})
        env_params = env_config.get('parameters', {})
        self.res = env_params.get("resolution")
        self.dur = env_params.get("duration", 24 * 60 * 60)
        self.total_step = self.dur // self.res
        self.simulation_start_time = env_params.get('simulation_start_time')

        # Get configuration sections
        self.clusters_config = self.config.get('clusters', {})
        self.buildings_config = self.config.get('buildings', {})
        self.systems_config = self.config.get('systems', {})
        self.controllers_config = self.config.get('controllers', {})
        self.thermal_zone_modules_config = self.config.get('thermal_zone_modules', {})
        self.electrical_zone_modules_config = self.config.get('electrical_zone_modules', {})
        self.water_zone_modules_config = self.config.get('water_zone_modules', {})
        self.system_building_map = self.config.get('system_building_map', {})
        self.building_system_map = self.config.get('building_system_map', {})

        # Initialize cluster states
        self.cluster_states: Dict[str, ClusterState] = {}
        self.cluster_actions: Dict[str, ClusterAction] = {}
        self.cluster_observations: Dict[str, ClusterObservation] = {}

        # Global disturbance
        self.disturbance = Disturbance()

        # Tracking
        self.current_step = 0
        self.done = False

        # Module container
        self.system_modules: Dict[str, BaseModule] = {}
        self.system_controllers: Dict[str, BaseModule] = {}
        self.thermal_zone_modules: Dict[str, BaseModule] = {}
        self.water_zone_modules: Dict[str, BaseModule] = {}
        self.electrical_zone_modules: Dict[str, BaseModule] = {}
        self.disturbance_modules: Dict[str, BaseModule] = {}

        # Build runtime environment
        self._create_thermal_zone_modules()  # Create thermal modules
        self._warmup_thermal_modules()  # Warmup for ModNN encoder
        self._create_electrical_zone_modules()  # Create thermal modules
        self._initialize_clusters()  # Initialize hierarchy cluster states
        self._create_all_modules()  # Create module instance

        # State manager with new structure
        self.state_manager = StateManager(configuration, self.cluster_states)

        self._validate_configuration()
        self.logger.info(f"Environment configuration finished")

    def _create_thermal_zone_modules(self):
        """Create thermal zone modules before cluster initialization."""
        created_zones = {}  # Track which zones are created for validation

        for zone_key, zone_config in self.thermal_zone_modules_config.items():
            # Validate zone key format
            if '.' not in zone_key:
                self.logger.error(f"Invalid zone key format: {zone_key}. Expected 'building_id.zone_id'")
                continue

            building_id, zone_id = zone_key.split('.', 1)  # Use maxsplit=1 in case zone_id has dots

            # Verify this building exists in configuration
            if building_id not in self.buildings_config:
                self.logger.warning(f"Zone {zone_key} references non-existent building {building_id}")
                continue

            instance = self._create_instance(
                name=zone_key,
                config=zone_config,
                component_type="thermal_zone",
                must_subclass=BaseModule
            )

            if instance:
                self.thermal_zone_modules[zone_key] = instance

                # Track created zones per building for validation
                if building_id not in created_zones:
                    created_zones[building_id] = []
                created_zones[building_id].append(zone_id)

                self.logger.debug(f"Created thermal zone module: {zone_key} (Building: {building_id}, Zone: {zone_id})")

        # Validate all configured zones have modules
        for building_id, building_config in self.buildings_config.items():
            configured_zones = building_config.get('thermal_zones', ['zone0'])
            created_for_building = created_zones.get(building_id, [])

            for zone_id in configured_zones:
                if zone_id not in created_for_building:
                    self.logger.warning(f"Building {building_id} expects zone {zone_id} but no module was created")

            # Log zone configuration
            self.logger.info(f"Building {building_id}: {len(created_for_building)} zones created "
                             f"({', '.join(created_for_building) if created_for_building else 'none'})")

        self.logger.info(f"Created {len(self.thermal_zone_modules)} thermal zone modules total")

    def _create_electrical_zone_modules(self):
        """Create electrical zone modules before cluster initialization."""
        created_zones = {}  # Track which zones are created for validation

        for zone_key, zone_config in self.electrical_zone_modules_config.items():
            # Validate zone key format
            if '.' not in zone_key:
                self.logger.error(f"Invalid zone key format: {zone_key}. Expected 'building_id.zone_id'")
                continue

            building_id, zone_id = zone_key.split('.', 1)  # Use maxsplit=1 in case zone_id has dots

            # Verify this building exists in configuration
            if building_id not in self.buildings_config:
                self.logger.warning(f"Zone {zone_key} references non-existent building {building_id}")
                continue

            instance = self._create_instance(
                name=zone_key,
                config=zone_config,
                component_type="electrical_zone",
                must_subclass=BaseModule
            )

            if instance:
                self.electrical_zone_modules[zone_key] = instance
                if building_id not in created_zones:
                    created_zones[building_id] = []
                created_zones[building_id].append(zone_id)
                self.logger.debug(
                    f"Created electrical zone module: {zone_key} (Building: {building_id}, Zone: {zone_id})")

        # Validate all configured zones have modules
        for building_id, building_config in self.buildings_config.items():
            configured_zones = building_config.get('electrical_zones', ['zone0'])
            created_for_building = created_zones.get(building_id, [])

            for zone_id in configured_zones:
                if zone_id not in created_for_building:
                    self.logger.warning(f"Building {building_id} expects zone {zone_id} but no module was created")

            # Log zone configuration
            self.logger.info(f"Building {building_id}: {len(created_for_building)} zones created "
                             f"({', '.join(created_for_building) if created_for_building else 'none'})")

        self.logger.info(f"Created {len(self.thermal_zone_modules)} thermal zone modules total")

    def _warmup_thermal_modules(self):
        """Warmup thermal dynamics modules <Since ModNN needs to finish the encoder calculation first>."""
        if not self.thermal_zone_modules:
            self.logger.info("No thermal zone modules to warmup")
            return

        self.logger.info(f"Starting automatic warmup for {len(self.thermal_zone_modules)} thermal dynamics modules...")

        # Cache for initial temperatures
        self._thermal_initial_temps = {}

        for zone_key, module in self.thermal_zone_modules.items():
            try:
                self.logger.info(f"Warming up thermal module: {zone_key}")

                # Prepare module for simulation
                if hasattr(module, 'prepare_for_simulation'):
                    module.prepare_for_simulation(self.simulation_start_time)

                    if hasattr(module, 'get_state_summary'):
                        summary = module.get_state_summary()
                        buffer_size = summary.get('buffer_size', 0)
                        self.logger.info(f"Warmup completed for {zone_key}: {buffer_size} timesteps loaded")

                        # Cache initial temperature
                        if 'current_step_room_temp' in summary:
                            self._thermal_initial_temps[zone_key] = summary['current_step_room_temp']
                else:
                    self.logger.warning(f"Thermal module {zone_key} has no prepare_for_simulation method")

            except Exception as e:
                self.logger.error(f"Failed to warmup {zone_key}: {e}")
                raise RuntimeError(f"Thermal dynamics warmup failed for {zone_key}: {e}")

        self.logger.info(f"Successfully warmed up {len(self.thermal_zone_modules)} thermal dynamics modules")

    def _initialize_clusters(self):
        """Initialize cluster states based on configuration."""
        for cluster_id in self.clusters_config.keys():
            # Create cluster state
            cluster_state = ClusterState(cluster_id=cluster_id)

            # Get buildings in this cluster
            cluster_buildings = [
                bid for bid, bconf in self.buildings_config.items()
                if bconf.get('cluster_id') == cluster_id
            ]

            # Initialize systems for each building
            for building_id in cluster_buildings:
                self._initialize_building_systems(cluster_state, building_id)

            self.cluster_states[cluster_id] = cluster_state
            self.cluster_actions[cluster_id] = ClusterAction(cluster_id=cluster_id)
            self.cluster_observations[cluster_id] = ClusterObservation(cluster_id=cluster_id)

            self.logger.info(f"Initialized cluster: {cluster_id} with {len(cluster_buildings)} buildings")

    def _initialize_building_systems(self, cluster_state: ClusterState, building_id: str):
        """Initialize all systems for a building within the cluster."""
        building_config = self.buildings_config.get(building_id, {})
        building_systems = self.building_system_map.get(building_id, {})

        # Create Building System (spans all domains)
        building_system = BuildingSystemState(system_id=f"{building_id}_building")

        # Get configured thermal zones for this building
        thermal_zones = building_config.get('thermal_zones', ['zone0'])

        # Validate and add thermal zones with warmed-up initial temperatures
        zones_added = 0
        for zone_id in thermal_zones:
            zone_key = f"{building_id}.{zone_id}"

            # Check if thermal module exists for this zone
            if zone_key not in self.thermal_zone_modules:
                self.logger.warning(f"No thermal module for zone {zone_key}, using defaults")

            # Get initial temperature from warmup if available
            initial_temp = self._thermal_initial_temps.get(zone_key)

            tz_state = ThermalZoneComponentState(
                component_id=f"{building_id}_{zone_id}",
                component_type=ComponentType.THERMAL_ZONE,
                system_id=building_system.system_id
            )
            tz_state.temperature = initial_temp
            building_system.components[zone_id] = tz_state
            zones_added += 1

        self.logger.debug(f"Building {building_id}: Added {zones_added} thermal zones")

        # Add electrical zone (typically one per building)
        ez_state = ElectricalZoneComponentState(
            component_id=f"{building_id}_elec_zone",
            component_type=ComponentType.ELECTRICAL_ZONE,
            system_id=building_system.system_id
        )
        building_system.components['electrical'] = ez_state

        # Add water zone if water system is configured
        if 'water' in building_systems:
            wz_state = WaterZoneComponentState(
                component_id=f"{building_id}_water_zone",
                component_type=ComponentType.WATER_ZONE,
                system_id=building_system.system_id
            )
            building_system.components['water'] = wz_state

        # Add building system to relevant domains
        cluster_state.thermal.systems[building_system.system_id] = building_system
        cluster_state.electrical.systems[building_system.system_id] = building_system
        cluster_state.water.systems[building_system.system_id] = building_system

        # Initialize HVAC System State
        if 'thermal' in building_systems:
            hvac_system_id = building_systems['thermal']
            hvac_system = HVACSystemState(system_id=hvac_system_id)
            # Add to relevant domains
            cluster_state.thermal.systems[hvac_system_id] = hvac_system
            cluster_state.electrical.systems[hvac_system_id] = hvac_system

        # Initialize DER System State
        if 'electrical' in building_systems:
            der_system_id = building_systems['electrical']
            der_system = DERSystemState(system_id=der_system_id)
            # Add to relevant domains
            cluster_state.electrical.systems[der_system_id] = der_system

        # Initialize Water System State
        if 'water' in building_systems:
            water_system_id = building_systems['water']
            water_system = WaterSystemState(system_id=water_system_id)
            # Add to relevant domains
            cluster_state.water.systems[water_system_id] = water_system

    def _create_all_modules(self):
        """Create all modules: systems, controllers, and disturbances. Buildings already finished"""

        # Create system modules
        for system_id, system_config in self.systems_config.items():
            instance = self._create_instance(
                name=system_id,
                config=system_config,
                component_type="system",
                must_subclass=BaseModule
            )
            if instance:
                self.system_modules[system_id] = instance
                self._state_registration(system_id, instance)
                self.logger.info(f"Created {system_config['system_type']} system: {system_id}")

        # Create system controllers
        for controller_id, controller_config in self.controllers_config.items():
            system_id = controller_config.get('system_id')
            instance = self._create_instance(
                name=controller_id,
                config=controller_config,
                component_type="controller",
                must_subclass=BaseModule
            )
            if instance and system_id:
                self.system_controllers[system_id] = instance
                self.logger.info(f"Created controller {controller_id} for system {system_id}")

        # Create disturbance modules
        disturbances_config = self.config.get('disturbances', {})
        for dist_name, dist_info in disturbances_config.items():
            instance = self._create_instance(
                name=dist_name,
                config=dist_info,
                component_type="disturbance",
                must_subclass=BaseModule
            )
            if instance:
                self.disturbance_modules[dist_name] = instance

    def _state_registration(self, system_id: str, module: BaseModule):
        """Register component states for a system module."""
        # Find the system state in all clusters
        for cluster_state in self.cluster_states.values():
            # Check each domain
            for domain in [cluster_state.electrical, cluster_state.thermal, cluster_state.water]:
                if system_id in domain.systems:
                    system_state = domain.systems[system_id]

                    if hasattr(module, 'register_component_state'):
                        module.register_component_state(system_state)
                        self.logger.debug(f"Registered components for system {system_id}")
                        return

                    if not hasattr(module, 'register_component_state'):
                        self.logger.warning(f"Module for system {system_id} has no register_component_state method")
                        return

    def _create_instance(self, name: str, config: Dict[str, Any],
                         component_type: str, must_subclass: Optional[type] = None) -> Optional[BaseModule]:
        """Generic function to create component instances."""
        if not config:
            self.logger.warning(f"{component_type} '{name}' has empty config; skipping.")
            return None

        class_path = config.get('class_path')
        if not class_path:
            self.logger.warning(f"{component_type} '{name}' missing 'class_path'; skipping.")
            return None

        try:
            component_class = self._import_class(class_path, must_subclass)
        except Exception as e:
            self.logger.error(f"Failed to import {component_type} '{name}' from {class_path}: {e}")
            return None

        params = config.get('parameters', {})

        try:
            instance = component_class(config=params, name=name)
            instance.initialize()
            instance._initialized = True
            self.logger.info(f"Created {component_type}: {name}")
            return instance
        except Exception as e:
            self.logger.error(f"Failed to create {component_type} '{name}': {e}")
            raise

    @staticmethod
    def _import_class(class_path: str, must_subclass: Optional[type] = None):
        """Import a class from a module path."""
        parts = class_path.split('.')
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        if must_subclass and not issubclass(cls, must_subclass):
            raise TypeError(f"{class_path} must inherit from {must_subclass.__name__}")
        return cls

    def step(self, external_actions: Optional[Dict[str, ClusterAction]] = None) -> Tuple[
        Dict[str, ClusterObservation], bool, Dict[str, Any]]:
        """
        Execute simulation step with proper phase ordering and information flow.

        Execution order:
        1. Update disturbances
        2. Water systems (controller → execution → update thermal demand)
        3. Thermal systems (controller → HVAC execution → zones → update electrical load)
        4. PV generation calculation
        5. Electrical systems (controller → DER execution)
        6. Update observations
        """
        if self.done:
            self.logger.warning("Environment is done. Call reset() to restart.")
            return self.cluster_observations, True, {}

        # Clear state manager cache at start of timestep
        self.state_manager.clear_cache()

        # Phase 1: Update global disturbances
        self._update_disturbances()

        # Phase 2: Process each cluster with proper system ordering
        for cluster_id, cluster_state in self.cluster_states.items():
            # Initialize cluster action container
            cluster_action = ClusterAction(cluster_id=cluster_id) if not external_actions else \
                external_actions.get(cluster_id, ClusterAction(cluster_id=cluster_id))

            # Update observation for current state
            self._update_cluster_observation(cluster_id, cluster_state)

            # Execute systems in dependency order with interleaved control
            self._execute_cluster_systems_ordered(cluster_id, cluster_state, cluster_action)

            # Store final cluster action
            self.cluster_actions[cluster_id] = cluster_action

            # Propagate cross-domain effects
            propagate_cross_domain_effects(cluster_state)

            # Final observation update
            self._update_cluster_observation(cluster_id, cluster_state)

        # Update simulation tracking
        self.current_step += 1
        if self.current_step >= self.total_step:
            self.done = True
            self.logger.info(f"Simulation completed after {self.current_step} steps")

        info = self._collect_step_info()
        return self.cluster_observations, self.done, info

    def _execute_cluster_systems_ordered(self, cluster_id: str, cluster_state: ClusterState,
                                         cluster_action: ClusterAction):
        """
        Execute systems in proper dependency order with interleaved control decisions.

        Order: Water → Thermal → Electrical
        Each stage: Controller → System Execution → State Update
        """
        # Get water control action and execute water systems
        water_systems = [sid for sid, state in cluster_state.water.systems.items()
                         if state.system_type == SystemType.WATER]
        # This is just a placeholder
        for system_id in water_systems:
            if system_id in self.system_controllers:
                # Get water controller action
                water_controller = self.system_controllers[system_id]
                water_action = water_controller.step(
                    state=cluster_state.water.systems[system_id],
                    observation=self.cluster_observations[cluster_id],
                    disturbance=self.disturbance,
                    timestep=self.current_step
                )
                cluster_action.water.system_actions[system_id] = water_action

                # Execute water system immediately
                if system_id in self.system_modules:
                    water_module = self.system_modules[system_id]
                    water_result = water_module.step(
                        state=cluster_state.water.systems[system_id],
                        action=water_action,
                        disturbance=self.disturbance,
                        timestep=self.current_step
                    )

                    # Cache thermal demand from water system for thermal controller
                    if 'thermal_demand' in water_result:
                        self.state_manager.cache['water_thermal_demand'][system_id] = water_result['thermal_demand']

        # Get thermal control action with updated water thermal demand
        hvac_systems = [sid for sid, state in cluster_state.thermal.systems.items()
                        if state.system_type == SystemType.HVAC]

        for system_id in hvac_systems:
            # Find the building that system support
            # @todo use get function later
            building_id = f"{self.system_building_map[system_id][0]}_building"
            if system_id in self.system_controllers:
                # Thermal controller can now use updated water thermal demand
                thermal_controller = self.system_controllers[system_id]

                # Pass water thermal demand to controller via observation or state
                if hasattr(thermal_controller, 'set_water_thermal_demand'):
                    water_demand = sum(self.state_manager.cache.get('water_thermal_demand', {}).values())
                    # PLace holder
                    thermal_controller.set_water_thermal_demand(water_demand)

                thermal_action = thermal_controller.step(
                    state=cluster_state.thermal.systems[system_id],
                    observation=(cluster_state.electrical.systems[system_id], cluster_state.electrical.systems[building_id]), #@todo again, need obs function
                    disturbance=self.disturbance,
                    timestep=self.current_step
                )
                cluster_action.thermal.system_actions[system_id] = thermal_action

                # Execute HVAC system immediately
                if system_id in self.system_modules:
                    hvac_module = self.system_modules[system_id]
                    # Set return air temperature from zones
                    # @todo I still feel something wrong here, the actions should be handled by controller
                    # as a dynamic model, it should not take any obs as input
                    # right now, the coil still need return air temperature as input, which is not make sense to me....
                    # as a local component, it dont need such access, all it need to do is actually maintain the pressure? or in other words, track the setpoint...
                    hvac_module.update_return_air_temperature(cluster_state.electrical.systems[building_id].components['zone0'].temperature)

                    # all_zone_temps = []
                    # for building_id in building_ids:
                    #     building_system_id = f"{building_id}_building"
                    #     building_system = cluster_state.thermal.systems.get(building_system_id)
                    #     if building_system:
                    #         for component in building_system.components.values():
                    #             if component.component_type == ComponentType.THERMAL_ZONE:
                    #                 all_zone_temps.append(component.temperature)
                    #
                    # if all_zone_temps:
                    #     return_air_temp = sum(all_zone_temps) / len(all_zone_temps)
                    #     hvac_module.update_return_air_temperature(return_air_temp)

                    hvac_result = hvac_module.step(
                        state=cluster_state.thermal.systems[system_id],
                        action=thermal_action,
                        disturbance=self.disturbance,
                        timestep=self.current_step
                    )

                    # Cache HVAC results
                    if 'FCU_power_total_W' in hvac_result:
                        self.state_manager.cache['hvac_power'][system_id] = hvac_result['FCU_power_total_W']
                    if 'Q_zone_actual_W' in hvac_result:
                        building_id = self.system_building_map.get(system_id, ['unknown'])[0]
                        if not hasattr(self.state_manager.cache, 'hvac_thermal_load'):
                            self.state_manager.cache['hvac_thermal_load'] = {}
                        self.state_manager.cache['hvac_thermal_load'][building_id] = hvac_result['Q_zone_actual_W']

        # Execute thermal zones with HVAC loads
        self._execute_thermal_zones(cluster_state, cluster_action)
        self._execute_electrical_zones(cluster_state, cluster_action)
        # Update building electrical loads based on HVAC power
        # self._update_building_loads(cluster_state)

        # Calculate PV generation before electrical control decision
        self._execute_pv_generation(cluster_state, cluster_action)
        # print(cluster_state.electrical.systems['SFH_1_building'].components['electrical'].building_power_w)
        # print(cluster_state.electrical.systems['der_system_1'].components['pv_1'].generation_w)

        # Get electrical control action with all updated information
        der_systems = [sid for sid, state in cluster_state.electrical.systems.items()
                       if state.system_type == SystemType.DER]
        # @todo ADD centralized/decentralized
        for system_id in der_systems:
            # Find the building that system support
            building_id = f"{self.system_building_map[system_id][0]}_building"

            if system_id in self.system_controllers:
                electrical_controller = self.system_controllers[system_id]
                electrical_action = electrical_controller.step(
                    state=cluster_state.electrical.systems[system_id], #@todo In the early stage, I didn't separate state/obs clearly just for simplifacation, now we need to gradually update these functions
                    observation=(cluster_state.electrical.systems[system_id], cluster_state.electrical.systems[building_id]),
                    disturbance=self.disturbance,
                    timestep=self.current_step
                )
                cluster_action.electrical.system_actions[system_id] = electrical_action
                #
                # Execute DER system immediately
                if system_id in self.system_modules:
                    der_module = self.system_modules[system_id]
                    der_result = der_module.step(
                        state=cluster_state.electrical.systems[system_id],
                        action=electrical_action,
                        disturbance=self.disturbance,
                        resolution=self.res,
                        timestep=self.current_step,
                    )

    def _get_controller_action_for_system(self, system_id: str, cluster_state: ClusterState,
                                          cluster_observation: ClusterObservation,
                                          domain_type: DomainType) -> Any:
        """
        Get control action for a single system.
        This replaces the bulk action collection with targeted action generation.
        """
        if system_id not in self.system_controllers:
            return None

        controller = self.system_controllers[system_id]
        domain_state = cluster_state.get_domain(domain_type)
        system_state = domain_state.systems.get(system_id)

        if not system_state:
            return None

        # Get any cached information relevant to this controller
        cached_info = {}
        if domain_type == DomainType.THERMAL:
            cached_info['water_thermal_demand'] = sum(
                self.state_manager.cache.get('water_thermal_demand', {}).values()
            )
        elif domain_type == DomainType.ELECTRICAL:
            cached_info['hvac_power'] = sum(
                self.state_manager.cache.get('hvac_power', {}).values()
            )
            cached_info['building_load'] = self._calculate_building_load(cluster_state)
            cached_info['pv_generation'] = sum(
                self.state_manager.cache.get('pv_generation', {}).values()
            )

        # Call controller with additional context if supported
        if hasattr(controller, 'step_with_context'):
            return controller.step_with_context(
                state=system_state,
                observation=cluster_observation,
                disturbance=self.disturbance,
                timestep=self.current_step,
                context=cached_info
            )
        else:
            return controller.step(
                state=system_state,
                observation=cluster_observation,
                disturbance=self.disturbance,
                timestep=self.current_step
            )

    def _update_disturbances(self):
        """Update global disturbances from modules."""
        for dist_name, dist_module in self.disturbance_modules.items():
            try:
                dist_update = dist_module.step(current_step=self.current_step)
                #@todo again, the module, state management should aligh with system module
                if dist_name == "weather":
                    self.disturbance.weather = dist_update
                    # if self.disturbance.weather_buffer is None:
                    #     self.disturbance.weather_buffer = []
                    # else:
                    #     self.disturbance.weather_buffer.append(dist_update)
                elif dist_name == "price":
                    self.disturbance.prices = dist_update
                    # if self.disturbance.prices_buffer is None:
                    #     self.disturbance.prices_buffer = []
                    # else:
                    #     self.disturbance.prices_buffer.append(dist_update)
                elif dist_name == "occupancy":
                    self.disturbance.occupancy = dist_update
                    # if self.disturbance.occupancy_buffer is None:
                    #     self.disturbance.occupancy_buffer = []
                    # else:
                    #     self.disturbance.occupancy_buffer.append(dist_update)

            except Exception as e:
                self.logger.error(f"Disturbance {dist_name} update failed: {e}")

    def _get_cluster_action(self, cluster_id: str, cluster_state: ClusterState) -> ClusterAction:
        """Get control actions for all systems in cluster."""
        self._update_cluster_observation(cluster_id, cluster_state)
        cluster_action = ClusterAction(cluster_id=cluster_id)

        # Get actions for each domain's systems
        for domain_type in [DomainType.THERMAL, DomainType.ELECTRICAL, DomainType.WATER]:
            domain_state = cluster_state.get_domain(domain_type)

            for system_id, system_state in domain_state.systems.items():
                if system_id in self.system_controllers:
                    controller = self.system_controllers[system_id]

                    # Controllers work with system states directly
                    system_action = controller.step(
                        state=system_state,
                        observation=self.cluster_observations[cluster_id],
                        disturbance=self.disturbance,
                        timestep=self.current_step
                    )

                    # Assign action to appropriate domain
                    if domain_type == DomainType.THERMAL:
                        cluster_action.thermal.system_actions[system_id] = system_action
                    elif domain_type == DomainType.ELECTRICAL:
                        cluster_action.electrical.system_actions[system_id] = system_action
                    elif domain_type == DomainType.WATER:
                        cluster_action.water.system_actions[system_id] = system_action

        return cluster_action


    def _execute_pv_generation(self, cluster_state: ClusterState, cluster_action: ClusterAction):
        """Calculate PV generation for all DER systems."""
        for system_id, system_state in cluster_state.electrical.systems.items():
            if system_state.system_type == SystemType.DER and system_id in self.system_modules:
                der_module = self.system_modules[system_id]

                # Pass the system state to calculate_pv_generation
                pv_generation = der_module.calculate_pv_generation(
                    state=system_state,  # Pass the actual system state
                    disturbance=self.disturbance,
                    timestep=self.current_step,
                )

                # Cache for electrical control decisions
                self.state_manager.cache['pv_generation'][system_id] = pv_generation

                # Log the generation
                self.logger.debug(f"DER {system_id}: Total PV generation = {pv_generation:.1f}W")

    def _execute_thermal_systems(self, cluster_state: ClusterState, cluster_action: ClusterAction):
        """Execute HVAC systems."""
        for system_id, system_state in cluster_state.thermal.systems.items():
            if system_state.system_type == SystemType.HVAC and system_id in self.system_modules:
                hvac_module = self.system_modules[system_id]

                # Find building(s) this HVAC serves
                building_ids = self.system_building_map.get(system_id, [])

                # Collect zone temperatures from all buildings served by this HVAC
                all_zone_temps = []

                for building_id in building_ids:
                    building_system_id = f"{building_id}_building"
                    building_system = cluster_state.thermal.systems.get(building_system_id)

                    if building_system:
                        for component in building_system.components.values():
                            if component.component_type == ComponentType.THERMAL_ZONE:
                                all_zone_temps.append(component.temperature)
                                self.logger.debug(f"Zone {component.component_id}: {component.temperature:.1f}°C")

                return_air_temp = sum(all_zone_temps) / len(all_zone_temps)
                hvac_module.update_return_air_temperature(return_air_temp)
                self.logger.debug(f"HVAC {system_id} return air temp set to {return_air_temp:.1f}°C "
                                  f"(avg of {len(all_zone_temps)} zones)")

                # Get thermal action for this system
                thermal_action = cluster_action.thermal.system_actions.get(system_id)

                hvac_result = hvac_module.step(
                    state=system_state,
                    action=thermal_action,
                    disturbance=self.disturbance,
                    timestep=self.current_step
                )

                # Cache HVAC power and thermal load
                if 'FCU_power_total_W' in hvac_result:
                    self.state_manager.cache['hvac_power'][system_id] = hvac_result['FCU_power_total_W']
                if 'Q_zone_actual_W' in hvac_result:
                    # Cache for thermal zones
                    building_id = self.system_building_map.get(system_id, ['unknown'])[0]
                    if not hasattr(self.state_manager.cache, 'hvac_thermal_load'):
                        self.state_manager.cache['hvac_thermal_load'] = {}
                    self.state_manager.cache['hvac_thermal_load'][building_id] = hvac_result['Q_zone_actual_W']

    def _execute_thermal_zones(self, cluster_state: ClusterState, cluster_action: ClusterAction):
        """Execute thermal zone dynamics using HVAC actions directly."""

        # Process each building's thermal zones
        for system_id, system_state in cluster_state.thermal.systems.items():
            if system_state.system_type == SystemType.BUILDING:
                building_id = system_id.replace('_building', '')

                # Find HVAC system serving this building
                hvac_system_id = self.building_system_map.get(building_id, {}).get('thermal')

                # Get HVAC thermal load from the HVAC module results or action
                hvac_load_total = 0.0

                if hvac_system_id and hvac_system_id in self.system_modules:
                    hvac_module = self.system_modules[hvac_system_id]
                    if hasattr(hvac_module, 'Q_zone_actual_W'):
                        hvac_load_total = hvac_module.Q_zone_actual_W

                # Count and distribute load
                thermal_zones = [comp for comp in system_state.components.values()
                                 if comp.component_type == ComponentType.THERMAL_ZONE]
                num_zones = len(thermal_zones)

                if num_zones == 0:
                    continue

                hvac_load_per_zone = hvac_load_total / num_zones

                # Process each zone with its share of HVAC load
                for zone_id, zone_state in system_state.components.items():
                    if zone_state.component_type == ComponentType.THERMAL_ZONE:
                        zone_key = f"{building_id}.{zone_id}"

                        if zone_key not in self.thermal_zone_modules:
                            continue

                        zone_module = self.thermal_zone_modules[zone_key]

                        # Pass HVAC thermal load to zone
                        zone_action = type('ZoneAction', (), {
                            'hvac_thermal_load': hvac_load_per_zone,
                            'hvac_system_id': hvac_system_id,
                            'zone_key': zone_key
                        })()
                        #@todo the state and module should be handled like system, instead add separatly outside the module
                        zone_result = zone_module.step(
                            state=zone_state,
                            action=zone_action,
                            disturbance=self.disturbance,
                            timestep=self.current_step
                        )


    def _execute_electrical_zones(self, cluster_state: ClusterState,
                                  cluster_action: ClusterAction):
        """Execute thermal zone dynamics using HVAC actions directly."""

        # Process each building's thermal zones
        for system_id, system_state in cluster_state.thermal.systems.items():
            if system_state.system_type == SystemType.BUILDING:
                building_id = system_id.replace('_building', '')

                # Find HVAC system serving this building
                hvac_system_id = self.building_system_map.get(building_id, {}).get(
                    'thermal')

                # Get HVAC thermal load from the HVAC module results or action
                hvac_load_total = 0.0

                if hvac_system_id and hvac_system_id in self.system_modules:
                    hvac_module = self.system_modules[hvac_system_id]
                    if hasattr(hvac_module, 'FCU_power_total_W'):
                        hvac_load_total = hvac_module.FCU_power_total_W

                # Process each zone with its share of HVAC load
                for zone_id, zone_state in system_state.components.items():
                    if zone_state.component_type == ComponentType.ELECTRICAL_ZONE:
                        zone_key = f"{building_id}.{zone_id}"
                        # @ todo the name and structure need to be revised!
                        # @ todo need to check the name for multi-building cluster, forget if it can work or not
                        zone_module = self.electrical_zone_modules['SFH_1.zone0']
                        building_power = zone_module.step(
                            disturbance=self.disturbance,
                            timestep=self.current_step
                        )
                        #@todo should update inside the module step
                        zone_state.total_load_w = building_power + hvac_load_total
                        zone_state.hvac_load_w = hvac_load_total
                        zone_state.building_power_w = building_power

    def _update_building_loads(self, cluster_state: ClusterState):
        """Update electrical loads based on HVAC and building operations."""
        # Sum HVAC power from all HVAC systems
        total_hvac_power = sum(self.state_manager.cache.get('hvac_power', {}).values())

        # Update building electrical zones
        for system_state in cluster_state.electrical.systems.values():
            if system_state.system_type == SystemType.BUILDING:
                for comp_id, component in system_state.components.items():
                    if component.component_type == ComponentType.ELECTRICAL_ZONE:
                        # Update loads
                        component.plug_load_w = 1000.0  # Base load
                        component.lighting_load_w = 500.0  # Lighting
                        component.total_load_w = (
                                component.plug_load_w +
                                component.lighting_load_w +
                                total_hvac_power
                        )
                        component.set_domain_impact(DomainType.ELECTRICAL, 'load', component.total_load_w)

    def _execute_electrical_systems(self, cluster_state: ClusterState, cluster_action: ClusterAction):
        """Execute DER systems with updated load information."""
        for system_id, system_state in cluster_state.electrical.systems.items():
            if system_state.system_type == SystemType.DER and system_id in self.system_modules:
                der_module = self.system_modules[system_id]
                electrical_action = cluster_action.electrical.system_actions.get(system_id)
                # Calculate total building load for this DER system
                building_load = self._calculate_building_load(cluster_state)

                if electrical_action or True:  # Execute even without action
                    der_result = der_module.step(
                        state=system_state,
                        action=electrical_action,
                        disturbance=self.disturbance,
                        resolution=self.res,
                        timestep=self.current_step,
                        building_load=building_load  # Pass building load
                    )

    def _calculate_building_load(self, cluster_state: ClusterState) -> float:
        """Calculate total electrical load from buildings."""
        total_load = 0.0

        # Sum loads from all building electrical zones
        for system_state in cluster_state.electrical.systems.values():
            if system_state.system_type == SystemType.BUILDING:
                for component in system_state.components.values():
                    if component.component_type == ComponentType.ELECTRICAL_ZONE:
                        total_load += component.total_load_w

        # Add HVAC power consumption
        total_load += sum(self.state_manager.cache.get('hvac_power', {}).values())

        return total_load

    def _execute_water_systems(self, cluster_state: ClusterState, cluster_action: ClusterAction):
        """Execute water systems if any."""
        for system_id, system_state in cluster_state.water.systems.items():
            if system_id in self.system_modules:
                water_module = self.system_modules[system_id]
                water_action = cluster_action.water.system_actions.get(system_id)

                if water_action or True:
                    water_module.step(
                        state=system_state,
                        action=water_action,
                        disturbance=self.disturbance,
                        timestep=self.current_step
                    )

    def _update_cluster_observation(self, cluster_id: str, cluster_state: ClusterState):
        """Update observation for cluster with individual zone temperatures."""
        obs = self.cluster_observations[cluster_id]

        # Time information
        obs.time_of_day = (self.current_step * self.res / 3600) % 24
        obs.day_of_week = int((self.current_step * self.res / 86400)) % 7 + 1
        obs.day_of_year = int((self.current_step * self.res / 86400)) % 365 + 1
        obs.timestamp = self.current_step * self.res

        # Store zone temperatures in thermal observation metrics
        zone_temperatures = {}
        for system_id, system_state in cluster_state.thermal.systems.items():
            if system_state.system_type == SystemType.BUILDING:
                building_id = system_id.replace('_building', '')
                for comp_id, component in system_state.components.items():
                    if component.component_type == ComponentType.THERMAL_ZONE:
                        zone_key = f"{building_id}.{comp_id}"
                        zone_temperatures[zone_key] = component.temperature

        # Store all zone temperatures in thermal observation
        obs.thermal.aggregated_metrics['zone_temperatures'] = zone_temperatures

        # Update domain observations with actual metrics
        for domain_type in [DomainType.ELECTRICAL, DomainType.THERMAL, DomainType.WATER]:
            domain_state = cluster_state.get_domain(domain_type)
            domain_obs = getattr(obs, domain_type.value)

            # Update system observations
            for system_id, system_state in domain_state.systems.items():
                # Create system observation if needed
                if system_id not in domain_obs.system_observations:
                    domain_obs.system_observations[system_id] = SystemObservation(
                        system_id=system_id,
                        system_type=system_state.system_type
                    )

                # Update metrics
                system_state.update_domain_metrics()
                system_obs = domain_obs.system_observations[system_id]

                # Store domain-specific metrics
                if domain_type in system_state.domain_metrics:
                    system_obs.metrics.update(system_state.domain_metrics[domain_type])

                # Store component states
                for comp_id, comp_state in system_state.components.items():
                    if comp_state.component_type == ComponentType.THERMAL_ZONE:
                        system_obs.component_states[comp_id] = {
                            'temperature': comp_state.temperature,
                        }
                    elif comp_state.component_type == ComponentType.ELECTRICAL_ZONE:
                        system_obs.component_states[comp_id] = {
                            'total_load_w': comp_state.total_load_w
                        }

        # Set forecasts
        obs.weather_forecast = [self.disturbance.weather] * 4
        obs.price_forecast = [self.disturbance.grid.electricity_price] * 4
        obs.occupancy_forecast = [self.disturbance.occupancy.occupancy_fraction] * 4

    def _collect_step_info(self) -> Dict[str, Any]:
        """Collect information about the current step."""
        info = {
            "step": self.current_step,
            "time_hours": self.current_step * self.res / 3600,
            "clusters": {}
        }

        for cluster_id, cluster_state in self.cluster_states.items():
            cluster_state.update_all()

            info["clusters"][cluster_id] = {
                "electrical": {
                    "total_power": cluster_state.electrical.total_power,
                    "interdomain_flows": dict(cluster_state.electrical.interdomain_flows)
                },
                "thermal": {
                    "total_heat_flow": cluster_state.thermal.total_heat_flow,
                    "interdomain_flows": dict(cluster_state.thermal.interdomain_flows)
                },
                "water": {
                    "total_water_flow": cluster_state.water.total_water_flow,
                    "interdomain_flows": dict(cluster_state.water.interdomain_flows)
                }
            }

        return info

    def reset(self) -> Dict[str, ClusterObservation]:
        """Reset the environment to initial state."""
        self.logger.info("Resetting environment")

        # Reset tracking
        self.current_step = 0
        self.done = False

        # Reset all modules
        for module in self.system_modules.values():
            module.reset()
        for module in self.system_controllers.values():
            module.reset()
        for module in self.thermal_zone_modules.values():
            module.reset()
        for module in self.disturbance_modules.values():
            module.reset()

        # Re-warmup thermal modules
        self._warmup_thermal_modules()

        # Reinitialize clusters
        self.cluster_states.clear()
        self.cluster_actions.clear()
        self.cluster_observations.clear()
        self._initialize_clusters()

        # Reset state manager
        self.state_manager = StateManager(self.config, self.cluster_states)

        return self.cluster_observations

    def _validate_configuration(self):
        """Validate the environment configuration."""
        if self.dur % self.res != 0:
            self.logger.warning(
                f"duration ({self.dur}) not divisible by resolution ({self.res}); "
                f"sim will run {self.total_step} steps (= floor)."
            )
        if not isinstance(self.dur, int) or self.dur <= 0:
            raise ValueError(f"'duration' must be a positive int (seconds); got {self.dur}")
        if self.res is None:
            raise ValueError("Missing required config key: 'resolution'")
        if not isinstance(self.res, int) or self.res <= 0:
            raise ValueError(f"'resolution' must be a positive int (seconds); got {self.res}")

        """Validate the runtime configuration."""
        for cluster_id, cluster_state in self.cluster_states.items():
            # Check system types
            has_hvac = any(s.system_type == SystemType.HVAC for s in cluster_state.thermal.systems.values())
            has_der = any(s.system_type == SystemType.DER for s in cluster_state.electrical.systems.values())
            has_building = any(s.system_type == SystemType.BUILDING for s in cluster_state.thermal.systems.values())

            # Check thermal zone modules
            thermal_zones_count = sum(
                1 for s in cluster_state.thermal.systems.values()
                if s.system_type == SystemType.BUILDING
                for c in s.components.values()
                if c.component_type == ComponentType.THERMAL_ZONE
            )

            self.logger.info(
                f"Cluster {cluster_id} - HVAC: {has_hvac}, DER: {has_der}, "
                f"Building: {has_building}, Thermal Zones: {thermal_zones_count}"
            )

            if not has_building:
                self.logger.warning(f"Cluster {cluster_id} has no building system")

            # Check that each system has a controller
            for domain in [cluster_state.thermal, cluster_state.electrical, cluster_state.water]:
                for system_id in domain.systems.keys():
                    if system_id not in self.system_controllers and not system_id.endswith('_building'):
                        self.logger.warning(f"System {system_id} has no controller assigned")

    def get_system_info(self, system_id: str) -> Dict[str, Any]:
        """Get information about a specific system."""
        if system_id not in self.system_modules:
            return {}

        system_config = self.systems_config.get(system_id, {})
        return {
            "system_type": system_config.get("system_type"),
            "supported_buildings": self.system_building_map.get(system_id, []),
            "has_controller": system_id in self.system_controllers,
            "parameters": system_config.get("parameters", {})
        }

    def get_building_systems(self, building_id: str) -> Dict[str, str]:
        """Get systems assigned to a building."""
        return self.building_system_map.get(building_id, {})
