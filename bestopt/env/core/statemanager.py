"""
State Manager to Handle cross-domain state dependencies and aggregations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from .data_structure import (
    ClusterState, ClusterAction, DomainType, SystemType, ComponentType,
    SystemState, SystemAction, propagate_cross_domain_effects
)


class StateManager:
    """
    Centralized state manager for cross-domain state dependencies.
    Works with the new hierarchical cluster-domain-system-component structure.
    """

    def __init__(self, config: Dict[str, Any], cluster_states: Dict[str, ClusterState]):
        """
        Initialize state manager with references to cluster states.

        Args:
            config: Environment configuration
            cluster_states: Dictionary of cluster states from environment
        """
        self.config = config
        self.logger = logging.getLogger("StateManagerV2")

        # Reference to cluster states (not creating new ones)
        self.cluster_states = cluster_states

        # Cache for intermediate results and cross-domain flows
        self.cache = {
            'pv_generation': {},  # system_id -> total generation
            'building_loads': {},  # building_id -> total load
            'hvac_power': {},  # system_id -> power consumption
            'cross_domain_flows': {}  # (from_domain, to_domain, system_id) -> flow
        }

        # System mappings from config
        self.system_building_map = config.get('system_building_map', {})
        self.building_system_map = config.get('building_system_map', {})

    def update_pv_generation(self, cluster_id: str, system_id: str) -> float:
        """Calculate and cache total PV generation for a DER system."""
        cluster = self.cluster_states.get(cluster_id)
        if not cluster:
            return 0.0

        total_generation = 0.0

        # Find DER system in electrical domain
        if system_id in cluster.electrical.systems:
            system_state = cluster.electrical.systems[system_id]
            if system_state.system_type == SystemType.DER:
                # Sum generation from all PV components
                for component in system_state.components.values():
                    if component.component_type == ComponentType.PV:
                        total_generation += component.generation_w

        self.cache['pv_generation'][system_id] = total_generation
        self.logger.debug(f"Cached PV generation for {system_id}: {total_generation:.2f}W")

        return total_generation

    def update_building_electrical_load(self, cluster_id: str, building_id: str) -> float:
        """Update total electrical load for a building including HVAC."""
        cluster = self.cluster_states.get(cluster_id)
        if not cluster:
            return 0.0

        total_load = 0.0

        # Get base load from building electrical zones
        building_system_id = f"{building_id}_building"
        if building_system_id in cluster.electrical.systems:
            building_system = cluster.electrical.systems[building_system_id]
            for component in building_system.components.values():
                if component.component_type == ComponentType.ELECTRICAL_ZONE:
                    total_load += component.total_load_w

        # Add HVAC power consumption
        hvac_power = self.get_hvac_power_consumption(cluster_id, building_id)
        total_load += hvac_power

        self.cache['building_loads'][building_id] = total_load
        self.logger.debug(f"Updated electrical load for {building_id}: {total_load:.2f}W")

        return total_load

    def get_hvac_power_consumption(self, cluster_id: str, building_id: str) -> float:
        """Get HVAC power consumption for a building."""
        cluster = self.cluster_states.get(cluster_id)
        if not cluster:
            return 0.0

        # Get HVAC system for this building
        building_systems = self.building_system_map.get(building_id, {})
        hvac_system_id = building_systems.get('thermal')

        if not hvac_system_id:
            return 0.0

        # Check cache first
        if hvac_system_id in self.cache['hvac_power']:
            return self.cache['hvac_power'][hvac_system_id]

        # Calculate from HVAC system components
        total_power = 0.0
        if hvac_system_id in cluster.thermal.systems:
            hvac_system = cluster.thermal.systems[hvac_system_id]
            hvac_system.update_domain_metrics()

            # Get electrical power impact
            total_power = hvac_system.domain_metrics.get(
                DomainType.ELECTRICAL, {}
            ).get('power', 0.0)

        self.cache['hvac_power'][hvac_system_id] = total_power
        return total_power

    def get_cluster_metrics(self, cluster_id: str) -> Dict[str, Any]:
        """Get aggregated metrics for a cluster."""
        cluster = self.cluster_states.get(cluster_id)
        if not cluster:
            return {}

        # Update all domain aggregations
        cluster.update_all()

        metrics = {
            'electrical': {
                'total_generation': 0.0,
                'total_load': 0.0,
                'total_storage': 0.0,
                'net_power': 0.0
            },
            'thermal': {
                'total_cooling': 0.0,
                'total_heating': 0.0,
                'average_temperature': 0.0
            },
            'water': {
                'total_consumption': 0.0,
                'total_heating': 0.0
            },
            'cross_domain': {}
        }

        # Aggregate electrical metrics
        for system in cluster.electrical.systems.values():
            system.update_domain_metrics()

            if system.system_type == SystemType.DER:
                # Sum generation from PV
                for comp in system.components.values():
                    if comp.component_type == ComponentType.PV:
                        metrics['electrical']['total_generation'] += comp.generation_w
                    elif comp.component_type == ComponentType.BATTERY:
                        metrics['electrical']['total_storage'] += abs(comp.power_w)

            elif system.system_type == SystemType.BUILDING:
                # Sum building loads
                for comp in system.components.values():
                    if comp.component_type == ComponentType.ELECTRICAL_ZONE:
                        metrics['electrical']['total_load'] += comp.total_load_w

        # Aggregate thermal metrics
        for system in cluster.thermal.systems.values():
            system.update_domain_metrics()

            if system.system_type == SystemType.HVAC:
                cooling = system.domain_metrics.get(DomainType.THERMAL, {}).get('cooling', 0.0)
                heating = system.domain_metrics.get(DomainType.THERMAL, {}).get('heating', 0.0)
                metrics['thermal']['total_cooling'] += abs(cooling)
                metrics['thermal']['total_heating'] += heating

            elif system.system_type == SystemType.BUILDING:
                # Average temperature from thermal zones
                temps = []
                for comp in system.components.values():
                    if comp.component_type == ComponentType.THERMAL_ZONE:
                        temps.append(comp.temperature_c)
                if temps:
                    metrics['thermal']['average_temperature'] = sum(temps) / len(temps)

        # Calculate net power
        metrics['electrical']['net_power'] = (
                metrics['electrical']['total_generation'] -
                metrics['electrical']['total_load']
        )

        # Track cross-domain flows
        metrics['cross_domain']['hvac_to_electrical'] = sum(self.cache['hvac_power'].values())

        return metrics

    def get_system_aggregated_state(self, cluster_id: str, system_id: str) -> Optional[SystemState]:
        """Get aggregated state for a system (for centralized control)."""
        cluster = self.cluster_states.get(cluster_id)
        if not cluster:
            return None

        # Search all domains for the system
        for domain in [cluster.electrical, cluster.thermal, cluster.water]:
            if system_id in domain.systems:
                system_state = domain.systems[system_id]
                system_state.update_domain_metrics()
                return system_state

        return None

    def distribute_system_action(self, action: SystemAction,
                                 supported_buildings: List[str]) -> Dict[str, SystemAction]:
        """
        Distribute a centralized system action to multiple buildings.

        For systems that support multiple buildings, this distributes
        the centralized control action appropriately.
        """
        distributed_actions = {}

        # For now, same action applies to all supported buildings
        # Could implement more sophisticated distribution logic
        for building_id in supported_buildings:
            distributed_actions[building_id] = action

        return distributed_actions

    def track_cross_domain_flow(self, from_domain: DomainType, to_domain: DomainType,
                                system_id: str, flow_value: float):
        """Track energy/resource flow between domains."""
        flow_key = (from_domain, to_domain, system_id)
        self.cache['cross_domain_flows'][flow_key] = flow_value

        self.logger.debug(
            f"Cross-domain flow: {from_domain.value} → {to_domain.value} "
            f"via {system_id}: {flow_value:.2f}"
        )

    def get_cross_domain_flows(self, cluster_id: str) -> Dict[Tuple, float]:
        """Get all cross-domain flows for a cluster."""
        cluster = self.cluster_states.get(cluster_id)
        if not cluster:
            return {}

        flows = {}

        # Calculate HVAC → Electrical flows
        for system in cluster.thermal.systems.values():
            if system.system_type == SystemType.HVAC:
                system.update_domain_metrics()
                power = system.domain_metrics.get(DomainType.ELECTRICAL, {}).get('power', 0.0)
                if power > 0:
                    flows[(DomainType.THERMAL, DomainType.ELECTRICAL, system.system_id)] = power

        # Calculate Water → Thermal flows (hot water heating)
        for system in cluster.water.systems.values():
            system.update_domain_metrics()
            heating = system.domain_metrics.get(DomainType.THERMAL, {}).get('heating', 0.0)
            if heating > 0:
                flows[(DomainType.WATER, DomainType.THERMAL, system.system_id)] = heating

        return flows

    def validate_power_balance(self, cluster_id: str) -> Dict[str, float]:
        """Validate electrical power balance in the cluster."""
        metrics = self.get_cluster_metrics(cluster_id)

        generation = metrics['electrical']['total_generation']
        load = metrics['electrical']['total_load']
        storage_net = metrics['electrical']['total_storage']  # Positive=charging

        # Power balance: Generation = Load + Storage + Export (or Import if negative)
        imbalance = generation - load - storage_net

        return {
            'generation': generation,
            'load': load,
            'storage_net': storage_net,
            'imbalance': imbalance,
            'balanced': abs(imbalance) < 1.0  # Within 1W tolerance
        }

    def clear_cache(self):
        """Clear all cached values (call at start of each timestep)."""
        for cache_dict in self.cache.values():
            cache_dict.clear()

    def get_building_systems(self, building_id: str) -> Dict[DomainType, str]:
        """Get all systems serving a building, organized by domain."""
        systems_by_domain = {}

        building_systems = self.building_system_map.get(building_id, {})

        # Map old domain strings to new enums
        domain_mapping = {
            'thermal': DomainType.THERMAL,
            'electrical': DomainType.ELECTRICAL,
            'water': DomainType.WATER
        }

        for old_domain, system_id in building_systems.items():
            if old_domain in domain_mapping:
                systems_by_domain[domain_mapping[old_domain]] = system_id

        return systems_by_domain

    def get_system_buildings(self, system_id: str) -> List[str]:
        """Get all buildings supported by a system."""
        return self.system_building_map.get(system_id, [])