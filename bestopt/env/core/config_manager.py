"""
Configuration Manager for Cluster-Building-System Architecture
"""
import json
import copy
import logging
from typing import Dict, Any, List, Optional, Literal, Tuple
from pathlib import Path
from dataclasses import is_dataclass, asdict

SystemType = Literal["hvac_systems", "der_systems", "water_systems"]
DomainType = Literal["thermal", "electrical", "water"]


class ConfigurationManager:
    """Manages configuration for cluster-building-system architecture."""

    def __init__(self, json_file_path: Optional[str] = None):
        self.logger = logging.getLogger("ConfigurationManager")
        self.config_path = Path(json_file_path) if json_file_path else None

        # Selected configurations
        self.selected_clusters: Dict[str, Dict[str, Any]] = {}
        self.selected_buildings: Dict[str, Dict[str, Any]] = {}
        self.selected_systems: Dict[str, Dict[str, Any]] = {}
        self.selected_controllers: Dict[str, Dict[str, Any]] = {}
        self.selected_disturbances: Dict[str, Dict[str, Any]] = {}
        self.selected_environment: Dict[str, Any] = {}

        # Track system-building mappings
        self.system_building_map: Dict[str, List[str]] = {}  # system_id -> [building_ids]
        self.building_system_map: Dict[str, Dict[str, str]] = {}  # building_id -> {domain -> system_id}

        if self.config_path and self.config_path.exists():
            self.config = self._load_json()
            self.logger.info(f"Loaded existing configuration from {json_file_path}")
            self._select_all()
        else:
            self.config = self._create_empty_config()
            if json_file_path:
                self.logger.info(f"JSON file {json_file_path} not found, starting with empty configuration")
            else:
                self.logger.info("Starting with empty configuration (no file specified)")



    def _select_all(self) -> None:
        """Select all clusters, buildings, systems, controllers, disturbances, and environment.
        
        Useful when loading an existing configuration that you want to modify
        without manually selecting each component.
        """
        # Select all clusters
        for cluster_id in self.config.get("clusters", {}):
            self.select_cluster(cluster_id)
        
        # Select all buildings
        building_ids = list(self.config.get("buildings", {}).keys())
        if building_ids:
            self.select_buildings(building_ids)
        
        # Select all systems
        system_ids = list(self.config.get("systems", {}).keys())
        if system_ids:
            self.select_systems(system_ids)
        
        # Rebuild system-building mappings from loaded config
        for system_id, system_config in self.config.get("systems", {}).items():
            supported_buildings = system_config.get("supported_buildings", [])
            if supported_buildings:
                self.system_building_map[system_id] = list(supported_buildings)
                domain = self._get_domain_from_system_type(system_config["system_type"])
                for building_id in supported_buildings:
                    if building_id not in self.building_system_map:
                        self.building_system_map[building_id] = {}
                    self.building_system_map[building_id][domain] = system_id
        
        # Select controllers and assign to systems
        for controller_id, ctrl_config in self.config.get("controllers", {}).items():
            system_id = ctrl_config.get("system_id")
            if system_id and system_id in self.selected_systems:
                self.select_controller_for_system(system_id, controller_id)
        
        # Select all disturbances
        disturbance_names = list(self.config.get("disturbances", {}).keys())
        if disturbance_names:
            self.select_disturbances(disturbance_names)
        
        # Select environment
        self.select_environment()
        
        self.logger.info("Auto-selected all items from loaded configuration")

    # Cluster Management
    def add_cluster(self, cluster_id: str,
                    *, parameters: Optional[Dict[str, Any]] = None,
                    overwrite: bool = True) -> None:
        """Add a cluster to the configuration."""
        clusters = self.config.setdefault("clusters", {})

        if not overwrite and cluster_id in clusters:
            self.logger.info(f"Cluster {cluster_id} exists, skipping")
            return

        clusters[cluster_id] = {
            "parameters": parameters or {},
            "buildings": [],
            "systems": []
        }
        self.logger.info(f"Added cluster: {cluster_id}")

    def select_cluster(self, cluster_id: str) -> None:
        """Select a cluster for simulation."""
        if cluster_id not in self.config.get("clusters", {}):
            self.logger.warning(f"Cluster {cluster_id} not found")
            return

        self.selected_clusters[cluster_id] = self._deep_copy_dict(
            self.config["clusters"][cluster_id]
        )
        self.logger.info(f"Selected cluster: {cluster_id}")

    # Building Management
    def add_building(self, cluster_id: str, building_id: str,
                     *, parameters: Optional[Dict[str, Any]] = None,
                     thermal_zones: Optional[List[str]] = None,
                     electrical_zones: Optional[List[str]] = None,
                     overwrite: bool = True) -> None:
        """Add a building to a cluster."""
        clusters = self.config.setdefault("clusters", {})
        if cluster_id not in clusters:
            self.logger.warning(f"Cluster {cluster_id} not found, creating it")
            self.add_cluster(cluster_id)

        buildings = self.config.setdefault("buildings", {})

        if not overwrite and building_id in buildings:
            self.logger.info(f"Building {building_id} exists, skipping")
            return

        buildings[building_id] = {
            "cluster_id": cluster_id,
            "parameters": parameters or {},
            "thermal_zones": thermal_zones or ["zone0"],
            "electrical_zones": electrical_zones or ["zone0"],
            # Default single zone
        }

        # Add building to cluster
        if building_id not in clusters[cluster_id]["buildings"]:
            clusters[cluster_id]["buildings"].append(building_id)

        self.logger.info(f"Added building {building_id} to cluster {cluster_id}")

    def select_buildings(self, building_ids: List[str]) -> None:
        """Select buildings for simulation."""
        available_buildings = self.config.get('buildings', {})

        for building_id in building_ids:
            if building_id in available_buildings:
                self.selected_buildings[building_id] = self._deep_copy_dict(
                    available_buildings[building_id]
                )
                self.logger.info(f"Selected building: {building_id}")
            else:
                self.logger.warning(f"Building '{building_id}' not found in configuration")

    def add_thermal_zone_module(self, building_id: str, zone_id: str,
                                *, parameters: Optional[Dict[str, Any]] = None,
                                class_path: Optional[str] = None,
                                dataclass_obj: Optional[object] = None,
                                overwrite: bool = True) -> None:
        """Add thermal zone module configuration for a building."""
        zone_modules = self.config.setdefault("thermal_zone_modules", {})
        zone_key = f"{building_id}.{zone_id}"

        if not overwrite and zone_key in zone_modules:
            self.logger.info(f"Thermal zone module {zone_key} exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        zone_modules[zone_key] = entry
        self.logger.info(f"Added thermal zone module: {zone_key}")

    def add_electrical_zone_module(self, building_id: str, zone_id: str,
                                *, parameters: Optional[Dict[str, Any]] = None,
                                class_path: Optional[str] = None,
                                dataclass_obj: Optional[object] = None,
                                overwrite: bool = True) -> None:
        """Add electrical zone module configuration for a building."""
        zone_modules = self.config.setdefault("electrical_zone_modules", {})
        zone_key = f"{building_id}.{zone_id}"

        if not overwrite and zone_key in zone_modules:
            self.logger.info(f"Electrical zone module {zone_key} exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        zone_modules[zone_key] = entry
        self.logger.info(f"Added Electrical zone module: {zone_key}")

    def add_water_zone_module(self, building_id: str, zone_id: str,
                                *, parameters: Optional[Dict[str, Any]] = None,
                                class_path: Optional[str] = None,
                                dataclass_obj: Optional[object] = None,
                                overwrite: bool = True) -> None:
        pass

    # System Management
    def add_system(self, cluster_id: str, system_id: str, system_type: SystemType,
                   *, parameters: Optional[Dict[str, Any]] = None,
                   components: Optional[Dict[str, Dict[str, Any]]] = None,
                   class_path: Optional[str] = None,
                   overwrite: bool = True) -> None:
        """Add a system to a cluster."""
        clusters = self.config.setdefault("clusters", {})
        if cluster_id not in clusters:
            self.logger.warning(f"Cluster {cluster_id} not found, creating it")
            self.add_cluster(cluster_id)

        systems = self.config.setdefault("systems", {})

        if not overwrite and system_id in systems:
            self.logger.info(f"System {system_id} exists, skipping")
            return

        systems[system_id] = {
            "cluster_id": cluster_id,
            "system_type": system_type,
            "parameters": parameters or {},
            "components": components or {},
            "class_path": class_path,
            "supported_buildings": []  # Will be populated when assigning
        }

        # Add system to cluster
        if system_id not in clusters[cluster_id]["systems"]:
            clusters[cluster_id]["systems"].append(system_id)

        self.logger.info(f"Added {system_type} system {system_id} to cluster {cluster_id}")

    def assign_system_to_buildings(self, system_id: str, building_ids: List[str]) -> None:
        """Assign a system to support one or more buildings."""
        systems = self.config.get("systems", {})
        if system_id not in systems:
            self.logger.warning(f"System {system_id} not found")
            return

        system = systems[system_id]
        system_type = system["system_type"]
        domain = self._get_domain_from_system_type(system_type)

        for building_id in building_ids:
            if building_id not in self.config.get("buildings", {}):
                self.logger.warning(f"Building {building_id} not found")
                continue

            # Update system's supported buildings
            if building_id not in system["supported_buildings"]:
                system["supported_buildings"].append(building_id)

            # Track mapping
            if system_id not in self.system_building_map:
                self.system_building_map[system_id] = []
            if building_id not in self.system_building_map[system_id]:
                self.system_building_map[system_id].append(building_id)

            if building_id not in self.building_system_map:
                self.building_system_map[building_id] = {}
            self.building_system_map[building_id][domain] = system_id

            self.logger.info(f"Assigned {system_id} to support building {building_id}")

    def select_systems(self, system_ids: List[str]) -> None:
        """Select systems for simulation."""
        available_systems = self.config.get('systems', {})

        for system_id in system_ids:
            if system_id in available_systems:
                self.selected_systems[system_id] = self._deep_copy_dict(
                    available_systems[system_id]
                )
                self.logger.info(f"Selected system: {system_id}")
            else:
                self.logger.warning(f"System '{system_id}' not found")

    # Controller Management
    def add_system_controller(self, controller_id: str, system_id: str,
                              *, parameters: Optional[Dict[str, Any]] = None,
                              class_path: Optional[str] = None,
                              dataclass_obj: Optional[object] = None,
                              overwrite: bool = True) -> None:
        """Add a controller for a system."""
        controllers = self.config.setdefault("controllers", {})

        if not overwrite and controller_id in controllers:
            self.logger.info(f"Controller {controller_id} exists, skipping")
            return

        # Get system info
        systems = self.config.get("systems", {})
        if system_id not in systems:
            self.logger.warning(f"System {system_id} not found")
            return

        system = systems[system_id]
        domain = self._get_domain_from_system_type(system["system_type"])

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        controllers[controller_id] = {
            "system_id": system_id,
            "domain": domain,
            "parameters": merged_params,
            "class_path": class_path
        }

        self.logger.info(f"Added controller {controller_id} for system {system_id}")

    def select_controller_for_system(self, system_id: str, controller_id: str) -> None:
        """Assign a controller to a system."""
        if system_id not in self.selected_systems:
            self.logger.warning(f"System {system_id} not selected")
            return

        if controller_id not in self.config.get('controllers', {}):
            self.logger.warning(f"Controller {controller_id} not available")
            return

        controller_config = self._deep_copy_dict(self.config['controllers'][controller_id])
        self.selected_controllers[controller_id] = controller_config
        self.selected_systems[system_id]['controller_id'] = controller_id

        self.logger.info(f"Assigned controller {controller_id} to system {system_id}")

    # Disturbances & Environment
    def add_disturbance(self, disturbance_name: str,
                        *, parameters: Optional[Dict[str, Any]] = None,
                        class_path: Optional[str] = None,
                        dataclass_obj: Optional[object] = None,
                        overwrite: bool = True) -> None:
        """Add a new disturbance."""
        disturbances = self.config.setdefault("disturbances", {})

        if not overwrite and disturbance_name in disturbances:
            self.logger.info(f"Disturbance {disturbance_name} exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        disturbances[disturbance_name] = entry
        self.logger.info(f"Added disturbance {disturbance_name}")

    def select_disturbances(self, disturbance_names: List[str]) -> None:
        """Select disturbances for simulation."""
        available_disturbances = self.config.get('disturbances', {})

        for name in disturbance_names:
            if name in available_disturbances:
                self.selected_disturbances[name] = self._deep_copy_dict(
                    available_disturbances[name]
                )
                self.logger.info(f"Selected disturbance: {name}")

    def add_environment(self, *, parameters: Optional[Dict[str, Any]] = None,
                        class_path: Optional[str] = None,
                        dataclass_obj: Optional[object] = None,
                        overwrite: bool = True) -> None:
        """Add environment configuration."""
        if not overwrite and "environment" in self.config:
            self.logger.info("Environment exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        self.config["environment"] = entry
        self.logger.info("Added environment")

    def select_environment(self) -> None:
        """Select environment configuration."""
        env_config = self.config.get('environment', {})
        if env_config:
            self.selected_environment = self._deep_copy_dict(env_config)
            self.logger.info("Environment configuration selected")

    # Configuration Export
    def get_final_configuration(self) -> Dict[str, Any]:
        """Generate final configuration for the environment."""
        # Collect thermal zone modules for selected buildings
        selected_thermal_zones = {}
        selected_electrical_zones = {}
        thermal_zone_modules = self.config.get("thermal_zone_modules", {})
        electrical_zone_modules = self.config.get("electrical_zone_modules", {})

        for building_id in self.selected_buildings:
            building_config = self.selected_buildings[building_id]
            for zone_id in building_config.get("thermal_zones", []):
                zone_key = f"{building_id}.{zone_id}"
                if zone_key in thermal_zone_modules:
                    selected_thermal_zones[zone_key] = self._deep_copy_dict(
                        thermal_zone_modules[zone_key]
                    )
            for zone_id in building_config.get("electrical_zones", []):
                zone_key = f"{building_id}.{zone_id}"
                if zone_key in electrical_zone_modules:
                    selected_electrical_zones[zone_key] = self._deep_copy_dict(
                        electrical_zone_modules[zone_key]
                    )

        return {
            "clusters": self.selected_clusters,
            "buildings": self.selected_buildings,
            "systems": self.selected_systems,
            "controllers": self.selected_controllers,
            "thermal_zone_modules": selected_thermal_zones,
            "electrical_zone_modules": selected_electrical_zones,
            "system_building_map": self.system_building_map,
            "building_system_map": self.building_system_map,
            "disturbances": self.selected_disturbances,
            "environment": self.selected_environment
        }

    def validate_configuration(self) -> List[str]:
        """Validate the current configuration."""
        warnings = []

        # Check clusters
        if not self.selected_clusters:
            warnings.append("No clusters selected")

        # Check buildings
        if not self.selected_buildings:
            warnings.append("No buildings selected")

        # Check systems
        if not self.selected_systems:
            warnings.append("No systems selected")

        # Check that each building has necessary systems
        for building_id in self.selected_buildings:
            if building_id not in self.building_system_map:
                warnings.append(f"Building {building_id} has no assigned systems")
            else:
                assigned_domains = set(self.building_system_map[building_id].keys())
                required_domains = {"thermal", "electrical"}  # water is optional
                missing_domains = required_domains - assigned_domains
                for domain in missing_domains:
                    warnings.append(f"Building {building_id} missing {domain} system")

        # Check that each system has a controller
        for system_id, system_config in self.selected_systems.items():
            if 'controller_id' not in system_config:
                warnings.append(f"System {system_id} has no assigned controller")

        # Check environment
        if not self.selected_environment:
            warnings.append("Environment not selected")

        return warnings

    def print_summary(self) -> None:
        """Print configuration summary."""
        print(f"\n{'=' * 50}")
        print("Configuration Summary")
        print(f"{'=' * 50}")

        # Clusters
        print(f"\nClusters ({len(self.selected_clusters)}):")
        for cluster_id in self.selected_clusters:
            print(f"  - {cluster_id}")

        # Buildings
        print(f"\nBuildings ({len(self.selected_buildings)}):")
        for building_id, building_config in self.selected_buildings.items():
            cluster_id = building_config.get("cluster_id", "unknown")
            zones = building_config.get("thermal_zones", [])
            print(f"  - {building_id} (cluster: {cluster_id}, zones: {zones})")

            # Show assigned systems
            if building_id in self.building_system_map:
                for domain, system_id in self.building_system_map[building_id].items():
                    print(f"    └─ {domain}: {system_id}")

        # Systems
        print(f"\nSystems ({len(self.selected_systems)}):")
        for system_id, system_config in self.selected_systems.items():
            system_type = system_config.get("system_type")
            supported = system_config.get("supported_buildings", [])
            controller_id = system_config.get("controller_id", "none")
            print(f"  - {system_id} ({system_type})")
            print(f"    ├─ Controller: {controller_id}")
            print(f"    └─ Supporting: {supported}")

        # Disturbances
        print(f"\nDisturbances: {list(self.selected_disturbances.keys())}")

        # Environment
        print(f"Environment: {'Selected' if self.selected_environment else 'Not selected'}")

        # Validation
        warnings = self.validate_configuration()
        if warnings:
            print(f"\n⚠ Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("\n✓ Configuration validation passed")

        print(f"{'=' * 50}\n")

    def save_final_configuration(self, filepath: str) -> None:
        """Save final configuration to JSON file."""
        final_config = self.get_final_configuration()
        with open(filepath, 'w') as f:
            json.dump(final_config, f, indent=2)
        self.logger.info(f"Saved configuration to {filepath}")

    # Helper Methods
    def _get_domain_from_system_type(self, system_type: SystemType) -> DomainType:
        """Map system type to domain."""
        mapping = {
            "hvac_systems": "thermal",
            "der_systems": "electrical",
            "water_systems": "water"
        }
        return mapping.get(system_type, "unknown")

    def _load_json(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _create_empty_config(self) -> Dict[str, Any]:
        """Create an empty configuration structure."""
        return {
            "clusters": {},
            "buildings": {},
            "systems": {},
            "controllers": {},
            "thermal_zone_modules": {},
            "disturbances": {},
            "environment": {}
        }

    @staticmethod
    def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a dictionary."""
        return copy.deepcopy(d)

    @staticmethod
    def _merge_params(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge parameters with overrides."""
        result = dict(base or {})
        if override:
            result.update(override)
        return result

    def _dataclass_to_dict(self, dataclass_obj: Optional[object]) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        if dataclass_obj is None:
            return {}
        try:
            if is_dataclass(dataclass_obj):
                return asdict(dataclass_obj)
            self.logger.warning("Provided dataclass_obj is not a dataclass; ignored.")
        except Exception as e:
            self.logger.warning(f"Failed to read dataclass: {e}")
        return {}