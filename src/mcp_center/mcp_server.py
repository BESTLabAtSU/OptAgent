"""
MCP Server for BESTOpt Building-to-Grid Platform
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from pathlib import Path
import logging
from datetime import datetime

from pydantic import Field
from mcp.server.fastmcp import FastMCP

import sys
from pathlib import Path

import json

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.env.core.simulation_manager import SimulationManager

# Initialize FastMCP server with metadata
mcp = FastMCP(
    "bestopt-mcp-server",
    instructions="BESTOpt Building-to-Grid Platform MCP Server for building energy simulation and optimization",
)


class ToolResponse(TypedDict):
    success: bool
    data: Optional[Dict]
    error: Optional[str]
    message: Optional[str]


class BESTOptServerState:
    """Server State Management with Auto-Initialization"""

    DEFAULT_CONFIG_NAME = "default_config"
    DEFAULT_CONFIG_PATH = "./bestopt_workspace/configs/default.json"

    def __init__(self, base_path: str = "./bestopt_workspace"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.configs_path = self.base_path / "configs"
        self.configs_path.mkdir(exist_ok=True)
        self.results_path = self.base_path / "results"
        self.results_path.mkdir(exist_ok=True)

        self.config_managers: Dict[str, ConfigurationManager] = {}
        self.active_config: Optional[str] = None
        self.environments: Dict[str, BESTOptEnvironment] = {}
        self.simulation_results: Dict[str, Dict] = {}
        self.simulation_managers: Dict[str, SimulationManager] = {}
        self.active_simulations: Dict[str, Dict] = {}
        self.simulation_manager = SimulationManager(str(self.results_path))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BESTOptMCPServer")

        # Auto-initialize default configuration
        self._auto_initialize_config()

    def _auto_initialize_config(self):
        """Automatically initialize a default configuration on startup"""
        try:
            config_path = Path(self.DEFAULT_CONFIG_PATH)

            # Load the configuration
            self.config_managers[self.DEFAULT_CONFIG_NAME] = ConfigurationManager(
                str(config_path)
            )
            self.active_config = self.DEFAULT_CONFIG_NAME

            self.logger.info(
                f"Auto-initialized default configuration '{self.DEFAULT_CONFIG_NAME}' "
                f"from {config_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to auto-initialize configuration: {e}")

    def ensure_active_config(self) -> bool:
        """
        Ensure there's an active configuration.
        Returns True if active config exists, False otherwise.
        """
        if self.active_config and self.active_config in self.config_managers:
            return True

        # Try to re-initialize if somehow lost
        if not self.active_config:
            self._auto_initialize_config()

        return self.active_config is not None

    def get_active_config_manager(self) -> Optional[ConfigurationManager]:
        """Get the active configuration manager, with auto-recovery"""
        if self.ensure_active_config():
            return self.config_managers.get(self.active_config)
        return None


# Initialize global state (this triggers auto-initialization)
state = BESTOptServerState()


# ============= CONFIGURATION MANAGEMENT TOOLS =============

@mcp.tool(
    description="""Initialize a new named configuration instance in memory.
    This is the ONLY tool that creates a new configuration object.
    Use when the user wants to 'create', 'initialize', or 'start' a new configuration.
    """
)
async def config_create(
    name: Annotated[Optional[str], Field(description="Unique name for this configuration")] = None,
    from_file: Annotated[Optional[str], Field(description="Path to existing configuration JSON file to load")] = None,
) -> Dict[str, Any]:
    """
    Creates a new configuration manager instance or loads from an existing file.
    """
    try:
        if not name:
            name = "config0"

        if not from_file:
            from_file = "./bestopt_workspace/configs/default.json"

        state.config_managers[name] = ConfigurationManager(from_file)
        state.active_config = name

        return {
            "success": True,
            "data": {"config_name": name, "from_file": from_file},
            "message": f"Configuration '{name}' loaded from {from_file}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(
    description="Save current active configuration to a JSON file. If a new name is provided, creates a copy and optionally sets it as active.")
async def config_save(
        name: Annotated[
            Optional[str],
            Field(description="Name for the saved configuration. If different from active, creates a copy."),
        ] = None,
        set_active: Annotated[
            bool,
            Field(description="If True and name is new, set the saved config as active"),
        ] = False,
) -> Dict[str, Any]:
    """Saves the current active configuration to a JSON file."""
    cm = state.config_managers[state.active_config]

    if not name:
        name = state.active_config

    filepath = state.configs_path / f"{name}.json"

    try:
        with open(filepath, 'w') as f:
            json.dump(cm.config, f, indent=2)

        result_data = {"saved_to": str(filepath), "config_name": name}

        # KEY ADDITION: If saving with a NEW name, register it as a config
        if name != state.active_config and name not in state.config_managers:
            state.config_managers[name] = ConfigurationManager(str(filepath))
            result_data["created_new_config"] = True

            if set_active:
                state.active_config = name
                result_data["now_active"] = name

        return {
            "success": True,
            "data": result_data,
            "message": f"Configuration saved as '{name}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Validate the completeness and correctness of current configuration")
async def config_validate() -> Dict[str, Any]:
    """Validates the current active configuration for completeness and consistency."""
    cm = state.config_managers[state.active_config]
    warnings = cm.validate_configuration()

    return {
        "success": True,
        "data": {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "warning_count": len(warnings),
        },
        "message": "Configuration is valid" if len(warnings) == 0
                   else f"Configuration has {len(warnings)} warnings",
    }


@mcp.tool(description="Switch the active configuration to a different one that already exists. Use only when multiple configurations have been created and you need to switch between them.")
async def config_set_active(
    name: Annotated[str, Field(description="Name of the configuration to make active")]
) -> Dict[str, Any]:
    """Sets the specified configuration as the active one for operations."""
    if name not in state.config_managers:
        return {
            "success": False,
            "error": f"Configuration '{name}' not found",
            "data": {"available_configs": list(state.config_managers.keys())},
        }

    state.active_config = name
    return {
        "success": True,
        "data": {"active_config": name},
        "message": f"Configuration '{name}' is now active",
    }


@mcp.tool(description="Show detailed information about the current active configuration including all clusters, buildings, systems, controllers, and disturbances. Also shows which configuration is active.")
async def config_list() -> Dict[str, Any]:
    """Show detailed information about all available configurations."""
    cm = state.config_managers[state.active_config]
    info = cm.get_final_configuration()
    return {
        "success": True,
        "data": {
            "information": info,
            "active": state.active_config,
            "auto_initialized": state.DEFAULT_CONFIG_NAME in state.config_managers,
        },
    }


# ============= CLUSTER MANAGEMENT TOOLS =============

@mcp.tool(description="Add a new cluster to the active configuration")
async def cluster_add(
        cluster_id: Annotated[str, Field(description="Unique identifier for the cluster")],
        parameters: Annotated[Optional[Dict], Field(description="Additional cluster parameters")] = None,
        overwrite: Annotated[bool, Field(description="If True, overwrite existing cluster with same ID")] = True,
) -> Dict[str, Any]:
    """
    Adds a new cluster to the current configuration.

    Clusters are the top-level organizational unit that contain buildings and systems.
    """
    cm = state.config_managers[state.active_config]

    try:
        cm.add_cluster(
            cluster_id=cluster_id,
            parameters=parameters,
            overwrite=overwrite,
        )
        return {
            "success": True,
            "data": {
                "cluster_id": cluster_id,
                "parameters": parameters or {},
                "overwrite": overwrite,
            },
            "message": f"Cluster '{cluster_id}' added successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Query cluster information from configuration")
async def cluster_query(
        cluster_id: Annotated[Optional[str], Field(description="ID of specific cluster to query")] = None,
) -> Dict[str, Any]:
    """Queries cluster information from the current configuration."""
    cm = state.config_managers[state.active_config]
    clusters = cm.config.get("clusters", {})

    if cluster_id:
        if cluster_id in clusters:
            return {
                "success": True,
                "data": clusters[cluster_id],
                "message": f"Cluster '{cluster_id}' found",
            }
        else:
            return {
                "success": False,
                "error": f"Cluster '{cluster_id}' not found",
                "data": {"available_clusters": list(clusters.keys())},
            }

    return {
        "success": True,
        "data": {
            "clusters": clusters,
            "count": len(clusters),
            "cluster_ids": list(clusters.keys()),
        },
    }


@mcp.tool(description="Select a cluster for simulation")
async def cluster_select(
        cluster_id: Annotated[str, Field(description="ID of the cluster to select")]
) -> Dict[str, Any]:
    """Selects a cluster for the simulation."""
    cm = state.config_managers[state.active_config]

    try:
        cm.select_cluster(cluster_id)
        return {
            "success": True,
            "data": {"selected": cluster_id},
            "message": f"Cluster '{cluster_id}' selected for simulation",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Remove a cluster from configuration")
async def cluster_remove(
        cluster_id: Annotated[str, Field(description="ID of the cluster to remove")]
) -> Dict[str, Any]:
    """Removes a cluster from the configuration."""
    cm = state.config_managers[state.active_config]

    try:
        if cluster_id not in cm.config.get("clusters", {}):
            return {
                "success": False,
                "error": f"Cluster '{cluster_id}' not found",
                "data": {"available_clusters": list(cm.config.get("clusters", {}).keys())},
            }

        cluster_info = cm.config["clusters"][cluster_id]
        buildings = cluster_info.get("buildings", [])
        systems = cluster_info.get("systems", [])

        del cm.config["clusters"][cluster_id]
        if cluster_id in cm.selected_clusters:
            del cm.selected_clusters[cluster_id]

        return {
            "success": True,
            "data": {
                "removed": cluster_id,
                "orphaned_buildings": buildings,
                "orphaned_systems": systems,
            },
            "message": f"Cluster '{cluster_id}' removed successfully. Note: {len(buildings)} buildings and {len(systems)} systems may now be orphaned.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= BUILDING MANAGEMENT TOOLS =============

@mcp.tool(description="Add a new building to a cluster in the active configuration")
async def building_add(
    building_id: Annotated[str, Field(description="Unique identifier for the building")],
    cluster_id: Annotated[str, Field(description="ID of the cluster to add the building to")],
    building_type: Annotated[Optional[str], Field(description="Type of building (e.g., 'SFH', 'MFH', 'Office')")] = None,
    thermal_zones: Annotated[Optional[List[str]], Field(description="List of thermal zone identifiers")] = None,
    electrical_zones: Annotated[Optional[List[str]], Field(description="List of electrical zone identifiers")] = None,
    parameters: Annotated[Optional[Dict], Field(description="Additional building parameters")] = None,
) -> Dict[str, Any]:
    """Adds a new building to the specified cluster."""
    cm = state.config_managers[state.active_config]
    params = parameters or {}
    if building_type:
        params["building_type"] = building_type

    try:
        cm.add_building(
            cluster_id=cluster_id,
            building_id=building_id,
            parameters=params,
            thermal_zones=thermal_zones or ["zone0"],
            electrical_zones=electrical_zones or ["zone0"],
        )
        return {
            "success": True,
            "data": {
                "building_id": building_id,
                "cluster_id": cluster_id,
                "building_type": building_type,
                "thermal_zones": thermal_zones or ["zone0"],
                "electrical_zones": electrical_zones or ["zone0"],
            },
            "message": f"Building '{building_id}' added to cluster '{cluster_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Update building configuration parameters")
async def building_update(
        building_id: Annotated[Optional[str], Field(
            description="ID of the building to update. Use default if not provide")] = None,
        updates: Annotated[Dict, Field(description="Dictionary containing updates")] = None,
) -> Dict[str, Any]:
    """Updates configuration parameters for an existing building."""
    cm = state.config_managers[state.active_config]

    # Resolve building_id if not provided
    used_default = False
    if building_id is None:
        buildings = cm.config.get("buildings", {})
        if not buildings:
            return {"success": False, "error": "No buildings found in configuration"}
        building_id = next(iter(buildings.keys()))
        used_default = True

    try:
        if building_id not in cm.config.get("buildings", {}):
            return {
                "success": False,
                "error": f"Building '{building_id}' not found",
                "data": {"available_buildings": list(cm.config.get("buildings", {}).keys())}
            }

        building = cm.config["buildings"][building_id]
        if updates:
            if "parameters" in updates:
                building["parameters"].update(updates["parameters"])
            if "thermal_zones" in updates:
                building["thermal_zones"] = updates["thermal_zones"]
            if "electrical_zones" in updates:
                building["electrical_zones"] = updates["electrical_zones"]

        message = f"Building '{building_id}' updated successfully"
        if used_default:
            message = f"No specific building_id provided, using default '{building_id}'. " + message

        return {
            "success": True,
            "data": {
                "building_id": building_id,
                "updates_applied": list(updates.keys()) if updates else [],
                "used_default": used_default
            },
            "message": message,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Remove a building from the configuration")
async def building_remove(
    building_id: Annotated[str, Field(description="ID of the building to remove")]
) -> Dict[str, Any]:
    """Removes a building from the current configuration."""
    cm = state.config_managers[state.active_config]

    try:
        cluster_id = cm.config["buildings"][building_id]["cluster_id"]
        if cluster_id in cm.config["clusters"]:
            buildings = cm.config["clusters"][cluster_id]["buildings"]
            if building_id in buildings:
                buildings.remove(building_id)

        del cm.config["buildings"][building_id]
        if building_id in cm.selected_buildings:
            del cm.selected_buildings[building_id]

        return {
            "success": True,
            "data": {"removed": building_id, "from_cluster": cluster_id},
            "message": f"Building '{building_id}' removed successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Query building information from configuration")
async def building_query(
    building_id: Annotated[Optional[str], Field(description="ID of specific building to query")] = None,
) -> Dict[str, Any]:
    """Queries building information from the current configuration."""
    cm = state.config_managers[state.active_config]
    buildings = cm.config.get("buildings", {})

    if building_id:
        if building_id in buildings:
            return {
                "success": True,
                "data": buildings[building_id],
                "message": f"Building '{building_id}' found",
            }
        else:
            return {
                "success": False,
                "error": f"Building '{building_id}' not found",
                "data": {"available_buildings": list(buildings.keys())},
            }

    return {
        "success": True,
        "data": {
            "buildings": buildings,
            "count": len(buildings),
            "building_ids": list(buildings.keys()),
        },
    }


@mcp.tool(description="Select buildings for simulation")
async def building_select(
    building_ids: Annotated[List[str], Field(description="List of building IDs to select")]
) -> Dict[str, Any]:
    """Selects specified buildings for the simulation."""
    cm = state.config_managers[state.active_config]
    cm.select_buildings(building_ids)

    return {
        "success": True,
        "data": {"selected": building_ids, "count": len(building_ids)},
        "message": f"Selected {len(building_ids)} buildings for simulation",
    }


@mcp.tool(description="Add thermal zone module of a building with detailed training parameters")
async def building_add_thermal_zone(
    building_id: Annotated[str, Field(description="ID of the building")],
    zone_id: Annotated[str, Field(description="ID of the thermal zone")],
        parameters: Annotated[Optional[Dict], Field(
            description="Thermal zone parameters: {"
                        "'model_args': {"
                        "'modeltype': 'LSTM'|'GRU'|'PI-modnn', "
                        "'trainday': <int days of training data>, "
                        "'testday': <int days of test data>, "
                        "'temp_unit': 'C'|'F', "
                        "'device': 'cpu'|'cuda:0'|'cuda:1'},"
        )] = None,
) -> Dict[str, Any]:
    """Adds a thermal zone module to the specified building."""
    cm = state.config_managers[state.active_config]

    try:
        cm.add_thermal_zone_module(building_id, zone_id, parameters=parameters)
        return {
            "success": True,
            "data": {
                "building_id": building_id,
                "zone_id": zone_id,
                "zone_key": f"{building_id}.{zone_id}",
            },
            "message": f"Thermal zone '{zone_id}' added to building '{building_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Add an electrical zone module to a building")
async def building_add_electrical_zone(
    building_id: Annotated[str, Field(description="ID of the building")],
    zone_id: Annotated[str, Field(description="ID of the electrical zone")],
    parameters: Annotated[Optional[Dict], Field(description="Electrical zone parameters")] = None,
) -> Dict[str, Any]:
    """Adds an electrical zone module to the specified building."""
    cm = state.config_managers[state.active_config]

    try:
        cm.add_electrical_zone_module(building_id, zone_id, parameters=parameters)
        return {
            "success": True,
            "data": {
                "building_id": building_id,
                "zone_id": zone_id,
                "zone_key": f"{building_id}.{zone_id}",
            },
            "message": f"Electrical zone '{zone_id}' added to building '{building_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= HVAC SYSTEM TOOLS =============

@mcp.tool(description="Add a brand new HVAC system to a cluster. Use ONLY when the system does not already exist. To modify an existing HVAC system's parameters, use hvac_update instead.")
async def hvac_add(
        system_id: Annotated[str, Field(description="Unique identifier for the HVAC system")],
        cluster_id: Annotated[str, Field(description="ID of the cluster to add the HVAC system to")],
        system_name: Annotated[Optional[str], Field(
            description="Display name for the system (e.g., 'FCU System', 'Office HVAC')"
        )] = None,
        system_config: Annotated[Optional[Dict], Field(
            description="HVAC system configuration: {"
                        "'fan': {'rated_flow_m3s': <float m3/s>, 'rated_power_W': <float watts>}, "
                        "'fan_ctrl': {'ctrl_type': 'constant'|'staged'|'vfd', 'rated_flow_m3s': <float>, 'stages': <int for staged mode>}, "
                        "'coil': {'effectiveness': <float 0-1>}, "
                        "'pump': {'rated_flow_m3s': <float m3/s>, 'rated_power_W': <float watts>}, "
                        "'chiller': {'rated_capacity_W': <float watts>, 'rated_cop': <float COP>}, "
                        "'tower': {'rated_capacity_W': <float watts>, 'rated_fan_power_W': <float watts>, "
                        "'pump_power_per_flow': <float W/(m3/s)>, 'min_approach_C': <float °C>, 'max_approach_C': <float °C>}}"
        )] = None,
        parameters: Annotated[Optional[Dict], Field(
            description="Additional HVAC parameters beyond system_config"
        )] = None,
) -> Dict[str, Any]:
    """
    Adds a new HVAC system to the specified cluster.

    Example system_config for a typical residential FCU:
    {
        "fan": {"rated_flow_m3s": 0.4, "rated_power_W": 400},
        "fan_ctrl": {"ctrl_type": "constant", "rated_flow_m3s": 0.4},
        "coil": {"effectiveness": 0.7},
        "pump": {"rated_flow_m3s": 0.01, "rated_power_W": 1500},
        "chiller": {"rated_capacity_W": 15000, "rated_cop": 4.5},
        "tower": {
            "rated_capacity_W": 15000,
            "rated_fan_power_W": 400,
            "pump_power_per_flow": 85000,
            "min_approach_C": 3.0,
            "max_approach_C": 7.0
        }
    }

    Fan control types:
    - "constant": Fixed speed fan
    - "staged": Multi-stage fan (requires 'stages' parameter)
    - "vfd": Variable frequency drive for continuous speed control
    """
    cm = state.config_managers[state.active_config]
    params = parameters or {}

    if system_name:
        params["system_name"] = system_name
    if system_config:
        params["system_config"] = system_config

    try:
        cm.add_system(
            cluster_id=cluster_id,
            system_id=system_id,
            system_type="hvac_systems",
            parameters=params,
        )
        return {
            "success": True,
            "data": {
                "system_id": system_id,
                "cluster_id": cluster_id,
                "system_type": "hvac_systems",
                "system_name": system_name,
                "system_config": system_config,
            },
            "message": f"HVAC system '{system_id}' added to cluster '{cluster_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Modify parameters of an HVAC system that already exists. Supports partial updates — only specified fields are changed. Do NOT use this to create a new system; use hvac_add for that.")
async def hvac_update(
        system_id: Annotated[
            Optional[str], Field(description="ID of the HVAC system to update. Use default if not provide")] = None,
        updates: Annotated[Dict, Field(
            description="Updates dictionary. MUST have one of these top-level keys."
                        "- 'system_config': for component updates."
                        "- 'parameters': for general parameter updates."
                        "Structure: {"
                        "'parameters': {<general params to update>}, "
                        "'system_config': {"
                        "'fan': {'rated_flow_m3s': <float>, 'rated_power_W': <float>}, "
                        "'fan_ctrl': {'ctrl_type': 'constant'|'staged'|'vfd', 'rated_flow_m3s': <float>, 'stages': <int>}, "
                        "'coil': {'effectiveness': <float 0-1>}, "
                        "'pump': {'rated_flow_m3s': <float>, 'rated_power_W': <float>}, "
                        "'chiller': {'rated_capacity_W': <float>, 'rated_cop': <float>}, "
                        "'tower': {'rated_capacity_W': <float>, 'rated_fan_power_W': <float>, "
                        "'pump_power_per_flow': <float>, 'min_approach_C': <float>, 'max_approach_C': <float>}"
                        "}} - Only include fields you want to update"
        )] = None,
) -> Dict[str, Any]:
    """
    Updates configuration for an existing HVAC system with DEEP MERGE support.
    Partial updates are supported - only specified fields are changed.
    """
    cm = state.config_managers[state.active_config]
    # Resolve system_id if not provided
    used_default = False
    if system_id is None:
        hvac_systems = {sid: s for sid, s in cm.config.get("systems", {}).items()
                        if s.get("system_type") == "hvac_systems"}
        if not hvac_systems:
            return {"success": False, "error": "No HVAC systems found in configuration"}
        system_id = next(iter(hvac_systems.keys()))
        used_default = True

    try:
        if system_id not in cm.config.get("systems", {}):
            return {"success": False, "error": f"System '{system_id}' not found"}

        system = cm.config["systems"][system_id]
        if system["system_type"] != "hvac_systems":
            return {"success": False, "error": f"System '{system_id}' is not an HVAC system"}

        if updates:
            # Handle general parameters
            if "parameters" in updates:
                system["parameters"].update(updates["parameters"])

            # Handle system_config with DEEP MERGE
            if "system_config" in updates:
                if "system_config" not in system["parameters"]:
                    system["parameters"]["system_config"] = {}

                existing_config = system["parameters"]["system_config"]
                new_config = updates["system_config"]

                # Deep merge each component (fan, coil, chiller, etc.)
                for component, config in new_config.items():
                    if component in existing_config and isinstance(existing_config[component], dict):
                        # Deep merge for existing dict components
                        existing_config[component].update(config)
                    else:
                        # Add new component
                        existing_config[component] = config

        message = f"HVAC system '{system_id}' updated successfully"
        if used_default:
            message = f"No specific system_id provided, using default '{system_id}'. " + message

        return {
            "success": True,
            "data": {
                "system_id": system_id,
                "updates_applied": list(updates.keys()) if updates else [],
                "used_default": used_default
            },
            "message": message,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Delete an existing HVAC system from the configuration. Use when the user wants to remove or delete an HVAC system.")
async def hvac_remove(
    system_id: Annotated[str, Field(description="ID of the HVAC system to remove")]
) -> Dict[str, Any]:
    """Removes an HVAC system from the configuration."""
    cm = state.config_managers[state.active_config]

    try:
        if cm.config["systems"][system_id]["system_type"] != "hvac_systems":
            return {"success": False, "error": f"System '{system_id}' is not an HVAC system"}

        cluster_id = cm.config["systems"][system_id]["cluster_id"]
        if cluster_id in cm.config["clusters"]:
            systems = cm.config["clusters"][cluster_id]["systems"]
            if system_id in systems:
                systems.remove(system_id)

        del cm.config["systems"][system_id]
        if system_id in cm.selected_systems:
            del cm.selected_systems[system_id]

        return {
            "success": True,
            "data": {"removed": system_id, "from_cluster": cluster_id},
            "message": f"HVAC system '{system_id}' removed successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Retrieve information about HVAC systems. If system_id is provided, returns details for that specific system. Otherwise returns all HVAC systems. This is a read-only operation.")
async def hvac_query(
    system_id: Annotated[Optional[str], Field(description="ID of specific HVAC system")] = None,
) -> Dict[str, Any]:
    """Queries HVAC system information from the configuration."""
    cm = state.config_managers[state.active_config]
    systems = cm.config.get("systems", {})

    if system_id:
        system = systems.get(system_id, {})
        if not system:
            return {"success": False, "error": f"System '{system_id}' not found"}
        if system.get("system_type") != "hvac_systems":
            return {"success": False, "error": f"System '{system_id}' is not an HVAC system"}
        return {"success": True, "data": system, "message": f"HVAC system '{system_id}' found"}

    hvac_systems = {sid: s for sid, s in systems.items() if s.get("system_type") == "hvac_systems"}
    return {
        "success": True,
        "data": {"hvac_systems": hvac_systems, "count": len(hvac_systems), "system_ids": list(hvac_systems.keys())},
    }


@mcp.tool(description="Assign HVAC system to serve specific buildings")
async def hvac_assign_to_buildings(
    system_id: Annotated[str, Field(description="ID of the HVAC system")],
    building_ids: Annotated[List[str], Field(description="List of building IDs")],
) -> Dict[str, Any]:
    """Assigns an HVAC system to serve specified buildings."""
    cm = state.config_managers[state.active_config]

    try:
        cm.assign_system_to_buildings(system_id, building_ids)
        return {
            "success": True,
            "data": {"system_id": system_id, "buildings": building_ids, "building_count": len(building_ids)},
            "message": f"HVAC system '{system_id}' assigned to {len(building_ids)} buildings",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Select HVAC systems for simulation")
async def hvac_select(
    system_ids: Annotated[List[str], Field(description="List of HVAC system IDs to select")]
) -> Dict[str, Any]:
    """Selects specified HVAC systems for the simulation."""
    cm = state.config_managers[state.active_config]
    cm.select_systems(system_ids)

    return {
        "success": True,
        "data": {"selected": system_ids, "count": len(system_ids)},
        "message": f"Selected {len(system_ids)} HVAC systems for simulation",
    }


# ============= DER SYSTEM TOOLS =============

@mcp.tool(description="Add a brand new DER (Distributed Energy Resource) system to a cluster. Use ONLY when the system does not already exist. To modify an existing DER system, use der_update instead.")
async def der_add(
        system_id: Annotated[str, Field(description="Unique identifier for the DER system")],
        cluster_id: Annotated[str, Field(description="ID of the cluster to add the DER system to")],
        system_name: Annotated[Optional[str], Field(
            description="Display name for the system (e.g., 'PV-Battery-EV System')"
        )] = None,
        system_config: Annotated[Optional[Dict], Field(
            description="DER system configuration: {"
                        "'pv': {'rated_capacity_kW': <float kW>}, "
                        "'bat': {'rated_capacity_kWh': <float kWh>, 'initial_soc': <float 0-1>, "
                        "'charge_speed': <float 0-1>, 'discharge_speed': <float 0-1>, 'charge_efficiency': <float 0-1>}, "
                        "'evs': [{'id': <string>, 'rated_capacity_kWh': <float>, 'initial_soc': <float 0-1>, "
                        "'charge_speed': <float>, 'discharge_speed': <float>, 'charge_efficiency': <float>, "
                        "'initially_connected': <bool>}, ...]}"
        )] = None,
        parameters: Annotated[Optional[Dict], Field(
            description="Additional DER parameters beyond system_config"
        )] = None,
) -> Dict[str, Any]:
    """
    Adds a new DER system to the specified cluster.

    Example system_config for a residential PV-Battery-EV system:
    {
        "pv": {"rated_capacity_kW": 20},
        "bat": {
            "rated_capacity_kWh": 15,
            "initial_soc": 0.3,
            "charge_speed": 0.25,
            "discharge_speed": 0.5,
            "charge_efficiency": 0.95
        },
        "evs": [
            {
                "id": "ev_tesla",
                "rated_capacity_kWh": 60,
                "initial_soc": 0.2,
                "charge_speed": 0.25,
                "discharge_speed": 0.5,
                "charge_efficiency": 0.95,
                "initially_connected": True
            },
            {
                "id": "ev_nissan",
                "rated_capacity_kWh": 40,
                "initial_soc": 0.8,
                "charge_speed": 0.25,
                "discharge_speed": 0.5,
                "charge_efficiency": 0.95,
                "initially_connected": False
            }
        ]
    }
    """
    cm = state.config_managers[state.active_config]
    params = parameters or {}

    if system_name:
        params["system_name"] = system_name
    if system_config:
        params["system_config"] = system_config

    try:
        cm.add_system(
            cluster_id=cluster_id,
            system_id=system_id,
            system_type="der_systems",
            parameters=params,
        )
        return {
            "success": True,
            "data": {
                "system_id": system_id,
                "cluster_id": cluster_id,
                "system_type": "der_systems",
                "system_name": system_name,
                "system_config": system_config,
            },
            "message": f"DER system '{system_id}' added to cluster '{cluster_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Modify parameters of a DER system that already exists. Supports deep merge for component updates (PV, battery, EVs). Do NOT use this to create a new system; use der_add for that.")
async def der_update(
        system_id: Annotated[
            Optional[str], Field(description="ID of the DER system to update. Use default if not provide")] = None,
        updates: Annotated[Dict, Field(
            description="Updates dictionary. MUST have one of these top-level keys."
                        "- 'system_config': for component updates."
                        "- 'parameters': for general parameter updates."
                        "Structure: {"
                        "'parameters': {<general params to update>}, "
                        "'system_config': {"
                        "'pv': {'rated_capacity_kW': <float>}, "
                        "'bat': {'rated_capacity_kWh': <float>, 'initial_soc': <float 0-1>, "
                        "'charge_speed': <float>, 'discharge_speed': <float>, 'charge_efficiency': <float>}, "
                        "'evs': [{'id': <str>, 'rated_capacity_kWh': <float>, 'initial_soc': <float>, ...}]"
                        "}} - Only include fields you want to update"
        )] = None,
) -> Dict[str, Any]:
    """
    Updates configuration for an existing DER system with DEEP MERGE support.
    Partial updates are supported - only specified fields are changed.
    """
    cm = state.config_managers[state.active_config]

    # Resolve system_id if not provided
    used_default = False
    if system_id is None:
        der_systems = {sid: s for sid, s in cm.config.get("systems", {}).items()
                       if s.get("system_type") == "der_systems"}
        if not der_systems:
            return {"success": False, "error": "No DER systems found in configuration"}
        system_id = next(iter(der_systems.keys()))
        used_default = True

    try:
        if system_id not in cm.config.get("systems", {}):
            return {"success": False, "error": f"System '{system_id}' not found"}

        system = cm.config["systems"][system_id]
        if system["system_type"] != "der_systems":
            return {"success": False, "error": f"System '{system_id}' is not a DER system"}

        # Update system parameters
        if updates:
            # Handle general parameters (shallow merge is OK here)
            if "parameters" in updates:
                system["parameters"].update(updates["parameters"])

            # Handle system_config with DEEP MERGE
            if "system_config" in updates:
                if "system_config" not in system["parameters"]:
                    system["parameters"]["system_config"] = {}

                existing_config = system["parameters"]["system_config"]
                new_config = updates["system_config"]

                # Deep merge each component
                for component, config in new_config.items():
                    if component == "evs":
                        # EVs is a list - replace entirely or merge by id
                        existing_config["evs"] = config
                    elif component in existing_config and isinstance(existing_config[component], dict):
                        # Deep merge for dict components (pv, bat, etc.)
                        existing_config[component].update(config)
                    else:
                        # Add new component
                        existing_config[component] = config

        # Sync to associated controller(s)
        controllers_updated = []
        if updates and "system_config" in updates:
            for ctrl_id, ctrl in cm.config.get("controllers", {}).items():
                if ctrl.get("system_id") == system_id:
                    if "system_config" not in ctrl["parameters"]:
                        ctrl["parameters"]["system_config"] = {}

                    ctrl_sys_config = ctrl["parameters"]["system_config"]
                    update_sys_config = updates["system_config"]

                    # Deep merge for controller sync too
                    if "pv" in update_sys_config:
                        if "pv" not in ctrl_sys_config:
                            ctrl_sys_config["pv"] = {}
                        ctrl_sys_config["pv"].update(update_sys_config["pv"])

                    if "bat" in update_sys_config:
                        if "bat" not in ctrl_sys_config:
                            ctrl_sys_config["bat"] = {}
                        ctrl_sys_config["bat"].update(update_sys_config["bat"])

                    if "evs" in update_sys_config and update_sys_config["evs"]:
                        if "ev" not in ctrl_sys_config:
                            ctrl_sys_config["ev"] = {}
                        evs = update_sys_config["evs"]
                        total_capacity = sum(ev.get("rated_capacity_kWh", 0) for ev in evs)
                        avg_soc = sum(ev.get("initial_soc", 0) for ev in evs) / len(evs)
                        ctrl_sys_config["ev"]["rated_capacity_kWh"] = total_capacity
                        ctrl_sys_config["ev"]["initial_soc"] = round(avg_soc, 2)

                    controllers_updated.append(ctrl_id)

        message = f"DER system '{system_id}' updated successfully"
        if controllers_updated:
            message += f", synced to controller(s): {controllers_updated}"
        if used_default:
            message = f"No specific system_id provided, using default '{system_id}'. " + message

        return {
            "success": True,
            "data": {
                "system_id": system_id,
                "updates_applied": list(updates.keys()) if updates else [],
                "controllers_synced": controllers_updated,
                "used_default": used_default
            },
            "message": message,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Delete an existing DER system from the configuration. Use when the user wants to remove or delete a DER system.")
async def der_remove(
    system_id: Annotated[str, Field(description="ID of the DER system to remove")]
) -> Dict[str, Any]:
    """Removes a DER system from the configuration."""
    cm = state.config_managers[state.active_config]

    try:
        if cm.config["systems"][system_id]["system_type"] != "der_systems":
            return {"success": False, "error": f"System '{system_id}' is not a DER system"}

        cluster_id = cm.config["systems"][system_id]["cluster_id"]
        if cluster_id in cm.config["clusters"]:
            systems = cm.config["clusters"][cluster_id]["systems"]
            if system_id in systems:
                systems.remove(system_id)

        del cm.config["systems"][system_id]
        if system_id in cm.selected_systems:
            del cm.selected_systems[system_id]

        return {
            "success": True,
            "data": {"removed": system_id, "from_cluster": cluster_id},
            "message": f"DER system '{system_id}' removed successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Retrieve information about DER systems. If system_id is provided, returns details for that specific system. Otherwise returns all DER systems. This is a read-only operation.")
async def der_query(
    system_id: Annotated[Optional[str], Field(description="ID of specific DER system")] = None,
) -> Dict[str, Any]:
    """Queries DER system information from the configuration."""
    cm = state.config_managers[state.active_config]
    systems = cm.config.get("systems", {})

    if system_id:
        system = systems.get(system_id, {})
        if not system:
            return {"success": False, "error": f"System '{system_id}' not found"}
        if system.get("system_type") != "der_systems":
            return {"success": False, "error": f"System '{system_id}' is not a DER system"}
        return {"success": True, "data": system, "message": f"DER system '{system_id}' found"}

    der_systems = {sid: s for sid, s in systems.items() if s.get("system_type") == "der_systems"}
    return {
        "success": True,
        "data": {"der_systems": der_systems, "count": len(der_systems), "system_ids": list(der_systems.keys())},
    }


@mcp.tool(description="Assign DER system to serve specific buildings")
async def der_assign_to_buildings(
    system_id: Annotated[str, Field(description="ID of the DER system")],
    building_ids: Annotated[List[str], Field(description="List of building IDs")],
) -> Dict[str, Any]:
    """Assigns a DER system to serve specified buildings."""
    cm = state.config_managers[state.active_config]

    try:
        cm.assign_system_to_buildings(system_id, building_ids)
        return {
            "success": True,
            "data": {"system_id": system_id, "buildings": building_ids, "building_count": len(building_ids)},
            "message": f"DER system '{system_id}' assigned to {len(building_ids)} buildings",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Select DER systems for simulation")
async def der_select(
    system_ids: Annotated[List[str], Field(description="List of DER system IDs to select")]
) -> Dict[str, Any]:
    """Selects specified DER systems for the simulation."""
    cm = state.config_managers[state.active_config]
    cm.select_systems(system_ids)

    return {
        "success": True,
        "data": {"selected": system_ids, "count": len(system_ids)},
        "message": f"Selected {len(system_ids)} DER systems for simulation",
    }


# ============= CONTROLLER TOOLS =============

@mcp.tool(description="Add a controller for an HVAC system with thermal domain control")
async def controller_add_hvac(
        controller_id: Annotated[str, Field(description="Unique identifier for the controller")],
        system_id: Annotated[str, Field(description="ID of the HVAC system to control")],
        parameters: Annotated[Optional[Dict], Field(
            description="Controller parameters: {"
                        "'type': 'rule-based'|'mpc'|'rl', "
                        "'mode': 'cooling'|'heating'|'auto', "
                        "'precooling': {'degree': <float °C>, 'hours': <int hours before peak>}, "
                        "'base_cooling': <float °C setpoint for cooling>, "
                        "'base_heating': <float °C setpoint for heating>, "
                        "'deadband': <float °C>, "
                        "'cooling_power_max': <float watts>, "
                        "'heating_power_max': <float watts>} "
                        " - Only include fields you want to add"
        )] = None,
) -> Dict[str, Any]:
    """
    Adds a controller for an HVAC system.

    Example parameters for a rule-based cooling controller:
    {
        "type": "rule-based",
        "mode": "cooling",
        "precooling": {"degree": 0, "hours": 0},
        "base_cooling": 24.0,
        "base_heating": 18.0,
        "deadband": 0.5,
        "cooling_power_max": 4000.0,
        "heating_power_max": 4000.0
    }
    """
    cm = state.config_managers[state.active_config]
    params = parameters or {}
    params["domain"] = "thermal"

    try:
        cm.add_system_controller(controller_id=controller_id, system_id=system_id, parameters=params)
        return {
            "success": True,
            "data": {"controller_id": controller_id, "system_id": system_id, "domain": "thermal", "parameters": params},
            "message": f"HVAC controller '{controller_id}' added for system '{system_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Add a controller for a DER system with electrical domain control. "
                      "IMPORTANT: Only include parameters explicitly mentioned by the user - do not add defaults or infer values.")
async def controller_add_der(
        controller_id: Annotated[str, Field(description="Unique identifier for the controller")],
        system_id: Annotated[str, Field(description="ID of the DER system to control")],
        parameters: Annotated[Optional[Dict], Field(
            description="Controller parameters - ONLY include fields the user EXPLICITLY mentioned. "
                        "Do NOT add default values or infer missing parameters. "
                        "Do NOT include None values. "
                        "Available fields: "
                        "'type': 'rule-based'|'mpc'|'rl', "
                        "'mode': 'self_consumption'|'peak_shaving'|'arbitrage', "
                        "'bat_soc_min': <float 0-1>, "
                        "'bat_soc_max': <float 0-1>, "
                        "'ev_soc_min': <float 0-1>, "
                        "'ev_soc_target': <float 0-1>, "
                        "'ev_v2g_enabled': <bool>, "
                        "'max_grid_import': <float watts>, "
                        "'max_grid_export': <float watts>, "
                        "'system_config': <dict> - Configuration for DER components: {"
                        "  'pv': {'rated_capacity_kW': <float>}, "
                        "  'bat': {'rated_capacity_kWh': <float>, 'initial_soc': <float 0-1>}, "
                        "  'ev': {'rated_capacity_kWh': <float>, 'initial_soc': <float 0-1>}"
                        "}"
        )] = None,
) -> Dict[str, Any]:
    """
    Adds a controller for a DER system.

    Example parameters for a self-consumption DER controller:
    {
        "type": "rule-based",
        "mode": "self_consumption",
        "bat_soc_min": 0.1,
        "bat_soc_max": 0.9,
        "ev_soc_min": 0.2,
        "ev_soc_target": 0.8,
        "ev_v2g_enabled": True,
        "max_grid_import": 10000,
        "max_grid_export": 5000,
        "system_config": {
            "pv": {"rated_capacity_kW": 2},
            "bat": {"rated_capacity_kWh": 5, "initial_soc": 0.3},
            "ev": {"rated_capacity_kWh": 5, "initial_soc": 0.3}
        }
    }
    """
    cm = state.config_managers[state.active_config]
    params = parameters or {}
    params["domain"] = "electrical"

    try:
        cm.add_system_controller(controller_id=controller_id, system_id=system_id, parameters=params)
        return {
            "success": True,
            "data": {"controller_id": controller_id, "system_id": system_id, "domain": "electrical",
                     "parameters": params},
            "message": f"DER controller '{controller_id}' added for system '{system_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Update controller parameters")
async def controller_update(
        controller_id: Annotated[Optional[str], Field(description="ID of the controller to update. Use default if not provide")] = None,
        updates: Annotated[Dict, Field(
            description="Dictionary containing parameter updates, ONLY include fields the user EXPLICITLY mentioned. "
                        "Do NOT add default values or infer missing parameters. "
                        "For HVAC controllers: {"
                        "'type': 'rule-based'|'mpc'|'rl', "
                        "'mode': 'cooling'|'heating'|'auto', "
                        "'base_cooling': <float °C>, "
                        "'base_heating': <float °C>, "
                        "'deadband': <float °C>, "
                        "'cooling_power_max': <float W>, "
                        "'heating_power_max': <float W>, "
                        "'precooling': {'degree': <float>, 'hours': <int>}}. "
                        "For DER controllers: {"
                        "'mode': 'self_consumption'|'peak_shaving'|'arbitrage', "
                        "'bat_soc_min': <float 0-1>, "
                        "'bat_soc_max': <float 0-1>, "
                        "'ev_soc_min': <float 0-1>, "
                        "'ev_soc_target': <float 0-1>, "
                        "'ev_v2g_enabled': <bool>, "
                        "'max_grid_import': <float W>, "
                        "'max_grid_export': <float W>, "
                        "'system_config': <dict> - Configuration for DER components: {"
                        "  'pv': {'rated_capacity_kW': <float>}, "
                        "  'bat': {'rated_capacity_kWh': <float>, 'initial_soc': <float 0-1>}, "
                        "  'ev': {'rated_capacity_kWh': <float>, 'initial_soc': <float 0-1>}"
                        "}}"
        )] = None,
) -> Dict[str, Any]:
    """
    Updates parameters for an existing controller.

    Example updates for HVAC controller:
    {
        "base_cooling": 22.0,
        "deadband": 0.4
    }

    Example updates for DER controller:
    {
        "ev_v2g_enabled": True,
        "ev_soc_target": 0.85,
        "bat_soc_max": 0.95
    }

    Example updates for DER system_config:
    {
        "system_config": {
            "pv": {"rated_capacity_kW": 3},
            "bat": {"rated_capacity_kWh": 10, "initial_soc": 0.5}
        }
    }

    Note: Updates are merged directly into controller parameters,
    not nested under 'parameters' key.
    """
    cm = state.config_managers[state.active_config]

    # Resolve controller_id if not provided
    used_default = False
    if controller_id is None:
        controllers = cm.config.get("controllers", {})
        if not controllers:
            return {"success": False, "error": "No controllers found in configuration"}
        controller_id = next(iter(controllers.keys()))
        used_default = True

    try:
        if controller_id not in cm.config.get("controllers", {}):
            return {
                "success": False,
                "error": f"Controller '{controller_id}' not found",
                "data": {"available_controllers": list(cm.config.get("controllers", {}).keys())}
            }

        controller = cm.config["controllers"][controller_id]

        if updates:
            # Handle nested system_config updates (deep merge)
            if "system_config" in updates and "system_config" in controller["parameters"]:
                existing_config = controller["parameters"]["system_config"]
                new_config = updates.pop("system_config")
                for component, config in new_config.items():
                    if component in existing_config:
                        existing_config[component].update(config)
                    else:
                        existing_config[component] = config

            controller["parameters"].update(updates)

        message = f"Controller '{controller_id}' updated successfully"
        if used_default:
            message = f"No specific controller_id provided, using default '{controller_id}'. " + message

        return {
            "success": True,
            "data": {
                "controller_id": controller_id,
                "updates_applied": list(updates.keys()) if updates else [],
                "used_default": used_default
            },
            "message": message,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool(description="Remove a controller from configuration")
async def controller_remove(
    controller_id: Annotated[str, Field(description="ID of the controller to remove")]
) -> Dict[str, Any]:
    """Removes a controller from the configuration."""
    cm = state.config_managers[state.active_config]

    try:
        system_id = cm.config["controllers"][controller_id].get("system_id")
        del cm.config["controllers"][controller_id]
        if controller_id in cm.selected_controllers:
            del cm.selected_controllers[controller_id]

        for sid, system in cm.selected_systems.items():
            if system.get("controller_id") == controller_id:
                del system["controller_id"]

        return {
            "success": True,
            "data": {"removed": controller_id, "from_system": system_id},
            "message": f"Controller '{controller_id}' removed successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Query controller information")
async def controller_query(
    controller_id: Annotated[Optional[str], Field(description="ID of specific controller")] = None,
) -> Dict[str, Any]:
    """Queries controller information from the configuration."""
    cm = state.config_managers[state.active_config]
    controllers = cm.config.get("controllers", {})

    if controller_id:
        if controller_id in controllers:
            return {"success": True, "data": controllers[controller_id], "message": f"Controller '{controller_id}' found"}
        else:
            return {
                "success": False,
                "error": f"Controller '{controller_id}' not found",
                "data": {"available_controllers": list(controllers.keys())},
            }

    return {
        "success": True,
        "data": {"controllers": controllers, "count": len(controllers), "controller_ids": list(controllers.keys())},
    }


@mcp.tool(description="Assign a controller to a system")
async def controller_assign_to_system(
    system_id: Annotated[str, Field(description="ID of the system")],
    controller_id: Annotated[str, Field(description="ID of the controller to assign")],
) -> Dict[str, Any]:
    """Assigns a controller to manage a specific system."""
    cm = state.config_managers[state.active_config]

    try:
        cm.select_controller_for_system(system_id, controller_id)
        return {
            "success": True,
            "data": {"system_id": system_id, "controller_id": controller_id},
            "message": f"Controller '{controller_id}' assigned to system '{system_id}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= DISTURBANCE TOOLS =============

@mcp.tool(description="Add weather disturbance data for simulation")
async def disturbance_add_weather(
    file_path: Annotated[str, Field(description="Path to weather data file")],
    simulation_start: Annotated[str, Field(description="Simulation start time")] = "2023-08-01 00:00:00",
) -> Dict[str, Any]:
    """Adds weather disturbance data to the simulation."""
    cm = state.config_managers[state.active_config]

    try:
        cm.add_disturbance("weather", parameters={"file_path": file_path, "simulation_start_time": simulation_start})
        return {
            "success": True,
            "data": {"disturbance": "weather", "file_path": file_path, "simulation_start": simulation_start},
            "message": "Weather disturbance added successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Add occupancy disturbance data for simulation")
async def disturbance_add_occupancy(
    file_path: Annotated[str, Field(description="Path to occupancy data file")],
    simulation_start: Annotated[str, Field(description="Simulation start time")] = "2023-08-01 00:00:00",
) -> Dict[str, Any]:
    """Adds occupancy disturbance data to the simulation."""
    cm = state.config_managers[state.active_config]

    try:
        cm.add_disturbance("occupancy", parameters={"file_path": file_path, "simulation_start_time": simulation_start})
        return {
            "success": True,
            "data": {"disturbance": "occupancy", "file_path": file_path, "simulation_start": simulation_start},
            "message": "Occupancy disturbance added successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Add electricity price disturbance for simulation")
async def disturbance_add_price(
    parameters: Annotated[Optional[Dict], Field(description="Price parameters")] = None,
) -> Dict[str, Any]:
    """Adds electricity price disturbance to the simulation."""
    cm = state.config_managers[state.active_config]

    try:
        cm.add_disturbance("price", parameters=parameters or {})
        return {
            "success": True,
            "data": {"disturbance": "price", "parameters": parameters or {}},
            "message": "Price disturbance added successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Update disturbance parameters")
async def disturbance_update(
    disturbance_name: Annotated[Optional[str], Field(description="Name of the disturbance to update. Use default if not provide")] = None,
    updates: Annotated[Dict, Field(description="Dictionary containing parameter updates")] = None,
) -> Dict[str, Any]:
    """Updates parameters for an existing disturbance."""
    cm = state.config_managers[state.active_config]

    # Resolve disturbance_name if not provided
    used_default = False
    if disturbance_name is None:
        disturbances = cm.config.get("disturbances", {})
        if not disturbances:
            return {"success": False, "error": "No disturbances found in configuration"}
        disturbance_name = next(iter(disturbances.keys()))
        used_default = True

    try:
        if disturbance_name not in cm.config.get("disturbances", {}):
            return {
                "success": False,
                "error": f"Disturbance '{disturbance_name}' not found",
                "data": {"available_disturbances": list(cm.config.get("disturbances", {}).keys())}
            }

        disturbance = cm.config["disturbances"][disturbance_name]
        if updates:
            disturbance["parameters"].update(updates)

        message = f"Disturbance '{disturbance_name}' updated successfully"
        if used_default:
            message = f"No specific disturbance_name provided, using default '{disturbance_name}'. " + message

        return {
            "success": True,
            "data": {
                "disturbance": disturbance_name,
                "updates_applied": list(updates.keys()) if updates else [],
                "used_default": used_default
            },
            "message": message,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Remove a disturbance from configuration")
async def disturbance_remove(
    disturbance_name: Annotated[str, Field(description="Name of the disturbance to remove")]
) -> Dict[str, Any]:
    """Removes a disturbance from the configuration."""
    cm = state.config_managers[state.active_config]

    try:
        del cm.config["disturbances"][disturbance_name]
        if disturbance_name in cm.selected_disturbances:
            del cm.selected_disturbances[disturbance_name]

        return {
            "success": True,
            "data": {"removed": disturbance_name},
            "message": f"Disturbance '{disturbance_name}' removed successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Query all disturbances in configuration")
async def disturbance_query() -> Dict[str, Any]:
    """Queries all disturbances in the current configuration."""
    cm = state.config_managers[state.active_config]
    disturbances = cm.config.get("disturbances", {})

    return {
        "success": True,
        "data": {"disturbances": disturbances, "count": len(disturbances), "disturbance_names": list(disturbances.keys())},
    }


@mcp.tool(description="Select disturbances for simulation")
async def disturbance_select(
    disturbance_names: Annotated[List[str], Field(description="List of disturbance names to select")]
) -> Dict[str, Any]:
    """Selects specified disturbances for the simulation."""
    cm = state.config_managers[state.active_config]
    cm.select_disturbances(disturbance_names)

    return {
        "success": True,
        "data": {"selected": disturbance_names, "count": len(disturbance_names)},
        "message": f"Selected {len(disturbance_names)} disturbances for simulation",
    }


# ============= ENVIRONMENT TOOLS =============

@mcp.tool(description="Configure the simulation environment with time resolution, duration, and start time. Use when initially setting up or resetting the simulation environment.")
async def environment_setup(
    resolution: Annotated[int, Field(description="Time resolution in seconds")] = 900,
    duration: Annotated[int, Field(description="Total simulation duration in seconds")] = 86400,
    simulation_start: Annotated[str, Field(description="Simulation start time")] = "2023-08-01 00:00:00",
) -> Dict[str, Any]:
    """Sets up the simulation environment configuration."""
    cm = state.config_managers[state.active_config]

    try:
        cm.add_environment(
            parameters={
                "resolution": resolution,
                "duration": duration,
                "simulation_start_time": simulation_start,
                "enable_history": True,
            }
        )

        steps = duration // resolution
        return {
            "success": True,
            "data": {"resolution": resolution, "duration": duration, "simulation_start": simulation_start, "total_steps": steps},
            "message": f"Environment configured: {steps} steps at {resolution}s resolution",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Modify specific parameters of the already-configured environment. Do NOT use if environment has not been set up yet; use environment_setup first.")
async def environment_update(
    updates: Annotated[Dict, Field(description="Dictionary containing parameter updates")]
) -> Dict[str, Any]:
    """Updates environment configuration parameters."""
    cm = state.config_managers[state.active_config]

    try:
        env_config = cm.config["environment"]
        env_config["parameters"].update(updates)

        return {
            "success": True,
            "data": {"updates_applied": list(updates.keys())},
            "message": "Environment parameters updated successfully",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Query environment configuration")
async def environment_query() -> Dict[str, Any]:
    """Queries the current environment configuration."""
    cm = state.config_managers[state.active_config]
    env_config = cm.config.get("environment", {})

    if env_config:
        params = env_config.get("parameters", {})
        return {
            "success": True,
            "data": {
                "environment": env_config,
                "resolution": params.get("resolution"),
                "duration": params.get("duration"),
                "simulation_start": params.get("simulation_start_time"),
            },
        }
    else:
        return {
            "success": False,
            "error": "Environment not configured",
            "message": "Use environment_setup to configure the environment",
        }


@mcp.tool(description="Select environment for simulation")
async def environment_select() -> Dict[str, Any]:
    """Selects the environment configuration for simulation."""
    cm = state.config_managers[state.active_config]

    try:
        cm.select_environment()
        return {
            "success": True,
            "data": {"environment": "selected"},
            "message": "Environment configuration selected for simulation",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= SIMULATION TOOLS =============

# @mcp.tool(description="Create simulation environment from current configuration")
# async def simulation_create_environment() -> Dict[str, Any]:
#     """Creates the BESTOpt environment from the current configuration."""
#     cm = state.config_managers[state.active_config]
#
#     try:
#         env = BESTOptEnvironment(cm.config)
#         state.environments[state.active_config] = env
#
#         return {
#             "success": True,
#             "data": {"environment": state.active_config, "total_steps": env.total_step if hasattr(env, "total_step") else 0},
#             "message": f"Environment created for configuration '{state.active_config}'",
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}


@mcp.tool(description="Run building energy simulation for specified timesteps")
async def simulation_run(
    steps: Annotated[int, Field(description="Number of simulation timesteps to run")] = 96,
    save_name: Annotated[Optional[str], Field(description="Name for saving simulation results")] = None,
) -> Dict[str, Any]:
    """Executes the building energy simulation."""
    cm = state.config_managers[state.active_config]
    env = BESTOptEnvironment(cm.config)

    save_name = save_name or f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    state.simulation_manager.initialize_simulation(save_name)
    state.active_simulations[save_name] = {
        "config": state.active_config,
        "start_time": datetime.now().isoformat(),
        "status": "running",
    }

    try:
        actual_steps = 0
        for step in range(min(steps, env.total_step)):
            observations, done, info = env.step()
            state.simulation_manager.add_simulation_step(save_name, env, step)
            actual_steps += 1

            if step % 8 == 0:
                state.logger.info(f"Simulation {save_name}: Step {step}/{steps}")

            if done:
                break

        state.simulation_manager.finalize_simulation(save_name)
        state.active_simulations[save_name]["status"] = "completed"
        state.active_simulations[save_name]["end_time"] = datetime.now().isoformat()
        state.active_simulations[save_name]["total_steps"] = actual_steps

        save_path = str(state.results_path / "operation" / f"{save_name}.pkl")

        return {
            "success": True,
            "data": {"name": save_name, "steps_completed": actual_steps, "saved_to": save_path, "configuration": state.active_config},
            "message": f"Simulation '{save_name}' completed with {actual_steps} steps",
        }
    except Exception as e:
        state.active_simulations[save_name]["status"] = "failed"
        state.active_simulations[save_name]["error"] = str(e)
        return {"success": False, "error": str(e)}


@mcp.tool(description="Get current simulation status and progress")
async def simulation_get_status() -> Dict[str, Any]:
    """Gets the current status of simulations and environment."""
    status_data = {
        "active_config": state.active_config,
        "environment_created": state.active_config in state.environments,
        "active_simulations": {},
        "completed_simulations": [],
        "failed_simulations": [],
    }

    for name, info in state.active_simulations.items():
        if info["status"] == "completed":
            status_data["completed_simulations"].append(name)
        elif info["status"] == "failed":
            status_data["failed_simulations"].append(name)
        else:
            status_data["active_simulations"][name] = {"status": info["status"], "start_time": info.get("start_time")}

    if state.active_config in state.environments:
        env = state.environments[state.active_config]
        status_data["environment_info"] = {
            "current_step": env.current_step if hasattr(env, "current_step") else 0,
            "total_steps": env.total_step if hasattr(env, "total_step") else 0,
        }

    return {"success": True, "data": status_data}


@mcp.tool(description="List all available simulation results")
async def simulation_list_results() -> Dict[str, Any]:
    """Lists all simulation results."""
    results = []

    for name, info in state.active_simulations.items():
        results.append({
            "name": name,
            "status": info["status"],
            "steps": info.get("total_steps", 0),
            "config": info["config"],
            "start_time": info.get("start_time"),
            "end_time": info.get("end_time"),
        })

    operation_path = state.results_path / "operation"
    saved_files = []
    if operation_path.exists():
        for pkl_file in operation_path.glob("*.pkl"):
            sim_name = pkl_file.stem
            if sim_name not in state.active_simulations:
                saved_files.append(sim_name)

    return {"success": True, "data": {"results": results, "count": len(results), "saved_files": saved_files}}


# ============= ANALYSIS TOOLS =============

@mcp.tool(description="Analyze thermal comfort metrics from simulation results")
async def analysis_comfort(
    simulation_name: Annotated[str, Field(description="Name of the simulation to analyze")]
) -> Dict[str, Any]:
    """Analyzes thermal comfort metrics."""
    try:
        analysis = state.simulation_manager.analyze_comfort(simulation_name)
        if "error" in analysis:
            return {"success": False, "error": analysis["error"]}
        return {"success": True, "data": analysis, "message": f"Comfort analysis complete for '{simulation_name}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Analyze energy consumption and generation metrics")
async def analysis_energy(
    simulation_name: Annotated[str, Field(description="Name of the simulation to analyze")]
) -> Dict[str, Any]:
    """Analyzes energy metrics."""
    try:
        analysis = state.simulation_manager.analyze_energy(simulation_name)
        if "error" in analysis:
            return {"success": False, "error": analysis["error"]}
        return {"success": True, "data": analysis, "message": f"Energy analysis complete for '{simulation_name}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Analyze cost and economic metrics")
async def analysis_cost(
    simulation_name: Annotated[str, Field(description="Name of the simulation to analyze")]
) -> Dict[str, Any]:
    """Analyzes cost metrics."""
    try:
        analysis = state.simulation_manager.analyze_cost(simulation_name)
        if "error" in analysis:
            return {"success": False, "error": analysis["error"]}
        return {"success": True, "data": analysis, "message": f"Cost analysis complete for '{simulation_name}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Analyze grid flexibility and battery cycling metrics")
async def analysis_flexibility(
    simulation_name: Annotated[str, Field(description="Name of the simulation to analyze")]
) -> Dict[str, Any]:
    """Analyzes flexibility metrics."""
    try:
        analysis = state.simulation_manager.analyze_flexibility(simulation_name)
        if "error" in analysis:
            return {"success": False, "error": analysis["error"]}
        return {"success": True, "data": analysis, "message": f"Flexibility analysis complete for '{simulation_name}'"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Perform comprehensive analysis of all metrics")
async def analysis_comprehensive(
    simulation_name: Annotated[str, Field(description="Name of the simulation to analyze")]
) -> Dict[str, Any]:
    """Performs comprehensive analysis."""
    try:
        comfort = state.simulation_manager.analyze_comfort(simulation_name)
        energy = state.simulation_manager.analyze_energy(simulation_name)
        cost = state.simulation_manager.analyze_cost(simulation_name)
        flexibility = state.simulation_manager.analyze_flexibility(simulation_name)

        for analysis_type, analysis in [("comfort", comfort), ("energy", energy), ("cost", cost), ("flexibility", flexibility)]:
            if "error" in analysis:
                return {"success": False, "error": f"{analysis_type} analysis failed: {analysis['error']}"}

        summary = {
            "violation_rate": comfort.get("violation_rate", 0),
            "avg_temperature": comfort.get("avg_temperature", 0),
            "total_consumption_kwh": energy.get("total_consumption_kwh", 0),
            "renewable_fraction": energy.get("renewable_fraction", 0),
            "self_consumption_rate": energy.get("self_consumption_rate", 0),
            "peak_demand_kw": energy.get("peak_demand_kw", 0),
            "net_cost": cost.get("net_cost", 0),
            "battery_cycles": flexibility.get("battery_cycles_efc", 0),
        }

        return {
            "success": True,
            "data": {"comfort": comfort, "energy": energy, "cost": cost, "flexibility": flexibility, "summary": summary},
            "message": f"Comprehensive analysis complete for '{simulation_name}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= COMPARISON TOOLS =============

@mcp.tool(description="Compare thermal comfort metrics between two simulations")
async def comparison_comfort(
    sim1: Annotated[str, Field(description="Name of first simulation")],
    sim2: Annotated[str, Field(description="Name of second simulation")],
) -> Dict[str, Any]:
    """Compares comfort metrics between two simulations."""
    try:
        analysis1 = state.simulation_manager.analyze_comfort(sim1)
        analysis2 = state.simulation_manager.analyze_comfort(sim2)

        if "error" in analysis1 or "error" in analysis2:
            return {"success": False, "error": "Failed to analyze one or both simulations"}

        differences = {
            "avg_temperature_diff": analysis2["avg_temperature"] - analysis1["avg_temperature"],
            "violation_rate_diff": analysis2["violation_rate"] - analysis1["violation_rate"],
            "total_violation_c_h_diff": analysis2["total_violation_c_h"] - analysis1["total_violation_c_h"],
            "temp_range_diff": analysis2["temp_range"] - analysis1["temp_range"],
        }

        better_comfort = sim1 if analysis1["violation_rate"] < analysis2["violation_rate"] else sim2
        improvement = (
            (analysis1["violation_rate"] - analysis2["violation_rate"]) / analysis1["violation_rate"] * 100
            if analysis1["violation_rate"] > 0 else 0
        )

        return {
            "success": True,
            "data": {
                "sim1": {"name": sim1, "metrics": analysis1},
                "sim2": {"name": sim2, "metrics": analysis2},
                "differences": differences,
                "better_comfort": better_comfort,
                "improvement_percentage": improvement,
            },
            "message": f"Comfort comparison complete: '{better_comfort}' has better comfort performance",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Compare energy metrics between two simulations")
async def comparison_energy(
    sim1: Annotated[str, Field(description="Name of first simulation")],
    sim2: Annotated[str, Field(description="Name of second simulation")],
) -> Dict[str, Any]:
    """Compares energy metrics between two simulations."""
    try:
        analysis1 = state.simulation_manager.analyze_energy(sim1)
        analysis2 = state.simulation_manager.analyze_energy(sim2)

        if "error" in analysis1 or "error" in analysis2:
            return {"success": False, "error": "Failed to analyze one or both simulations"}

        differences = {
            "total_consumption_diff": analysis2.get("total_consumption_kwh", 0) - analysis1.get("total_consumption_kwh", 0),
            "peak_demand_diff": analysis2.get("peak_demand_kw", 0) - analysis1.get("peak_demand_kw", 0),
            "renewable_fraction_diff": analysis2.get("renewable_fraction", 0) - analysis1.get("renewable_fraction", 0),
            "self_consumption_rate_diff": analysis2.get("self_consumption_rate", 0) - analysis1.get("self_consumption_rate", 0),
        }

        better_efficiency = sim1 if analysis1.get("renewable_fraction", 0) > analysis2.get("renewable_fraction", 0) else sim2

        return {
            "success": True,
            "data": {
                "sim1": {"name": sim1, "metrics": analysis1},
                "sim2": {"name": sim2, "metrics": analysis2},
                "differences": differences,
                "better_efficiency": better_efficiency,
            },
            "message": f"Energy comparison complete: '{better_efficiency}' has better energy efficiency",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Comprehensive comparison of all metrics between two simulations with detailed analysis")
async def comparison_comprehensive(
    sim1: Annotated[str, Field(description="Name of baseline simulation")],
    sim2: Annotated[str, Field(description="Name of updated simulation")],
) -> Dict[str, Any]:
    """Performs comprehensive comparison with detailed metrics and recommendations."""
    try:
        # Get all analyses
        comfort1 = state.simulation_manager.analyze_comfort(sim1)
        comfort2 = state.simulation_manager.analyze_comfort(sim2)
        energy1 = state.simulation_manager.analyze_energy(sim1)
        energy2 = state.simulation_manager.analyze_energy(sim2)
        cost1 = state.simulation_manager.analyze_cost(sim1)
        cost2 = state.simulation_manager.analyze_cost(sim2)
        flexibility1 = state.simulation_manager.analyze_flexibility(sim1)
        flexibility2 = state.simulation_manager.analyze_flexibility(sim2)

        # Check for errors
        for name, analysis in [("comfort", comfort1), ("comfort", comfort2),
                               ("energy", energy1), ("energy", energy2),
                               ("cost", cost1), ("cost", cost2),
                               ("flexibility", flexibility1), ("flexibility", flexibility2)]:
            if "error" in analysis:
                return {"success": False, "error": f"{name} analysis failed: {analysis['error']}"}

        # === COMFORT COMPARISON ===
        comfort_comparison = {
            "baseline": {
                "avg_temperature": comfort1.get("avg_temperature"),
                "min_temperature": comfort1.get("min_temperature"),
                "max_temperature": comfort1.get("max_temperature"),
                "temp_range": comfort1.get("temp_range"),
                "violation_rate_pct": comfort1.get("violation_rate"),
                "total_violation_c_h": comfort1.get("total_violation_c_h"),
                "temp_above_cooling_c_h": comfort1.get("temp_above_cooling_c_h"),
                "temp_below_heating_c_h": comfort1.get("temp_below_heating_c_h"),
            },
            "updated": {
                "avg_temperature": comfort2.get("avg_temperature"),
                "min_temperature": comfort2.get("min_temperature"),
                "max_temperature": comfort2.get("max_temperature"),
                "temp_range": comfort2.get("temp_range"),
                "violation_rate_pct": comfort2.get("violation_rate"),
                "total_violation_c_h": comfort2.get("total_violation_c_h"),
                "temp_above_cooling_c_h": comfort2.get("temp_above_cooling_c_h"),
                "temp_below_heating_c_h": comfort2.get("temp_below_heating_c_h"),
            },
            "differences": {
                "avg_temperature_diff": round(comfort2.get("avg_temperature", 0) - comfort1.get("avg_temperature", 0), 3),
                "violation_rate_diff": round(comfort2.get("violation_rate", 0) - comfort1.get("violation_rate", 0), 3),
                "total_violation_c_h_diff": round(comfort2.get("total_violation_c_h", 0) - comfort1.get("total_violation_c_h", 0), 3),
                "temp_range_diff": round(comfort2.get("temp_range", 0) - comfort1.get("temp_range", 0), 3),
            },
            "improvement": {
                "violation_reduced": comfort2.get("violation_rate", 0) < comfort1.get("violation_rate", 0),
                "violation_reduction_pct": round((comfort1.get("violation_rate", 0) - comfort2.get("violation_rate", 0)) / comfort1.get("violation_rate", 1) * 100, 2) if comfort1.get("violation_rate", 0) > 0 else 0,
            }
        }

        # === ENERGY COMPARISON ===
        energy_comparison = {
            "baseline": {
                "total_consumption_kwh": energy1.get("total_consumption_kwh"),
                "peak_demand_kw": energy1.get("peak_demand_kw"),
                "avg_demand_kw": energy1.get("avg_demand_kw"),
                "load_factor_pct": energy1.get("load_factor"),
                "pv_generation_kwh": energy1.get("pv_total_generation_kwh"),
                "self_consumption_kwh": energy1.get("self_consumption_kwh"),
                "self_consumption_rate_pct": energy1.get("self_consumption_rate"),
                "renewable_fraction_pct": energy1.get("renewable_fraction"),
                "grid_import_kwh": energy1.get("grid_import_total_kwh"),
                "grid_export_kwh": energy1.get("grid_export_total_kwh"),
                "net_grid_energy_kwh": energy1.get("net_grid_energy_kwh"),
            },
            "updated": {
                "total_consumption_kwh": energy2.get("total_consumption_kwh"),
                "peak_demand_kw": energy2.get("peak_demand_kw"),
                "avg_demand_kw": energy2.get("avg_demand_kw"),
                "load_factor_pct": energy2.get("load_factor"),
                "pv_generation_kwh": energy2.get("pv_total_generation_kwh"),
                "self_consumption_kwh": energy2.get("self_consumption_kwh"),
                "self_consumption_rate_pct": energy2.get("self_consumption_rate"),
                "renewable_fraction_pct": energy2.get("renewable_fraction"),
                "grid_import_kwh": energy2.get("grid_import_total_kwh"),
                "grid_export_kwh": energy2.get("grid_export_total_kwh"),
                "net_grid_energy_kwh": energy2.get("net_grid_energy_kwh"),
            },
            "differences": {
                "total_consumption_diff_kwh": round(energy2.get("total_consumption_kwh", 0) - energy1.get("total_consumption_kwh", 0), 3),
                "peak_demand_diff_kw": round(energy2.get("peak_demand_kw", 0) - energy1.get("peak_demand_kw", 0), 3),
                "self_consumption_rate_diff_pct": round(energy2.get("self_consumption_rate", 0) - energy1.get("self_consumption_rate", 0), 3),
                "renewable_fraction_diff_pct": round(energy2.get("renewable_fraction", 0) - energy1.get("renewable_fraction", 0), 3),
                "grid_import_diff_kwh": round(energy2.get("grid_import_total_kwh", 0) - energy1.get("grid_import_total_kwh", 0), 3),
                "grid_export_diff_kwh": round(energy2.get("grid_export_total_kwh", 0) - energy1.get("grid_export_total_kwh", 0), 3),
            },
            "improvement": {
                "consumption_reduced": energy2.get("total_consumption_kwh", 0) < energy1.get("total_consumption_kwh", 0),
                "consumption_reduction_pct": round((energy1.get("total_consumption_kwh", 0) - energy2.get("total_consumption_kwh", 0)) / energy1.get("total_consumption_kwh", 1) * 100, 2) if energy1.get("total_consumption_kwh", 0) > 0 else 0,
                "peak_reduced": energy2.get("peak_demand_kw", 0) < energy1.get("peak_demand_kw", 0),
                "peak_reduction_pct": round((energy1.get("peak_demand_kw", 0) - energy2.get("peak_demand_kw", 0)) / energy1.get("peak_demand_kw", 1) * 100, 2) if energy1.get("peak_demand_kw", 0) > 0 else 0,
                "renewable_improved": energy2.get("renewable_fraction", 0) > energy1.get("renewable_fraction", 0),
                "self_consumption_improved": energy2.get("self_consumption_rate", 0) > energy1.get("self_consumption_rate", 0),
            }
        }

        # === COST COMPARISON ===
        cost_comparison = {
            "baseline": {
                "import_cost": cost1.get("import_cost"),
                "export_revenue": cost1.get("export_revenue"),
                "net_cost": cost1.get("net_cost"),
                "peak_consumption_kwh": cost1.get("peak_consumption_kwh"),
                "off_peak_consumption_kwh": cost1.get("off_peak_consumption_kwh"),
                "peak_charges": cost1.get("peak_charges"),
            },
            "updated": {
                "import_cost": cost2.get("import_cost"),
                "export_revenue": cost2.get("export_revenue"),
                "net_cost": cost2.get("net_cost"),
                "peak_consumption_kwh": cost2.get("peak_consumption_kwh"),
                "off_peak_consumption_kwh": cost2.get("off_peak_consumption_kwh"),
                "peak_charges": cost2.get("peak_charges"),
            },
            "differences": {
                "import_cost_diff": round(cost2.get("import_cost", 0) - cost1.get("import_cost", 0), 4),
                "export_revenue_diff": round(cost2.get("export_revenue", 0) - cost1.get("export_revenue", 0), 4),
                "net_cost_diff": round(cost2.get("net_cost", 0) - cost1.get("net_cost", 0), 4),
                "peak_charges_diff": round(cost2.get("peak_charges", 0) - cost1.get("peak_charges", 0), 4),
            },
            "improvement": {
                "cost_reduced": cost2.get("net_cost", 0) < cost1.get("net_cost", 0),
                "cost_savings": round(cost1.get("net_cost", 0) - cost2.get("net_cost", 0), 4),
                "cost_savings_pct": round((cost1.get("net_cost", 0) - cost2.get("net_cost", 0)) / cost1.get("net_cost", 1) * 100, 2) if cost1.get("net_cost", 0) > 0 else 0,
                "export_revenue_improved": cost2.get("export_revenue", 0) > cost1.get("export_revenue", 0),
            }
        }

        # === FLEXIBILITY COMPARISON ===
        flexibility_comparison = {
            "baseline": {
                "battery_soc_avg_pct": flexibility1.get("battery_soc_avg"),
                "battery_soc_min_pct": flexibility1.get("battery_soc_min"),
                "battery_soc_max_pct": flexibility1.get("battery_soc_max"),
                "battery_cycles_efc": flexibility1.get("battery_cycles_efc"),
                "ev_tesla_cycles_efc": flexibility1.get("ev_tesla_cycles_efc"),
                "ev_nissan_cycles_efc": flexibility1.get("ev_nissan_cycles_efc"),
                "load_factor_pct": flexibility1.get("load_factor"),
                "peak_to_average_ratio": flexibility1.get("peak_to_average_ratio"),
                "peak_max_kw": flexibility1.get("peak_max_kw"),
                "peak_reduction_pct": flexibility1.get("peak_reduction"),
            },
            "updated": {
                "battery_soc_avg_pct": flexibility2.get("battery_soc_avg"),
                "battery_soc_min_pct": flexibility2.get("battery_soc_min"),
                "battery_soc_max_pct": flexibility2.get("battery_soc_max"),
                "battery_cycles_efc": flexibility2.get("battery_cycles_efc"),
                "ev_tesla_cycles_efc": flexibility2.get("ev_tesla_cycles_efc"),
                "ev_nissan_cycles_efc": flexibility2.get("ev_nissan_cycles_efc"),
                "load_factor_pct": flexibility2.get("load_factor"),
                "peak_to_average_ratio": flexibility2.get("peak_to_average_ratio"),
                "peak_max_kw": flexibility2.get("peak_max_kw"),
                "peak_reduction_pct": flexibility2.get("peak_reduction"),
            },
            "differences": {
                "battery_cycles_diff": round(flexibility2.get("battery_cycles_efc", 0) - flexibility1.get("battery_cycles_efc", 0), 3),
                "ev_tesla_cycles_diff": round(flexibility2.get("ev_tesla_cycles_efc", 0) - flexibility1.get("ev_tesla_cycles_efc", 0), 3),
                "ev_nissan_cycles_diff": round(flexibility2.get("ev_nissan_cycles_efc", 0) - flexibility1.get("ev_nissan_cycles_efc", 0), 3),
                "load_factor_diff": round(flexibility2.get("load_factor", 0) - flexibility1.get("load_factor", 0), 3),
                "peak_reduction_diff": round(flexibility2.get("peak_reduction", 0) - flexibility1.get("peak_reduction", 0), 3),
            },
            "improvement": {
                "battery_cycling_reduced": flexibility2.get("battery_cycles_efc", 0) < flexibility1.get("battery_cycles_efc", 0),
                "load_factor_improved": flexibility2.get("load_factor", 0) > flexibility1.get("load_factor", 0),
                "peak_reduction_improved": flexibility2.get("peak_reduction", 0) > flexibility1.get("peak_reduction", 0),
            }
        }

        # === OVERALL SUMMARY ===
        improvements_count = 0
        metrics_improved = []
        metrics_degraded = []

        # Comfort assessment
        if comfort_comparison["improvement"]["violation_reduced"]:
            improvements_count += 1
            metrics_improved.append("comfort")
        elif comfort2.get("violation_rate", 0) > comfort1.get("violation_rate", 0):
            metrics_degraded.append("comfort")

        # Energy assessment
        if energy_comparison["improvement"]["consumption_reduced"] or energy_comparison["improvement"]["renewable_improved"]:
            improvements_count += 1
            metrics_improved.append("energy")
        elif energy2.get("total_consumption_kwh", 0) > energy1.get("total_consumption_kwh", 0):
            metrics_degraded.append("energy")

        # Cost assessment
        if cost_comparison["improvement"]["cost_reduced"]:
            improvements_count += 1
            metrics_improved.append("cost")
        elif cost2.get("net_cost", 0) > cost1.get("net_cost", 0):
            metrics_degraded.append("cost")

        # Flexibility assessment (lower battery cycling is generally better for battery life)
        if flexibility_comparison["improvement"]["battery_cycling_reduced"] or flexibility_comparison["improvement"]["load_factor_improved"]:
            improvements_count += 1
            metrics_improved.append("flexibility")
        elif flexibility2.get("battery_cycles_efc", 0) > flexibility1.get("battery_cycles_efc", 0) * 1.2:  # 20% tolerance
            metrics_degraded.append("flexibility")

        overall_winner = sim2 if improvements_count >= 2 else sim1

        summary = {
            "baseline_simulation": sim1,
            "updated_simulation": sim2,
            "overall_winner": overall_winner,
            "improvements_count": f"{improvements_count}/4",
            "metrics_improved": metrics_improved,
            "metrics_degraded": metrics_degraded,
            "key_changes": {
                "comfort_violation_change_pct": comfort_comparison["differences"]["violation_rate_diff"],
                "energy_consumption_change_kwh": energy_comparison["differences"]["total_consumption_diff_kwh"],
                "cost_savings": cost_comparison["improvement"]["cost_savings"],
                "battery_cycle_change": flexibility_comparison["differences"]["battery_cycles_diff"],
            },
            "recommendation": f"'{overall_winner}' performs better overall. Improved: {', '.join(metrics_improved) if metrics_improved else 'none'}. Degraded: {', '.join(metrics_degraded) if metrics_degraded else 'none'}."
        }

        return {
            "success": True,
            "data": {
                "summary": summary,
                "comfort": comfort_comparison,
                "energy": energy_comparison,
                "cost": cost_comparison,
                "flexibility": flexibility_comparison,
            },
            "message": f"Comprehensive comparison complete: '{overall_winner}' is recommended with {improvements_count}/4 metrics improved",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool(description="Compare cost and economic metrics between two simulations")
async def comparison_cost(
    sim1: Annotated[str, Field(description="Name of first simulation (baseline)")],
    sim2: Annotated[str, Field(description="Name of second simulation (updated)")],
) -> Dict[str, Any]:
    """Compares cost metrics between two simulations."""
    try:
        cost1 = state.simulation_manager.analyze_cost(sim1)
        cost2 = state.simulation_manager.analyze_cost(sim2)

        if "error" in cost1:
            return {"success": False, "error": f"Failed to analyze {sim1}: {cost1['error']}"}
        if "error" in cost2:
            return {"success": False, "error": f"Failed to analyze {sim2}: {cost2['error']}"}

        differences = {
            "import_cost_diff": round(cost2.get("import_cost", 0) - cost1.get("import_cost", 0), 4),
            "export_revenue_diff": round(cost2.get("export_revenue", 0) - cost1.get("export_revenue", 0), 4),
            "net_cost_diff": round(cost2.get("net_cost", 0) - cost1.get("net_cost", 0), 4),
            "peak_charges_diff": round(cost2.get("peak_charges", 0) - cost1.get("peak_charges", 0), 4),
            "peak_consumption_diff_kwh": round(cost2.get("peak_consumption_kwh", 0) - cost1.get("peak_consumption_kwh", 0), 3),
        }

        # Determine which is more cost-effective
        better_cost = sim1 if cost1.get("net_cost", 0) < cost2.get("net_cost", 0) else sim2
        cost_savings = abs(cost2.get("net_cost", 0) - cost1.get("net_cost", 0))
        savings_pct = (cost_savings / cost1.get("net_cost", 1) * 100) if cost1.get("net_cost", 0) > 0 else 0

        return {
            "success": True,
            "data": {
                "sim1": {
                    "name": sim1,
                    "metrics": {
                        "import_cost": cost1.get("import_cost"),
                        "export_revenue": cost1.get("export_revenue"),
                        "net_cost": cost1.get("net_cost"),
                        "peak_charges": cost1.get("peak_charges"),
                        "peak_consumption_kwh": cost1.get("peak_consumption_kwh"),
                        "off_peak_consumption_kwh": cost1.get("off_peak_consumption_kwh"),
                    }
                },
                "sim2": {
                    "name": sim2,
                    "metrics": {
                        "import_cost": cost2.get("import_cost"),
                        "export_revenue": cost2.get("export_revenue"),
                        "net_cost": cost2.get("net_cost"),
                        "peak_charges": cost2.get("peak_charges"),
                        "peak_consumption_kwh": cost2.get("peak_consumption_kwh"),
                        "off_peak_consumption_kwh": cost2.get("off_peak_consumption_kwh"),
                    }
                },
                "differences": differences,
                "analysis": {
                    "better_cost": better_cost,
                    "cost_savings": round(cost_savings, 4),
                    "savings_percentage": round(savings_pct, 2),
                    "import_cost_reduced": cost2.get("import_cost", 0) < cost1.get("import_cost", 0),
                    "export_revenue_increased": cost2.get("export_revenue", 0) > cost1.get("export_revenue", 0),
                    "peak_charges_reduced": cost2.get("peak_charges", 0) < cost1.get("peak_charges", 0),
                }
            },
            "message": f"Cost comparison complete: '{better_cost}' is more cost-effective (saves ${round(cost_savings, 2)} / {round(savings_pct, 1)}%)",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Compare grid flexibility and battery cycling metrics between two simulations")
async def comparison_flexibility(
        sim1: Annotated[str, Field(description="Name of first simulation (baseline)")],
        sim2: Annotated[str, Field(description="Name of second simulation (updated)")],
) -> Dict[str, Any]:
    """Compares flexibility metrics between two simulations."""
    try:
        flex1 = state.simulation_manager.analyze_flexibility(sim1)
        flex2 = state.simulation_manager.analyze_flexibility(sim2)

        if "error" in flex1:
            return {"success": False, "error": f"Failed to analyze {sim1}: {flex1['error']}"}
        if "error" in flex2:
            return {"success": False, "error": f"Failed to analyze {sim2}: {flex2['error']}"}

        differences = {
            "battery_soc_avg_diff_pct": round(flex2.get("battery_soc_avg", 0) - flex1.get("battery_soc_avg", 0), 2),
            "battery_cycles_diff": round(flex2.get("battery_cycles_efc", 0) - flex1.get("battery_cycles_efc", 0), 3),
            "ev_tesla_cycles_diff": round(flex2.get("ev_tesla_cycles_efc", 0) - flex1.get("ev_tesla_cycles_efc", 0), 3),
            "ev_nissan_cycles_diff": round(flex2.get("ev_nissan_cycles_efc", 0) - flex1.get("ev_nissan_cycles_efc", 0),
                                           3),
            "load_factor_diff_pct": round(flex2.get("load_factor", 0) - flex1.get("load_factor", 0), 2),
            "peak_to_average_ratio_diff": round(
                flex2.get("peak_to_average_ratio", 0) - flex1.get("peak_to_average_ratio", 0), 3),
            "peak_reduction_diff_pct": round(flex2.get("peak_reduction", 0) - flex1.get("peak_reduction", 0), 2),
        }

        # Analysis - lower battery cycling is better for battery longevity
        battery_cycling_better = sim2 if flex2.get("battery_cycles_efc", 0) < flex1.get("battery_cycles_efc",
                                                                                        0) else sim1
        load_flexibility_better = sim2 if flex2.get("load_factor", 0) > flex1.get("load_factor", 0) else sim1

        # Total EV cycles
        ev_cycles_1 = flex1.get("ev_tesla_cycles_efc", 0) + flex1.get("ev_nissan_cycles_efc", 0)
        ev_cycles_2 = flex2.get("ev_tesla_cycles_efc", 0) + flex2.get("ev_nissan_cycles_efc", 0)

        return {
            "success": True,
            "data": {
                "sim1": {
                    "name": sim1,
                    "metrics": {
                        "battery_soc_avg_pct": flex1.get("battery_soc_avg"),
                        "battery_soc_min_pct": flex1.get("battery_soc_min"),
                        "battery_soc_max_pct": flex1.get("battery_soc_max"),
                        "battery_cycles_efc": flex1.get("battery_cycles_efc"),
                        "ev_tesla_cycles_efc": flex1.get("ev_tesla_cycles_efc"),
                        "ev_nissan_cycles_efc": flex1.get("ev_nissan_cycles_efc"),
                        "total_ev_cycles_efc": round(ev_cycles_1, 3),
                        "load_factor_pct": flex1.get("load_factor"),
                        "peak_to_average_ratio": flex1.get("peak_to_average_ratio"),
                        "peak_max_kw": flex1.get("peak_max_kw"),
                        "peak_reduction_pct": flex1.get("peak_reduction"),
                    }
                },
                "sim2": {
                    "name": sim2,
                    "metrics": {
                        "battery_soc_avg_pct": flex2.get("battery_soc_avg"),
                        "battery_soc_min_pct": flex2.get("battery_soc_min"),
                        "battery_soc_max_pct": flex2.get("battery_soc_max"),
                        "battery_cycles_efc": flex2.get("battery_cycles_efc"),
                        "ev_tesla_cycles_efc": flex2.get("ev_tesla_cycles_efc"),
                        "ev_nissan_cycles_efc": flex2.get("ev_nissan_cycles_efc"),
                        "total_ev_cycles_efc": round(ev_cycles_2, 3),
                        "load_factor_pct": flex2.get("load_factor"),
                        "peak_to_average_ratio": flex2.get("peak_to_average_ratio"),
                        "peak_max_kw": flex2.get("peak_max_kw"),
                        "peak_reduction_pct": flex2.get("peak_reduction"),
                    }
                },
                "differences": differences,
                "analysis": {
                    "battery_cycling_better": battery_cycling_better,
                    "load_flexibility_better": load_flexibility_better,
                    "battery_cycles_reduced": flex2.get("battery_cycles_efc", 0) < flex1.get("battery_cycles_efc", 0),
                    "battery_cycles_change_pct": round(
                        (flex2.get("battery_cycles_efc", 0) - flex1.get("battery_cycles_efc", 0)) / flex1.get(
                            "battery_cycles_efc", 1) * 100, 2) if flex1.get("battery_cycles_efc", 0) > 0 else 0,
                    "ev_cycles_reduced": ev_cycles_2 < ev_cycles_1,
                    "load_factor_improved": flex2.get("load_factor", 0) > flex1.get("load_factor", 0),
                    "peak_shaving_improved": flex2.get("peak_reduction", 0) > flex1.get("peak_reduction", 0),
                }
            },
            "message": f"Flexibility comparison complete: '{battery_cycling_better}' has better battery management, '{load_flexibility_better}' has better load flexibility",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# # ============= COMMUNICATION/SUMMARY TOOLS =============
#
# @mcp.tool(description="Get system-wide summary of current configuration and simulation status")
# async def communication_get_summary() -> Dict[str, Any]:
#     """Provides a high-level summary of the system state."""
#     cm = state.config_managers[state.active_config]
#
#     summary = {
#         "active_configuration": state.active_config,
#         "total_configurations": len(state.config_managers),
#         "auto_initialized": state.DEFAULT_CONFIG_NAME in state.config_managers,
#         "infrastructure": {
#             "clusters": len(cm.config.get("clusters", {})),
#             "buildings": len(cm.config.get("buildings", {})),
#             "hvac_systems": len([s for s in cm.config.get("systems", {}).values() if s.get("system_type") == "hvac_systems"]),
#             "der_systems": len([s for s in cm.config.get("systems", {}).values() if s.get("system_type") == "der_systems"]),
#             "controllers": len(cm.config.get("controllers", {})),
#             "disturbances": list(cm.config.get("disturbances", {}).keys()),
#         },
#         "simulation": {
#             "environment_ready": state.active_config in state.environments,
#             "completed_simulations": len([s for s in state.active_simulations.values() if s["status"] == "completed"]),
#             "running_simulations": len([s for s in state.active_simulations.values() if s["status"] == "running"]),
#             "failed_simulations": len([s for s in state.active_simulations.values() if s["status"] == "failed"]),
#         },
#         "validation": {
#             "warnings": len(cm.validate_configuration()),
#             "is_valid": len(cm.validate_configuration()) == 0,
#         },
#     }
#
#     return {"success": True, "data": summary, "message": "System summary generated successfully"}
#
#
# @mcp.tool(description="Get detailed validation report for current configuration")
# async def communication_get_validation_report() -> Dict[str, Any]:
#     """Provides a detailed validation report identifying any configuration issues."""
#     cm = state.config_managers[state.active_config]
#     warnings = cm.validate_configuration()
#
#     report = {
#         "configuration": state.active_config,
#         "is_valid": len(warnings) == 0,
#         "warning_count": len(warnings),
#         "warnings": warnings,
#         "ready_for_simulation": len(warnings) == 0,
#     }
#
#     message = "Configuration is valid and ready for simulation" if len(warnings) == 0 else f"Configuration has {len(warnings)} issues that should be addressed"
#
#     return {"success": True, "data": report, "message": message}
#
#
# @mcp.tool(description="Get summary of recent simulation results")
# async def communication_get_recent_results(
#     limit: Annotated[int, Field(description="Maximum number of recent results to return")] = 5,
# ) -> Dict[str, Any]:
#     """Provides a summary of the most recent simulation results with key metrics."""
#     try:
#         recent_sims = sorted(
#             [(name, info) for name, info in state.active_simulations.items() if info["status"] == "completed"],
#             key=lambda x: x[1].get("end_time", ""),
#             reverse=True,
#         )[:limit]
#
#         if not recent_sims:
#             return {"success": True, "data": {"results": [], "count": 0}, "message": "No completed simulations found"}
#
#         results = []
#         for sim_name, info in recent_sims:
#             comfort = state.simulation_manager.analyze_comfort(sim_name)
#             energy = state.simulation_manager.analyze_energy(sim_name)
#             cost = state.simulation_manager.analyze_cost(sim_name)
#
#             results.append({
#                 "name": sim_name,
#                 "config": info["config"],
#                 "steps": info.get("total_steps", 0),
#                 "completed_at": info.get("end_time"),
#                 "key_metrics": {
#                     "comfort_violation_rate": comfort.get("violation_rate", "N/A"),
#                     "net_cost": cost.get("net_cost", "N/A"),
#                     "renewable_fraction": energy.get("renewable_fraction", "N/A"),
#                     "total_consumption_kwh": energy.get("total_consumption_kwh", "N/A"),
#                 },
#             })
#
#         return {"success": True, "data": {"results": results, "count": len(results)}, "message": f"Retrieved {len(results)} recent simulation results"}
#     except Exception as e:
#         return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")