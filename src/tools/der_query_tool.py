import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bestopt.env.core.config_manager import ConfigurationManager
from .base_tool import BaseTool, ToolResult


class DERQueryTool(BaseTool):
    """
    Tool for querying DER systems and controllers
    Read-only operations for system and controller information
    """

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(
            name="der_query",
            description="Query DER systems and controllers",
            parameters_schema={
                "query_type": {
                    "type": "string",
                    "enum": ["system", "controller", "all"],
                    "description": "Type of query to perform"
                },
                "system_id": {
                    "type": "string",
                    "description": "Specific system ID to query (optional)"
                },
                "controller_id": {
                    "type": "string",
                    "description": "Specific controller ID to query (optional)"
                },
                "parameters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific parameters to retrieve"
                }
            }
        )

        self.config_dir = config_dir or Path(".")
        self.config_manager = ConfigurationManager(self.config_dir)

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute query operation"""
        try:
            query_type = parameters.get("query_type", "all")

            if query_type == "system":
                result = await self._query_systems(parameters)
            elif query_type == "controller":
                result = await self._query_controllers(parameters)
            else:  # "all"
                systems = await self._query_systems(parameters)
                controllers = await self._query_controllers(parameters)
                result = ToolResult(
                    success=True,
                    data={
                        "systems": systems.data if systems.success else {},
                        "controllers": controllers.data if controllers.success else {}
                    }
                )

            return result

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Query failed: {str(e)}"
            )

    async def _query_systems(self, parameters: Dict[str, Any]) -> ToolResult:
        """Query DER systems"""
        system_id = parameters.get("system_id")
        requested_params = parameters.get("parameters", [])

        if system_id:
            # Query specific system
            system_config = self.config_manager.get_config_by_id("systems", system_id)
            if not system_config:
                return ToolResult(
                    success=False,
                    error=f"System {system_id} not found"
                )

            # Filter parameters if requested
            if requested_params:
                filtered_config = self._filter_parameters(system_config, requested_params)
                return ToolResult(success=True, data={system_id: filtered_config})
            else:
                return ToolResult(success=True, data={system_id: system_config})
        else:
            # Query all systems
            all_systems = self.config_manager.list_all("systems")
            systems_data = {}

            for sid in all_systems:
                system_config = self.config_manager.get_config_by_id("systems", sid)
                if system_config and system_config.get("system_type") == "der_systems":
                    systems_data[sid] = system_config
                    # @todo should filter using llm
                    # if requested_params:
                    #     systems_data[sid] = self._filter_parameters(system_config, requested_params)
                    # else:
                    #     systems_data[sid] = system_config

            return ToolResult(
                success=True,
                data={"systems": systems_data, "total": len(systems_data)}
            )

    async def _query_controllers(self, parameters: Dict[str, Any]) -> ToolResult:
        """Query DER controllers"""
        controller_id = parameters.get("controller_id")
        system_id = parameters.get("system_id")
        requested_params = parameters.get("parameters", [])

        if controller_id:
            # Query specific controller
            controller_config = self.config_manager.get_config_by_id("controllers", controller_id)
            if not controller_config:
                return ToolResult(
                    success=False,
                    error=f"Controller {controller_id} not found"
                )

            if requested_params:
                filtered_config = self._filter_parameters(controller_config, requested_params)
                return ToolResult(success=True, data={controller_id: filtered_config})
            else:
                return ToolResult(success=True, data={controller_id: controller_config})
        else:
            # Query all controllers (optionally filtered by system)
            all_controllers = self.config_manager.list_all("controllers")
            controllers_data = {}

            for cid in all_controllers:
                controller_config = self.config_manager.get_config_by_id("controllers", cid)
                if controller_config:
                    # Filter by system if specified
                    if system_id and controller_config.get("system_id") != system_id:
                        continue

                    # Check if it's a DER controller
                    if controller_config.get("domain") == "electrical":
                        controllers_data[cid] = controller_config
                        # if requested_params:
                        #     controllers_data[cid] = self._filter_parameters(controller_config, requested_params)
                        # else:
                        #     controllers_data[cid] = controller_config

            return ToolResult(
                success=True,
                data={"controllers": controllers_data, "total": len(controllers_data)}
            )

    def _filter_parameters(self, config: Dict[str, Any], params: List[str]) -> Dict[str, Any]:
        """Filter configuration to only include requested parameters"""
        #@todo should be replaced by LLM
        filtered = {}
        for param in params:
            # Handle nested parameters with dot notation
            if "." in param:
                parts = param.split(".")
                current = config
                for part in parts[:-1]:
                    if part in current:
                        current = current[part]
                    else:
                        break
                if parts[-1] in current:
                    # Recreate nested structure
                    self._set_nested(filtered, parts, current[parts[-1]])
            elif param in config:
                filtered[param] = config[param]
        return filtered

    def _set_nested(self, d: Dict, keys: List[str], value: Any) -> None:
        """Set nested dictionary value"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value