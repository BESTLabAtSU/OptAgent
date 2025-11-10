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


class DERSystemTool(BaseTool):
    """
    Tool for managing DER systems (add, update, delete)
    Handles PV, Battery, EV, and other DER components
    """

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(
            name="der_system",
            description="Manage DER systems",
            parameters_schema={
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "delete"],
                    "description": "Action to perform"
                },
                "system_id": {
                    "type": "string",
                    "description": "System identifier"
                },
                "config": {
                    "type": "object",
                    "description": "System configuration"
                },
                "updates": {
                    "type": "object",
                    "description": "Parameters to update"
                },
                "merge": {
                    "type": "boolean",
                    "default": True,
                    "description": "Merge with existing config or replace"
                }
            }
        )

        self.config_dir = config_dir or Path(".")
        self.config_manager = ConfigurationManager(self.config_dir)
        self.default_cluster = "residential_cluster_1"

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute system operation"""
        try:
            action = parameters.get("action", "update")

            if action == "add":
                return await self._add_system(parameters)
            elif action == "update":
                return await self._update_system(parameters)
            elif action == "delete":
                return await self._delete_system(parameters)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"System operation failed: {str(e)}"
            )

    async def _add_system(self, parameters: Dict[str, Any]) -> ToolResult:
        """Add new DER system"""
        config = parameters.get("config", {})
        system_id = config.get("system_id") or parameters.get("system_id")

        if not system_id:
            return ToolResult(success=False, error="system_id is required")

        # Build system configuration
        system_config = {
            "system_name": config.get("system_name", f"DER System {system_id}"),
            "system_config": {}
        }

        # Add components
        components = config.get("components", {})
        if "pv" in components:
            system_config["system_config"]["pv"] = components["pv"]
        if "bat" in components:
            system_config["system_config"]["bat"] = components["bat"]
        if "ev" in components:
            system_config["system_config"]["ev"] = components["ev"]
        if "evs" in components:
            system_config["system_config"]["evs"] = components["evs"]

        # Add system to configuration manager
        self.config_manager.add_system(
            cluster_id=config.get("cluster_id", self.default_cluster),
            system_id=system_id,
            system_type="der_systems",
            parameters=system_config,
            class_path=config.get("class_path", "bestopt.env.modules.ders.system.der.DERModule")
        )

        # Save configuration
        self.config_manager.save_all_configurations(self.config_dir)

        return ToolResult(
            success=True,
            data={
                "action": "add",
                "system_id": system_id,
                "config": system_config,
                "message": f"DER system {system_id} added successfully"
            }
        )

    async def _delete_system(self, parameters: Dict[str, Any]) -> ToolResult:
        """Delete DER system"""
        system_id = parameters.get("system_id")

        if not system_id:
            return ToolResult(success=False, error="system_id is required")

        # Remove system
        success = self.config_manager.remove_item("systems", system_id)

        if success:
            # Save configuration
            self.config_manager.save_all_configurations(self.config_dir)

            return ToolResult(
                success=True,
                data={
                    "action": "delete",
                    "system_id": system_id,
                    "message": f"DER system {system_id} deleted successfully"
                }
            )
        else:
            return ToolResult(
                success=False,
                error=f"Failed to delete system {system_id}"
            )

    async def _update_system(self, parameters: Dict[str, Any]) -> ToolResult:
        """Update existing DER system"""
        #grab system id from existing config and assume we only have one
        #@todo need to figure how to pass correct system_id here
        system_id = "der_system_1"
        # system_id = parameters.get("system_id")
        updates = parameters.get("updates", {})
        merge = parameters.get("merge", True)

        if not system_id:
            return ToolResult(success=False, error="system_id is required")

        if not updates:
            return ToolResult(success=False, error="No updates provided")

        try:
            # Get existing system configuration
            existing_system = self.config_manager.get_config_by_id("systems", system_id)

            if not existing_system:
                return ToolResult(
                    success=False,
                    error=f"System {system_id} not found"
                )

            # Extract current parameters
            current_params = existing_system.get("parameters", {})
            current_config = current_params.get("system_config", {})

            # Prepare the updates for ConfigurationManager
            updated_params = {}

            # Handle component updates
            if "components" in updates:
                # Build new system_config
                new_system_config = current_config.copy() if merge else {}

                for comp_name, comp_config in updates["components"].items():
                    if comp_config is None or comp_config == {}:
                        # Remove component if None or empty
                        new_system_config.pop(comp_name, None)
                    else:
                        # Update or add component
                        if merge and comp_name in new_system_config:
                            # Merge component configuration
                            new_system_config[comp_name] = {
                                **new_system_config[comp_name],
                                **comp_config
                            }
                        else:
                            new_system_config[comp_name] = comp_config

                updated_params["system_config"] = new_system_config

            # Handle other parameter updates (system_name, etc.)
            for key, value in updates.items():
                if key != "components":
                    updated_params[key] = value

            # Preserve system_name if not in updates
            if "system_name" not in updated_params:
                updated_params["system_name"] = current_params.get(
                    "system_name",
                    f"DER System {system_id}"
                )

            # If we're merging and system_config wasn't updated, preserve it
            if merge and "system_config" not in updated_params:
                updated_params["system_config"] = current_config

            # Use the configuration manager's update_system method
            self.config_manager.update_system(
                system_id=system_id,
                parameters=updated_params,
                merge=False  # We've already handled merging above
            )

            # Save configuration
            self.config_manager.save_all_configurations(self.config_dir)

            # Get the updated system for response
            updated_system = self.config_manager.get_config_by_id("systems", system_id)

            return ToolResult(
                success=True,
                data={
                    "action": "update",
                    "system_id": system_id,
                    "previous_config": current_params,
                    "updated_config": updated_system.get("parameters", {}),
                    "merge_mode": merge,
                    "message": f"DER system {system_id} updated successfully"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to update system {system_id}: {str(e)}"
            )