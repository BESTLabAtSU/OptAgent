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


class DERControllerTool(BaseTool):
    """
    Tool for managing DER controllers (add, update, delete)
    Handles controller configuration and parameters
    """

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(
            name="der_controller",
            description="Manage DER controllers",
            parameters_schema={
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "delete"],
                    "description": "Action to perform"
                },
                "controller_id": {
                    "type": "string",
                    "description": "Controller identifier"
                },
                "system_id": {
                    "type": "string",
                    "description": "Associated system ID"
                },
                "config": {
                    "type": "object",
                    "description": "Controller configuration"
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
        self.config_manager = ConfigurationManager()

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute controller operation"""
        try:
            action = parameters.get("action", "update")

            if action == "add":
                return await self._add_controller(parameters)
            elif action == "update":
                return await self._update_controller(parameters)
            elif action == "delete":
                return await self._delete_controller(parameters)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Controller operation failed: {str(e)}"
            )

    async def _add_controller(self, parameters: Dict[str, Any]) -> ToolResult:
        """Add new DER controller"""
        config = parameters.get("config", {})
        controller_id = config.get("controller_id") or parameters.get("controller_id")
        system_id = config.get("system_id") or parameters.get("system_id")

        if not controller_id:
            return ToolResult(success=False, error="controller_id is required")
        if not system_id:
            return ToolResult(success=False, error="system_id is required")

        # Build controller configuration
        controller_params = config.get("parameters", {})

        # Set default parameters if not provided
        default_params = {
            "domain": "electrical",
            "type": controller_params.get("type", "rule-based"),
            "mode": controller_params.get("mode", "self_consumption"),
            "bat_soc_min": controller_params.get("bat_soc_min", 0.1),
            "bat_soc_max": controller_params.get("bat_soc_max", 0.9),
            "ev_soc_min": controller_params.get("ev_soc_min", 0.2),
            "ev_soc_target": controller_params.get("ev_soc_target", 0.8),
            "ev_v2g_enabled": controller_params.get("ev_v2g_enabled", False),
            "max_grid_import": controller_params.get("max_grid_import", 10000),
            "max_grid_export": controller_params.get("max_grid_export", 5000)
        }

        # Merge with provided parameters
        controller_params = {**default_params, **controller_params}

        # Add controller to configuration manager
        self.config_manager.add_system_controller(
            controller_id=controller_id,
            system_id=system_id,
            parameters=controller_params,
            class_path=config.get("class_path", "bestopt.env.controllers.electrical.SupervisoryController")
        )

        # Save configuration
        self.config_manager.save_all_configurations(self.config_dir / "der_controllers.json")

        return ToolResult(
            success=True,
            data={
                "action": "add",
                "controller_id": controller_id,
                "system_id": system_id,
                "parameters": controller_params,
                "message": f"DER controller {controller_id} added successfully"
            }
        )

    async def _update_controller(self, parameters: Dict[str, Any]) -> ToolResult:
        """Update existing DER controller"""
        controller_id = parameters.get("controller_id")
        updates = parameters.get("updates", {})
        merge = parameters.get("merge", True)

        if not controller_id:
            return ToolResult(success=False, error="controller_id is required")

        # Check if controller exists
        existing = self.config_manager.get_config_by_id("controllers", controller_id)
        if not existing:
            return ToolResult(success=False, error=f"Controller {controller_id} not found")

        # Update controller
        self.config_manager.update_controller(
            controller_id=controller_id,
            parameters=updates,
            merge=merge
        )

        # Save configuration
        self.config_manager.save_all_configurations(self.config_dir / "der_controllers.json")

        # Get updated configuration
        updated_config = self.config_manager.get_config_by_id("controllers", controller_id)

        return ToolResult(
            success=True,
            data={
                "action": "update",
                "controller_id": controller_id,
                "updates": updates,
                "merge": merge,
                "config": updated_config,
                "message": f"DER controller {controller_id} updated successfully"
            }
        )

    async def _delete_controller(self, parameters: Dict[str, Any]) -> ToolResult:
        """Delete DER controller"""
        controller_id = parameters.get("controller_id")

        if not controller_id:
            return ToolResult(success=False, error="controller_id is required")

        # Remove controller
        success = self.config_manager.remove_item("controllers", controller_id)

        if success:
            # Save configuration
            self.config_manager.save_all_configurations(self.config_dir / "der_controllers.json")

            return ToolResult(
                success=True,
                data={
                    "action": "delete",
                    "controller_id": controller_id,
                    "message": f"DER controller {controller_id} deleted successfully"
                }
            )
        else:
            return ToolResult(
                success=False,
                error=f"Failed to delete controller {controller_id}"
            )

    async def _update_system(self, parameters: Dict[str, Any]) -> ToolResult:
        """Update existing DER system"""
        system_id = parameters.get("system_id")
        updates = parameters.get("updates", {})
        merge = parameters.get("merge", True)

        if not system_id:
            return ToolResult(success=False, error="system_id is required")

        # Check if system exists
        existing = self.config_manager.get_config_by_id("systems", system_id)
        if not existing:
            return ToolResult(success=False, error=f"System {system_id} not found")

        # Prepare updates
        if "components" in updates:
            # Convert components to system_config format
            system_config_updates = {}
            for comp_name, comp_config in updates["components"].items():
                system_config_updates[comp_name] = comp_config

            updates = {
                "system_config": system_config_updates
            }

        # Update system
        self.config_manager.update_system(
            system_id=system_id,
            parameters=updates if not merge else None,
            merge=merge
        )

        # If updating specific components
        if merge and "system_config" in updates:
            for comp_name, comp_config in updates["system_config"].items():
                self.config_manager.update_system_component(
                    system_id=system_id,
                    component_name=comp_name,
                    component_config=comp_config,
                    merge=merge
                )

        # Save configuration
        self.config_manager.save_all_configurations(self.config_dir / "der_systems.json")

        return ToolResult(
            success=True,
            data={
                "action": "update",
                "system_id": system_id,
                "updates": updates,
                "merge": merge,
                "message": f"DER system_id {system_id} updated successfully"})