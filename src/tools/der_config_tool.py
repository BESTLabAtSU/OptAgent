# src/tools/der_config_tool.py
"""
DER Configuration Tool - Manages DER system configurations
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from .base_tool import BaseTool, ToolResult


@dataclass
class DERSystemConfig:
    """DER System configuration structure"""
    cluster_id: str
    system_id: str
    system_type: str
    parameters: Dict[str, Any]
    class_path: str
    timestamp: Optional[str] = None

    def to_cm_add_system_call(self) -> str:
        """Convert to cm.add_system() call format"""
        return f"""cm.add_system(
    cluster_id="{self.cluster_id}",
    system_id="{self.system_id}",
    system_type="{self.system_type}",
    parameters={json.dumps(self.parameters, indent=8).replace('"', "'")},
    class_path="{self.class_path}"
)"""


class DERConfigTool(BaseTool):
    """
    Tool for configuring DER systems
    Handles updates to PV, Battery, EV, and other DER components
    """

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(
            name="der_config",
            description="Configure and update DER system parameters",
            parameters_schema={
                "action": {
                    "type": "string",
                    "enum": ["update", "create", "delete", "query"],
                    "description": "Action to perform"
                },
                "config": {
                    "type": "object",
                    "description": "Configuration parameters",
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "system_id": {"type": "string"},
                        "system_type": {"type": "string"},
                        "parameters": {"type": "object"},
                        "class_path": {"type": "string"}
                    }
                },
                "validate": {
                    "type": "boolean",
                    "default": True,
                    "description": "Validate configuration before applying"
                }
            }
        )

        self.config_dir = config_dir or Path("./der_configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage of current configurations
        self.current_configs: Dict[str, DERSystemConfig] = {}

        # Load existing configurations
        self._load_existing_configs()

    def _load_existing_configs(self) -> None:
        """Load existing configurations from disk"""
        config_file = self.config_dir / "current_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
                for system_id, config_data in data.items():
                    self.current_configs[system_id] = DERSystemConfig(**config_data)

    def _save_configs(self) -> None:
        """Save current configurations to disk"""
        config_file = self.config_dir / "current_config.json"
        data = {
            system_id: asdict(config)
            for system_id, config in self.current_configs.items()
        }
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the DER configuration tool

        Args:
            parameters: Tool parameters including action and config

        Returns:
            ToolResult with status and output
        """
        try:
            action = parameters.get("action", "update")
            config_data = parameters.get("config", {})
            validate = parameters.get("validate", True)

            if action == "update":
                result = await self._update_config(config_data, validate)
            elif action == "create":
                result = await self._create_config(config_data, validate)
            elif action == "delete":
                result = await self._delete_config(config_data.get("system_id"))
            elif action == "query":
                result = await self._query_config(config_data.get("system_id"))
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )

            return result

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"DER configuration failed: {str(e)}"
            )

    async def _update_config(
            self,
            config_data: Dict[str, Any],
            validate: bool
    ) -> ToolResult:
        """Update an existing DER configuration"""
        system_id = config_data.get("system_id")

        if not system_id:
            return ToolResult(
                success=False,
                error="system_id is required for update"
            )

        # Validate if requested
        if validate:
            validation_result = await self._validate_config(config_data)
            if not validation_result["valid"]:
                return ToolResult(
                    success=False,
                    error=f"Validation failed: {validation_result['errors']}"
                )

        # Create DERSystemConfig object
        config = DERSystemConfig(
            cluster_id=config_data.get("cluster_id", "residential_cluster_1"),
            system_id=system_id,
            system_type=config_data.get("system_type", "der_systems"),
            parameters=config_data.get("parameters", {}),
            class_path=config_data.get("class_path", "bestopt.env.modules.ders.system.der.DERModule"),
            timestamp=datetime.now().isoformat()
        )

        # Store previous config for rollback if needed
        previous_config = self.current_configs.get(system_id)

        # Update configuration
        self.current_configs[system_id] = config

        # Save to disk
        self._save_configs()

        # Generate the configuration code
        config_code = config.to_cm_add_system_call()

        # Write to Python file for the runtime system
        config_file = self.config_dir / f"{system_id}_config.py"
        with open(config_file, 'w') as f:
            f.write(f"# Generated at {config.timestamp}\n")
            f.write(f"# DER System Configuration for {system_id}\n\n")
            f.write(config_code)

        # Simulate applying to runtime (in real implementation, this would call the actual runtime API)
        runtime_result = await self._apply_to_runtime(config)

        return ToolResult(
            success=True,
            data={
                "system_id": system_id,
                "action": "update",
                "config": asdict(config),
                "config_code": config_code,
                "config_file": str(config_file),
                "runtime_status": runtime_result,
                "previous_config": asdict(previous_config) if previous_config else None
            }
        )

    async def _create_config(
            self,
            config_data: Dict[str, Any],
            validate: bool
    ) -> ToolResult:
        """Create a new DER configuration"""
        system_id = config_data.get("system_id")

        if not system_id:
            return ToolResult(
                success=False,
                error="system_id is required"
            )

        if system_id in self.current_configs:
            return ToolResult(
                success=False,
                error=f"System {system_id} already exists. Use update action instead."
            )

        # Proceed with update logic for new config
        return await self._update_config(config_data, validate)

    async def _delete_config(self, system_id: str) -> ToolResult:
        """Delete a DER configuration"""
        if not system_id:
            return ToolResult(
                success=False,
                error="system_id is required for deletion"
            )

        if system_id not in self.current_configs:
            return ToolResult(
                success=False,
                error=f"System {system_id} not found"
            )

        # Remove configuration
        deleted_config = self.current_configs.pop(system_id)

        # Save updated configs
        self._save_configs()

        # Remove config file
        config_file = self.config_dir / f"{system_id}_config.py"
        if config_file.exists():
            config_file.unlink()

        return ToolResult(
            success=True,
            data={
                "system_id": system_id,
                "action": "delete",
                "deleted_config": asdict(deleted_config)
            }
        )

    async def _query_config(self, system_id: Optional[str] = None) -> ToolResult:
        """Query DER configurations"""
        if system_id:
            if system_id not in self.current_configs:
                return ToolResult(
                    success=False,
                    error=f"System {system_id} not found"
                )

            config = self.current_configs[system_id]
            return ToolResult(
                success=True,
                data={
                    "system_id": system_id,
                    "config": asdict(config),
                    "config_code": config.to_cm_add_system_call()
                }
            )
        else:
            # Return all configurations
            all_configs = {
                sid: asdict(config)
                for sid, config in self.current_configs.items()
            }
            return ToolResult(
                success=True,
                data={
                    "total_systems": len(all_configs),
                    "systems": all_configs
                }
            )

    async def _validate_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DER configuration

        Returns:
            Dict with 'valid' boolean and 'errors' list
        """
        errors = []

        # Check required fields
        required_fields = ["system_id", "parameters"]
        for field in required_fields:
            if field not in config_data:
                errors.append(f"Missing required field: {field}")

        # Validate parameters structure
        if "parameters" in config_data:
            params = config_data["parameters"]

            if "system_config" in params:
                system_config = params["system_config"]

                # Validate PV configuration
                if "pv" in system_config:
                    pv_config = system_config["pv"]
                    if "rated_capacity_kW" in pv_config:
                        capacity = pv_config["rated_capacity_kW"]
                        if not isinstance(capacity, (int, float)) or capacity <= 0:
                            errors.append("PV rated_capacity_kW must be a positive number")

                # Validate Battery configuration
                if "bat" in system_config:
                    bat_config = system_config["bat"]
                    if "rated_capacity_kWh" in bat_config:
                        capacity = bat_config["rated_capacity_kWh"]
                        if not isinstance(capacity, (int, float)) or capacity <= 0:
                            errors.append("Battery rated_capacity_kWh must be a positive number")

                    if "initial_soc" in bat_config:
                        soc = bat_config["initial_soc"]
                        if not isinstance(soc, (int, float)) or soc < 0 or soc > 1:
                            errors.append("Battery initial_soc must be between 0 and 1")

                # Validate EV configuration
                if "ev" in system_config:
                    ev_config = system_config["ev"]
                    if "rated_capacity_kWh" in ev_config:
                        capacity = ev_config["rated_capacity_kWh"]
                        if not isinstance(capacity, (int, float)) or capacity <= 0:
                            errors.append("EV rated_capacity_kWh must be a positive number")

                    if "initial_soc" in ev_config:
                        soc = ev_config["initial_soc"]
                        if not isinstance(soc, (int, float)) or soc < 0 or soc > 1:
                            errors.append("EV initial_soc must be between 0 and 1")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def _apply_to_runtime(self, config: DERSystemConfig) -> Dict[str, Any]:
        """
        Apply configuration to the runtime system
        In production, this would call the actual building-grid-ders runtime API
        """
        # Simulate API call to runtime
        await asyncio.sleep(0.1)  # Simulate network delay

        # In production, this would be something like:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         f"{runtime_api_url}/configure",
        #         json=asdict(config)
        #     ) as response:
        #         return await response.json()

        return {
            "status": "success",
            "message": f"Configuration applied to runtime for {config.system_id}",
            "timestamp": datetime.now().isoformat()
        }