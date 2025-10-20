"""
Base tool class for all tools in the system
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for tools"""

    def __init__(
            self,
            name: str,
            description: str,
            parameters_schema: Dict[str, Any]
    ):
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against schema"""
        # Simple validation - can be enhanced with jsonschema
        required = [k for k, v in self.parameters_schema.items()
                    if isinstance(v, dict) and v.get("required", False)]

        for req in required:
            if req not in parameters:
                return False
        return True