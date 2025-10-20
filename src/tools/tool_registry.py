"""
Tool registry for managing available tools
"""
from typing import Dict, Optional, List, Any
from .base_tool import BaseTool


class ToolRegistry:
    """Registry for managing tools"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool"""
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self._tools.keys())

    def get_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all tools"""
        return {
            name: {
                "description": tool.description,
                "parameters": tool.parameters_schema
            }
            for name, tool in self._tools.items()
        }


# Global registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry