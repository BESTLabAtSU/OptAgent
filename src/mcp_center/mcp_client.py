"""
MCP Client
"""
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path
import logging
import asyncio
import json


class MCPClient:
    """MCP Client for tool communication with enhanced discovery"""

    def __init__(self, server_script_path: str = None):
        """
        Initialize MCP Client

        Args:
            server_script_path: Path to the MCP server script
        """
        self.server_script_path = server_script_path or str(Path(__file__).parent / "mcp_server.py")
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.available_tools: Dict[str, Any] = {}
        self.logger = logging.getLogger("MCPClient")
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._connected and self.session is not None

    async def connect(self):
        """Connect to an MCP server"""
        if self._connected:
            self.logger.warning("Already connected to MCP server")
            return

        try:
            self.exit_stack = AsyncExitStack()

            server_params = StdioServerParameters(
                command="python",
                args=[self.server_script_path],
                env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            self.stdio, self.write = stdio_transport

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            # Discover available tools with enhanced parsing
            await self.discover_tools()

            self._connected = True
            self.logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            if self.exit_stack:
                await self.exit_stack.aclose()
            raise

    async def disconnect(self):
        """Disconnect from server"""
        if not self._connected:
            return

        try:
            if self.exit_stack:
                await self.exit_stack.aclose()
            self._connected = False
            self.session = None
            self.available_tools = {}
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    async def discover_tools(self):
        """Discover and cache available tools with structured metadata"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self.session.list_tools()
            self.available_tools = {}

            for tool in response.tools:
                # Get the raw schema
                input_schema = getattr(tool, 'inputSchema', {})

                # Parse the schema into structured format
                parameters = self._parse_parameters(input_schema)

                self.available_tools[tool.name] = {
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                    "category": self._categorize_tool(tool.name),
                    "parameters": parameters,
                    "returns": self._parse_return_type(tool),
                    "raw_schema": input_schema  # Keep raw schema as backup
                }

            self.logger.info(f"Discovered {len(self.available_tools)} tools")
        except Exception as e:
            self.logger.error(f"Failed to discover tools: {e}")
            raise

    def _parse_parameters(self, schema: Dict) -> List[Dict]:
        """Extract structured parameter information from JSON schema"""
        parameters = []
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        for param_name, param_schema in properties.items():
            # Handle the anyOf pattern for optional parameters
            param_type = param_schema.get('type', 'any')
            is_optional = False

            if 'anyOf' in param_schema:
                # This is an Optional type
                types = param_schema['anyOf']
                is_optional = any(t.get('type') == 'null' for t in types)
                # Get the actual type (non-null one)
                for t in types:
                    if t.get('type') != 'null':
                        param_type = t.get('type', 'string')
                        break

            # Handle array types
            if param_type == 'array':
                items = param_schema.get('items', {})
                item_type = items.get('type', 'any')
                param_type = f"List[{item_type}]"

            # Handle object types
            elif param_type == 'object':
                param_type = "Dict"

            param_info = {
                'name': param_name,
                'type': param_type,
                'description': param_schema.get('description', ''),
                'required': param_name in required,
                'optional': is_optional,
                'default': param_schema.get('default'),
                'title': param_schema.get('title', param_name),
                'enum': param_schema.get('enum'),  # If there are specific allowed values
                'example': param_schema.get('example'),  # If examples are provided
            }

            # Add constraints if present
            if 'minLength' in param_schema:
                param_info['min_length'] = param_schema['minLength']
            if 'maxLength' in param_schema:
                param_info['max_length'] = param_schema['maxLength']
            if 'minimum' in param_schema:
                param_info['minimum'] = param_schema['minimum']
            if 'maximum' in param_schema:
                param_info['maximum'] = param_schema['maximum']

            parameters.append(param_info)

        return parameters

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tools based on their names"""
        if 'config' in tool_name:
            return 'configuration'
        elif 'building' in tool_name:
            return 'building_management'
        elif 'cluster' in tool_name:
            return 'cluster_management'
        elif 'simulation' in tool_name or 'sim' in tool_name:
            return 'simulation'
        elif 'environment' in tool_name or 'env' in tool_name:
            return 'environment'
        elif 'result' in tool_name or 'output' in tool_name:
            return 'results'
        elif 'analysis' in tool_name or 'analyze' in tool_name:
            return 'analysis'
        # Add more categories as needed
        return 'general'

    def _parse_return_type(self, tool) -> Dict:
        """Parse the return type information from tool metadata"""
        # This is a default implementation - adjust based on your FastMCP setup
        return {
            'type': 'Dict[str, Any]',
            'description': 'Returns a dictionary with success status, data, and optional error message',
            'schema': {
                'success': 'bool - Whether the operation succeeded',
                'data': 'Dict - Operation results',
                'message': 'str - Human-readable status message',
                'error': 'str (optional) - Error message if operation failed'
            }
        }

    def list_all_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.available_tools.keys())

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None):
        """
        Call a tool

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {self.list_all_tools()}")

        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            self.logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information or None if not found
        """
        return self.available_tools.get(tool_name)

    def get_tools_by_category(self) -> Dict[str, List[Dict]]:
        """
        Get tools organized by category

        Returns:
            Dictionary of category -> list of tool info
        """
        categories = {}
        for tool_name, tool_info in self.available_tools.items():
            category = tool_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(tool_info)
        return categories

    def get_tool_signature(self, tool_name: str) -> str:
        """Generate a clean function signature for a tool"""
        if tool_name not in self.available_tools:
            return f"Unknown tool: {tool_name}"

        tool = self.available_tools[tool_name]
        params = []

        for p in tool['parameters']:
            # Format each parameter cleanly
            param_str = f"{p['name']}: {p['type']}"
            if not p['required']:
                param_str += " (optional)"
            if p.get('default') is not None:
                param_str += f" = {repr(p['default'])}"
            params.append(param_str)

        signature = f"{tool_name}({', '.join(params)})"
        return signature

    def get_llm_prompt(self) -> str:
        """Generate clean, structured tool documentation for the Orchestrator (LLM-optimized)"""
        lines = []

        # Group tools by category
        tools_by_category = self.get_tools_by_category()

        for category, tools in sorted(tools_by_category.items()):
            lines.append(f"{category.upper()} TOOLS:")

            for tool in tools:
                lines.append(f"\nTool: {tool['name']}")
                lines.append(f"Purpose: {tool['description']}")

                if tool['parameters']:
                    lines.append("Parameters:")
                    for param in tool['parameters']:
                        param_line = f"- {param['name']} ({param['type']}"
                        if param['required']:
                            param_line += ", required"
                        else:
                            param_line += ", optional"
                        param_line += f"): {param['description']}"

                        if param.get('default') is not None:
                            param_line += f", default={param['default']}"

                        lines.append(param_line)
                else:
                    lines.append("Parameters: none")

                lines.append(f"Returns: {tool['returns']['description']}")

        return "\n".join(lines)

    def get_tools_as_structured_json(self) -> Dict:
        """Get tools in a structured JSON format for programmatic use"""
        structured = {
            "total_tools": len(self.available_tools),
            "categories": {}
        }

        tools_by_category = self.get_tools_by_category()

        for category, tools in tools_by_category.items():
            structured['categories'][category] = []

            for tool in tools:
                structured_tool = {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": [
                        {
                            "name": p['name'],
                            "type": p['type'],
                            "required": p['required'],
                            "description": p['description'],
                            "default": p.get('default'),
                            "constraints": {
                                k: v for k, v in p.items()
                                if k in ['enum', 'min_length', 'max_length', 'minimum', 'maximum']
                            }
                        }
                        for p in tool['parameters']
                    ],
                    "returns": tool['returns'],
                    "example_call": self._generate_example_call(tool['name'], tool, as_dict=True)
                }
                structured['categories'][category].append(structured_tool)

        return structured

    def _generate_example_call(self, tool_name: str, tool_info: Dict, as_dict: bool = False) -> Any:
        """Generate an example call for the tool"""
        example_args = {}

        for param in tool_info['parameters']:
            # Only include required parameters in the example
            if param['required']:
                # Use example if provided, otherwise generate based on type
                if param.get('example'):
                    example_args[param['name']] = param['example']
                elif param['type'] == 'string':
                    example_args[param['name']] = f"example_{param['name']}"
                elif param['type'] == 'integer':
                    example_args[param['name']] = 1
                elif param['type'] == 'number':
                    example_args[param['name']] = 1.0
                elif param['type'] == 'boolean':
                    example_args[param['name']] = True
                elif 'List' in param['type']:
                    example_args[param['name']] = ["item1", "item2"]
                elif param['type'] == 'Dict':
                    example_args[param['name']] = {"key": "value"}
                else:
                    example_args[param['name']] = f"<{param['type']}_value>"

        if as_dict:
            return {
                "tool": tool_name,
                "arguments": example_args
            }
        else:
            # Format as function call string
            args_str = json.dumps(example_args, indent=None)
            return f"await client.call_tool('{tool_name}', {args_str})"

    def print_tool_summary(self):
        """Print a formatted summary of all available tools"""
        print(self.get_llm_prompt())

    def export_tool_documentation(self, filepath: str = "mcp_tools_documentation.md"):
        """Export tool documentation to a markdown file"""
        lines = ["# MCP Tools Documentation\n"]
        lines.append(f"Total tools available: {len(self.available_tools)}\n")

        tools_by_category = self.get_tools_by_category()

        for category, tools in sorted(tools_by_category.items()):
            lines.append(f"\n## {category.replace('_', ' ').title()}\n")

            for tool in tools:
                lines.append(f"### `{tool['name']}`\n")
                lines.append(f"**Description:** {tool['description']}\n")

                if tool['parameters']:
                    lines.append("\n**Parameters:**\n")
                    for param in tool['parameters']:
                        req = "required" if param['required'] else "optional"
                        lines.append(f"- `{param['name']}` ({param['type']}, {req}): {param['description']}")
                        if param.get('default') is not None:
                            lines.append(f"  - Default: `{param['default']}`")
                        if param.get('enum'):
                            lines.append(f"  - Allowed values: {param['enum']}")
                        lines.append("")

                lines.append(f"\n**Returns:** {tool['returns']['description']}\n")

                example = self._generate_example_call(tool['name'], tool)
                lines.append(f"\n**Example:**\n```python\n{example}\n```\n")
                lines.append("---\n")

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))

        self.logger.info(f"Tool documentation exported to {filepath}")


class MCPClientPool:
    """Pool of MCP clients for concurrent operations"""

    def __init__(self, pool_size: int = 3, server_script_path: str = None):
        """
        Initialize MCP client pool

        Args:
            pool_size: Number of clients in the pool
            server_script_path: Path to the MCP server script
        """
        self.pool_size = pool_size
        self.server_script_path = server_script_path
        self.clients: List[MCPClient] = []
        self.available_clients: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger("MCPClientPool")
        self._initialized = False

    async def initialize(self):
        """Initialize all clients in the pool"""
        if self._initialized:
            return

        try:
            for i in range(self.pool_size):
                client = MCPClient(server_script_path=self.server_script_path)
                await client.connect()
                self.clients.append(client)
                await self.available_clients.put(client)

            self._initialized = True
            self.logger.info(f"Initialized pool with {self.pool_size} clients")
        except Exception as e:
            self.logger.error(f"Failed to initialize client pool: {e}")
            await self.shutdown()
            raise

    async def get_client(self) -> MCPClient:
        """
        Get an available client from the pool

        Returns:
            Available MCP client
        """
        if not self._initialized:
            raise RuntimeError("Client pool not initialized")
        return await self.available_clients.get()

    async def return_client(self, client: MCPClient):
        """
        Return a client to the pool

        Args:
            client: Client to return
        """
        await self.available_clients.put(client)

    async def execute_with_client(self, func, *args, **kwargs):
        """
        Execute a function with a client from the pool

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Function result
        """
        client = await self.get_client()
        try:
            result = await func(client, *args, **kwargs)
            return result
        finally:
            await self.return_client(client)

    async def shutdown(self):
        """Shutdown all clients in the pool"""
        for client in self.clients:
            try:
                await client.disconnect()
            except Exception as e:
                self.logger.error(f"Error shutting down client: {e}")

        self.clients.clear()
        self._initialized = False
        self.logger.info("Client pool shut down")

