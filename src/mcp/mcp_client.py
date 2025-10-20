# =====================================
# src/mcp/mcp_client.py
# =====================================
"""
MCP Client for connecting to MCP servers
"""
import asyncio
import websockets
import json
from typing import Dict, Any, Optional


class MCPClient:
    """Client for MCP protocol communication"""

    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self):
        """Connect to MCP server"""
        self.websocket = await websockets.connect(self.uri)

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.websocket:
            await self.websocket.close()

    async def send_context_update(self, context_id: str, context: Dict[str, Any]):
        """Send context update to server"""
        message = {
            "type": "context_update",
            "context_id": context_id,
            "context": context
        }
        if self.websocket:
            await self.websocket.send(json.dumps(message))

    async def query_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Query context from server"""
        message = {
            "type": "context_query",
            "context_id": context_id
        }
        if self.websocket:
            await self.websocket.send(json.dumps(message))
            response = await self.websocket.recv()
            return json.loads(response)
        return None