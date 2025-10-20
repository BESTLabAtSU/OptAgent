# =====================================
# src/mcp/mcp_server.py
# =====================================
"""
MCP (Model Context Protocol) Server implementation
"""
import asyncio
import websockets
import json
from typing import Dict, Any, Set
from datetime import datetime


class MCPServer:
    """
    MCP Server for handling model context protocol communications
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.contexts: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        """Start the MCP server"""
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"MCP Server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def handle_client(self, websocket, path):
        """Handle client connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        finally:
            self.clients.remove(websocket)

    async def process_message(self, websocket, message: str):
        """Process incoming MCP messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "context_update":
                await self.update_context(data)
            elif message_type == "context_query":
                await self.send_context(websocket, data)
            elif message_type == "broadcast":
                await self.broadcast_message(data)

        except Exception as e:
            error_response = {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_response))

    async def update_context(self, data: Dict[str, Any]):
        """Update shared context"""
        context_id = data.get("context_id")
        context_data = data.get("context")

        if context_id:
            self.contexts[context_id] = {
                "data": context_data,
                "timestamp": datetime.now().isoformat()
            }

    async def send_context(self, websocket, data: Dict[str, Any]):
        """Send context to requesting client"""
        context_id = data.get("context_id")

        if context_id in self.contexts:
            response = {
                "type": "context_response",
                "context_id": context_id,
                "context": self.contexts[context_id]
            }
            await websocket.send(json.dumps(response))

    async def broadcast_message(self, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        message = json.dumps(data)
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )